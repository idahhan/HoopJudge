"""Persistent ball-handler identity tracker.

The possession tracker assigns raw detector IDs that can change when a player
is briefly occluded or leaves/re-enters the frame.  This module adds a
post-processing layer that decides whether a raw-ID change is a *relabel*
(same physical player, different detector token) or a genuine *possession switch*
(a different person now has the ball).

Re-id scoring
-------------
  bbox_proximity   0.30   distance between bbox centres, normalised by frame diagonal
  bbox_size        0.15   ratio of bbox areas (min/max)
  pose_similarity  0.25   cosine similarity of available normalised keypoints
                          (left/right wrist + elbow from player_selection data)
  color_hist       0.30   HSV torso histogram intersection (upper 60 % of bbox)

Decision rules
--------------
  reid_score >= relabel_threshold          → immediate relabel (same player, new ID)
  reid_score <  relabel_threshold for
      >= switch_hysteresis_frames          → true possession switch accepted
  otherwise                               → hold current persistent ID (hysteresis)

Entry point
-----------
  track_handler_identity(data, video_path, config)
      Reads data["possession"] and data["player_selection"], opens the video to
      sample torso colour, and writes data["handler_identity"] with per-frame
      records.  Returns the updated data dict.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    # Re-id score weights (must sum to 1.0)
    "w_bbox_proximity": 0.30,
    "w_bbox_size":      0.15,
    "w_pose":           0.25,
    "w_color":          0.30,
    # Scores above this threshold → relabel (same player, new raw detector ID)
    "relabel_threshold": 0.60,
    # How many consecutive frames a challenger must hold a low re-id score
    # before a true switch is accepted
    "switch_hysteresis_frames": 4,
    # Fraction of bbox height used as the torso region (from top of bbox)
    "torso_fraction": 0.60,
    # Number of HSV histogram bins per channel
    "hist_bins_h": 18,
    "hist_bins_s": 8,
    "hist_bins_v": 8,
    # Minimum number of pixels to compute a colour histogram (otherwise skip)
    "min_torso_pixels": 200,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_centre(bbox: List[int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_area(bbox: List[int]) -> float:
    return max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))


def _frame_diagonal(W: int, H: int) -> float:
    return math.sqrt(W * W + H * H) or 1.0


# ---------------------------------------------------------------------------
# Per-feature scoring helpers
# ---------------------------------------------------------------------------

def _score_bbox_proximity(
    bbox_a: List[int], bbox_b: List[int], W: int, H: int
) -> float:
    """1 − normalised centre distance (0 = far apart, 1 = same centre)."""
    ca = _bbox_centre(bbox_a)
    cb = _bbox_centre(bbox_b)
    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
    return max(0.0, 1.0 - dist / _frame_diagonal(W, H))


def _score_bbox_size(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Area ratio min/max (1 = identical size, 0 = wildly different)."""
    area_a = _bbox_area(bbox_a)
    area_b = _bbox_area(bbox_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def _keypoints_vec(player_dict: Dict) -> Optional[np.ndarray]:
    """Return a fixed-length 8-element normalised keypoint vector.

    Slots: [lw_x, lw_y, rw_x, rw_y, le_x, le_y, re_x, re_y]
    Missing keypoints are zeroed out.  Returns None if fewer than 2 keypoints
    are available (not enough signal for cosine similarity).
    """
    keys = ("left_wrist_norm", "right_wrist_norm",
            "left_elbow_norm", "right_elbow_norm")
    pts = np.zeros(8, dtype=np.float32)
    n_valid = 0
    for i, key in enumerate(keys):
        v = player_dict.get(key)
        if v and len(v) >= 2:
            pts[i * 2]     = float(v[0])
            pts[i * 2 + 1] = float(v[1])
            n_valid += 1
    if n_valid < 2:
        return None
    return pts


def _score_pose(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    """Cosine similarity of normalised upper-body keypoint vectors."""
    if vec_a is None or vec_b is None:
        return 0.5  # neutral when data is missing
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < 1e-6 or nb < 1e-6:
        return 0.5
    return float(np.dot(vec_a, vec_b) / (na * nb))


def _torso_hist(frame_bgr: np.ndarray, bbox: List[int],
                torso_frac: float, bins_h: int, bins_s: int, bins_v: int,
                min_pixels: int) -> Optional[np.ndarray]:
    """Compute a normalised HSV histogram for the torso region of a bbox."""
    x1, y1, x2, y2 = bbox
    H_frame, W_frame = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W_frame, x2), min(H_frame, y2)
    bh = y2 - y1
    if bh <= 0 or x2 <= x1:
        return None
    torso_y2 = y1 + int(bh * torso_frac)
    roi = frame_bgr[y1:torso_y2, x1:x2]
    if roi.size < min_pixels * 3:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [bins_h, bins_s, bins_v],
        [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def _score_color(hist_a: Optional[np.ndarray],
                 hist_b: Optional[np.ndarray]) -> float:
    """Histogram intersection similarity (0–1)."""
    if hist_a is None or hist_b is None:
        return 0.5  # neutral when frames not available
    # Intersection of two L1-normalised histograms lies in [0, 1]
    return float(np.minimum(hist_a, hist_b).sum())


# ---------------------------------------------------------------------------
# State object for a tracked handler
# ---------------------------------------------------------------------------

class _HandlerState:
    """Stores the last known appearance of the persistent ball-handler."""

    __slots__ = ("persistent_id", "detector_id", "bbox", "kp_vec", "color_hist")

    def __init__(self, persistent_id: int, detector_id: int,
                 bbox: List[int],
                 kp_vec: Optional[np.ndarray],
                 color_hist: Optional[np.ndarray]) -> None:
        self.persistent_id = persistent_id
        self.detector_id   = detector_id
        self.bbox          = bbox
        self.kp_vec        = kp_vec
        self.color_hist    = color_hist


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class HandlerIdentityTracker:
    """Post-processing re-id layer for ball-handler identity persistence.

    Usage::

        tracker = HandlerIdentityTracker(config)
        tracker.initialize(W, H)
        for frame_idx, poss_record, ps_players, frame_bgr in ...:
            result = tracker.update(frame_idx, poss_record, ps_players, frame_bgr)
            # result: dict with persistent_handler_id, handler_id_source, reid_score
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(_DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self._cfg = cfg

        self._W: int = 640
        self._H: int = 480

        self._state: Optional[_HandlerState] = None
        self._next_persistent_id: int = 0

        # Hysteresis bookkeeping for a challenger
        self._challenger_det_id: Optional[int] = None
        self._challenger_frames:  int           = 0

    def initialize(self, W: int, H: int) -> None:
        self._W = W
        self._H = H

    def _reid_score(self,
                    candidate_bbox: List[int],
                    candidate_kp:   Optional[np.ndarray],
                    candidate_hist: Optional[np.ndarray],
                    state: _HandlerState) -> float:
        cfg = self._cfg
        s_prox  = _score_bbox_proximity(candidate_bbox, state.bbox, self._W, self._H)
        s_size  = _score_bbox_size(candidate_bbox, state.bbox)
        s_pose  = _score_pose(candidate_kp, state.kp_vec)
        s_color = _score_color(candidate_hist, state.color_hist)

        score = (cfg["w_bbox_proximity"] * s_prox +
                 cfg["w_bbox_size"]      * s_size  +
                 cfg["w_pose"]           * s_pose  +
                 cfg["w_color"]          * s_color)
        return float(score)

    def _find_ps_player(self,
                        bbox: List[int],
                        ps_players: List[Dict]) -> Optional[Dict]:
        """Find the player_selection entry closest to *bbox* by IoU."""
        best, best_iou = None, 0.0
        for p in ps_players:
            pb = p.get("bbox")
            if pb is None:
                continue
            ax1, ay1, ax2, ay2 = bbox
            bx1, by1, bx2, by2 = pb
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
            iou   = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best = p
        return best

    def update(self,
               frame_idx: int,
               poss_record: Dict,
               ps_players:  List[Dict],
               frame_bgr:   Optional[np.ndarray]) -> Dict[str, Any]:
        """Process one frame and return the handler identity result."""

        cfg = self._cfg
        det_id   = poss_record.get("possessor_player_id")
        det_bbox = poss_record.get("possessor_bbox")

        # No possession detected this frame
        if det_id is None or det_bbox is None:
            return {
                "frame_idx":            frame_idx,
                "persistent_handler_id": (self._state.persistent_id
                                          if self._state else None),
                "raw_detector_id":      det_id,
                "handler_id_source":    "no_possession",
                "reid_score":           None,
            }

        # Compute candidate features
        ps_player   = self._find_ps_player(det_bbox, ps_players)
        kp_vec      = _keypoints_vec(ps_player) if ps_player else None

        if frame_bgr is not None:
            color_hist = _torso_hist(
                frame_bgr, det_bbox,
                cfg["torso_fraction"],
                cfg["hist_bins_h"], cfg["hist_bins_s"], cfg["hist_bins_v"],
                cfg["min_torso_pixels"],
            )
        else:
            color_hist = None

        # First ever detection — initialise state
        if self._state is None:
            pid = self._next_persistent_id
            self._next_persistent_id += 1
            self._state = _HandlerState(pid, det_id, det_bbox, kp_vec, color_hist)
            logger.debug("f%04d  handler INIT  P%s → H%d", frame_idx, det_id, pid)
            return {
                "frame_idx":             frame_idx,
                "persistent_handler_id": pid,
                "raw_detector_id":       det_id,
                "handler_id_source":     "initial",
                "reid_score":            None,
            }

        # Raw ID unchanged — update state, reset challenger
        if det_id == self._state.detector_id:
            self._state.bbox       = det_bbox
            self._state.kp_vec     = kp_vec
            self._state.color_hist = color_hist
            self._challenger_det_id   = None
            self._challenger_frames   = 0
            return {
                "frame_idx":             frame_idx,
                "persistent_handler_id": self._state.persistent_id,
                "raw_detector_id":       det_id,
                "handler_id_source":     "held",
                "reid_score":            None,
            }

        # Raw ID changed — compute re-id score
        reid = self._reid_score(det_bbox, kp_vec, color_hist, self._state)

        if reid >= cfg["relabel_threshold"]:
            # Same physical player, new detector token → relabel
            old_det = self._state.detector_id
            self._state.detector_id = det_id
            self._state.bbox        = det_bbox
            self._state.kp_vec      = kp_vec
            self._state.color_hist  = color_hist
            self._challenger_det_id  = None
            self._challenger_frames  = 0
            logger.info(
                "f%04d  ID RELABEL   raw P%s→P%s  persistent H%d  (reid=%.3f)",
                frame_idx, old_det, det_id, self._state.persistent_id, reid,
            )
            return {
                "frame_idx":             frame_idx,
                "persistent_handler_id": self._state.persistent_id,
                "raw_detector_id":       det_id,
                "handler_id_source":     "relabel",
                "reid_score":            round(reid, 4),
            }

        # Low re-id score — may be a true switch, apply hysteresis
        if det_id == self._challenger_det_id:
            self._challenger_frames += 1
        else:
            self._challenger_det_id  = det_id
            self._challenger_frames  = 1

        if self._challenger_frames >= cfg["switch_hysteresis_frames"]:
            # True possession switch
            old_pid = self._state.persistent_id
            new_pid = self._next_persistent_id
            self._next_persistent_id += 1
            self._state = _HandlerState(new_pid, det_id, det_bbox, kp_vec, color_hist)
            self._challenger_det_id  = None
            self._challenger_frames  = 0
            logger.info(
                "f%04d  SWITCH       raw P%s  H%d→H%d  (reid=%.3f)",
                frame_idx, det_id, old_pid, new_pid, reid,
            )
            return {
                "frame_idx":             frame_idx,
                "persistent_handler_id": new_pid,
                "raw_detector_id":       det_id,
                "handler_id_source":     "switch",
                "reid_score":            round(reid, 4),
            }

        # Still in hysteresis window — hold current persistent ID
        return {
            "frame_idx":             frame_idx,
            "persistent_handler_id": self._state.persistent_id,
            "raw_detector_id":       det_id,
            "handler_id_source":     "held_hysteresis",
            "reid_score":            round(reid, 4),
        }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def track_handler_identity(
    data:       Dict[str, Any],
    video_path: str,
    config:     Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Post-process possession results with persistent handler identity.

    Parameters
    ----------
    data:        The enriched pipeline dict (must have "possession" and
                 "player_selection" keys).
    video_path:  Path to the original video (used to sample torso colour).
    config:      Optional overrides for _DEFAULT_CONFIG.

    Returns the same *data* dict with a new top-level key "handler_identity":
    {
        "per_frame": [ {frame_idx, persistent_handler_id, raw_detector_id,
                        handler_id_source, reid_score}, ... ],
        "summary": { n_relabels, n_switches, n_frames_held, n_frames_no_possession },
        "config": { ... },
    }
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    poss_per_frame = data.get("possession", {}).get("per_frame", [])
    ps_per_frame   = data.get("player_selection", {}).get("per_frame", [])

    # Build per-frame player_selection index: frame_idx → list of player dicts
    ps_index: Dict[int, List[Dict]] = {}
    for entry in ps_per_frame:
        fidx = entry.get("frame_idx")
        if fidx is not None:
            ps_index[fidx] = entry.get("players", [])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video for colour sampling: %s", video_path)
        cap = None

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if cap else 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 480

    tracker = HandlerIdentityTracker(config=cfg)
    tracker.initialize(W, H)

    results: List[Dict] = []
    n_relabels      = 0
    n_switches      = 0
    n_held          = 0
    n_no_possession = 0

    # We need to iterate frames in order and read video frames to get colour.
    # Build a sorted index of frame_idx values from possession data.
    frame_indices = sorted({r["frame_idx"] for r in poss_per_frame
                            if "frame_idx" in r})

    # Build possession lookup
    poss_index = {r["frame_idx"]: r for r in poss_per_frame}

    video_frame_idx = 0  # tracks how far we've read into the video

    for fidx in frame_indices:
        # Seek video to this frame if cap is open
        frame_bgr: Optional[np.ndarray] = None
        if cap is not None:
            if video_frame_idx != fidx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                video_frame_idx = fidx
            ret, frame_bgr = cap.read()
            video_frame_idx += 1

        poss_rec  = poss_index.get(fidx, {})
        ps_players = ps_index.get(fidx, [])

        result = tracker.update(fidx, poss_rec, ps_players, frame_bgr)
        results.append(result)

        src = result["handler_id_source"]
        if src == "relabel":
            n_relabels += 1
        elif src == "switch":
            n_switches += 1
        elif src in ("held", "held_hysteresis"):
            n_held += 1
        elif src == "no_possession":
            n_no_possession += 1

    if cap is not None:
        cap.release()

    summary = {
        "n_relabels":         n_relabels,
        "n_switches":         n_switches,
        "n_frames_held":      n_held,
        "n_frames_no_possession": n_no_possession,
        "n_frames_total":     len(results),
    }
    logger.info(
        "Handler identity: %d relabels, %d switches, %d held, %d no-possession "
        "(out of %d frames)",
        n_relabels, n_switches, n_held, n_no_possession, len(results),
    )

    data["handler_identity"] = {
        "per_frame": results,
        "summary":   summary,
        "config":    cfg,
    }
    return data
