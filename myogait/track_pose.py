"""Per-track pose state and hard-frame pose refinement.

Problem
-------
In overlap / contact / occlusion frames the full-frame YOLO person-detector
can assign keypoints that are contaminated by an adjacent player's body.
``select_target_player`` already writes track-specific keypoints into
``data["frames"][i]["landmarks"]``, but those keypoints come from a
full-frame inference where two bbox regions overlap – so YOLO may fit
skeleton joints to the wrong person.

This module adds three layers on top:

1. **HardFrameDetector** – flags a frame as "hard" when one or more signals
   suggest the global pose cannot be trusted for the selected handler:
   - selected-handler bbox overlaps another player by > threshold
   - ball is near multiple players simultaneously
   - raw pose jumps too far from the track's own smoothed history
   - lower-body keypoints are mostly missing / low-confidence

2. **TrackPoseState** – each ByteTrack track owns a rolling pose state:
   - raw observations are blended via EMA to produce a smoothed pose
   - carry-forward kicks in when no trustworthy observation is available
   - the state remembers the source for each frame (global / crop / carry)

3. **Crop-pose refinement** – in hard frames, the selected handler's bbox is
   used to crop the frame, YOLO pose is re-run on the crop (more isolated),
   the result is mapped back to full-frame coordinates, and – if confident –
   replaces the contaminated full-frame keypoints.

Entry point
-----------
``refine_handler_pose(data, video_path, config, person_detector)``
    Post-processes ``data["frames"][i]["landmarks"]`` for every frame that
    has a selected handler.  Adds ``data["track_pose"]`` with per-frame
    diagnostics.  Returns updated data dict.

Data flow
---------
select_target_player()
    └─ extract_selected_player_pose()   # writes raw YOLO kp → landmarks
refine_handler_pose()                   # reads landmarks, refines in-place
    ├─ normal frame  → EMA-smooth current obs, write back
    ├─ hard frame    → crop + YOLO → map to full-frame → EMA-smooth, write back
    └─ unrecoverable → carry-forward smoothed state, write back
track_possession()                      # consumes refined landmarks
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Joint definitions (subset of MediaPipe that COCO-17 covers)
# ---------------------------------------------------------------------------

# Ordered list used as a consistent internal array index
_MP_JOINTS: List[str] = [
    "NOSE",
    "LEFT_SHOULDER",  "RIGHT_SHOULDER",
    "LEFT_ELBOW",     "RIGHT_ELBOW",
    "LEFT_WRIST",     "RIGHT_WRIST",
    "LEFT_HIP",       "RIGHT_HIP",
    "LEFT_KNEE",      "RIGHT_KNEE",
    "LEFT_ANKLE",     "RIGHT_ANKLE",
]
_N_JOINTS = len(_MP_JOINTS)
_MP_JOINT_IDX = {name: i for i, name in enumerate(_MP_JOINTS)}

# COCO-17 index → MediaPipe name
_COCO_TO_MP: Dict[int, str] = {
    0:  "NOSE",
    5:  "LEFT_SHOULDER",   6:  "RIGHT_SHOULDER",
    7:  "LEFT_ELBOW",      8:  "RIGHT_ELBOW",
    9:  "LEFT_WRIST",      10: "RIGHT_WRIST",
    11: "LEFT_HIP",        12: "RIGHT_HIP",
    13: "LEFT_KNEE",       14: "RIGHT_KNEE",
    15: "LEFT_ANKLE",      16: "RIGHT_ANKLE",
}

# Lower-body joint indices within _MP_JOINTS
_LOWER_BODY_IDXS = [
    _MP_JOINT_IDX["LEFT_HIP"],   _MP_JOINT_IDX["RIGHT_HIP"],
    _MP_JOINT_IDX["LEFT_KNEE"],  _MP_JOINT_IDX["RIGHT_KNEE"],
    _MP_JOINT_IDX["LEFT_ANKLE"], _MP_JOINT_IDX["RIGHT_ANKLE"],
]


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    # ── Hard-frame detection ─────────────────────────────────────────────────
    # IoU between selected handler and ANY other player bbox → overlap flag
    "overlap_iou_threshold":       0.15,
    # Mean normalised keypoint displacement from smoothed history → jump flag
    "pose_jump_threshold":         0.08,
    # Lower-body kp visibility below this is counted as "missing"
    "min_lower_body_conf":         0.30,
    # More than this many missing lower-body kps → lower-body-instability flag
    "max_bad_lower_body_kps":      3,
    # Ball within this many pixels of N+ players → multi-proximity flag
    "ball_proximity_radius_px":    150.0,
    "ball_multi_player_count":     2,

    # ── Crop-pose refinement (used in hard frames) ───────────────────────────
    # Expand the track bbox by this fraction on each side before cropping
    "crop_padding_ratio":          0.20,
    # Skip crop refinement if either crop dimension is below this (pixels)
    "crop_min_size_px":            80,
    # Reject refined pose if mean visible-kp confidence < this
    "min_refined_pose_conf":       0.30,
    # Person detector used for crop inference
    "person_detector":             "yolo_pose",
    "person_detector_kwargs": {
        "model_path":           "yolov8n-pose.pt",
        "confidence_threshold": 0.15,   # lower: isolated crops have fewer dets
        "min_area_frac":        0.0,    # no area filter on crops
    },

    # ── Per-track pose state (smoothing + carry-forward) ─────────────────────
    # EMA weight for new observations (1.0 = no smoothing, 0.0 = full carry)
    "pose_smoothing_alpha":        0.65,
    # Minimum observation confidence to be blended (below → position kept)
    "pose_smoothing_min_conf":     0.30,
    # Maximum consecutive frames to carry the smoothed pose forward
    "carry_forward_max_frames":    8,
    # Confidence decay per carry-forward frame
    "carry_forward_conf_decay":    0.90,
    # History length for jump detection
    "history_length":              5,
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lm_dict_to_array(lm: Dict) -> np.ndarray:
    """MediaPipe landmark dict → ``(_N_JOINTS, 3)`` float32 array [x, y, vis]."""
    arr = np.zeros((_N_JOINTS, 3), dtype=np.float32)
    for name, idx in _MP_JOINT_IDX.items():
        kp = lm.get(name)
        if kp:
            arr[idx, 0] = float(kp.get("x", 0.0))
            arr[idx, 1] = float(kp.get("y", 0.0))
            arr[idx, 2] = float(kp.get("visibility", 0.0))
    return arr


def _array_to_lm_dict(arr: np.ndarray) -> Dict:
    """``(_N_JOINTS, 3)`` float32 → MediaPipe landmark dict."""
    result: Dict = {}
    for name, idx in _MP_JOINT_IDX.items():
        x, y, v = float(arr[idx, 0]), float(arr[idx, 1]), float(arr[idx, 2])
        if v > 0.01:
            result[name] = {"x": round(x, 6), "y": round(y, 6),
                            "visibility": round(v, 4)}
    return result


def _coco17_crop_to_array(
    kp_coco: np.ndarray,        # (17, 3) keypoints normalised to crop dims
    crop_x1: int, crop_y1: int,
    crop_w:  int, crop_h:  int,
    frame_w: int, frame_h: int,
) -> np.ndarray:
    """Map COCO-17 crop-normalised keypoints → ``(_N_JOINTS, 3)`` full-frame."""
    arr = np.zeros((_N_JOINTS, 3), dtype=np.float32)
    for coco_idx, mp_name in _COCO_TO_MP.items():
        if coco_idx >= len(kp_coco):
            continue
        x_crop, y_crop, conf = (float(kp_coco[coco_idx, 0]),
                                float(kp_coco[coco_idx, 1]),
                                float(kp_coco[coco_idx, 2]))
        x_full = (x_crop * crop_w + crop_x1) / max(frame_w, 1)
        y_full = (y_crop * crop_h + crop_y1) / max(frame_h, 1)
        arr[_MP_JOINT_IDX[mp_name]] = [x_full, y_full, conf]
    return arr


def _mean_visible_conf(arr: np.ndarray, min_conf: float = 0.01) -> float:
    """Mean confidence of visible keypoints in a (_N_JOINTS, 3) array."""
    vis = arr[:, 2]
    mask = vis > min_conf
    if mask.sum() == 0:
        return 0.0
    return float(vis[mask].mean())


def _pose_jump(arr_new: np.ndarray, arr_prev: np.ndarray,
               min_conf: float = 0.30) -> float:
    """Mean normalised L2 displacement for keypoints visible in both poses."""
    both = (arr_new[:, 2] > min_conf) & (arr_prev[:, 2] > min_conf)
    if both.sum() < 3:
        return 0.0
    diff = arr_new[both, :2] - arr_prev[both, :2]
    return float(np.mean(np.sqrt((diff ** 2).sum(axis=1))))


def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(union, 1e-6)


# ---------------------------------------------------------------------------
# Hard-frame detector
# ---------------------------------------------------------------------------

class HardFrameDetector:
    """Flags frames where the global full-frame pose cannot be trusted.

    Returns a ``(is_hard, reasons)`` tuple where ``reasons`` is a dict of
    individual signal flags (for debug overlay).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config

    def check(
        self,
        handler_bbox:    List[int],           # selected handler smoothed bbox
        other_bboxes:    List[List[int]],     # all other player bboxes
        ball_center:     Optional[Tuple],     # (cx, cy) in pixels, or None
        all_player_wrist_px: List[Optional[Tuple]],  # wrist positions for all players
        current_arr:     np.ndarray,          # (_N_JOINTS,3) current raw obs
        smoothed_arr:    Optional[np.ndarray],# (_N_JOINTS,3) track smoothed state
    ) -> Tuple[bool, Dict[str, bool]]:
        cfg = self._cfg
        reasons: Dict[str, bool] = {
            "overlap":        False,
            "pose_jump":      False,
            "lower_body":     False,
            "ball_proximity": False,
        }

        # 1. Bbox overlap with other players
        for ob in other_bboxes:
            if _iou(tuple(handler_bbox), tuple(ob)) > cfg["overlap_iou_threshold"]:
                reasons["overlap"] = True
                break

        # 2. Pose jump vs smoothed history
        if smoothed_arr is not None:
            jump = _pose_jump(current_arr, smoothed_arr,
                              cfg["pose_smoothing_min_conf"])
            if jump > cfg["pose_jump_threshold"]:
                reasons["pose_jump"] = True

        # 3. Lower-body instability
        lb_conf = current_arr[_LOWER_BODY_IDXS, 2]
        n_bad   = int((lb_conf < cfg["min_lower_body_conf"]).sum())
        if n_bad > cfg["max_bad_lower_body_kps"]:
            reasons["lower_body"] = True

        # 4. Ball near multiple players
        if ball_center is not None:
            radius = cfg["ball_proximity_radius_px"]
            close  = sum(
                1 for wp in all_player_wrist_px
                if wp is not None
                and math.sqrt((wp[0] - ball_center[0]) ** 2 +
                              (wp[1] - ball_center[1]) ** 2) <= radius
            )
            if close >= cfg["ball_multi_player_count"]:
                reasons["ball_proximity"] = True

        is_hard = any(reasons.values())
        return is_hard, reasons


# ---------------------------------------------------------------------------
# Per-track pose state
# ---------------------------------------------------------------------------

class TrackPoseState:
    """Owns the authoritative pose estimate for one ByteTrack track.

    Each call to ``observe()`` or ``carry_forward()`` returns the final
    keypoint array and the source label that should be written to
    ``data["frames"]``.
    """

    def __init__(self, track_id: int, config: Dict[str, Any]) -> None:
        self.track_id       = track_id
        self._alpha         = config["pose_smoothing_alpha"]
        self._min_conf      = config["pose_smoothing_min_conf"]
        self._max_carry     = config["carry_forward_max_frames"]
        self._decay         = config["carry_forward_conf_decay"]

        self._smoothed:     Optional[np.ndarray] = None   # (_N_JOINTS, 3)
        self._carry_count:  int                  = 0
        self._last_valid:   int                  = -1

    # ── Public API ──────────────────────────────────────────────────────────

    def observe(self, raw_arr: np.ndarray, frame_idx: int) -> np.ndarray:
        """Accept a new raw observation; return the EMA-smoothed pose."""
        if self._smoothed is None:
            self._smoothed = raw_arr.copy()
        else:
            alpha = self._alpha
            conf  = raw_arr[:, 2:3]          # (N, 1)
            high  = conf >= self._min_conf   # (N, 1) bool

            # Blend positions for confident keypoints; keep previous otherwise
            blended_xy = np.where(
                high,
                alpha * raw_arr[:, :2] + (1.0 - alpha) * self._smoothed[:, :2],
                self._smoothed[:, :2],
            )
            blended_conf = alpha * raw_arr[:, 2] + (1.0 - alpha) * self._smoothed[:, 2]

            self._smoothed = np.concatenate(
                [blended_xy, blended_conf[:, None]], axis=1
            ).astype(np.float32)

        self._carry_count = 0
        self._last_valid  = frame_idx
        return self._smoothed.copy()

    def carry_forward(self) -> Tuple[Optional[np.ndarray], bool]:
        """Return the last smoothed pose with decayed confidence.

        Returns ``(array, ok)`` where ``ok`` is False when max carry exceeded.
        """
        self._carry_count += 1
        if self._smoothed is None or self._carry_count > self._max_carry:
            return None, False
        decayed = self._smoothed.copy()
        decayed[:, 2] *= self._decay ** self._carry_count
        return decayed, True

    def current_smoothed(self) -> Optional[np.ndarray]:
        """Return current smoothed array without advancing carry counter."""
        return self._smoothed.copy() if self._smoothed is not None else None

    def pose_jump_from_smoothed(self, raw_arr: np.ndarray) -> float:
        if self._smoothed is None:
            return 0.0
        return _pose_jump(raw_arr, self._smoothed, self._min_conf)


# ---------------------------------------------------------------------------
# Crop-pose inference
# ---------------------------------------------------------------------------

def _best_person_in_crop(
    detections,                  # List[PersonDetection] from YOLO on crop
    crop_cx: float, crop_cy: float,
) -> Optional[Any]:
    """Pick the detection whose centre is closest to the crop centre."""
    if not detections:
        return None
    best, best_d = None, float("inf")
    for det in detections:
        cx = (det.bbox[0] + det.bbox[2]) / 2.0
        cy = (det.bbox[1] + det.bbox[3]) / 2.0
        d  = math.sqrt((cx - crop_cx) ** 2 + (cy - crop_cy) ** 2)
        if d < best_d:
            best_d, best = d, det
    return best


def _run_crop_pose(
    frame_bgr:   np.ndarray,
    handler_bbox: List[int],       # [x1, y1, x2, y2] in pixels
    person_detector,
    config:      Dict[str, Any],
    frame_w:     int,
    frame_h:     int,
) -> Optional[np.ndarray]:
    """Run YOLO pose on a padded crop of the handler bbox.

    Returns a ``(_N_JOINTS, 3)`` array in full-frame normalised coordinates,
    or ``None`` if the crop is too small or the refined pose is low-confidence.
    """
    pad   = config["crop_padding_ratio"]
    x1, y1, x2, y2 = handler_bbox

    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(frame_w, x2 + px)
    cy2 = min(frame_h, y2 + py)

    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    min_sz = config["crop_min_size_px"]
    if crop_w < min_sz or crop_h < min_sz:
        return None

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    dets = person_detector.detect(crop)
    if not dets:
        return None

    # The target player should dominate the crop; pick by centre proximity
    crop_cx_norm = crop_w / 2.0
    crop_cy_norm = crop_h / 2.0
    best = _best_person_in_crop(dets, crop_cx_norm, crop_cy_norm)
    if best is None or best.keypoints_norm is None:
        return None
    if best.keypoints_norm.shape != (17, 3):
        return None

    arr = _coco17_crop_to_array(
        best.keypoints_norm,
        cx1, cy1, crop_w, crop_h, frame_w, frame_h,
    )
    mean_conf = _mean_visible_conf(arr)
    if mean_conf < config["min_refined_pose_conf"]:
        logger.debug("  crop pose rejected: mean_conf=%.3f < %.3f",
                     mean_conf, config["min_refined_pose_conf"])
        return None

    return arr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def refine_handler_pose(
    data:            Dict[str, Any],
    video_path:      str,
    config:          Optional[Dict[str, Any]] = None,
    person_detector  = None,
) -> Dict[str, Any]:
    """Refine selected-handler pose with per-track state and crop inference.

    Reads
    -----
    - ``data["player_selection"]["per_frame"]``  – target bbox, track ID, players
    - ``data["ball"]["per_frame"]``               – ball position
    - ``data["frames"][i]["landmarks"]``          – raw global pose (already
      track-specific from ``extract_selected_player_pose``)

    Writes
    ------
    - ``data["frames"][i]["landmarks"]``          – **replaced** with the
      track-owned final pose (smoothed, refined crop, or carry-forward)
    - ``data["track_pose"]``                      – per-frame diagnostics

    Parameters
    ----------
    data :           The enriched pivot dict.
    video_path :     Path to the source video (for crop frame reads).
    config :         Optional overrides for ``_DEFAULT_CONFIG``.
    person_detector: Pre-built YOLOPersonDetector (built from config if None).

    Returns
    -------
    dict   The same *data* with updated landmarks and new ``track_pose`` key.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    ps_per_frame    = data.get("player_selection", {}).get("per_frame", [])
    ball_per_frame  = data.get("ball", {}).get("per_frame", [])
    frames_data     = data.get("frames", [])
    meta            = data.get("meta", {})
    frame_w         = meta.get("width",  1920)
    frame_h         = meta.get("height", 1080)

    if not ps_per_frame:
        logger.warning("track_pose: no player_selection data; skipping refinement")
        return data

    # ── Build indices ────────────────────────────────────────────────────────
    ball_index: Dict[int, Dict] = {
        r["frame_idx"]: r.get("ball", {})
        for r in ball_per_frame
        if "frame_idx" in r
    }

    # ── Ambiguity index (if available from compute_ambiguity step) ───────────
    ambiguity_index: Dict[int, Dict] = {
        r["frame_idx"]: r
        for r in data.get("ambiguity", {}).get("per_frame", [])
        if "frame_idx" in r
    }
    use_ambiguity = bool(ambiguity_index)
    if use_ambiguity:
        logger.info("track_pose: routing via ambiguity stage (%d records)", len(ambiguity_index))
    else:
        logger.info("track_pose: no ambiguity data; using internal HardFrameDetector")

    # ── Build person detector if needed ─────────────────────────────────────
    if person_detector is None:
        from .detectors.person_detector import create_person_detector
        person_detector = create_person_detector(
            cfg["person_detector"],
            **cfg["person_detector_kwargs"],
        )
        logger.info("track_pose: built crop person detector (%s)",
                    cfg["person_detector"])

    # ── Open video for frame reads ───────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("track_pose: cannot open video %s", video_path)
        return data

    # ── Per-track state registry ─────────────────────────────────────────────
    track_states:  Dict[int, TrackPoseState] = {}
    hard_detector  = HardFrameDetector(cfg)

    # ── Diagnostic counters ──────────────────────────────────────────────────
    n_hard_overlap  = 0
    n_hard_jump     = 0
    n_hard_lower    = 0
    n_hard_ball     = 0
    n_crop_ok       = 0
    n_crop_fail     = 0
    n_carry         = 0
    n_normal        = 0

    per_frame_diag: List[Dict] = []
    video_frame_idx = 0   # tracks current video position

    for ps_entry in ps_per_frame:
        fidx       = ps_entry.get("frame_idx")
        track_id   = ps_entry.get("target_track_id")
        bbox_raw   = ps_entry.get("smoothed_bbox") or ps_entry.get("target_bbox")
        players    = ps_entry.get("players", [])

        diag: Dict[str, Any] = {
            "frame_idx":       fidx,
            "track_id":        track_id,
            "is_hard":         False,
            "reasons":         {},
            "pose_source":     "no_handler",
            "pose_conf":       None,
            "crop_attempted":  False,
            "crop_accepted":   False,
        }

        # No selected player this frame
        if fidx is None or track_id is None or bbox_raw is None:
            per_frame_diag.append(diag)
            continue

        handler_bbox = list(bbox_raw)

        # Ensure we have current raw landmarks
        if fidx >= len(frames_data):
            per_frame_diag.append(diag)
            continue
        raw_lm = frames_data[fidx].get("landmarks") or {}
        if not raw_lm:
            per_frame_diag.append(diag)
            continue

        # ── Convert raw landmarks to array for analysis ──────────────────────
        raw_arr = _lm_dict_to_array(raw_lm)

        # ── Ensure track state exists ────────────────────────────────────────
        if track_id not in track_states:
            track_states[track_id] = TrackPoseState(track_id, cfg)

        state = track_states[track_id]

        # ── Gather other player bboxes ───────────────────────────────────────
        other_bboxes = [
            p["bbox"] for p in players
            if p.get("bbox") is not None
            and p.get("track_id") != track_id
        ]

        # ── Ball center + wrist positions of all players ─────────────────────
        ball_dict  = ball_index.get(fidx, {})
        ball_ctr   = None
        if ball_dict.get("tracked") and ball_dict.get("center"):
            bc = ball_dict["center"]
            ball_ctr = (float(bc[0]), float(bc[1]))

        all_wrist_px: List[Optional[Tuple]] = []
        for p in players:
            lw = p.get("left_wrist_norm")
            rw = p.get("right_wrist_norm")
            if lw:
                all_wrist_px.append((lw[0] * frame_w, lw[1] * frame_h))
            elif rw:
                all_wrist_px.append((rw[0] * frame_w, rw[1] * frame_h))
            else:
                all_wrist_px.append(None)

        # ── Hard-frame / routing decision ────────────────────────────────────
        # Prefer ambiguity-stage routing if available; fall back to internal
        # HardFrameDetector otherwise.
        pose_mode: str   = "normal"   # "normal" | "refined" | "carry_forward"
        is_hard:   bool  = False
        reasons:   Dict[str, bool] = {
            "overlap": False, "pose_jump": False,
            "lower_body": False, "ball_proximity": False,
        }

        if use_ambiguity:
            amb_rec  = ambiguity_index.get(fidx, {})
            pose_mode = amb_rec.get("recommended_pose_mode", "normal")
            amb_state = amb_rec.get("state", "clear")
            is_hard   = pose_mode != "normal"
            # Populate reasons from ambiguity signals for the diag / overlay
            signals = amb_rec.get("signals", {})
            cfg_thr  = data.get("ambiguity", {}).get("config", {})
            ov_thr   = cfg_thr.get("bbox_overlap_threshold", 0.15)
            jmp_thr  = cfg_thr.get("pose_jump_threshold", 0.06)
            reasons["overlap"]   = signals.get("bbox_overlap",   0.0) > ov_thr
            reasons["pose_jump"] = signals.get("pose_jump",      0.0) > jmp_thr
            lb_inst  = signals.get("lower_body_instability", 0.0)
            lb_frac  = cfg_thr.get("lower_body_fraction_bad", 0.50)
            reasons["lower_body"] = lb_inst > lb_frac
            # add ambiguity state to diag for renderer
            diag["ambiguity_state"] = amb_state
            diag["ambiguity_score"] = amb_rec.get("score", 0.0)
        else:
            is_hard, reasons = hard_detector.check(
                handler_bbox, other_bboxes, ball_ctr,
                all_wrist_px, raw_arr, state.current_smoothed(),
            )
            pose_mode = "carry_forward" if is_hard else "normal"
            # For internal detection: prefer refined crop over carry when overlap/jump
            if is_hard and (reasons["overlap"] or reasons["pose_jump"]):
                pose_mode = "refined"
            diag["ambiguity_state"] = "occluded" if pose_mode == "carry_forward" else (
                "overlap" if is_hard else "clear"
            )
            diag["ambiguity_score"] = 1.0 if not is_hard else 0.5

        diag["is_hard"] = is_hard
        diag["reasons"] = reasons

        if reasons["overlap"]:      n_hard_overlap += 1
        if reasons["pose_jump"]:    n_hard_jump    += 1
        if reasons["lower_body"]:   n_hard_lower   += 1
        if reasons.get("ball_proximity"): n_hard_ball += 1

        # ── Determine final pose ─────────────────────────────────────────────
        final_arr: Optional[np.ndarray] = None
        source: str = "global_pose"

        if pose_mode == "normal":
            # Normal frame — trust the current observation, update state
            final_arr = state.observe(raw_arr, fidx)
            source    = "global_pose"
            n_normal += 1

        elif pose_mode == "refined":
            # Prefer crop-pose; fall back to carry-forward then raw global
            if video_frame_idx != fidx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                video_frame_idx = fidx

            ret, frame_bgr = cap.read()
            video_frame_idx += 1

            refined_arr: Optional[np.ndarray] = None
            if ret and frame_bgr is not None:
                diag["crop_attempted"] = True
                refined_arr = _run_crop_pose(
                    frame_bgr, handler_bbox, person_detector, cfg, frame_w, frame_h,
                )

            if refined_arr is not None:
                final_arr = state.observe(refined_arr, fidx)
                source    = "refined_crop_pose"
                diag["crop_accepted"] = True
                n_crop_ok += 1
                logger.debug("f%04d  T%d  crop pose accepted (conf=%.3f)",
                             fidx, track_id, _mean_visible_conf(refined_arr))
            else:
                if ret:
                    n_crop_fail += 1
                carried, ok = state.carry_forward()
                if ok and carried is not None:
                    final_arr = carried
                    source    = "carry_forward"
                    n_carry  += 1
                    logger.debug("f%04d  T%d  carry-forward after crop fail (count=%d)",
                                 fidx, track_id, state._carry_count)
                else:
                    final_arr = state.observe(raw_arr, fidx)
                    source    = "global_pose_fallback"
                    logger.debug("f%04d  T%d  carry exceeded; using raw global", fidx, track_id)

        else:  # pose_mode == "carry_forward"
            # Ambiguity stage says: do not commit new pose observation
            carried, ok = state.carry_forward()
            if ok and carried is not None:
                final_arr = carried
                source    = "carry_forward"
                n_carry  += 1
                logger.debug("f%04d  T%d  carry-forward (ambiguity=%s, count=%d)",
                             fidx, track_id,
                             diag.get("ambiguity_state", "?"), state._carry_count)
            else:
                # Carry limit exceeded — attempt crop as last resort
                if video_frame_idx != fidx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    video_frame_idx = fidx
                ret, frame_bgr = cap.read()
                video_frame_idx += 1
                if ret and frame_bgr is not None:
                    diag["crop_attempted"] = True
                    crop_last = _run_crop_pose(
                        frame_bgr, handler_bbox, person_detector, cfg, frame_w, frame_h,
                    )
                    if crop_last is not None:
                        final_arr = state.observe(crop_last, fidx)
                        source    = "refined_crop_pose"
                        diag["crop_accepted"] = True
                        n_crop_ok += 1
                    else:
                        n_crop_fail += 1
                if final_arr is None:
                    final_arr = state.observe(raw_arr, fidx)
                    source    = "global_pose_fallback"
                    logger.debug("f%04d  T%d  carry exceeded + crop fail; using raw", fidx, track_id)

        # ── Write final pose back into data["frames"] ────────────────────────
        if final_arr is not None:
            frames_data[fidx]["landmarks"] = _array_to_lm_dict(final_arr)
            frames_data[fidx]["landmarks_source"] = source
            pose_conf = _mean_visible_conf(final_arr)
            diag["pose_source"] = source
            diag["pose_conf"]   = round(pose_conf, 4)

        per_frame_diag.append(diag)

    cap.release()

    # ── Summary stats ────────────────────────────────────────────────────────
    n_hard = sum(1 for d in per_frame_diag if d["is_hard"])
    summary = {
        "n_frames":          len(per_frame_diag),
        "n_hard_frames":     n_hard,
        "n_hard_overlap":    n_hard_overlap,
        "n_hard_jump":       n_hard_jump,
        "n_hard_lower_body": n_hard_lower,
        "n_hard_ball":       n_hard_ball,
        "n_normal":          n_normal,
        "n_crop_ok":         n_crop_ok,
        "n_crop_fail":       n_crop_fail,
        "n_carry_forward":   n_carry,
    }
    logger.info(
        "track_pose: %d/%d hard frames "
        "(overlap=%d jump=%d lower=%d ball=%d) | "
        "crop_ok=%d fail=%d carry=%d normal=%d",
        n_hard, len(per_frame_diag),
        n_hard_overlap, n_hard_jump, n_hard_lower, n_hard_ball,
        n_crop_ok, n_crop_fail, n_carry, n_normal,
    )

    data["track_pose"] = {
        "config":    cfg,
        "summary":   summary,
        "per_frame": per_frame_diag,
    }
    return data
