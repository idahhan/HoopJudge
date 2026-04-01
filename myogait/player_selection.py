"""Multi-person target-player selection for basketball clips.

Pipeline
--------
1. Detect all visible persons per frame using a YOLO pose model.
2. Score each person against the tracked ball position using:
   - distance from ball center to left/right wrist          (w_wrist=0.40)
   - distance from ball center to left/right hand proxy     (w_hand_proxy=0.15)
   - whether the ball centre lies inside the player's bbox  (w_inside=0.20)
   - normalised distance from ball to nearest bbox edge     (w_edge=0.10)
   - temporal continuity with the previous assignment       (w_temporal=0.15)
3. Select the highest-scoring person as the target player for each frame.
4. Apply temporal smoothing (sliding-window IoU consensus + gap fill).
5. Extract selected-player pose: convert target player's COCO-17 keypoints
   to MediaPipe-format landmarks and write into data["frames"][i]["landmarks"].
6. Optionally run foot-event detection (detect_events) on those landmarks.

Entry points
------------
select_target_player(video_path, data, config, person_detector)
    Full pipeline.  Writes ``data["player_selection"]`` (and optionally
    updates ``data["frames"]`` landmarks + ``data["events"]``).

extract_selected_player_pose(data, assignments)
    Convert target player COCO-17 keypoints → MediaPipe landmarks and write
    into data["frames"].  Call after select_target_player().

run_selected_player_analysis(data, config)
    Run detect_events() on the selected-player landmarks.

render_player_selection_video(video_path, data, output_path, ...)
    Debug video: all players, target (green), skeleton overlay,
    wrists + hand proxies, ball, foot events.

score_player_for_ball(person, ball, frame_w, frame_h, ...)
    Public scoring function (returns float in [0,1]).

coco17_to_landmarks(kp_norm)
    Convert (17, 3) COCO keypoint array to MediaPipe-format landmark dict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detectors.ball_detector import BallDetection
from .detectors.person_detector import (
    PersonDetection,
    PersonDetector,
    create_person_detector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO-17 keypoint indices used in scoring and rendering
# ---------------------------------------------------------------------------
_COCO_LEFT_SHOULDER  = 5
_COCO_RIGHT_SHOULDER = 6
_COCO_LEFT_ELBOW     = 7
_COCO_RIGHT_ELBOW    = 8
_COCO_LEFT_WRIST     = 9
_COCO_RIGHT_WRIST    = 10
_COCO_LEFT_HIP       = 11
_COCO_RIGHT_HIP      = 12
_COCO_LEFT_KNEE      = 13
_COCO_RIGHT_KNEE     = 14
_COCO_LEFT_ANKLE     = 15
_COCO_RIGHT_ANKLE    = 16

# Skeleton connections for debug overlay (index pairs)
_SKELETON_UPPER = [
    (_COCO_LEFT_SHOULDER,  _COCO_RIGHT_SHOULDER),
    (_COCO_LEFT_SHOULDER,  _COCO_LEFT_ELBOW),
    (_COCO_LEFT_ELBOW,     _COCO_LEFT_WRIST),
    (_COCO_RIGHT_SHOULDER, _COCO_RIGHT_ELBOW),
    (_COCO_RIGHT_ELBOW,    _COCO_RIGHT_WRIST),
]
_SKELETON_TORSO = [
    (_COCO_LEFT_SHOULDER, _COCO_LEFT_HIP),
    (_COCO_RIGHT_SHOULDER, _COCO_RIGHT_HIP),
    (_COCO_LEFT_HIP,       _COCO_RIGHT_HIP),
]
_SKELETON_LOWER = [
    (_COCO_LEFT_HIP,   _COCO_LEFT_KNEE),
    (_COCO_LEFT_KNEE,  _COCO_LEFT_ANKLE),
    (_COCO_RIGHT_HIP,  _COCO_RIGHT_KNEE),
    (_COCO_RIGHT_KNEE, _COCO_RIGHT_ANKLE),
]

# Minimum keypoint confidence to use in scoring / rendering
_MIN_KP_CONF = 0.30


# ---------------------------------------------------------------------------
# COCO-17 → MediaPipe landmark name mapping
# (only joints we can derive; downstream code handles missing ones gracefully)
# ---------------------------------------------------------------------------
_COCO_TO_MP_NAME: Dict[int, str] = {
    0:  "NOSE",
    5:  "LEFT_SHOULDER",
    6:  "RIGHT_SHOULDER",
    7:  "LEFT_ELBOW",
    8:  "RIGHT_ELBOW",
    9:  "LEFT_WRIST",
    10: "RIGHT_WRIST",
    11: "LEFT_HIP",
    12: "RIGHT_HIP",
    13: "LEFT_KNEE",
    14: "RIGHT_KNEE",
    15: "LEFT_ANKLE",
    16: "RIGHT_ANKLE",
}


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- Person detector ---------------------------------------------------
    "person_detector": "yolo_pose",
    "person_detector_kwargs": {
        "model_path": "yolov8n-pose.pt",
        "confidence_threshold": 0.30,
    },
    # ---- Scoring weights (must sum to 1.0) ---------------------------------
    # Wrist proximity is the primary signal for ball possession.
    "w_wrist":       0.40,   # distance from ball to nearest wrist
    "w_hand_proxy":  0.15,   # distance from ball to hand proxy (beyond wrist)
    "w_inside":      0.20,   # ball centre lies inside player bbox (binary)
    "w_edge":        0.10,   # normalised distance from ball to nearest bbox edge
    "w_temporal":    0.15,   # IoU with previous frame's target bbox
    # Extend factor: hand_proxy = wrist + (wrist - elbow) * extend
    "hand_proxy_extend": 0.5,
    # Search radius for wrist/hand proximity (pixels, or multiple of ball radius)
    "wrist_search_radius_px":   120.0,
    "wrist_search_radius_ball": 7.0,   # radius = max(ball.radius * factor, _px)
    # ---- Temporal smoothing ------------------------------------------------
    "smoothing_window":     7,    # sliding-window half-width in frames
    "smoothing_min_iou":    0.20, # min mean IoU for window consensus
    "smoothing_max_gap_fill": 30, # max frames to carry forward during gaps
    # ---- Ball-score override -----------------------------------------------
    # If a challenger's wrist+hand_proxy score exceeds the current target's
    # by at least this margin, always switch regardless of temporal bonus.
    # Prevents temporal inertia from overriding clear ball-proximity evidence.
    "ball_score_switch_margin": 0.30,
    # Min IoU to consider a detected person as the "same" previous target
    # when no ball is present (track-by-inertia).
    "no_ball_tracking_iou": 0.20,
    # ---- Fallback ----------------------------------------------------------
    "fallback_to_largest": True,  # use highest-confidence plausible player when no ball
    # Standing players are taller than wide; filter detections whose width-to-
    # height ratio exceeds this threshold (e.g. 1.8 = floor/crowd blobs).
    "max_bbox_aspect_ratio": 1.8,
    # ---- Landmark gating (legacy — superseded by extract_selected_pose) ----
    "gate_landmarks": False,
    "gate_min_iou":   0.25,
    "landmark_bbox_vis_thresh": 0.10,
    # ---- Selected-player pose extraction -----------------------------------
    # Convert target player COCO-17 keypoints → MediaPipe landmarks and write
    # them into data["frames"][i]["landmarks"] for downstream analysis.
    "extract_selected_pose": True,
    # For gap-filled frames carry the last known keypoints forward.
    "pose_gap_fill": True,
    # ---- Lower-body event detection ----------------------------------------
    "run_events": True,
    "events_method": "zeni",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_iou(a: Tuple, b: Tuple) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-6)


def _point_in_bbox(px: float, py: float, bbox: Tuple) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def _edge_distance(px: float, py: float, bbox: Tuple) -> float:
    """Pixel distance from point to nearest bbox edge (0 if inside)."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _hand_proxy_norm(
    kp_norm: np.ndarray,
    wrist_idx: int,
    elbow_idx: int,
    extend: float = 0.5,
    conf_thresh: float = _MIN_KP_CONF,
) -> Optional[Tuple[float, float]]:
    """Return normalised (x, y) of the hand proxy beyond the wrist.

    The proxy sits at ``wrist + (wrist - elbow) * extend``, which models the
    natural grip extension past the wrist joint.  Returns ``None`` if either
    keypoint has insufficient confidence.
    """
    wrist = kp_norm[wrist_idx]
    elbow = kp_norm[elbow_idx]
    if wrist[2] < conf_thresh or elbow[2] < conf_thresh:
        return None
    dx = (wrist[0] - elbow[0]) * extend
    dy = (wrist[1] - elbow[1]) * extend
    return (float(wrist[0] + dx), float(wrist[1] + dy))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_player_components(
    person: PersonDetection,
    ball: BallDetection,
    frame_w: int,
    frame_h: int,
    prev_target_bbox: Optional[Tuple] = None,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute full per-component score breakdown for one player vs ball.

    Returns
    -------
    dict with keys:
        score, wrist_left, wrist_right, hand_proxy_left, hand_proxy_right,
        wrist, hand_proxy, inside, edge, temporal
    """
    cfg = config or _DEFAULT_CONFIG

    w_wrist       = cfg.get("w_wrist",       _DEFAULT_CONFIG["w_wrist"])
    w_hand_proxy  = cfg.get("w_hand_proxy",  _DEFAULT_CONFIG["w_hand_proxy"])
    w_inside      = cfg.get("w_inside",      _DEFAULT_CONFIG["w_inside"])
    w_edge        = cfg.get("w_edge",        _DEFAULT_CONFIG["w_edge"])
    w_temporal    = cfg.get("w_temporal",    _DEFAULT_CONFIG["w_temporal"])
    extend        = cfg.get("hand_proxy_extend", _DEFAULT_CONFIG["hand_proxy_extend"])
    sr_px         = cfg.get("wrist_search_radius_px",   _DEFAULT_CONFIG["wrist_search_radius_px"])
    sr_ball       = cfg.get("wrist_search_radius_ball", _DEFAULT_CONFIG["wrist_search_radius_ball"])

    bcx, bcy = ball.center  # type: ignore[misc]
    x1, y1, x2, y2 = person.bbox
    bw   = max(x2 - x1, 1)
    bh   = max(y2 - y1, 1)
    diag = math.sqrt(bw * bw + bh * bh)

    search_radius = max(float(ball.radius) * sr_ball, sr_px) if ball.radius else sr_px

    # ---- 1 & 2. Wrist + hand-proxy proximity ----------------------------
    wrist_left = wrist_right = hand_proxy_left = hand_proxy_right = 0.0

    kp = person.keypoints_norm
    if kp is not None and kp.shape == (17, 3):
        def _prox(px_norm: Optional[Tuple[float, float]]) -> float:
            if px_norm is None:
                return 0.0
            px, py = px_norm[0] * frame_w, px_norm[1] * frame_h
            d = _dist2(bcx, bcy, px, py)
            return max(0.0, 1.0 - d / search_radius)

        # Raw wrist proximity
        lw = kp[_COCO_LEFT_WRIST]
        rw = kp[_COCO_RIGHT_WRIST]
        if lw[2] >= _MIN_KP_CONF:
            wrist_left  = _prox((lw[0], lw[1]))
        if rw[2] >= _MIN_KP_CONF:
            wrist_right = _prox((rw[0], rw[1]))

        # Hand proxy (beyond wrist in elbow→wrist direction)
        lp = _hand_proxy_norm(kp, _COCO_LEFT_WRIST,  _COCO_LEFT_ELBOW,  extend)
        rp = _hand_proxy_norm(kp, _COCO_RIGHT_WRIST, _COCO_RIGHT_ELBOW, extend)
        if lp is not None:
            hand_proxy_left  = _prox(lp)
        if rp is not None:
            hand_proxy_right = _prox(rp)
    else:
        # Fall back to wrists_norm property (no keypoints_norm array)
        wrists = person.wrists_norm
        if wrists is not None:
            for i, wn in enumerate(wrists):
                if wn is not None:
                    px, py = wn[0] * frame_w, wn[1] * frame_h
                    val = max(0.0, 1.0 - _dist2(bcx, bcy, px, py) / search_radius)
                    if i == 0:
                        wrist_left = val
                    else:
                        wrist_right = val

    wrist_score      = max(wrist_left, wrist_right)
    hand_proxy_score = max(hand_proxy_left, hand_proxy_right)

    # ---- 3. Ball inside bbox (binary) ------------------------------------
    inside_score = 1.0 if _point_in_bbox(bcx, bcy, person.bbox) else 0.0

    # ---- 4. Edge proximity -----------------------------------------------
    if inside_score:
        edge_score = 1.0
    else:
        d = _edge_distance(bcx, bcy, person.bbox)
        edge_score = max(0.0, 1.0 - d / diag)

    # ---- 5. Temporal continuity ------------------------------------------
    temporal_score = 0.0
    if prev_target_bbox is not None:
        temporal_score = _bbox_iou(person.bbox, prev_target_bbox)

    score = min(1.0, max(0.0,
        w_wrist      * wrist_score      +
        w_hand_proxy * hand_proxy_score +
        w_inside     * inside_score     +
        w_edge       * edge_score       +
        w_temporal   * temporal_score
    ))

    return {
        "score":            score,
        "wrist_left":       wrist_left,
        "wrist_right":      wrist_right,
        "hand_proxy_left":  hand_proxy_left,
        "hand_proxy_right": hand_proxy_right,
        "wrist":            wrist_score,
        "hand_proxy":       hand_proxy_score,
        "inside":           inside_score,
        "edge":             edge_score,
        "temporal":         temporal_score,
    }


def score_player_for_ball(
    person: PersonDetection,
    ball: BallDetection,
    frame_w: int,
    frame_h: int,
    prev_target_bbox: Optional[Tuple] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score one player as the likely ball carrier.

    The dominant signal is proximity between the ball center and the player's
    wrist keypoints / hand-proxy points (computed slightly beyond the wrist in
    the elbow→wrist direction).

    Parameters
    ----------
    person : PersonDetection
        Candidate player (bbox + optional COCO-17 keypoints).
    ball : BallDetection
        Current-frame ball detection.  Must have ``detected=True``.
    frame_w, frame_h : int
        Frame pixel dimensions (used to convert normalised keypoints).
    prev_target_bbox : tuple, optional
        Previous-frame target bbox for temporal-continuity bonus.
    weights : dict, optional
        Override scoring weights (``w_wrist``, ``w_hand_proxy``, ``w_inside``,
        ``w_edge``, ``w_temporal``).

    Returns
    -------
    float
        Score in [0, 1].  Higher → more likely ball carrier.
    """
    if not ball.detected or ball.center is None:
        return 0.0
    cfg = {**_DEFAULT_CONFIG, **(weights or {})}
    return _score_player_components(person, ball, frame_w, frame_h, prev_target_bbox, cfg)["score"]


# ---------------------------------------------------------------------------
# Per-frame assignment
# ---------------------------------------------------------------------------

@dataclass
class FrameAssignment:
    """Target-player assignment result for one frame."""
    target_bbox:           Optional[Tuple[int, int, int, int]] = None
    target_score:          float = 0.0
    target_source:         str   = "none"    # "ball_proximity" | "largest" | "none"
    n_players:             int   = 0
    players:               List[dict] = field(default_factory=list)
    # COCO-17 keypoints for the selected player (17, 3) or None
    target_keypoints_norm: Optional[np.ndarray] = field(default=None, repr=False)
    # Per-component score breakdown
    score_breakdown:       Optional[Dict[str, float]] = None
    # Filled by smooth_assignments():
    smoothed_bbox:         Optional[Tuple[int, int, int, int]] = None
    smoothed_source:       str = "none"   # + "temporal"
    landmarks_gated:       bool = False


def assign_target_player(
    persons: List[PersonDetection],
    ball: BallDetection,
    frame_w: int,
    frame_h: int,
    prev_assignment: Optional[FrameAssignment] = None,
    config: Optional[Dict] = None,
) -> FrameAssignment:
    """Select the most likely ball carrier from a list of detected persons.

    When a ball position is available every person is scored; the highest-
    scorer is chosen.  When no ball is available and
    ``config["fallback_to_largest"]`` is True, the highest-confidence
    plausible-aspect-ratio player is returned as a best-guess subject.
    """
    cfg      = {**_DEFAULT_CONFIG, **(config or {})}
    prev_bbox = prev_assignment.target_bbox if prev_assignment else None
    max_ar    = cfg.get("max_bbox_aspect_ratio", 1.8)

    # Filter out implausible detections (floor/crowd blobs wider than tall)
    def _plausible(p: PersonDetection) -> bool:
        x1, y1, x2, y2 = p.bbox
        h = max(y2 - y1, 1)
        w = max(x2 - x1, 1)
        return (w / h) <= max_ar

    plausible = [p for p in persons if _plausible(p)] or persons  # fallback: keep all

    result = FrameAssignment(n_players=len(persons))

    if not persons:
        return result

    if ball.detected and ball.center is not None:
        # Score every plausible player and record their breakdown
        all_components = [
            (_score_player_components(p, ball, frame_w, frame_h, prev_bbox, cfg), p)
            for p in plausible
        ]

        # Build players list (all persons, plausible ones get a real score)
        scored_set = {id(p) for _, p in all_components}
        comp_map   = {id(p): comp for comp, p in all_components}
        for p in persons:
            pd = p.to_dict()
            if id(p) in scored_set:
                comp = comp_map[id(p)]
                pd["score"] = round(comp["score"], 4)
                pd["score_breakdown"] = {
                    k: round(v, 4) for k, v in comp.items() if k != "score"
                }
            else:
                pd["score"] = 0.0
            result.players.append(pd)

        best_comp, best_person = max(all_components, key=lambda x: x[0]["score"])

        # Ball-score override: if the top scorer wins purely because of temporal
        # inertia but a challenger has clearly better wrist proximity, switch.
        # This prevents temporal bonus from keeping the wrong player when ball
        # evidence strongly favours someone else.
        switch_margin = cfg.get("ball_score_switch_margin",
                                _DEFAULT_CONFIG["ball_score_switch_margin"])
        if prev_bbox is not None and switch_margin > 0 and len(all_components) > 1:
            def _ball_prox(comp):
                return comp.get("wrist", 0.0) + comp.get("hand_proxy", 0.0)

            # Find the current target (highest IoU with prev_bbox)
            curr_entry = max(
                all_components,
                key=lambda x: _bbox_iou(x[1].bbox, prev_bbox),
            )
            curr_iou = _bbox_iou(curr_entry[1].bbox, prev_bbox)
            if curr_iou >= cfg.get("no_ball_tracking_iou",
                                   _DEFAULT_CONFIG["no_ball_tracking_iou"]):
                curr_prox = _ball_prox(curr_entry[0])
                best_prox = _ball_prox(best_comp)
                # If the winner of the score race IS the current target, no override
                # needed. Only override if a different player is the overall winner
                # AND if the current target's ball proximity is well behind.
                if (id(best_person) != id(curr_entry[1]) and
                        curr_prox > best_prox - switch_margin):
                    # Not enough margin to justify switching away from current target
                    best_comp, best_person = curr_entry

        result.target_bbox           = best_person.bbox
        result.target_score          = best_comp["score"]
        result.target_source         = "ball_proximity"
        result.target_keypoints_norm = best_person.keypoints_norm
        result.score_breakdown       = best_comp

    else:
        # No ball detected this frame — track the previous target by IoU
        # (track-by-inertia) rather than jumping to highest-confidence player.
        # This prevents one undetected ball frame from resetting the tracker.
        for p in persons:
            pd = p.to_dict()
            pd["score"] = 0.0
            result.players.append(pd)

        no_ball_iou_thresh = cfg.get("no_ball_tracking_iou",
                                     _DEFAULT_CONFIG["no_ball_tracking_iou"])
        if prev_bbox is not None:
            # Find the person that best matches the previous target
            iou_scored = [(p, _bbox_iou(p.bbox, prev_bbox)) for p in plausible]
            best_match, best_iou = max(iou_scored, key=lambda x: x[1])
            if best_iou >= no_ball_iou_thresh:
                result.target_bbox           = best_match.bbox
                result.target_score          = best_iou
                result.target_source         = "tracked_no_ball"
                result.target_keypoints_norm = best_match.keypoints_norm
                return result

        # No prev bbox or no IoU match — fall back to highest-confidence player
        if cfg.get("fallback_to_largest"):
            best = max(plausible, key=lambda p: p.confidence)
            result.target_bbox           = best.bbox
            result.target_score          = best.confidence
            result.target_source         = "highest_confidence"
            result.target_keypoints_norm = best.keypoints_norm

    return result


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_assignments(
    raw: List[FrameAssignment],
    window: int = 7,
    min_iou: float = 0.20,
    max_gap_fill: int = 30,
) -> List[FrameAssignment]:
    """Smooth per-frame target-player assignments.

    Two-pass algorithm:

    1. **Consensus pass** — for each frame look at ±``window//2`` neighbours
       and choose the bbox with the highest mean IoU to all others in the
       window.  Suppresses 1-3 frame flickers caused by scoring noise.

    2. **Gap-fill pass** — carry the last valid assignment forward (and
       backward at clip start) across gaps where no player was detected, up
       to ``max_gap_fill`` frames.

    Parameters
    ----------
    raw : list of FrameAssignment
        Unsmoothed per-frame assignments (mutated in-place).
    window : int
        Sliding-window size (frames). Default: 7.
    min_iou : float
        Minimum mean-IoU for the consensus bbox to be accepted. Default: 0.20.
    max_gap_fill : int
        Maximum consecutive None frames to fill by carry-forward. Default: 30.

    Returns
    -------
    list of FrameAssignment
        Same list with ``smoothed_bbox`` and ``smoothed_source`` set.
    """
    n    = len(raw)
    half = window // 2

    # Pass 1: sliding-window IoU consensus
    smoothed: List[Optional[Tuple]] = [None] * n
    sources:  List[str]             = ["none"] * n

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        window_items = [
            (j, raw[j].target_bbox)
            for j in range(lo, hi)
            if raw[j].target_bbox is not None
        ]

        if not window_items:
            smoothed[i] = raw[i].target_bbox
            sources[i]  = raw[i].target_source
            continue

        bboxes     = [b for _, b in window_items]
        frame_idxs = [j for j, _ in window_items]

        if len(bboxes) == 1:
            smoothed[i] = bboxes[0]
            sources[i]  = raw[frame_idxs[0]].target_source
            continue

        mean_ious = np.array([
            float(np.mean([_bbox_iou(b, other) for other in bboxes]))
            for b in bboxes
        ])
        best = int(np.argmax(mean_ious))

        if mean_ious[best] >= min_iou:
            smoothed[i] = bboxes[best]
            sources[i]  = (
                raw[i].target_source or "ball_proximity"
                if frame_idxs[best] == i
                else "temporal"
            )
        else:
            smoothed[i] = raw[i].target_bbox
            sources[i]  = raw[i].target_source

    # Pass 2a: gap-fill forward
    last_bbox: Optional[Tuple] = None
    gap = 0
    for i in range(n):
        if smoothed[i] is not None:
            last_bbox, gap = smoothed[i], 0
        elif last_bbox is not None and gap < max_gap_fill:
            smoothed[i] = last_bbox
            sources[i]  = "temporal"
            gap += 1
        else:
            gap += 1

    # Pass 2b: gap-fill backward (clip-start gaps)
    last_bbox = None
    for i in range(n - 1, -1, -1):
        if smoothed[i] is not None:
            last_bbox = smoothed[i]
        elif last_bbox is not None:
            smoothed[i] = last_bbox
            sources[i]  = "temporal"

    # Write back
    for i, asgn in enumerate(raw):
        asgn.smoothed_bbox   = smoothed[i]
        asgn.smoothed_source = sources[i]

    return raw


# ---------------------------------------------------------------------------
# Landmark gating (legacy helper, kept for backward compatibility)
# ---------------------------------------------------------------------------

def _estimate_extracted_person_bbox(
    frame: dict,
    frame_w: int,
    frame_h: int,
    vis_thresh: float = 0.10,
) -> Optional[Tuple[int, int, int, int]]:
    """Estimate the extracted person's pixel bbox from MediaPipe landmarks."""
    lm = frame.get("landmarks") or {}
    if not lm:
        return None

    xs, ys = [], []
    for kp in lm.values():
        if kp.get("visibility", 0.0) >= vis_thresh:
            x = kp.get("x", float("nan"))
            y = kp.get("y", float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x * frame_w)
                ys.append(y * frame_h)

    if not xs:
        return None

    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def gate_landmarks_for_player_selection(
    data: dict,
    assignments: List[FrameAssignment],
    min_iou: float = 0.25,
    vis_thresh: float = 0.10,
) -> int:
    """Clear data["frames"][i]["landmarks"] for frames where the already-
    extracted single-person landmarks belong to a different player than the
    identified target.

    Deprecated in favour of :func:`extract_selected_player_pose`, which
    replaces landmarks with the correct player's keypoints rather than just
    clearing them.  Kept for backward compatibility.

    Returns
    -------
    int  — number of frames gated.
    """
    frames  = data.get("frames", [])
    meta    = data.get("meta", {})
    frame_w = meta.get("width",  1920)
    frame_h = meta.get("height", 1080)
    n_gated = 0

    for frame, asgn in zip(frames, assignments):
        target = asgn.smoothed_bbox
        if target is None:
            continue
        extracted = _estimate_extracted_person_bbox(frame, frame_w, frame_h, vis_thresh)
        if extracted is None:
            continue
        if _bbox_iou(extracted, target) < min_iou:
            if "landmarks_unfiltered" not in frame:
                frame["landmarks_unfiltered"] = frame.get("landmarks")
            frame["landmarks"] = {}
            asgn.landmarks_gated = True
            n_gated += 1

    return n_gated


# ---------------------------------------------------------------------------
# COCO-17 → MediaPipe landmark conversion
# ---------------------------------------------------------------------------

def coco17_to_landmarks(kp_norm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Convert a COCO-17 keypoint array to a MediaPipe-format landmark dict.

    Parameters
    ----------
    kp_norm : np.ndarray, shape (17, 3)
        Columns are (x_norm, y_norm, confidence) all in [0, 1].

    Returns
    -------
    dict
        ``{landmark_name: {"x": float, "y": float, "visibility": float}}``
        in normalised [0, 1] coordinates.  Only joints with confidence ≥ 0.01
        are included (very permissive — let downstream code apply its own
        threshold).
    """
    result: Dict[str, Dict[str, float]] = {}
    for coco_idx, mp_name in _COCO_TO_MP_NAME.items():
        if coco_idx >= len(kp_norm):
            continue
        x, y, conf = float(kp_norm[coco_idx, 0]), float(kp_norm[coco_idx, 1]), float(kp_norm[coco_idx, 2])
        if conf >= 0.01:
            result[mp_name] = {"x": x, "y": y, "visibility": conf}
    return result


# ---------------------------------------------------------------------------
# Selected-player pose extraction
# ---------------------------------------------------------------------------

def extract_selected_player_pose(
    data: dict,
    assignments: List[FrameAssignment],
    pose_gap_fill: bool = True,
) -> int:
    """Write selected player's COCO-17 keypoints into data["frames"] as
    MediaPipe-format landmarks, replacing whatever single-person landmarks
    were set by extract().

    For frames where the assignment has ``target_keypoints_norm`` set (i.e.
    a real YOLO pose detection was made), the COCO-17 keypoints are converted
    and stored.  For gap-filled frames (source="temporal" with no keypoints),
    the last known keypoints are carried forward when ``pose_gap_fill=True``.

    The original landmarks from extract() are preserved under the key
    ``"landmarks_original_extract"`` (set only the first time this function
    runs on a given frame).

    Parameters
    ----------
    data : dict
        Pivot data dict.  ``data["frames"]`` is mutated in-place.
    assignments : list of FrameAssignment
        Per-frame smoothed assignments (from smooth_assignments()).
    pose_gap_fill : bool
        Carry the last known keypoints forward for gap-filled frames.

    Returns
    -------
    int  — number of frames whose landmarks were updated.
    """
    frames  = data.get("frames", [])
    updated = 0
    last_kp: Optional[np.ndarray] = None

    for frame, asgn in zip(frames, assignments):
        kp = asgn.target_keypoints_norm

        # Propagate last known keypoints into gap frames
        if kp is None and pose_gap_fill and last_kp is not None:
            if asgn.smoothed_bbox is not None:   # only if a player is assigned
                kp = last_kp

        if kp is not None and kp.shape == (17, 3):
            # Preserve original extract() landmarks once
            if "landmarks_original_extract" not in frame:
                frame["landmarks_original_extract"] = frame.get("landmarks")
            frame["landmarks"] = coco17_to_landmarks(kp)
            last_kp = kp
            updated += 1
        elif asgn.smoothed_bbox is None:
            # No target player at all — clear landmarks so events aren't misled
            if frame.get("landmarks"):
                if "landmarks_original_extract" not in frame:
                    frame["landmarks_original_extract"] = frame.get("landmarks")
                frame["landmarks"] = {}

    return updated


# ---------------------------------------------------------------------------
# Selected-player lower-body event analysis
# ---------------------------------------------------------------------------

def run_selected_player_analysis(
    data: dict,
    config: Optional[Dict] = None,
) -> dict:
    """Run lower-body event detection on the selected player's landmarks.

    Should be called **after** :func:`extract_selected_player_pose` has
    written the target player's COCO-derived landmarks into
    ``data["frames"]``.

    Parameters
    ----------
    data : dict
        Pivot data dict with updated landmarks.
    config : dict, optional
        Supports key ``"events_method"`` (default: ``"zeni"``).

    Returns
    -------
    dict  — same *data* with ``data["events"]`` populated.
    """
    from .events import detect_events  # local import to avoid circular deps

    cfg    = {**_DEFAULT_CONFIG, **(config or {})}
    method = cfg.get("events_method", "zeni")

    n_frames_with_lm = sum(
        1 for f in data.get("frames", []) if f.get("landmarks")
    )
    logger.info(
        "Running event detection (method=%s) on %d/%d frames with landmarks",
        method, n_frames_with_lm, len(data.get("frames", [])),
    )

    try:
        data = detect_events(data, method=method)
        ev = data.get("events", {})
        logger.info(
            "Events detected — left_hs=%d right_hs=%d left_to=%d right_to=%d",
            len(ev.get("left_hs", [])),
            len(ev.get("right_hs", [])),
            len(ev.get("left_to", [])),
            len(ev.get("right_to", [])),
        )
    except Exception as exc:
        logger.warning("Event detection failed: %s", exc)

    return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_target_player(
    video_path: str,
    data: dict,
    config: Optional[Dict] = None,
    person_detector: Optional[PersonDetector] = None,
) -> dict:
    """Multi-person target-player selection pipeline.

    Steps
    -----
    1. Detect all persons per frame (YOLO pose).
    2. Score each against the tracked ball (wrist proximity dominant).
    3. Temporal smoothing (IoU consensus + gap fill).
    4. Extract selected-player pose: write COCO-17 → MediaPipe landmarks into
       ``data["frames"]`` so downstream analysis sees the right player.
    5. Optionally run lower-body event detection.

    Writes ``data["player_selection"]`` and returns *data*.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    data : dict
        Pivot data dict (output of ``extract()``).
    config : dict, optional
        Override default config keys.
    person_detector : PersonDetector, optional
        Pre-built detector.  Constructed from config if ``None``.

    Returns
    -------
    dict
        Same ``data`` with ``data["player_selection"]`` (and optionally
        ``data["events"]``) added.
    """
    cfg     = {**_DEFAULT_CONFIG, **(config or {})}
    frames  = data.get("frames", [])
    meta    = data.get("meta", {})
    frame_w = meta.get("width",  1920)
    frame_h = meta.get("height", 1080)
    fps     = meta.get("fps", 30.0)
    n       = len(frames)

    logger.info(
        "Player selection: %d frames, %dx%d @ %.1f fps", n, frame_w, frame_h, fps
    )

    # Build person detector
    if person_detector is None:
        person_detector = create_person_detector(
            cfg.get("person_detector", "yolo_pose"),
            **cfg.get("person_detector_kwargs", {}),
        )

    # Collect tracked ball positions
    ball_per_frame: List[BallDetection] = []
    if "ball" in data and "per_frame" in data["ball"]:
        for pf in data["ball"]["per_frame"]:
            bd = pf.get("ball", {})
            if bd.get("tracked") and bd.get("center"):
                cx, cy  = float(bd["center"][0]), float(bd["center"][1])
                raw_box = bd.get("bbox")
                bbox    = tuple(int(v) for v in raw_box) if raw_box else None
                ball_per_frame.append(BallDetection(
                    detected   = True,
                    bbox       = bbox,
                    center     = (cx, cy),
                    radius     = bd.get("radius"),
                    confidence = bd.get("confidence", 0.0),
                ))
            else:
                ball_per_frame.append(BallDetection())
    else:
        ball_per_frame = [BallDetection()] * n
        logger.warning(
            "No data['ball'] found. Run analyze_ball() first for best results. "
            "Falling back to 'largest player' heuristic."
        )

    # Step 1: Per-frame person detection + scoring
    logger.info("Step 1/3: Detecting persons and scoring against ball …")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    raw_assignments: List[FrameAssignment] = []
    prev_asgn: Optional[FrameAssignment]   = None

    try:
        for idx in range(n):
            ret, frame_bgr = cap.read()
            if not ret:
                raw_assignments.append(FrameAssignment())
                continue

            persons = person_detector.detect(frame_bgr)
            ball    = ball_per_frame[idx] if idx < len(ball_per_frame) else BallDetection()
            asgn    = assign_target_player(
                persons, ball, frame_w, frame_h,
                prev_assignment=prev_asgn,
                config=cfg,
            )
            raw_assignments.append(asgn)
            prev_asgn = asgn
    finally:
        cap.release()

    # Step 2: Temporal smoothing
    logger.info(
        "Step 2/3: Smoothing assignments (window=%d, max_gap_fill=%d) …",
        cfg["smoothing_window"], cfg["smoothing_max_gap_fill"],
    )
    assignments = smooth_assignments(
        raw_assignments,
        window       = cfg["smoothing_window"],
        min_iou      = cfg["smoothing_min_iou"],
        max_gap_fill = cfg["smoothing_max_gap_fill"],
    )

    # Step 3: Extract selected-player pose
    n_pose_updated = 0
    if cfg.get("extract_selected_pose", True):
        logger.info("Step 3/3: Extracting selected-player pose into data['frames'] …")
        n_pose_updated = extract_selected_player_pose(
            data, assignments,
            pose_gap_fill=cfg.get("pose_gap_fill", True),
        )
        logger.info("  %d/%d frames updated with selected-player landmarks", n_pose_updated, n)

    # Legacy: gate_landmarks (off by default — extract_selected_pose supersedes it)
    n_gated = 0
    if cfg.get("gate_landmarks"):
        n_gated = gate_landmarks_for_player_selection(
            data, assignments,
            min_iou    = cfg["gate_min_iou"],
            vis_thresh = cfg["landmark_bbox_vis_thresh"],
        )

    # Step 4 (optional): lower-body event detection
    if cfg.get("run_events", True):
        data = run_selected_player_analysis(data, cfg)

    # Build output
    n_found      = sum(1 for a in assignments if a.smoothed_bbox is not None)
    n_no_players = sum(1 for a in assignments if a.n_players == 0)
    coverage     = n_found / max(n, 1)

    per_frame_out = []
    for i, (frame, asgn) in enumerate(zip(frames, assignments)):
        entry = {
            "frame_idx":       i,
            "time_s":          round(frame.get("time_s", i / fps), 4),
            "n_players":       asgn.n_players,
            "players":         asgn.players,
            "target_bbox":     list(asgn.target_bbox)    if asgn.target_bbox    else None,
            "target_score":    round(asgn.target_score, 4),
            "target_source":   asgn.target_source,
            "score_breakdown": {k: round(v, 4) for k, v in asgn.score_breakdown.items()}
                               if asgn.score_breakdown else None,
            "smoothed_bbox":   list(asgn.smoothed_bbox)  if asgn.smoothed_bbox  else None,
            "smoothed_source": asgn.smoothed_source,
            "landmarks_gated": asgn.landmarks_gated,
        }
        per_frame_out.append(entry)

    data["player_selection"] = {
        "config":  {k: v for k, v in cfg.items() if not callable(v)},
        "summary": {
            "n_frames":                  n,
            "n_frames_player_found":     n_found,
            "n_frames_no_players":       n_no_players,
            "n_frames_landmarks_gated":  n_gated,
            "n_frames_pose_updated":     n_pose_updated,
            "player_coverage_rate":      round(coverage, 4),
            "detector":                  cfg.get("person_detector", "unknown"),
        },
        "per_frame": per_frame_out,
    }

    logger.info(
        "Player selection done. Coverage=%.1f%% | No-player=%d | Pose-updated=%d",
        coverage * 100, n_no_players, n_pose_updated,
    )
    return data


# ---------------------------------------------------------------------------
# Debug video renderer
# ---------------------------------------------------------------------------

# BGR colour palette
_CLR_ALL_PERSON   = (90,  90,  90)   # dim grey  — non-target players
_CLR_TARGET       = (0,  200,   0)   # green     — target player box
_CLR_BALL_DET     = (0,  140, 255)   # orange    — direct ball detection
_CLR_BALL_INT     = (0,  255, 255)   # yellow    — interpolated
_CLR_BALL_PRED    = (255, 200,  0)   # light-blue — predicted
_CLR_WRIST_L      = (255, 128,  0)   # blue      — left wrist
_CLR_WRIST_R      = (0,    0, 220)   # red       — right wrist
_CLR_HAND_PROXY   = (255,   0, 200)  # magenta   — hand proxy
_CLR_SKEL_UPPER   = (200, 200,   0)  # cyan-ish  — upper-body skeleton
_CLR_SKEL_TORSO   = (200, 200, 200)  # white     — torso
_CLR_SKEL_LOWER   = (0,   200, 200)  # yellow    — lower-body skeleton
_CLR_EVENT_HS     = (0,   255,   0)  # green     — heel strike
_CLR_EVENT_TO     = (0,   100, 255)  # orange    — toe off
_CLR_HUD          = (220, 220, 220)
_CLR_LABEL_BG     = (30,   30,  30)


def _ball_color(source: str) -> tuple:
    if source == "interpolated":
        return _CLR_BALL_INT
    if source == "predicted":
        return _CLR_BALL_PRED
    return _CLR_BALL_DET


def _draw_skeleton(
    frame: np.ndarray,
    kp_norm: np.ndarray,
    frame_w: int,
    frame_h: int,
    conf_thresh: float = _MIN_KP_CONF,
) -> None:
    """Draw COCO-17 skeleton on *frame* for the target player."""
    def pt(idx: int) -> Optional[Tuple[int, int]]:
        if idx >= len(kp_norm) or kp_norm[idx, 2] < conf_thresh:
            return None
        return (int(kp_norm[idx, 0] * frame_w), int(kp_norm[idx, 1] * frame_h))

    for a, b in _SKELETON_UPPER:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, _CLR_SKEL_UPPER, 2, cv2.LINE_AA)
    for a, b in _SKELETON_TORSO:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, _CLR_SKEL_TORSO, 2, cv2.LINE_AA)
    for a, b in _SKELETON_LOWER:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, _CLR_SKEL_LOWER, 2, cv2.LINE_AA)


def render_player_selection_video(
    video_path: str,
    data: dict,
    output_path: str,
    fps: Optional[float] = None,
    show_ball: bool = True,
    show_all_players: bool = True,
    show_wrists: bool = True,
    show_skeleton: bool = True,
    show_events: bool = True,
    show_score: bool = True,
    codec: str = "mp4v",
) -> str:
    """Render a debug video with full player-selection overlay.

    Overlay key
    -----------
    - Thin grey box       — every detected player (non-target)
    - Thick green box     — target player (after smoothing)
    - Score label         — assignment score + component breakdown on target
    - Skeleton overlay    — COCO-17 skeleton for target player (when pose extracted)
    - Blue / red dot      — left / right wrist of target player
    - Magenta dot         — hand proxy points
    - Orange circle       — directly detected ball
    - Yellow circle       — interpolated ball
    - Green ankle marker  — heel strike event frame
    - Orange ankle marker — toe off event frame
    - HUD                 — frame index, player count, event label

    Parameters
    ----------
    video_path : str
        Source video path.
    data : dict
        Pivot data dict with ``player_selection`` (and optionally ``ball``,
        ``events``).
    output_path : str
        Destination MP4 path.
    fps, show_ball, show_all_players, show_wrists, show_skeleton,
    show_events, show_score, codec : see defaults above.

    Returns
    -------
    str — path to the written video file.
    """
    ps       = data.get("player_selection", {})
    pf_map   = {pf["frame_idx"]: pf for pf in ps.get("per_frame", [])}
    ball_map = {
        entry["frame_idx"]: entry
        for entry in data.get("ball", {}).get("per_frame", [])
    }
    # Build per-frame event lookup
    events = data.get("events", {})
    left_hs_frames  = {e["frame"] for e in events.get("left_hs",  [])}
    right_hs_frames = {e["frame"] for e in events.get("right_hs", [])}
    left_to_frames  = {e["frame"] for e in events.get("left_to",  [])}
    right_to_frames = {e["frame"] for e in events.get("right_to", [])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    frames_data = data.get("frames", [])

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pf   = pf_map.get(idx, {})
        ball = ball_map.get(idx, {})

        target_bbox = pf.get("smoothed_bbox")
        tb_tuple    = tuple(target_bbox) if target_bbox else None

        # ---- All non-target players (dim) --------------------------------
        if show_all_players:
            for p in pf.get("players", []):
                pb = p.get("bbox")
                if pb is None:
                    continue
                is_tgt = tb_tuple is not None and _bbox_iou(tuple(pb), tb_tuple) > 0.5
                if not is_tgt:
                    cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), _CLR_ALL_PERSON, 1)

        # ---- Skeleton overlay for target (from extracted COCO-17 pose) ---
        if show_skeleton and idx < len(frames_data):
            lm = frames_data[idx].get("landmarks") or {}
            # Reconstruct a COCO-17-like array from MediaPipe-format landmarks
            kp_arr = np.zeros((17, 3), dtype=np.float32)
            for coco_idx, mp_name in _COCO_TO_MP_NAME.items():
                if mp_name in lm:
                    kp_arr[coco_idx, 0] = lm[mp_name]["x"]
                    kp_arr[coco_idx, 1] = lm[mp_name]["y"]
                    kp_arr[coco_idx, 2] = lm[mp_name].get("visibility", 0.0)
            if kp_arr[:, 2].max() > _MIN_KP_CONF:
                _draw_skeleton(frame, kp_arr, W, H)

        # ---- Target player (highlighted box + wrists + hand proxies) ----
        if tb_tuple is not None:
            tx1, ty1, tx2, ty2 = tb_tuple
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), _CLR_TARGET, 3)

            if show_score:
                bd = pf.get("score_breakdown") or {}
                score = pf.get("target_score", 0.0)
                if score > 0 or pf.get("smoothed_source") == "temporal":
                    label = (
                        f"score={score:.2f}"
                        f" w={bd.get('wrist', 0):.2f}"
                        f" h={bd.get('hand_proxy', 0):.2f}"
                        f" [{pf.get('smoothed_source', '')}]"
                    )
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    lx = tx1
                    ly = max(ty1 - 5, th + 4)
                    cv2.rectangle(frame, (lx, ly-th-4), (lx+tw+4, ly+bl), _CLR_LABEL_BG, -1)
                    cv2.putText(frame, label, (lx+2, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_TARGET, 1, cv2.LINE_AA)

            # Wrists + hand proxies for target player
            if show_wrists:
                for p in pf.get("players", []):
                    pb = p.get("bbox")
                    if pb and _bbox_iou(tuple(pb), tb_tuple) > 0.5:
                        # Wrist dots
                        for wkey, col in (
                            ("left_wrist_norm",  _CLR_WRIST_L),
                            ("right_wrist_norm", _CLR_WRIST_R),
                        ):
                            wn = p.get(wkey)
                            if wn:
                                wx, wy = int(wn[0] * W), int(wn[1] * H)
                                cv2.circle(frame, (wx, wy), 6, col, -1)
                                cv2.circle(frame, (wx, wy), 8, (255,255,255), 1)
                        # Elbow→wrist line
                        for ek, wk in (
                            ("left_elbow_norm",  "left_wrist_norm"),
                            ("right_elbow_norm", "right_wrist_norm"),
                        ):
                            en = p.get(ek)
                            wn = p.get(wk)
                            if en and wn:
                                ep = (int(en[0]*W), int(en[1]*H))
                                wp = (int(wn[0]*W), int(wn[1]*H))
                                cv2.line(frame, ep, wp, _CLR_SKEL_UPPER, 2, cv2.LINE_AA)
                                # Hand proxy dot (0.5× beyond wrist)
                                hpx = int(wp[0] + 0.5*(wp[0]-ep[0]))
                                hpy = int(wp[1] + 0.5*(wp[1]-ep[1]))
                                cv2.circle(frame, (hpx, hpy), 5, _CLR_HAND_PROXY, -1)
                        break

        # ---- Ball --------------------------------------------------------
        if show_ball:
            bd = ball.get("ball", {})
            if bd.get("tracked") and bd.get("center"):
                bcx, bcy = int(bd["center"][0]), int(bd["center"][1])
                br       = max(int(bd.get("radius", 15)), 8)
                col      = _ball_color(bd.get("source", "detected"))
                cv2.circle(frame, (bcx, bcy), br, col, 2)
                cv2.circle(frame, (bcx, bcy), 3,  col, -1)

        # ---- Foot event markers ------------------------------------------
        if show_events and idx < len(frames_data):
            lm = frames_data[idx].get("landmarks") or {}
            event_label = ""
            for side, ankle_key, hs_set, to_set in (
                ("L", "LEFT_ANKLE",  left_hs_frames,  left_to_frames),
                ("R", "RIGHT_ANKLE", right_hs_frames, right_to_frames),
            ):
                if ankle_key in lm:
                    ax = int(lm[ankle_key]["x"] * W)
                    ay = int(lm[ankle_key]["y"] * H)
                    if idx in hs_set:
                        cv2.circle(frame, (ax, ay), 10, _CLR_EVENT_HS, -1)
                        cv2.putText(frame, f"{side}HS", (ax+8, ay),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_EVENT_HS, 1, cv2.LINE_AA)
                        event_label += f"{side}HS "
                    elif idx in to_set:
                        cv2.circle(frame, (ax, ay), 10, _CLR_EVENT_TO, -1)
                        cv2.putText(frame, f"{side}TO", (ax+8, ay),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_EVENT_TO, 1, cv2.LINE_AA)
                        event_label += f"{side}TO "

        # ---- HUD ---------------------------------------------------------
        hud = f"f{idx:04d}  players={pf.get('n_players', 0)}"
        if pf.get("landmarks_gated"):
            hud += "  [gated]"
        cv2.putText(frame, hud, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, _CLR_HUD, 1, cv2.LINE_AA)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    logger.info(
        "Player selection debug video written → %s (%d frames)", output_path, idx
    )
    return output_path
