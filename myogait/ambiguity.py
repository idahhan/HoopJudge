"""Overlap and ambiguity analysis for the selected ball handler.

Purpose
-------
Between target-player selection and final pose/gait commitment there is a
structural gap: when two players are geometrically close, the raw landmarks
chosen for the handler may reflect mixed body evidence.  This module adds an
explicit decision stage that classifies every frame into one of four states
and recommends how downstream pose/gait logic should treat it.

States
------
clear       All signals below threshold.  Use normal pose flow.
overlap     Bbox overlap detected but handler identity is unambiguous
            (clear score-gap winner).  Prefer refined crop pose.
ambiguous   Multiple signals firing or handler identity uncertain.
            Prefer refined crop pose + stronger smoothing.
occluded    Handler bbox missing, track lost, or severe overlap.
            Do not emit hard pose/gait changes — use carry-forward.

Pipeline position
-----------------
select_target_player()      → data["player_selection"]
        ↓
compute_ambiguity()          ← THIS MODULE
        ↓                    → data["ambiguity"]
refine_handler_pose()        reads data["ambiguity"]["per_frame"] for routing
        ↓
track_possession() …

Signal set
----------
bbox_overlap            max IoU(handler, other_player) — spatial contamination risk
score_gap               handler_score − second_best_score (small → competitors)
handler_score           absolute possession score (low → uncertain handler)
pose_jump               mean normalised keypoint displacement vs EMA history
lower_body_instability  fraction of lower-body kps below confidence threshold
track_id_change         handler track ID changed from previous frame (swap risk)
bbox_area_change        fractional bbox area change vs previous frame (geometry jitter)

Output schema
-------------
data["ambiguity"]["per_frame"][i] = {
    "frame_idx":             int,
    "selected_track_id":     int | None,
    "state":                 "clear" | "overlap" | "ambiguous" | "occluded",
    "score":                 float,          # overall ambiguity score [0, 1]
    "signals": {
        "bbox_overlap":          float,      # raw max IoU with other player
        "score_gap":             float,      # handler - 2nd best score (0 if no ball)
        "handler_score":         float,      # raw handler possession score
        "pose_jump":             float,      # normalised L2 kp displacement
        "lower_body_instability":float,      # fraction of bad lower-body kps [0,1]
        "track_id_change":       float,      # 1.0 if track ID changed else 0.0
        "bbox_area_change":      float,      # |Δarea| / prev_area
    },
    "signal_scores": { ... },               # normalised [0,1] contribution per signal
    "nearby_tracks":         [int, ...],    # tracks overlapping handler this frame
    "recommended_pose_mode": "normal" | "refined" | "carry_forward",
}
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    # ── Signal weights (weighted sum → ambiguity score) ──────────────────────
    "w_bbox_overlap":         0.35,
    "w_score_gap":            0.25,
    "w_handler_score":        0.10,
    "w_pose_jump":            0.15,
    "w_lower_body":           0.08,
    "w_track_id_change":      0.05,
    "w_bbox_area_change":     0.02,

    # ── Raw signal normalisation thresholds ──────────────────────────────────
    # bbox_overlap: IoU above this → signal starts firing; above severe → occluded
    "bbox_overlap_threshold":  0.15,
    "bbox_overlap_severe":     0.45,

    # score_gap: gap below this means near-tie; above → handler is clear winner
    # signal = max(0, 1 − gap/full_margin)
    "score_gap_full_margin":   0.30,

    # handler_score: below this → handler identity itself is uncertain
    # signal = max(0, 1 − score/confident_threshold)
    "handler_score_confident": 0.35,

    # pose_jump: displacement above this → starts firing; above severe → strong
    "pose_jump_threshold":     0.06,
    "pose_jump_severe":        0.14,

    # lower_body: fraction of 6 lower-body kps below conf threshold counts as bad
    "lower_body_conf_min":     0.30,
    "lower_body_fraction_bad": 0.50,   # signal = fraction_bad / this

    # bbox_area_change: fractional area change (|ΔA|/prev_A) above this → signal
    "bbox_area_change_max":    0.50,

    # ── State classification thresholds (on overall score [0,1]) ─────────────
    "clear_threshold":         0.18,
    "ambiguous_threshold":     0.36,
    "occluded_threshold":      0.58,

    # ── History ──────────────────────────────────────────────────────────────
    # Number of past frames kept per track for EMA / jump detection
    "history_length":          5,
    # EMA alpha for per-track landmark smoothing used in jump detection
    "ema_alpha":               0.60,
}

# ---------------------------------------------------------------------------
# Shared joint definitions (duplicated from track_pose to avoid circular import)
# ---------------------------------------------------------------------------

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

_LOWER_BODY_IDXS = [
    _MP_JOINT_IDX["LEFT_HIP"],   _MP_JOINT_IDX["RIGHT_HIP"],
    _MP_JOINT_IDX["LEFT_KNEE"],  _MP_JOINT_IDX["RIGHT_KNEE"],
    _MP_JOINT_IDX["LEFT_ANKLE"], _MP_JOINT_IDX["RIGHT_ANKLE"],
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-6)


def _bbox_area(bbox: List[int]) -> float:
    return max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))


def _lm_to_array(lm: Dict) -> np.ndarray:
    """MediaPipe landmark dict → (_N_JOINTS, 3) float32."""
    arr = np.zeros((_N_JOINTS, 3), dtype=np.float32)
    for name, idx in _MP_JOINT_IDX.items():
        kp = lm.get(name)
        if kp:
            arr[idx, 0] = float(kp.get("x", 0.0))
            arr[idx, 1] = float(kp.get("y", 0.0))
            arr[idx, 2] = float(kp.get("visibility", 0.0))
    return arr


def _pose_jump(arr_new: np.ndarray, arr_ref: np.ndarray,
               min_conf: float = 0.25) -> float:
    """Mean normalised L2 displacement for joints visible in both arrays."""
    both = (arr_new[:, 2] > min_conf) & (arr_ref[:, 2] > min_conf)
    if both.sum() < 3:
        return 0.0
    diff = arr_new[both, :2] - arr_ref[both, :2]
    return float(np.mean(np.sqrt((diff ** 2).sum(axis=1))))


# ---------------------------------------------------------------------------
# Per-track history (for pose-jump and bbox-area-change)
# ---------------------------------------------------------------------------

class _TrackHistory:
    """Rolling landmark EMA and bbox area history for one ByteTrack track."""

    def __init__(self, alpha: float, history_length: int) -> None:
        self._alpha  = alpha
        self._lm_ema: Optional[np.ndarray] = None   # (_N_JOINTS, 3) EMA
        self._bbox_areas: deque = deque(maxlen=history_length)

    def update(self, lm_arr: np.ndarray, bbox: List[int]) -> None:
        """Update EMA and bbox history with this frame's observation."""
        if self._lm_ema is None:
            self._lm_ema = lm_arr.copy()
        else:
            a = self._alpha
            conf  = lm_arr[:, 2:3] >= 0.25
            self._lm_ema[:, :2] = np.where(
                conf,
                a * lm_arr[:, :2] + (1 - a) * self._lm_ema[:, :2],
                self._lm_ema[:, :2],
            )
            self._lm_ema[:, 2] = a * lm_arr[:, 2] + (1 - a) * self._lm_ema[:, 2]
        self._bbox_areas.append(_bbox_area(bbox))

    def pose_jump_from_ema(self, lm_arr: np.ndarray) -> float:
        if self._lm_ema is None:
            return 0.0
        return _pose_jump(lm_arr, self._lm_ema)

    def bbox_area_change(self, current_bbox: List[int]) -> float:
        """Fractional change in bbox area vs recent average."""
        if not self._bbox_areas:
            return 0.0
        prev_area = float(np.mean(self._bbox_areas))
        curr_area = _bbox_area(current_bbox)
        if prev_area < 1.0:
            return 0.0
        return abs(curr_area - prev_area) / prev_area


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals(
    handler_bbox:    List[int],
    handler_score:   float,
    second_score:    float,
    ball_detected:   bool,
    players:         List[Dict],
    handler_track_id: Optional[int],
    lm_arr:          np.ndarray,
    track_hist:      _TrackHistory,
    prev_track_id:   Optional[int],
    cfg:             Dict[str, Any],
) -> Tuple[Dict[str, float], List[int]]:
    """Compute raw signal values for one frame.

    Returns (signals_dict, nearby_track_ids).
    """
    # 1. Bbox overlap + nearby tracks
    max_iou   = 0.0
    nearby: List[int] = []
    for p in players:
        pb = p.get("bbox")
        if pb is None:
            continue
        if p.get("track_id") == handler_track_id:
            continue
        iou = _iou(handler_bbox, pb)
        if iou > cfg["bbox_overlap_threshold"]:
            nearby.append(p.get("track_id"))
        max_iou = max(max_iou, iou)

    # 2. Score gap (only meaningful when ball is detected)
    score_gap = (handler_score - second_score) if ball_detected else 1.0

    # 3. Pose jump vs EMA reference
    jump = track_hist.pose_jump_from_ema(lm_arr)

    # 4. Lower-body instability
    lb_conf    = lm_arr[_LOWER_BODY_IDXS, 2]
    frac_bad   = float((lb_conf < cfg["lower_body_conf_min"]).sum()) / len(_LOWER_BODY_IDXS)

    # 5. Track ID change (binary)
    id_changed = 1.0 if (prev_track_id is not None and
                          handler_track_id is not None and
                          handler_track_id != prev_track_id) else 0.0

    # 6. Bbox area change
    area_change = track_hist.bbox_area_change(handler_bbox)

    return {
        "bbox_overlap":          round(max_iou,       4),
        "score_gap":             round(score_gap,     4),
        "handler_score":         round(handler_score, 4),
        "pose_jump":             round(jump,          4),
        "lower_body_instability":round(frac_bad,      4),
        "track_id_change":       id_changed,
        "bbox_area_change":      round(min(area_change, 2.0), 4),
    }, nearby


def _normalise_signals(signals: Dict[str, float],
                       cfg: Dict[str, Any]) -> Dict[str, float]:
    """Map raw signal values to [0, 1] contribution scores."""
    s = signals

    # bbox_overlap: linear from 0 → severe
    ov_n = min(1.0, s["bbox_overlap"] / max(cfg["bbox_overlap_severe"], 1e-6))

    # score_gap: small gap → high score  (inverted, clamped)
    gap_n = max(0.0, 1.0 - s["score_gap"] / max(cfg["score_gap_full_margin"], 1e-6))
    gap_n = min(1.0, gap_n)

    # handler_score: low score → high uncertainty
    hs_n = max(0.0, 1.0 - s["handler_score"] / max(cfg["handler_score_confident"], 1e-6))
    hs_n = min(1.0, hs_n)

    # pose_jump: linear from threshold → severe
    j_range = max(cfg["pose_jump_severe"] - cfg["pose_jump_threshold"], 1e-6)
    j_n = max(0.0, (s["pose_jump"] - cfg["pose_jump_threshold"]) / j_range)
    j_n = min(1.0, j_n)

    # lower_body_instability: already a fraction [0,1]; scale by bad threshold
    lb_n = min(1.0, s["lower_body_instability"] / max(cfg["lower_body_fraction_bad"], 1e-6))

    # track_id_change: already 0 or 1
    tc_n = s["track_id_change"]

    # bbox_area_change: linear up to max
    ba_n = min(1.0, s["bbox_area_change"] / max(cfg["bbox_area_change_max"], 1e-6))

    return {
        "bbox_overlap":          round(ov_n,  4),
        "score_gap":             round(gap_n, 4),
        "handler_score":         round(hs_n,  4),
        "pose_jump":             round(j_n,   4),
        "lower_body_instability":round(lb_n,  4),
        "track_id_change":       round(tc_n,  4),
        "bbox_area_change":      round(ba_n,  4),
    }


def _weighted_score(norm_signals: Dict[str, float],
                    cfg: Dict[str, Any]) -> float:
    """Compute overall ambiguity score [0, 1] as weighted sum."""
    score = (
        cfg["w_bbox_overlap"]     * norm_signals["bbox_overlap"]     +
        cfg["w_score_gap"]        * norm_signals["score_gap"]        +
        cfg["w_handler_score"]    * norm_signals["handler_score"]    +
        cfg["w_pose_jump"]        * norm_signals["pose_jump"]        +
        cfg["w_lower_body"]       * norm_signals["lower_body_instability"] +
        cfg["w_track_id_change"]  * norm_signals["track_id_change"]  +
        cfg["w_bbox_area_change"] * norm_signals["bbox_area_change"]
    )
    return round(min(1.0, score), 4)


def _classify(
    score:          float,
    signals:        Dict[str, float],
    handler_present: bool,
    cfg:            Dict[str, Any],
) -> Tuple[str, str]:
    """Return (state, recommended_pose_mode)."""
    if not handler_present:
        return "occluded", "carry_forward"

    ov = signals["bbox_overlap"]

    if ov > cfg["bbox_overlap_severe"] or score >= cfg["occluded_threshold"]:
        return "occluded", "carry_forward"

    if score >= cfg["ambiguous_threshold"]:
        return "ambiguous", "refined"

    if ov > cfg["bbox_overlap_threshold"] or score >= cfg["clear_threshold"]:
        return "overlap", "refined"

    return "clear", "normal"


# ---------------------------------------------------------------------------
# Main analyser (stateful, processes frames in order)
# ---------------------------------------------------------------------------

class AmbiguityAnalyzer:
    """Stateful per-frame ambiguity classifier for the selected handler.

    Maintains per-track landmark EMA and bbox history to compute pose-jump
    and bbox-geometry-change signals without requiring video frame reads.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg              = config
        self._prev_track_id:   Optional[int] = None
        self._track_histories: Dict[int, _TrackHistory] = {}

    def _get_history(self, track_id: int) -> _TrackHistory:
        if track_id not in self._track_histories:
            self._track_histories[track_id] = _TrackHistory(
                self._cfg["ema_alpha"],
                self._cfg["history_length"],
            )
        return self._track_histories[track_id]

    def analyze(
        self,
        frame_idx:     int,
        ps_entry:      Dict,           # player_selection.per_frame[i]
        lm_dict:       Dict,           # data["frames"][i]["landmarks"]
        ball_dict:     Dict,           # data["ball"].per_frame[i]["ball"]
    ) -> Dict[str, Any]:
        """Analyse one frame; return the ambiguity record."""
        cfg = self._cfg

        track_id  = ps_entry.get("target_track_id")
        bbox_raw  = ps_entry.get("smoothed_bbox") or ps_entry.get("target_bbox")
        players   = ps_entry.get("players", [])

        # No handler assigned
        if track_id is None or bbox_raw is None:
            rec = _empty_record(frame_idx, track_id)
            self._prev_track_id = track_id
            return rec

        handler_bbox = list(bbox_raw)
        lm_arr = _lm_to_array(lm_dict) if lm_dict else np.zeros((_N_JOINTS, 3), np.float32)

        # Ball info
        ball_detected = bool(ball_dict.get("tracked") and ball_dict.get("center"))

        # Handler + second-best possession scores
        handler_score = 0.0
        second_score  = 0.0
        all_scores_sorted = sorted(
            [p for p in players if p.get("score", 0) > 0],
            key=lambda p: -p.get("score", 0),
        )
        if all_scores_sorted:
            if all_scores_sorted[0].get("track_id") == track_id:
                handler_score = all_scores_sorted[0].get("score", 0.0)
                second_score  = all_scores_sorted[1].get("score", 0.0) if len(all_scores_sorted) > 1 else 0.0
            else:
                # Selected handler is not the top scorer (temporal override)
                for p in all_scores_sorted:
                    if p.get("track_id") == track_id:
                        handler_score = p.get("score", 0.0)
                        break
                second_score = all_scores_sorted[0].get("score", 0.0)

        # Per-track history (update AFTER computing signals so history is from prior frame)
        hist = self._get_history(track_id)

        # Compute signals
        signals, nearby = _compute_signals(
            handler_bbox, handler_score, second_score, ball_detected,
            players, track_id, lm_arr, hist, self._prev_track_id, cfg,
        )
        norm_sig = _normalise_signals(signals, cfg)
        score    = _weighted_score(norm_sig, cfg)
        state, mode = _classify(score, signals, True, cfg)

        # Update history with this frame's observation
        hist.update(lm_arr, handler_bbox)
        self._prev_track_id = track_id

        # Log notable frames
        if state in ("ambiguous", "occluded"):
            logger.debug(
                "f%04d  T%s  %-10s  score=%.3f  ov=%.2f  gap=%.2f  jump=%.3f",
                frame_idx, track_id, state, score,
                signals["bbox_overlap"], signals["score_gap"], signals["pose_jump"],
            )

        return {
            "frame_idx":             frame_idx,
            "selected_track_id":     track_id,
            "state":                 state,
            "score":                 score,
            "signals":               signals,
            "signal_scores":         norm_sig,
            "nearby_tracks":         [t for t in nearby if t is not None],
            "recommended_pose_mode": mode,
        }


def _empty_record(frame_idx: int, track_id: Optional[int]) -> Dict:
    empty_sig = {k: 0.0 for k in [
        "bbox_overlap", "score_gap", "handler_score", "pose_jump",
        "lower_body_instability", "track_id_change", "bbox_area_change",
    ]}
    return {
        "frame_idx":             frame_idx,
        "selected_track_id":     track_id,
        "state":                 "occluded",
        "score":                 1.0,
        "signals":               empty_sig,
        "signal_scores":         empty_sig,
        "nearby_tracks":         [],
        "recommended_pose_mode": "carry_forward",
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def compute_ambiguity(
    data:   Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify every frame's handler state and write data["ambiguity"].

    Must be called AFTER ``select_target_player()`` (needs player_selection
    and landmarks) and BEFORE ``refine_handler_pose()`` (which reads the
    recommended_pose_mode to route its logic).

    Parameters
    ----------
    data :   Pivot dict with player_selection, ball, and frames already set.
    config : Optional overrides for _DEFAULT_CONFIG.

    Returns
    -------
    The same dict with a new ``data["ambiguity"]`` key.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    ps_per_frame   = data.get("player_selection", {}).get("per_frame", [])
    ball_per_frame = data.get("ball", {}).get("per_frame", [])
    frames_data    = data.get("frames", [])

    ball_index: Dict[int, Dict] = {
        r["frame_idx"]: r.get("ball", {})
        for r in ball_per_frame if "frame_idx" in r
    }

    analyser = AmbiguityAnalyzer(cfg)
    results: List[Dict] = []

    for ps_entry in ps_per_frame:
        fidx    = ps_entry.get("frame_idx")
        if fidx is None:
            continue
        lm_dict  = (frames_data[fidx].get("landmarks") or {}) if fidx < len(frames_data) else {}
        ball_d   = ball_index.get(fidx, {})
        rec      = analyser.analyze(fidx, ps_entry, lm_dict, ball_d)
        results.append(rec)

    # ── Summary ──────────────────────────────────────────────────────────────
    state_counts: Dict[str, int] = {"clear": 0, "overlap": 0, "ambiguous": 0, "occluded": 0}
    mode_counts:  Dict[str, int] = {"normal": 0, "refined": 0, "carry_forward": 0}
    for r in results:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
        mode_counts[r["recommended_pose_mode"]] = mode_counts.get(r["recommended_pose_mode"], 0) + 1

    summary = {
        "n_frames":  len(results),
        "states":    state_counts,
        "modes":     mode_counts,
    }

    logger.info(
        "Ambiguity: clear=%d  overlap=%d  ambiguous=%d  occluded=%d  "
        "(normal=%d  refined=%d  carry=%d)",
        state_counts["clear"], state_counts["overlap"],
        state_counts["ambiguous"], state_counts["occluded"],
        mode_counts["normal"], mode_counts["refined"], mode_counts["carry_forward"],
    )

    data["ambiguity"] = {
        "config":    cfg,
        "summary":   summary,
        "per_frame": results,
    }
    return data
