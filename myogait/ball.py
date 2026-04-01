"""Ball possession analysis for single-player basketball clips.

Pipeline
--------
1. detect_ball_frames  – run YOLO over every video frame
2. track_ball_frames   – interpolate/predict across gaps in detections
3. classify_ball_state – per-frame state from ball + hand geometry
4. smooth_ball_states  – temporal window smoothing
5. analyze_ball        – orchestrate all four; writes ``data["ball"]``
6. render_ball_video   – debug overlay video

Tracking sources
----------------
Each per-frame ball dict carries a ``source`` field:

``detected``      Direct YOLO hit.
``interpolated``  Linear interpolation across a short detection gap.
``predicted``     Velocity extrapolation at the clip start/end (optional).
``none``          No position available.

States
------
``left_hand_control``   Ball within control radius of left hand.
``right_hand_control``  Ball within control radius of right hand.
``both_uncertain``      Ball within control radius of both hands.
``free``                Ball detected but not near either hand.
``no_ball_detected``    No ball position available (even after tracking).
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detectors.ball_detector import BallDetection, BallDetector, create_ball_detector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

BALL_STATES = (
    "left_hand_control",
    "right_hand_control",
    "both_uncertain",
    "free",
    "no_ball_detected",
)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "detector": "roboflow",
    "detector_kwargs": {
        # xil7x (basketball-xil7x/1) is the default.
        # api_key can be set here or via ROBOFLOW_API_KEY env var.
        "project_id": "basketball-xil7x",
        "version": 1,
        "target_classes": ["ball"],
    },
    # Fraction of body_scale below which ball is considered "in hand"
    "control_threshold": 0.40,
    # Minimum landmark visibility to trust a hand position.
    "min_visibility": 0.05,
    # Temporal smoothing window (frames, odd recommended)
    "smoothing_window": 7,
    # When True, attach per-frame class_label debug info to detection results
    "debug_candidates": False,
    # --- Temporal tracking ---
    # Max number of consecutive non-detected frames to bridge with interpolation.
    # Set to 0 to disable interpolation entirely.
    "max_interp_gap": 8,
    # Max frames to extrapolate using velocity at the clip start/end.
    # 0 = disabled (default; edge prediction is riskier than gap interpolation).
    "max_predict_frames": 0,
    # Min YOLO confidence on each end of a gap to allow interpolation.
    "interp_min_conf": 0.10,
    # Max implied ball speed (px/frame) across a gap.
    # Gaps that require faster motion are skipped (likely scene cut or wrong detection).
    "max_ball_speed_px": 150.0,
}

# Hand landmark names (MediaPipe 33) ordered by priority for centroid
_HAND_LANDMARKS = {
    "left":  ["LEFT_WRIST",  "LEFT_INDEX",  "LEFT_PINKY",  "LEFT_THUMB"],
    "right": ["RIGHT_WRIST", "RIGHT_INDEX", "RIGHT_PINKY", "RIGHT_THUMB"],
}

# Body-scale reference landmarks
_BODY_SCALE_LANDMARKS = [
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _norm_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _hand_centroid(
    side: str,
    landmarks: dict,
    min_visibility: float,
) -> Optional[Tuple[float, float]]:
    """Weighted centroid of hand landmarks in normalised [0,1] space."""
    keys = _HAND_LANDMARKS[side]
    xs, ys, ws = [], [], []
    for key in keys:
        lm = landmarks.get(key)
        if lm is None:
            continue
        vis = lm.get("visibility", 1.0)
        if vis < min_visibility:
            continue
        x = lm.get("x")
        y = lm.get("y")
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
        ws.append(vis)
    if not xs:
        return None
    total_w = sum(ws)
    cx = sum(x * w for x, w in zip(xs, ws)) / total_w
    cy = sum(y * w for y, w in zip(ys, ws)) / total_w
    return (cx, cy)


def _body_scale(landmarks: dict) -> float:
    """Return an approximate body height scale in normalised coords."""
    lengths = []
    for (a_name, b_name) in _BODY_SCALE_LANDMARKS:
        a = landmarks.get(a_name)
        b = landmarks.get(b_name)
        if a is None or b is None:
            continue
        ax, ay = a.get("x", float("nan")), a.get("y", float("nan"))
        bx, by = b.get("x", float("nan")), b.get("y", float("nan"))
        if any(math.isnan(v) for v in (ax, ay, bx, by)):
            continue
        d = _norm_dist(ax, ay, bx, by)
        if d > 1e-4:
            lengths.append(d)
    return float(np.mean(lengths)) if lengths else 0.4


# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------

def detect_ball_frames(
    video_path: str,
    data: dict,
    detector: Optional[BallDetector] = None,
    config: Optional[dict] = None,
) -> List[dict]:
    """Detect the ball in every frame of *video_path* using YOLO.

    Returns raw per-frame detection dicts.  Call ``track_ball_frames()``
    afterwards to fill gaps with interpolation.

    Returns
    -------
    list of dict
        One entry per frame.  Each dict comes from ``BallDetection.to_dict()``
        and has ``detected``, ``bbox``, ``center``, ``radius``, ``confidence``.
        After this step ``tracked`` and ``source`` are NOT yet set; that is
        done by ``track_ball_frames()``.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    debug = cfg.get("debug_candidates", False)

    if detector is None:
        det_kwargs = cfg.get("detector_kwargs", {})
        detector = create_ball_detector(cfg["detector"], **det_kwargs)

    if hasattr(detector, "reset"):
        detector.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames_data = data.get("frames", [])
    n_frames = len(frames_data)
    results: List[dict] = []
    prev_detection: Optional[BallDetection] = None

    frame_pos = 0
    while frame_pos < n_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            while len(results) < n_frames:
                results.append(BallDetection().to_dict(include_debug=debug))
            break

        lm = frames_data[frame_pos].get("landmarks") if frame_pos < n_frames else None
        detection = detector.detect(frame_bgr, landmarks=lm, prev_detection=prev_detection)
        results.append(detection.to_dict(include_debug=debug))

        if detection.detected:
            prev_detection = detection
        frame_pos += 1

    cap.release()
    n_det = sum(1 for r in results if r["detected"])
    logger.info(
        "YOLO detection: %d/%d frames (%.1f%%)",
        n_det, len(results), 100 * n_det / max(len(results), 1),
    )
    return results


# ---------------------------------------------------------------------------
# Temporal tracking / gap interpolation
# ---------------------------------------------------------------------------

def _estimate_velocity(
    frames: List[dict],
    det_indices: List[int],
) -> Optional[Tuple[float, float]]:
    """Return mean velocity (px/frame) from up to the last 3 consecutive detections."""
    valid = [(i, frames[i]) for i in det_indices if frames[i].get("center")]
    if len(valid) < 2:
        return None
    vxs, vys = [], []
    for k in range(len(valid) - 1):
        i0, f0 = valid[k]
        i1, f1 = valid[k + 1]
        dt = i1 - i0
        if dt == 0:
            continue
        c0, c1 = f0["center"], f1["center"]
        vxs.append((c1[0] - c0[0]) / dt)
        vys.append((c1[1] - c0[1]) / dt)
    if not vxs:
        return None
    return (float(np.mean(vxs)), float(np.mean(vys)))


def track_ball_frames(
    raw_detections: List[dict],
    config: Optional[dict] = None,
) -> List[dict]:
    """Apply temporal tracking/interpolation on top of raw YOLO detections.

    For each gap of non-detected frames between two valid YOLO detections,
    if the gap is short enough and the implied speed is plausible, the gap
    is filled with linear interpolation.

    Optionally extrapolates at the clip start/end using ``max_predict_frames``
    (disabled by default — edge prediction is conservative by design).

    New fields added to every frame dict
    -------------------------------------
    ``tracked`` : bool
        True if a ball position is available from any source.
    ``source`` : str
        ``"detected"`` | ``"interpolated"`` | ``"predicted"`` | ``"none"``

    Interpolated confidence
    -----------------------
    Confidence is linearly interpolated between the two anchor detections,
    then reduced by a sinusoidal penalty that peaks at the centre of the gap
    (max –40 %).  This makes far-from-anchor interpolations visibly less
    confident in the debug output without making them useless.

    Parameters
    ----------
    raw_detections : list of dict
        Output of ``detect_ball_frames()``.
    config : dict, optional
        Accepted keys (see ``_DEFAULT_CONFIG`` for defaults):
        ``max_interp_gap``, ``max_predict_frames``,
        ``interp_min_conf``, ``max_ball_speed_px``.

    Returns
    -------
    list of dict
        One augmented dict per frame.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    max_gap = int(cfg["max_interp_gap"])
    max_pred = int(cfg["max_predict_frames"])
    min_conf = float(cfg["interp_min_conf"])
    max_speed = float(cfg["max_ball_speed_px"])

    n = len(raw_detections)
    result = [dict(d) for d in raw_detections]

    # Initialise tracking metadata
    for d in result:
        if d.get("detected"):
            d["tracked"] = True
            d["source"] = "detected"
        else:
            d["tracked"] = False
            d["source"] = "none"

    det_indices = [i for i, d in enumerate(result) if d.get("detected")]
    if not det_indices:
        logger.info("Ball tracking: no YOLO detections — nothing to interpolate")
        return result

    # ------------------------------------------------------------------
    # Interpolate gaps between consecutive detections
    # ------------------------------------------------------------------
    if max_gap > 0:
        for k in range(len(det_indices) - 1):
            i = det_indices[k]      # left anchor (detected)
            j = det_indices[k + 1]  # right anchor (detected)
            gap = j - i - 1
            if gap < 1 or gap > max_gap:
                continue

            left = result[i]
            right = result[j]

            # Both anchors need enough confidence to trust them as anchors
            if (left.get("confidence", 0) < min_conf
                    or right.get("confidence", 0) < min_conf):
                continue

            lc = left.get("center")
            rc = right.get("center")
            if lc is None or rc is None:
                continue

            dx = rc[0] - lc[0]
            dy = rc[1] - lc[1]
            dist = math.hypot(dx, dy)
            speed = dist / (gap + 1)   # px/frame across the gap

            if speed > max_speed:
                logger.debug(
                    "Gap [%d→%d] skipped: %.1fpx/frame > limit %.1f",
                    i, j, speed, max_speed,
                )
                continue

            lb = left.get("bbox")
            rb = right.get("bbox")
            lr = float(left.get("radius") or 20.0)
            rr = float(right.get("radius") or 20.0)
            l_conf = float(left.get("confidence", 0.5))
            r_conf = float(right.get("confidence", 0.5))

            for step in range(1, gap + 1):
                t = step / (gap + 1)   # position in (0, 1)
                idx = i + step

                cx = lc[0] + t * dx
                cy = lc[1] + t * dy
                radius = lr + t * (rr - lr)

                if lb is not None and rb is not None:
                    bbox = [
                        int(round(lb[0] + t * (rb[0] - lb[0]))),
                        int(round(lb[1] + t * (rb[1] - lb[1]))),
                        int(round(lb[2] + t * (rb[2] - lb[2]))),
                        int(round(lb[3] + t * (rb[3] - lb[3]))),
                    ]
                else:
                    r = max(1, int(radius))
                    bbox = [int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r]

                # Linear-interpolated confidence with a sinusoidal centre penalty
                base_conf = l_conf * (1.0 - t) + r_conf * t
                centre_penalty = 1.0 - 0.40 * math.sin(t * math.pi)
                interp_conf = base_conf * centre_penalty

                result[idx].update({
                    "tracked": True,
                    "source": "interpolated",
                    "center": [round(cx, 1), round(cy, 1)],
                    "bbox": bbox,
                    "radius": round(radius, 1),
                    "confidence": round(interp_conf, 4),
                })

    # ------------------------------------------------------------------
    # Extrapolate at the leading edge (before first detection)
    # ------------------------------------------------------------------
    if max_pred > 0 and det_indices[0] > 0:
        vel = _estimate_velocity(result, det_indices[:min(3, len(det_indices))])
        if vel is not None:
            vx, vy = vel
            anchor = result[det_indices[0]]
            lc = anchor.get("center")
            lr = float(anchor.get("radius") or 20.0)
            lb = anchor.get("bbox")
            base_conf = float(anchor.get("confidence", 0.3))

            for step in range(1, min(det_indices[0], max_pred) + 1):
                idx = det_indices[0] - step
                if result[idx]["tracked"]:
                    continue
                cx = lc[0] - step * vx
                cy = lc[1] - step * vy
                r = max(1, int(lr))
                if lb:
                    bbox = [
                        int(lb[0] - step * vx), int(lb[1] - step * vy),
                        int(lb[2] - step * vx), int(lb[3] - step * vy),
                    ]
                else:
                    bbox = [int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r]
                pred_conf = base_conf * (0.60 ** step)
                result[idx].update({
                    "tracked": True,
                    "source": "predicted",
                    "center": [round(cx, 1), round(cy, 1)],
                    "bbox": bbox,
                    "radius": round(lr, 1),
                    "confidence": round(pred_conf, 4),
                })

    # ------------------------------------------------------------------
    # Extrapolate at the trailing edge (after last detection)
    # ------------------------------------------------------------------
    if max_pred > 0 and det_indices[-1] < n - 1:
        vel = _estimate_velocity(result, det_indices[max(0, len(det_indices) - 3):])
        if vel is not None:
            vx, vy = vel
            anchor = result[det_indices[-1]]
            lc = anchor.get("center")
            lr = float(anchor.get("radius") or 20.0)
            lb = anchor.get("bbox")
            base_conf = float(anchor.get("confidence", 0.3))

            for step in range(1, min(n - 1 - det_indices[-1], max_pred) + 1):
                idx = det_indices[-1] + step
                if result[idx]["tracked"]:
                    break
                cx = lc[0] + step * vx
                cy = lc[1] + step * vy
                r = max(1, int(lr))
                if lb:
                    bbox = [
                        int(lb[0] + step * vx), int(lb[1] + step * vy),
                        int(lb[2] + step * vx), int(lb[3] + step * vy),
                    ]
                else:
                    bbox = [int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r]
                pred_conf = base_conf * (0.60 ** step)
                result[idx].update({
                    "tracked": True,
                    "source": "predicted",
                    "center": [round(cx, 1), round(cy, 1)],
                    "bbox": bbox,
                    "radius": round(lr, 1),
                    "confidence": round(pred_conf, 4),
                })

    n_interp = sum(1 for d in result if d.get("source") == "interpolated")
    n_pred = sum(1 for d in result if d.get("source") == "predicted")
    n_tracked = sum(1 for d in result if d.get("tracked"))
    logger.info(
        "Ball tracking: %d detected + %d interpolated + %d predicted = %d/%d (%.1f%%) covered",
        len(det_indices), n_interp, n_pred, n_tracked, n,
        100 * n_tracked / max(n, 1),
    )
    return result


# ---------------------------------------------------------------------------
# Per-frame state classification
# ---------------------------------------------------------------------------

def classify_ball_state(
    ball: dict,
    landmarks: dict,
    config: Optional[dict] = None,
) -> dict:
    """Classify ball possession state for a single frame.

    Uses ``ball["tracked"]`` (set by ``track_ball_frames``) rather than
    ``ball["detected"]`` so that interpolated and predicted positions also
    produce meaningful states.

    Returns
    -------
    dict with keys:
        ``state``, ``left_dist``, ``right_dist``, ``body_scale``,
        ``left_hand``, ``right_hand``, ``ball_norm``
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    threshold = cfg["control_threshold"]
    min_vis = cfg["min_visibility"]

    result = {
        "state": "no_ball_detected",
        "left_dist": None,
        "right_dist": None,
        "body_scale": None,
        "left_hand": None,
        "right_hand": None,
        "ball_norm": None,
    }

    # Use "tracked" (covers detected + interpolated + predicted)
    if not ball.get("tracked"):
        return result

    center = ball.get("center")
    if center is None:
        return result

    frame_w = ball.get("_frame_w")
    frame_h = ball.get("_frame_h")

    cx_px, cy_px = center
    if frame_w and frame_h:
        bx_n = cx_px / frame_w
        by_n = cy_px / frame_h
    else:
        bx_n = cx_px
        by_n = cy_px

    result["ball_norm"] = [round(bx_n, 4), round(by_n, 4)]

    scale = _body_scale(landmarks)
    result["body_scale"] = round(scale, 4)

    left_hand = _hand_centroid("left", landmarks, min_vis)
    right_hand = _hand_centroid("right", landmarks, min_vis)
    result["left_hand"] = [round(v, 4) for v in left_hand] if left_hand else None
    result["right_hand"] = [round(v, 4) for v in right_hand] if right_hand else None

    left_dist = right_dist = None
    if left_hand is not None:
        d = _norm_dist(bx_n, by_n, left_hand[0], left_hand[1])
        left_dist = d / scale if scale > 1e-4 else d
    if right_hand is not None:
        d = _norm_dist(bx_n, by_n, right_hand[0], right_hand[1])
        right_dist = d / scale if scale > 1e-4 else d

    result["left_dist"] = round(left_dist, 4) if left_dist is not None else None
    result["right_dist"] = round(right_dist, 4) if right_dist is not None else None

    left_ctrl = left_dist is not None and left_dist < threshold
    right_ctrl = right_dist is not None and right_dist < threshold

    if left_ctrl and right_ctrl:
        state = "both_uncertain"
    elif left_ctrl:
        state = "left_hand_control"
    elif right_ctrl:
        state = "right_hand_control"
    else:
        state = "free"

    result["state"] = state
    return result


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_ball_states(
    per_frame: List[dict],
    config: Optional[dict] = None,
) -> List[str]:
    """Apply a sliding-window mode filter to frame-level ball states."""
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    window = max(1, cfg["smoothing_window"])
    half = window // 2

    raw_states = [f["state"] for f in per_frame]
    smoothed: List[str] = []

    for i, state in enumerate(raw_states):
        if state == "no_ball_detected":
            smoothed.append(state)
            continue
        start = max(0, i - half)
        end = min(len(raw_states), i + half + 1)
        window_states = [
            s for s in raw_states[start:end]
            if s != "no_ball_detected"
        ]
        if not window_states:
            smoothed.append(state)
        else:
            vote = Counter(window_states).most_common(1)[0][0]
            smoothed.append(vote)

    return smoothed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_ball(
    video_path: str,
    data: dict,
    config: Optional[dict] = None,
    detector: Optional[BallDetector] = None,
) -> dict:
    """Run the full ball possession pipeline and store results in *data*.

    Steps
    -----
    1. YOLO detection per frame (``detect_ball_frames``)
    2. Gap interpolation / tracking  (``track_ball_frames``)
    3. Per-frame state classification (``classify_ball_state``)
    4. Temporal smoothing            (``smooth_ball_states``)

    Adds ``data["ball"]`` with per-frame results + summary stats.

    Returns
    -------
    dict
        The updated *data* dict.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    # Step 1: YOLO
    logger.info("Step 1/4: Detecting ball in %s …", os.path.basename(video_path))
    raw_detections = detect_ball_frames(video_path, data, detector=detector, config=cfg)

    # Step 2: Tracking
    logger.info("Step 2/4: Tracking ball across gaps …")
    ball_detections = track_ball_frames(raw_detections, config=cfg)

    # Inject frame dimensions (needed for normalisation in classify_ball_state)
    meta = data.get("meta", {})
    frame_w = meta.get("width")
    frame_h = meta.get("height")
    for bd in ball_detections:
        bd["_frame_w"] = frame_w
        bd["_frame_h"] = frame_h

    # Step 3: Classify
    logger.info("Step 3/4: Classifying ball state per frame …")
    frames_data = data.get("frames", [])
    per_frame: List[dict] = []

    for i, (frame_info, ball_det) in enumerate(zip(frames_data, ball_detections)):
        landmarks = frame_info.get("landmarks", {})
        classification = classify_ball_state(ball_det, landmarks, config=cfg)
        entry = {
            "frame_idx": frame_info.get("frame_idx", i),
            "time_s": round(frame_info.get("time_s", 0.0), 4),
            "ball": {k: v for k, v in ball_det.items() if not k.startswith("_")},
            **classification,
        }
        per_frame.append(entry)

    # Step 4: Smooth
    logger.info("Step 4/4: Smoothing states (window=%d) …", cfg["smoothing_window"])
    smoothed_states = smooth_ball_states(per_frame, config=cfg)
    for entry, sm in zip(per_frame, smoothed_states):
        entry["state_smoothed"] = sm

    # Summary
    state_counts = Counter(e["state_smoothed"] for e in per_frame)
    n_frames = len(per_frame)
    fps = meta.get("fps", 1.0)

    n_yolo = sum(1 for d in ball_detections if d.get("detected"))
    n_tracked = sum(1 for d in ball_detections if d.get("tracked"))
    n_interp = sum(1 for d in ball_detections if d.get("source") == "interpolated")
    n_pred = sum(1 for d in ball_detections if d.get("source") == "predicted")

    data["ball"] = {
        "method": cfg["detector"],
        "config": {
            "control_threshold": cfg["control_threshold"],
            "min_visibility": cfg["min_visibility"],
            "smoothing_window": cfg["smoothing_window"],
            "max_interp_gap": cfg["max_interp_gap"],
            "max_ball_speed_px": cfg["max_ball_speed_px"],
        },
        "summary": {
            "n_frames": n_frames,
            # Raw YOLO
            "n_ball_detected": n_yolo,
            "yolo_detection_rate": round(n_yolo / max(n_frames, 1), 3),
            # Tracking (backward-compat alias)
            "detection_rate": round(n_yolo / max(n_frames, 1), 3),
            # Tracked total
            "n_tracked": n_tracked,
            "n_interpolated": n_interp,
            "n_predicted": n_pred,
            "tracked_coverage_rate": round(n_tracked / max(n_frames, 1), 3),
            # State analysis on smoothed states
            "state_counts": dict(state_counts),
            "state_durations_s": {
                state: round(count / fps, 3)
                for state, count in state_counts.items()
            },
        },
        "per_frame": per_frame,
    }

    dominant = state_counts.most_common(1)
    logger.info(
        "Ball analysis done. YOLO: %.1f%% | tracked: %.1f%% | dominant: %s",
        100 * data["ball"]["summary"]["yolo_detection_rate"],
        100 * data["ball"]["summary"]["tracked_coverage_rate"],
        dominant[0][0] if dominant else "n/a",
    )
    return data


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def ball_to_csv(data: dict, output_path: str) -> str:
    """Write per-frame ball results to a CSV file."""
    ball_data = data.get("ball", {})
    per_frame = ball_data.get("per_frame", [])
    if not per_frame:
        raise ValueError("No ball data found. Run analyze_ball() first.")

    fieldnames = [
        "frame_idx", "time_s",
        "ball_detected", "ball_tracked", "ball_source",
        "ball_cx_px", "ball_cy_px", "ball_confidence",
        "left_hand_x", "left_hand_y",
        "right_hand_x", "right_hand_y",
        "left_dist", "right_dist", "body_scale",
        "state", "state_smoothed",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in per_frame:
            ball = entry.get("ball", {})
            center = ball.get("center") or [None, None]
            lh = entry.get("left_hand") or [None, None]
            rh = entry.get("right_hand") or [None, None]
            writer.writerow({
                "frame_idx": entry.get("frame_idx"),
                "time_s": entry.get("time_s"),
                "ball_detected": ball.get("detected"),
                "ball_tracked": ball.get("tracked"),
                "ball_source": ball.get("source", "none"),
                "ball_cx_px": center[0],
                "ball_cy_px": center[1],
                "ball_confidence": ball.get("confidence"),
                "left_hand_x": lh[0],
                "left_hand_y": lh[1],
                "right_hand_x": rh[0],
                "right_hand_y": rh[1],
                "left_dist": entry.get("left_dist"),
                "right_dist": entry.get("right_dist"),
                "body_scale": entry.get("body_scale"),
                "state": entry.get("state"),
                "state_smoothed": entry.get("state_smoothed"),
            })
    logger.info("Ball CSV written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Debug video rendering
# ---------------------------------------------------------------------------

# BGR colour palette for states
_STATE_COLORS = {
    "left_hand_control":  (255, 140, 0),
    "right_hand_control": (0, 60, 255),
    "both_uncertain":     (0, 200, 200),
    "free":               (180, 180, 180),
    "no_ball_detected":   (50, 50, 50),
}
_COLOR_WHITE          = (255, 255, 255)
_COLOR_BALL_DETECTED  = (0, 165, 255)    # orange  — direct YOLO hit
_COLOR_BALL_INTERP    = (0, 230, 255)    # yellow  — interpolated
_COLOR_BALL_PRED      = (130, 170, 255)  # light blue/peach — predicted
_COLOR_LEFT_HAND      = (255, 100, 0)
_COLOR_RIGHT_HAND     = (0, 0, 255)


def render_ball_video(
    video_path: str,
    data: dict,
    output_path: str,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    show_skeleton: bool = True,
) -> str:
    """Render a debug video with ball detection/tracking and hand state overlay.

    Ball colour codes by source
    ---------------------------
    Orange   — direct YOLO detection
    Yellow   — interpolated (gap-filled)
    Blue/peach — predicted (velocity extrapolation)

    Parameters
    ----------
    video_path : str
        Source video.
    data : dict
        Pivot dict with ``data["ball"]`` populated by ``analyze_ball()``.
    output_path : str
        Destination path.
    fps, codec, show_skeleton
        Standard rendering options.

    Returns
    -------
    str
        *output_path* written.
    """
    from .video import render_skeleton_frame

    ball_data = data.get("ball", {})
    per_frame = ball_data.get("per_frame", [])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps if fps is not None else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _codecs = [codec]
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4":
        _codecs += [c for c in ("mp4v", "avc1", "XVID") if c != codec]
    writer = None
    for c in _codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError(f"Could not open video writer for '{output_path}'")

    frames_data = data.get("frames", []) if show_skeleton else []
    was_flipped = data.get("extraction", {}).get("was_flipped", False)

    frame_pos = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_pos < len(per_frame):
            entry = per_frame[frame_pos]
            frame = _draw_ball_overlay(frame, entry, width, height)

            if show_skeleton and frame_pos < len(frames_data):
                fd = frames_data[frame_pos]
                lm = fd.get("landmarks", {})
                if was_flipped and lm:
                    lm = {
                        name: {**val, "x": 1.0 - val["x"]}
                        for name, val in lm.items()
                        if val.get("x") is not None
                    }
                frame = render_skeleton_frame(frame, lm)

        writer.write(frame)
        frame_pos += 1

    cap.release()
    writer.release()
    logger.info("Ball debug video written to %s", output_path)
    return output_path


def _draw_ball_overlay(
    frame: "np.ndarray",
    entry: dict,
    frame_w: int,
    frame_h: int,
) -> "np.ndarray":
    """Draw ball + hand overlay on a single frame (in-place copy)."""
    frame = frame.copy()
    state = entry.get("state_smoothed") or entry.get("state", "no_ball_detected")
    state_color = _STATE_COLORS.get(state, _COLOR_WHITE)

    # --- Ball ---
    ball = entry.get("ball", {})
    ball_center_px = None
    source = ball.get("source", "detected" if ball.get("detected") else "none")
    tracked = ball.get("tracked", ball.get("detected", False))

    if tracked and ball.get("center"):
        if source == "detected":
            ball_color = _COLOR_BALL_DETECTED
            src_tag = "det"
            box_thick = 2
            circ_thick = 2
        elif source == "interpolated":
            ball_color = _COLOR_BALL_INTERP
            src_tag = "interp"
            box_thick = 1
            circ_thick = 1
        elif source == "predicted":
            ball_color = _COLOR_BALL_PRED
            src_tag = "pred"
            box_thick = 1
            circ_thick = 1
        else:
            ball_color = _COLOR_BALL_DETECTED
            src_tag = "det"
            box_thick = 2
            circ_thick = 2

        center = ball["center"]
        bbox = ball.get("bbox")
        radius = ball.get("radius")
        cx, cy = int(center[0]), int(center[1])
        ball_center_px = (cx, cy)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, box_thick, cv2.LINE_AA)
        r = int(radius) if radius else 15
        cv2.circle(frame, (cx, cy), r, ball_color, circ_thick, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, ball_color, -1, cv2.LINE_AA)

        conf_label = f"{src_tag}:{ball.get('confidence', 0):.2f}"
        cv2.putText(
            frame, conf_label, (cx + r + 4, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 1, cv2.LINE_AA,
        )

    # --- Hand centroids ---
    left_hand = entry.get("left_hand")
    right_hand = entry.get("right_hand")
    left_px = right_px = None

    if left_hand:
        lx = int(left_hand[0] * frame_w)
        ly = int(left_hand[1] * frame_h)
        left_px = (lx, ly)
        cv2.circle(frame, (lx, ly), 8, _COLOR_LEFT_HAND, -1, cv2.LINE_AA)
        cv2.circle(frame, (lx, ly), 8, _COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "L", (lx - 4, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_WHITE, 1, cv2.LINE_AA)

    if right_hand:
        rx = int(right_hand[0] * frame_w)
        ry = int(right_hand[1] * frame_h)
        right_px = (rx, ry)
        cv2.circle(frame, (rx, ry), 8, _COLOR_RIGHT_HAND, -1, cv2.LINE_AA)
        cv2.circle(frame, (rx, ry), 8, _COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "R", (rx - 4, ry + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_WHITE, 1, cv2.LINE_AA)

    # --- Distance lines ---
    if ball_center_px:
        left_dist = entry.get("left_dist")
        right_dist = entry.get("right_dist")

        if left_px and left_dist is not None:
            cv2.line(frame, ball_center_px, left_px, _COLOR_LEFT_HAND, 1, cv2.LINE_AA)
            mid = ((ball_center_px[0] + left_px[0]) // 2,
                   (ball_center_px[1] + left_px[1]) // 2)
            cv2.putText(frame, f"{left_dist:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_LEFT_HAND, 1, cv2.LINE_AA)

        if right_px and right_dist is not None:
            cv2.line(frame, ball_center_px, right_px, _COLOR_RIGHT_HAND, 1, cv2.LINE_AA)
            mid = ((ball_center_px[0] + right_px[0]) // 2,
                   (ball_center_px[1] + right_px[1]) // 2)
            cv2.putText(frame, f"{right_dist:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COLOR_RIGHT_HAND, 1, cv2.LINE_AA)

    # --- State banner ---
    h, w = frame.shape[:2]
    label_sm = f"smooth: {state}"
    label_raw = f"raw: {entry.get('state', '?')}"
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, label_sm, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, label_raw, (w - 240, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # --- Source legend (top-right corner) ---
    lx0 = w - 110
    for row, (tag, color) in enumerate([
        ("■ det",    _COLOR_BALL_DETECTED),
        ("■ interp", _COLOR_BALL_INTERP),
        ("■ pred",   _COLOR_BALL_PRED),
    ]):
        cv2.putText(frame, tag, (lx0, 20 + row * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)

    # --- Frame counter ---
    t = entry.get("time_s", 0)
    fidx = entry.get("frame_idx", "?")
    cv2.putText(frame, f"t={t:.2f}s  f={fidx}", (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
