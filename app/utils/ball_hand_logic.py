"""Hand proximity and state classification helpers (app-layer wrappers).

These functions wrap the core myogait.ball logic and provide
thin, testable wrappers used by the service layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from myogait.ball import (  # type: ignore[import]
    classify_ball_state,
    smooth_ball_states,
    _body_scale,
    _hand_centroid,
    BALL_STATES,
)


def get_hand_positions(
    landmarks: dict,
    min_visibility: float = 0.25,
) -> Dict[str, Optional[List[float]]]:
    """Return left and right hand centroids from pose landmarks.

    Parameters
    ----------
    landmarks : dict
        MediaPipe landmark dict (name → {x, y, visibility}).
    min_visibility : float
        Minimum visibility threshold.

    Returns
    -------
    dict with ``"left"`` and ``"right"`` keys; values are ``[x, y]`` or ``None``.
    """
    left = _hand_centroid("left", landmarks, min_visibility)
    right = _hand_centroid("right", landmarks, min_visibility)
    return {
        "left": list(left) if left else None,
        "right": list(right) if right else None,
    }


def compute_body_scale_for_frame(landmarks: dict) -> float:
    """Return the body-scale estimate for a set of landmarks."""
    return _body_scale(landmarks)


def classify_frame(
    ball: dict,
    landmarks: dict,
    control_threshold: float = 0.40,
    min_visibility: float = 0.25,
) -> dict:
    """Classify ball state for a single frame.

    Thin wrapper around ``myogait.ball.classify_ball_state``.
    """
    config = {
        "control_threshold": control_threshold,
        "min_visibility": min_visibility,
    }
    return classify_ball_state(ball, landmarks, config=config)


def apply_smoothing(
    per_frame: List[dict],
    window: int = 7,
) -> List[str]:
    """Apply temporal smoothing and return smoothed state labels."""
    return smooth_ball_states(per_frame, config={"smoothing_window": window})


def state_summary(per_frame: List[dict]) -> Dict[str, Any]:
    """Count raw and smoothed state occurrences across frames."""
    from collections import Counter

    raw = Counter(e.get("state", "no_ball_detected") for e in per_frame)
    smoothed = Counter(e.get("state_smoothed", "no_ball_detected") for e in per_frame)
    return {"raw": dict(raw), "smoothed": dict(smoothed)}
