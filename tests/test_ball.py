"""Tests for ball possession analysis.

Covers:
- Hand centroid computation
- Body scale computation
- Per-frame state classification
- Temporal smoothing
- Full pipeline integration (real clip, skipped if unavailable)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from myogait.detectors.ball_detector import BallDetection, BallDetector, create_ball_detector
from myogait.ball import (
    _body_scale,
    _hand_centroid,
    analyze_ball,
    classify_ball_state,
    smooth_ball_states,
    render_ball_video,
    ball_to_csv,
    BALL_STATES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_VIDEO = Path(__file__).parent.parent / "trimmed-2.mp4"
_REAL_JSON = Path(__file__).parent.parent / "trimmed-2.json"

try:
    import ultralytics  # noqa: F401
    _HAS_ULTRALYTICS = True
except (ImportError, PermissionError, OSError):
    _HAS_ULTRALYTICS = False


def _fake_landmarks(
    left_wrist=(0.3, 0.5),
    right_wrist=(0.7, 0.5),
) -> dict:
    """Minimal landmark dict for testing."""
    lx, ly = left_wrist
    rx, ry = right_wrist
    return {
        "LEFT_WRIST":   {"x": lx, "y": ly, "visibility": 0.95},
        "LEFT_PINKY":   {"x": lx + 0.02, "y": ly + 0.02, "visibility": 0.90},
        "LEFT_INDEX":   {"x": lx + 0.01, "y": ly - 0.01, "visibility": 0.92},
        "LEFT_THUMB":   {"x": lx - 0.01, "y": ly, "visibility": 0.88},
        "RIGHT_WRIST":  {"x": rx, "y": ry, "visibility": 0.95},
        "RIGHT_PINKY":  {"x": rx - 0.02, "y": ry + 0.02, "visibility": 0.90},
        "RIGHT_INDEX":  {"x": rx - 0.01, "y": ry - 0.01, "visibility": 0.92},
        "RIGHT_THUMB":  {"x": rx + 0.01, "y": ry, "visibility": 0.88},
        "LEFT_SHOULDER":  {"x": 0.35, "y": 0.25, "visibility": 0.99},
        "RIGHT_SHOULDER": {"x": 0.65, "y": 0.25, "visibility": 0.99},
        "LEFT_HIP":  {"x": 0.35, "y": 0.60, "visibility": 0.99},
        "RIGHT_HIP": {"x": 0.65, "y": 0.60, "visibility": 0.99},
    }


def _fake_ball_detection(cx_px=200, cy_px=240, frame_w=640, frame_h=480) -> dict:
    r = 30
    return {
        "detected": True,
        "tracked": True,
        "source": "detected",
        "bbox": [cx_px - r, cy_px - r, cx_px + r, cy_px + r],
        "center": [float(cx_px), float(cy_px)],
        "radius": float(r),
        "confidence": 0.75,
        "_frame_w": frame_w,
        "_frame_h": frame_h,
    }


class _AlwaysDetectStub(BallDetector):
    """Stub detector that always returns a detection at a fixed pixel position."""

    def __init__(self, cx_px=192, cy_px=240):
        self._cx = cx_px
        self._cy = cy_px

    def detect(self, frame_bgr, landmarks=None, prev_detection=None) -> BallDetection:
        return BallDetection(
            detected=True,
            bbox=(self._cx - 30, self._cy - 30, self._cx + 30, self._cy + 30),
            center=(float(self._cx), float(self._cy)),
            radius=30.0,
            confidence=0.85,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Only 'yolo' is supported"):
            create_ball_detector("unicorn")

    def test_color_raises(self):
        with pytest.raises(ValueError):
            create_ball_detector("color")

    @pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="ultralytics not installed")
    def test_yolo_creates_detector(self):
        from myogait.detectors.ball_detector import YOLOBallDetector
        det = create_ball_detector("yolo")
        assert isinstance(det, YOLOBallDetector)

    def test_ball_detection_to_dict_contract(self):
        d = BallDetection(detected=True, center=(10.0, 20.0), confidence=0.5).to_dict()
        assert set(d.keys()) >= {"detected", "bbox", "center", "radius", "confidence"}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestBodyScale:
    def test_returns_positive(self):
        lm = _fake_landmarks()
        scale = _body_scale(lm)
        assert scale > 0.0

    def test_fallback_on_empty(self):
        scale = _body_scale({})
        assert scale == 0.4

    def test_proportional_to_skeleton_size(self):
        tall_lm = {
            "LEFT_SHOULDER":  {"x": 0.5, "y": 0.1, "visibility": 1.0},
            "RIGHT_SHOULDER": {"x": 0.5, "y": 0.1, "visibility": 1.0},
            "LEFT_HIP":  {"x": 0.5, "y": 0.9, "visibility": 1.0},
            "RIGHT_HIP": {"x": 0.5, "y": 0.9, "visibility": 1.0},
        }
        short_lm = {
            "LEFT_SHOULDER":  {"x": 0.5, "y": 0.4, "visibility": 1.0},
            "RIGHT_SHOULDER": {"x": 0.5, "y": 0.4, "visibility": 1.0},
            "LEFT_HIP":  {"x": 0.5, "y": 0.6, "visibility": 1.0},
            "RIGHT_HIP": {"x": 0.5, "y": 0.6, "visibility": 1.0},
        }
        assert _body_scale(tall_lm) > _body_scale(short_lm)


class TestHandCentroid:
    def test_returns_tuple_for_visible_landmarks(self):
        lm = _fake_landmarks()
        left = _hand_centroid("left", lm, min_visibility=0.0)
        assert left is not None
        assert len(left) == 2

    def test_returns_none_below_visibility(self):
        lm = {
            "LEFT_WRIST": {"x": 0.3, "y": 0.5, "visibility": 0.1},
            "LEFT_PINKY": {"x": 0.3, "y": 0.5, "visibility": 0.1},
            "LEFT_INDEX": {"x": 0.3, "y": 0.5, "visibility": 0.1},
            "LEFT_THUMB": {"x": 0.3, "y": 0.5, "visibility": 0.1},
        }
        result = _hand_centroid("left", lm, min_visibility=0.5)
        assert result is None

    def test_centroid_weighted_toward_high_vis(self):
        lm = {
            "LEFT_WRIST": {"x": 0.0, "y": 0.0, "visibility": 1.0},
            "LEFT_PINKY": {"x": 1.0, "y": 1.0, "visibility": 0.01},
            "LEFT_INDEX": {"x": 0.0, "y": 0.0, "visibility": 0.99},
            "LEFT_THUMB": {"x": 0.0, "y": 0.0, "visibility": 0.99},
        }
        cx, cy = _hand_centroid("left", lm, min_visibility=0.0)
        assert cx < 0.1
        assert cy < 0.1


# ---------------------------------------------------------------------------
# classify_ball_state
# ---------------------------------------------------------------------------

class TestClassifyBallState:
    def test_no_ball(self):
        ball = {"detected": False}
        lm = _fake_landmarks()
        result = classify_ball_state(ball, lm)
        assert result["state"] == "no_ball_detected"

    def test_left_hand_control(self):
        ball = _fake_ball_detection(cx_px=192, cy_px=240, frame_w=640, frame_h=480)
        lm = _fake_landmarks(left_wrist=(0.3, 0.5), right_wrist=(0.7, 0.5))
        result = classify_ball_state(ball, lm, config={"control_threshold": 0.5, "min_visibility": 0.0})
        assert result["state"] == "left_hand_control"
        assert result["left_dist"] is not None
        assert result["left_dist"] < 0.1

    def test_right_hand_control(self):
        ball = _fake_ball_detection(cx_px=448, cy_px=240, frame_w=640, frame_h=480)
        lm = _fake_landmarks(left_wrist=(0.3, 0.5), right_wrist=(0.7, 0.5))
        result = classify_ball_state(ball, lm, config={"control_threshold": 0.5, "min_visibility": 0.0})
        assert result["state"] == "right_hand_control"

    def test_free_ball(self):
        ball = _fake_ball_detection(cx_px=320, cy_px=480, frame_w=640, frame_h=480)
        lm = _fake_landmarks(left_wrist=(0.3, 0.2), right_wrist=(0.7, 0.2))
        result = classify_ball_state(ball, lm, config={"control_threshold": 0.2, "min_visibility": 0.0})
        assert result["state"] == "free"

    def test_both_uncertain(self):
        lm = _fake_landmarks(left_wrist=(0.49, 0.5), right_wrist=(0.51, 0.5))
        ball = _fake_ball_detection(
            cx_px=int(0.50 * 640), cy_px=int(0.5 * 480), frame_w=640, frame_h=480
        )
        result = classify_ball_state(ball, lm, config={"control_threshold": 0.5, "min_visibility": 0.0})
        assert result["state"] in ("both_uncertain", "left_hand_control", "right_hand_control")

    def test_body_scale_and_ball_norm_present(self):
        ball = _fake_ball_detection()
        lm = _fake_landmarks()
        result = classify_ball_state(ball, lm, config={"min_visibility": 0.0})
        assert result["body_scale"] is not None and result["body_scale"] > 0
        assert result["ball_norm"] is not None and len(result["ball_norm"]) == 2


# ---------------------------------------------------------------------------
# smooth_ball_states
# ---------------------------------------------------------------------------

class TestSmoothBallStates:
    def _make_per_frame(self, states):
        return [{"state": s} for s in states]

    def test_no_change_on_consistent_states(self):
        states = ["left_hand_control"] * 10
        per_frame = self._make_per_frame(states)
        smoothed = smooth_ball_states(per_frame, config={"smoothing_window": 5})
        assert all(s == "left_hand_control" for s in smoothed)

    def test_smooths_single_frame_flip(self):
        states = ["left_hand_control"] * 4 + ["right_hand_control"] + ["left_hand_control"] * 4
        per_frame = self._make_per_frame(states)
        smoothed = smooth_ball_states(per_frame, config={"smoothing_window": 7})
        assert smoothed[4] == "left_hand_control"

    def test_preserves_no_ball_frames(self):
        states = ["no_ball_detected"] * 5
        per_frame = self._make_per_frame(states)
        smoothed = smooth_ball_states(per_frame, config={"smoothing_window": 3})
        assert all(s == "no_ball_detected" for s in smoothed)

    def test_output_length_matches_input(self):
        states = ["free", "left_hand_control", "no_ball_detected", "right_hand_control"]
        per_frame = self._make_per_frame(states)
        smoothed = smooth_ball_states(per_frame, config={"smoothing_window": 3})
        assert len(smoothed) == len(states)

    def test_all_states_valid(self):
        states = ["free", "left_hand_control", "right_hand_control",
                  "both_uncertain", "no_ball_detected"] * 3
        per_frame = self._make_per_frame(states)
        smoothed = smooth_ball_states(per_frame, config={"smoothing_window": 3})
        for s in smoothed:
            assert s in BALL_STATES, f"Unexpected state: {s}"


# ---------------------------------------------------------------------------
# Integration test on real clip (skipped if video or ultralytics not available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _REAL_VIDEO.is_file() or not _REAL_JSON.is_file() or not _HAS_ULTRALYTICS,
    reason="Real clip or ultralytics not available",
)
class TestRealClipIntegration:
    def test_analyze_ball_runs(self, tmp_path):
        import json as json_mod
        with open(_REAL_JSON) as f:
            data = json_mod.load(f)
        result = analyze_ball(str(_REAL_VIDEO), data, config={"smoothing_window": 5})
        ball = result["ball"]
        assert "per_frame" in ball
        assert "summary" in ball
        assert len(ball["per_frame"]) == len(data["frames"])
        for entry in ball["per_frame"]:
            assert entry["state"] in BALL_STATES
            assert entry["state_smoothed"] in BALL_STATES

    def test_ball_to_csv(self, tmp_path):
        import json as json_mod
        with open(_REAL_JSON) as f:
            data = json_mod.load(f)
        data = analyze_ball(str(_REAL_VIDEO), data)
        csv_path = tmp_path / "ball.csv"
        ball_to_csv(data, str(csv_path))
        assert csv_path.is_file()
        content = csv_path.read_text()
        assert "frame_idx" in content
        assert "state_smoothed" in content

    def test_render_ball_video(self, tmp_path):
        import json as json_mod
        with open(_REAL_JSON) as f:
            data = json_mod.load(f)
        data = analyze_ball(str(_REAL_VIDEO), data)
        out = tmp_path / "ball_debug.mp4"
        render_ball_video(str(_REAL_VIDEO), data, str(out))
        assert out.is_file()
        assert out.stat().st_size > 1000

    def test_summary_fields(self, tmp_path):
        import json as json_mod
        with open(_REAL_JSON) as f:
            data = json_mod.load(f)
        data = analyze_ball(str(_REAL_VIDEO), data)
        summary = data["ball"]["summary"]
        assert "n_frames" in summary
        assert "detection_rate" in summary
        assert 0.0 <= summary["detection_rate"] <= 1.0
        assert "state_counts" in summary
        assert isinstance(summary["state_counts"], dict)
