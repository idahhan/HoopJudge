"""Tests for YOLOBallDetector and the ball detector interface.

Covers:
- BallDetection dataclass serialization contract
- YOLOBallDetector with a mocked ultralytics model (no real weights needed)
- Factory rejects unknown methods
- YOLO detector integration (skipped if ultralytics is not installed)
- Ball analysis pipeline with a stub detector
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from myogait.detectors.ball_detector import (
    BallDetection,
    BallDetector,
    YOLOBallDetector,
    create_ball_detector,
    _SPORTS_BALL_CLASS,
)
from myogait.ball import track_ball_frames

_REAL_VIDEO = Path(__file__).parent.parent / "trimmed-2.mp4"
_REAL_JSON = Path(__file__).parent.parent / "trimmed-2.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mock_ultralytics(
    xyxy=(100.0, 100.0, 200.0, 200.0),
    conf: float = 0.85,
    cls: int = _SPORTS_BALL_CLASS,
    no_detections: bool = False,
):
    """Return a mock 'ultralytics' module whose YOLO class yields a fixed result."""
    xyxy_arr = np.array(list(xyxy), dtype=float)

    if no_detections:
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.names = {}
    else:
        mock_cpu_result = MagicMock()
        mock_cpu_result.numpy.return_value = xyxy_arr

        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_cpu_result

        mock_box = MagicMock()
        mock_box.cls = [cls]
        mock_box.conf = [conf]
        mock_box.xyxy = [mock_tensor]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {_SPORTS_BALL_CLASS: "sports ball"}

    mock_yolo_instance = MagicMock()
    mock_yolo_instance.return_value = [mock_result]
    mock_yolo_instance.to = MagicMock(return_value=mock_yolo_instance)

    mock_mod = MagicMock()
    mock_mod.YOLO = MagicMock(return_value=mock_yolo_instance)
    return mock_mod


class _StubBallDetector(BallDetector):
    """Minimal detector stub that returns a fixed detection without any model."""

    def __init__(self, fixed: Optional[BallDetection] = None):
        self._fixed = fixed or BallDetection(
            detected=True,
            bbox=(100, 100, 200, 200),
            center=(150.0, 150.0),
            radius=50.0,
            confidence=0.85,
        )

    def detect(self, frame_bgr, landmarks=None, prev_detection=None) -> BallDetection:
        return self._fixed


# ---------------------------------------------------------------------------
# BallDetection dataclass
# ---------------------------------------------------------------------------

class TestBallDetectionDataclass:
    def test_empty_detection_defaults(self):
        d = BallDetection()
        assert not d.detected
        assert d.center is None
        assert d.bbox is None
        assert d.confidence == 0.0

    def test_to_dict_required_keys(self):
        det = BallDetection(detected=True, center=(10.0, 20.0), confidence=0.8)
        d = det.to_dict()
        assert {"detected", "bbox", "center", "radius", "confidence"} <= set(d.keys())

    def test_to_dict_with_debug_includes_class_label(self):
        det = BallDetection(
            detected=True, center=(10.0, 20.0), confidence=0.8, class_label="sports ball"
        )
        d = det.to_dict(include_debug=True)
        assert d["class_label"] == "sports ball"

    def test_to_dict_without_debug_omits_class_label(self):
        det = BallDetection(
            detected=True, center=(10.0, 20.0), confidence=0.8, class_label="sports ball"
        )
        d = det.to_dict(include_debug=False)
        assert "class_label" not in d

    def test_confidence_rounded_to_4dp(self):
        det = BallDetection(detected=True, confidence=0.123456789)
        d = det.to_dict()
        assert d["confidence"] == round(0.123456789, 4)

    def test_center_serialised_as_list(self):
        det = BallDetection(detected=True, center=(10.5, 20.5))
        d = det.to_dict()
        assert d["center"] == [10.5, 20.5]

    def test_bbox_serialised_as_list(self):
        det = BallDetection(detected=True, bbox=(10, 20, 110, 120))
        d = det.to_dict()
        assert d["bbox"] == [10, 20, 110, 120]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Only 'yolo' is supported"):
            create_ball_detector("magic")

    def test_color_raises_value_error(self):
        # color detector has been removed
        with pytest.raises(ValueError):
            create_ball_detector("color")

    def test_case_insensitive(self):
        with pytest.raises(ValueError):
            create_ball_detector("COLOR")


# ---------------------------------------------------------------------------
# YOLOBallDetector with mocked ultralytics
# ---------------------------------------------------------------------------

class TestYOLOBallDetectorMock:
    """Tests that do not require a real ultralytics installation or model weights."""

    def test_detect_returns_correct_center(self):
        mock_mod = _build_mock_ultralytics(xyxy=(100.0, 100.0, 200.0, 200.0), conf=0.85)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result.detected
        assert result.center == pytest.approx((150.0, 150.0))  # (100+200)/2

    def test_detect_returns_correct_radius(self):
        mock_mod = _build_mock_ultralytics(xyxy=(100.0, 100.0, 200.0, 250.0))
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        # radius = max(200-100, 250-100) / 2 = 150/2 = 75
        assert result.radius == pytest.approx(75.0)

    def test_detect_returns_int_bbox(self):
        mock_mod = _build_mock_ultralytics(xyxy=(100.7, 99.3, 200.9, 201.1))
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result.bbox is not None
        assert all(isinstance(v, int) for v in result.bbox)

    def test_detect_returns_confidence(self):
        mock_mod = _build_mock_ultralytics(conf=0.77)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result.confidence == pytest.approx(0.77)

    def test_no_detections_returns_empty(self):
        mock_mod = _build_mock_ultralytics(no_detections=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert not result.detected
        assert result.center is None
        assert result.confidence == 0.0

    def test_target_class_filters_wrong_class(self):
        """A detection of class 0 (person) must be ignored when target_class=32."""
        mock_mod = _build_mock_ultralytics(cls=0)  # person, not sports ball
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt", target_class=_SPORTS_BALL_CLASS)
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert not result.detected

    def test_target_class_none_accepts_any_class(self):
        """target_class=None accepts highest-confidence detection regardless of class."""
        mock_mod = _build_mock_ultralytics(cls=0)  # person
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt", target_class=None)
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert result.detected

    def test_to_dict_shape_from_yolo_result(self):
        """Result of detect() serialises to the expected dict shape."""
        mock_mod = _build_mock_ultralytics()
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            det = YOLOBallDetector(model_path="fake.pt")
            result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        d = result.to_dict()
        assert set(d.keys()) >= {"detected", "bbox", "center", "radius", "confidence"}
        assert d["detected"] is True
        assert len(d["center"]) == 2
        assert len(d["bbox"]) == 4

    def test_missing_explicit_path_raises_file_not_found(self):
        """A path with a directory component that does not exist should raise."""
        mock_mod = _build_mock_ultralytics()
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "ultralytics", mock_mod)
            with pytest.raises(FileNotFoundError, match="not found"):
                YOLOBallDetector(model_path="/nonexistent/path/basketball.pt")


# ---------------------------------------------------------------------------
# Stub detector — pipeline integration without real model
# ---------------------------------------------------------------------------

class TestPipelineWithStubDetector:
    """Verify analyze_ball works end-to-end when given a pre-built stub detector."""

    def _make_data(self, n_frames=10, width=640, height=480, fps=30.0):
        frames = [
            {
                "frame_idx": i,
                "time_s": round(i / fps, 4),
                "landmarks": {
                    "LEFT_WRIST":    {"x": 0.3, "y": 0.5, "visibility": 0.95},
                    "LEFT_PINKY":    {"x": 0.32, "y": 0.52, "visibility": 0.90},
                    "LEFT_INDEX":    {"x": 0.31, "y": 0.49, "visibility": 0.92},
                    "LEFT_THUMB":    {"x": 0.29, "y": 0.50, "visibility": 0.88},
                    "RIGHT_WRIST":   {"x": 0.7, "y": 0.5, "visibility": 0.95},
                    "RIGHT_PINKY":   {"x": 0.68, "y": 0.52, "visibility": 0.90},
                    "RIGHT_INDEX":   {"x": 0.69, "y": 0.49, "visibility": 0.92},
                    "RIGHT_THUMB":   {"x": 0.71, "y": 0.50, "visibility": 0.88},
                    "LEFT_SHOULDER": {"x": 0.35, "y": 0.25, "visibility": 0.99},
                    "RIGHT_SHOULDER":{"x": 0.65, "y": 0.25, "visibility": 0.99},
                    "LEFT_HIP":      {"x": 0.35, "y": 0.60, "visibility": 0.99},
                    "RIGHT_HIP":     {"x": 0.65, "y": 0.60, "visibility": 0.99},
                },
            }
            for i in range(n_frames)
        ]
        return {
            "meta": {"width": width, "height": height, "fps": fps},
            "frames": frames,
        }

    def test_analyze_ball_with_stub_produces_expected_keys(self, tmp_path):
        from myogait.ball import analyze_ball, BALL_STATES

        # Write a minimal 1-frame synthetic video
        import cv2
        video_path = str(tmp_path / "test.mp4")
        n_frames = 5
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        for _ in range(n_frames):
            out.write(np.zeros((480, 640, 3), dtype=np.uint8))
        out.release()

        data = self._make_data(n_frames=n_frames)
        stub = _StubBallDetector()
        result = analyze_ball(video_path, data, detector=stub)

        ball = result["ball"]
        assert "per_frame" in ball
        assert "summary" in ball
        assert "method" in ball
        assert len(ball["per_frame"]) == n_frames
        for entry in ball["per_frame"]:
            assert entry["state"] in BALL_STATES
            assert entry["state_smoothed"] in BALL_STATES

    def test_analyze_ball_summary_detection_rate(self, tmp_path):
        from myogait.ball import analyze_ball

        import cv2
        video_path = str(tmp_path / "test2.mp4")
        n_frames = 6
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480)
        )
        for _ in range(n_frames):
            out.write(np.zeros((480, 640, 3), dtype=np.uint8))
        out.release()

        data = self._make_data(n_frames=n_frames)
        # Stub always detects → detection_rate should be 1.0
        stub = _StubBallDetector()
        result = analyze_ball(video_path, data, detector=stub)

        summary = result["ball"]["summary"]
        assert summary["detection_rate"] == pytest.approx(1.0)
        assert summary["n_ball_detected"] == n_frames


# ---------------------------------------------------------------------------
# YOLOBallDetector — real ultralytics (skipped if not installed)
# ---------------------------------------------------------------------------

class TestYOLODetectorWithUltralytics:
    """Integration tests requiring a real ultralytics installation."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ultralytics(self):
        try:
            import ultralytics  # noqa: F401
        except (ImportError, PermissionError, OSError):
            pytest.skip("ultralytics not available or not usable in this environment")

    def test_instantiation(self):
        det = YOLOBallDetector(model_path="yolo11n.pt")
        assert det is not None

    def test_returns_ball_detection_type_on_grey_frame(self):
        det = YOLOBallDetector(model_path="yolo11n.pt")
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = det.detect(frame)
        assert isinstance(result, BallDetection)

    def test_factory_yolo_creates_yolo_detector(self):
        det = create_ball_detector("yolo", model_path="yolo11n.pt")
        assert isinstance(det, YOLOBallDetector)

    @pytest.mark.skipif(not _REAL_VIDEO.is_file(), reason="Real clip not available")
    def test_yolo_on_real_clip_does_not_crash(self):
        import cv2
        det = YOLOBallDetector(model_path="yolo11n.pt", confidence_threshold=0.01)
        cap = cv2.VideoCapture(str(_REAL_VIDEO))
        n_detected = 0
        for _ in range(20):
            ret, frame = cap.read()
            if not ret:
                break
            r = det.detect(frame)
            assert isinstance(r, BallDetection)
            if r.detected:
                n_detected += 1
        cap.release()
        print(f"\n  YOLO detected ball in {n_detected}/20 frames")


# ---------------------------------------------------------------------------
# track_ball_frames
# ---------------------------------------------------------------------------

def _make_raw(detections):
    """Build a raw detections list from a list of (detected, cx, cy, conf) tuples.
    Pass (False, None, None, 0.0) for missing frames.
    """
    out = []
    for i, (det, cx, cy, conf) in enumerate(detections):
        frame = {"frame_idx": i, "detected": det, "tracked": det, "source": "detected" if det else "none",
                 "confidence": conf}
        if det:
            frame["center"] = [float(cx), float(cy)]
            frame["bbox"] = [int(cx) - 10, int(cy) - 10, int(cx) + 10, int(cy) + 10]
            frame["radius"] = 10.0
        else:
            frame["center"] = None
            frame["bbox"] = None
            frame["radius"] = None
        out.append(frame)
    return out


class TestTrackBallFrames:
    def test_passthrough_when_no_gaps(self):
        """All frames detected → track_ball_frames should not change source."""
        raw = _make_raw([(True, 100, 100, 0.9)] * 10)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 8, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        assert len(tracked) == 10
        for f in tracked:
            assert f["source"] == "detected"
            assert f["tracked"] is True

    def test_fills_short_gap(self):
        """A 2-frame gap between two detections should be filled with interpolated frames."""
        data = (
            [(True, 100, 100, 0.8)]
            + [(False, None, None, 0.0)] * 2
            + [(True, 130, 100, 0.8)]
        )
        raw = _make_raw(data)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 8, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        assert tracked[1]["source"] == "interpolated"
        assert tracked[2]["source"] == "interpolated"
        assert tracked[1]["tracked"] is True
        assert tracked[1]["center"] is not None
        # Interpolated center should be between anchors
        cx1 = tracked[1]["center"][0]
        assert 100.0 < cx1 < 130.0

    def test_does_not_fill_gap_exceeding_max_interp_gap(self):
        """A gap longer than max_interp_gap must NOT be filled."""
        data = (
            [(True, 100, 100, 0.8)]
            + [(False, None, None, 0.0)] * 10
            + [(True, 200, 100, 0.8)]
        )
        raw = _make_raw(data)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 5, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        # All gap frames should remain "none"
        for f in tracked[1:11]:
            assert f["source"] == "none"
            assert f["tracked"] is False

    def test_does_not_fill_gap_with_excessive_speed(self):
        """A gap where implied speed > max_ball_speed_px must NOT be filled."""
        # 2-frame gap, 500px jump → 250px/frame > 150 limit
        data = (
            [(True, 0, 100, 0.8)]
            + [(False, None, None, 0.0)] * 2
            + [(True, 500, 100, 0.8)]
        )
        raw = _make_raw(data)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 8, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        for f in tracked[1:3]:
            assert f["source"] == "none"

    def test_output_length_unchanged(self):
        data = [(True, 100, 100, 0.8)] * 3 + [(False, None, None, 0.0)] * 3 + [(True, 120, 100, 0.8)] * 3
        raw = _make_raw(data)
        tracked = track_ball_frames(raw)
        assert len(tracked) == len(raw)

    def test_all_sources_valid(self):
        valid_sources = {"detected", "interpolated", "predicted", "none"}
        data = [(True, 50*i, 100, 0.8) if i % 3 == 0 else (False, None, None, 0.0) for i in range(12)]
        raw = _make_raw(data)
        tracked = track_ball_frames(raw)
        for f in tracked:
            assert f["source"] in valid_sources

    def test_low_conf_anchor_not_used(self):
        """Anchors with conf < interp_min_conf must not trigger interpolation."""
        data = (
            [(True, 100, 100, 0.05)]   # conf below threshold
            + [(False, None, None, 0.0)] * 2
            + [(True, 130, 100, 0.8)]
        )
        raw = _make_raw(data)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 8, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        # Gap frames should remain "none" since left anchor is below threshold
        for f in tracked[1:3]:
            assert f["source"] == "none"

    def test_interpolated_confidence_lower_than_anchors(self):
        """Mid-gap confidence should be reduced relative to anchor confidence."""
        data = (
            [(True, 100, 100, 0.9)]
            + [(False, None, None, 0.0)] * 5
            + [(True, 160, 100, 0.9)]
        )
        raw = _make_raw(data)
        tracked = track_ball_frames(raw, config={"max_interp_gap": 8, "max_predict_frames": 0,
                                                  "interp_min_conf": 0.10, "max_ball_speed_px": 150.0})
        # Middle frame (index 3) should have lower confidence than 0.9
        assert tracked[3]["confidence"] < 0.9
