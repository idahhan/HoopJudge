"""Ball detection utilities.

Provides a YOLO11-based detector for locating a basketball in a video frame.

Detector
--------
YOLOBallDetector
    ultralytics YOLO11 (or YOLOv8/v9) detector.

    When using a COCO-trained model, it targets class 32 (``sports ball``).
    When using a custom basketball checkpoint, set ``target_class=0`` (first
    class) or ``target_class=None`` to accept the highest-confidence detection
    regardless of class.

    Requires ``pip install ultralytics``.

Factory
-------
create_ball_detector(method="yolo", **kwargs)
    Only ``"yolo"`` is supported.  Pass keyword arguments to
    ``YOLOBallDetector.__init__``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# COCO class index for "sports ball"
_SPORTS_BALL_CLASS = 32


@dataclass
class BallDetection:
    """Result of ball detection for a single frame.

    All coordinates are in **pixel space** of the source frame.

    Attributes
    ----------
    detected : bool
    bbox : (x1, y1, x2, y2) pixels, or None
    center : (cx, cy) pixels, or None
    radius : estimated radius pixels (half the larger bbox dimension), or None
    confidence : YOLO confidence score ∈ [0, 1]
    class_label : YOLO class name (e.g. ``"sports ball"``), or None
    """

    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    confidence: float = 0.0
    class_label: Optional[str] = None

    def to_dict(self, include_debug: bool = False) -> dict:
        d: Dict[str, Any] = {
            "detected": self.detected,
            "bbox": list(self.bbox) if self.bbox else None,
            "center": list(self.center) if self.center else None,
            "radius": self.radius,
            "confidence": round(self.confidence, 4),
        }
        if include_debug:
            d["class_label"] = self.class_label
        return d


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BallDetector(ABC):
    """Abstract base class for ball detectors."""

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[dict] = None,
        prev_detection: Optional[BallDetection] = None,
    ) -> BallDetection:
        """Detect ball in a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image (H, W, 3).
        landmarks : dict, optional
            MediaPipe landmark dict ``{name: {x, y, visibility}}`` in
            normalised [0, 1] coordinates.  Provided for interface
            compatibility; YOLO does not use it internally.
        prev_detection : BallDetection, optional
            Detection from the immediately preceding frame.  Provided for
            interface compatibility; YOLO does not use it internally.

        Returns
        -------
        BallDetection
        """

    def detect_batch(
        self,
        frames: List[np.ndarray],
        landmarks_list: Optional[List[Optional[dict]]] = None,
    ) -> List[BallDetection]:
        """Detect ball in a sequence of frames with temporal continuity."""
        prev: Optional[BallDetection] = None
        results: List[BallDetection] = []
        lms_list = landmarks_list or [None] * len(frames)
        for frame, lm in zip(frames, lms_list):
            det = self.detect(frame, landmarks=lm, prev_detection=prev)
            results.append(det)
            if det.detected:
                prev = det
        return results


# ---------------------------------------------------------------------------
# YOLO11 detector
# ---------------------------------------------------------------------------


class YOLOBallDetector(BallDetector):
    """Ball detector backed by a YOLO model (ultralytics YOLO11/YOLOv8/v9).

    By default uses the COCO ``sports ball`` class (class 32).  When using a
    custom basketball checkpoint, set ``target_class=0`` (first class) or
    ``target_class=None`` to accept the highest-confidence detection regardless
    of class.

    Parameters
    ----------
    model_path : str
        Path to model weights file or a hub model name.

        - Simple name (no path separators), e.g. ``"yolo11n.pt"``: ultralytics
          downloads it on first use from the ultralytics hub.
        - Absolute or relative path, e.g. ``"/models/basketball.pt"``: the file
          must exist; a clear ``FileNotFoundError`` is raised otherwise.

    confidence_threshold : float
        Minimum YOLO confidence to accept a detection.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    iou_threshold : float
        Non-maximum suppression IoU threshold.
    imgsz : int
        Inference image size (long edge in pixels).
    target_class : int or None
        YOLO class index to filter on.

        - ``32`` (default) → COCO ``sports ball``
        - ``0`` → first class in a custom single-class basketball model
        - ``None`` → accept highest-confidence detection regardless of class
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.20,
        device: str = "cpu",
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        target_class: Optional[int] = _SPORTS_BALL_CLASS,
    ):
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "YOLOBallDetector requires ultralytics: pip install ultralytics"
            ) from exc

        # Validate explicit paths (those with directory separators) before loading.
        # Simple names like "yolo11n.pt" are handled by ultralytics (auto-download).
        p = Path(model_path)
        if p.parent != Path(".") and not p.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {model_path}\n"
                "Provide a valid path to a .pt file, or a model name that "
                "ultralytics can download (e.g. 'yolo11n.pt')."
            )

        logger.info("Loading YOLO model: %s on %s", model_path, device)
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.target_class = target_class

    def detect(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[dict] = None,
        prev_detection: Optional[BallDetection] = None,
    ) -> BallDetection:
        results = self.model(
            frame_bgr,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
        )

        best_conf = 0.0
        best_box = None
        best_cls_name: Optional[str] = None

        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                if self.target_class is not None and cls_idx != self.target_class:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()
                    names = getattr(r, "names", {})
                    best_cls_name = names.get(cls_idx)

        if best_box is None:
            return BallDetection()

        x1, y1, x2, y2 = best_box
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        radius = float(max(x2 - x1, y2 - y1) / 2.0)
        return BallDetection(
            detected=True,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=(cx, cy),
            radius=radius,
            confidence=best_conf,
            class_label=best_cls_name,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ball_detector(method: str = "yolo", **kwargs) -> BallDetector:
    """Return a BallDetector for the given method name.

    Parameters
    ----------
    method : str
        Only ``"yolo"`` is supported.
    **kwargs
        Forwarded to ``YOLOBallDetector.__init__``.

    Returns
    -------
    BallDetector

    Raises
    ------
    ValueError
        If *method* is not ``"yolo"``.
    """
    method = method.lower()
    if method == "yolo":
        return YOLOBallDetector(**kwargs)
    raise ValueError(
        f"Unknown ball detector method: {method!r}. Only 'yolo' is supported."
    )
