"""Person detection utilities.

Provides a YOLO-pose-based detector that returns all visible persons in a
frame along with their bounding boxes and 17 COCO keypoints (used for
wrist-proximity scoring during ball-carrier assignment).

Detectors
---------
YOLOPersonDetector
    Uses a YOLO pose model (default: yolov8n-pose.pt) to detect all persons
    and return their bboxes + COCO-17 keypoints.  Keypoints are stored in
    *normalised* [0, 1] coordinates.

    Requires ``pip install ultralytics``.

Factory
-------
create_person_detector(method, **kwargs)
    Supported methods:
    - ``"yolo_pose"`` or ``"yolo"`` → :class:`YOLOPersonDetector`
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices (upper body used for ball-carrier scoring)
_COCO_LEFT_ELBOW  = 7
_COCO_RIGHT_ELBOW = 8
_COCO_LEFT_WRIST  = 9
_COCO_RIGHT_WRIST = 10

# Minimum keypoint confidence to trust a position
_MIN_KP_CONF = 0.30


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PersonDetection:
    """A single detected person in one video frame.

    Attributes
    ----------
    bbox : (x1, y1, x2, y2) in pixels, top-left / bottom-right.
    confidence : person detection confidence [0, 1].
    keypoints_norm : (17, 3) float array, columns = (x_norm, y_norm, conf)
        in COCO-17 order, *normalised* to [0, 1].  ``None`` when a
        detection-only model is used.
    area : bbox area in pixels².  Set automatically from bbox.
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    keypoints_norm: Optional[np.ndarray] = field(default=None, repr=False)
    area: int = field(init=False)
    track_id: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.area = max(0, (x2 - x1) * (y2 - y1))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def center(self) -> Tuple[float, float]:
        """Pixel-space bbox centre."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def wrists_norm(self) -> Optional[List[Optional[Tuple[float, float]]]]:
        """Return ``[(lx, ly), (rx, ry)]`` in *normalised* [0, 1] space.

        Each element is ``None`` if the wrist keypoint confidence is below
        ``_MIN_KP_CONF``.  Returns ``None`` when no keypoints are available.
        """
        if self.keypoints_norm is None:
            return None
        result: List[Optional[Tuple[float, float]]] = []
        for idx in (_COCO_LEFT_WRIST, _COCO_RIGHT_WRIST):
            kp = self.keypoints_norm[idx]
            if kp[2] >= _MIN_KP_CONF:
                result.append((float(kp[0]), float(kp[1])))
            else:
                result.append(None)
        return result

    @property
    def elbows_norm(self) -> Optional[List[Optional[Tuple[float, float]]]]:
        """Return ``[(lx, ly), (rx, ry)]`` for elbows in normalised [0,1] space.

        Each element is ``None`` if keypoint confidence is below ``_MIN_KP_CONF``.
        Returns ``None`` when no keypoints are available.
        """
        if self.keypoints_norm is None:
            return None
        result: List[Optional[Tuple[float, float]]] = []
        for idx in (_COCO_LEFT_ELBOW, _COCO_RIGHT_ELBOW):
            kp = self.keypoints_norm[idx]
            if kp[2] >= _MIN_KP_CONF:
                result.append((float(kp[0]), float(kp[1])))
            else:
                result.append(None)
        return result

    def to_dict(self) -> dict:
        """Serialisable dict for storing in per-frame JSON."""
        d: dict = {
            "bbox":       list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "area":       int(self.area),
            "track_id":   self.track_id,
        }
        if self.keypoints_norm is not None:
            elbows = self.elbows_norm
            wrists = self.wrists_norm
            d["left_elbow_norm"]  = list(elbows[0]) if elbows and elbows[0] is not None else None
            d["right_elbow_norm"] = list(elbows[1]) if elbows and elbows[1] is not None else None
            d["left_wrist_norm"]  = list(wrists[0]) if wrists and wrists[0] is not None else None
            d["right_wrist_norm"] = list(wrists[1]) if wrists and wrists[1] is not None else None
        return d


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PersonDetector(ABC):
    """Abstract base class for person detectors."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        """Return all detected persons in *frame_bgr*, sorted by confidence."""

    def detect_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[PersonDetection]]:
        """Detect persons in a list of frames (default: sequential)."""
        return [self.detect(f) for f in frames]


# ---------------------------------------------------------------------------
# YOLO pose implementation
# ---------------------------------------------------------------------------

class YOLOPersonDetector(PersonDetector):
    """Detect all persons with a YOLO pose model.

    Uses a YOLO pose checkpoint (default: ``yolov8n-pose.pt``, already
    downloaded as part of the myogait pose-extraction step) to return every
    person in the frame with their bbox and 17 COCO keypoints.  Wrist
    keypoints (COCO indices 9 and 10) are used downstream for ball-to-hand
    proximity scoring.

    Requires ``pip install ultralytics``.

    Parameters
    ----------
    model_path : str
        Path or hub name for a YOLO pose model.
        Default: ``"yolov8n-pose.pt"`` (auto-downloaded if absent).
    confidence_threshold : float
        Minimum detection confidence to keep a person. Default: 0.30.
    device : str
        Inference device (``"cpu"``, ``"cuda"``, etc.). Default: ``"cpu"``.
    iou_threshold : float
        NMS IoU threshold. Default: 0.45.
    imgsz : int
        Inference image size (longest side). Default: 640.
    min_area_frac : float
        Minimum person bbox area as a fraction of the total frame area.
        Filters out tiny / partial silhouettes near clip edges.
        Default: 0.002 (0.2 % of frame).
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.30,
        device: str = "cpu",
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        min_area_frac: float = 0.002,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.min_area_frac = min_area_frac
        self._model = self._load_model()

    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "YOLOPersonDetector requires ultralytics:\n"
                "  pip install ultralytics"
            ) from exc

        path = str(Path(self.model_path).expanduser())
        logger.info(
            "Loading YOLO person detector: %s on %s", path, self.device
        )
        model = YOLO(path)
        model.to(self.device)
        return model

    def detect(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        h, w = frame_bgr.shape[:2]
        min_area = h * w * self.min_area_frac

        results = self._model(
            frame_bgr,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[0],        # COCO class 0 = person
        )

        persons: List[PersonDetection] = []

        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints  # None for detection-only models

            n = len(boxes) if boxes is not None else 0
            for i in range(n):
                x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy().astype(int)
                conf = float(boxes[i].conf[0].cpu().numpy())

                area = max(0, int((x2 - x1) * (y2 - y1)))
                if area < min_area:
                    continue

                # Normalise keypoints from pixel coords to [0, 1]
                kp_norm: Optional[np.ndarray] = None
                if keypoints is not None and i < len(keypoints.data):
                    kp_px = keypoints.data[i].cpu().numpy()   # (17, 3): x_px, y_px, conf
                    if kp_px.shape == (17, 3):
                        kp_norm = kp_px.copy()
                        kp_norm[:, 0] /= max(w, 1)
                        kp_norm[:, 1] /= max(h, 1)

                persons.append(
                    PersonDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        keypoints_norm=kp_norm,
                    )
                )

        persons.sort(key=lambda p: p.confidence, reverse=True)
        return persons


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_person_detector(
    method: str = "yolo_pose",
    **kwargs,
) -> PersonDetector:
    """Return a PersonDetector for the given method.

    Parameters
    ----------
    method : str
        ``"yolo_pose"`` or ``"yolo"`` → :class:`YOLOPersonDetector`.
    **kwargs
        Forwarded to the detector's ``__init__``.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    m = method.lower()
    if m in ("yolo_pose", "yolo"):
        return YOLOPersonDetector(**kwargs)
    raise ValueError(
        f"Unknown person detector: {method!r}. Supported: 'yolo_pose'."
    )
