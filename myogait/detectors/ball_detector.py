"""Ball detection utilities.

Single detector: RoboflowBallDetector backed by the basketball-xil7x model
(cricket-qnb5l workspace, v1 — 97.85 % mAP on basketball/rim/human classes,
filtered to class "ball").

On first run the ONNX weights are downloaded from Roboflow and cached under
~/.inference/.  All subsequent frames run **local inference** — no network
traffic after the initial download.

Requires:
    pip install inference

Factory
-------
create_ball_detector(**kwargs)
    Returns a :class:`RoboflowBallDetector` configured for xil7x.
    All kwargs are forwarded to ``__init__``.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROBOFLOW_DEFAULT_PROJECT   = "basketball-xil7x"
_ROBOFLOW_DEFAULT_WORKSPACE = "cricket-qnb5l"
_ROBOFLOW_DEFAULT_VERSION   = 1
_ROBOFLOW_DEFAULT_CLASSES   = ["ball"]   # xil7x has 3 classes: ball / rim / human


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


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
    confidence : model confidence score ∈ [0, 1]
    class_label : model class name (e.g. ``"ball"``), or None
    """

    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    confidence: float = 0.0
    class_label: Optional[str] = None

    def to_dict(self, include_debug: bool = False) -> dict:
        d: Dict[str, Any] = {
            "detected":   self.detected,
            "bbox":       list(self.bbox)   if self.bbox   else None,
            "center":     list(self.center) if self.center else None,
            "radius":     self.radius,
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
        """Detect ball in a single BGR frame."""

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
# Roboflow inference-SDK detector
# ---------------------------------------------------------------------------


class RoboflowBallDetector(BallDetector):
    """Ball detector backed by a Roboflow basketball-specific model.

    Uses the ``inference`` SDK from Roboflow.  On the first call the ONNX
    model weights are downloaded from Roboflow and cached locally under
    ``~/.inference/``.  All subsequent frames run **local inference** with no
    further network I/O.

    Default model
    -------------
    project  : basketball-xil7x  (cricket-qnb5l workspace)
    version  : 1
    classes  : ["ball"]  (the model also detects "rim" and "human"; we filter
                          to "ball" only)

    Parameters
    ----------
    api_key : str, optional
        Roboflow API key.  Falls back to the ``ROBOFLOW_API_KEY`` environment
        variable.  Required only for the initial weight download.
    project_id : str
        Roboflow project slug.
    version : int
        Model version number.
    confidence_threshold : float
        Minimum confidence to accept a detection.
    target_classes : list of str, optional
        Accepted class names.  Defaults to ``["ball"]`` for xil7x.

    Setup
    -----
    pip install inference
    export ROBOFLOW_API_KEY=<your-key>
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: str = _ROBOFLOW_DEFAULT_PROJECT,
        version: int = _ROBOFLOW_DEFAULT_VERSION,
        confidence_threshold: float = 0.20,
        target_classes: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "RoboflowBallDetector requires a Roboflow API key.\n"
                "Pass api_key=... or set the ROBOFLOW_API_KEY environment variable.\n"
                "Get a free key at https://app.roboflow.com"
            )

        self.project_id           = project_id
        self.version              = version
        self.confidence_threshold = confidence_threshold
        # Default: filter to "ball" class for xil7x
        self.target_classes = (
            target_classes if target_classes is not None
            else list(_ROBOFLOW_DEFAULT_CLASSES)
        )

        self._model = self._load_model()

    def _load_model(self):
        try:
            from inference import get_model  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "RoboflowBallDetector requires the inference SDK:\n"
                "  pip install inference\n"
                "See https://inference.roboflow.com for details."
            ) from exc

        model_id = f"{self.project_id}/{self.version}"
        logger.info(
            "Loading Roboflow ball model %s (api_key=***%s)",
            model_id, self.api_key[-4:],
        )
        model = get_model(model_id=model_id, api_key=self.api_key)
        logger.info("Roboflow ball model ready: %s", model_id)
        return model

    def detect(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[dict] = None,
        prev_detection: Optional[BallDetection] = None,
    ) -> BallDetection:
        results = self._model.infer(
            frame_bgr,
            confidence=self.confidence_threshold,
        )

        best_conf = 0.0
        best_pred = None

        for r in results:
            preds = getattr(r, "predictions", [])
            for pred in preds:
                cls  = getattr(pred, "class_name", "") or ""
                conf = float(getattr(pred, "confidence", 0.0))
                if cls not in self.target_classes:
                    continue
                if conf < self.confidence_threshold:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_pred = pred

        if best_pred is None:
            return BallDetection()

        cx = float(best_pred.x)
        cy = float(best_pred.y)
        w  = float(best_pred.width)
        h  = float(best_pred.height)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        return BallDetection(
            detected    = True,
            bbox        = (x1, y1, x2, y2),
            center      = (cx, cy),
            radius      = max(w, h) / 2.0,
            confidence  = best_conf,
            class_label = getattr(best_pred, "class_name", None),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ball_detector(method: str = "roboflow", **kwargs) -> BallDetector:
    """Return a :class:`RoboflowBallDetector`.

    Parameters
    ----------
    method : str
        Only ``"roboflow"`` is supported.  Passing any other value raises
        ``ValueError``.
    **kwargs
        Forwarded to :class:`RoboflowBallDetector.__init__`.

    Returns
    -------
    RoboflowBallDetector

    Raises
    ------
    ValueError
        If *method* is not ``"roboflow"``.
    """
    if method.lower() != "roboflow":
        raise ValueError(
            f"Unknown ball detector method: {method!r}.\n"
            "Only 'roboflow' is supported (xil7x model by default).\n"
            "Example: create_ball_detector('roboflow', api_key='...')"
        )
    return RoboflowBallDetector(**kwargs)
