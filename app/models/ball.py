"""Pydantic models for ball detection API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class BallAnalyzeRequest(BaseModel):
    """Request body for POST /ball/analyze/{clip_id}."""

    yolo_model_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to YOLO model weights, or a simple name for ultralytics "
            "auto-download (e.g. 'yolo11n.pt'). Defaults to YOLO_MODEL_PATH env var."
        ),
    )
    yolo_conf_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum YOLO confidence to accept a detection. Defaults to YOLO_CONF_THRESHOLD env var.",
    )
    control_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=2.0,
        description=(
            "Normalised-distance threshold (fraction of body scale) "
            "below which the ball is considered in-hand."
        ),
    )
    min_visibility: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum MediaPipe landmark visibility to use a hand point.",
    )
    smoothing_window: int = Field(
        default=7,
        ge=1,
        le=60,
        description="Temporal smoothing window in frames.",
    )
    render_debug_video: bool = Field(
        default=False,
        description="If true, also write a debug overlay video.",
    )
    export_csv: bool = Field(
        default=False,
        description="If true, also export results to CSV.",
    )


# ---------------------------------------------------------------------------
# Response components
# ---------------------------------------------------------------------------

class BallDetectionInfo(BaseModel):
    detected: bool
    tracked: bool = False
    source: str = "none"
    bbox: Optional[List[int]] = None
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    confidence: float = 0.0


class FrameBallResult(BaseModel):
    frame_idx: int
    time_s: float
    ball: BallDetectionInfo
    ball_norm: Optional[List[float]] = None
    left_hand: Optional[List[float]] = None
    right_hand: Optional[List[float]] = None
    left_dist: Optional[float] = None
    right_dist: Optional[float] = None
    body_scale: Optional[float] = None
    state: str
    state_smoothed: str


class BallSummary(BaseModel):
    n_frames: int
    n_ball_detected: int
    detection_rate: float
    yolo_detection_rate: Optional[float] = None
    n_tracked: Optional[int] = None
    n_interpolated: Optional[int] = None
    n_predicted: Optional[int] = None
    tracked_coverage_rate: Optional[float] = None
    state_counts: Dict[str, int]
    state_durations_s: Dict[str, float]


class BallConfig(BaseModel):
    control_threshold: float
    min_visibility: float
    smoothing_window: int


class BallAnalysisResult(BaseModel):
    clip_id: str
    method: str
    config: BallConfig
    summary: BallSummary
    per_frame: List[FrameBallResult]
    output_files: Dict[str, Optional[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Status / error responses
# ---------------------------------------------------------------------------

class ClipStatus(BaseModel):
    clip_id: str
    json_path: str
    ball_result_path: Optional[str] = None
    has_ball_data: bool = False
    has_debug_video: bool = False
    has_csv: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
