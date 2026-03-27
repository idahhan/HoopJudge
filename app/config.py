"""Application configuration."""

import os
from pathlib import Path

# Directory where clip JSON pivot files live (can be overridden via env)
DATA_DIR: Path = Path(os.getenv("BALL_DATA_DIR", str(Path(__file__).parent.parent / "myogait")))

# Directory where output files (ball JSON, CSV, debug videos) are written
OUTPUT_DIR: Path = Path(os.getenv("BALL_OUTPUT_DIR", str(DATA_DIR)))

# YOLO model path — use a simple name to let ultralytics auto-download,
# or set an absolute path to a custom basketball checkpoint.
YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")

# Minimum YOLO confidence to accept a ball detection
YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.20"))

# Default ball analysis config
DEFAULT_BALL_CONFIG = {
    "detector": "yolo",
    "detector_kwargs": {
        "model_path": YOLO_MODEL_PATH,
        "confidence_threshold": YOLO_CONF_THRESHOLD,
    },
    "control_threshold": 0.40,
    "min_visibility": 0.05,
    "smoothing_window": 7,
}
