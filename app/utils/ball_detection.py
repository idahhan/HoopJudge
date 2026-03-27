"""Ball detection helpers for the API layer.

Bridges the app layer and the myogait.ball core module.
Handles clip lookup, file resolution, and result persistence.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

# File-naming conventions
_BALL_RESULT_SUFFIX = "_ball.json"
_BALL_VIDEO_SUFFIX = "_ball_debug.mp4"
_BALL_CSV_SUFFIX = "_ball.csv"


def resolve_clip_json(clip_id: str) -> str:
    """Return the absolute path to the pivot JSON for *clip_id*.

    Resolution order:
    1. Exact match if *clip_id* is an absolute path ending in ``.json``.
    2. ``DATA_DIR / clip_id.json``
    3. ``DATA_DIR / clip_id / clip_id.json``

    Raises
    ------
    FileNotFoundError
        If the JSON file cannot be found.
    """
    # Absolute path given directly
    if os.path.isabs(clip_id) and clip_id.endswith(".json"):
        if os.path.isfile(clip_id):
            return clip_id
        raise FileNotFoundError(f"JSON file not found: {clip_id}")

    # Stem resolution
    stem = clip_id.removesuffix(".json")
    candidates = [
        DATA_DIR / f"{stem}.json",
        DATA_DIR / stem / f"{stem}.json",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find pivot JSON for clip_id={clip_id!r}. Tried: {tried}"
    )


def resolve_video_path(clip_id: str, json_path: str) -> str:
    """Resolve the source video path for a clip.

    Tries the path stored in the JSON meta first, then looks for a video
    file with the same stem alongside the JSON.

    Raises
    ------
    FileNotFoundError
        If no video file can be located.
    """
    with open(json_path) as f:
        meta_video = json.load(f).get("meta", {}).get("video_path", "")

    if meta_video and os.path.isfile(meta_video):
        return meta_video

    # Look for video next to the JSON
    json_dir = Path(json_path).parent
    stem = Path(json_path).stem.removesuffix("_ball")
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"):
        candidate = json_dir / f"{stem}{ext}"
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not locate video for clip_id={clip_id!r}. "
        f"Checked meta path '{meta_video}' and siblings of '{json_path}'."
    )


def ball_result_path(clip_id: str) -> Path:
    """Return the path where ball results are stored for *clip_id*."""
    stem = Path(clip_id).stem.removesuffix(".json")
    return OUTPUT_DIR / f"{stem}{_BALL_RESULT_SUFFIX}"


def ball_video_path(clip_id: str) -> Path:
    """Return the path where the ball debug video is stored."""
    stem = Path(clip_id).stem.removesuffix(".json")
    return OUTPUT_DIR / f"{stem}{_BALL_VIDEO_SUFFIX}"


def ball_csv_path(clip_id: str) -> Path:
    """Return the path where the ball CSV export is stored."""
    stem = Path(clip_id).stem.removesuffix(".json")
    return OUTPUT_DIR / f"{stem}{_BALL_CSV_SUFFIX}"


def load_ball_result(clip_id: str) -> Optional[dict]:
    """Load previously computed ball results, or ``None`` if not found."""
    path = ball_result_path(clip_id)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def save_ball_result(clip_id: str, data: dict) -> str:
    """Persist the ``data["ball"]`` section to a JSON file.

    Returns the path written.
    """
    path = ball_result_path(clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ball_section = data.get("ball", {})
    with open(path, "w") as f:
        json.dump(ball_section, f, indent=2, default=_json_safe)
    logger.info("Ball result saved to %s", path)
    return str(path)


def _json_safe(obj):
    """Fallback JSON serializer for numpy types."""
    import numpy as np  # noqa: PLC0415
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
