"""Ball possession analysis service.

Orchestrates clip loading, ball detection, state classification,
temporal smoothing, result persistence, and optional video/CSV export.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DEFAULT_BALL_CONFIG
from app.utils.ball_detection import (
    ball_csv_path,
    ball_result_path,
    ball_video_path,
    load_ball_result,
    resolve_clip_json,
    resolve_video_path,
    save_ball_result,
)

logger = logging.getLogger(__name__)


def _load_pivot(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def run_ball_analysis(
    clip_id: str,
    config: Optional[Dict[str, Any]] = None,
    render_debug_video: bool = False,
    export_csv: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the full ball possession pipeline for *clip_id*.

    Parameters
    ----------
    clip_id : str
        Identifier for the clip (stem of the JSON pivot file).
    config : dict, optional
        Ball analysis configuration overrides.
    render_debug_video : bool
        If True, render and save a debug overlay video.
    export_csv : bool
        If True, export per-frame results to CSV.
    force : bool
        If True, re-run even if results already exist.

    Returns
    -------
    dict
        The ``data["ball"]`` section plus ``output_files`` and ``clip_id``.
    """
    # Lazy imports (keeps startup fast)
    from myogait.ball import analyze_ball, render_ball_video, ball_to_csv  # type: ignore[import]
    from myogait.schema import load_json  # type: ignore[import]

    # Resolve paths
    json_path = resolve_clip_json(clip_id)
    video_path = resolve_video_path(clip_id, json_path)

    # Return cached result unless forced
    if not force:
        cached = load_ball_result(clip_id)
        if cached is not None:
            logger.info("Returning cached ball result for %s", clip_id)
            output_files = _collect_output_files(clip_id)
            return {"clip_id": clip_id, **cached, "output_files": output_files}

    # Build merged config, deep-merging detector_kwargs so that env defaults
    # (e.g. YOLO_MODEL_PATH) are preserved when the caller only overrides one key.
    incoming = config or {}
    merged_dkw = {
        **DEFAULT_BALL_CONFIG.get("detector_kwargs", {}),
        **incoming.get("detector_kwargs", {}),
    }
    merged_config = {**DEFAULT_BALL_CONFIG, **incoming, "detector_kwargs": merged_dkw}

    # Load pivot data
    logger.info("Loading pivot JSON from %s", json_path)
    data = _load_pivot(json_path)

    if not data.get("frames"):
        raise ValueError(
            f"Clip '{clip_id}' has no frames. Run extract() first."
        )

    # Run analysis
    data = analyze_ball(video_path, data, config=merged_config)

    # Persist result
    result_path = save_ball_result(clip_id, data)
    output_files: Dict[str, Optional[str]] = {"ball_json": result_path}

    # Optional: debug video
    if render_debug_video:
        vid_path = str(ball_video_path(clip_id))
        try:
            render_ball_video(video_path, data, vid_path)
            output_files["debug_video"] = vid_path
        except Exception as exc:
            logger.warning("Debug video rendering failed: %s", exc)
            output_files["debug_video"] = None

    # Optional: CSV
    if export_csv:
        csv_path = str(ball_csv_path(clip_id))
        try:
            ball_to_csv(data, csv_path)
            output_files["csv"] = csv_path
        except Exception as exc:
            logger.warning("CSV export failed: %s", exc)
            output_files["csv"] = None

    return {"clip_id": clip_id, **data["ball"], "output_files": output_files}


def get_ball_results(clip_id: str) -> Optional[Dict[str, Any]]:
    """Fetch previously computed ball results for *clip_id*.

    Returns ``None`` if no results exist yet.
    """
    cached = load_ball_result(clip_id)
    if cached is None:
        return None
    output_files = _collect_output_files(clip_id)
    return {"clip_id": clip_id, **cached, "output_files": output_files}


def get_clip_status(clip_id: str) -> Dict[str, Any]:
    """Return a status dict for *clip_id* (paths + existence flags)."""
    try:
        json_path = resolve_clip_json(clip_id)
    except FileNotFoundError:
        return {
            "clip_id": clip_id,
            "json_path": None,
            "has_ball_data": False,
            "has_debug_video": False,
            "has_csv": False,
            "error": "Clip JSON not found",
        }

    return {
        "clip_id": clip_id,
        "json_path": json_path,
        "ball_result_path": str(ball_result_path(clip_id)),
        "has_ball_data": ball_result_path(clip_id).is_file(),
        "has_debug_video": ball_video_path(clip_id).is_file(),
        "has_csv": ball_csv_path(clip_id).is_file(),
    }


def render_debug_video_for_clip(clip_id: str) -> str:
    """Render (or re-render) the debug video for *clip_id*.

    Returns the path to the output video.

    Raises
    ------
    ValueError
        If no ball results exist yet for this clip.
    FileNotFoundError
        If the pivot JSON or video cannot be found.
    """
    from myogait.ball import render_ball_video  # type: ignore[import]

    cached = load_ball_result(clip_id)
    if cached is None:
        raise ValueError(
            f"No ball results found for clip '{clip_id}'. Run analyze first."
        )

    json_path = resolve_clip_json(clip_id)
    video_path = resolve_video_path(clip_id, json_path)

    # Rebuild minimal data dict with ball section for render_ball_video
    data = _load_pivot(json_path)
    data["ball"] = cached

    vid_path = str(ball_video_path(clip_id))
    render_ball_video(video_path, data, vid_path)
    return vid_path


def _collect_output_files(clip_id: str) -> Dict[str, Optional[str]]:
    """Return a dict of output file paths, with None where files don't exist."""
    rp = ball_result_path(clip_id)
    vp = ball_video_path(clip_id)
    cp = ball_csv_path(clip_id)
    return {
        "ball_json": str(rp) if rp.is_file() else None,
        "debug_video": str(vp) if vp.is_file() else None,
        "csv": str(cp) if cp.is_file() else None,
    }
