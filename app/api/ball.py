"""Ball possession API routes.

Endpoints
---------
POST   /ball/analyze/{clip_id}       Run (or re-run) ball detection pipeline
GET    /ball/results/{clip_id}        Fetch stored results
GET    /ball/status/{clip_id}         Check processing status + file existence
POST   /ball/debug-video/{clip_id}   (Re)render debug overlay video
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.models.ball import (
    BallAnalyzeRequest,
    BallAnalysisResult,
    BallConfig,
    BallDetectionInfo,
    BallSummary,
    ClipStatus,
    ErrorResponse,
    FrameBallResult,
)
from app.services.ball_service import (
    get_ball_results,
    get_clip_status,
    render_debug_video_for_clip,
    run_ball_analysis,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ball", tags=["ball"])


def _build_analysis_response(clip_id: str, result: dict) -> BallAnalysisResult:
    """Convert raw service result dict to a validated response model."""
    cfg = result.get("config", {})
    summary = result.get("summary", {})
    per_frame_raw = result.get("per_frame", [])

    per_frame = []
    for entry in per_frame_raw:
        ball_raw = entry.get("ball", {})
        per_frame.append(
            FrameBallResult(
                frame_idx=entry["frame_idx"],
                time_s=entry["time_s"],
                ball=BallDetectionInfo(**{
                    k: v for k, v in ball_raw.items()
                    if k in BallDetectionInfo.model_fields
                }),
                ball_norm=entry.get("ball_norm"),
                left_hand=entry.get("left_hand"),
                right_hand=entry.get("right_hand"),
                left_dist=entry.get("left_dist"),
                right_dist=entry.get("right_dist"),
                body_scale=entry.get("body_scale"),
                state=entry.get("state", "no_ball_detected"),
                state_smoothed=entry.get("state_smoothed", "no_ball_detected"),
            )
        )

    return BallAnalysisResult(
        clip_id=clip_id,
        method=result.get("method", "yolo"),
        config=BallConfig(
            control_threshold=cfg.get("control_threshold", 0.40),
            min_visibility=cfg.get("min_visibility", 0.25),
            smoothing_window=cfg.get("smoothing_window", 7),
        ),
        summary=BallSummary(
            n_frames=summary.get("n_frames", 0),
            n_ball_detected=summary.get("n_ball_detected", 0),
            detection_rate=summary.get("detection_rate", 0.0),
            state_counts=summary.get("state_counts", {}),
            state_durations_s=summary.get("state_durations_s", {}),
        ),
        per_frame=per_frame,
        output_files=result.get("output_files", {}),
    )


@router.post(
    "/analyze/{clip_id}",
    response_model=BallAnalysisResult,
    summary="Run ball possession analysis for a clip",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def analyze_clip(
    clip_id: str,
    body: BallAnalyzeRequest = BallAnalyzeRequest(),
    force: bool = Query(default=False, description="Re-run even if results exist"),
) -> BallAnalysisResult:
    """Detect the basketball and classify possession state for every frame.

    - **clip_id**: stem of the JSON pivot file (e.g. ``trimmed-2`` for ``trimmed-2.json``)
    - **force**: set to ``true`` to reprocess even if cached results exist
    """
    detector_kwargs: dict = {}
    if body.yolo_model_path is not None:
        detector_kwargs["model_path"] = body.yolo_model_path
    if body.yolo_conf_threshold is not None:
        detector_kwargs["confidence_threshold"] = body.yolo_conf_threshold

    config: dict = {
        "control_threshold": body.control_threshold,
        "min_visibility": body.min_visibility,
        "smoothing_window": body.smoothing_window,
    }
    if detector_kwargs:
        config["detector_kwargs"] = detector_kwargs
    try:
        result = run_ball_analysis(
            clip_id,
            config=config,
            render_debug_video=body.render_debug_video,
            export_csv=body.export_csv,
            force=force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ball analysis failed for clip_id=%s", clip_id)
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    return _build_analysis_response(clip_id, result)


@router.get(
    "/results/{clip_id}",
    response_model=BallAnalysisResult,
    summary="Fetch stored ball results",
    responses={404: {"model": ErrorResponse}},
)
async def fetch_results(clip_id: str) -> BallAnalysisResult:
    """Return previously computed ball analysis results.

    Returns 404 if analysis has not been run for this clip yet.
    """
    result = get_ball_results(clip_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ball results found for clip '{clip_id}'. Run POST /ball/analyze/{clip_id} first.",
        )
    return _build_analysis_response(clip_id, result)


@router.get(
    "/status/{clip_id}",
    response_model=ClipStatus,
    summary="Check clip processing status",
)
async def clip_status(clip_id: str) -> ClipStatus:
    """Return processing status and file existence flags for a clip."""
    status = get_clip_status(clip_id)
    return ClipStatus(
        clip_id=status["clip_id"],
        json_path=status.get("json_path") or "",
        ball_result_path=status.get("ball_result_path"),
        has_ball_data=status.get("has_ball_data", False),
        has_debug_video=status.get("has_debug_video", False),
        has_csv=status.get("has_csv", False),
    )


@router.post(
    "/debug-video/{clip_id}",
    summary="Render (or re-render) the debug overlay video",
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def render_video(clip_id: str) -> Dict[str, str]:
    """Render a debug overlay video showing ball detection and state labels.

    Returns the output file path.  The video is saved alongside the pivot JSON.
    Requires that ball analysis has already been run for this clip.
    """
    try:
        video_path = render_debug_video_for_clip(clip_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Debug video rendering failed for clip_id=%s", clip_id)
        raise HTTPException(status_code=500, detail=f"Render error: {exc}")

    return {"clip_id": clip_id, "video_path": video_path}
