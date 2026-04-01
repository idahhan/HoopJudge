"""Compare ball detectors (YOLO + up to two Roboflow models) on a clip.

Usage examples
--------------
# Full three-way comparison (YOLO + sod9e + xil7x):
python run_ball_compare.py \\
    --clip trimmed-.mov \\
    --json trimmed-.json \\
    --roboflow-api-key YOUR_KEY \\
    --roboflow2-project basketball-xil7x \\
    --roboflow2-version 1 \\
    --roboflow2-target-classes ball

# Just YOLO vs sod9e (original two-way):
python run_ball_compare.py \\
    --clip trimmed-.mov \\
    --json trimmed-.json \\
    --roboflow-api-key YOUR_KEY

# YOLO baseline only:
python run_ball_compare.py \\
    --clip trimmed-.mov \\
    --json trimmed-.json \\
    --yolo-only

Outputs (written alongside --json)
-------
<stem>_yolo_ball_debug.mp4
<stem>_rf1_ball_debug.mp4          (first Roboflow model)
<stem>_rf2_ball_debug.mp4          (second Roboflow model, if requested)
<stem>_yolo_ball.csv
<stem>_rf1_ball.csv
<stem>_rf2_ball.csv
<stem>_comparison.txt              (side-by-side metric table + FP analysis)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ball_compare")

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _load_pivot(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _run_pipeline(
    video_path: str,
    data: dict,
    config: dict,
    debug_video_path: str,
    csv_path: str,
    label: str,
) -> dict:
    """Run analyze_ball + render + CSV.  Returns augmented summary dict."""
    import copy
    from myogait.ball import analyze_ball, render_ball_video, ball_to_csv

    data_copy = copy.deepcopy(data)

    logger.info("=== Running detector: %s ===", label)
    t0 = time.time()
    data_copy = analyze_ball(video_path, data_copy, config=config)
    elapsed = time.time() - t0

    summary = data_copy["ball"]["summary"]
    summary["_elapsed_s"] = round(elapsed, 1)
    summary["_label"] = label
    summary["_per_frame"] = data_copy["ball"]["per_frame"]   # kept for FP analysis

    logger.info("  Detection rate  : %.1f%%", 100 * summary["yolo_detection_rate"])
    logger.info("  Tracked coverage: %.1f%%", 100 * summary["tracked_coverage_rate"])
    logger.info("  Elapsed         : %.1fs", elapsed)

    logger.info("Rendering debug video → %s", debug_video_path)
    render_ball_video(video_path, data_copy, debug_video_path)

    logger.info("Writing CSV → %s", csv_path)
    ball_to_csv(data_copy, csv_path)

    return summary


def _failure_segments(per_frame: list, min_gap: int = 10) -> list:
    """Return (start_s, end_s, n_frames) for runs of no_ball_detected ≥ min_gap."""
    segments = []
    gap_start: Optional[float] = None
    consec = 0
    last_t = 0.0

    for entry in per_frame:
        state = entry.get("state_smoothed", "no_ball_detected")
        t = entry.get("time_s", 0.0)
        last_t = t
        if state == "no_ball_detected":
            consec += 1
            if gap_start is None:
                gap_start = t
        else:
            if consec >= min_gap and gap_start is not None:
                segments.append((gap_start, t, consec))
            consec = 0
            gap_start = None

    if consec >= min_gap and gap_start is not None:
        segments.append((gap_start, last_t, consec))

    return segments


def _class_label_counts(per_frame: list) -> dict:
    """Count class_label values seen in detected frames (FP / class-mix analysis)."""
    counts: Counter = Counter()
    for entry in per_frame:
        ball = entry.get("ball", {})
        if ball.get("detected"):
            lbl = ball.get("class_label") or "unknown"
            counts[lbl] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _print_comparison(summaries: list[dict], out_path: Optional[str] = None):
    labels = [s["_label"] for s in summaries]
    n_frames = summaries[0]["n_frames"]
    n_cols = len(summaries)
    col_w = max(24, *(len(l) + 2 for l in labels))

    lines: list[str] = []
    sep = "=" * (30 + col_w * n_cols)

    lines += ["", sep, "  BALL DETECTOR COMPARISON", sep,
              f"  Frames analysed: {n_frames}", ""]

    # ---- metric rows -------------------------------------------------------
    hdr = f"{'Metric':<30}" + "".join(f"{l:>{col_w}}" for l in labels)
    lines += [hdr, "-" * len(hdr)]

    def row(name, *vals):
        return f"  {name:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

    lines.append(row("Raw detection rate",
                     *[f"{100*s['yolo_detection_rate']:.1f}%" for s in summaries]))
    lines.append(row("Tracked coverage rate",
                     *[f"{100*s['tracked_coverage_rate']:.1f}%" for s in summaries]))
    lines.append(row("Frames detected",
                     *[str(s["n_ball_detected"]) for s in summaries]))
    lines.append(row("Frames tracked total",
                     *[str(s["n_tracked"]) for s in summaries]))
    lines.append(row("Frames interpolated",
                     *[str(s["n_interpolated"]) for s in summaries]))
    lines.append(row("Frames predicted",
                     *[str(s.get("n_predicted", 0)) for s in summaries]))
    lines.append(row("Pipeline time",
                     *[f"{s['_elapsed_s']:.1f}s" for s in summaries]))
    lines.append("")

    # ---- state distribution ------------------------------------------------
    all_states = []
    seen = set()
    for st in ["left_hand_control", "right_hand_control",
               "both_uncertain", "free", "no_ball_detected"]:
        for s in summaries:
            if st in s.get("state_counts", {}):
                if st not in seen:
                    all_states.append(st)
                    seen.add(st)

    lines.append(f"  {'State (smoothed)':<28}" +
                 "".join(f"{'%':>{col_w}}" for _ in summaries))
    lines.append("  " + "-" * (28 + col_w * n_cols))
    for st in all_states:
        vals = []
        for s in summaries:
            cnt = s.get("state_counts", {}).get(st, 0)
            pct = 100 * cnt / max(n_frames, 1)
            vals.append(f"{pct:.1f}% ({cnt}f)")
        lines.append(row(f"  {st}", *vals))
    lines.append("")

    # ---- class label distribution (FP / wrong-object analysis) ------------
    lines.append("  Class labels seen on detected frames (FP / object-mix check):")
    for s in summaries:
        lbl_counts = s.get("_class_labels", {})
        lines.append(f"    [{s['_label']}]")
        if not lbl_counts:
            lines.append("      (class_label not recorded — rerun with debug_candidates=True)")
        else:
            for cls, cnt in sorted(lbl_counts.items(), key=lambda x: -x[1]):
                pct = 100 * cnt / max(s["n_ball_detected"], 1)
                lines.append(f"      {cls:<20} {cnt:>5} detections  ({pct:.1f}%)")
        lines.append("")

    # ---- failure segments --------------------------------------------------
    lines.append("  Failure segments (≥10 consecutive no_ball_detected):")
    for s in summaries:
        segs = s.get("_failure_segments", [])
        lines.append(f"    [{s['_label']}]")
        if not segs:
            lines.append("      none")
        else:
            for (t0, t1, nf) in segs[:15]:
                lines.append(f"      {t0:.2f}s – {t1:.2f}s  ({nf} frames)")
        lines.append("")

    lines += [sep, ""]
    text = "\n".join(lines)
    print(text)

    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        logger.info("Comparison saved → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ball detectors on a clip (YOLO + up to two Roboflow models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("--clip", required=True,
                        help="Path to source video (.mov/.mp4)")
    parser.add_argument("--json", required=True,
                        help="Path to pivot JSON (from myogait extract)")

    # --- YOLO ---------------------------------------------------------------
    yolo_grp = parser.add_argument_group("YOLO baseline")
    yolo_grp.add_argument("--yolo-model", default="yolo11n.pt",
                          help="YOLO model path or hub name (default: yolo11n.pt)")
    yolo_grp.add_argument("--yolo-conf", type=float, default=0.20,
                          help="YOLO confidence threshold (default: 0.20)")
    yolo_grp.add_argument("--yolo-only", action="store_true",
                          help="Skip all Roboflow models; run YOLO baseline only")
    yolo_grp.add_argument("--target-class", type=int, default=None,
                          help="YOLO class index to filter (default: 32 COCO sports ball)")

    # --- shared Roboflow ----------------------------------------------------
    rf_grp = parser.add_argument_group("Roboflow (shared)")
    rf_grp.add_argument("--roboflow-api-key",
                        default=os.environ.get("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    rf_grp.add_argument("--max-interp-gap", type=int, default=8,
                        help="Max gap frames to interpolate (default: 8)")

    # --- Roboflow model 1 (sod9e, current default) -------------------------
    rf1_grp = parser.add_argument_group("Roboflow model 1 (current default: sod9e)")
    rf1_grp.add_argument("--roboflow-project", default="basketball-sod9e-yhrkx",
                         help="Roboflow project slug for model 1")
    rf1_grp.add_argument("--roboflow-version", type=int, default=2,
                         help="Roboflow version for model 1 (default: 2)")
    rf1_grp.add_argument("--roboflow-conf", type=float, default=0.20,
                         help="Confidence threshold for model 1 (default: 0.20)")
    rf1_grp.add_argument("--roboflow-target-classes", default=None,
                         help="Comma-separated class names to accept for model 1 "
                              "(default: accept all)")

    # --- Roboflow model 2 (xil7x, challenger) ------------------------------
    rf2_grp = parser.add_argument_group("Roboflow model 2 (challenger)")
    rf2_grp.add_argument("--roboflow2-project", default=None,
                         help="Roboflow project slug for model 2 "
                              "(e.g. basketball-xil7x).  Omit to skip.")
    rf2_grp.add_argument("--roboflow2-version", type=int, default=1,
                         help="Roboflow version for model 2 (default: 1)")
    rf2_grp.add_argument("--roboflow2-conf", type=float, default=0.20,
                         help="Confidence threshold for model 2 (default: 0.20)")
    rf2_grp.add_argument("--roboflow2-target-classes", default=None,
                         help="Comma-separated class names to accept for model 2 "
                              "(e.g. 'ball' for xil7x to filter out rim+human)")

    args = parser.parse_args()

    clip = str(Path(args.clip).resolve())
    json_path = str(Path(args.json).resolve())
    out_dir = Path(json_path).parent
    stem = Path(json_path).stem

    if not os.path.isfile(clip):
        parser.error(f"Clip not found: {clip}")
    if not os.path.isfile(json_path):
        parser.error(f"Pivot JSON not found: {json_path}")

    pivot = _load_pivot(json_path)
    if not pivot.get("frames"):
        parser.error("Pivot JSON has no frames. Run myogait extract first.")

    summaries = []

    # ------------------------------------------------------------------ YOLO
    yolo_cfg = {
        "detector": "yolo",
        "detector_kwargs": {
            "model_path": args.yolo_model,
            "confidence_threshold": args.yolo_conf,
            **({"target_class": args.target_class} if args.target_class is not None else {}),
        },
        "max_interp_gap": args.max_interp_gap,
        "debug_candidates": True,
    }
    s = _run_pipeline(
        video_path=clip, data=pivot, config=yolo_cfg,
        debug_video_path=str(out_dir / f"{stem}_yolo_ball_debug.mp4"),
        csv_path=str(out_dir / f"{stem}_yolo_ball.csv"),
        label=f"YOLO ({Path(args.yolo_model).name})",
    )
    s["_failure_segments"] = _failure_segments(s["_per_frame"])
    s["_class_labels"] = _class_label_counts(s["_per_frame"])
    summaries.append(s)

    if args.yolo_only:
        _print_comparison(summaries, out_path=str(out_dir / f"{stem}_comparison.txt"))
        return

    if not args.roboflow_api_key:
        logger.warning(
            "No Roboflow API key (--roboflow-api-key or ROBOFLOW_API_KEY). "
            "Skipping Roboflow models."
        )
        _print_comparison(summaries, out_path=str(out_dir / f"{stem}_comparison.txt"))
        return

    # --------------------------------------------------------- Roboflow model 1
    def _parse_classes(s: Optional[str]) -> Optional[list]:
        if not s:
            return None
        return [c.strip() for c in s.split(",") if c.strip()]

    rf1_classes = _parse_classes(args.roboflow_target_classes)
    rf1_cfg = {
        "detector": "roboflow",
        "detector_kwargs": {
            "api_key": args.roboflow_api_key,
            "project_id": args.roboflow_project,
            "version": args.roboflow_version,
            "confidence_threshold": args.roboflow_conf,
            "target_classes": rf1_classes,
        },
        "max_interp_gap": args.max_interp_gap,
        "debug_candidates": True,
    }
    rf1_label = f"RF {args.roboflow_project}/v{args.roboflow_version}"
    try:
        s = _run_pipeline(
            video_path=clip, data=pivot, config=rf1_cfg,
            debug_video_path=str(out_dir / f"{stem}_rf1_ball_debug.mp4"),
            csv_path=str(out_dir / f"{stem}_rf1_ball.csv"),
            label=rf1_label,
        )
        s["_failure_segments"] = _failure_segments(s["_per_frame"])
        s["_class_labels"] = _class_label_counts(s["_per_frame"])
        summaries.append(s)
    except ImportError as exc:
        logger.error("Roboflow inference SDK not installed: %s", exc)
        logger.error("Run:  pip install inference")
    except Exception as exc:
        logger.error("Roboflow model 1 failed: %s", exc, exc_info=True)

    # --------------------------------------------------------- Roboflow model 2
    if args.roboflow2_project:
        rf2_classes = _parse_classes(args.roboflow2_target_classes)
        rf2_cfg = {
            "detector": "roboflow",
            "detector_kwargs": {
                "api_key": args.roboflow_api_key,
                "project_id": args.roboflow2_project,
                "version": args.roboflow2_version,
                "confidence_threshold": args.roboflow2_conf,
                "target_classes": rf2_classes,
            },
            "max_interp_gap": args.max_interp_gap,
            "debug_candidates": True,
        }
        rf2_label = f"RF {args.roboflow2_project}/v{args.roboflow2_version}"
        try:
            s = _run_pipeline(
                video_path=clip, data=pivot, config=rf2_cfg,
                debug_video_path=str(out_dir / f"{stem}_rf2_ball_debug.mp4"),
                csv_path=str(out_dir / f"{stem}_rf2_ball.csv"),
                label=rf2_label,
            )
            s["_failure_segments"] = _failure_segments(s["_per_frame"])
            s["_class_labels"] = _class_label_counts(s["_per_frame"])
            summaries.append(s)
        except ImportError as exc:
            logger.error("Roboflow inference SDK not installed: %s", exc)
        except Exception as exc:
            logger.error("Roboflow model 2 failed: %s", exc, exc_info=True)

    # ----------------------------------------------------------------- output
    comparison_path = str(out_dir / f"{stem}_comparison.txt")
    _print_comparison(summaries, out_path=comparison_path)

    print("\nOutput files written:")
    tags = ["yolo"] + [f"rf{i+1}" for i in range(len(summaries) - 1)]
    for tag, s in zip(tags, summaries):
        print(f"  {out_dir / f'{stem}_{tag}_ball_debug.mp4'}")
        print(f"  {out_dir / f'{stem}_{tag}_ball.csv'}")
    print(f"  {comparison_path}")
    print()


if __name__ == "__main__":
    main()
