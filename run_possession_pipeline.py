"""Run the full ball possession pipeline on a video.

Steps:
  1. extract  — pose landmarks via MediaPipe
  2. select_target_player — pick the player most likely handling the ball
  3. analyze_ball — Roboflow xil7x ball detection + tracking
  4. track_possession — trajectory-based possession tracker
  5. render_possession_video — debug overlay video

Usage:
    cd /Users/idahhan/myogait/myogait
    python run_possession_pipeline.py yywHHDu8as4.mp4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Redirect ultralytics config to a writable temp dir to avoid sandbox issues
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics_cfg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Make sure the package is importable when running from the myogait/ subdir
sys.path.insert(0, str(Path(__file__).parent))

import math
from collections import deque

import cv2
import numpy as np

from myogait import (
    extract,
    select_target_player,
    analyze_ball,
    track_possession,
    track_handler_identity,
    save_json,
)
from myogait.video import render_skeleton_frame
from myogait.possession import (
    _iou, _conf_color,
    _CLR_ALL, _CLR_POSSESSOR, _CLR_BALL, _CLR_BALL_INTERP,
    _CLR_TRAIL_BALL, _CLR_TRAIL_WRIST, _CLR_HUD, _CLR_LABEL_BG,
    _DEFAULT_CONFIG as _POSS_DEFAULT_CONFIG,
)


_CLR_RELABEL = (0, 220, 255)   # cyan — same player, new detector token
_CLR_SWITCH  = (0,  80, 255)   # orange-red — genuine possession switch


def _render_combined(video_path: str, data: dict, output_path: str) -> str:
    """Render skeleton + gait events + ball + possession overlays on one video."""
    poss    = data.get("possession", {})
    pf_map  = {r["frame_idx"]: r for r in poss.get("per_frame", [])}
    ball_map = {
        f["frame_idx"]: f.get("ball", {})
        for f in data.get("ball", {}).get("per_frame", [])
    }
    # Handler identity (re-id fallback) lookup
    hi_map: dict = {
        r["frame_idx"]: r
        for r in data.get("handler_identity", {}).get("per_frame", [])
    }
    # Player-selection per-frame index (for ByteTrack track IDs)
    ps_index: dict = {
        r["frame_idx"]: r
        for r in data.get("player_selection", {}).get("per_frame", [])
    }
    frames_data  = data.get("frames", [])
    angles_data  = data.get("angles", {})
    angle_frames = angles_data.get("frames", []) if angles_data else []
    was_flipped  = data.get("extraction", {}).get("was_flipped", False)
    ps_frames    = data.get("player_selection", {}).get("per_frame", [])
    trail_len    = poss.get("config", {}).get("trail_length",
                   _POSS_DEFAULT_CONFIG["trail_length"])

    # Build gait event lookup: frame_idx → {type, side}
    event_lookup: dict = {}
    events_dict = data.get("events", {})
    for key in ["left_hs", "right_hs", "left_to", "right_to"]:
        side    = "left"  if key.startswith("left")  else "right"
        ev_type = "HS"    if key.endswith("_hs")      else "TO"
        for ev in events_dict.get(key, []):
            fidx = ev.get("frame")
            if fidx is not None:
                event_lookup[fidx] = {"type": ev_type, "side": side}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    ball_trail:  deque = deque(maxlen=trail_len)
    wrist_trail: deque = deque(maxlen=trail_len)

    _CLR_LEFT  = (200, 80,  40)   # blue-ish for left side events
    _CLR_RIGHT = (40,  80, 200)   # red-ish for right side events

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. Possession + ball + handler identity data ---
        pr   = pf_map.get(frame_idx, {})
        ball = ball_map.get(frame_idx, {})
        hi   = hi_map.get(frame_idx, {})

        bc           = ball.get("center")
        ball_tracked = ball.get("tracked", False)
        ball_src     = ball.get("source", "detected")
        poss_bbox    = pr.get("possessor_bbox")
        poss_pid     = pr.get("possessor_player_id")   # possession tracker ID
        hi_pid       = hi.get("persistent_handler_id") # re-id fallback
        hi_src       = hi.get("handler_id_source", "")
        hi_reid      = hi.get("reid_score")

        # ByteTrack track ID for the selected player (primary identity)
        ps_entry     = ps_index.get(frame_idx, {})
        bt_track_id  = ps_entry.get("target_track_id")  # stable ByteTrack ID

        # --- 2. Skeleton overlay (only when skeleton is on the possessor) ---
        if frame_idx < len(frames_data):
            fd  = frames_data[frame_idx]
            lm  = fd.get("landmarks", {})
            if was_flipped and lm:
                lm = {n: {**v, "x": 1.0 - v["x"]} if v.get("x") is not None else v
                      for n, v in lm.items()}
            frame_angles = angle_frames[frame_idx] if frame_idx < len(angle_frames) else None
            frame_events = event_lookup.get(frame_idx)
            # Only draw skeleton when it belongs to the possessor;
            # skip if temporal smoothing locked the skeleton onto a different player.
            sel_bbox = None
            if frame_idx < len(ps_frames):
                psfr = ps_frames[frame_idx]
                sel_bbox = psfr.get("smoothed_bbox") or psfr.get("target_bbox")
            skeleton_on_possessor = (
                sel_bbox is None or poss_bbox is None or
                _iou(tuple(sel_bbox), tuple(poss_bbox)) > 0.3
            )
            if lm and skeleton_on_possessor:
                frame = render_skeleton_frame(frame, lm,
                                              angles=frame_angles,
                                              events=frame_events)
        confidence   = pr.get("confidence", 0.0)
        reason       = pr.get("reason", "")
        all_scores   = pr.get("scores", [])

        # Trails
        ball_trail.append(tuple(bc) if bc and ball_tracked else None)

        poss_wrist_norm = None
        if poss_pid is not None:
            ps_entry = next(
                (p for p in data.get("player_selection", {}).get("per_frame", [])
                 if p.get("frame_idx") == frame_idx), {}
            )
            for p in ps_entry.get("players", []):
                pb = p.get("bbox")
                if pb and poss_bbox and _iou(tuple(pb), tuple(poss_bbox)) > 0.5:
                    lw = p.get("left_wrist_norm")
                    rw = p.get("right_wrist_norm")
                    if lw and rw and bc:
                        dl = math.sqrt((lw[0]*W - bc[0])**2 + (lw[1]*H - bc[1])**2)
                        dr = math.sqrt((rw[0]*W - bc[0])**2 + (rw[1]*H - bc[1])**2)
                        poss_wrist_norm = lw if dl < dr else rw
                    else:
                        poss_wrist_norm = lw or rw
                    break

        wrist_trail.append(
            (int(poss_wrist_norm[0]*W), int(poss_wrist_norm[1]*H))
            if poss_wrist_norm else None
        )

        # Ball trail
        for ti, tp in enumerate(ball_trail):
            if tp is None:
                continue
            alpha = (ti + 1) / len(ball_trail)
            r = max(3, int(6 * alpha))
            col = tuple(int(c * alpha) for c in _CLR_TRAIL_BALL)
            cv2.circle(frame, (int(tp[0]), int(tp[1])), r, col, -1)

        # Wrist trail
        for ti, tp in enumerate(wrist_trail):
            if tp is None:
                continue
            alpha = (ti + 1) / len(wrist_trail)
            r = max(2, int(5 * alpha))
            col = tuple(int(c * alpha) for c in _CLR_TRAIL_WRIST)
            cv2.circle(frame, tp, r, col, -1)

        # Non-possessor players — draw their ByteTrack track IDs too
        for p in all_scores:
            sb = p.get("bbox")
            if sb is None:
                continue
            if poss_bbox and _iou(tuple(sb), tuple(poss_bbox)) > 0.5:
                continue   # skip possessor, handled below
            cv2.rectangle(frame, (sb[0], sb[1]), (sb[2], sb[3]), _CLR_ALL, 1)
            tid = p.get("track_id")
            if tid is not None:
                cv2.putText(frame, f"T{tid}",
                            (sb[0], max(sb[1] - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _CLR_ALL, 1, cv2.LINE_AA)

        # Possessor bbox — show ByteTrack ID (T#) as primary label
        if poss_bbox:
            col = _conf_color(confidence)
            cv2.rectangle(frame,
                          (poss_bbox[0], poss_bbox[1]),
                          (poss_bbox[2], poss_bbox[3]),
                          col, 3)
            if bt_track_id is not None:
                label_pid = f"T{bt_track_id}"   # ByteTrack is primary
            elif hi_pid is not None:
                label_pid = f"H{hi_pid}/P{poss_pid}"  # re-id fallback
            else:
                label_pid = f"P{poss_pid}"
            cv2.putText(frame, label_pid,
                        (poss_bbox[0], max(poss_bbox[1] - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # Ball circle
        if bc and ball_tracked:
            ball_col = _CLR_BALL_INTERP if ball_src == "interpolated" else _CLR_BALL
            br = max(int(ball.get("radius", 15)), 8)
            cv2.circle(frame, (int(bc[0]), int(bc[1])), br, ball_col, 2)
            cv2.circle(frame, (int(bc[0]), int(bc[1])), 3, ball_col, -1)

        # Score table
        if all_scores:
            y0 = H - 14 - 16 * len(all_scores)
            for si, s in enumerate(sorted(all_scores, key=lambda x: -x["total"])):
                line = (f"P{s['player_id']} "
                        f"tot={s['total']:.2f} "
                        f"prx={s['proximity']:.2f} "
                        f"trj={s['trajectory']:.2f} "
                        f"stb={s['stability']:.2f}")
                y = y0 + si * 16
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(frame, (6, y - th - 2), (6 + tw + 4, y + 3), _CLR_LABEL_BG, -1)
                cv2.putText(frame, line, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _CLR_HUD, 1, cv2.LINE_AA)

        # HUD — ByteTrack ID (T#) is primary; re-id fallback (H#) shown if no BT
        reason_short = reason[:60] + ("…" if len(reason) > 60 else "")
        if bt_track_id is not None:
            hud_id = f"T{bt_track_id}"
        elif hi_pid is not None:
            hud_id = f"H{hi_pid}/P{poss_pid}"
        else:
            hud_id = f"P{poss_pid}"
        for li, line in enumerate([
            f"f{frame_idx:04d}  {hud_id}  conf={confidence:.2f}",
            reason_short,
        ]):
            cv2.putText(frame, line, (10, 22 + li * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _CLR_HUD, 1, cv2.LINE_AA)

        # Re-id event flash (right side): RELABEL or SWITCH
        if hi_src in ("relabel", "switch"):
            flash_col  = _CLR_RELABEL if hi_src == "relabel" else _CLR_SWITCH
            flash_text = f"{'RELABEL' if hi_src == 'relabel' else 'SWITCH'}"
            if hi_reid is not None:
                flash_text += f" {hi_reid:.2f}"
            (fw, fh), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            fx = W - fw - 10
            fy = 42
            cv2.rectangle(frame, (fx - 4, fy - fh - 4), (fx + fw + 4, fy + 4),
                          _CLR_LABEL_BG, -1)
            cv2.putText(frame, flash_text, (fx, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, flash_col, 2, cv2.LINE_AA)

        # Gait event flash bar (top strip when HS/TO fires this frame)
        if frame_idx in event_lookup:
            ev = event_lookup[frame_idx]
            bar_col = _CLR_LEFT if ev["side"] == "left" else _CLR_RIGHT
            cv2.rectangle(frame, (0, 0), (W, 6), bar_col, -1)
            label = f"{ev['type']} {ev['side'].upper()}"
            cv2.putText(frame, label, (W - 160, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_col, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Combined debug video written → %s (%d frames)", output_path, frame_idx)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Ball possession pipeline")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--pivot-json", default=None,
                        help="Skip extraction and load an existing pivot JSON")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Load pivot JSON derived from video name (auto)")
    parser.add_argument("--model", default="yolo",
                        help="Pose extractor model (default: yolo)")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Roboflow detection confidence threshold (default 0.20)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for outputs (default: same as video)")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    pivot_path = Path(args.pivot_json) if args.pivot_json else out_dir / f"{stem}.json"

    # ------------------------------------------------------------------
    # Step 1: Extract (or load existing)
    # ------------------------------------------------------------------
    if args.skip_extract and pivot_path.exists():
        logger.info("Loading existing pivot JSON: %s", pivot_path)
        with open(pivot_path) as f:
            data = json.load(f)
    elif pivot_path.exists() and not args.pivot_json:
        logger.info("Found existing pivot JSON, skipping extraction: %s", pivot_path)
        with open(pivot_path) as f:
            data = json.load(f)
    else:
        logger.info("Step 1/5 — Extracting pose landmarks from %s", video_path.name)
        data = extract(str(video_path), model=args.model)
        save_json(data, str(pivot_path))
        logger.info("Pivot JSON saved → %s", pivot_path)

    # ------------------------------------------------------------------
    # Step 2: Ball analysis (Roboflow xil7x) — runs BEFORE player selection
    #         so select_target_player can use ball proximity to pick the right player
    # ------------------------------------------------------------------
    logger.info("Step 2/5 — Running ball detection (xil7x)")
    ball_config = {
        "detector": "roboflow",
        "detector_kwargs": {
            "api_key": args.api_key,
            "project_id": "basketball-xil7x",
            "version": 1,
            "target_classes": ["ball"],
            "confidence_threshold": args.conf,
        },
        "control_threshold": 0.40,
        "smoothing_window": 7,
        "max_interp_gap": 8,
    }
    data = analyze_ball(str(video_path), data, config=ball_config)

    ball_summary = data.get("ball", {}).get("summary", {})
    logger.info(
        "Ball detection: coverage=%.1f%%",
        100 * ball_summary.get("detection_rate", 0),
    )

    # ------------------------------------------------------------------
    # Step 3: Player selection — runs AFTER ball detection so it can use
    #         ball proximity to select the correct player
    # ------------------------------------------------------------------
    logger.info("Step 3/5 — Selecting target player (ball-guided)")
    data = select_target_player(str(video_path), data)
    # player_selection re-runs YOLO on the original (unflipped) video and
    # overwrites data["frames"] landmarks in unflipped coordinate space.
    # Clear was_flipped so the renderer doesn't double-flip the skeleton.
    if data.get("extraction", {}).get("was_flipped"):
        data["extraction"]["was_flipped"] = False
        logger.info("Cleared was_flipped flag after player_selection landmark rewrite")

    # ------------------------------------------------------------------
    # Step 4: Possession tracking
    # ------------------------------------------------------------------
    logger.info("Step 4/5 — Tracking possession")
    data = track_possession(data)

    poss_summary = data.get("possession", {}).get("summary", {})
    logger.info(
        "Possession: coverage=%.1f%%, switches=%d",
        100 * poss_summary.get("possession_coverage_rate", 0),
        poss_summary.get("n_possession_switches", 0),
    )

    # ------------------------------------------------------------------
    # Step 5: Persistent handler identity (re-id layer)
    # ------------------------------------------------------------------
    logger.info("Step 5/6 — Running handler identity tracker (re-id)")
    data = track_handler_identity(data, str(video_path))

    hi_summary = data.get("handler_identity", {}).get("summary", {})
    logger.info(
        "Handler identity: %d relabels, %d switches (out of %d frames)",
        hi_summary.get("n_relabels", 0),
        hi_summary.get("n_switches", 0),
        hi_summary.get("n_frames_total", 0),
    )

    # ------------------------------------------------------------------
    # Save enriched pivot JSON
    # ------------------------------------------------------------------
    enriched_path = out_dir / f"{stem}_possession.json"
    save_json(data, str(enriched_path))
    logger.info("Enriched pivot saved → %s", enriched_path)

    # ------------------------------------------------------------------
    # Step 6: Render combined skeleton + possession + gait debug video
    # ------------------------------------------------------------------
    debug_video = out_dir / f"{stem}_possession_debug.mp4"
    logger.info("Step 6/6 — Rendering combined debug video → %s", debug_video.name)
    _render_combined(str(video_path), data, str(debug_video))

    logger.info("Done. Output: %s", debug_video)
    print(f"\nDebug video: {debug_video}")
    print(f"Enriched JSON: {enriched_path}")


if __name__ == "__main__":
    main()
