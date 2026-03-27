"""Compare ContactTCN vs the zeni rule-based detector on labeled data.

Two evaluation modes are supported:

Mode 1 — NPZ sequences (``--npz-dir``)
    Each .npz file contains ``features`` (T, N_FEATURES) and ``labels``
    (T,) arrays.  The model runs on the features; ground-truth events are
    decoded from ``labels`` using the same hysteresis logic.  This mode
    gives both per-frame F1 and event-level timing metrics.

Mode 2 — myogait JSON files (``--json-dir``)
    Each JSON file is a myogait pivot dict that **must** contain
    ``data["ground_truth"]`` with the keys ``left_hs``, ``right_hs``,
    ``left_to``, ``right_to`` (same format as ``data["events"]``).
    Both ``learned_tcn`` and ``zeni`` are run on the pose data and
    compared against the stored ground-truth events.

Metrics
-------
``contact_f1``       — macro-averaged per-frame binary F1 (mode 1 only)
``hs_timing_error``  — mean |pred_frame - gt_frame| for matched HS events
``to_timing_error``  — same for toe-off events
``hs_count_diff``    — |n_pred_hs - n_gt_hs|  (event count mismatch)
``to_count_diff``    — same for toe-off

Usage
-----
.. code-block:: bash

    # Mode 1 — per-frame F1 + event metrics
    python -m myogait.training.evaluate_contact_detector \\
        --npz-dir /path/to/labeled_sequences/ \\
        [--weights ~/.myogait/contact_tcn.pt] \\
        [--window 24] [--threshold-on 0.6] [--threshold-off 0.4]

    # Mode 2 — event metrics on myogait JSON files
    python -m myogait.training.evaluate_contact_detector \\
        --json-dir /path/to/myogait_json_files/ \\
        [--weights ~/.myogait/contact_tcn.pt]

.. note::
   Dummy sequences from ``generate_dummy_dataset.py`` are **not** valid
   for measuring real detection quality.  Use properly annotated clips.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric helpers                                                                #
# --------------------------------------------------------------------------- #

def _per_frame_f1(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Binary F1 on per-frame contact labels."""
    pred = (pred_labels >= 0.5).astype(bool)
    gt   = (gt_labels  >= 0.5).astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    return float(2 * prec * rec / max(prec + rec, 1e-8))


def _matched_timing_errors(
    pred_events: List[dict],
    gt_events: List[dict],
) -> Tuple[float, int]:
    """Greedy nearest-neighbour matching of predicted to GT events.

    Returns
    -------
    mean_error : float
        Mean |pred_frame - gt_frame| across matched pairs (frames).
        NaN when there are no GT events.
    count_diff : int
        Absolute difference in event counts: |n_pred - n_gt|.
    """
    n_pred = len(pred_events)
    n_gt   = len(gt_events)
    count_diff = abs(n_pred - n_gt)

    if n_gt == 0:
        return float("nan"), count_diff
    if n_pred == 0:
        return float("nan"), count_diff

    gt_frames   = np.array([e["frame"] for e in gt_events], dtype=float)
    pred_frames = np.array([e["frame"] for e in pred_events], dtype=float)

    # For each GT event find the closest predicted event (greedy)
    errors = []
    used = set()
    for gf in gt_frames:
        dists = np.abs(pred_frames - gf)
        order = np.argsort(dists)
        for idx in order:
            if idx not in used:
                errors.append(float(dists[idx]))
                used.add(idx)
                break

    return float(np.mean(errors)) if errors else float("nan"), count_diff


def _events_from_labels(labels: np.ndarray, fps: float, config: dict) -> dict:
    """Decode events from a binary label vector using hysteresis (for GT)."""
    from myogait.detectors.learned_contact_detector import (
        _causal_smooth, _hysteresis_events, DEFAULT_TCN_CONFIG,
    )
    cfg = {**DEFAULT_TCN_CONFIG, **config}
    smooth = _causal_smooth(labels.astype(np.float32), cfg["smoothing_frames"])
    return {
        **_hysteresis_events(smooth, fps, cfg["threshold_on"], cfg["threshold_off"],
                             cfg["min_event_gap"], "left_hs", "left_to"),
        **_hysteresis_events(smooth, fps, cfg["threshold_on"], cfg["threshold_off"],
                             cfg["min_event_gap"], "right_hs", "right_to"),
    }


# --------------------------------------------------------------------------- #
# Per-sequence metrics                                                          #
# --------------------------------------------------------------------------- #

def _sequence_metrics_npz(
    path: Path,
    model,
    config: dict,
    fps: float,
) -> Optional[dict]:
    """Evaluate one .npz sequence; return metrics dict or None on error."""
    try:
        from myogait.detectors.learned_contact_detector import (
            infer_contact_probs, decode_contact_events, N_FEATURES,
        )
        data = np.load(str(path))
        feats  = data["features"].astype(np.float32)  # (T, N_FEATURES)
        labels = data["labels"].astype(np.float32)    # (T,)
    except Exception as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
        return None

    if feats.shape[1] != N_FEATURES:
        logger.warning(
            "Skipping %s: feature dim=%d, expected %d",
            path.name, feats.shape[1], N_FEATURES,
        )
        return None

    # Model predictions
    probs = infer_contact_probs(feats, model, config)
    pred_binary = (probs >= config.get("threshold_on", 0.6)).astype(np.float32)

    # Per-frame F1 (left and right are not distinguished in NPZ mode;
    # features and labels represent one foot at a time)
    f1 = _per_frame_f1(pred_binary, labels)

    # Event-level: decode from predicted probs and from ground-truth labels
    # Use the same hysteresis config for both so the comparison is fair.
    pred_events = decode_contact_events(probs, probs, fps, config)
    gt_events   = _events_from_labels(labels, fps, config)

    hs_err,  hs_cnt_diff  = _matched_timing_errors(
        pred_events["left_hs"], gt_events["left_hs"])
    to_err, to_cnt_diff = _matched_timing_errors(
        pred_events["left_to"], gt_events["left_to"])

    return {
        "contact_f1":      f1,
        "hs_timing_error": hs_err,
        "to_timing_error": to_err,
        "hs_count_diff":   hs_cnt_diff,
        "to_count_diff":   to_cnt_diff,
        "n_frames":        len(labels),
    }


def _sequence_metrics_json(
    path: Path,
    model,
    config: dict,
) -> Optional[dict]:
    """Evaluate one myogait JSON file against stored ground-truth events."""
    try:
        from myogait.detectors.learned_contact_detector import (
            build_features, infer_contact_probs, decode_contact_events,
        )
        from myogait.events import _detect_zeni  # type: ignore[import]
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
        return None

    gt = data.get("ground_truth")
    if not gt:
        logger.warning(
            "Skipping %s: no 'ground_truth' key. "
            "Annotate the file with data['ground_truth'] = "
            "{'left_hs': [...], 'right_hs': [...], ...}",
            path.name,
        )
        return None

    frames = data.get("frames", [])
    fps    = data.get("meta", {}).get("fps", 30.0)
    if not frames:
        logger.warning("Skipping %s: no frames.", path.name)
        return None

    # ---- learned_tcn ----
    feats_l, _ = build_features(frames, "left")
    feats_r, _ = build_features(frames, "right")
    probs_l = infer_contact_probs(feats_l, model, config)
    probs_r = infer_contact_probs(feats_r, model, config)
    tcn_events = decode_contact_events(probs_l, probs_r, fps, config)

    # ---- zeni ----
    zeni_events = _detect_zeni(frames, fps)

    def _side_metrics(pred: dict, side: str) -> dict:
        hs_err, hs_cnt = _matched_timing_errors(
            pred.get(f"{side}_hs", []), gt.get(f"{side}_hs", []))
        to_err, to_cnt = _matched_timing_errors(
            pred.get(f"{side}_to", []), gt.get(f"{side}_to", []))
        return {
            "hs_timing_error": hs_err,
            "to_timing_error": to_err,
            "hs_count_diff":   hs_cnt,
            "to_count_diff":   to_cnt,
        }

    tcn_l  = _side_metrics(tcn_events,  "left")
    tcn_r  = _side_metrics(tcn_events,  "right")
    zeni_l = _side_metrics(zeni_events, "left")
    zeni_r = _side_metrics(zeni_events, "right")

    def _avg(a: dict, b: dict) -> dict:
        keys = set(a) | set(b)
        out = {}
        for k in keys:
            va, vb = a.get(k, float("nan")), b.get(k, float("nan"))
            vals = [v for v in (va, vb) if not (isinstance(v, float) and np.isnan(v))]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    return {
        "learned_tcn": _avg(tcn_l, tcn_r),
        "zeni":        _avg(zeni_l, zeni_r),
        "n_frames":    len(frames),
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                   #
# --------------------------------------------------------------------------- #

def _nanmean(values: list) -> float:
    arr = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(arr)) if arr else float("nan")


def _print_npz_summary(results: List[dict]) -> None:
    keys = ["contact_f1", "hs_timing_error", "to_timing_error",
            "hs_count_diff", "to_count_diff"]
    print("\n=== ContactTCN vs GT (NPZ mode) ===")
    for k in keys:
        vals = [r[k] for r in results if r and k in r]
        print(f"  {k:<22} {_nanmean(vals):.4f}  (n={len(vals)})")
    print()


def _print_json_summary(results: List[dict]) -> None:
    print("\n=== learned_tcn vs zeni vs GT (JSON mode) ===")
    metrics = ["hs_timing_error", "to_timing_error", "hs_count_diff", "to_count_diff"]
    print(f"  {'metric':<22}  {'learned_tcn':>12}  {'zeni':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}")
    for m in metrics:
        tcn_vals  = [r["learned_tcn"][m] for r in results if r and "learned_tcn" in r]
        zeni_vals = [r["zeni"][m]         for r in results if r and "zeni" in r]
        print(
            f"  {m:<22}  {_nanmean(tcn_vals):>12.4f}  {_nanmean(zeni_vals):>12.4f}"
        )
    print()


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def _load_model_or_exit(weights_path: Path):
    try:
        import torch
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import N_FEATURES, FEATURE_SCHEMA
    except ImportError:
        logger.error("PyTorch is required for evaluation: pip install torch")
        sys.exit(1)

    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        sys.exit(1)

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    # Schema check
    saved_schema = checkpoint.get("feature_schema")
    if saved_schema and list(saved_schema) != list(FEATURE_SCHEMA):
        logger.warning(
            "Feature schema mismatch between checkpoint and current code!\n"
            "  checkpoint: %s\n  current:    %s",
            saved_schema, list(FEATURE_SCHEMA),
        )

    model = ContactTCN(
        n_features=int(checkpoint.get("n_features", N_FEATURES)),
        channels=int(checkpoint.get("channels",    32)),
        kernel_size=int(checkpoint.get("kernel_size", 3)),
        n_dilations=int(checkpoint.get("n_dilations", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(
        "Loaded ContactTCN  n_dilations=%d  RF=%d frames  val_f1=%.4f",
        model.n_dilations, model.receptive_field,
        float(checkpoint.get("val_f1", float("nan"))),
    )
    return model


def evaluate(
    npz_dir: Optional[Path] = None,
    json_dir: Optional[Path] = None,
    weights: Optional[Path] = None,
    window: int = 24,
    threshold_on: float = 0.6,
    threshold_off: float = 0.4,
    min_event_gap: int = 5,
    smoothing_frames: int = 5,
    fps: float = 30.0,
) -> None:
    """Run evaluation in NPZ mode, JSON mode, or both."""
    if npz_dir is None and json_dir is None:
        logger.error("Provide at least one of --npz-dir or --json-dir.")
        sys.exit(1)

    from myogait.detectors.learned_contact_detector import DEFAULT_WEIGHTS_PATH

    weights_path = weights or _DEFAULT_WEIGHTS_PATH
    model = _load_model_or_exit(weights_path)

    config = {
        "window_size":     window,
        "threshold_on":    threshold_on,
        "threshold_off":   threshold_off,
        "min_event_gap":   min_event_gap,
        "smoothing_frames": smoothing_frames,
    }

    if npz_dir is not None:
        files = sorted(npz_dir.glob("*.npz"))
        logger.info("NPZ mode: %d files in %s", len(files), npz_dir)
        results = [_sequence_metrics_npz(p, model, config, fps) for p in files]
        results = [r for r in results if r is not None]
        if results:
            _print_npz_summary(results)
        else:
            logger.warning("No valid results in NPZ mode.")

    if json_dir is not None:
        files = sorted(json_dir.glob("*.json"))
        logger.info("JSON mode: %d files in %s", len(files), json_dir)
        results = [_sequence_metrics_json(p, model, config) for p in files]
        results = [r for r in results if r is not None]
        if results:
            _print_json_summary(results)
        else:
            logger.warning("No valid results in JSON mode.")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

_DEFAULT_WEIGHTS_PATH = Path.home() / ".myogait" / "contact_tcn.pt"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate ContactTCN vs zeni on labeled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz-dir",        type=Path, default=None,
                   help="Directory of labeled .npz sequences (mode 1).")
    p.add_argument("--json-dir",       type=Path, default=None,
                   help="Directory of myogait JSON files with ground_truth (mode 2).")
    p.add_argument("--weights",        type=Path, default=_DEFAULT_WEIGHTS_PATH,
                   help="Path to contact_tcn.pt checkpoint.")
    p.add_argument("--window",         type=int,   default=24)
    p.add_argument("--threshold-on",   type=float, default=0.6)
    p.add_argument("--threshold-off",  type=float, default=0.4)
    p.add_argument("--min-event-gap",  type=int,   default=5)
    p.add_argument("--smoothing",      type=int,   default=5)
    p.add_argument("--fps",            type=float, default=30.0)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()
    evaluate(
        npz_dir=args.npz_dir,
        json_dir=args.json_dir,
        weights=args.weights,
        window=args.window,
        threshold_on=args.threshold_on,
        threshold_off=args.threshold_off,
        min_event_gap=args.min_event_gap,
        smoothing_frames=args.smoothing,
        fps=args.fps,
    )
