"""Learned temporal contact detector based on a causal TCN.

This module provides an alternative to rule-based gait event detection
for situations where those methods are brittle: noisy keypoints,
non-periodic motion (jumps, pivots, sports), or ambiguous contact states.

Public helpers (unit-testable without PyTorch)
----------------------------------------------
build_features(frames, side)
    Extract a (n_frames, N_FEATURES) feature matrix for one foot.
    Returns (feats, validity_mask) where validity_mask[i] is True when
    the ankle landmark was genuinely observed (not imputed) on frame i.
decode_contact_events(probs_left, probs_right, fps, config)
    Convert probability traces to HS/TO event dicts with hysteresis.

Requires PyTorch for inference
-------------------------------
infer_contact_probs(feats, model, config)
    Batch sliding-window TCN inference → per-frame probabilities.
    Raises if any NaN reaches the model; use build_features() to ensure
    clean input.

Integration
-----------
On import this module registers ``"learned_tcn"`` in the myogait event
method registry, so ``detect_events(data, method="learned_tcn")`` works
out of the box.  If PyTorch is not installed or no weights file is found
the detector **visibly** falls back to the rule-based ``"zeni"`` method
with a WARNING-level log explaining the exact reason.

Configuration
-------------
Override defaults before calling ``detect_events``::

    from myogait.detectors import learned_contact_detector as lcd
    lcd.configure({"threshold_on": 0.65, "smoothing_frames": 7})

Or pass a config dict to any of the public helper functions directly.

Weights
-------
Default search path: ``~/.myogait/contact_tcn.pt``
Override:            ``MYOGAIT_TCN_WEIGHTS`` environment variable

Checkpoint format (produced by train_contact_tcn.py)::

    {
        "state_dict":     OrderedDict(...),
        "n_features":     17,
        "channels":       32,
        "kernel_size":    3,
        "n_dilations":    3,
        "feature_schema": [...],   # list of 17 column names (see FEATURE_SCHEMA)
        "val_f1":         float,
        "epoch":          int,
    }

Feature schema
--------------
The 17-column feature vector produced by :func:`build_features` is
documented in :data:`FEATURE_SCHEMA`.  The checkpoint stores a copy so
that future code can detect schema mismatches between a saved model and
the current extractor.

Visualization
-------------
After a successful inference run, per-frame probabilities are cached::

    probs = lcd.get_last_contact_probs()  # {"left": ndarray, "right": ndarray}

Store in ``data["contact_probs"]`` to enable the ``show_contact_probs``
overlay in ``render_skeleton_video``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Feature schema (single source of truth for column layout)                    #
# --------------------------------------------------------------------------- #

#: Ordered tuple of column names for the 17-feature vector.
#: Indices must stay stable once a model is trained — changing them
#: requires retraining and updating the checkpoint's ``feature_schema`` field.
FEATURE_SCHEMA: Tuple[str, ...] = (
    # Foot joint coordinates (normalised [0,1] image coords)
    "ankle_x",    # col 0
    "ankle_y",    # col 1
    "heel_x",     # col 2
    "heel_y",     # col 3
    "toe_x",      # col 4  (LEFT/RIGHT_FOOT_INDEX landmark)
    "toe_y",      # col 5
    "knee_x",     # col 6
    "knee_y",     # col 7
    "hip_x",      # col 8
    "hip_y",      # col 9
    # Landmark visibility scores (extractor confidence, [0,1])
    "ankle_vis",  # col 10
    "heel_vis",   # col 11
    "toe_vis",    # col 12
    # Causal temporal derivatives (forward-difference, no future leakage)
    "vy_ankle",   # col 13  Δy ankle  (frame[i].y − frame[i-1].y)
    "vy_heel",    # col 14  Δy heel
    "ay_ankle",   # col 15  Δ²y ankle (Δvy)
    "ay_heel",    # col 16  Δ²y heel
)

#: Total number of features per foot per frame.
N_FEATURES: int = len(FEATURE_SCHEMA)  # 17

# --------------------------------------------------------------------------- #
# Constants / defaults                                                          #
# --------------------------------------------------------------------------- #

DEFAULT_TCN_CONFIG: dict = {
    # ── Inference ──────────────────────────────────────────────────────────
    # Sliding window length fed to the TCN.  Must be ≥ model receptive field.
    "window_size": 24,

    # ── Post-processing (hysteresis state machine) ──────────────────────────
    # Probability threshold to *enter* the "planted" (contact) state.
    "threshold_on": 0.6,
    # Probability threshold to *exit* the "planted" state.
    # MUST be < threshold_on — the gap between them is the hysteresis band.
    "threshold_off": 0.4,
    # Minimum frames between two events of the same type on the same foot
    # (debounce).  At 30 fps, gap=5 ≈ 167 ms.
    "min_event_gap": 5,
    # Length of the causal (backward-looking) moving-average smoothing window
    # applied to raw probabilities before the state machine runs.
    "smoothing_frames": 5,

    # ── Model architecture (used at training time; ignored at inference) ────
    # Set n_dilations here (or via --n-dilations CLI flag) to change the
    # temporal receptive field.  At inference the value is read from the
    # checkpoint; this entry only guides train_contact_tcn.py.
    #
    # Receptive field table (kernel_size=3):
    #   n_dilations=3 → RF=15  (~0.5 s at 30 fps)  ← default
    #   n_dilations=4 → RF=31  (~1.0 s)
    #   n_dilations=5 → RF=63  (~2.1 s)
    #   n_dilations=6 → RF=127 (~4.2 s)
    "n_dilations": 3,
}

_DEFAULT_WEIGHTS_ENV = "MYOGAIT_TCN_WEIGHTS"
_DEFAULT_WEIGHTS_PATH = Path.home() / ".myogait" / "contact_tcn.pt"

# Module-level config override (see configure())
_active_config: dict = {}

# Module-level cache populated after each successful inference run
_last_contact_probs: Dict[str, np.ndarray] = {}

# Landmark name mapping per foot side
_FOOT_LANDMARKS: Dict[str, Dict[str, str]] = {
    "left": {
        "ankle": "LEFT_ANKLE",
        "heel":  "LEFT_HEEL",
        "toe":   "LEFT_FOOT_INDEX",
        "knee":  "LEFT_KNEE",
        "hip":   "LEFT_HIP",
    },
    "right": {
        "ankle": "RIGHT_ANKLE",
        "heel":  "RIGHT_HEEL",
        "toe":   "RIGHT_FOOT_INDEX",
        "knee":  "RIGHT_KNEE",
        "hip":   "RIGHT_HIP",
    },
}

# --------------------------------------------------------------------------- #
# Configuration                                                                 #
# --------------------------------------------------------------------------- #

def configure(overrides: dict) -> None:
    """Override default TCN detector configuration.

    Call this once before ``detect_events(data, method="learned_tcn")``
    to customise thresholds, window size, or smoothing.  Unspecified
    keys keep their defaults from :data:`DEFAULT_TCN_CONFIG`.

    Parameters
    ----------
    overrides : dict
        Partial config dict.  Valid keys (see :data:`DEFAULT_TCN_CONFIG`):
        ``window_size``, ``threshold_on``, ``threshold_off``,
        ``min_event_gap``, ``smoothing_frames``, ``n_dilations``.
    """
    global _active_config
    _active_config = {**DEFAULT_TCN_CONFIG, **overrides}


def get_last_contact_probs() -> Dict[str, np.ndarray]:
    """Return per-frame contact probabilities from the last inference run.

    Returns
    -------
    dict
        ``{"left": ndarray, "right": ndarray}`` of shape ``(n_frames,)``.
        Empty dict if no inference has been run yet.
    """
    return {k: v.copy() for k, v in _last_contact_probs.items()}


# --------------------------------------------------------------------------- #
# Feature extraction                                                            #
# --------------------------------------------------------------------------- #

def _fill_nan_series(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN; zero-fill if all NaN."""
    out = arr.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return np.nan_to_num(out, nan=0.0)


def build_features(
    frames: list,
    side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-frame feature matrix for one foot.

    Feature layout is defined by :data:`FEATURE_SCHEMA` (17 columns):

    ===  ============  ===================================================
    Col  Name          Description
    ===  ============  ===================================================
    0    ankle_x       Ankle X coordinate (normalised [0,1])
    1    ankle_y       Ankle Y coordinate (normalised [0,1])
    2    heel_x        Heel X
    3    heel_y        Heel Y
    4    toe_x         Toe X  (FOOT_INDEX landmark)
    5    toe_y         Toe Y
    6    knee_x        Knee X
    7    knee_y        Knee Y
    8    hip_x         Hip X
    9    hip_y         Hip Y
    10   ankle_vis     Ankle visibility score [0,1]
    11   heel_vis      Heel visibility score
    12   toe_vis       Toe visibility score
    13   vy_ankle      Δy ankle  (frame[i].y − frame[i-1].y; frame 0 = 0)
    14   vy_heel       Δy heel
    15   ay_ankle      Δ²y ankle (Δvy; frame 0 = 0)
    16   ay_heel       Δ²y heel
    ===  ============  ===================================================

    NaN handling
    ~~~~~~~~~~~~
    1. Extract raw values; missing landmarks produce NaN.
    2. Forward-fill then backward-fill NaN (deterministic imputation).
    3. Zero-fill any residual NaN (only when the entire sequence is
       missing for a landmark, e.g. model does not output that joint).
    4. Velocity/acceleration computed *after* fill → no NaN propagation.
    5. The returned ``validity_mask`` records which frames had genuine
       (non-imputed) ankle data so callers can weight or flag them.

    Parameters
    ----------
    frames : list
        Frame list from a myogait data dict (``data["frames"]``).
    side : str
        ``"left"`` or ``"right"``.

    Returns
    -------
    feats : np.ndarray, shape (n_frames, N_FEATURES), dtype float32
        Feature matrix.  Guaranteed NaN-free.
    validity_mask : np.ndarray, shape (n_frames,), dtype bool
        ``True`` where the ankle landmark was genuinely observed
        (not forward/backward-filled from a neighbouring frame).
    """
    if side not in _FOOT_LANDMARKS:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    lm_names = _FOOT_LANDMARKS[side]
    n = len(frames)

    if n == 0:
        empty_feats = np.zeros((0, N_FEATURES), dtype=np.float32)
        empty_mask  = np.zeros(0, dtype=bool)
        return empty_feats, empty_mask

    def _series(lm_name: str, coord: str) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        for i, f in enumerate(frames):
            lm = f.get("landmarks", {}).get(lm_name)
            if lm is None:
                continue
            v = lm.get(coord)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out[i] = float(v)
        return out

    # Build validity mask from ankle (primary contact landmark) before fill
    raw_ankle_y = _series(lm_names["ankle"], "y")
    validity_mask = ~np.isnan(raw_ankle_y)   # True = real observation

    ankle_x   = _fill_nan_series(_series(lm_names["ankle"], "x"))
    ankle_y   = _fill_nan_series(raw_ankle_y)
    heel_x    = _fill_nan_series(_series(lm_names["heel"],  "x"))
    heel_y    = _fill_nan_series(_series(lm_names["heel"],  "y"))
    toe_x     = _fill_nan_series(_series(lm_names["toe"],   "x"))
    toe_y     = _fill_nan_series(_series(lm_names["toe"],   "y"))
    knee_x    = _fill_nan_series(_series(lm_names["knee"],  "x"))
    knee_y    = _fill_nan_series(_series(lm_names["knee"],  "y"))
    hip_x     = _fill_nan_series(_series(lm_names["hip"],   "x"))
    hip_y     = _fill_nan_series(_series(lm_names["hip"],   "y"))
    ankle_vis = _fill_nan_series(_series(lm_names["ankle"], "visibility"))
    heel_vis  = _fill_nan_series(_series(lm_names["heel"],  "visibility"))
    toe_vis   = _fill_nan_series(_series(lm_names["toe"],   "visibility"))

    # Causal velocity: Δy[i] = y[i] - y[i-1]; padded with 0 at frame 0
    vy_ankle = np.empty(n, dtype=np.float64)
    vy_heel  = np.empty(n, dtype=np.float64)
    vy_ankle[0] = 0.0
    vy_heel[0]  = 0.0
    if n > 1:
        vy_ankle[1:] = np.diff(ankle_y)
        vy_heel[1:]  = np.diff(heel_y)

    # Causal acceleration: Δ²y
    ay_ankle = np.empty(n, dtype=np.float64)
    ay_heel  = np.empty(n, dtype=np.float64)
    ay_ankle[0] = 0.0
    ay_heel[0]  = 0.0
    if n > 1:
        ay_ankle[1:] = np.diff(vy_ankle)
        ay_heel[1:]  = np.diff(vy_heel)

    # Stack in FEATURE_SCHEMA order (columns 0-16)
    feats = np.stack([
        ankle_x, ankle_y,          # 0-1
        heel_x,  heel_y,           # 2-3
        toe_x,   toe_y,            # 4-5
        knee_x,  knee_y,           # 6-7
        hip_x,   hip_y,            # 8-9
        ankle_vis, heel_vis, toe_vis,  # 10-12
        vy_ankle, vy_heel,         # 13-14
        ay_ankle, ay_heel,         # 15-16
    ], axis=1)  # (n_frames, 17)

    feats = feats.astype(np.float32)

    # Final NaN guard: should never trigger after the fill pipeline above,
    # but provides a deterministic fallback and a loud error if it does.
    nan_count = int(np.isnan(feats).sum())
    if nan_count > 0:
        logger.error(
            "build_features(%s): %d NaN values remain after imputation — "
            "replacing with 0. This indicates a bug in the fill pipeline.",
            side, nan_count,
        )
        feats = np.nan_to_num(feats, nan=0.0)

    return feats, validity_mask


# --------------------------------------------------------------------------- #
# Inference                                                                     #
# --------------------------------------------------------------------------- #

def infer_contact_probs(
    feats: np.ndarray,
    model,
    config: Optional[dict] = None,
) -> np.ndarray:
    """Run causal sliding-window TCN inference.

    Builds a batch of all windows in one pass (no Python loop over
    frames), then extracts the causal prediction from the last position
    of each window.

    NaN contract
    ~~~~~~~~~~~~
    ``feats`` must be NaN-free.  Use :func:`build_features` which
    guarantees this.  A hard error is raised (not silently ignored)
    if NaN is detected, so training data corruption surfaces early.

    Parameters
    ----------
    feats : np.ndarray, shape (n_frames, n_features)
        NaN-free feature matrix from :func:`build_features`.
    model : ContactTCN
        Loaded TCN model (must be in ``eval`` mode).
    config : dict, optional
        Detector config overrides.  Relevant key: ``window_size``.

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Contact probability in [0, 1] for each frame.

    Raises
    ------
    ValueError
        If ``feats`` contains NaN values.
    """
    import torch

    cfg = {**DEFAULT_TCN_CONFIG, **_active_config, **(config or {})}
    window = int(cfg["window_size"])
    n = len(feats)

    if n == 0:
        return np.array([], dtype=np.float32)

    # Hard NaN check — fail loudly rather than silently corrupt predictions
    nan_count = int(np.isnan(feats).sum())
    if nan_count > 0:
        raise ValueError(
            f"infer_contact_probs: feature matrix contains {nan_count} NaN "
            f"values. Call build_features() which guarantees a NaN-free "
            f"output before passing features to the model."
        )

    # Left-pad with the first frame so that frame 0 gets a full window
    # (pure causal padding — no future leakage).
    pad = window - 1
    if pad > 0:
        padded = np.concatenate(
            [np.tile(feats[:1], (pad, 1)), feats], axis=0
        )  # (pad + n, n_features)
    else:
        padded = feats

    # Build all windows at once via numpy index broadcasting.
    # Row i spans padded[i : i+window], which is the causal context for frame i.
    row_idx = np.arange(n)[:, None] + np.arange(window)[None, :]  # (n, window)
    windows = padded[row_idx]  # (n, window, n_features)

    model.eval()
    x = torch.from_numpy(windows)  # (n, window, n_features)
    with torch.no_grad():
        logits = model(x)                       # (n, window)
        probs  = torch.sigmoid(logits).numpy()  # (n, window)

    # The last position of each window is the causal estimate for that frame
    return probs[:, -1].astype(np.float32)


# --------------------------------------------------------------------------- #
# Post-processing: hysteresis state machine                                     #
# --------------------------------------------------------------------------- #

def _causal_smooth(probs: np.ndarray, window: int) -> np.ndarray:
    """Causal (backward-looking) moving average."""
    if window <= 1 or len(probs) == 0:
        return probs.copy().astype(np.float32)
    cs = np.cumsum(probs, dtype=np.float64)
    out = np.empty(len(probs), dtype=np.float32)
    for i in range(len(probs)):
        start = max(0, i - window + 1)
        out[i] = float(cs[i] - (cs[start - 1] if start > 0 else 0.0)) / (i - start + 1)
    return out


def _hysteresis_events(
    probs: np.ndarray,
    fps: float,
    threshold_on: float,
    threshold_off: float,
    min_gap: int,
    hs_key: str,
    to_key: str,
) -> dict:
    """Run hysteresis state machine on a single foot's probability trace.

    State transitions
    -----------------
    ``airborne`` → ``planted`` when prob ≥ ``threshold_on``   → heel strike
    ``planted``  → ``airborne`` when prob < ``threshold_off``  → toe-off

    Signals in the hysteresis band (threshold_off ≤ prob < threshold_on)
    do **not** change state, preventing oscillation near the boundary.

    The ``min_gap`` debounce suppresses a second event of the same type
    that occurs within ``min_gap`` frames of the previous one.
    Confidence equals the probability value at the transition frame.
    """
    events: dict = {hs_key: [], to_key: []}
    if len(probs) == 0:
        return events

    state = "airborne"
    last_hs = -(min_gap + 1)
    last_to = -(min_gap + 1)

    for i, p in enumerate(probs):
        p = float(p)
        if state == "airborne":
            if p >= threshold_on and (i - last_hs) > min_gap:
                state = "planted"
                last_hs = i
                events[hs_key].append({
                    "frame":      int(i),
                    "time":       round(float(i) / fps, 4),
                    "confidence": round(p, 3),
                })
        else:  # planted
            if p < threshold_off and (i - last_to) > min_gap:
                state = "airborne"
                last_to = i
                events[to_key].append({
                    "frame":      int(i),
                    "time":       round(float(i) / fps, 4),
                    "confidence": round(1.0 - p, 3),
                })

    return events


def decode_contact_events(
    probs_left: np.ndarray,
    probs_right: np.ndarray,
    fps: float,
    config: Optional[dict] = None,
) -> dict:
    """Convert contact probability traces to HS/TO event dicts.

    Applies causal smoothing then a hysteresis state machine with
    debouncing.  The on-threshold is deliberately higher than the
    off-threshold to prevent rapid toggling near the decision boundary.

    Parameters
    ----------
    probs_left : np.ndarray, shape (n_frames,)
        Per-frame contact probabilities for the left foot.
    probs_right : np.ndarray, shape (n_frames,)
        Per-frame contact probabilities for the right foot.
    fps : float
        Video frame rate (used to compute event timestamps).
    config : dict, optional
        Detector config overrides.

    Returns
    -------
    dict
        Identical format to rule-based detectors::

            {
                "left_hs":  [{"frame": int, "time": float, "confidence": float}, ...],
                "left_to":  [...],
                "right_hs": [...],
                "right_to": [...],
            }
    """
    cfg = {**DEFAULT_TCN_CONFIG, **_active_config, **(config or {})}
    threshold_on  = float(cfg["threshold_on"])
    threshold_off = float(cfg["threshold_off"])
    min_gap       = int(cfg["min_event_gap"])
    smooth_w      = int(cfg["smoothing_frames"])

    # Validate hysteresis invariant
    if threshold_off >= threshold_on:
        logger.warning(
            "learned_tcn: threshold_off (%.2f) >= threshold_on (%.2f). "
            "Hysteresis requires threshold_off < threshold_on. "
            "Clamping threshold_off to threshold_on - 0.1.",
            threshold_off, threshold_on,
        )
        threshold_off = max(0.0, threshold_on - 0.1)

    smooth_left  = _causal_smooth(probs_left,  smooth_w)
    smooth_right = _causal_smooth(probs_right, smooth_w)

    return {
        **_hysteresis_events(smooth_left,  fps, threshold_on, threshold_off,
                             min_gap, "left_hs",  "left_to"),
        **_hysteresis_events(smooth_right, fps, threshold_on, threshold_off,
                             min_gap, "right_hs", "right_to"),
    }


# --------------------------------------------------------------------------- #
# Model loading with explicit fallback reason                                   #
# --------------------------------------------------------------------------- #

# Fallback reason codes
_REASON_OK           = "ok"
_REASON_NO_TORCH     = "torch_not_installed"
_REASON_NO_FILE      = "weights_file_not_found"
_REASON_LOAD_FAILED  = "weights_load_failed"


def _load_model() -> Tuple[Optional[object], str]:
    """Load ContactTCN weights; return (model, reason) tuple.

    Returns
    -------
    model : ContactTCN or None
    reason : str
        One of the ``_REASON_*`` constants.  Callers should log
        :func:`_log_fallback` when reason != ``_REASON_OK``.
    """
    try:
        import torch
        from .contact_tcn import ContactTCN
    except ImportError:
        return None, _REASON_NO_TORCH

    weights_path = Path(
        os.environ.get(_DEFAULT_WEIGHTS_ENV, str(_DEFAULT_WEIGHTS_PATH))
    )
    if not weights_path.exists():
        return None, _REASON_NO_FILE

    try:
        checkpoint = torch.load(
            str(weights_path), map_location="cpu", weights_only=True
        )
        _validate_checkpoint_schema(checkpoint, weights_path)
        model = ContactTCN(
            n_features=int(checkpoint.get("n_features", N_FEATURES)),
            channels=int(checkpoint.get("channels",    32)),
            kernel_size=int(checkpoint.get("kernel_size", 3)),
            n_dilations=int(checkpoint.get("n_dilations", 3)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        logger.info("learned_tcn: loaded weights from %s", weights_path)
        return model, _REASON_OK
    except Exception as exc:
        return None, f"{_REASON_LOAD_FAILED}:{exc}"


def _validate_checkpoint_schema(checkpoint: dict, path: Path) -> None:
    """Warn if the checkpoint's feature schema differs from FEATURE_SCHEMA."""
    saved_schema = checkpoint.get("feature_schema")
    if saved_schema is None:
        logger.debug(
            "learned_tcn: checkpoint %s has no 'feature_schema' key "
            "(saved before schema tracking was added).", path.name,
        )
        return
    if list(saved_schema) != list(FEATURE_SCHEMA):
        logger.warning(
            "learned_tcn: feature schema mismatch!\n"
            "  checkpoint (%s): %s\n"
            "  current code:    %s\n"
            "The model may produce incorrect predictions.  Retrain with "
            "the current version of build_features().",
            path.name, saved_schema, list(FEATURE_SCHEMA),
        )


def _log_fallback(reason: str, weights_path: Path) -> None:
    """Emit a clearly formatted WARNING explaining why TCN fell back to zeni."""
    if reason == _REASON_NO_TORCH:
        logger.warning(
            "learned_tcn ► FALLBACK TO ZENI\n"
            "  Reason : PyTorch is not installed.\n"
            "  Fix    : pip install torch"
        )
    elif reason == _REASON_NO_FILE:
        logger.warning(
            "learned_tcn ► FALLBACK TO ZENI\n"
            "  Reason : No weights file found at '%s'.\n"
            "  Fix    : Set MYOGAIT_TCN_WEIGHTS=<path> or place a "
            "checkpoint at the default path.\n"
            "           Train one with: python -m myogait.training.train_contact_tcn",
            weights_path,
        )
    else:
        # _REASON_LOAD_FAILED:<message>
        detail = reason.replace(f"{_REASON_LOAD_FAILED}:", "", 1)
        logger.warning(
            "learned_tcn ► FALLBACK TO ZENI\n"
            "  Reason : Failed to load weights from '%s'.\n"
            "  Detail : %s",
            weights_path, detail,
        )


# --------------------------------------------------------------------------- #
# Registry-compatible detector function                                         #
# --------------------------------------------------------------------------- #

def _detect_learned_tcn(
    frames: list,
    fps: float,
    min_cycle_duration: float = 0.4,
    cutoff_freq: float = 6.0,
) -> dict:
    """Gait event detector using the learned ContactTCN.

    Signature matches the myogait event-method registry contract:
    ``(frames, fps, min_cycle_duration, cutoff_freq) -> dict``.

    Falls back to the rule-based ``"zeni"`` detector with a visible
    WARNING when PyTorch is unavailable or no checkpoint is found.
    Module-level config is read from :data:`_active_config`; call
    :func:`configure` to override defaults before detection.
    """
    global _last_contact_probs

    weights_path = Path(
        os.environ.get(_DEFAULT_WEIGHTS_ENV, str(_DEFAULT_WEIGHTS_PATH))
    )
    model, reason = _load_model()
    if model is None:
        _log_fallback(reason, weights_path)
        from ..events import _detect_zeni  # type: ignore[import]
        return _detect_zeni(frames, fps, min_cycle_duration, cutoff_freq)

    cfg = {**DEFAULT_TCN_CONFIG, **_active_config}

    feats_left,  mask_left  = build_features(frames, "left")
    feats_right, mask_right = build_features(frames, "right")

    # Log coverage so users can spot badly occluded sequences
    if len(mask_left) > 0:
        cov_l = int(mask_left.sum())
        cov_r = int(mask_right.sum())
        n = len(mask_left)
        logger.debug(
            "learned_tcn: landmark coverage — left ankle %d/%d (%.0f%%), "
            "right ankle %d/%d (%.0f%%)",
            cov_l, n, 100 * cov_l / n,
            cov_r, n, 100 * cov_r / n,
        )

    probs_left  = infer_contact_probs(feats_left,  model, cfg)
    probs_right = infer_contact_probs(feats_right, model, cfg)

    # Cache for downstream visualization
    _last_contact_probs = {"left": probs_left, "right": probs_right}

    return decode_contact_events(probs_left, probs_right, fps, cfg)


# --------------------------------------------------------------------------- #
# Auto-register on import                                                       #
# --------------------------------------------------------------------------- #

from ..events import register_event_method as _register  # noqa: E402

_register("learned_tcn", _detect_learned_tcn)
