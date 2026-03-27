"""Generate synthetic contact sequences for smoke-testing the training loop.

.. warning::
   Dummy data is **not** suitable for validating model quality or
   benchmarking detection performance.  Use it only to verify that the
   training loop runs without errors on your system.

The synthetic sequences simulate a simplified gait cycle:
- Ankle Y follows a sinusoidal trajectory (foot rises during swing).
- Contact label = 1 when the foot is in the lower half of its range
  (stance phase), 0 otherwise.
- Small Gaussian noise is added to every feature.

Usage
-----
.. code-block:: bash

    python -m myogait.training.generate_dummy_dataset \\
        --out /tmp/dummy_gait_data \\
        [--n-sequences 50] [--min-frames 120] [--max-frames 300] \\
        [--fps 30] [--seed 0]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from myogait.detectors.learned_contact_detector import N_FEATURES
except ImportError:
    N_FEATURES = 17


def _make_sequence(
    T: int,
    fps: float,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic gait sequence.

    Parameters
    ----------
    T : int
        Number of frames.
    fps : float
        Frame rate (controls stride frequency).
    noise_std : float
        Gaussian noise standard deviation added to all features.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    features : np.ndarray, shape (T, N_FEATURES)
    labels   : np.ndarray, shape (T,), float32, {0.0, 1.0}
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(T) / fps
    stride_hz = rng.uniform(0.8, 1.2)  # ~1 stride/s ≈ 2 steps/s

    # Ankle Y: high value = foot near ground (contact); low = foot raised (swing)
    ankle_y  = 0.75 + 0.15 * np.sin(2 * np.pi * stride_hz * t)
    heel_y   = ankle_y + rng.uniform(-0.02, 0.02, T)
    toe_y    = ankle_y - 0.03 + 0.01 * np.sin(2 * np.pi * stride_hz * t + 0.5)

    # Horizontal positions: slow forward progression
    ankle_x  = 0.3 + 0.2 * t / max(t[-1], 1.0)
    heel_x   = ankle_x - 0.02
    toe_x    = ankle_x + 0.03

    knee_y   = 0.55 + 0.05 * np.sin(2 * np.pi * stride_hz * t + np.pi)
    knee_x   = ankle_x + rng.uniform(-0.02, 0.02, T)
    hip_y    = np.full(T, 0.40)
    hip_x    = ankle_x

    ankle_vis = np.clip(rng.normal(0.9, 0.05, T), 0.0, 1.0)
    heel_vis  = np.clip(rng.normal(0.85, 0.07, T), 0.0, 1.0)
    toe_vis   = np.clip(rng.normal(0.80, 0.08, T), 0.0, 1.0)

    vy_ankle = np.empty(T)
    vy_ankle[0] = 0.0
    vy_ankle[1:] = np.diff(ankle_y)

    vy_heel = np.empty(T)
    vy_heel[0] = 0.0
    vy_heel[1:] = np.diff(heel_y)

    ay_ankle = np.empty(T)
    ay_ankle[0] = 0.0
    ay_ankle[1:] = np.diff(vy_ankle)

    ay_heel = np.empty(T)
    ay_heel[0] = 0.0
    ay_heel[1:] = np.diff(vy_heel)

    feats = np.stack([
        ankle_x, ankle_y, heel_x, heel_y, toe_x, toe_y,
        knee_x, knee_y, hip_x, hip_y,
        ankle_vis, heel_vis, toe_vis,
        vy_ankle, vy_heel, ay_ankle, ay_heel,
    ], axis=1).astype(np.float32)

    feats += rng.normal(0.0, noise_std, feats.shape).astype(np.float32)

    # Contact = 1 when foot is near the ground (ankle_y above midpoint)
    mid_y = float(np.mean(ankle_y))
    labels = (ankle_y >= mid_y).astype(np.float32)

    return feats, labels


def generate(
    out_dir: Path,
    n_sequences: int = 50,
    min_frames: int = 120,
    max_frames: int = 300,
    fps: float = 30.0,
    seed: int = 0,
) -> None:
    """Write synthetic .npz files to *out_dir*.

    Parameters
    ----------
    out_dir : Path
        Output directory (created if it does not exist).
    n_sequences : int
        Number of .npz files to generate.
    min_frames, max_frames : int
        Range for randomly sampled sequence lengths.
    fps : float
        Simulated frame rate.
    seed : int
        Base random seed; each sequence gets its own derived seed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(seed)

    for i in range(n_sequences):
        T = int(rng_master.integers(min_frames, max_frames + 1))
        seq_rng = np.random.default_rng(seed + i)
        feats, labels = _make_sequence(T, fps, rng=seq_rng)

        out_path = out_dir / f"dummy_seq_{i:04d}.npz"
        np.savez(str(out_path), features=feats, labels=labels)

    n_contact = sum(
        1 for f in out_dir.glob("dummy_seq_*.npz")
        if np.load(str(f))["labels"].mean() > 0
    )
    logger.info(
        "Generated %d sequences in %s (features=%d, fps=%.0f)",
        n_sequences, out_dir, N_FEATURES, fps,
    )
    logger.info(
        "NOTE: These sequences are for smoke-testing only. "
        "Do NOT use them as real validation data."
    )


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic gait sequences for smoke-testing "
            "train_contact_tcn.py. NOT for real validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out",          type=Path, required=True,
                   help="Output directory for .npz files.")
    p.add_argument("--n-sequences",  type=int,   default=50)
    p.add_argument("--min-frames",   type=int,   default=120)
    p.add_argument("--max-frames",   type=int,   default=300)
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--seed",         type=int,   default=0)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()
    generate(
        out_dir=args.out,
        n_sequences=args.n_sequences,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        fps=args.fps,
        seed=args.seed,
    )
