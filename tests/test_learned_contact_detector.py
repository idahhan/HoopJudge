"""Unit tests for the learned TCN contact detector.

These tests intentionally avoid PyTorch and cover only the pure-numpy
helpers, making them fast and dependency-free.  A separate section
tests the fallback behaviour and registry integration.  A final section
(torch-gated) tests the model itself and checkpoint roundtrip.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Make sure the package root is importable when running pytest from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from myogait.detectors.learned_contact_detector import (
    N_FEATURES,
    FEATURE_SCHEMA,
    DEFAULT_TCN_CONFIG,
    build_features,
    decode_contact_events,
    _causal_smooth,
    _hysteresis_events,
    get_last_contact_probs,
    configure,
    _active_config,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _make_frames(n: int, side: str = "left") -> list:
    """Create minimal synthetic frame dicts with all required landmarks."""
    prefix = "LEFT" if side == "left" else "RIGHT"
    frames = []
    for i in range(n):
        t = i / 30.0
        frames.append({
            "frame_idx": i,
            "landmarks": {
                f"{prefix}_ANKLE": {"x": 0.3 + 0.05 * np.sin(t), "y": 0.80 + 0.05 * np.cos(t), "visibility": 0.95},
                f"{prefix}_HEEL":  {"x": 0.28 + 0.05 * np.sin(t), "y": 0.82 + 0.05 * np.cos(t), "visibility": 0.90},
                f"{prefix}_FOOT_INDEX": {"x": 0.33, "y": 0.83, "visibility": 0.85},
                f"{prefix}_KNEE":  {"x": 0.30, "y": 0.55, "visibility": 0.97},
                f"{prefix}_HIP":   {"x": 0.30, "y": 0.40, "visibility": 0.98},
            },
        })
    return frames


def _make_contact_probs(n: int, period: int = 30) -> np.ndarray:
    """Synthetic probability trace with clear on/off alternation."""
    t = np.arange(n)
    return (0.5 + 0.45 * np.sin(2 * np.pi * t / period)).astype(np.float32)


# --------------------------------------------------------------------------- #
# build_features                                                                #
# --------------------------------------------------------------------------- #

class TestBuildFeatures:
    def test_output_shape(self):
        frames = _make_frames(60, "left")
        feats, mask = build_features(frames, "left")
        assert feats.shape == (60, N_FEATURES), (
            f"Expected (60, {N_FEATURES}), got {feats.shape}"
        )
        assert mask.shape == (60,)
        assert mask.dtype == bool

    def test_dtype(self):
        feats, _ = build_features(_make_frames(30, "right"), "right")
        assert feats.dtype == np.float32

    def test_no_nan(self):
        feats, _ = build_features(_make_frames(30, "left"), "left")
        assert not np.any(np.isnan(feats)), "build_features should not produce NaN"

    def test_missing_landmarks_filled(self):
        """Frames with missing landmarks should be filled, not NaN."""
        frames = _make_frames(10, "left")
        frames[5]["landmarks"] = {}  # wipe frame 5
        feats, mask = build_features(frames, "left")
        assert not np.any(np.isnan(feats)), "NaN after fill for missing frame"
        assert mask[5] == False, "Frame 5 should be invalid in validity_mask"

    def test_validity_mask_true_when_data_present(self):
        """All frames with real ankle data should show True in mask."""
        frames = _make_frames(20, "left")
        _, mask = build_features(frames, "left")
        assert mask.all(), "All frames have ankle data, so all should be valid"

    def test_velocity_column_is_causal(self):
        """vy_ankle[0] must be 0 (no future leakage at frame 0)."""
        feats, _ = build_features(_make_frames(30, "left"), "left")
        # Column 13 = vy_ankle  (per FEATURE_SCHEMA)
        assert feats[0, 13] == pytest.approx(0.0), "vy_ankle[0] should be 0"

    def test_feature_schema_alignment(self):
        """Column 13 must be vy_ankle per FEATURE_SCHEMA."""
        assert FEATURE_SCHEMA[13] == "vy_ankle"
        assert FEATURE_SCHEMA[14] == "vy_heel"
        assert FEATURE_SCHEMA[15] == "ay_ankle"
        assert len(FEATURE_SCHEMA) == N_FEATURES

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side must be"):
            build_features(_make_frames(10), "center")

    def test_single_frame(self):
        feats, mask = build_features(_make_frames(1, "left"), "left")
        assert feats.shape == (1, N_FEATURES)
        assert len(mask) == 1

    def test_empty_frames(self):
        feats, mask = build_features([], "left")
        assert feats.shape == (0, N_FEATURES)
        assert len(mask) == 0

    # ── NEW: no-future-leakage (numpy level) ──────────────────────────────

    def test_velocity_no_future_leakage(self):
        """Changing frame k must not affect features of frames 0..k-1."""
        frames = _make_frames(15, "left")
        feats_orig, _ = build_features(frames, "left")

        # Deep-copy and modify frame 10 (in the middle of the sequence)
        frames_mod = [dict(f) for f in frames]
        frames_mod[10] = dict(frames[10])
        frames_mod[10]["landmarks"] = dict(frames[10]["landmarks"])
        frames_mod[10]["landmarks"]["LEFT_ANKLE"] = {
            "x": 0.99, "y": 0.99, "visibility": 0.1
        }
        feats_mod, _ = build_features(frames_mod, "left")

        # Frames 0-9 must be identical
        np.testing.assert_array_equal(
            feats_orig[:10],
            feats_mod[:10],
            err_msg="Modifying frame 10 changed features of earlier frames (leakage)",
        )
        # Frame 10 itself must differ
        assert not np.allclose(feats_orig[10], feats_mod[10]), (
            "Frame 10 should be different after modification"
        )

    # ── NEW: stable handling of missing keypoints ─────────────────────────

    def test_all_landmarks_missing_produces_zeros(self):
        """All-empty frames → all-zero features, no NaN."""
        frames = [{"frame_idx": i, "landmarks": {}} for i in range(30)]
        feats, mask = build_features(frames, "left")
        assert feats.shape == (30, N_FEATURES)
        assert not np.any(np.isnan(feats))
        assert np.all(feats == 0.0), "All-missing frames should yield zero after fill"
        assert not mask.any(), "No valid observations → mask should be all-False"

    def test_sparse_keypoints_no_nan(self):
        """Frames with only some landmarks present should fill gracefully."""
        frames = _make_frames(30, "left")
        # Remove heel from odd frames
        for i in range(1, 30, 2):
            frames[i]["landmarks"].pop("LEFT_HEEL", None)
        feats, _ = build_features(frames, "left")
        assert not np.any(np.isnan(feats)), "Sparse heel data should not produce NaN"

    def test_single_valid_frame_fills_all(self):
        """A single valid frame surrounded by empties → constant-filled sequence."""
        frames = [{"frame_idx": i, "landmarks": {}} for i in range(10)]
        frames[5]["landmarks"] = {
            "LEFT_ANKLE": {"x": 0.4, "y": 0.8, "visibility": 0.9},
            "LEFT_HEEL":  {"x": 0.38, "y": 0.82, "visibility": 0.85},
            "LEFT_FOOT_INDEX": {"x": 0.42, "y": 0.83, "visibility": 0.8},
            "LEFT_KNEE":  {"x": 0.4, "y": 0.55, "visibility": 0.95},
            "LEFT_HIP":   {"x": 0.4, "y": 0.4, "visibility": 0.97},
        }
        feats, mask = build_features(frames, "left")
        assert not np.any(np.isnan(feats))
        # Only frame 5 is genuinely valid
        assert int(mask.sum()) == 1 and mask[5]
        # ankle_x (col 0) should be 0.4 everywhere after fill
        np.testing.assert_allclose(feats[:, 0], 0.4, atol=1e-5)


# --------------------------------------------------------------------------- #
# _causal_smooth                                                                #
# --------------------------------------------------------------------------- #

class TestCausalSmooth:
    def test_window_1_is_identity(self):
        arr = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
        out = _causal_smooth(arr, 1)
        np.testing.assert_allclose(out, arr, rtol=1e-5)

    def test_no_future_leakage(self):
        """At each position, smoothed value must equal mean of past ≤ window frames."""
        arr = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = _causal_smooth(arr, 3)
        # Position 0: only 1 frame → mean([1.0]) = 1.0
        assert out[0] == pytest.approx(1.0)
        # Position 1: mean([1.0, 0.0]) = 0.5
        assert out[1] == pytest.approx(0.5)
        # Position 2: mean([1.0, 0.0, 0.0]) = 1/3
        assert out[2] == pytest.approx(1.0 / 3.0, abs=1e-5)
        # Position 3: mean([0.0, 0.0, 0.0]) = 0
        assert out[3] == pytest.approx(0.0)

    def test_empty_array(self):
        out = _causal_smooth(np.array([], dtype=np.float32), 5)
        assert len(out) == 0


# --------------------------------------------------------------------------- #
# _hysteresis_events                                                            #
# --------------------------------------------------------------------------- #

class TestHysteresisEvents:
    def test_single_on_off_cycle(self):
        # Step function: stays low then high then low
        probs = np.array([0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2], dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=0,
                                  hs_key="left_hs", to_key="left_to")
        assert len(evs["left_hs"]) == 1, "Should detect exactly one HS"
        assert len(evs["left_to"]) == 1, "Should detect exactly one TO"
        assert evs["left_hs"][0]["frame"] == 2
        assert evs["left_to"][0]["frame"] == 5

    def test_hysteresis_prevents_oscillation(self):
        # Oscillate around 0.5 — should not fire because never reaches threshold_on
        probs = np.full(20, 0.5, dtype=np.float32)
        evs = _hysteresis_events(probs, fps=30.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=2,
                                  hs_key="left_hs", to_key="left_to")
        assert evs["left_hs"] == []
        assert evs["left_to"] == []

    def test_min_gap_debounce(self):
        # Two rapid transitions within min_gap should collapse to one event
        probs = np.array([0.2, 0.7, 0.3, 0.7, 0.3], dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=3,
                                  hs_key="left_hs", to_key="left_to")
        # Second HS at frame 3 is within min_gap=3 of first HS at frame 1 → suppressed
        assert len(evs["left_hs"]) == 1

    def test_confidence_at_hs_is_probability(self):
        probs = np.array([0.2, 0.75, 0.75], dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=0,
                                  hs_key="left_hs", to_key="left_to")
        assert evs["left_hs"][0]["confidence"] == pytest.approx(0.75, abs=0.01)

    def test_confidence_at_to_is_one_minus_probability(self):
        # TO confidence = 1 - prob at transition
        probs = np.array([0.9, 0.9, 0.3], dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=0,
                                  hs_key="left_hs", to_key="left_to")
        assert len(evs["left_to"]) == 1
        assert evs["left_to"][0]["confidence"] == pytest.approx(1.0 - 0.3, abs=0.01)

    def test_empty_probs(self):
        evs = _hysteresis_events(np.array([], dtype=np.float32), fps=30.0,
                                  threshold_on=0.6, threshold_off=0.4, min_gap=2,
                                  hs_key="left_hs", to_key="left_to")
        assert evs == {"left_hs": [], "left_to": []}

    # ── NEW: hysteresis band — no re-triggering ────────────────────────────

    def test_hysteresis_band_no_retrigger(self):
        """Signal that dips into the hysteresis band (0.4 ≤ p < 0.6) but does
        not cross threshold_off should NOT produce a TO event, and when it
        climbs back above threshold_on it should NOT fire a second HS (already
        planted).
        """
        # Timeline: airborne → crosses 0.6 (HS) → dips to 0.5 (in band) →
        #           back to 0.9 → finally drops to 0.1 (TO)
        probs = np.array([0.1, 0.1, 0.8, 0.8, 0.5, 0.5, 0.9, 0.9, 0.1],
                          dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=0,
                                  hs_key="hs", to_key="to")
        assert len(evs["hs"]) == 1, (
            "Signal entering hysteresis band then recovering should not re-trigger HS"
        )
        assert len(evs["to"]) == 1, "Exactly one TO at the final drop"
        assert evs["to"][0]["frame"] == 8

    def test_hysteresis_debounce_combined(self):
        """Rapid oscillations above threshold_on should fire only one HS
        due to debounce, even though the signal never dips below threshold_off."""
        # All values above threshold_on — debounce min_gap=5 suppresses extras
        probs = np.array([0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        evs = _hysteresis_events(probs, fps=1.0, threshold_on=0.6,
                                  threshold_off=0.4, min_gap=5,
                                  hs_key="hs", to_key="to")
        assert len(evs["hs"]) == 1, "Only one HS even with long planted stretch"


# --------------------------------------------------------------------------- #
# decode_contact_events                                                         #
# --------------------------------------------------------------------------- #

class TestDecodeContactEvents:
    def test_output_keys(self):
        probs = _make_contact_probs(90)
        result = decode_contact_events(probs, probs, fps=30.0)
        assert set(result.keys()) == {"left_hs", "left_to", "right_hs", "right_to"}

    def test_event_dict_structure(self):
        probs = _make_contact_probs(90)
        result = decode_contact_events(probs, probs, fps=30.0)
        for key in result:
            for ev in result[key]:
                assert "frame" in ev and isinstance(ev["frame"], int)
                assert "time" in ev and isinstance(ev["time"], float)
                assert "confidence" in ev
                assert 0.0 <= ev["confidence"] <= 1.0

    def test_hysteresis_invariant_warning(self, caplog):
        """threshold_off >= threshold_on should warn and auto-correct."""
        import logging
        probs = _make_contact_probs(60)
        with caplog.at_level(logging.WARNING):
            decode_contact_events(
                probs, probs, fps=30.0,
                config={"threshold_on": 0.5, "threshold_off": 0.6},
            )
        assert any("hysteresis" in r.message.lower() for r in caplog.records)

    def test_all_zeros_no_events(self):
        probs = np.zeros(60, dtype=np.float32)
        result = decode_contact_events(probs, probs, fps=30.0)
        assert all(len(v) == 0 for v in result.values())

    def test_all_ones_no_to_events(self):
        probs = np.ones(60, dtype=np.float32)
        result = decode_contact_events(probs, probs, fps=30.0)
        # Starts planted immediately → no HS from airborne, and no TO
        assert len(result["left_to"]) == 0
        assert len(result["right_to"]) == 0

    def test_config_override(self):
        """Using a wide threshold should suppress events near 0.5."""
        probs = np.full(60, 0.55, dtype=np.float32)
        result = decode_contact_events(
            probs, probs, fps=30.0,
            config={"threshold_on": 0.8, "threshold_off": 0.2},
        )
        assert len(result["left_hs"]) == 0, "0.55 < threshold_on=0.8; should be no HS"


# --------------------------------------------------------------------------- #
# Registry integration and fallback                                             #
# --------------------------------------------------------------------------- #

class TestRegistryAndFallback:
    def test_registered_after_import(self):
        from myogait.events import list_event_methods
        # Import triggers registration
        import myogait.detectors.learned_contact_detector  # noqa: F401
        assert "learned_tcn" in list_event_methods()

    def test_fallback_to_zeni_when_no_weights(self, tmp_path):
        """_detect_learned_tcn must fall back when no weights file exists."""
        import myogait.detectors.learned_contact_detector as lcd

        # Point weights path at non-existent file
        with patch.dict("os.environ", {"MYOGAIT_TCN_WEIGHTS": str(tmp_path / "nonexistent.pt")}):
            frames = _make_frames(60, "left") + _make_frames(60, "right")
            # We expect fallback: zeni needs ankle + hip landmarks
            # Build a minimal data set that zeni can run on
            for f in frames:
                f["landmarks"].setdefault("LEFT_ANKLE",  {"x": 0.3, "y": 0.8, "visibility": 0.9})
                f["landmarks"].setdefault("RIGHT_ANKLE", {"x": 0.7, "y": 0.8, "visibility": 0.9})
                f["landmarks"].setdefault("LEFT_HIP",    {"x": 0.4, "y": 0.4, "visibility": 0.9})
                f["landmarks"].setdefault("RIGHT_HIP",   {"x": 0.6, "y": 0.4, "visibility": 0.9})

            result = lcd._detect_learned_tcn(frames, fps=30.0)

        # Result must have the standard event keys regardless of which path ran
        assert set(result.keys()) == {"left_hs", "left_to", "right_hs", "right_to"}

    def test_get_last_contact_probs_empty_before_inference(self):
        """Before any inference, get_last_contact_probs returns empty dict or arrays."""
        import myogait.detectors.learned_contact_detector as lcd
        # Reset module-level cache
        lcd._last_contact_probs = {}
        probs = get_last_contact_probs()
        assert isinstance(probs, dict)

    def test_configure_updates_active_config(self):
        import myogait.detectors.learned_contact_detector as lcd
        original = lcd._active_config.copy()
        configure({"threshold_on": 0.75, "window_size": 32})
        assert lcd._active_config["threshold_on"] == pytest.approx(0.75)
        assert lcd._active_config["window_size"] == 32
        # Restore
        lcd._active_config = original


# --------------------------------------------------------------------------- #
# TCN model (only run when torch is available)                                  #
# --------------------------------------------------------------------------- #

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_torch = pytest.mark.skipif(not _torch_available(), reason="torch not installed")


@_skip_no_torch
class TestContactTCN:
    def _make_model(self):
        from myogait.detectors.contact_tcn import ContactTCN
        return ContactTCN(n_features=N_FEATURES, channels=16, kernel_size=3, n_dilations=3)

    def test_output_shape(self):
        import torch
        model = self._make_model()
        x = torch.randn(2, 24, N_FEATURES)   # (batch, time, features)
        out = model(x)
        assert out.shape == (2, 24), f"Expected (2, 24), got {out.shape}"

    def test_output_is_logits_not_probs(self):
        """Model must output raw logits — values outside [0,1] are expected."""
        import torch
        model = self._make_model()
        x = torch.randn(1, 24, N_FEATURES) * 10  # large inputs
        out = model(x)
        # At least some logits should be outside [0,1]
        assert bool((out.abs() > 1).any()), "Model output looks like probabilities, not logits"

    def test_causality(self):
        """Changing a future frame must not affect the output of past frames."""
        import torch
        model = self._make_model()
        model.eval()
        T = 16
        x = torch.randn(1, T, N_FEATURES)
        out_orig = model(x).detach().clone()

        # Modify frame T-1 (last frame)
        x_mod = x.clone()
        x_mod[0, T - 1, :] += 100.0
        out_mod = model(x_mod).detach()

        # All frames except the last should be identical (causal network)
        np.testing.assert_allclose(
            out_orig[0, :-1].numpy(),
            out_mod[0, :-1].numpy(),
            rtol=1e-4, atol=1e-5,
            err_msg="Modifying a future frame changed past outputs (non-causal!)",
        )

    def test_param_count_is_small(self):
        model = self._make_model()
        # With channels=16, should be well under 50k parameters
        assert model.param_count() < 50_000, (
            f"Model has {model.param_count()} params; expected < 50k for CPU inference"
        )

    def test_receptive_field(self):
        model = self._make_model()
        # kernel=3, dilations=1,2,4; rf = 2*(1+2+4)+1 = 15
        assert model.receptive_field == 15

    def test_receptive_field_wider(self):
        """n_dilations=5 should give RF=63."""
        from myogait.detectors.contact_tcn import ContactTCN
        model = ContactTCN(n_features=N_FEATURES, channels=16, kernel_size=3, n_dilations=5)
        assert model.receptive_field == 63, (
            f"Expected RF=63 for n_dilations=5, got {model.receptive_field}"
        )

    # ── NEW: checkpoint save / load roundtrip ─────────────────────────────

    def test_checkpoint_roundtrip(self, tmp_path):
        """Model saved to disk and reloaded must produce bit-identical outputs."""
        import torch
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import FEATURE_SCHEMA

        model = self._make_model()
        model.eval()
        x = torch.randn(2, 10, N_FEATURES)
        out_before = model(x).detach().clone()

        ckpt_path = tmp_path / "roundtrip_tcn.pt"
        torch.save({
            "state_dict":     model.state_dict(),
            "n_features":     N_FEATURES,
            "channels":       16,
            "kernel_size":    3,
            "n_dilations":    3,
            "feature_schema": list(FEATURE_SCHEMA),
            "val_f1":         0.99,
            "epoch":          42,
        }, str(ckpt_path))

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        # Verify metadata round-trips correctly
        assert checkpoint["n_features"] == N_FEATURES
        assert checkpoint["feature_schema"] == list(FEATURE_SCHEMA)
        assert checkpoint["val_f1"] == pytest.approx(0.99)
        assert checkpoint["epoch"] == 42

        model2 = ContactTCN(
            n_features=checkpoint["n_features"],
            channels=checkpoint["channels"],
            kernel_size=checkpoint["kernel_size"],
            n_dilations=checkpoint["n_dilations"],
        )
        model2.load_state_dict(checkpoint["state_dict"])
        model2.eval()

        out_after = model2(x).detach()
        np.testing.assert_allclose(
            out_before.numpy(), out_after.numpy(),
            rtol=1e-5, atol=1e-6,
            err_msg="Checkpoint roundtrip changed model outputs",
        )

    def test_checkpoint_schema_mismatch_warns(self, tmp_path, caplog):
        """Checkpoint with a mismatched feature_schema should log a warning."""
        import logging
        import torch
        from myogait.detectors.learned_contact_detector import _validate_checkpoint_schema

        fake_checkpoint = {"feature_schema": ["wrong_col"] * N_FEATURES}
        with caplog.at_level(logging.WARNING):
            _validate_checkpoint_schema(fake_checkpoint, tmp_path / "fake.pt")
        assert any("schema mismatch" in r.message.lower() for r in caplog.records)


@_skip_no_torch
class TestInferContactProbs:
    def test_output_shape(self):
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import infer_contact_probs
        model = ContactTCN(n_features=N_FEATURES, channels=8)
        feats = np.random.rand(45, N_FEATURES).astype(np.float32)
        probs = infer_contact_probs(feats, model)
        assert probs.shape == (45,), f"Expected (45,), got {probs.shape}"

    def test_output_range(self):
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import infer_contact_probs
        model = ContactTCN(n_features=N_FEATURES, channels=8)
        feats = np.random.rand(30, N_FEATURES).astype(np.float32)
        probs = infer_contact_probs(feats, model)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_nan_input_raises(self):
        """NaN in feature matrix must raise ValueError, not silently corrupt."""
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import infer_contact_probs
        model = ContactTCN(n_features=N_FEATURES, channels=8)
        feats = np.random.rand(20, N_FEATURES).astype(np.float32)
        feats[5, 3] = float("nan")  # inject one NaN
        with pytest.raises(ValueError, match="NaN"):
            infer_contact_probs(feats, model)

    def test_empty_feats(self):
        from myogait.detectors.contact_tcn import ContactTCN
        from myogait.detectors.learned_contact_detector import infer_contact_probs
        model = ContactTCN(n_features=N_FEATURES, channels=8)
        probs = infer_contact_probs(np.array([]).reshape(0, N_FEATURES), model)
        assert len(probs) == 0
