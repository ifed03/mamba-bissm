import importlib
import sys
import types

import numpy as np
import pytest

from data.noise_injection import (
    VALID_NOISE_TYPES,
    inject_noise_at_snr,
    load_noise_record,
    resample_noise,
    scale_noise_to_snr,
    select_noise_segment,
)


def test_valid_noise_types_are_supported():
    assert VALID_NOISE_TYPES == {"bw", "em", "ma"}


@pytest.mark.parametrize("noise_type", ["foo", "118e24", "119e00"])
def test_invalid_noise_type_raises(noise_type):
    with pytest.raises(ValueError, match="Invalid noise_type"):
        inject_noise_at_snr(
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            0.0,
            noise_type=noise_type,
            noise_channel=0,
            seed=1,
            noise_start_index=0,
            noise_original_fs=360,
            target_fs=100,
        )


def test_non_numeric_snr_raises():
    with pytest.raises(ValueError, match="snr_db must be a real numeric scalar"):
        scale_noise_to_snr(np.array([1.0, 2.0]), np.array([0.1, 0.2]), "bad")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_snr_raises(bad):
    with pytest.raises(ValueError, match="snr_db must be finite"):
        scale_noise_to_snr(np.array([1.0, 2.0]), np.array([0.1, 0.2]), bad)


def test_negative_finite_snr_is_accepted():
    _, meta = scale_noise_to_snr(np.array([1.0, -1.0]), np.array([0.1, -0.1]), -6)
    assert np.isclose(meta["snr_db"], -6.0)


def test_mocked_wfdb_loader_returns_expected_values(monkeypatch):
    class Rec:
        p_signal = np.ones((8, 2), dtype=np.float64)
        fs = 360
        sig_name = ["n0", "n1"]

    fake_mod = types.SimpleNamespace(rdrecord=lambda _: Rec())
    monkeypatch.setitem(sys.modules, "wfdb", fake_mod)

    noise, fs, names = load_noise_record("/tmp/noise", "bw")
    assert noise.shape == (8, 2)
    assert fs == 360.0
    assert names == ["n0", "n1"]


def test_channel_selection_by_integer_index_works():
    noise = np.column_stack([np.arange(10), np.arange(10) + 100]).astype(np.float64)
    seg, ch, start = select_noise_segment(noise, target_length=5, seed=0, channel=1)
    assert ch == 1
    assert start >= 0
    assert np.all(seg >= 100)


def test_invalid_channel_index_raises():
    noise = np.ones((10, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="out of range"):
        select_noise_segment(noise, target_length=4, seed=0, channel=2)


def test_channel_selection_is_deterministic_with_seed():
    noise = np.column_stack([np.arange(20), np.arange(20) + 100]).astype(np.float64)
    a = select_noise_segment(noise, target_length=6, seed=42, channel=None)
    b = select_noise_segment(noise, target_length=6, seed=42, channel=None)
    assert a[1] == b[1]
    assert a[2] == b[2]
    assert np.array_equal(a[0], b[0])


def test_resampling_length_is_approximate_expected():
    noise = np.ones((650, 2), dtype=np.float64)
    out = resample_noise(noise, original_fs=360, target_fs=100)
    expected = int(round(650 * 100 / 360))
    assert abs(out.shape[0] - expected) <= 1
    assert out.shape[1] == 2


def test_crop_tile_logic_returns_exact_length_for_long_short_equal():
    equal_noise = np.ones((10, 2), dtype=np.float64)
    seg_eq, _, _ = select_noise_segment(equal_noise, target_length=10, seed=1, channel=0)
    assert seg_eq.shape == (10,)

    long_noise = np.ones((20, 2), dtype=np.float64)
    seg_long, _, _ = select_noise_segment(long_noise, target_length=10, seed=1, channel=0)
    assert seg_long.shape == (10,)

    short_noise = np.ones((4, 2), dtype=np.float64)
    seg_short, _, start_short = select_noise_segment(short_noise, target_length=10, seed=1, channel=0)
    assert seg_short.shape == (10,)
    assert start_short == 0


def test_injection_output_shape_matches_input_and_input_unchanged():
    clean = np.linspace(0, 1, 100, dtype=np.float64)
    clean_copy = clean.copy()
    noise = np.sin(np.linspace(0, 2 * np.pi, 100))
    noisy, _ = inject_noise_at_snr(
        clean,
        noise,
        0.0,
        noise_type="bw",
        noise_channel=0,
        seed=1,
        noise_start_index=10,
        noise_original_fs=360,
        target_fs=100,
    )
    assert noisy.shape == clean.shape
    assert np.array_equal(clean, clean_copy)


def test_snr_scaling_numerically_correct():
    clean = np.array([1.0, -1.0, 1.0, -1.0])
    noise = np.array([1.0, -1.0, 0.5, -0.5])
    scaled, meta = scale_noise_to_snr(clean, noise, snr_db=6.0)
    measured = 10 * np.log10(np.mean(clean**2) / np.mean(scaled**2))
    assert np.isclose(meta["measured_snr_db"], 6.0, atol=1e-8)
    assert np.isclose(measured, 6.0, atol=1e-8)


def test_repeated_injection_with_same_seed_identical_outputs_and_metadata():
    clean = np.linspace(-1, 1, 200, dtype=np.float64)
    noise2d = np.column_stack([np.arange(500), np.arange(500) + 100]).astype(np.float64)
    seg1, ch1, st1 = select_noise_segment(noise2d, 200, seed=12, channel=None)
    seg2, ch2, st2 = select_noise_segment(noise2d, 200, seed=12, channel=None)
    out1, m1 = inject_noise_at_snr(clean, seg1, -6, noise_type="em", noise_channel=ch1, seed=12, noise_start_index=st1, noise_original_fs=360, target_fs=100)
    out2, m2 = inject_noise_at_snr(clean, seg2, -6, noise_type="em", noise_channel=ch2, seed=12, noise_start_index=st2, noise_original_fs=360, target_fs=100)
    assert np.array_equal(out1, out2)
    assert m1 == m2


def test_different_seeds_can_produce_different_start_indices():
    noise = np.column_stack([np.arange(200), np.arange(200) + 1]).astype(np.float64)
    _, _, s1 = select_noise_segment(noise, target_length=50, seed=1, channel=0)
    _, _, s2 = select_noise_segment(noise, target_length=50, seed=2, channel=0)
    assert s1 != s2


def test_zero_clean_power_raises():
    with pytest.raises(ValueError, match="clean ECG power is zero or near-zero"):
        scale_noise_to_snr(np.zeros(10), np.arange(10), 0)


def test_zero_raw_noise_power_raises():
    with pytest.raises(ValueError, match="raw noise power is zero or near-zero"):
        scale_noise_to_snr(np.arange(10), np.ones(10), 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_ecg_raises(bad):
    clean = np.array([0.0, 1.0, bad])
    noise = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="clean_ecg must contain only finite values"):
        scale_noise_to_snr(clean, noise, 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_noise_raises(bad):
    clean = np.array([1.0, 2.0, 3.0])
    noise = np.array([0.0, 1.0, bad])
    with pytest.raises(ValueError, match="noise_segment must contain only finite values"):
        scale_noise_to_snr(clean, noise, 0)


def test_missing_wfdb_raises_only_when_loading(monkeypatch):
    monkeypatch.delitem(sys.modules, "wfdb", raising=False)

    real_import = importlib.import_module

    def _guarded_import(name, package=None):
        if name == "wfdb":
            raise ImportError("wfdb unavailable")
        return real_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _guarded_import)
    import builtins
    real_import_builtin = builtins.__import__

    def _guarded_builtin_import(name, *args, **kwargs):
        if name == "wfdb":
            raise ImportError("wfdb unavailable")
        return real_import_builtin(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_builtin_import)

    # Synthetic functions still work.
    scale_noise_to_snr(np.array([1.0, -1.0]), np.array([1.0, -1.0]), 0.0)

    with pytest.raises(ImportError, match="wfdb is required"):
        load_noise_record("/tmp/noise", "ma")


def test_resample_360_to_100_uses_expected_rational_ratio(monkeypatch):
    from data import noise_injection as ni

    captured = {}

    def _fake_resample_poly(arr, up, down, axis):
        captured["up"] = up
        captured["down"] = down
        captured["axis"] = axis
        return arr

    monkeypatch.setattr(ni, "resample_poly", _fake_resample_poly)
    noise = np.ones((100, 2), dtype=np.float64)
    out = ni.resample_noise(noise, original_fs=360, target_fs=100)

    assert np.array_equal(out, noise)
    assert captured["up"] == 5
    assert captured["down"] == 18
    assert captured["axis"] == 0
