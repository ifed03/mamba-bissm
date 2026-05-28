"""Low-level NSTDB noise injection utilities.

This module implements the low-level operation for injecting MIT-BIH NSTDB raw
noise into single-lead ECG signals. It is intentionally independent of split
logic (train/val/test policy) and should be used in the preprocessing pipeline
*after ECG resampling* and *before window extraction and per-window z-score
normalisation*.

Only raw NSTDB noise records are supported:
- ``bw``: baseline wander
- ``em``: electrode motion
- ``ma``: muscle artefact

The ``118eXX`` and ``119eXX`` records must *not* be used as raw noise sources
because those are already noisy benchmark ECG examples, not raw standalone
noise channels for injection.

SNR scaling follows:
- signal power: mean(x**2)
- raw noise power: mean(n**2), after demeaning n
- target noise power: signal_power / (10 ** (snr_db / 10))
- scale factor: sqrt(target_noise_power / raw_noise_power)
- noisy ECG: x + scale_factor * n

Negative SNR values are valid as long as they are finite. Reproducibility is
controlled via deterministic RNG using a provided integer seed.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

VALID_NOISE_TYPES = {"bw", "em", "ma"}
_EPS = 1e-12


def _validate_noise_type(noise_type: str) -> str:
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Invalid noise_type '{noise_type}'. Expected one of {sorted(VALID_NOISE_TYPES)}.")
    return noise_type


def _validate_finite_real_scalar(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric scalar; got {value!r}.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out!r}.")
    return out


def _as_finite_1d_float(x: np.ndarray | list[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def load_noise_record(noise_root: str | Path = "data", noise_type: str = "bw"):
    """Load an NSTDB raw noise record using WFDB.

    Parameters
    ----------
    noise_root:
        Root directory containing exact raw NSTDB ``bw.*``, ``em.*``, and ``ma.*`` WFDB files. Defaults to ``data``.
    noise_type:
        One of ``bw``, ``em``, ``ma``.

    Returns
    -------
    noise : np.ndarray
        2D noise array of shape (num_samples, num_channels).
    fs : float
        Source sampling rate from the WFDB header.
    sig_names : list[str]
        Signal/channel names from the WFDB header.
    """

    noise_type = _validate_noise_type(noise_type)
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required to load NSTDB WFDB records (.hea/.dat). "
            "Install wfdb to use load_noise_record."
        ) from exc

    record_path = str(Path(noise_root) / noise_type)
    record = wfdb.rdrecord(record_path)
    noise = np.asarray(record.p_signal, dtype=np.float64)
    if noise.ndim != 2 or noise.shape[1] < 1:
        raise ValueError(f"Loaded noise record must be 2D with >=1 channel; got shape {noise.shape}.")
    if not np.all(np.isfinite(noise)):
        raise ValueError("Loaded noise record contains non-finite values.")
    fs = float(record.fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Loaded sampling frequency must be positive finite; got {record.fs!r}.")
    sig_names = list(getattr(record, "sig_name", []))
    return noise, fs, sig_names


def resample_noise(noise: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Resample 2D noise [time, channels] to target sampling rate with polyphase FIR.

    Uses ``scipy.signal.resample_poly`` so downsampling includes anti-aliasing
    low-pass filtering. For example, 360 Hz -> 100 Hz uses the exact rational
    factor 5/18.
    """
    original_fs = _validate_finite_real_scalar(original_fs, "original_fs")
    target_fs = _validate_finite_real_scalar(target_fs, "target_fs")
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("original_fs and target_fs must be > 0.")

    arr = np.asarray(noise, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"noise must be 2D [time, channels]; got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("noise must be non-empty with at least one channel.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("noise must contain only finite values.")

    if np.isclose(original_fs, target_fs):
        return arr.copy()

    ratio = Fraction(target_fs / original_fs).limit_denominator(10_000)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(arr, up=up, down=down, axis=0)
    return np.asarray(resampled, dtype=np.float64)


def select_noise_segment(noise: np.ndarray, target_length: int, seed: int, channel: int | None = None):
    arr = np.asarray(noise, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"noise must be 2D [time, channels]; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("noise must contain only finite values.")
    if target_length <= 0:
        raise ValueError(f"target_length must be positive; got {target_length}.")

    rng = np.random.default_rng(seed)
    num_samples, num_channels = arr.shape

    if channel is None:
        ch = int(rng.integers(0, num_channels))
    else:
        if not isinstance(channel, (int, np.integer)):
            raise ValueError(f"channel must be an integer index or None; got {channel!r}.")
        ch = int(channel)
        if ch < 0 or ch >= num_channels:
            raise ValueError(f"channel index {ch} out of range for {num_channels} channels.")

    ch_sig = arr[:, ch]

    if num_samples >= target_length:
        start = int(rng.integers(0, num_samples - target_length + 1))
        segment = ch_sig[start : start + target_length].copy()
    else:
        start = 0
        reps = int(np.ceil(target_length / num_samples))
        segment = np.tile(ch_sig, reps)[:target_length].copy()

    return segment, ch, start


def scale_noise_to_snr(clean_ecg: np.ndarray | list[float], noise_segment: np.ndarray | list[float], snr_db: float):
    snr_db = _validate_finite_real_scalar(snr_db, "snr_db")
    x = _as_finite_1d_float(clean_ecg, "clean_ecg")
    n = _as_finite_1d_float(noise_segment, "noise_segment")
    if x.shape != n.shape:
        raise ValueError(f"clean_ecg and noise_segment must have same shape; got {x.shape} vs {n.shape}.")

    n = n - n.mean()
    signal_power = float(np.mean(x**2))
    raw_noise_power = float(np.mean(n**2))

    if signal_power <= _EPS:
        raise ValueError(f"clean ECG power is zero or near-zero ({signal_power}).")
    if raw_noise_power <= _EPS:
        raise ValueError(f"raw noise power is zero or near-zero ({raw_noise_power}).")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale_factor = float(np.sqrt(target_noise_power / raw_noise_power))
    scaled_noise = n * scale_factor
    scaled_noise_power = float(np.mean(scaled_noise**2))
    measured_snr_db = float(10.0 * np.log10(signal_power / scaled_noise_power))

    metadata = {
        "snr_db": snr_db,
        "scale_factor": scale_factor,
        "signal_power": signal_power,
        "raw_noise_power": raw_noise_power,
        "scaled_noise_power": scaled_noise_power,
        "measured_snr_db": measured_snr_db,
    }
    return scaled_noise.astype(np.float64), metadata


def inject_noise_at_snr(
    clean_ecg: np.ndarray | list[float],
    noise_segment: np.ndarray | list[float],
    snr_db: float,
    *,
    noise_type: str,
    noise_channel: int,
    seed: int,
    noise_start_index: int,
    noise_original_fs: float,
    target_fs: float,
):
    _validate_noise_type(noise_type)
    scaled_noise, snr_meta = scale_noise_to_snr(clean_ecg, noise_segment, snr_db)
    x = _as_finite_1d_float(clean_ecg, "clean_ecg")
    noisy = x + scaled_noise
    metadata = {
        "noise_type": noise_type,
        "snr_db": float(snr_meta["snr_db"]),
        "noise_channel": int(noise_channel),
        "seed": int(seed),
        "noise_start_index": int(noise_start_index),
        "noise_original_fs": float(_validate_finite_real_scalar(noise_original_fs, "noise_original_fs")),
        "target_fs": float(_validate_finite_real_scalar(target_fs, "target_fs")),
        "scale_factor": float(snr_meta["scale_factor"]),
        "signal_power": float(snr_meta["signal_power"]),
        "raw_noise_power": float(snr_meta["raw_noise_power"]),
        "scaled_noise_power": float(snr_meta["scaled_noise_power"]),
        "measured_snr_db": float(snr_meta["measured_snr_db"]),
    }
    return noisy.astype(np.float64), metadata
