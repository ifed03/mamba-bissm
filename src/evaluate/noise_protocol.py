"""Protocol helpers for controlled NSTDB noise evaluation and training.

Zero-shot noisy-test evaluation remains test-only and uses clean validation for
checkpoint and threshold selection. Noisy-input training/evaluation uses noisy
train, validation, and test splits; noisy validation drives checkpoint and
threshold selection. In both protocols, noise is applied to the already
resampled ECG signal before window extraction and per-window normalisation in
:mod:`data.parquet_dataset`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.noise_injection import (
    VALID_NOISE_TYPES,
    inject_noise_at_snr,
    load_noise_record,
    resample_noise,
    select_noise_segment,
)

DEFAULT_NOISE_ROOT = Path("data")
DEFAULT_SNR_DB = [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]
REQUIRED_NSTDB_FILES = ("bw.hea", "bw.dat", "em.hea", "em.dat", "ma.hea", "ma.dat")
ZERO_SHOT_EVAL_NAME = "zero-shot"
NOISY_INPUT_TRAINING_NAME = "noisy-input-training"
NOISY_INPUT_THRESHOLD_SOURCE = "noisy_val"
NOISY_INPUT_CHECKPOINT_SOURCE = "noisy_val"


@dataclass(frozen=True)
class NoiseCondition:
    noise_type: str
    snr_db: float

    def __post_init__(self) -> None:
        validate_noise_type(self.noise_type)
        if not np.isfinite(float(self.snr_db)):
            raise ValueError(f"snr_db must be finite; got {self.snr_db!r}.")


def validate_noise_type(noise_type: str) -> str:
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Invalid noise type {noise_type!r}. Expected one of {sorted(VALID_NOISE_TYPES)}.")
    return noise_type


def validate_nstdb_root(noise_root: str | Path) -> Path:
    root = Path(noise_root)
    if not root.exists():
        raise FileNotFoundError(f"NSTDB noise_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"NSTDB noise_root is not a directory: {root}")
    missing = [name for name in REQUIRED_NSTDB_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"NSTDB noise_root {root} is missing required raw noise files: {', '.join(missing)}"
        )
    return root


def condition_key(condition: NoiseCondition) -> str:
    snr = f"{condition.snr_db:g}".replace("-", "neg")
    return f"noise_type={condition.noise_type}__snr_db={snr}"


def metrics_filename(condition: NoiseCondition) -> str:
    return f"metrics_{ZERO_SHOT_EVAL_NAME}_{condition_key(condition)}.json"


def noisy_input_condition_name(condition: NoiseCondition) -> str:
    snr = f"{condition.snr_db:g}".replace("-", "neg")
    return f"noisy_input_training_{condition.noise_type}_{snr}dB"


def noisy_input_metrics_filename(condition: NoiseCondition) -> str:
    return f"metrics_{NOISY_INPUT_TRAINING_NAME}_{condition_key(condition)}.json"


def noisy_input_threshold_filename(condition: NoiseCondition) -> str:
    return f"threshold_{NOISY_INPUT_TRAINING_NAME}_{condition_key(condition)}.json"


def ensure_clean_split(split_name: str) -> None:
    if split_name != "test":
        raise ValueError(f"Noise injection is only allowed for test split, got split={split_name!r}.")


def deterministic_example_seed(
    *, base_seed: int, record_id: str, split: str, noise_type: str, snr_db: float
) -> int:
    validate_noise_type(noise_type)
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "record_id": str(record_id),
            "split": split,
            "noise_type": noise_type,
            "snr_db": float(snr_db),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def metadata_for_noisy_example(
    *,
    base_metadata: dict[str, Any],
    split: str,
    condition: NoiseCondition,
    threshold_source: str,
    checkpoint_source: str,
) -> dict[str, Any]:
    ensure_clean_split(split)
    if threshold_source != "clean_val":
        raise ValueError("tau* must come from clean validation only.")
    if checkpoint_source != "clean_val":
        raise ValueError("checkpoint selection must come from clean validation only.")

    out = dict(base_metadata)
    out.update(
        {
            "original_record_id": base_metadata.get("original_record_id", base_metadata.get("record_id")),
            "split": split,
            "noise_type": condition.noise_type,
            "snr_db": float(condition.snr_db),
            "threshold_source": threshold_source,
            "checkpoint_source": checkpoint_source,
        }
    )
    return out


def condition_output_dir(root: str | Path, condition: NoiseCondition) -> Path:
    return Path(root) / condition_key(condition)


class _BaseDeterministicNoiseInjector:
    """Deterministic ECG noise injector applied after ECG resampling."""

    threshold_source = "unspecified"
    checkpoint_source = "unspecified"

    def __init__(
        self,
        *,
        noise_type: str,
        snr_db: float,
        base_seed: int = 123,
        target_fs: float = 100,
        noise_root: str | Path = DEFAULT_NOISE_ROOT,
        noise: np.ndarray | None = None,
        noise_fs: float | None = None,
    ) -> None:
        self.condition = NoiseCondition(validate_noise_type(noise_type), float(snr_db))
        self.base_seed = int(base_seed)
        self.target_fs = float(target_fs)
        self.noise_root = Path(noise_root)
        if noise is None:
            validate_nstdb_root(self.noise_root)
            noise, source_fs, _ = load_noise_record(self.noise_root, self.condition.noise_type)
        else:
            if noise_fs is None:
                raise ValueError("noise_fs is required when synthetic noise is provided.")
            source_fs = float(noise_fs)
        self.noise_original_fs = float(source_fs)
        self.noise = resample_noise(np.asarray(noise, dtype=np.float64), self.noise_original_fs, self.target_fs)

    def _validate_split(self, split: str) -> None:
        return None

    def inject(self, clean_resampled_ecg: np.ndarray, *, record_id: str, split: str = "test"):
        self._validate_split(split)
        seed = deterministic_example_seed(
            base_seed=self.base_seed,
            record_id=str(record_id),
            split=split,
            noise_type=self.condition.noise_type,
            snr_db=self.condition.snr_db,
        )
        segment, channel, start = select_noise_segment(self.noise, len(clean_resampled_ecg), seed=seed, channel=None)
        noisy, noise_meta = inject_noise_at_snr(
            clean_resampled_ecg,
            segment,
            self.condition.snr_db,
            noise_type=self.condition.noise_type,
            noise_channel=channel,
            seed=seed,
            noise_start_index=start,
            noise_original_fs=self.noise_original_fs,
            target_fs=self.target_fs,
        )
        metadata = {
            "record_id": str(record_id),
            "original_record_id": str(record_id),
            "split": split,
            "noise_type": self.condition.noise_type,
            "snr_db": float(self.condition.snr_db),
            "seed": int(seed),
            "noise_channel": int(channel),
            "noise_start_index": int(start),
            "measured_snr_db": float(noise_meta["measured_snr_db"]),
            "threshold_source": self.threshold_source,
            "checkpoint_source": self.checkpoint_source,
        }
        metadata.update(noise_meta)
        metadata["split"] = split
        metadata["original_record_id"] = str(record_id)
        return noisy, metadata


class ZeroShotNoiseInjector(_BaseDeterministicNoiseInjector):
    """Deterministic ECG noise injector for the test split only."""

    threshold_source = "clean_val"
    checkpoint_source = "clean_val"

    def _validate_split(self, split: str) -> None:
        ensure_clean_split(split)


class NoisyInputNoiseInjector(_BaseDeterministicNoiseInjector):
    """Deterministic ECG noise injector for noisy-input train/val/test runs."""

    threshold_source = NOISY_INPUT_THRESHOLD_SOURCE
    checkpoint_source = NOISY_INPUT_CHECKPOINT_SOURCE

    def _validate_split(self, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"noisy-input noise injection requires split to be train, val, or test; got {split!r}.")


def load_clean_threshold(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text())
    if payload.get("threshold_source", "clean_val") != "clean_val":
        raise ValueError("clean threshold metadata must have threshold_source='clean_val'.")
    if "threshold" not in payload:
        raise ValueError(f"clean threshold metadata is missing 'threshold': {path}")
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"clean validation threshold must be finite; got {threshold!r}.")
    return threshold
