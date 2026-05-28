"""Protocol guards for zero-shot noisy-test evaluation.

This module enforces that noisy evaluation is test-only and that model/threshold
selection remains based on clean validation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NoiseCondition:
    noise_type: str
    snr_db: float


def condition_key(condition: NoiseCondition) -> str:
    snr = f"{condition.snr_db:g}".replace("-", "neg")
    return f"noise_type={condition.noise_type}__snr_db={snr}"


def ensure_clean_split(split_name: str) -> None:
    if split_name != "test":
        raise ValueError(f"Noise injection is only allowed for test split, got split={split_name!r}.")


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
