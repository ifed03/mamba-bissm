#!/usr/bin/env python
"""Run the full clean-data AF/NSR experiment matrix in one command."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WINDOWS = (4, 6, 8, 10)


def controlled_ecgmamba_backbone_configs() -> list[str]:
    """Configs for the controlled ECGMamba backbone comparison."""
    configs: list[str] = []
    for w in WINDOWS:
        configs.append(f"configs/binary_bissm_d64_n2_s64_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_mamba_d64_n2_s16_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_bimamba_d128_n2_s64_slowpath_fp32_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_ecgmamba_bilstm_d64_n2_100hz_win{w}s_stride2s.yaml")
    return configs


def depth_sweep_configs() -> list[str]:
    """Additional depth-sweep configs, separate from the controlled comparison."""
    configs: list[str] = []
    for w in WINDOWS:
        configs.append(f"configs/binary_bissm_d64_n4_s64_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_mamba_d64_n4_s16_100hz_win{w}s_stride2s.yaml")
    return configs


def external_baseline_configs() -> list[str]:
    """Standalone baseline configs kept separate from backbone comparisons."""
    return [f"configs/binary_bilstm_100hz_win{w}s_stride2s.yaml" for w in WINDOWS]


def clean_matrix_configs() -> list[str]:
    return (
        controlled_ecgmamba_backbone_configs()
        + depth_sweep_configs()
        + external_baseline_configs()
    )


def _default_batch_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch-tag",
        default=None,
        help="Optional tag appended to run names for this matrix batch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing training.",
    )
    args = p.parse_args()

    config_groups = [
        ("controlled_ecgmamba_backbone", controlled_ecgmamba_backbone_configs()),
        ("depth_sweep", depth_sweep_configs()),
        ("external_baseline", external_baseline_configs()),
    ]
    configs = [cfg for _, group in config_groups for cfg in group]
    missing = [c for c in configs if not Path(c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing config(s): {missing}")

    batch_tag = args.batch_tag or _default_batch_tag()
    total = len(configs)
    idx = 0
    for group_name, group_configs in config_groups:
        print(f"# {group_name}")
        for cfg in group_configs:
            idx += 1
            stem = Path(cfg).stem
            run_name = f"{stem}__{batch_tag}"
            cmd = [
                "python",
                "scripts/train_model.py",
                "--config",
                cfg,
                "--run-name",
                run_name,
            ]
            print(f"[{idx:02d}/{total}] {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
