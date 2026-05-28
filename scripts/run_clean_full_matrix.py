#!/usr/bin/env python
"""Run the full clean-data AF/NSR experiment matrix in one command."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WINDOWS = (4, 6, 8, 10)


def clean_matrix_configs() -> list[str]:
    configs: list[str] = []
    for w in WINDOWS:
        configs.append(f"configs/binary_bissm_reduced2_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_bissm_reduced4_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_bilstm_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_mamba_2layer_100hz_win{w}s_stride2s.yaml")
    for w in WINDOWS:
        configs.append(f"configs/binary_mamba_4layer_100hz_win{w}s_stride2s.yaml")
    return configs


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

    configs = clean_matrix_configs()
    missing = [c for c in configs if not Path(c).exists()]
    if missing:
        raise FileNotFoundError(f"Missing config(s): {missing}")

    batch_tag = args.batch_tag or _default_batch_tag()
    for idx, cfg in enumerate(configs, start=1):
        stem = Path(cfg).stem
        run_name = f"{stem}__{batch_tag}"
        cmd = ["python", "scripts/train_model.py", "--config", cfg, "--run-name", run_name]
        print(f"[{idx:02d}/{len(configs)}] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
