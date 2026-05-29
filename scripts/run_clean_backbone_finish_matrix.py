#!/usr/bin/env python
"""Run the clean-data backbone completion matrix in one command."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


WINDOWS = (4, 6, 8, 10)
GENERATED_CONFIG_DIR = Path("configs/generated_clean_backbone_finish")


def mamba_backbone_configs() -> list[str]:
    return [
        str(
            GENERATED_CONFIG_DIR
            / f"binary_mamba_d128_n4_s64_slowpath_fp32_lr1e-4_100hz_win{w}s_stride2s.yaml"
        )
        for w in WINDOWS
    ]


def bimamba_backbone_configs() -> list[str]:
    return [
        str(
            GENERATED_CONFIG_DIR
            / f"binary_bimamba_d128_n2_s64_slowpath_fp32_lr1e-3_100hz_win{w}s_stride2s.yaml"
        )
        for w in WINDOWS
    ]


def ecgmamba_bilstm_backbone_configs() -> list[str]:
    return [
        str(
            GENERATED_CONFIG_DIR
            / f"binary_ecgmamba_bilstm_d128_h64_n2_fp32_100hz_win{w}s_stride2s.yaml"
        )
        for w in WINDOWS
    ]


def standalone_bilstm_configs() -> list[str]:
    return [f"configs/binary_bilstm_100hz_win{w}s_stride2s.yaml" for w in WINDOWS]


def standalone_cnn1d_configs() -> list[str]:
    return [
        f"configs/binary_cnn1d_c256_n3_k7_100hz_win{w}s_stride2s.yaml"
        for w in WINDOWS
    ]


def clean_backbone_finish_configs() -> list[str]:
    return (
        mamba_backbone_configs()
        + bimamba_backbone_configs()
        + ecgmamba_bilstm_backbone_configs()
        + standalone_bilstm_configs()
        + standalone_cnn1d_configs()
    )


def _default_batch_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _mamba_template_path(window_seconds: int) -> Path:
    return Path(f"configs/binary_mamba_d64_n4_s16_100hz_win{window_seconds}s_stride2s.yaml")


def _ecgmamba_bilstm_template_path(window_seconds: int) -> Path:
    return Path(f"configs/binary_ecgmamba_bilstm_d64_n2_100hz_win{window_seconds}s_stride2s.yaml")


def prepare_generated_configs() -> None:
    """Materialize requested ECGMamba backbone configs for each window length."""
    for idx, window_seconds in enumerate(WINDOWS):
        template_path = _mamba_template_path(window_seconds)
        if not template_path.exists():
            raise FileNotFoundError(f"Missing template config: {template_path}")

        mamba_cfg = _load_yaml(template_path)
        mamba_cfg["model"].update(
            {
                "name": "mamba",
                "backbone": "mamba",
                "d_model": 128,
                "n_layers": 4,
                "d_state": 64,
                "use_fast_path": False,
            }
        )
        mamba_cfg["training"]["lr"] = 1e-4
        mamba_cfg["training"]["mixed_precision"] = False
        _write_yaml(Path(mamba_backbone_configs()[idx]), mamba_cfg)

        bimamba_cfg = _load_yaml(template_path)
        bimamba_cfg["model"].update(
            {
                "name": "bimamba",
                "backbone": "bimamba",
                "d_model": 128,
                "n_layers": 2,
                "d_state": 64,
                "use_fast_path": False,
            }
        )
        bimamba_cfg["training"]["lr"] = 1e-3
        bimamba_cfg["training"]["mixed_precision"] = False
        _write_yaml(Path(bimamba_backbone_configs()[idx]), bimamba_cfg)

        bilstm_template_path = _ecgmamba_bilstm_template_path(window_seconds)
        if not bilstm_template_path.exists():
            raise FileNotFoundError(f"Missing template config: {bilstm_template_path}")

        bilstm_cfg = _load_yaml(bilstm_template_path)
        bilstm_cfg["model"].update(
            {
                "name": "ecgmamba",
                "backbone": "bilstm",
                "d_model": 128,
                "n_layers": 2,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "lstm_bidirectional": True,
            }
        )
        bilstm_cfg["training"]["mixed_precision"] = False
        _write_yaml(Path(ecgmamba_bilstm_backbone_configs()[idx]), bilstm_cfg)


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
    p.add_argument(
        "--python",
        default="python",
        help="Python executable used to launch scripts/train_model.py.",
    )
    args = p.parse_args()

    prepare_generated_configs()

    config_groups = [
        ("ecgmamba_mamba_backbone", mamba_backbone_configs()),
        ("ecgmamba_bimamba_backbone", bimamba_backbone_configs()),
        ("ecgmamba_bilstm_backbone", ecgmamba_bilstm_backbone_configs()),
        ("standalone_bilstm_baseline", standalone_bilstm_configs()),
        ("standalone_cnn1d_baseline", standalone_cnn1d_configs()),
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
                args.python,
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
