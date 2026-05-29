#!/usr/bin/env python
"""Preflight audits for the clean-data ECGMamba backbone matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_clean_full_matrix import WINDOWS, controlled_ecgmamba_backbone_configs  # noqa: E402


BACKBONE_LABELS = {
    "bissm": "BiSSM",
    "mamba": "Mamba",
    "bimamba": "BiMamba",
    "bilstm": "BiLSTM-backbone",
}


def _config_path(path: str) -> Path:
    return ROOT / path


def _window_seconds(cfg: dict) -> int:
    seconds = cfg["preprocessing"]["windowing"]["window_seconds"]
    return int(seconds)


def _input_samples(cfg: dict) -> int:
    seconds = float(cfg["preprocessing"]["windowing"]["window_seconds"])
    fs = int(cfg["preprocessing"]["fs_target"])
    return int(round(seconds * fs))


def _backbone_label(cfg: dict) -> str:
    backbone = cfg["model"].get("backbone", "bissm")
    return BACKBONE_LABELS.get(str(backbone), str(backbone))


def _parameter_count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _load_config(path: str) -> dict:
    from utils.config import load_config

    cfg = load_config(str(_config_path(path)))
    # CPU audits use reference kernels; this does not alter parameter counts.
    if cfg.get("model", {}).get("backbone") in {"mamba", "bimamba"}:
        cfg["model"]["use_fast_path"] = False
    return cfg


def _build_model(cfg: dict):
    from models import build_model

    return build_model(cfg)


def _encoded_sequence(model, x):
    seq = model._to_sequence(x)
    return model.pos(seq)


def _backbone_output(model, seq):
    if model.backbone_name == "bissm":
        out = seq
        for block in model.backbone:
            out = block(out)
        return out
    return model.backbone(seq)


def run_parameter_count_audit(configs: list[str]) -> list[dict]:
    rows: list[dict] = []
    counts_by_backbone: dict[str, set[int]] = defaultdict(set)
    for path in configs:
        cfg = _load_config(path)
        model = _build_model(cfg)
        total, trainable = _parameter_count(model)
        label = _backbone_label(cfg)
        counts_by_backbone[label].add(total)
        rows.append(
            {
                "window": _window_seconds(cfg),
                "backbone": label,
                "config": path,
                "total": total,
                "trainable": trainable,
            }
        )

    offenders = {
        backbone: sorted(counts)
        for backbone, counts in counts_by_backbone.items()
        if len(counts) != 1
    }
    if offenders:
        raise AssertionError(
            "Parameter count changed across window lengths: "
            + ", ".join(f"{name}={counts}" for name, counts in offenders.items())
        )
    return rows


def run_shape_audit(configs: list[str], batch_size: int, device: str = "auto") -> list[dict]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict] = []
    for path in configs:
        cfg = _load_config(path)
        model = _build_model(cfg).to(device)
        model.eval()

        d_model = int(cfg["model"]["d_model"])
        samples = _input_samples(cfg)
        x = torch.randn(batch_size, 1, samples, device=device)

        with torch.no_grad():
            encoded = _encoded_sequence(model, x)
            backbone_out = _backbone_output(model, encoded)
            pooled = backbone_out.mean(dim=1)
            logits, model_pooled = model(x)

        expected_encoded = (batch_size, encoded.shape[1], d_model)
        assert tuple(x.shape) == (batch_size, 1, samples), (path, tuple(x.shape))
        assert tuple(encoded.shape) == expected_encoded, (path, tuple(encoded.shape))
        assert tuple(backbone_out.shape) == expected_encoded, (
            path,
            tuple(backbone_out.shape),
        )
        assert tuple(pooled.shape) == (batch_size, d_model), (path, tuple(pooled.shape))
        assert tuple(logits.shape) == (batch_size,), (path, tuple(logits.shape))
        assert tuple(model_pooled.shape) == (batch_size, d_model), (
            path,
            tuple(model_pooled.shape),
        )

        rows.append(
            {
                "window": _window_seconds(cfg),
                "backbone": _backbone_label(cfg),
                "config": path,
                "input": tuple(x.shape),
                "encoded": tuple(encoded.shape),
                "backbone_output": tuple(backbone_out.shape),
                "pooled": tuple(pooled.shape),
                "logit": tuple(logits.shape),
            }
        )
    return rows


def save_dry_run_listing(batch_tag: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"clean_full_matrix_dry_run_{batch_tag}.txt"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_clean_full_matrix.py"),
        "--dry-run",
        "--batch-tag",
        batch_tag,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out_path.write_text(result.stdout, encoding="utf-8")
    return out_path


def _print_parameter_rows(rows: list[dict]) -> None:
    print("Parameter-count audit")
    print("window_s,backbone,total_parameters,trainable_parameters,config")
    for row in sorted(rows, key=lambda r: (r["window"], r["backbone"])):
        print(
            f"{row['window']},{row['backbone']},{row['total']},"
            f"{row['trainable']},{row['config']}"
        )


def _print_shape_rows(rows: list[dict]) -> None:
    print("\nShape audit")
    print("window_s,backbone,input,encoded,backbone_output,pooled,logit,config")
    for row in sorted(rows, key=lambda r: (r["window"], r["backbone"])):
        print(
            f"{row['window']},{row['backbone']},{row['input']},{row['encoded']},"
            f"{row['backbone_output']},{row['pooled']},{row['logit']},{row['config']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-tag", default="preflight")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default="audits")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    configs = controlled_ecgmamba_backbone_configs()
    expected = len(WINDOWS) * len(BACKBONE_LABELS)
    if len(configs) != expected:
        raise AssertionError(f"Expected {expected} controlled configs, got {len(configs)}")

    param_rows = run_parameter_count_audit(configs)
    shape_rows = run_shape_audit(configs, batch_size=args.batch_size, device=args.device)
    dry_run_path = save_dry_run_listing(args.batch_tag, ROOT / args.output_dir)

    _print_parameter_rows(param_rows)
    _print_shape_rows(shape_rows)
    print(f"\nDry-run listing saved to {dry_run_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
