#!/usr/bin/env python
"""Profile architecture-level CPU efficiency for completed result runs.

This is intentionally lighter than profile_efficiency_batch.py: it avoids data
loading and record iteration, and times synthetic windows at the configured
input length. That makes it useful for checking that comparable model folders
all have parameter counts and forward-pass efficiency metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SUMMARY_FIELDS = [
    "run_name",
    "run_dir",
    "model_name",
    "backbone",
    "window_seconds",
    "stride_seconds",
    "input_length_samples",
    "total_parameters",
    "trainable_parameters",
    "mean_window_latency_ms_batch1",
    "mean_window_latency_ms_batch16",
    "windows_per_second_batch16",
    "timing_device",
    "warmup_iterations",
    "measured_repeats",
    "throughput_batch_size",
    "efficiency_file_path",
    "status",
    "error_message",
]


def _run_dirs(root: Path) -> list[Path]:
    if (root / "config_resolved.yaml").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("config_resolved.yaml"))


def _window_metadata(cfg: dict) -> dict[str, Any]:
    preprocessing = cfg.get("preprocessing", {}) or {}
    windowing = preprocessing.get("windowing", {}) or {}
    window_seconds = windowing.get("window_seconds", preprocessing.get("target_seconds"))
    stride_seconds = windowing.get("stride_seconds", window_seconds)
    fs_target = preprocessing.get("fs_target")
    target_seconds = preprocessing.get("target_seconds", window_seconds)
    input_length_samples = None
    if fs_target is not None and target_seconds is not None:
        input_length_samples = int(round(float(fs_target) * float(target_seconds)))
    return {
        "window_seconds": float(window_seconds) if window_seconds is not None else None,
        "stride_seconds": float(stride_seconds) if stride_seconds is not None else None,
        "input_length_samples": input_length_samples,
    }


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _cpu_reference_overrides(cfg: dict) -> bool:
    model_cfg = cfg.get("model", {}) or {}
    backbone = str(model_cfg.get("backbone", model_cfg.get("name", ""))).lower()
    if backbone not in {"mamba", "bimamba"}:
        return False
    fast_path_overridden = bool(model_cfg.get("use_fast_path", True))
    model_cfg["use_fast_path"] = False
    try:
        import mamba_ssm.modules.mamba_simple as mamba_simple
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = selective_scan_ref
    except Exception:
        pass
    return fast_path_overridden


def _latencies_ms(model: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> list[float]:
    out: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _summary(vals: list[float]) -> dict[str, float]:
    t = torch.tensor(vals, dtype=torch.float64)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "p50": float(torch.quantile(t, 0.50).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _profile_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from models import build_model
    from utils.config import load_config

    cfg_path = run_dir / "config_resolved.yaml"
    cfg = load_config(str(cfg_path))
    metadata = _window_metadata(cfg)
    input_len = metadata["input_length_samples"]
    if not input_len:
        raise ValueError("Could not infer input_length_samples from config")

    torch.set_num_threads(max(1, int(args.torch_threads)))
    fast_path_overridden = _cpu_reference_overrides(cfg)
    device = torch.device("cpu")
    model = build_model(cfg).to(device=device, dtype=torch.float32)
    model.eval()
    total_params, trainable_params = _count_parameters(model)

    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    x1 = torch.randn(1, 1, int(input_len), generator=generator, device=device)
    xN = x1.repeat(int(args.throughput_batch_size), 1, 1)

    t1 = _latencies_ms(model, x1, int(args.warmup), int(args.repeats))
    tN = _latencies_ms(model, xN, int(args.warmup), int(args.repeats))
    s1 = _summary(t1)
    sN = _summary(tN)
    throughput = float(int(args.throughput_batch_size) / (sN["mean"] / 1000.0))

    payload: dict[str, Any] = {
        "config_path": str(cfg_path.resolve()),
        "checkpoint_path": str((run_dir / "checkpoints" / "best.ckpt").resolve())
        if (run_dir / "checkpoints" / "best.ckpt").exists()
        else "",
        "model_name": cfg.get("model", {}).get("name"),
        "backbone": cfg.get("model", {}).get("backbone"),
        **metadata,
        "timing_device": "cpu",
        "device": "cpu",
        "timing_scope": "synthetic_model_forward_only_excludes_loader_io",
        "window_input_source": "synthetic_random_window",
        "cpu_fast_path_override": fast_path_overridden,
        "precision": "fp32",
        "warmup_iterations": int(args.warmup),
        "measured_repeats": int(args.repeats),
        "latency_batch_size": 1,
        "throughput_batch_size": int(args.throughput_batch_size),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "mean_window_latency_ms_batch1": s1["mean"],
        "std_window_latency_ms_batch1": s1["std"],
        "p50_window_latency_ms_batch1": s1["p50"],
        "p95_window_latency_ms_batch1": s1["p95"],
        "mean_window_latency_ms_batch16": sN["mean"],
        "std_window_latency_ms_batch16": sN["std"],
        "p50_window_latency_ms_batch16": sN["p50"],
        "p95_window_latency_ms_batch16": sN["p95"],
        "windows_per_second_batch16": throughput,
        "num_records": 0,
        "num_windows": 0,
    }

    window_rows = (
        [{"batch_size": 1, "repeat_idx": i, "latency_ms": v} for i, v in enumerate(t1)]
        + [
            {"batch_size": int(args.throughput_batch_size), "repeat_idx": i, "latency_ms": v}
            for i, v in enumerate(tN)
        ]
    )
    _write_csv(
        run_dir / "efficiency_window_latency.csv",
        window_rows,
        ["batch_size", "repeat_idx", "latency_ms"],
    )
    _write_csv(
        run_dir / "efficiency_record_latency.csv",
        [],
        ["record_id", "num_windows", "latency_ms"],
    )
    with (run_dir / "efficiency.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--summary", default="results/synthetic_efficiency_summary.csv")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--throughput-batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    run_dirs: list[Path] = []
    for root_arg in args.roots:
        root = Path(root_arg)
        if not root.is_absolute():
            root = ROOT / root
        run_dirs.extend(_run_dirs(root))
    run_dirs = sorted(set(run_dirs))

    rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        efficiency_path = run_dir / "efficiency.json"
        if efficiency_path.exists() and not args.overwrite:
            try:
                payload = json.loads(efficiency_path.read_text(encoding="utf-8"))
                status = "existing"
                error = ""
            except Exception as exc:
                payload = {}
                status = "failed"
                error = str(exc)
        else:
            try:
                print(f"[{idx}/{len(run_dirs)}] profile {run_dir.relative_to(ROOT)}", flush=True)
                payload = _profile_run(run_dir, args)
                status = "success"
                error = ""
            except Exception as exc:
                if not args.keep_going:
                    raise
                payload = {}
                status = "failed"
                error = str(exc)
                print(f"[{idx}/{len(run_dirs)}] failed {run_dir.relative_to(ROOT)}: {exc}", flush=True)
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "model_name": payload.get("model_name", ""),
                "backbone": payload.get("backbone", ""),
                "window_seconds": payload.get("window_seconds", ""),
                "stride_seconds": payload.get("stride_seconds", ""),
                "input_length_samples": payload.get("input_length_samples", ""),
                "total_parameters": payload.get("total_parameters", ""),
                "trainable_parameters": payload.get("trainable_parameters", ""),
                "mean_window_latency_ms_batch1": payload.get("mean_window_latency_ms_batch1", ""),
                "mean_window_latency_ms_batch16": payload.get("mean_window_latency_ms_batch16", ""),
                "windows_per_second_batch16": payload.get("windows_per_second_batch16", ""),
                "timing_device": payload.get("timing_device", ""),
                "warmup_iterations": payload.get("warmup_iterations", ""),
                "measured_repeats": payload.get("measured_repeats", ""),
                "throughput_batch_size": payload.get("throughput_batch_size", ""),
                "efficiency_file_path": str(efficiency_path),
                "status": status,
                "error_message": error,
            }
        )

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_path, rows, SUMMARY_FIELDS)
    print(f"Wrote summary: {summary_path.relative_to(ROOT)} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
