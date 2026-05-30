#!/usr/bin/env python
"""Profile CPU inference efficiency for completed runs without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_name",
    "run_dir",
    "model_name",
    "backbone",
    "window_seconds",
    "stride_seconds",
    "input_length_samples",
    "num_trainable_params",
    "total_parameters",
    "training_time_seconds",
    "mean_epoch_time_seconds",
    "best_epoch",
    "inference_time_seconds_total",
    "inference_latency_ms_per_record",
    "inference_latency_ms_per_window",
    "throughput_records_per_second",
    "throughput_windows_per_second",
    "timing_device",
    "warmup_iterations",
    "measured_repeats",
    "throughput_batch_size",
    "efficiency_file_path",
    "metrics_file_path",
    "status",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_dirs(root: Path) -> list[Path]:
    if (root / "config_resolved.yaml").exists() and (root / "checkpoints" / "best.ckpt").exists():
        return [root]
    return sorted(
        p.parent
        for p in root.rglob("config_resolved.yaml")
        if (p.parent / "checkpoints" / "best.ckpt").exists()
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics.json"


def _efficiency_summary(run_dir: Path) -> dict[str, Any]:
    efficiency_path = run_dir / "efficiency.json"
    metrics_path = _metrics_path(run_dir)
    efficiency = _load_json(efficiency_path)
    metrics = _load_json(metrics_path)
    record_rows = []
    record_csv = run_dir / "efficiency_record_latency.csv"
    if record_csv.exists():
        with record_csv.open(newline="", encoding="utf-8") as f:
            record_rows = list(csv.DictReader(f))
    total_record_ms = sum(float(row["latency_ms"]) for row in record_rows if row.get("latency_ms"))

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": efficiency.get("model_name", ""),
        "backbone": efficiency.get("backbone", ""),
        "window_seconds": efficiency.get("window_seconds", ""),
        "stride_seconds": efficiency.get("stride_seconds", ""),
        "input_length_samples": efficiency.get("input_length_samples", ""),
        "num_trainable_params": efficiency.get("trainable_parameters", ""),
        "total_parameters": efficiency.get("total_parameters", ""),
        "training_time_seconds": metrics.get("training_time_seconds", ""),
        "mean_epoch_time_seconds": metrics.get("mean_epoch_time_seconds", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "inference_time_seconds_total": total_record_ms / 1000.0 if record_rows else "",
        "inference_latency_ms_per_record": efficiency.get("mean_record_latency_ms", ""),
        "inference_latency_ms_per_window": efficiency.get("mean_window_latency_ms_batch1", ""),
        "throughput_records_per_second": efficiency.get("records_per_second", ""),
        "throughput_windows_per_second": efficiency.get("windows_per_second_batch16", ""),
        "timing_device": efficiency.get("timing_device", ""),
        "warmup_iterations": efficiency.get("warmup_iterations", ""),
        "measured_repeats": efficiency.get("measured_repeats", ""),
        "throughput_batch_size": efficiency.get("throughput_batch_size", ""),
        "efficiency_file_path": str(efficiency_path),
        "metrics_file_path": str(metrics_path) if metrics_path.exists() else "",
        "status": "success" if efficiency else "missing_efficiency",
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _profile_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/profile_efficiency.py",
        "--config",
        str(run_dir / "config_resolved.yaml"),
        "--ckpt",
        str(run_dir / "checkpoints" / "best.ckpt"),
        "--device",
        "cpu",
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--throughput-batch-size",
        str(args.throughput_batch_size),
    ]
    if args.max_records is not None:
        cmd.extend(["--max-records", str(args.max_records)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Completed run dir(s) or parent directories containing runs.")
    parser.add_argument("--summary", default="runs/efficiency_summary.csv")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--throughput-batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    run_dirs: list[Path] = []
    for root_arg in args.roots:
        root = Path(root_arg)
        if not root.is_absolute():
            root = repo_root / root
        run_dirs.extend(_run_dirs(root))
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        raise ValueError("No completed runs found with config_resolved.yaml and checkpoints/best.ckpt.")

    rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        efficiency_path = run_dir / "efficiency.json"
        if args.overwrite or not efficiency_path.exists():
            cmd = _profile_command(args, run_dir)
            print(f"[{idx}/{len(run_dirs)}] profile {run_dir.relative_to(repo_root)}")
            if args.dry_run:
                print(" ".join(cmd))
            else:
                subprocess.run(cmd, cwd=repo_root, check=True)
        else:
            print(f"[{idx}/{len(run_dirs)}] skip existing {run_dir.relative_to(repo_root)}")
        rows.append(_efficiency_summary(run_dir))

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if not args.dry_run:
        _write_summary(summary_path, rows)
        print(f"Wrote summary: {summary_path.relative_to(repo_root)} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
