#!/usr/bin/env python
"""Profile CPU inference efficiency for completed runs without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    "log_file_path",
    "return_code",
    "error_message",
    "status",
]


@dataclass(frozen=True)
class ProfileJob:
    index: int
    total: int
    run_dir: Path
    cmd: list[str]
    log_path: Path


@dataclass(frozen=True)
class ProfileResult:
    run_dir: Path
    status: str
    return_code: int | None = None
    log_path: Path | None = None
    error_message: str = ""


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


def _efficiency_summary(
    run_dir: Path,
    *,
    status: str | None = None,
    return_code: int | None = None,
    error_message: str = "",
    log_path: Path | None = None,
) -> dict[str, Any]:
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
        "log_file_path": str(log_path) if log_path is not None else "",
        "return_code": "" if return_code is None else return_code,
        "error_message": error_message,
        "status": status or ("success" if efficiency else "missing_efficiency"),
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


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _safe_log_name(run_dir: Path) -> str:
    return "__".join(run_dir.parts[-2:]) + ".log"


def _run_profile_job(job: ProfileJob, repo_root: Path) -> ProfileResult:
    import os

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with job.log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(job.cmd)}\n")
        f.flush()
        completed = subprocess.run(job.cmd, cwd=repo_root, stdout=f, stderr=subprocess.STDOUT, env=env)
    if completed.returncode != 0:
        return ProfileResult(
            run_dir=job.run_dir,
            status="failed",
            return_code=int(completed.returncode),
            log_path=job.log_path,
            error_message=f"Efficiency profiling failed with return code {completed.returncode}",
        )
    if not (job.run_dir / "efficiency.json").exists():
        return ProfileResult(
            run_dir=job.run_dir,
            status="failed",
            return_code=0,
            log_path=job.log_path,
            error_message="Profiler exited successfully but did not write efficiency.json",
        )
    return ProfileResult(job.run_dir, "success", 0, job.log_path)


def _run_jobs(
    jobs: list[ProfileJob],
    *,
    repo_root: Path,
    max_workers: int,
    keep_going: bool,
) -> dict[Path, ProfileResult]:
    results: dict[Path, ProfileResult] = {}
    if max_workers <= 1:
        for job in jobs:
            print(f"[{job.index}/{job.total}] profile {_display_path(job.run_dir, repo_root)}")
            result = _run_profile_job(job, repo_root)
            results[job.run_dir] = result
            print(f"[{job.index}/{job.total}] {result.status} {_display_path(job.run_dir, repo_root)}")
            if result.status == "failed" and not keep_going:
                raise subprocess.CalledProcessError(result.return_code or 1, job.cmd)
        return results

    print(
        "Running profiling jobs in parallel. "
        "Use --jobs 1 for the cleanest CPU latency measurements."
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(_run_profile_job, job, repo_root): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results[job.run_dir] = result
            print(f"[{job.index}/{job.total}] {result.status} {_display_path(job.run_dir, repo_root)}")
            if result.status == "failed" and not keep_going:
                for pending in future_to_job:
                    pending.cancel()
                raise subprocess.CalledProcessError(result.return_code or 1, job.cmd)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Completed run dir(s) or parent directories containing runs.")
    parser.add_argument("--summary", default="runs/efficiency_summary.csv")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--throughput-batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Number of profiling subprocesses to run. Use 1 for publication-quality CPU latency.")
    parser.add_argument("--log-dir", default="runs/efficiency_profile_logs")
    parser.add_argument("--keep-going", action="store_true", help="Continue profiling remaining runs if one run fails.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Only report and summarize which runs have/miss efficiency metrics.")
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

    existing = [run_dir for run_dir in run_dirs if (run_dir / "efficiency.json").exists()]
    missing = [run_dir for run_dir in run_dirs if not (run_dir / "efficiency.json").exists()]
    to_profile = run_dirs if args.overwrite else missing
    print(
        "Efficiency audit: "
        f"completed_runs={len(run_dirs)}, with_efficiency={len(existing)}, "
        f"missing_efficiency={len(missing)}, to_profile={0 if args.audit_only else len(to_profile)}"
    )

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir

    rows: list[dict[str, Any]] = []
    results: dict[Path, ProfileResult] = {}
    if args.audit_only:
        pass
    elif args.dry_run:
        for idx, run_dir in enumerate(to_profile, start=1):
            cmd = _profile_command(args, run_dir)
            print(f"[{idx}/{len(to_profile)}] {' '.join(cmd)}")
    else:
        jobs = [
            ProfileJob(
                index=idx,
                total=len(to_profile),
                run_dir=run_dir,
                cmd=_profile_command(args, run_dir),
                log_path=log_dir / _safe_log_name(run_dir),
            )
            for idx, run_dir in enumerate(to_profile, start=1)
        ]
        results = _run_jobs(jobs, repo_root=repo_root, max_workers=max(1, args.jobs), keep_going=args.keep_going)

    for run_dir in run_dirs:
        result = results.get(run_dir)
        if result is not None:
            rows.append(
                _efficiency_summary(
                    run_dir,
                    status=result.status,
                    return_code=result.return_code,
                    error_message=result.error_message,
                    log_path=result.log_path,
                )
            )
        else:
            rows.append(_efficiency_summary(run_dir))

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if not args.dry_run:
        _write_summary(summary_path, rows)
        print(f"Wrote summary: {_display_path(summary_path, repo_root)} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
