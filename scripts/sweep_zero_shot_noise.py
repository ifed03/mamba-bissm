#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data.noise_injection import VALID_NOISE_TYPES
from evaluate.noise_protocol import NoiseCondition, condition_key, metrics_filename
from utils.config import load_config
from utils.io import model_label_from_config
from utils.logging import save_json

DEFAULT_NOISE_TYPES = tuple(sorted(VALID_NOISE_TYPES))
DEFAULT_SNRS = (18.0, 6.0, 0.0, -6.0)
DEFAULT_OUTPUT_ROOT = Path("outputs") / "zero_shot_noise_sweep"
SUMMARY_COLUMNS = [
    "run_name",
    "model_family",
    "checkpoint_path",
    "config_path",
    "noise_type",
    "snr_db",
    "threshold",
    "threshold_source",
    "auroc",
    "auprc",
    "f1",
    "accuracy",
    "sensitivity",
    "specificity",
    "tn",
    "fp",
    "fn",
    "tp",
    "num_test_examples",
    "result_path",
    "status",
]


@dataclass(frozen=True)
class SuccessfulRun:
    run_name: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    threshold_path: Path | None
    model_family: str


@dataclass(frozen=True)
class EvaluationJob:
    run: SuccessfulRun
    condition: NoiseCondition
    output_root: Path
    result_path: Path


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _resolve_manifest_entry(path: str | Path, *, manifest_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_relative = (manifest_dir / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return candidate.resolve()


def _read_run_entries(successful_runs: Path) -> list[Path]:
    if successful_runs.is_dir():
        return sorted(p for p in successful_runs.iterdir() if p.is_dir())

    if not successful_runs.is_file():
        raise FileNotFoundError(f"successful_runs path does not exist: {successful_runs}")

    text = successful_runs.read_text(encoding="utf-8")
    base = successful_runs.parent
    if successful_runs.suffix.lower() == ".json":
        payload = json.loads(text)
        entries = payload.get("runs", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("JSON successful_runs manifest must be a list or an object with a 'runs' list.")
        raw_paths = [entry.get("path", entry.get("run_dir")) if isinstance(entry, dict) else entry for entry in entries]
    else:
        raw_paths = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

    paths = []
    for raw in raw_paths:
        if raw is None:
            raise ValueError(f"Invalid empty run entry in {successful_runs}")
        paths.append(_resolve_manifest_entry(str(raw), manifest_dir=base))
    return sorted(paths)


def _locate_config(run_dir: Path) -> Path:
    candidates = [run_dir / "config_resolved.yaml", run_dir / "config.yaml", run_dir / "config_resolved.yml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    yaml_files = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    if yaml_files:
        return yaml_files[0]
    raise FileNotFoundError(f"Missing config_resolved.yaml or equivalent in {run_dir}")


def _locate_checkpoint(run_dir: Path) -> Path:
    candidates = [run_dir / "checkpoints" / "best.ckpt", run_dir / "best.ckpt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt")) if (run_dir / "checkpoints").is_dir() else []
    ckpts += sorted(run_dir.glob("*.ckpt"))
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError(f"Missing checkpoint, preferably checkpoints/best.ckpt, in {run_dir}")


def _locate_threshold(run_dir: Path) -> Path | None:
    path = run_dir / "clean_validation_threshold.json"
    return path if path.is_file() else None


def discover_successful_runs(successful_runs: str | Path) -> list[SuccessfulRun]:
    root = _resolve_path(successful_runs)
    runs: list[SuccessfulRun] = []
    errors: list[str] = []
    for run_dir in _read_run_entries(root):
        try:
            config_path = _locate_config(run_dir)
            checkpoint_path = _locate_checkpoint(run_dir)
            cfg = load_config(str(config_path))
            runs.append(
                SuccessfulRun(
                    run_name=run_dir.name,
                    run_dir=run_dir,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    threshold_path=_locate_threshold(run_dir),
                    model_family=model_label_from_config(cfg),
                )
            )
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")

    if errors:
        raise RuntimeError("Failed to resolve successful run(s):\n" + "\n".join(errors))
    if not runs:
        raise ValueError(f"No successful run directories found in {root}")
    return runs


def validate_noise_types(noise_types: list[str]) -> list[str]:
    invalid = [nt for nt in noise_types if nt not in VALID_NOISE_TYPES]
    if invalid:
        raise ValueError(f"Unsupported noise type(s) {invalid}. Expected one of {sorted(VALID_NOISE_TYPES)}.")
    return noise_types


def build_evaluation_matrix(
    runs: list[SuccessfulRun],
    *,
    noise_types: list[str],
    snrs: list[float],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> list[EvaluationJob]:
    validate_noise_types(noise_types)
    root = Path(output_root)
    jobs: list[EvaluationJob] = []
    for run in runs:
        run_output_root = root / run.model_family / run.run_name / "zero-shot"
        for noise_type in noise_types:
            for snr_db in snrs:
                condition = NoiseCondition(noise_type, float(snr_db))
                jobs.append(
                    EvaluationJob(
                        run=run,
                        condition=condition,
                        output_root=run_output_root,
                        result_path=run_output_root / metrics_filename(condition),
                    )
                )
    return jobs


def result_json_complete(path: str | Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required_top = {"eval", "noise_type", "snr_db", "threshold", "threshold_source", "checkpoint", "num_test_examples", "test"}
    if not required_top.issubset(payload):
        return False
    metrics = payload.get("test") or {}
    required_metrics = {"auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity", "confusion_matrix"}
    return isinstance(metrics, dict) and required_metrics.issubset(metrics)


def _confusion_entries(metrics: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    cm = metrics.get("confusion_matrix", metrics.get("cm"))
    try:
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    except (TypeError, IndexError):
        return "", "", "", ""


def summary_row_from_result(job: EvaluationJob, payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    metrics = payload.get("test") or {}
    tn, fp, fn, tp = _confusion_entries(metrics)
    return {
        "run_name": job.run.run_name,
        "model_family": job.run.model_family,
        "checkpoint_path": str(job.run.checkpoint_path),
        "config_path": str(job.run.config_path),
        "noise_type": payload.get("noise_type", job.condition.noise_type),
        "snr_db": payload.get("snr_db", float(job.condition.snr_db)),
        "threshold": payload.get("threshold", ""),
        "threshold_source": payload.get("threshold_source", ""),
        "auroc": metrics.get("auroc", ""),
        "auprc": metrics.get("auprc", ""),
        "f1": metrics.get("f1", ""),
        "accuracy": metrics.get("accuracy", metrics.get("acc", "")),
        "sensitivity": metrics.get("sensitivity", ""),
        "specificity": metrics.get("specificity", ""),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "num_test_examples": payload.get("num_test_examples", ""),
        "result_path": str(job.result_path),
        "status": status,
    }


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = output_root / "zero_shot_noise_summary.csv"
    json_path = output_root / "zero_shot_noise_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    save_json(json_path, {"rows": rows})
    return csv_path, json_path


def _ordered_unique(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _jobs_by_run(jobs: list[EvaluationJob]) -> list[tuple[SuccessfulRun, list[EvaluationJob]]]:
    groups: list[tuple[SuccessfulRun, list[EvaluationJob]]] = []
    index: dict[SuccessfulRun, list[EvaluationJob]] = {}
    for job in jobs:
        if job.run not in index:
            index[job.run] = []
            groups.append((job.run, index[job.run]))
        index[job.run].append(job)
    return groups


def _evaluate_run_command(run: SuccessfulRun, jobs: list[EvaluationJob], args: argparse.Namespace) -> list[str]:
    noise_types = _ordered_unique([job.condition.noise_type for job in jobs])
    snrs = _ordered_unique([float(job.condition.snr_db) for job in jobs])
    output_roots = {job.output_root for job in jobs}
    if len(output_roots) != 1:
        raise ValueError(f"Expected one output root for run {run.run_name}, got {sorted(str(p) for p in output_roots)}")
    cmd = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--config",
        str(run.config_path),
        "--checkpoint",
        str(run.checkpoint_path),
        "--noise-eval",
        "zero-shot",
        "--noise-types",
        *noise_types,
        "--snr-db",
        *[f"{snr:g}" for snr in snrs],
        "--noise-root",
        args.noise_root,
        "--base-seed",
        str(args.base_seed),
        "--ecg-fs",
        str(args.ecg_fs),
        "--output-root",
        str(next(iter(output_roots))),
        "--skip-existing",
    ]
    if run.threshold_path is not None:
        cmd.extend(["--clean-threshold-path", str(run.threshold_path)])
    return cmd


def failure_row(job: EvaluationJob, status: str) -> dict[str, Any]:
    return {
        "run_name": job.run.run_name,
        "model_family": job.run.model_family,
        "checkpoint_path": str(job.run.checkpoint_path),
        "config_path": str(job.run.config_path),
        "noise_type": job.condition.noise_type,
        "snr_db": float(job.condition.snr_db),
        "result_path": str(job.result_path),
        "status": status,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a resumable zero-shot ECG noise sweep over successful runs.")
    p.add_argument("--successful-runs", default="successful_runs")
    p.add_argument("--snrs", nargs="+", type=float, default=list(DEFAULT_SNRS))
    p.add_argument("--noise-types", nargs="+", default=list(DEFAULT_NOISE_TYPES))
    p.add_argument("--noise-root", default="data")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--ecg-fs", type=float, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    noise_types = validate_noise_types(args.noise_types)
    runs = discover_successful_runs(args.successful_runs)
    jobs = build_evaluation_matrix(runs, noise_types=noise_types, snrs=args.snrs, output_root=args.output_root)
    output_root = Path(args.output_root)

    print(f"Discovered successful runs: {len(runs)}")
    print(f"Noise types: {noise_types}")
    print(f"SNR dB levels: {[float(s) for s in args.snrs]}")
    print(f"Planned evaluations: {len(jobs)}")

    rows: list[dict[str, Any]] = []
    failures = 0
    for run, run_jobs in _jobs_by_run(jobs):
        for job in run_jobs:
            cond = condition_key(job.condition)
            prefix = f"{job.run.run_name} {cond}"
            if args.dry_run:
                print(f"DRY-RUN would evaluate: {prefix}")
                print(f"DRY-RUN result path: {job.result_path}")

        if args.dry_run:
            continue

        complete_jobs = [job for job in run_jobs if job.result_path.is_file() and result_json_complete(job.result_path)]
        incomplete_jobs = [job for job in run_jobs if args.overwrite or job not in complete_jobs]

        for job in complete_jobs:
            if not args.overwrite:
                payload = json.loads(job.result_path.read_text(encoding="utf-8"))
                rows.append(summary_row_from_result(job, payload, status="skipped_existing"))

        if not incomplete_jobs:
            print(f"SKIP run with all conditions complete: {run.run_name}")
            continue

        print(f"RUN batched zero-shot noise for {run.run_name}: {len(incomplete_jobs)} pending condition(s)")
        result = subprocess.run(_evaluate_run_command(run, incomplete_jobs, args), cwd=Path(__file__).resolve().parents[1], text=True)
        if result.returncode != 0:
            failures += len(incomplete_jobs)
            rows.extend(failure_row(job, f"failed_exit_{result.returncode}") for job in incomplete_jobs)
            print(f"FAILED {run.run_name}: evaluate_model.py exited with {result.returncode}")
            continue

        for job in incomplete_jobs:
            if not result_json_complete(job.result_path):
                failures += 1
                rows.append(failure_row(job, "failed_missing_metrics"))
                print(f"FAILED {job.run.run_name} {condition_key(job.condition)}: result JSON missing or incomplete at {job.result_path}")
                continue
            payload = json.loads(job.result_path.read_text(encoding="utf-8"))
            rows.append(summary_row_from_result(job, payload, status="completed"))
            print(f"COMPLETED {job.result_path}")

    if args.dry_run:
        print("DRY-RUN complete: no evaluations executed and no files written.")
        return 0

    csv_path, json_path = _write_summary(output_root, rows)
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Completed/skipped rows: {len([r for r in rows if str(r.get('status', '')).startswith(('completed', 'skipped'))])}")
    print(f"Failed evaluations: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
