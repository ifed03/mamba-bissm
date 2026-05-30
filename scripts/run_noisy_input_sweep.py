#!/usr/bin/env python
"""Launch controlled noisy-input AF/NSR training sweeps."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NOISE_TYPES = ("bw", "em", "ma")
SNR_DB = (24.0, 18.0, 12.0, 6.0, 0.0, -6.0)
REQUIRED_NSTDB_FILES = ("bw.hea", "bw.dat", "em.hea", "em.dat", "ma.hea", "ma.dat")
PROCESSING_ORDER = "resample->noise->window->normalize"
NOISY_VAL = "noisy_val"
LOCK_NAME = ".run.lock"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_family: str
    backbone: str
    config_path: str
    smoke_config_path: str | None = None


@dataclass(frozen=True)
class WorkItem:
    kind: str
    entry: dict[str, Any]


@dataclass
class WorkResult:
    run_name: str
    kind: str
    status: str
    return_code: int | None = None
    log_file: str | None = None
    error_message: str | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="ecgmamba_mamba",
        model_family="ecgmamba",
        backbone="mamba",
        config_path="configs/binary_mamba_d64_n2_s16_100hz_win4s_stride2s.yaml",
        smoke_config_path="configs/smoke_ecgmamba_mamba_ssm_reduced_fp32_win4s_3epoch.yaml",
    ),
    ModelSpec(
        key="ecgmamba_bimamba",
        model_family="ecgmamba",
        backbone="bimamba",
        config_path="configs/binary_bimamba_d128_n2_s64_slowpath_fp32_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="ecgmamba_bilstm",
        model_family="ecgmamba",
        backbone="bilstm",
        config_path="configs/binary_ecgmamba_bilstm_d64_n2_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="ecgmamba_bissm",
        model_family="ecgmamba",
        backbone="bissm",
        config_path="configs/binary_bissm_d64_n2_s32_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="cnn1d",
        model_family="cnn1d",
        backbone="baseline",
        config_path="configs/binary_cnn1d_c256_n3_k7_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="bilstm",
        model_family="bilstm",
        backbone="baseline",
        config_path="configs/binary_bilstm_100hz_win4s_stride2s.yaml",
        smoke_config_path="configs/smoke_bilstm_win4s_3epoch.yaml",
    ),
)


SUMMARY_FIELDS = (
    "run_name",
    "model_family",
    "backbone",
    "dimensions",
    "noise_type",
    "snr_db",
    "best_epoch",
    "best_val_metric_name",
    "best_val_auroc",
    "noisy_val_tau_star",
    "test_auroc",
    "test_auprc",
    "test_f1",
    "test_accuracy",
    "test_sensitivity",
    "test_specificity",
    "checkpoint_source",
    "threshold_source",
    "metrics_file_path",
    "threshold_file_path",
    "checkpoint_path",
    "num_trainable_params",
    "inference_time_seconds_total",
    "inference_latency_ms_per_record",
    "inference_latency_ms_per_window",
    "throughput_records_per_second",
    "throughput_windows_per_second",
    "efficiency_file_path",
    "log_file_path",
    "return_code",
    "error_message",
    "status",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _snr_token(snr_db: float) -> str:
    raw = f"{float(snr_db):g}"
    return raw.replace("-", "neg")


def _run_snr_token(snr_db: float) -> str:
    return f"{_snr_token(snr_db)}dB"


def _condition_key(noise_type: str, snr_db: float) -> str:
    return f"noise_type={noise_type}__snr_db={_snr_token(snr_db)}"


def noisy_input_metrics_filename(noise_type: str, snr_db: float) -> str:
    return f"metrics_noisy-input-training_{_condition_key(noise_type, snr_db)}.json"


def noisy_input_threshold_filename(noise_type: str, snr_db: float) -> str:
    return f"threshold_noisy-input-training_{_condition_key(noise_type, snr_db)}.json"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def dimensions_from_config(cfg: dict[str, Any]) -> str:
    mcfg = cfg.get("model", {}) or {}
    model_name = str(mcfg.get("name", "")).lower()
    backbone = str(mcfg.get("backbone", model_name)).lower()

    if model_name == "cnn1d":
        channels = mcfg.get("cnn_channels", mcfg.get("channels", []))
        if isinstance(channels, int):
            channels = [channels]
        final_channels = channels[-1] if channels else "na"
        return f"c{final_channels}_n{len(channels)}_k{mcfg.get('cnn_kernel_size', 'na')}"

    if model_name == "bilstm":
        return f"h{mcfg.get('hidden_size', 'na')}_n{mcfg.get('num_layers', 'na')}"

    d_model = mcfg.get("d_model", "na")
    n_layers = mcfg.get("n_layers", "na")
    if backbone in {"mamba", "bimamba"}:
        return f"d{d_model}_n{n_layers}_s{mcfg.get('d_state', mcfg.get('state_dim', 'na'))}"
    if backbone == "bissm":
        return f"d{d_model}_n{n_layers}_s{mcfg.get('state_dim', 'na')}"
    if backbone == "bilstm":
        return f"d{d_model}_n{n_layers}_h{mcfg.get('lstm_hidden_size', 'na')}"
    return f"d{d_model}_n{n_layers}"


def run_name(spec: ModelSpec, dimensions: str, noise_type: str, snr_db: float) -> str:
    return f"{spec.key}_{dimensions}_{noise_type}_{_run_snr_token(snr_db)}"


def selected_specs(model_keys: list[str] | None, *, smoke: bool) -> list[ModelSpec]:
    keys = model_keys or [spec.key for spec in MODEL_SPECS]
    lookup = {spec.key: spec for spec in MODEL_SPECS}
    unknown = sorted(set(keys) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Valid keys: {sorted(lookup)}")
    if smoke and model_keys is None:
        keys = ["ecgmamba_mamba", "bilstm"]
    return [lookup[key] for key in keys]


def build_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    models: list[str] | None = None,
    noise_types: list[str] | None = None,
    snr_db: list[float] | None = None,
    seed: int = 123,
    ecg_fs: float = 100,
    noise_root: str = "data",
    python_executable: str = sys.executable,
    smoke: bool = False,
) -> dict[str, Any]:
    chosen_noise = noise_types or list(NOISE_TYPES)
    chosen_snr = snr_db or list(SNR_DB)
    if smoke:
        chosen_noise = noise_types or ["bw"]
        chosen_snr = snr_db or [18.0]

    entries: list[dict[str, Any]] = []
    for spec in selected_specs(models, smoke=smoke):
        cfg_rel = spec.smoke_config_path if smoke and spec.smoke_config_path else spec.config_path
        cfg_path = repo_root / cfg_rel
        cfg_exists = cfg_path.is_file()
        cfg = load_config(cfg_path) if cfg_exists else {}
        dimensions = dimensions_from_config(cfg) if cfg_exists else "unknown"

        for noise_type in chosen_noise:
            for snr in chosen_snr:
                name = run_name(spec, dimensions, noise_type, snr)
                out_dir = output_root / name
                command = [
                    python_executable,
                    "scripts/train_model.py",
                    "--config",
                    cfg_rel,
                    "--run-name",
                    name,
                    "--noise-training-mode",
                    "noisy-input",
                    "--noise-root",
                    noise_root,
                    "--noise-types",
                    noise_type,
                    "--snr-db",
                    f"{float(snr):g}",
                    "--base-seed",
                    str(seed),
                    "--ecg-fs",
                    f"{float(ecg_fs):g}",
                    "--output-root",
                    str(output_root),
                ]
                entries.append(
                    {
                        "run_name": name,
                        "model_family": spec.model_family,
                        "backbone": spec.backbone,
                        "config_path": cfg_rel,
                        "dimensions": dimensions,
                        "noise_type": noise_type,
                        "snr_db": float(snr),
                        "output_dir": str(out_dir),
                        "command": command,
                        "command_str": " ".join(command),
                        "seed": int(seed),
                        "ecg_fs": float(ecg_fs),
                        "expected_metrics_file": str(out_dir / noisy_input_metrics_filename(noise_type, snr)),
                        "expected_threshold_file": str(out_dir / noisy_input_threshold_filename(noise_type, snr)),
                        "expected_checkpoint_file": str(out_dir / "checkpoints" / "best.ckpt"),
                        "status": "planned" if cfg_exists else "missing_config",
                    }
                )

    return {
        "protocol": "noisy-input-training",
        "smoke": bool(smoke),
        "output_root": str(output_root),
        "noise_root": noise_root,
        "seed": int(seed),
        "ecg_fs": float(ecg_fs),
        "expected_run_count": len(entries),
        "entries": entries,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    entries = manifest["entries"]
    names = [entry["run_name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate run names in noisy-input sweep manifest.")
    output_dirs = [entry["output_dir"] for entry in entries]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Duplicate output directories in noisy-input sweep manifest.")

    missing_configs = [entry["config_path"] for entry in entries if entry["status"] == "missing_config"]
    if missing_configs:
        unique = sorted(set(missing_configs))
        raise FileNotFoundError(f"Missing config(s) for noisy-input sweep: {unique}")

    noise_root = repo_root / str(manifest["noise_root"])
    missing_noise = [name for name in REQUIRED_NSTDB_FILES if not (noise_root / name).is_file()]
    if missing_noise:
        raise FileNotFoundError(f"NSTDB noise root {noise_root} is missing: {missing_noise}")

    duplicates = [path for path in output_dirs if Path(path).exists()]
    if duplicates and not (overwrite or resume):
        raise FileExistsError(
            "Output run directories already exist. Use --overwrite or --resume explicitly: "
            f"{duplicates[:5]}"
        )

    for entry in entries:
        if not (repo_root / entry["config_path"]).is_file():
            raise FileNotFoundError(f"Config does not exist: {entry['config_path']}")
        command = entry["command"]
        if command.count("--noise-types") != 1 or command.count("--snr-db") != 1:
            raise ValueError(f"Command must include one --noise-types and one --snr-db flag: {entry['run_name']}")
        noise_idx = command.index("--noise-types") + 1
        snr_idx = command.index("--snr-db") + 1
        if command[noise_idx] not in NOISE_TYPES:
            raise ValueError(f"Invalid noise type in command for {entry['run_name']}: {command[noise_idx]}")
        if noise_idx + 1 >= len(command) or command[noise_idx + 1] != "--snr-db":
            raise ValueError(f"Command has multiple noise types for {entry['run_name']}: {command}")
        try:
            float(command[snr_idx])
        except ValueError as exc:
            raise ValueError(f"Invalid SNR in command for {entry['run_name']}: {command[snr_idx]}") from exc
        if snr_idx + 1 < len(command) and not command[snr_idx + 1].startswith("--"):
            raise ValueError(f"Command has multiple SNR values for {entry['run_name']}: {command}")


def _entry_complete(entry: dict[str, Any]) -> bool:
    return (
        Path(entry["expected_metrics_file"]).is_file()
        and Path(entry["expected_threshold_file"]).is_file()
        and Path(entry["expected_checkpoint_file"]).is_file()
    )


def _efficiency_complete(entry: dict[str, Any]) -> bool:
    return (Path(entry["output_dir"]) / "efficiency.json").is_file()


def mark_resume_statuses(manifest: dict[str, Any]) -> None:
    for entry in manifest["entries"]:
        if entry.get("status") == "planned" and _entry_complete(entry):
            entry["status"] = "completed"


def _lock_path(entry: dict[str, Any]) -> Path:
    return Path(entry["output_dir"]) / LOCK_NAME


def acquire_run_lock(entry: dict[str, Any], *, stale_lock_minutes: float | None = None) -> Path | None:
    run_dir = Path(entry["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / LOCK_NAME
    if lock_path.exists() and stale_lock_minutes is not None:
        age_minutes = (time.time() - lock_path.stat().st_mtime) / 60.0
        if age_minutes >= stale_lock_minutes:
            lock_path.unlink()
    payload = json.dumps(
        {
            "run_name": entry["run_name"],
            "pid": os.getpid(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        sort_keys=True,
    )
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as f:
        f.write(payload + "\n")
    return lock_path


def release_run_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _metric(metrics: dict[str, Any], split: str, name: str) -> Any:
    return metrics.get(split, {}).get(name, "")


def parse_summary_row(entry: dict[str, Any]) -> dict[str, Any]:
    metrics_path = Path(entry["expected_metrics_file"])
    threshold_path = Path(entry["expected_threshold_file"])
    run_dir = Path(entry["output_dir"])
    efficiency_path = run_dir / "efficiency.json"
    record_latency_path = run_dir / "efficiency_record_latency.csv"
    row = {
        "run_name": entry["run_name"],
        "model_family": entry["model_family"],
        "backbone": entry["backbone"],
        "dimensions": entry["dimensions"],
        "noise_type": entry["noise_type"],
        "snr_db": entry["snr_db"],
        "metrics_file_path": str(metrics_path),
        "threshold_file_path": str(threshold_path),
        "checkpoint_path": entry.get("expected_checkpoint_file", ""),
        "efficiency_file_path": str(efficiency_path) if efficiency_path.is_file() else "",
        "log_file_path": entry.get("log_file_path", ""),
        "return_code": entry.get("return_code", ""),
        "error_message": entry.get("error_message", ""),
        "status": entry.get("status", "planned"),
    }
    if row["status"] in {"missing_config", "skipped", "failed", "locked"} or not metrics_path.is_file():
        row.update({field: "" for field in SUMMARY_FIELDS if field not in row})
        if row["status"] == "planned":
            row["status"] = "skipped"
        return row

    metrics = json.loads(metrics_path.read_text())
    threshold = json.loads(threshold_path.read_text()) if threshold_path.is_file() else {}
    expected_suffix = f"_{entry['noise_type']}_{_run_snr_token(float(entry['snr_db']))}"
    if not entry["run_name"].endswith(expected_suffix):
        raise ValueError(f"{entry['run_name']} does not end with expected condition suffix {expected_suffix}.")
    threshold_source = metrics.get("threshold_source", threshold.get("threshold_source"))
    checkpoint_source = metrics.get("checkpoint_source", threshold.get("checkpoint_source"))
    if threshold_source != NOISY_VAL:
        raise ValueError(f"{metrics_path} has threshold_source={threshold_source!r}; expected {NOISY_VAL!r}.")
    if checkpoint_source != NOISY_VAL:
        raise ValueError(f"{metrics_path} has checkpoint_source={checkpoint_source!r}; expected {NOISY_VAL!r}.")
    if threshold and threshold.get("threshold_source") != NOISY_VAL:
        raise ValueError(f"{threshold_path} has threshold_source={threshold.get('threshold_source')!r}; expected {NOISY_VAL!r}.")
    if threshold and threshold.get("checkpoint_source") != NOISY_VAL:
        raise ValueError(f"{threshold_path} has checkpoint_source={threshold.get('checkpoint_source')!r}; expected {NOISY_VAL!r}.")
    if metrics.get("noise_type") != entry["noise_type"]:
        raise ValueError(f"{metrics_path} noise_type does not match manifest entry {entry['run_name']}.")
    if float(metrics.get("snr_db")) != float(entry["snr_db"]):
        raise ValueError(f"{metrics_path} snr_db does not match manifest entry {entry['run_name']}.")

    metadata = metrics.get("noise_metadata", {})
    missing_splits = [split for split in ("train", "val", "test") if not metadata.get(split)]
    if missing_splits:
        raise ValueError(f"{metrics_path} is missing noisy metadata for split(s): {missing_splits}.")
    bad_order = [
        split
        for split in ("train", "val", "test")
        for item in metadata.get(split, [])
        if item.get("processing_order") != PROCESSING_ORDER
    ]
    if bad_order:
        raise ValueError(f"{metrics_path} has unexpected processing_order in split(s): {sorted(set(bad_order))}.")

    row.update(
        {
            "best_epoch": metrics.get("best_epoch", ""),
            "best_val_metric_name": metrics.get("best_val_metric_name", ""),
            "best_val_auroc": _metric(metrics, "val", "auroc"),
            "noisy_val_tau_star": metrics.get("threshold", threshold.get("threshold", "")),
            "test_auroc": _metric(metrics, "test", "auroc"),
            "test_auprc": _metric(metrics, "test", "auprc"),
            "test_f1": _metric(metrics, "test", "f1"),
            "test_accuracy": _metric(metrics, "test", "accuracy"),
            "test_sensitivity": _metric(metrics, "test", "sensitivity"),
            "test_specificity": _metric(metrics, "test", "specificity"),
            "checkpoint_source": checkpoint_source,
            "threshold_source": threshold_source,
            "checkpoint_path": threshold.get("checkpoint", entry.get("expected_checkpoint_file", "")),
            "status": "success",
        }
    )
    if efficiency_path.is_file():
        efficiency = json.loads(efficiency_path.read_text())
        total_latency_ms = 0.0
        if record_latency_path.is_file():
            with record_latency_path.open(newline="") as f:
                total_latency_ms = sum(
                    float(record["latency_ms"])
                    for record in csv.DictReader(f)
                    if record.get("latency_ms")
                )
        row.update(
            {
                "num_trainable_params": efficiency.get("trainable_parameters", ""),
                "inference_time_seconds_total": total_latency_ms / 1000.0 if total_latency_ms else "",
                "inference_latency_ms_per_record": efficiency.get("mean_record_latency_ms", ""),
                "inference_latency_ms_per_window": efficiency.get("mean_window_latency_ms_batch1", ""),
                "throughput_records_per_second": efficiency.get("records_per_second", ""),
                "throughput_windows_per_second": efficiency.get("windows_per_second_batch16", ""),
                "efficiency_file_path": str(efficiency_path),
            }
        )
    return row


def collect_summary(manifest: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    rows = [parse_summary_row(entry) for entry in manifest["entries"]]
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    json_path = output_root / "summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def efficiency_command(
    entry: dict[str, Any],
    *,
    warmup: int,
    repeats: int,
    throughput_batch_size: int,
    max_records: int | None,
) -> list[str]:
    run_dir = Path(entry["output_dir"])
    cmd = [
        entry["command"][0],
        "scripts/profile_efficiency.py",
        "--config",
        str(run_dir / "config_resolved.yaml"),
        "--ckpt",
        str(entry["expected_checkpoint_file"]),
        "--device",
        "cpu",
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--throughput-batch-size",
        str(throughput_batch_size),
    ]
    if max_records is not None:
        cmd.extend(["--max-records", str(max_records)])
    return cmd


def profile_entry_efficiency(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    warmup: int,
    repeats: int,
    throughput_batch_size: int,
    max_records: int | None,
    overwrite: bool,
    log_file: Path | None = None,
) -> None:
    run_dir = Path(entry["output_dir"])
    efficiency_path = run_dir / "efficiency.json"
    if efficiency_path.is_file() and not overwrite:
        print(f"Efficiency exists, skipping: {efficiency_path}")
        return
    config_path = run_dir / "config_resolved.yaml"
    checkpoint_path = Path(entry["expected_checkpoint_file"])
    if not config_path.is_file() or not checkpoint_path.is_file():
        print(f"Efficiency skipped, missing config/checkpoint: {run_dir}")
        return
    cmd = efficiency_command(
        entry,
        warmup=warmup,
        repeats=repeats,
        throughput_batch_size=throughput_batch_size,
        max_records=max_records,
    )
    print(f"Profiling CPU efficiency: {run_dir}")
    if log_file is None:
        subprocess.run(cmd, cwd=repo_root, check=True)
        return
    with log_file.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=repo_root, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def build_work_items(
    manifest: dict[str, Any],
    *,
    profile_efficiency: bool = False,
    overwrite_efficiency: bool = False,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    for entry in manifest["entries"]:
        if entry.get("status") not in {"planned", "completed"}:
            continue
        complete = _entry_complete(entry)
        if complete:
            entry["status"] = "completed"
            if profile_efficiency and (overwrite_efficiency or not _efficiency_complete(entry)):
                items.append(WorkItem("profile_efficiency", entry))
            continue
        if entry.get("status") == "planned":
            items.append(WorkItem("train_condition", entry))
    return items


def _run_subprocess_to_log(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return int(result.returncode)


def _run_work_item(
    item: WorkItem,
    *,
    repo_root: Path,
    profile_efficiency: bool,
    efficiency_warmup: int,
    efficiency_repeats: int,
    efficiency_throughput_batch_size: int,
    efficiency_max_records: int | None,
    overwrite_efficiency: bool,
    stale_lock_minutes: float | None,
) -> WorkResult:
    entry = item.entry
    run_dir = Path(entry["output_dir"])
    log_path = run_dir / ("efficiency.log" if item.kind == "profile_efficiency" else "run.log")
    lock_path = acquire_run_lock(entry, stale_lock_minutes=stale_lock_minutes)
    if lock_path is None:
        return WorkResult(
            run_name=entry["run_name"],
            kind=item.kind,
            status="locked",
            log_file=str(log_path),
            error_message=f"Run lock exists: {_lock_path(entry)}",
        )
    try:
        if item.kind == "profile_efficiency":
            try:
                profile_entry_efficiency(
                    entry,
                    repo_root=repo_root,
                    warmup=efficiency_warmup,
                    repeats=efficiency_repeats,
                    throughput_batch_size=efficiency_throughput_batch_size,
                    max_records=efficiency_max_records,
                    overwrite=overwrite_efficiency,
                    log_file=log_path,
                )
            except subprocess.CalledProcessError as exc:
                return WorkResult(
                    run_name=entry["run_name"],
                    kind=item.kind,
                    status="failed",
                    return_code=int(exc.returncode),
                    log_file=str(log_path),
                    error_message=f"Efficiency profiling failed with return code {exc.returncode}",
                )
            return WorkResult(entry["run_name"], item.kind, "profiled", 0, str(log_path))

        return_code = _run_subprocess_to_log(entry["command"], cwd=repo_root, log_path=log_path)
        if return_code != 0:
            return WorkResult(
                run_name=entry["run_name"],
                kind=item.kind,
                status="failed",
                return_code=return_code,
                log_file=str(log_path),
                error_message=f"Training command failed with return code {return_code}",
            )
        if profile_efficiency:
            try:
                profile_entry_efficiency(
                    entry,
                    repo_root=repo_root,
                    warmup=efficiency_warmup,
                    repeats=efficiency_repeats,
                    throughput_batch_size=efficiency_throughput_batch_size,
                    max_records=efficiency_max_records,
                    overwrite=overwrite_efficiency,
                    log_file=log_path,
                )
            except subprocess.CalledProcessError as exc:
                return WorkResult(
                    run_name=entry["run_name"],
                    kind=item.kind,
                    status="failed",
                    return_code=int(exc.returncode),
                    log_file=str(log_path),
                    error_message=f"Efficiency profiling failed with return code {exc.returncode}",
                )
        return WorkResult(entry["run_name"], item.kind, "completed", 0, str(log_path))
    finally:
        release_run_lock(lock_path)


def apply_result_to_entry(result: WorkResult, entry: dict[str, Any]) -> None:
    entry["log_file_path"] = result.log_file or ""
    entry["return_code"] = result.return_code
    entry["error_message"] = result.error_message or ""
    if result.status == "profiled":
        entry["status"] = "profiled"
    elif result.status == "completed":
        entry["status"] = "completed"
    elif result.status in {"failed", "locked"}:
        entry["status"] = result.status


def run_manifest(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    jobs: int = 1,
    fail_fast: bool = False,
    keep_going: bool = False,
    profile_efficiency: bool = False,
    efficiency_warmup: int = 20,
    efficiency_repeats: int = 100,
    efficiency_throughput_batch_size: int = 16,
    efficiency_max_records: int | None = None,
    overwrite_efficiency: bool = False,
    stale_lock_minutes: float | None = None,
) -> None:
    if keep_going:
        fail_fast = False
    entries_by_name = {entry["run_name"]: entry for entry in manifest["entries"]}
    items = build_work_items(
        manifest,
        profile_efficiency=profile_efficiency,
        overwrite_efficiency=overwrite_efficiency,
    )
    total = len(manifest["entries"])
    if not items:
        print(f"No runnable work items. completed={sum(1 for e in manifest['entries'] if e.get('status') == 'completed')} total={total}")
        return

    if jobs == 1:
        completed = 0
        failed = 0
        for item in items:
            entry = item.entry
            index = manifest["entries"].index(entry) + 1
            print(f"[{index}/{total}] Starting {item.kind} {entry['run_name']}")
            result = _run_work_item(
                item,
                repo_root=repo_root,
                profile_efficiency=profile_efficiency,
                efficiency_warmup=efficiency_warmup,
                efficiency_repeats=efficiency_repeats,
                efficiency_throughput_batch_size=efficiency_throughput_batch_size,
                efficiency_max_records=efficiency_max_records,
                overwrite_efficiency=overwrite_efficiency,
                stale_lock_minutes=stale_lock_minutes,
            )
            apply_result_to_entry(result, entry)
            completed += result.status in {"completed", "profiled"}
            failed += result.status == "failed"
            print(
                f"Finished {entry['run_name']} status={result.status} "
                f"completed={completed} failed={failed} remaining={len(items) - completed - failed}"
            )
            if result.status == "failed" and fail_fast:
                raise subprocess.CalledProcessError(result.return_code or 1, entry["command"])
        return

    print(f"Starting parallel noisy-input work: jobs={jobs}, work_items={len(items)}, total_runs={total}")
    completed = 0
    failed = 0
    running = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_item = {}
        for item in items:
            future = executor.submit(
                _run_work_item,
                item,
                repo_root=repo_root,
                profile_efficiency=profile_efficiency,
                efficiency_warmup=efficiency_warmup,
                efficiency_repeats=efficiency_repeats,
                efficiency_throughput_batch_size=efficiency_throughput_batch_size,
                efficiency_max_records=efficiency_max_records,
                overwrite_efficiency=overwrite_efficiency,
                stale_lock_minutes=stale_lock_minutes,
            )
            future_to_item[future] = item
            running += 1
            print(f"START {item.kind} {item.entry['run_name']} running={min(running, jobs)}")

        first_failure: subprocess.CalledProcessError | None = None
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive: worker should normally return WorkResult
                result = WorkResult(
                    run_name=item.entry["run_name"],
                    kind=item.kind,
                    status="failed",
                    return_code=1,
                    log_file=str(Path(item.entry["output_dir"]) / "run.log"),
                    error_message=str(exc),
                )
            entry = entries_by_name[result.run_name]
            apply_result_to_entry(result, entry)
            completed += result.status in {"completed", "profiled"}
            failed += result.status == "failed"
            running -= 1
            print(
                f"FINISH {result.kind} {result.run_name} status={result.status} "
                f"completed={completed} running={max(running, 0)} failed={failed}"
            )
            if result.status == "failed" and fail_fast and first_failure is None:
                first_failure = subprocess.CalledProcessError(result.return_code or 1, entry["command"])
        if first_failure is not None:
            raise first_failure


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Write manifest and preflight only.")
    p.add_argument("--smoke", action="store_true", help="Use the small Mamba/BiLSTM smoke subset.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--models", nargs="+", default=None, choices=[spec.key for spec in MODEL_SPECS])
    p.add_argument("--noise-types", nargs="+", default=None, choices=list(NOISE_TYPES))
    p.add_argument("--snr-db", nargs="+", type=float, default=None)
    p.add_argument("--noise-root", default="data")
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--ecg-fs", type=float, default=100)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="Maximum independent conditions to run concurrently.")
    p.add_argument("--stale-lock-minutes", type=float, default=None)
    p.add_argument("--profile-efficiency", action="store_true")
    p.add_argument("--overwrite-efficiency", action="store_true")
    p.add_argument("--efficiency-warmup", type=int, default=20)
    p.add_argument("--efficiency-repeats", type=int, default=100)
    p.add_argument("--efficiency-throughput-batch-size", type=int, default=16)
    p.add_argument("--efficiency-max-records", type=int, default=None)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    return args


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    default_root = "runs/noisy_input_sweep_smoke" if args.smoke else "runs/noisy_input_sweep"
    output_root = Path(args.output_root or default_root)
    manifest_path = Path(args.manifest_path or output_root / "manifest.json")

    manifest = build_manifest(
        repo_root=repo_root,
        output_root=output_root,
        models=args.models,
        noise_types=args.noise_types,
        snr_db=args.snr_db,
        seed=args.base_seed,
        ecg_fs=args.ecg_fs,
        noise_root=args.noise_root,
        python_executable=args.python,
        smoke=args.smoke,
    )
    if args.resume:
        mark_resume_statuses(manifest)
    write_manifest(manifest, manifest_path)
    print(f"Wrote noisy-input sweep manifest: {manifest_path}")
    print(f"Planned runs: {len(manifest['entries'])}")
    validate_manifest(manifest, repo_root=repo_root, overwrite=args.overwrite, resume=args.resume)
    for entry in manifest["entries"]:
        command = entry["command"]
        print(
            f"Preflight ok: {entry['run_name']} -> one condition "
            f"{command[command.index('--noise-types') + 1]} / {command[command.index('--snr-db') + 1]} dB"
        )

    if args.dry_run:
        print("DRY-RUN complete: manifest written, no training launched.")
        return

    try:
        run_manifest(
            manifest,
            repo_root=repo_root,
            jobs=args.jobs,
            fail_fast=args.fail_fast,
            keep_going=args.keep_going,
            profile_efficiency=args.profile_efficiency,
            efficiency_warmup=args.efficiency_warmup,
            efficiency_repeats=args.efficiency_repeats,
            efficiency_throughput_batch_size=args.efficiency_throughput_batch_size,
            efficiency_max_records=args.efficiency_max_records,
            overwrite_efficiency=args.overwrite_efficiency,
            stale_lock_minutes=args.stale_lock_minutes,
        )
    finally:
        write_manifest(manifest, manifest_path)
        rows = collect_summary(manifest, output_root)
        print(f"Wrote summary: {output_root / 'summary.csv'} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
