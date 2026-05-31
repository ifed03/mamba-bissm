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
        config_path=(
            "final_configs/generated_clean_backbone_finish/"
            "binary_mamba_d128_n4_s64_slowpath_fp32_lr1e-4_100hz_win4s_stride2s.yaml"
        ),
        smoke_config_path="configs/smoke_ecgmamba_mamba_ssm_reduced_fp32_win4s_3epoch.yaml",
    ),
    ModelSpec(
        key="ecgmamba_bimamba",
        model_family="ecgmamba",
        backbone="bimamba",
        config_path="final_configs/binary_bimamba_d128_n2_s64_slowpath_fp32_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="ecgmamba_bilstm",
        model_family="ecgmamba",
        backbone="bilstm",
        config_path=(
            "final_configs/generated_clean_backbone_finish/"
            "binary_ecgmamba_bilstm_d128_h64_n2_fp32_100hz_win4s_stride2s.yaml"
        ),
    ),
    ModelSpec(
        key="ecgmamba_bissm",
        model_family="ecgmamba",
        backbone="bissm",
        config_path="final_configs/binary_bissm_d64_n2_s32_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="cnn1d",
        model_family="cnn1d",
        backbone="baseline",
        config_path="final_configs/binary_cnn1d_c256_n3_k7_100hz_win4s_stride2s.yaml",
    ),
    ModelSpec(
        key="bilstm",
        model_family="bilstm",
        backbone="baseline",
        config_path="final_configs/binary_bilstm_100hz_win4s_stride2s.yaml",
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
    "efficiency_profile_key",
    "efficiency_profile_source_run_name",
    "efficiency_file_path",
    "log_file_path",
    "return_code",
    "error_message",
    "status",
)


CONTEXT_LENGTH_SUMMARY_FIELDS = (
    "model_family",
    "backbone",
    "dimensions",
    "efficiency_profile_key",
    "source_run_name",
    "source_checkpoint_path",
    "window_seconds",
    "input_length_samples",
    "stride_seconds",
    "num_trainable_params",
    "mean_window_latency_ms_batch1",
    "p50_window_latency_ms_batch1",
    "p95_window_latency_ms_batch1",
    "mean_record_latency_ms",
    "windows_per_second_batch16",
    "records_per_second",
    "timing_device",
    "timing_scope",
    "warmup_iterations",
    "measured_repeats",
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


def _format_seconds_token(value: Any) -> str:
    try:
        raw = f"{float(value):g}"
    except (TypeError, ValueError):
        raw = str(value)
    return raw.replace(".", "p")


def _safe_key_part(value: Any) -> str:
    text = str(value).strip().lower() or "unknown"
    allowed = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "unknown"


def efficiency_profile_fields(
    *,
    entry: dict[str, Any],
    cfg: dict[str, Any],
    config_path: str,
    timing_device: str = "cpu",
    precision: str = "fp32",
) -> dict[str, Any]:
    preprocessing = cfg.get("preprocessing", {}) or {}
    windowing = preprocessing.get("windowing", {}) or {}
    fs_target = preprocessing.get("fs_target", entry.get("ecg_fs", 100))
    window_seconds = windowing.get("window_seconds", preprocessing.get("target_seconds", "unknown"))
    stride_seconds = windowing.get("stride_seconds", window_seconds)
    try:
        input_length_samples: Any = int(round(float(fs_target) * float(window_seconds)))
    except (TypeError, ValueError):
        input_length_samples = "unknown"
    architecture_id = Path(config_path).stem
    profile_key = "__".join(
        [
            f"model-{_safe_key_part(entry.get('model_family'))}",
            f"backbone-{_safe_key_part(entry.get('backbone'))}",
            f"dims-{_safe_key_part(entry.get('dimensions'))}",
            f"arch-{_safe_key_part(architecture_id)}",
            f"input-{_safe_key_part(input_length_samples)}",
            f"win-{_format_seconds_token(window_seconds)}s",
            f"stride-{_format_seconds_token(stride_seconds)}s",
            f"device-{_safe_key_part(timing_device)}",
            f"precision-{_safe_key_part(precision)}",
        ]
    )
    return {
        "efficiency_architecture_id": architecture_id,
        "input_length_samples": input_length_samples,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "timing_device": timing_device,
        "precision": precision,
        "efficiency_profile_key": profile_key,
    }


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
                entry = {
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
                entry.update(efficiency_profile_fields(entry=entry, cfg=cfg, config_path=cfg_rel))
                entries.append(entry)

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
    metrics_path = Path(entry["expected_metrics_file"])
    threshold_path = Path(entry["expected_threshold_file"])
    checkpoint_path = Path(entry["expected_checkpoint_file"])
    if not (metrics_path.is_file() and threshold_path.is_file() and checkpoint_path.is_file()):
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    metadata = metrics.get("noise_metadata", {})
    if not isinstance(metadata, dict):
        return False
    for split in ("train", "val", "test"):
        split_metadata = metadata.get(split)
        if not split_metadata:
            return False
        if not all(item.get("processing_order") == PROCESSING_ORDER for item in split_metadata):
            return False
    return True


def _efficiency_complete(entry: dict[str, Any]) -> bool:
    return (Path(entry["output_dir"]) / "efficiency.json").is_file()


def mark_resume_statuses(manifest: dict[str, Any]) -> None:
    for entry in manifest["entries"]:
        if entry.get("status") == "planned" and _entry_complete(entry):
            entry["status"] = "completed"


def shared_efficiency_dir(output_root: Path, profile_key: str) -> Path:
    return output_root / "efficiency_profiles" / _safe_key_part(profile_key)


def _profile_representative_score(indexed_entry: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
    index, entry = indexed_entry
    return (
        0 if float(entry.get("snr_db", 9999)) == 18.0 else 1,
        0 if entry.get("noise_type") == "bw" else 1,
        index,
    )


def assign_efficiency_profile_sources(
    manifest: dict[str, Any],
    output_root: Path,
    *,
    per_condition: bool = False,
) -> None:
    if per_condition:
        for entry in manifest["entries"]:
            profile_key = entry.get("efficiency_profile_key", "")
            entry["efficiency_profile_source_run_name"] = entry["run_name"]
            entry["efficiency_profile_file"] = str(Path(entry["output_dir"]) / "efficiency.json")
            entry["efficiency_profile_key"] = profile_key
        return

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, entry in enumerate(manifest["entries"]):
        profile_key = entry.get("efficiency_profile_key")
        if profile_key:
            groups.setdefault(str(profile_key), []).append((index, entry))

    for profile_key, indexed_entries in groups.items():
        completed = [(index, entry) for index, entry in indexed_entries if _entry_complete(entry)]
        source_entry = min(completed, key=_profile_representative_score)[1] if completed else None
        profile_file = shared_efficiency_dir(output_root, profile_key) / "efficiency.json"
        for _, entry in indexed_entries:
            entry["efficiency_profile_file"] = str(profile_file)
            entry["efficiency_profile_source_run_name"] = source_entry["run_name"] if source_entry else ""


def _shared_efficiency_complete(entry: dict[str, Any]) -> bool:
    profile_file = entry.get("efficiency_profile_file")
    return bool(profile_file) and Path(profile_file).is_file()


def _is_efficiency_representative(entry: dict[str, Any]) -> bool:
    return entry.get("efficiency_profile_source_run_name") == entry.get("run_name")


def _entry_stride_seconds(entry: dict[str, Any], fallback_window_seconds: float | None = None) -> Any:
    stride = entry.get("stride_seconds")
    if stride not in {None, ""}:
        return stride
    if fallback_window_seconds is not None:
        return fallback_window_seconds
    return entry.get("window_seconds", "unknown")


def efficiency_profile_key_for_window(
    entry: dict[str, Any],
    *,
    window_seconds: float,
    timing_device: str = "cpu",
    precision: str = "fp32",
) -> tuple[str, int, Any]:
    fs_target = entry.get("ecg_fs", 100)
    input_length_samples = int(round(float(fs_target) * float(window_seconds)))
    stride_seconds = _entry_stride_seconds(entry, fallback_window_seconds=window_seconds)
    architecture_id = entry.get("efficiency_architecture_id") or Path(str(entry.get("config_path", "unknown"))).stem
    profile_key = "__".join(
        [
            f"model-{_safe_key_part(entry.get('model_family'))}",
            f"backbone-{_safe_key_part(entry.get('backbone'))}",
            f"dims-{_safe_key_part(entry.get('dimensions'))}",
            f"arch-{_safe_key_part(architecture_id)}",
            f"input-{_safe_key_part(input_length_samples)}",
            f"win-{_format_seconds_token(window_seconds)}s",
            f"stride-{_format_seconds_token(stride_seconds)}s",
            f"device-{_safe_key_part(timing_device)}",
            f"precision-{_safe_key_part(precision)}",
        ]
    )
    return profile_key, input_length_samples, stride_seconds


def _normalise_window_seconds(values: list[float] | None) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        window = float(value)
        if window <= 0:
            raise ValueError("--efficiency-window-seconds values must be positive")
        key = round(window, 9)
        if key not in seen:
            result.append(window)
            seen.add(key)
    return result


def build_efficiency_context_profile_entries(
    manifest: dict[str, Any],
    output_root: Path,
    window_seconds: list[float],
) -> list[dict[str, Any]]:
    windows = _normalise_window_seconds(window_seconds)
    if not windows:
        return []
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, entry in enumerate(manifest["entries"]):
        if _entry_complete(entry):
            groups.setdefault(str(entry.get("efficiency_profile_key", entry["run_name"])), []).append((index, entry))

    context_entries: list[dict[str, Any]] = []
    for _, indexed_entries in groups.items():
        source = min(indexed_entries, key=_profile_representative_score)[1]
        for window in windows:
            profile_key, input_length_samples, stride_seconds = efficiency_profile_key_for_window(source, window_seconds=window)
            profile_file = output_root / "efficiency_profiles_context_length" / _safe_key_part(profile_key) / "efficiency.json"
            context_entry = dict(source)
            context_entry.update(
                {
                    "efficiency_profile_key": profile_key,
                    "efficiency_profile_file": str(profile_file),
                    "efficiency_profile_source_run_name": source["run_name"],
                    "efficiency_window_seconds_override": float(window),
                    "efficiency_input_length_samples": input_length_samples,
                    "efficiency_stride_seconds": stride_seconds,
                    "efficiency_profile_mode": "context_length_scaling",
                    "status": "completed",
                    "log_file_path": "",
                    "return_code": "",
                    "error_message": "",
                }
            )
            context_entries.append(context_entry)
    manifest["efficiency_context_profile_entries"] = context_entries
    return context_entries


def _lock_path(entry: dict[str, Any]) -> Path:
    return Path(entry["output_dir"]) / LOCK_NAME


def acquire_run_lock(
    entry: dict[str, Any],
    *,
    stale_lock_minutes: float | None = None,
    lock_dir: Path | None = None,
) -> Path | None:
    run_dir = lock_dir or Path(entry["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / LOCK_NAME
    if lock_path.exists() and stale_lock_minutes is not None:
        age_minutes = (time.time() - lock_path.stat().st_mtime) / 60.0
        if age_minutes >= stale_lock_minutes:
            lock_path.unlink()
    payload = json.dumps(
        {
            "run_name": entry["run_name"],
            "efficiency_profile_key": entry.get("efficiency_profile_key", ""),
            "efficiency_window_seconds_override": entry.get("efficiency_window_seconds_override", ""),
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
    efficiency_path = Path(entry.get("efficiency_profile_file") or (run_dir / "efficiency.json"))
    record_latency_path = efficiency_path.parent / "efficiency_record_latency.csv"
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
        "efficiency_profile_key": entry.get("efficiency_profile_key", ""),
        "efficiency_profile_source_run_name": entry.get("efficiency_profile_source_run_name", ""),
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


def parse_efficiency_context_row(entry: dict[str, Any]) -> dict[str, Any]:
    efficiency_path = Path(entry.get("efficiency_profile_file", ""))
    row = {
        "model_family": entry.get("model_family", ""),
        "backbone": entry.get("backbone", ""),
        "dimensions": entry.get("dimensions", ""),
        "efficiency_profile_key": entry.get("efficiency_profile_key", ""),
        "source_run_name": entry.get("efficiency_profile_source_run_name", entry.get("run_name", "")),
        "source_checkpoint_path": entry.get("expected_checkpoint_file", ""),
        "window_seconds": entry.get("efficiency_window_seconds_override", entry.get("window_seconds", "")),
        "input_length_samples": entry.get("efficiency_input_length_samples", entry.get("input_length_samples", "")),
        "stride_seconds": entry.get("efficiency_stride_seconds", entry.get("stride_seconds", "")),
        "efficiency_file_path": str(efficiency_path) if efficiency_path.is_file() else str(efficiency_path),
        "log_file_path": entry.get("log_file_path", ""),
        "return_code": entry.get("return_code", ""),
        "error_message": entry.get("error_message", ""),
        "status": entry.get("status", "planned"),
    }
    if entry.get("status") in {"failed", "locked", "incompatible_input_length"}:
        return row
    if not efficiency_path.is_file():
        row["status"] = "planned"
        return row
    try:
        efficiency = json.loads(efficiency_path.read_text())
    except json.JSONDecodeError as exc:
        row["status"] = "failed"
        row["error_message"] = f"Invalid efficiency JSON: {exc}"
        return row
    row.update(
        {
            "num_trainable_params": efficiency.get("trainable_parameters", ""),
            "mean_window_latency_ms_batch1": efficiency.get("mean_window_latency_ms_batch1", ""),
            "p50_window_latency_ms_batch1": efficiency.get("p50_window_latency_ms_batch1", ""),
            "p95_window_latency_ms_batch1": efficiency.get("p95_window_latency_ms_batch1", ""),
            "mean_record_latency_ms": efficiency.get("mean_record_latency_ms", ""),
            "windows_per_second_batch16": efficiency.get("windows_per_second_batch16", ""),
            "records_per_second": efficiency.get("records_per_second", ""),
            "timing_device": efficiency.get("timing_device", efficiency.get("device", "")),
            "timing_scope": efficiency.get("timing_scope", ""),
            "warmup_iterations": efficiency.get("warmup_iterations", ""),
            "measured_repeats": efficiency.get("measured_repeats", ""),
            "window_seconds": efficiency.get("window_seconds", row["window_seconds"]),
            "input_length_samples": efficiency.get("input_length_samples", row["input_length_samples"]),
            "stride_seconds": efficiency.get("stride_seconds", row["stride_seconds"]),
            "status": "success",
        }
    )
    return row


def collect_efficiency_context_length_summary(
    manifest: dict[str, Any],
    output_root: Path,
    window_seconds: list[float],
) -> list[dict[str, Any]]:
    entries = manifest.get("efficiency_context_profile_entries")
    if entries is None:
        entries = build_efficiency_context_profile_entries(manifest, output_root, window_seconds)
    rows = [parse_efficiency_context_row(entry) for entry in entries]
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "efficiency_context_length_summary.csv"
    json_path = output_root / "efficiency_context_length_summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONTEXT_LENGTH_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CONTEXT_LENGTH_SUMMARY_FIELDS})
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


def efficiency_command(
    entry: dict[str, Any],
    *,
    warmup: int,
    repeats: int,
    throughput_batch_size: int,
    max_records: int | None,
    output_dir: Path | None = None,
    window_seconds: float | None = None,
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
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if window_seconds is not None:
        cmd.extend(["--window-seconds", f"{float(window_seconds):g}"])
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
    output_dir: Path | None = None,
    window_seconds: float | None = None,
) -> None:
    run_dir = Path(entry["output_dir"])
    target_dir = output_dir or run_dir
    efficiency_path = target_dir / "efficiency.json"
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
        output_dir=output_dir,
        window_seconds=window_seconds,
    )
    print(f"Profiling CPU efficiency: source={run_dir} output={target_dir}", flush=True)
    if log_file is None:
        subprocess.run(cmd, cwd=repo_root, check=True)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_file.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, cwd=repo_root, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if output_dir is not None and efficiency_path.is_file():
        payload = json.loads(efficiency_path.read_text())
        payload.update(
            {
                "efficiency_profile_key": entry.get("efficiency_profile_key", ""),
                "efficiency_profile_source_run_name": entry.get("run_name", ""),
                "efficiency_profile_source_output_dir": str(run_dir),
                "efficiency_profile_mode": entry.get("efficiency_profile_mode", "shared_model_config"),
                "latency_scaling_only": bool(entry.get("efficiency_window_seconds_override")),
            }
        )
        if entry.get("efficiency_window_seconds_override") is not None:
            payload["window_seconds_override"] = float(entry["efficiency_window_seconds_override"])
            payload["latency_scaling_note"] = (
                "CPU inference latency scaling by input length only; not a trained/evaluated noisy-input performance metric."
            )
        efficiency_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_work_items(
    manifest: dict[str, Any],
    *,
    profile_efficiency: bool = False,
    overwrite_efficiency: bool = False,
    profile_efficiency_per_condition: bool = False,
    efficiency_window_seconds: list[float] | None = None,
) -> list[WorkItem]:
    output_root = Path(str(manifest.get("output_root", ".")))
    context_windows = _normalise_window_seconds(efficiency_window_seconds)
    if profile_efficiency and context_windows:
        context_entries = build_efficiency_context_profile_entries(manifest, output_root, context_windows)
        items: list[WorkItem] = []
        scheduled_profile_keys: set[str] = set()
        for entry in context_entries:
            profile_key = str(entry.get("efficiency_profile_key", entry["run_name"]))
            if profile_key in scheduled_profile_keys:
                continue
            if overwrite_efficiency or not _shared_efficiency_complete(entry):
                items.append(WorkItem("profile_efficiency", entry))
            scheduled_profile_keys.add(profile_key)
        return items

    if profile_efficiency and not profile_efficiency_per_condition:
        assign_efficiency_profile_sources(manifest, output_root)
    items: list[WorkItem] = []
    scheduled_profile_keys: set[str] = set()
    for entry in manifest["entries"]:
        if entry.get("status") not in {"planned", "completed"}:
            continue
        complete = _entry_complete(entry)
        if complete:
            entry["status"] = "completed"
            if profile_efficiency:
                if profile_efficiency_per_condition:
                    if overwrite_efficiency or not _efficiency_complete(entry):
                        items.append(WorkItem("profile_efficiency", entry))
                elif _is_efficiency_representative(entry):
                    profile_key = str(entry.get("efficiency_profile_key", entry["run_name"]))
                    if profile_key not in scheduled_profile_keys and (overwrite_efficiency or not _shared_efficiency_complete(entry)):
                        items.append(WorkItem("profile_efficiency", entry))
                        scheduled_profile_keys.add(profile_key)
            continue
        if entry.get("status") == "planned":
            items.append(WorkItem("train_condition", entry))
    return items


def _run_subprocess_to_log(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
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
    profile_efficiency_per_condition: bool = False,
) -> WorkResult:
    entry = item.entry
    run_dir = Path(entry["output_dir"])
    profile_output_dir = None
    if item.kind == "profile_efficiency" and not profile_efficiency_per_condition and entry.get("efficiency_profile_file"):
        profile_output_dir = Path(entry["efficiency_profile_file"]).parent
    log_path = (profile_output_dir or run_dir) / ("efficiency.log" if item.kind == "profile_efficiency" else "run.log")
    lock_path = acquire_run_lock(entry, stale_lock_minutes=stale_lock_minutes, lock_dir=profile_output_dir)
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
                    output_dir=profile_output_dir,
                    window_seconds=entry.get("efficiency_window_seconds_override"),
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
        if profile_efficiency and profile_efficiency_per_condition:
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
    efficiency_warmup: int = 5,
    efficiency_repeats: int = 10,
    efficiency_throughput_batch_size: int = 16,
    efficiency_max_records: int | None = None,
    overwrite_efficiency: bool = False,
    stale_lock_minutes: float | None = None,
    profile_efficiency_per_condition: bool = False,
    efficiency_window_seconds: list[float] | None = None,
) -> None:
    if keep_going:
        fail_fast = False
    items = build_work_items(
        manifest,
        profile_efficiency=profile_efficiency,
        overwrite_efficiency=overwrite_efficiency,
        profile_efficiency_per_condition=profile_efficiency_per_condition,
        efficiency_window_seconds=efficiency_window_seconds,
    )
    total = len(manifest["entries"])
    status_counts = {status: sum(1 for e in manifest["entries"] if e.get("status") == status) for status in {"planned", "completed", "failed", "locked", "profiled", "missing_config"}}
    train_items = sum(1 for item in items if item.kind == "train_condition")
    profile_items = sum(1 for item in items if item.kind == "profile_efficiency")
    skipped_completed = status_counts.get("completed", 0) - profile_items
    print(
        "Resume/work summary: "
        f"completed_skip={max(skipped_completed, 0)}, train={train_items}, "
        f"profile={profile_items}, other_statuses={status_counts}"
    )
    if not items:
        print(f"No runnable work items. completed={sum(1 for e in manifest['entries'] if e.get('status') == 'completed')} total={total}")
        return

    if jobs == 1:
        completed = 0
        failed = 0
        for index, item in enumerate(items, start=1):
            entry = item.entry
            print(f"[{index}/{len(items)}] Starting {item.kind} {entry['run_name']}")
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
                profile_efficiency_per_condition=profile_efficiency_per_condition,
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
                profile_efficiency_per_condition=profile_efficiency_per_condition,
            )
            future_to_item[future] = item
            running += 1
            print(f"QUEUE {item.kind} {item.entry['run_name']} queued={running}/{len(items)} max_jobs={jobs}")

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
            entry = item.entry
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
    p.add_argument("--profile-efficiency-per-condition", action="store_true", help="Profile every noisy condition instead of one representative per model config.")
    p.add_argument("--overwrite-efficiency", action="store_true")
    p.add_argument("--efficiency-warmup", type=int, default=5)
    p.add_argument("--efficiency-repeats", type=int, default=10)
    p.add_argument("--efficiency-window-seconds", nargs="+", type=float, default=None)
    p.add_argument("--efficiency-throughput-batch-size", type=int, default=16)
    p.add_argument("--efficiency-max-records", type=int, default=None)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    if args.efficiency_window_seconds and any(value <= 0 for value in args.efficiency_window_seconds):
        p.error("--efficiency-window-seconds values must be positive")
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
    assign_efficiency_profile_sources(
        manifest,
        output_root,
        per_condition=args.profile_efficiency_per_condition,
    )
    write_manifest(manifest, manifest_path)
    print(f"Wrote noisy-input sweep manifest: {manifest_path}")
    print(f"Planned runs: {len(manifest['entries'])}")
    validate_manifest(manifest, repo_root=repo_root, overwrite=args.overwrite, resume=args.resume)
    status_counts = {status: sum(1 for e in manifest["entries"] if e.get("status") == status) for status in {"planned", "completed", "failed", "locked", "profiled", "missing_config"}}
    print(f"Manifest status summary: {status_counts}")
    verbose_preflight = args.dry_run and not args.resume
    for entry in manifest["entries"]:
        if not verbose_preflight and entry.get("status") != "planned":
            continue
        command = entry["command"]
        print(
            f"Preflight ok: {entry['run_name']} -> one condition "
            f"{command[command.index('--noise-types') + 1]} / {command[command.index('--snr-db') + 1]} dB"
        )

    if args.dry_run:
        dry_items = build_work_items(
            manifest,
            profile_efficiency=args.profile_efficiency,
            overwrite_efficiency=args.overwrite_efficiency,
            profile_efficiency_per_condition=args.profile_efficiency_per_condition,
            efficiency_window_seconds=args.efficiency_window_seconds,
        )
        dry_train = sum(1 for item in dry_items if item.kind == "train_condition")
        dry_profiles = sum(1 for item in dry_items if item.kind == "profile_efficiency")
        mode = "context-length" if args.efficiency_window_seconds else ("per-condition" if args.profile_efficiency_per_condition else "shared-profile")
        print(
            "DRY-RUN work summary: "
            f"train={dry_train}, efficiency_profiles={dry_profiles}, "
            f"efficiency_mode={mode}, warmup={args.efficiency_warmup}, repeats={args.efficiency_repeats}"
        )
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
            profile_efficiency_per_condition=args.profile_efficiency_per_condition,
            efficiency_window_seconds=args.efficiency_window_seconds,
        )
    finally:
        assign_efficiency_profile_sources(
            manifest,
            output_root,
            per_condition=args.profile_efficiency_per_condition,
        )
        write_manifest(manifest, manifest_path)
        rows = collect_summary(manifest, output_root)
        print(f"Wrote summary: {output_root / 'summary.csv'} ({len(rows)} row(s))")
        if args.efficiency_window_seconds:
            context_rows = collect_efficiency_context_length_summary(manifest, output_root, args.efficiency_window_seconds)
            print(
                f"Wrote context-length efficiency summary: "
                f"{output_root / 'efficiency_context_length_summary.csv'} ({len(context_rows)} row(s))"
            )


if __name__ == "__main__":
    main()
