#!/usr/bin/env python
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse

import torch

def _run_dir_from_ckpt(ckpt: str) -> Path:
    p = Path(ckpt)
    return p.parent.parent if p.parent.name == "checkpoints" else p.parent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--throughput-batch-size", type=int, default=16)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    from data.datamodule import make_dataloaders
    from data.splits import load_split
    from evaluate.efficiency import (
        count_parameters,
        cpu_info,
        efficiency_metadata_from_config,
        profile_record_latency,
        profile_window_latency,
        _repeat_batch,
        write_efficiency_outputs,
    )
    from models import build_model
    from train.checkpointing import load_checkpoint
    from utils.config import load_config

    cfg = load_config(args.config)
    device = torch.device(args.device)
    fast_path_overridden_for_cpu = False
    if device.type == "cpu":
        model_cfg = cfg.get("model", {}) or {}
        backbone = str(model_cfg.get("backbone", model_cfg.get("name", ""))).lower()
        if backbone in {"mamba", "bimamba"}:
            if model_cfg.get("use_fast_path", True):
                model_cfg["use_fast_path"] = False
                fast_path_overridden_for_cpu = True
            try:
                import mamba_ssm.modules.mamba_simple as mamba_simple
                from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

                mamba_simple.causal_conv1d_fn = None
                mamba_simple.selective_scan_fn = selective_scan_ref
            except Exception as exc:
                print(f"Warning: could not force Mamba CPU reference kernels: {exc}")
    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"holdout_seed{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)
    _, test_loader_for_window, test_loader = make_dataloaders(cfg, split)

    model = build_model(cfg)
    load_checkpoint(args.ckpt, model)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    first_batch = next(iter(test_loader_for_window))
    x_ref = first_batch["x"][0:1].to(device=device, dtype=torch.float32)
    x16 = _repeat_batch(x_ref, args.throughput_batch_size)

    t1 = profile_window_latency(model, x_ref, args.warmup, args.repeats)
    t16 = profile_window_latency(model, x16, args.warmup, args.repeats)
    window_rows = (
        [{"batch_size": 1, "repeat_idx": i, "latency_ms": v} for i, v in enumerate(t1)]
        + [{"batch_size": args.throughput_batch_size, "repeat_idx": i, "latency_ms": v} for i, v in enumerate(t16)]
    )

    record_rows = profile_record_latency(model, test_loader, device, args.max_records)
    record_lat = [r["latency_ms"] for r in record_rows]
    num_profiled_records = len(record_rows)
    num_profiled_windows = int(sum(r["num_windows"] for r in record_rows))

    run_dir = _run_dir_from_ckpt(args.ckpt)
    wps16 = float(args.throughput_batch_size / (sum(t16) / len(t16) / 1000.0))
    metadata = efficiency_metadata_from_config(cfg, x_ref)
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.ckpt).resolve()),
        "model_name": cfg.get("model", {}).get("name"),
        "backbone": cfg.get("model", {}).get("backbone"),
        **metadata,
        "timing_device": str(device),
        "device": str(device),
        "timing_scope": "model_forward_sigmoid_max_only_excludes_loader_tensor_transfer_io",
        "cpu_fast_path_override": fast_path_overridden_for_cpu,
        "precision": "fp32",
        "warmup_iterations": args.warmup,
        "num_warmup_batches": args.warmup,
        "warmup_passes": args.warmup,
        "measured_repeats": args.repeats,
        "timed_window_passes": args.repeats,
        "timed_passes": args.repeats,
        "timed_batches": num_profiled_records,
        "num_records": num_profiled_records,
        "num_windows": num_profiled_windows,
        "latency_batch_size": 1,
        "batch_size": 1,
        "throughput_batch_size": args.throughput_batch_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "mean_window_latency_ms_batch1": float(torch.tensor(t1).mean().item()),
        "std_window_latency_ms_batch1": float(torch.tensor(t1).std(unbiased=False).item()),
        "p50_window_latency_ms_batch1": float(torch.quantile(torch.tensor(t1), 0.5).item()),
        "p95_window_latency_ms_batch1": float(torch.quantile(torch.tensor(t1), 0.95).item()),
        "mean_window_latency_ms_batch16": float(torch.tensor(t16).mean().item()),
        "std_window_latency_ms_batch16": float(torch.tensor(t16).std(unbiased=False).item()),
        "p50_window_latency_ms_batch16": float(torch.quantile(torch.tensor(t16), 0.5).item()),
        "p95_window_latency_ms_batch16": float(torch.quantile(torch.tensor(t16), 0.95).item()),
        "windows_per_second_batch16": wps16,
        "mean_record_latency_ms": float(torch.tensor(record_lat).mean().item()),
        "std_record_latency_ms": float(torch.tensor(record_lat).std(unbiased=False).item()),
        "p50_record_latency_ms": float(torch.quantile(torch.tensor(record_lat), 0.5).item()),
        "p95_record_latency_ms": float(torch.quantile(torch.tensor(record_lat), 0.95).item()),
        "records_per_second": float(1.0 / (sum(record_lat) / len(record_lat) / 1000.0)),
        "cpu_info": cpu_info(),
        "window_input_source": "real_test_window",
    }
    write_efficiency_outputs(run_dir, payload, window_rows, record_rows)
    print(payload)
