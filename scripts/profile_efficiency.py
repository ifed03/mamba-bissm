#!/usr/bin/env python
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse

import torch

from data.datamodule import make_dataloaders
from data.splits import load_split
from evaluate.efficiency import (
    count_parameters,
    cpu_info,
    profile_record_latency,
    profile_window_latency,
    write_efficiency_outputs,
)
from models import build_model
from train.checkpointing import load_checkpoint
from utils.config import load_config


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

    cfg = load_config(args.config)
    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"holdout_seed{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)
    _, test_loader_for_window, test_loader = make_dataloaders(cfg, split)

    model = build_model(cfg)
    load_checkpoint(args.ckpt, model)
    device = torch.device(args.device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    first_batch = next(iter(test_loader_for_window))
    x_ref = first_batch["x"][0:1].to(device=device, dtype=torch.float32)
    x16 = x_ref.repeat(args.throughput_batch_size, 1)

    t1 = profile_window_latency(model, x_ref, args.warmup, args.repeats)
    t16 = profile_window_latency(model, x16, args.warmup, args.repeats)
    window_rows = (
        [{"batch_size": 1, "repeat_idx": i, "latency_ms": v} for i, v in enumerate(t1)]
        + [{"batch_size": args.throughput_batch_size, "repeat_idx": i, "latency_ms": v} for i, v in enumerate(t16)]
    )

    record_rows = profile_record_latency(model, test_loader, device, args.max_records)
    record_lat = [r["latency_ms"] for r in record_rows]

    run_dir = _run_dir_from_ckpt(args.ckpt)
    wps16 = float(args.throughput_batch_size / (sum(t16) / len(t16) / 1000.0))
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.ckpt).resolve()),
        "model_name": cfg.get("model", {}).get("name"),
        "backbone": cfg.get("model", {}).get("backbone"),
        "window_seconds": cfg["preprocessing"].get("window_seconds"),
        "stride_seconds": cfg["preprocessing"].get("stride_seconds"),
        "input_length_samples": int(x_ref.shape[-1]),
        "timing_device": str(device),
        "precision": "fp32",
        "warmup_iterations": args.warmup,
        "measured_repeats": args.repeats,
        "latency_batch_size": 1,
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
