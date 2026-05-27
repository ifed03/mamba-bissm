from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _repeat_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if x.ndim == 0:
        raise ValueError("expected a batched tensor")
    return x.repeat(batch_size, *([1] * (x.ndim - 1)))


def efficiency_metadata_from_config(cfg: dict, x_ref: torch.Tensor | None = None) -> dict:
    preprocessing = cfg.get("preprocessing", {}) or {}
    windowing = preprocessing.get("windowing", {}) or {}

    window_seconds = windowing.get("window_seconds", preprocessing.get("target_seconds"))
    stride_seconds = windowing.get("stride_seconds", window_seconds)

    if x_ref is not None:
        input_length_samples = int(x_ref.shape[-1])
    else:
        target_seconds = preprocessing.get("target_seconds", window_seconds)
        fs_target = preprocessing.get("fs_target")
        input_length_samples = None
        if target_seconds is not None and fs_target is not None:
            input_length_samples = int(round(float(target_seconds) * float(fs_target)))

    return {
        "window_seconds": float(window_seconds) if window_seconds is not None else None,
        "stride_seconds": float(stride_seconds) if stride_seconds is not None else None,
        "input_length_samples": input_length_samples,
    }


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _summary(arr_ms: list[float], prefix: str) -> dict:
    vals = np.array(arr_ms, dtype=float)
    return {
        f"mean_{prefix}": float(vals.mean()),
        f"std_{prefix}": float(vals.std(ddof=0)),
        f"p50_{prefix}": float(np.percentile(vals, 50)),
        f"p95_{prefix}": float(np.percentile(vals, 95)),
    }


def profile_window_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> list[float]:
    timings = []
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
    return timings


def _time_record_prediction(model: torch.nn.Module, x_cpu_fp32: torch.Tensor) -> float:
    """Time one deployed record prediction: forward + sigmoid + max only."""
    t0 = time.perf_counter()
    logits, _ = model(x_cpu_fp32)
    probs = torch.sigmoid(logits)
    _ = probs.max()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def profile_record_latency(model: torch.nn.Module, test_loader, device: torch.device, max_records: int | None = None):
    rows = []
    model.eval()
    with torch.no_grad():
        for i, b in enumerate(test_loader):
            if max_records is not None and i >= max_records:
                break
            # Keep loader iteration and tensor movement outside timed scope.
            x = b["x"].to(device=device, dtype=torch.float32)
            rid = b.get("record_id", [f"record_{i}"])
            record_id = rid[0] if isinstance(rid, (list, tuple)) else str(rid)
            latency_ms = _time_record_prediction(model, x)
            rows.append({"record_id": record_id, "num_windows": int(x.shape[0]), "latency_ms": latency_ms})
    return rows


def cpu_info() -> dict:
    out = {"processor": platform.processor(), "machine": platform.machine(), "platform": platform.platform()}
    try:
        import os

        out["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return out


def write_efficiency_outputs(
    run_dir: Path,
    payload: dict,
    window_rows: list[dict],
    record_rows: list[dict],
):
    pd.DataFrame(window_rows).to_csv(run_dir / "efficiency_window_latency.csv", index=False)
    pd.DataFrame(record_rows).to_csv(run_dir / "efficiency_record_latency.csv", index=False)
    with (run_dir / "efficiency.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def validate_positive_finite(values: list[float]) -> bool:
    return all(math.isfinite(v) and v > 0 for v in values)
