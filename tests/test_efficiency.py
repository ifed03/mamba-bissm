import math

import torch

from evaluate.efficiency import (
    count_parameters,
    efficiency_metadata_from_config,
    profile_record_latency,
    profile_window_latency,
    _repeat_batch,
    _time_record_prediction,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(x), None


def test_parameter_count_known_model():
    m = TinyModel()
    total, trainable = count_parameters(m)
    assert total == 9
    assert trainable == 9


def test_cpu_timing_positive_finite():
    m = TinyModel().to(dtype=torch.float32)
    x = torch.randn(1, 8, dtype=torch.float32)
    vals = profile_window_latency(m, x, warmup=2, repeats=5)
    assert len(vals) == 5
    assert all(v > 0 and math.isfinite(v) for v in vals)


def test_batch_size_outputs_present():
    m = TinyModel().to(dtype=torch.float32)
    x1 = torch.randn(1, 8, dtype=torch.float32)
    x16 = torch.randn(16, 8, dtype=torch.float32)
    t1 = profile_window_latency(m, x1, warmup=1, repeats=3)
    t16 = profile_window_latency(m, x16, warmup=1, repeats=3)
    assert len(t1) == 3 and len(t16) == 3


def test_repeat_batch_preserves_non_batch_dimensions():
    x2 = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    y2 = _repeat_batch(x2, 4)
    assert y2.shape == (4, 8)
    assert torch.equal(y2[0], x2[0])
    assert torch.equal(y2[-1], x2[0])

    x3 = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    y3 = _repeat_batch(x3, 4)
    assert y3.shape == (4, 1, 8)
    assert torch.equal(y3[0], x3[0])
    assert torch.equal(y3[-1], x3[0])


def test_efficiency_metadata_reads_nested_windowing_config():
    cfg = {
        "preprocessing": {
            "fs_target": 100,
            "target_seconds": 4.0,
            "windowing": {
                "enabled": True,
                "window_seconds": 4.0,
                "stride_seconds": 2.0,
            },
        }
    }

    metadata = efficiency_metadata_from_config(cfg)

    assert metadata["window_seconds"] == 4.0
    assert metadata["stride_seconds"] == 2.0
    assert metadata["input_length_samples"] == 400


def test_record_level_profiling_rows_and_fields():
    m = TinyModel().to(dtype=torch.float32)

    loader = [
        {"x": torch.randn(2, 8), "record_id": ["r1"]},
        {"x": torch.randn(5, 8), "record_id": ["r2"]},
    ]
    rows = profile_record_latency(m, loader, torch.device("cpu"))
    assert len(rows) == 2
    for r, n in zip(rows, [2, 5]):
        assert set(["record_id", "num_windows", "latency_ms"]).issubset(r.keys())
        assert r["num_windows"] == n
        assert r["latency_ms"] > 0 and math.isfinite(r["latency_ms"])


def test_record_level_includes_sigmoid_and_maxpool(monkeypatch):
    m = TinyModel().to(dtype=torch.float32)
    called = {"sigmoid": 0, "max": 0}

    orig_sigmoid = torch.sigmoid
    orig_max = torch.Tensor.max

    def sig(x):
        called["sigmoid"] += 1
        return orig_sigmoid(x)

    def tmax(self, *args, **kwargs):
        called["max"] += 1
        return orig_max(self, *args, **kwargs)

    monkeypatch.setattr(torch, "sigmoid", sig)
    monkeypatch.setattr(torch.Tensor, "max", tmax)
    _ = profile_record_latency(m, [{"x": torch.randn(3, 8), "record_id": ["ra"]}], torch.device("cpu"))
    assert called["sigmoid"] >= 1
    assert called["max"] >= 1

from pathlib import Path
import json
from evaluate.efficiency import write_efficiency_outputs


def test_output_schema_keys(tmp_path: Path):
    required = {
        "config_path","checkpoint_path","model_name","window_seconds","stride_seconds","input_length_samples","timing_device",
        "precision","warmup_iterations","num_warmup_batches","warmup_passes","measured_repeats",
        "timed_window_passes","timed_passes","timed_batches","num_records","num_windows",
        "latency_batch_size","batch_size","throughput_batch_size","timing_scope",
        "total_parameters","trainable_parameters","mean_window_latency_ms_batch1","std_window_latency_ms_batch1",
        "p50_window_latency_ms_batch1","p95_window_latency_ms_batch1","mean_window_latency_ms_batch16",
        "std_window_latency_ms_batch16","p50_window_latency_ms_batch16","p95_window_latency_ms_batch16",
        "windows_per_second_batch16","mean_record_latency_ms","std_record_latency_ms","p50_record_latency_ms",
        "p95_record_latency_ms","records_per_second","window_input_source"
    }
    payload = {k: 1 for k in required}
    payload.update({"backbone": None, "cpu_info": {}, "config_path": "a", "checkpoint_path": "b", "model_name": "m", "timing_device": "cpu", "device": "cpu", "precision": "fp32"})
    write_efficiency_outputs(tmp_path, payload, [{"batch_size":1,"repeat_idx":0,"latency_ms":1.0}], [{"record_id":"r","num_windows":1,"latency_ms":1.0}])
    data = json.loads((tmp_path / "efficiency.json").read_text())
    assert required.issubset(data.keys())

import subprocess


def test_cli_smoke_help():
    res = subprocess.run(["python", "scripts/profile_efficiency.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--config" in res.stdout and "--ckpt" in res.stdout


def test_window_latency_csv_has_batch_label(tmp_path: Path):
    payload = {
        "config_path": "a", "checkpoint_path": "b", "model_name": "m", "window_seconds": 10,
        "stride_seconds": 2, "input_length_samples": 1000, "timing_device": "cpu", "device": "cpu", "precision": "fp32",
        "warmup_iterations": 1, "num_warmup_batches": 1, "warmup_passes": 1, "measured_repeats": 1, "timed_window_passes": 1, "timed_passes": 1, "timed_batches": 1, "num_records": 1, "num_windows": 1, "latency_batch_size": 1, "batch_size": 1, "throughput_batch_size": 16, "timing_scope": "model_forward_sigmoid_max_only_excludes_loader_tensor_transfer_io",
        "total_parameters": 1, "trainable_parameters": 1,
        "mean_window_latency_ms_batch1": 1.0, "std_window_latency_ms_batch1": 0.0, "p50_window_latency_ms_batch1": 1.0, "p95_window_latency_ms_batch1": 1.0,
        "mean_window_latency_ms_batch16": 1.0, "std_window_latency_ms_batch16": 0.0, "p50_window_latency_ms_batch16": 1.0, "p95_window_latency_ms_batch16": 1.0,
        "windows_per_second_batch16": 100.0, "mean_record_latency_ms": 1.0, "std_record_latency_ms": 0.0,
        "p50_record_latency_ms": 1.0, "p95_record_latency_ms": 1.0, "records_per_second": 10.0, "cpu_info": {}, "backbone": None
    }
    write_efficiency_outputs(
        tmp_path,
        payload,
        [{"batch_size": 1, "repeat_idx": 0, "latency_ms": 1.0}, {"batch_size": 16, "repeat_idx": 0, "latency_ms": 2.0}],
        [{"record_id": "r", "num_windows": 1, "latency_ms": 1.0}],
    )
    df = __import__("pandas").read_csv(tmp_path / "efficiency_window_latency.csv")
    assert ["batch_size", "repeat_idx", "latency_ms"] == list(df.columns)
    assert set(df["batch_size"].tolist()) == {1, 16}


def test_efficiency_json_includes_requested_fields(tmp_path: Path):
    payload = {
        "config_path": "a", "checkpoint_path": "b", "model_name": "m", "window_seconds": 10,
        "stride_seconds": 2, "input_length_samples": 1000, "timing_device": "cpu", "device": "cpu", "precision": "fp32",
        "warmup_iterations": 20, "num_warmup_batches": 20, "warmup_passes": 20, "measured_repeats": 100, "timed_window_passes": 100, "timed_passes": 100, "timed_batches": 1, "num_records": 1, "num_windows": 3, "latency_batch_size": 1, "batch_size": 1, "throughput_batch_size": 16, "timing_scope": "model_forward_sigmoid_max_only_excludes_loader_tensor_transfer_io",
        "total_parameters": 100, "trainable_parameters": 100,
        "mean_window_latency_ms_batch1": 1.0, "std_window_latency_ms_batch1": 0.1,
        "p50_window_latency_ms_batch1": 1.0, "p95_window_latency_ms_batch1": 1.2,
        "mean_window_latency_ms_batch16": 2.0, "std_window_latency_ms_batch16": 0.2,
        "p50_window_latency_ms_batch16": 2.0, "p95_window_latency_ms_batch16": 2.4,
        "windows_per_second_batch16": 800.0,
        "mean_record_latency_ms": 6.0, "std_record_latency_ms": 0.5,
        "p50_record_latency_ms": 5.9, "p95_record_latency_ms": 6.8,
        "records_per_second": 160.0,
        "window_input_source": "real_test_window",
        "cpu_info": {}, "backbone": None,
    }
    write_efficiency_outputs(
        tmp_path,
        payload,
        [{"batch_size": 1, "repeat_idx": 0, "latency_ms": 1.0}],
        [{"record_id": "r", "num_windows": 3, "latency_ms": 6.0}],
    )
    data = json.loads((tmp_path / "efficiency.json").read_text())
    required_exact = {
        "mean_window_latency_ms_batch1",
        "std_window_latency_ms_batch1",
        "p50_window_latency_ms_batch1",
        "p95_window_latency_ms_batch1",
        "mean_window_latency_ms_batch16",
        "windows_per_second_batch16",
        "mean_record_latency_ms",
        "p95_record_latency_ms",
        "records_per_second",
        "total_parameters",
        "trainable_parameters",
        "timing_device",
        "device",
        "precision",
        "timing_scope",
        "num_records",
        "num_windows",
        "num_warmup_batches",
        "timed_batches",
        "window_input_source",
    }
    assert required_exact.issubset(data.keys())


def test_record_timing_helper_positive_and_finite():
    m = TinyModel().to(dtype=torch.float32)
    x = torch.randn(4, 8, dtype=torch.float32)
    latency_ms = _time_record_prediction(m, x)
    assert latency_ms > 0 and math.isfinite(latency_ms)
