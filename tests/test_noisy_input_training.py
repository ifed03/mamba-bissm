import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from data.datamodule import make_dataloaders
from evaluate.noise_protocol import (
    NOISY_INPUT_CHECKPOINT_SOURCE,
    NOISY_INPUT_THRESHOLD_SOURCE,
    NoiseCondition,
    NoisyInputNoiseInjector,
    condition_key,
    deterministic_example_seed,
    noisy_input_metrics_filename,
    noisy_input_threshold_filename,
    validate_nstdb_root,
)
from train.trainer import train_model


def _noise():
    t = np.linspace(0, 40 * np.pi, 2000)
    return np.column_stack([np.sin(t), np.cos(t)]).astype(np.float64)


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ecg.parquet"
    rows = []
    for split_name in ["train", "val", "test"]:
        for i, label in enumerate([0, 1]):
            phase = i + len(split_name)
            rows.append(
                {
                    "record_id": f"{split_name}_r{i}",
                    "x": (np.sin(np.linspace(0, 8 * np.pi, 200) + phase) + 0.2).tolist(),
                    "label": label,
                    "fs": 200,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _split():
    return {"train": [0, 1], "val": [2, 3], "test": [4, 5]}


def _cfg(data_path: Path):
    return {
        "paths": {"data_path": str(data_path), "splits_dir": "splits", "runs_dir": "runs"},
        "split": {"seed": 42},
        "preprocessing": {
            "fs_target": 100,
            "target_seconds": 1.0,
            "normalize": "none",
            "windowing": {"enabled": True, "window_seconds": 1.0, "stride_seconds": 1.0},
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "patience": 2,
            "grad_clip": 1.0,
            "mixed_precision": False,
            "weighted_sampler": False,
            "deterministic": True,
        },
    }


def _noise_cfg(base_seed=123, noise_type="bw", snr_db=0.0):
    return {
        "enabled": True,
        "mode": "noisy-input",
        "noise_type": noise_type,
        "snr_db": float(snr_db),
        "base_seed": base_seed,
        "target_fs": 100,
        "noise": _noise(),
        "noise_fs": 100,
    }


def test_noisy_input_noise_applies_to_all_splits_and_preserves_membership(tmp_path):
    data_path = _write_dataset(tmp_path)
    cfg = _cfg(data_path)
    split = _split()

    clean = make_dataloaders(cfg, split)
    noisy = make_dataloaders(cfg, split, noise_training_cfg=_noise_cfg(base_seed=7))

    required = {
        "record_id",
        "original_record_id",
        "split",
        "noise_type",
        "snr_db",
        "seed",
        "noise_channel",
        "noise_start_index",
        "measured_snr_db",
        "original_label",
        "threshold_source",
        "checkpoint_source",
        "processing_order",
    }

    for split_name, clean_loader, noisy_loader in zip(["train", "val", "test"], clean, noisy, strict=True):
        clean_ds = clean_loader.dataset
        noisy_ds = noisy_loader.dataset
        assert noisy_ds.noise_injector is not None
        assert len(noisy_ds.record_ids) == len(clean_ds.record_ids)
        assert len(noisy_ds) == len(clean_ds)
        assert noisy_ds.record_ids == clean_ds.record_ids
        assert noisy_ds.labels == clean_ds.labels
        assert noisy_ds.sample_record_ids == clean_ds.sample_record_ids
        assert noisy_ds.sample_labels == clean_ds.sample_labels
        assert len(noisy_ds.noise_metadata) == len(clean_ds.record_ids)
        assert required.issubset(noisy_ds.noise_metadata[0])
        assert noisy_ds.noise_metadata[0]["split"] == split_name
        assert noisy_ds.noise_metadata[0]["threshold_source"] == NOISY_INPUT_THRESHOLD_SOURCE
        assert noisy_ds.noise_metadata[0]["checkpoint_source"] == NOISY_INPUT_CHECKPOINT_SOURCE
        assert noisy_ds.noise_metadata[0]["processing_order"] == "resample->noise->window->normalize"
        assert not np.array_equal(noisy_ds._signals[0], clean_ds._signals[0])

    after = pd.read_parquet(data_path)
    assert after["record_id"].tolist() == [f"{s}_r{i}" for s in ["train", "val", "test"] for i in [0, 1]]
    assert after["label"].tolist() == [0, 1, 0, 1, 0, 1]


def test_noisy_input_seed_determinism_and_separable_conditions(tmp_path):
    data_path = _write_dataset(tmp_path)
    cfg = _cfg(data_path)
    split = _split()

    a = make_dataloaders(cfg, split, noise_training_cfg=_noise_cfg(base_seed=11, noise_type="bw", snr_db=24.0))
    b = make_dataloaders(cfg, split, noise_training_cfg=_noise_cfg(base_seed=11, noise_type="bw", snr_db=24.0))
    c = make_dataloaders(cfg, split, noise_training_cfg=_noise_cfg(base_seed=12, noise_type="bw", snr_db=24.0))
    d = make_dataloaders(cfg, split, noise_training_cfg=_noise_cfg(base_seed=11, noise_type="em", snr_db=-6.0))

    assert a[0].dataset.noise_metadata == b[0].dataset.noise_metadata
    assert np.array_equal(a[0].dataset._signals[0], b[0].dataset._signals[0])
    assert any(not np.array_equal(x, y) for x, y in zip(a[2].dataset._signals, c[2].dataset._signals, strict=True))
    assert a[0].dataset.noise_metadata[0]["seed"] == deterministic_example_seed(
        base_seed=11,
        record_id="train_r0",
        split="train",
        noise_type="bw",
        snr_db=24.0,
    )
    assert a[0].dataset.noise_metadata[0]["snr_db"] == 24.0
    assert d[0].dataset.noise_metadata[0]["noise_type"] == "em"
    assert d[0].dataset.noise_metadata[0]["snr_db"] == -6.0
    assert noisy_input_metrics_filename(NoiseCondition("bw", 24)) != noisy_input_metrics_filename(NoiseCondition("em", -6))


def test_noisy_input_validation_checkpoint_threshold_metadata_and_test_tau(tmp_path):
    data_path = _write_dataset(tmp_path)
    cfg = _cfg(data_path)
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg,
        _split(),
        noise_training_cfg=_noise_cfg(base_seed=5, noise_type="ma", snr_db=6.0),
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 1)

        def forward(self, x):
            flat = x.view(x.shape[0], -1)
            logit = self.linear(flat).reshape(-1)
            return logit, flat[:, :2]

    run_dir = tmp_path / "run"
    metrics = train_model(TinyModel(), train_loader, val_loader, test_loader, cfg, run_dir)
    condition = NoiseCondition("ma", 6.0)
    threshold_path = run_dir / noisy_input_threshold_filename(condition)
    metrics_path = run_dir / noisy_input_metrics_filename(condition)

    threshold_payload = json.loads(threshold_path.read_text())
    metrics_payload = json.loads(metrics_path.read_text())

    assert threshold_payload["threshold_source"] == NOISY_INPUT_THRESHOLD_SOURCE
    assert threshold_payload["checkpoint_source"] == NOISY_INPUT_CHECKPOINT_SOURCE
    assert threshold_payload["condition"] == condition_key(condition)
    assert metrics_payload["threshold"] == metrics["threshold"]
    assert metrics_payload["threshold_source"] == NOISY_INPUT_THRESHOLD_SOURCE
    assert metrics_payload["checkpoint_source"] == NOISY_INPUT_CHECKPOINT_SOURCE
    assert metrics_payload["noise_metadata"]["val"]
    assert metrics_payload["noise_metadata"]["test"]
    assert not (run_dir / "clean_validation_threshold.json").exists()


def test_noisy_input_validation_and_errors(tmp_path):
    with pytest.raises(ValueError, match="Invalid noise type"):
        NoisyInputNoiseInjector(noise_type="bad", snr_db=0, noise=_noise(), noise_fs=100)
    with pytest.raises(FileNotFoundError, match="noise_root does not exist"):
        validate_nstdb_root(tmp_path / "missing")
    injector = NoisyInputNoiseInjector(noise_type="bw", snr_db=6, noise=_noise(), noise_fs=100)
    clean = np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.1
    _, meta = injector.inject(clean, record_id="r", split="val")
    assert np.isclose(meta["measured_snr_db"], 6.0, atol=1e-8)


def test_noisy_input_dry_run_writes_nothing_and_reports_protocol(tmp_path):
    data_path = _write_dataset(tmp_path)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(_split()))
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        f"  data_path: {data_path}\n"
        "  splits_dir: splits\n"
        "  runs_dir: runs\n"
        "split:\n"
        "  seed: 42\n"
        "preprocessing:\n"
        "  fs_target: 100\n"
        "  target_seconds: 1.0\n"
        "  normalize: none\n"
        "training:\n"
        "  batch_size: 2\n"
    )
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/train_model.py",
        "--config",
        str(config),
        "--split",
        str(split_path),
        "--noise-training-mode",
        "noisy-input",
        "--noise-root",
        "data",
        "--noise-types",
        "bw",
        "em",
        "--snr-db",
        "24",
        "0",
        "--output-root",
        str(out),
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)

    assert "DRY-RUN complete" in result.stdout
    assert "Data materialisation: dynamic injection" in result.stdout
    assert "Checkpoint selection split: noisy validation" in result.stdout
    assert "Threshold tau* selection split: noisy validation" in result.stdout
    assert f"{out}/noisy_input_training_bw_24dB" in result.stdout
    assert f"{out}/noisy_input_training_bw_0dB" in result.stdout
    assert f"{out}/noisy_input_training_em_24dB" in result.stdout
    assert f"{out}/noisy_input_training_em_0dB" in result.stdout
    assert "metrics_noisy-input-training_noise_type=bw__snr_db=24.json" in result.stdout
    assert "metrics_noisy-input-training_noise_type=em__snr_db=0.json" in result.stdout
    assert "threshold_noisy-input-training_noise_type=bw__snr_db=24.json" in result.stdout
    assert "threshold_noisy-input-training_noise_type=em__snr_db=0.json" in result.stdout
    assert not out.exists()
