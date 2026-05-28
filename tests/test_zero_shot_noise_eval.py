import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from evaluate.noise_protocol import (
    DEFAULT_SNR_DB,
    NoiseCondition,
    ZeroShotNoiseInjector,
    deterministic_example_seed,
    load_clean_threshold,
    metrics_filename,
    validate_noise_type,
    validate_nstdb_root,
)


def _load_torch_dataset_modules():
    try:
        from data.datamodule import make_dataloaders
        from data.parquet_dataset import ParquetECGDataset
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyTorch-backed dataset tests skipped because torch failed to load: {exc}")
    return make_dataloaders, ParquetECGDataset


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ecg.parquet"
    df = pd.DataFrame(
        {
            "record_id": ["train_r", "val_r", "test_r0", "test_r1"],
            "x": [
                np.sin(np.linspace(0, 4 * np.pi, 200)).tolist(),
                np.cos(np.linspace(0, 4 * np.pi, 200)).tolist(),
                np.sin(np.linspace(0, 8 * np.pi, 200)).tolist(),
                np.cos(np.linspace(0, 8 * np.pi, 200)).tolist(),
            ],
            "label": [0, 1, 0, 1],
            "fs": [200, 200, 200, 200],
        }
    )
    df.to_parquet(path, index=False)
    return path


def _cfg(data_path: Path):
    return {
        "paths": {"data_path": str(data_path)},
        "preprocessing": {
            "fs_target": 100,
            "target_seconds": 1.0,
            "normalize": "none",
            "windowing": {"enabled": True, "window_seconds": 1.0, "stride_seconds": 1.0},
        },
        "training": {"batch_size": 2, "weighted_sampler": False},
    }


def _split():
    return {"train": [0], "val": [1], "test": [2, 3]}


def _noise():
    t = np.linspace(0, 20 * np.pi, 1000)
    return np.column_stack([np.sin(t), np.cos(t)]).astype(np.float64)


def _noise_cfg(base_seed=123, noise_type="bw", snr_db=0.0):
    return {
        "enabled": True,
        "noise_type": noise_type,
        "snr_db": snr_db,
        "base_seed": base_seed,
        "target_fs": 100,
        "noise": _noise(),
        "noise_fs": 100,
    }


def test_zero_shot_noise_is_test_only_and_preserves_clean_train_val(tmp_path):
    make_dataloaders, _ = _load_torch_dataset_modules()

    data_path = _write_dataset(tmp_path)
    cfg = _cfg(data_path)
    split = _split()

    clean_train, clean_val, clean_test = make_dataloaders(cfg, split)
    noisy_train, noisy_val, noisy_test = make_dataloaders(cfg, split, test_noise_cfg=_noise_cfg())

    assert noisy_train.dataset.noise_injector is None
    assert noisy_val.dataset.noise_injector is None
    assert noisy_test.dataset.noise_injector is not None
    assert np.array_equal(clean_train.dataset._signals[0], noisy_train.dataset._signals[0])
    assert np.array_equal(clean_val.dataset._signals[0], noisy_val.dataset._signals[0])
    assert len(noisy_test.dataset.record_ids) == len(clean_test.dataset.record_ids)
    assert noisy_test.dataset.record_ids == clean_test.dataset.record_ids
    assert noisy_test.dataset.labels == clean_test.dataset.labels
    assert not np.array_equal(noisy_test.dataset._signals[0], clean_test.dataset._signals[0])

    # The source parquet is not overwritten or rewritten by the noisy test loader.
    after = pd.read_parquet(data_path)
    assert after.loc[2, "record_id"] == "test_r0"
    assert after.loc[2, "label"] == 0


def test_metadata_fields_and_deterministic_per_example_seeds(tmp_path):
    make_dataloaders, _ = _load_torch_dataset_modules()

    data_path = _write_dataset(tmp_path)
    cfg = _cfg(data_path)
    split = _split()

    _, _, a = make_dataloaders(cfg, split, test_noise_cfg=_noise_cfg(base_seed=7))
    _, _, b = make_dataloaders(cfg, split, test_noise_cfg=_noise_cfg(base_seed=7))
    _, _, c = make_dataloaders(cfg, split, test_noise_cfg=_noise_cfg(base_seed=8))

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
    }

    assert required.issubset(a.dataset.noise_metadata[0])
    assert a.dataset.noise_metadata == b.dataset.noise_metadata
    assert np.array_equal(a.dataset._signals[0], b.dataset._signals[0])
    assert any(not np.array_equal(x, y) for x, y in zip(a.dataset._signals, c.dataset._signals, strict=True))
    assert a.dataset.noise_metadata[0]["seed"] == deterministic_example_seed(
        base_seed=7,
        record_id="test_r0",
        split="test",
        noise_type="bw",
        snr_db=0.0,
    )


def test_non_test_split_noise_config_fails_early(tmp_path):
    _, ParquetECGDataset = _load_torch_dataset_modules()

    data_path = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="only allowed for test split"):
        ParquetECGDataset(
            str(data_path),
            [0],
            train=False,
            preprocess_cfg=_cfg(data_path)["preprocessing"],
            split_name="val",
            noise_cfg=_noise_cfg(),
        )


def test_multiple_conditions_have_unambiguous_metrics_names_and_invalid_noise_fails():
    names = [metrics_filename(NoiseCondition(nt, snr)) for nt in ["bw", "em"] for snr in [24, -6]]

    assert len(names) == len(set(names))
    assert all(name.startswith("metrics_zero-shot_noise_type=") and name.endswith(".json") for name in names)
    assert DEFAULT_SNR_DB == [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]

    with pytest.raises(ValueError, match="Invalid noise type"):
        validate_noise_type("118e06")


def test_missing_nstdb_root_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="noise_root does not exist"):
        validate_nstdb_root(tmp_path / "missing")


def test_measured_snr_is_close_on_synthetic_example():
    injector = ZeroShotNoiseInjector(
        noise_type="ma",
        snr_db=6.0,
        base_seed=1,
        target_fs=100,
        noise=_noise(),
        noise_fs=100,
    )
    clean = np.sin(np.linspace(0, 6 * np.pi, 500)) + 0.1

    noisy, meta = injector.inject(clean, record_id="synthetic", split="test")

    assert noisy.shape == clean.shape
    assert np.isclose(meta["measured_snr_db"], 6.0, atol=1e-8)


def test_threshold_loading_requires_clean_val(tmp_path):
    path = tmp_path / "tau.json"

    path.write_text(json.dumps({"threshold": 0.37, "threshold_source": "clean_val"}))
    assert load_clean_threshold(path) == 0.37

    path.write_text(json.dumps({"threshold": 0.37, "threshold_source": "noisy_test"}))
    with pytest.raises(ValueError, match="clean_val"):
        load_clean_threshold(path)


def test_dry_run_zero_shot_evaluation_writes_nothing_and_missing_checkpoint_fails(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        "  data_path: data/cpsc_2018_labeled_single_lead.parquet\n"
        "  splits_dir: splits\n"
        "split:\n"
        "  seed: 42\n"
        "preprocessing:\n"
        "  fs_target: 100\n"
        "  target_seconds: 1.0\n"
        "training:\n"
        "  batch_size: 2\n"
    )

    out = tmp_path / "out"
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"placeholder")

    cmd = [
        sys.executable,
        "scripts/evaluate_model.py",
        "--config",
        str(config),
        "--checkpoint",
        str(ckpt),
        "--noise-eval",
        "zero-shot",
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

    result = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DRY-RUN complete" in result.stdout
    assert "metrics_zero-shot_noise_type=bw__snr_db=24.json" in result.stdout
    assert "metrics_zero-shot_noise_type=em__snr_db=0.json" in result.stdout
    assert not out.exists()

    missing = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_model.py",
            "--config",
            str(config),
            "--noise-eval",
            "zero-shot",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert missing.returncode != 0
    assert "required for zero-shot noise evaluation" in (missing.stderr + missing.stdout)
