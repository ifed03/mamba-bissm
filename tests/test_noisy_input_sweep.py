import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_noisy_input_sweep as sweep


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path, **kwargs):
    return sweep.build_manifest(
        repo_root=REPO_ROOT,
        output_root=tmp_path / "runs",
        python_executable="python",
        **kwargs,
    )


def _base_entry(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0])
    return manifest["entries"][0]


def _noise_metadata():
    row = {"processing_order": sweep.PROCESSING_ORDER}
    return {"train": [row], "val": [row], "test": [row]}


def _write_success_metrics(entry, *, threshold_source="noisy_val", checkpoint_source="noisy_val", metadata=None):
    metrics_path = Path(entry["expected_metrics_file"])
    threshold_path = Path(entry["expected_threshold_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": entry["run_name"],
        "best_epoch": 2,
        "best_val_metric_name": "auroc",
        "threshold": 0.42,
        "threshold_source": threshold_source,
        "checkpoint_source": checkpoint_source,
        "noise_type": entry["noise_type"],
        "snr_db": entry["snr_db"],
        "val": {"auroc": 0.81},
        "test": {
            "auroc": 0.75,
            "auprc": 0.76,
            "f1": 0.7,
            "accuracy": 0.72,
            "sensitivity": 0.73,
            "specificity": 0.71,
        },
        "noise_metadata": _noise_metadata() if metadata is None else metadata,
    }
    metrics_path.write_text(json.dumps(payload))
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": 0.42,
                "threshold_source": threshold_source,
                "checkpoint_source": checkpoint_source,
                "checkpoint": str(Path(entry["output_dir"]) / "checkpoints" / "best.ckpt"),
            }
        )
    )


def test_run_names_are_unique():
    manifest = _manifest(Path("/tmp/noisy-test"))
    names = [entry["run_name"] for entry in manifest["entries"]]
    assert len(names) == 108
    assert len(names) == len(set(names))


def test_negative_snr_is_encoded_safely(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_bissm"], noise_types=["bw"], snr_db=[-6.0])
    entry = manifest["entries"][0]
    assert entry["run_name"].endswith("bw_neg6dB")
    assert "-6dB" not in entry["run_name"]
    assert "snr_db=neg6" in entry["expected_metrics_file"]


def test_each_generated_command_has_exactly_one_noise_type_and_snr(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw", "em"], snr_db=[24.0, 0.0])
    for entry in manifest["entries"]:
        cmd = entry["command"]
        assert cmd.count("--noise-types") == 1
        assert cmd.count("--snr-db") == 1
        assert cmd[cmd.index("--noise-types") + 2] == "--snr-db"
        assert cmd[cmd.index("--snr-db") + 2] == "--base-seed"


def test_output_directories_are_condition_specific(tmp_path):
    manifest = _manifest(tmp_path, models=["bilstm"], noise_types=["bw", "ma"], snr_db=[18.0, -6.0])
    output_dirs = [entry["output_dir"] for entry in manifest["entries"]]
    assert len(output_dirs) == len(set(output_dirs))
    assert any("bw_18dB" in path for path in output_dirs)
    assert any("ma_neg6dB" in path for path in output_dirs)


def test_missing_config_files_are_reported_clearly(tmp_path):
    manifest = sweep.build_manifest(
        repo_root=tmp_path,
        output_root=tmp_path / "runs",
        models=["ecgmamba_mamba"],
        noise_types=["bw"],
        snr_db=[18.0],
    )
    assert manifest["entries"][0]["status"] == "missing_config"
    with pytest.raises(FileNotFoundError, match="Missing config"):
        sweep.validate_manifest(manifest, repo_root=tmp_path)


def test_manifest_contains_all_expected_fields(tmp_path):
    entry = _base_entry(tmp_path)
    expected = {
        "run_name",
        "model_family",
        "backbone",
        "config_path",
        "dimensions",
        "noise_type",
        "snr_db",
        "output_dir",
        "command",
        "seed",
        "expected_metrics_file",
        "expected_threshold_file",
    }
    assert expected.issubset(entry)


def test_summary_parser_extracts_metrics_from_synthetic_json(tmp_path):
    entry = _base_entry(tmp_path)
    _write_success_metrics(entry)
    row = sweep.parse_summary_row(entry)
    assert row["status"] == "success"
    assert row["best_epoch"] == 2
    assert row["best_val_metric_name"] == "auroc"
    assert row["best_val_auroc"] == 0.81
    assert row["noisy_val_tau_star"] == 0.42
    assert row["test_auroc"] == 0.75
    assert row["test_auprc"] == 0.76
    assert row["test_f1"] == 0.7
    assert row["test_accuracy"] == 0.72
    assert row["test_sensitivity"] == 0.73
    assert row["test_specificity"] == 0.71


def test_summary_validation_fails_if_threshold_source_is_not_noisy_val(tmp_path):
    entry = _base_entry(tmp_path)
    _write_success_metrics(entry, threshold_source="clean_val")
    with pytest.raises(ValueError, match="threshold_source"):
        sweep.parse_summary_row(entry)


def test_summary_validation_fails_if_checkpoint_source_is_not_noisy_val(tmp_path):
    entry = _base_entry(tmp_path)
    _write_success_metrics(entry, checkpoint_source="clean_val")
    with pytest.raises(ValueError, match="checkpoint_source"):
        sweep.parse_summary_row(entry)


def test_summary_validation_fails_if_train_val_test_metadata_are_not_all_present(tmp_path):
    entry = _base_entry(tmp_path)
    _write_success_metrics(entry, metadata={"train": [{"processing_order": sweep.PROCESSING_ORDER}], "val": []})
    with pytest.raises(ValueError, match="missing noisy metadata"):
        sweep.parse_summary_row(entry)


def test_dry_run_writes_manifest_but_does_not_launch_training(tmp_path):
    out = tmp_path / "dry_run"
    cmd = [
        sys.executable,
        "scripts/run_noisy_input_sweep.py",
        "--dry-run",
        "--models",
        "ecgmamba_mamba",
        "--noise-types",
        "bw",
        "--snr-db",
        "18",
        "--output-root",
        str(out),
        "--python",
        "python",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    manifest_path = out / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["entries"][0]
    assert "DRY-RUN complete" in result.stdout
    assert not Path(entry["output_dir"]).exists()
    assert not Path(entry["expected_metrics_file"]).exists()


def test_smoke_subset_contains_only_requested_small_subset(tmp_path):
    manifest = _manifest(tmp_path, smoke=True)
    assert [entry["run_name"].split("_")[0] for entry in manifest["entries"]] == ["ecgmamba", "bilstm"]
    assert len(manifest["entries"]) == 2
    assert {entry["noise_type"] for entry in manifest["entries"]} == {"bw"}
    assert {entry["snr_db"] for entry in manifest["entries"]} == {18.0}
    assert all("smoke" in entry["config_path"] for entry in manifest["entries"])
