import json
import os
import time
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



def _write_complete_outputs(entry):
    _write_success_metrics(entry)
    ckpt = Path(entry["expected_checkpoint_file"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("checkpoint")


def test_jobs_defaults_to_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_noisy_input_sweep.py", "--dry-run"])
    args = sweep._parse_args()
    assert args.jobs == 1


def test_invalid_jobs_zero_fails_clearly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_noisy_input_sweep.py", "--jobs", "0"])
    with pytest.raises(SystemExit):
        sweep._parse_args()
    assert "--jobs must be >= 1" in capsys.readouterr().err


def test_jobs_one_preserves_sequential_execution_order(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path,
        models=["ecgmamba_mamba"],
        noise_types=["bw"],
        snr_db=[24.0, 18.0, 12.0],
    )
    seen = []

    def fake_run(item, **kwargs):
        seen.append(item.entry["run_name"])
        return sweep.WorkResult(item.entry["run_name"], item.kind, "completed", 0, "log")

    monkeypatch.setattr(sweep, "_run_work_item", fake_run)
    sweep.run_manifest(manifest, repo_root=REPO_ROOT, jobs=1)
    assert seen == [entry["run_name"] for entry in manifest["entries"]]


def test_jobs_four_schedules_multiple_independent_conditions(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path,
        models=["ecgmamba_mamba"],
        noise_types=["bw", "em"],
        snr_db=[24.0, 18.0],
    )
    submitted = []

    class ImmediateFuture:
        def __init__(self, result):
            self._result = result
        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def submit(self, fn, item, **kwargs):
            submitted.append(item.entry["run_name"])
            return ImmediateFuture(sweep.WorkResult(item.entry["run_name"], item.kind, "completed", 0, "log"))

    monkeypatch.setattr(sweep, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(sweep, "as_completed", lambda futures: list(futures))
    sweep.run_manifest(manifest, repo_root=REPO_ROOT, jobs=4)
    assert len(submitted) == 4
    assert len(set(submitted)) == 4


def test_unique_run_names_and_output_dirs_before_parallel_launch(tmp_path):
    manifest = _manifest(tmp_path)
    sweep.validate_manifest(manifest, repo_root=REPO_ROOT)
    names = [entry["run_name"] for entry in manifest["entries"]]
    dirs = [entry["output_dir"] for entry in manifest["entries"]]
    assert len(names) == len(set(names))
    assert len(dirs) == len(set(dirs))


def test_completed_run_is_skipped_under_resume(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0])
    entry = manifest["entries"][0]
    _write_complete_outputs(entry)
    sweep.mark_resume_statuses(manifest)
    assert entry["status"] == "completed"
    assert sweep.build_work_items(manifest) == []


def test_completed_run_missing_efficiency_is_scheduled_for_profiling(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0])
    entry = manifest["entries"][0]
    _write_complete_outputs(entry)
    sweep.mark_resume_statuses(manifest)
    items = sweep.build_work_items(manifest, profile_efficiency=True)
    assert [(item.kind, item.entry["run_name"]) for item in items] == [("profile_efficiency", entry["run_name"])]


def test_existing_lock_prevents_duplicate_execution(tmp_path):
    entry = _base_entry(tmp_path)
    run_dir = Path(entry["output_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / sweep.LOCK_NAME).write_text("busy")
    result = sweep._run_work_item(
        sweep.WorkItem("train_condition", entry),
        repo_root=REPO_ROOT,
        profile_efficiency=False,
        efficiency_warmup=1,
        efficiency_repeats=1,
        efficiency_throughput_batch_size=1,
        efficiency_max_records=None,
        overwrite_efficiency=False,
        stale_lock_minutes=None,
    )
    assert result.status == "locked"


def test_stale_lock_can_be_replaced(tmp_path):
    entry = _base_entry(tmp_path)
    run_dir = Path(entry["output_dir"])
    run_dir.mkdir(parents=True)
    lock = run_dir / sweep.LOCK_NAME
    lock.write_text("old")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    acquired = sweep.acquire_run_lock(entry, stale_lock_minutes=1)
    try:
        assert acquired == lock
        assert "run_name" in lock.read_text()
    finally:
        sweep.release_run_lock(acquired)


def test_failed_subprocess_is_recorded_without_stopping_unrelated_runs(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path,
        models=["ecgmamba_mamba"],
        noise_types=["bw"],
        snr_db=[24.0, 18.0],
    )
    failing = manifest["entries"][0]["run_name"]

    def fake_run(cmd, *, cwd, log_path):
        return 7 if failing in str(log_path) else 0

    monkeypatch.setattr(sweep, "_run_subprocess_to_log", fake_run)
    sweep.run_manifest(manifest, repo_root=REPO_ROOT, jobs=2)
    statuses = {entry["run_name"]: entry["status"] for entry in manifest["entries"]}
    assert statuses[failing] == "failed"
    assert list(statuses.values()).count("completed") == 1
    assert manifest["entries"][0]["return_code"] == 7


def test_per_run_log_paths_are_created_and_recorded(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0])
    monkeypatch.setattr(sweep, "_run_subprocess_to_log", lambda cmd, *, cwd, log_path: (log_path.write_text("log"), 0)[1])
    sweep.run_manifest(manifest, repo_root=REPO_ROOT, jobs=1)
    entry = manifest["entries"][0]
    assert entry["log_file_path"].endswith("run.log")
    assert Path(entry["log_file_path"]).is_file()


def test_summary_aggregation_still_works_after_parallel_execution(tmp_path, monkeypatch):
    manifest = _manifest(
        tmp_path,
        models=["ecgmamba_mamba"],
        noise_types=["bw"],
        snr_db=[24.0, 18.0],
    )

    def fake_run(cmd, *, cwd, log_path):
        run_name = log_path.parent.name
        entry = next(e for e in manifest["entries"] if e["run_name"] == run_name)
        _write_complete_outputs(entry)
        log_path.write_text("ok")
        return 0

    monkeypatch.setattr(sweep, "_run_subprocess_to_log", fake_run)
    sweep.run_manifest(manifest, repo_root=REPO_ROOT, jobs=2)
    rows = sweep.collect_summary(manifest, tmp_path / "runs")
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"success"}


def test_dry_run_with_jobs_four_writes_manifest_but_launches_no_subprocesses(tmp_path):
    out = tmp_path / "dry_run_jobs"
    cmd = [
        sys.executable,
        "scripts/run_noisy_input_sweep.py",
        "--dry-run",
        "--jobs",
        "4",
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
    manifest = json.loads((out / "manifest.json").read_text())
    assert "DRY-RUN complete" in result.stdout
    assert manifest["entries"][0]["command"][0] == "python"
    assert not Path(manifest["entries"][0]["output_dir"]).exists()


def test_smoke_mode_accepts_jobs_two(tmp_path):
    manifest = _manifest(tmp_path, smoke=True)
    assert len(manifest["entries"]) == 2
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(sys, "argv", ["run_noisy_input_sweep.py", "--smoke", "--jobs", "2"])
        args = sweep._parse_args()
        assert args.smoke is True
        assert args.jobs == 2
    finally:
        monkeypatch.undo()



def _complete_manifest(manifest):
    for entry in manifest["entries"]:
        _write_complete_outputs(entry)
    sweep.mark_resume_statuses(manifest)


def _write_shared_efficiency(entry, *, trainable=1234, record_latency_ms=12.5):
    efficiency_path = Path(entry["efficiency_profile_file"])
    efficiency_path.parent.mkdir(parents=True, exist_ok=True)
    efficiency_path.write_text(
        json.dumps(
            {
                "trainable_parameters": trainable,
                "mean_record_latency_ms": record_latency_ms,
                "mean_window_latency_ms_batch1": 3.5,
                "records_per_second": 80.0,
                "windows_per_second_batch16": 160.0,
                "warmup_iterations": 5,
                "measured_repeats": 10,
            }
        )
    )
    (efficiency_path.parent / "efficiency_record_latency.csv").write_text("record_id,latency_ms\nr1,10\nr2,15\n")


def test_default_efficiency_profiles_once_per_unique_model_config(tmp_path):
    manifest = _manifest(tmp_path)
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs")
    items = sweep.build_work_items(manifest, profile_efficiency=True)
    profile_items = [item for item in items if item.kind == "profile_efficiency"]
    assert len(profile_items) == 6
    assert len({item.entry["efficiency_profile_key"] for item in profile_items}) == 6


def test_profile_efficiency_per_condition_schedules_all_completed_conditions(tmp_path):
    manifest = _manifest(tmp_path)
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs", per_condition=True)
    items = sweep.build_work_items(
        manifest,
        profile_efficiency=True,
        profile_efficiency_per_condition=True,
    )
    assert len([item for item in items if item.kind == "profile_efficiency"]) == 108


def test_summary_uses_shared_efficiency_values_for_every_run(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0, 12.0])
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs")
    representative = next(e for e in manifest["entries"] if e["run_name"] == e["efficiency_profile_source_run_name"])
    _write_shared_efficiency(representative)

    rows = sweep.collect_summary(manifest, tmp_path / "runs")
    assert len(rows) == 2
    assert {row["inference_latency_ms_per_record"] for row in rows} == {12.5}
    assert {row["num_trainable_params"] for row in rows} == {1234}
    assert {row["efficiency_file_path"] for row in rows} == {representative["efficiency_profile_file"]}
    assert {row["efficiency_profile_source_run_name"] for row in rows} == {representative["run_name"]}


def test_summary_writes_efficiency_profile_key_and_source_columns(tmp_path):
    manifest = _manifest(tmp_path, models=["ecgmamba_mamba"], noise_types=["bw"], snr_db=[18.0])
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs")
    _write_shared_efficiency(manifest["entries"][0])

    sweep.collect_summary(manifest, tmp_path / "runs")
    header = (tmp_path / "runs" / "summary.csv").read_text().splitlines()[0].split(",")
    assert "efficiency_profile_key" in header
    assert "efficiency_profile_source_run_name" in header
    row = sweep.parse_summary_row(manifest["entries"][0])
    assert row["efficiency_profile_key"]
    assert row["efficiency_profile_source_run_name"] == manifest["entries"][0]["run_name"]


def test_representative_selection_prefers_bw_18db(tmp_path):
    manifest = _manifest(
        tmp_path,
        models=["ecgmamba_mamba"],
        noise_types=["em", "bw"],
        snr_db=[24.0, 18.0],
    )
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs")
    expected = next(e for e in manifest["entries"] if e["noise_type"] == "bw" and e["snr_db"] == 18.0)
    assert {entry["efficiency_profile_source_run_name"] for entry in manifest["entries"]} == {expected["run_name"]}


def test_shared_efficiency_profiles_reused_under_resume_when_present(tmp_path):
    manifest = _manifest(tmp_path)
    _complete_manifest(manifest)
    sweep.assign_efficiency_profile_sources(manifest, tmp_path / "runs")
    representative_items = sweep.build_work_items(manifest, profile_efficiency=True)
    assert len(representative_items) == 6

    _write_shared_efficiency(representative_items[0].entry)
    items_after_one_shared_exists = sweep.build_work_items(manifest, profile_efficiency=True)
    assert len([item for item in items_after_one_shared_exists if item.kind == "profile_efficiency"]) == 5
    assert representative_items[0].entry["run_name"] not in {item.entry["run_name"] for item in items_after_one_shared_exists}
