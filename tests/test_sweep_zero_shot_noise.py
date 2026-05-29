import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_sweep_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_zero_shot_noise.py"
    spec = importlib.util.spec_from_file_location("sweep_zero_shot_noise", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_run(root: Path, name: str = "binary_bilstm_fake_run") -> Path:
    run_dir = root / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.ckpt").write_bytes(b"placeholder")
    (run_dir / "clean_validation_threshold.json").write_text(
        '{"threshold": 0.5, "threshold_source": "clean_val"}',
        encoding="utf-8",
    )
    (run_dir / "config_resolved.yaml").write_text(
        "paths:\n"
        "  data_path: data/fake.parquet\n"
        "  splits_dir: splits\n"
        "split:\n"
        "  seed: 42\n"
        "preprocessing:\n"
        "  fs_target: 100\n"
        "  target_seconds: 4.0\n"
        "  windowing:\n"
        "    enabled: true\n"
        "model:\n"
        "  name: bilstm\n"
        "  hidden_size: 128\n"
        "  num_layers: 2\n"
        "  bidirectional: true\n",
        encoding="utf-8",
    )
    return run_dir


def test_successful_runs_manifest_and_matrix_include_12_conditions_per_run(tmp_path):
    sweep = _load_sweep_module()
    run_dir = _write_fake_run(tmp_path / "successful_runs")
    manifest = tmp_path / "successful_runs.txt"
    manifest.write_text(f"{run_dir}\n", encoding="utf-8")

    runs = sweep.discover_successful_runs(manifest)
    jobs = sweep.build_evaluation_matrix(
        runs,
        noise_types=["bw", "em", "ma"],
        snrs=[18, 6, 0, -6],
        output_root=tmp_path / "out",
    )

    assert len(runs) == 1
    assert len(jobs) == 12
    assert {job.condition.noise_type for job in jobs} == {"bw", "em", "ma"}
    assert {job.condition.snr_db for job in jobs} == {18.0, 6.0, 0.0, -6.0}


def test_tmp_manifest_can_contain_cwd_relative_run_paths(tmp_path, monkeypatch):
    sweep = _load_sweep_module()
    repo_like = tmp_path / "repo"
    run_dir = _write_fake_run(repo_like / "successful_runs")
    manifest = tmp_path / "successful_runs_skip.txt"
    manifest.write_text(f"successful_runs/{run_dir.name}\n", encoding="utf-8")

    monkeypatch.chdir(repo_like)

    runs = sweep.discover_successful_runs(manifest)

    assert len(runs) == 1
    assert runs[0].run_dir == run_dir.resolve()


def test_output_filename_includes_zero_shot_noise_type_and_snr(tmp_path):
    sweep = _load_sweep_module()
    run_dir = _write_fake_run(tmp_path / "successful_runs")
    run = sweep.discover_successful_runs(run_dir.parent)[0]

    [job] = sweep.build_evaluation_matrix(
        [run],
        noise_types=["ma"],
        snrs=[-6],
        output_root=tmp_path / "out",
    )

    path = str(job.result_path)
    assert "bilstm_h128_n2_bi" in path
    assert run_dir.name in path
    assert "zero-shot" in path
    assert path.endswith("metrics_zero-shot_noise_type=ma__snr_db=neg6.json")


def test_dry_run_prints_matrix_and_writes_nothing(tmp_path):
    _write_fake_run(tmp_path / "successful_runs")
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_zero_shot_noise.py",
            "--successful-runs",
            str(tmp_path / "successful_runs"),
            "--noise-types",
            "bw",
            "em",
            "ma",
            "--snrs",
            "18",
            "6",
            "0",
            "-6",
            "--output-root",
            str(out),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Planned evaluations: 12" in result.stdout
    assert "DRY-RUN would evaluate" in result.stdout
    assert "metrics_zero-shot_noise_type=bw__snr_db=18.json" in result.stdout
    assert "metrics_zero-shot_noise_type=ma__snr_db=neg6.json" in result.stdout
    assert not out.exists()


def test_batched_run_command_contains_all_pending_conditions(tmp_path):
    sweep = _load_sweep_module()
    run_dir = _write_fake_run(tmp_path / "successful_runs")
    run = sweep.discover_successful_runs(run_dir.parent)[0]
    jobs = sweep.build_evaluation_matrix(
        [run],
        noise_types=["bw", "em", "ma"],
        snrs=[18, 6, 0, -6],
        output_root=tmp_path / "out",
    )

    args = type(
        "Args",
        (),
        {"noise_root": "data", "base_seed": 123, "ecg_fs": 100},
    )()
    cmd = sweep._evaluate_run_command(run, jobs, args)

    assert cmd.count("scripts/evaluate_model.py") == 1
    assert "--skip-existing" in cmd
    assert cmd[cmd.index("--noise-types") + 1 : cmd.index("--snr-db")] == ["bw", "em", "ma"]
    assert cmd[cmd.index("--snr-db") + 1 : cmd.index("--noise-root")] == ["18", "6", "0", "-6"]
    assert cmd.count("--clean-threshold-path") == 1
