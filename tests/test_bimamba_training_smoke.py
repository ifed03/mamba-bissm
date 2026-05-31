import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bimamba use_fast_path requires CUDA")
def test_bimamba_training_smoke_one_epoch(tmp_path: Path):
    pytest.importorskip("mamba_ssm")

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "final_configs/binary_bimamba_d128_n2_s64_slowpath_fp32_100hz_win4s_stride2s.yaml"
    cfg = yaml.safe_load(src.read_text())

    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["patience"] = 1
    cfg["training"]["mixed_precision"] = False

    cfg["model"]["d_model"] = 16
    cfg["model"]["n_layers"] = 1
    cfg["model"]["d_state"] = 4
    cfg["model"]["d_conv"] = 3
    cfg["model"]["expand"] = 2
    cfg["model"]["use_fast_path"] = True

    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    config_path = tmp_path / "bimamba_train_smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_model.py",
            "--config",
            str(config_path),
            "--run-name",
            "smoke_bimamba_manual",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    run_dir = tmp_path / "runs" / "smoke_bimamba_manual"
    assert (run_dir / "checkpoints" / "best.ckpt").is_file()
    assert (run_dir / "training_history.csv").is_file()
