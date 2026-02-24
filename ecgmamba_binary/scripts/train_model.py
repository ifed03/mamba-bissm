#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import make_dataloaders
from src.data.splits import load_split
from src.models.cnn_baseline import CNNBaseline
from src.models.ecgmamba import ECGMamba
from src.train.trainer import train_model
from src.utils.config import load_config, save_config
from src.utils.io import make_run_dir
from src.utils.seed import set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["split"]["seed"], cfg["training"].get("deterministic", True))

    split_path = args.split or str(
        Path(cfg["paths"]["splits_dir"]) / f"seed_{cfg['split']['seed']}" / "split.json"
    )
    split = load_split(split_path)
    train_loader, val_loader, test_loader = make_dataloaders(cfg, split)

    model = ECGMamba(cfg) if cfg["model"]["name"] == "ecgmamba" else CNNBaseline(cfg)
    run_dir = make_run_dir(cfg["paths"]["runs_dir"], args.run_name)
    save_config(run_dir / "config.yaml", cfg)
    metrics = train_model(model, train_loader, val_loader, test_loader, cfg, run_dir)
    print(metrics)
