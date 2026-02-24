#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

from data.datamodule import make_dataloaders
from data.splits import load_split
from models.cnn_baseline import CNNBaseline
from models.ecgmamba import ECGMamba
from train.trainer import train_model
from utils.config import load_config, save_config
from utils.io import make_run_dir
from utils.seed import set_seed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["split"]["seed"], cfg["training"].get("deterministic", True))

    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"seed_{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)
    train_loader, val_loader, test_loader = make_dataloaders(cfg, split)

    model = ECGMamba(cfg) if cfg["model"]["name"] == "ecgmamba" else CNNBaseline(cfg)
    run_dir = make_run_dir(cfg["paths"]["runs_dir"], args.run_name)
    save_config(run_dir / "config.yaml", cfg)
    metrics = train_model(model, train_loader, val_loader, test_loader, cfg, run_dir)
    print(metrics)
