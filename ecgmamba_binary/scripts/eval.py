#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

import numpy as np
import torch

from data.datamodule import make_dataloaders
from data.splits import load_split
from eval.evaluator import evaluate
from eval.plots import save_plots
from models.cnn_baseline import CNNBaseline
from models.ecgmamba import ECGMamba
from train.checkpointing import load_checkpoint
from train.metrics import choose_threshold_max_f1
from utils.config import load_config
from utils.logging import save_json


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"seed_{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)
    _, val_loader, test_loader = make_dataloaders(cfg, split)

    model = ECGMamba(cfg) if cfg["model"]["name"] == "ecgmamba" else CNNBaseline(cfg)
    load_checkpoint(args.ckpt, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vm, yv, pv = evaluate(model, val_loader, device, threshold=0.5)
    thr = choose_threshold_max_f1(yv, pv)
    tm, yt, pt = evaluate(model, test_loader, device, threshold=thr)
    run_dir = Path(args.ckpt).parent
    np.savez(run_dir / "preds_val.npz", y=yv, p=pv)
    np.savez(run_dir / "preds_test.npz", y=yt, p=pt)
    save_plots(run_dir, yt, pt, thr)
    save_json(run_dir / "metrics_eval.json", {"val": vm, "test": tm, "threshold": thr})
    print({"val": vm, "test": tm, "threshold": thr})
