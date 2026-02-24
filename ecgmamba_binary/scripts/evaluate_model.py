#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import make_dataloaders
from src.data.splits import load_split
from src.evaluate.evaluator import evaluate
from src.evaluate.plots import save_plots
from src.models.cnn_baseline import CNNBaseline
from src.models.ecgmamba import ECGMamba
from src.train.checkpointing import load_checkpoint
from src.train.metrics import choose_threshold_max_f1
from src.utils.config import load_config
from src.utils.logging import save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_path = args.split or str(
        Path(cfg["paths"]["splits_dir"]) / f"seed_{cfg['split']['seed']}" / "split.json"
    )
    split = load_split(split_path)
    _, val_loader, test_loader = make_dataloaders(cfg, split)

    model = ECGMamba(cfg) if cfg["model"]["name"] == "ecgmamba" else CNNBaseline(cfg)
    load_checkpoint(args.ckpt, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_metrics, y_val, p_val = evaluate(model, val_loader, device, threshold=0.5)
    threshold = choose_threshold_max_f1(y_val, p_val)
    test_metrics, y_test, p_test = evaluate(model, test_loader, device, threshold=threshold)

    run_dir = Path(args.ckpt).parent
    np.savez(run_dir / "preds_val.npz", y=y_val, p=p_val)
    np.savez(run_dir / "preds_test.npz", y=y_test, p=p_test)
    save_plots(run_dir, y_test, p_test, threshold)
    save_json(run_dir / "metrics_evaluate.json", {"val": val_metrics, "test": test_metrics, "threshold": threshold})
    print({"val": val_metrics, "test": test_metrics, "threshold": threshold})
