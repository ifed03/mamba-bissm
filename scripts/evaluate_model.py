#!/usr/bin/env python
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path) # .insert(0, ...) puts it at the front of the search list

import argparse
import torch
import pandas as pd

from data.datamodule import make_dataloaders
from data.splits import load_split
from evaluate.evaluator import evaluate_record_level
from evaluate.plots import save_plots
from models import build_model
from train.checkpointing import load_checkpoint
from train.metrics import choose_threshold_max_f1, compute_metrics
from utils.config import load_config
from utils.logging import save_json


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"holdout_seed{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)
    _, val_loader, test_loader = make_dataloaders(cfg, split)

    model = build_model(cfg)
    load_checkpoint(args.ckpt, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _, val_records, val_segments, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
    thr = choose_threshold_max_f1(val_records["y_true"], val_records["y_prob"])
    tm, test_records, test_segments, _ = evaluate_record_level(model, test_loader, device, threshold=thr)
    vm = compute_metrics(val_records["y_true"], val_records["y_prob"], thr)
    ckpt_path = Path(args.ckpt)
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent

    pd.DataFrame({"record_id": val_records["record_id"], "y_true": val_records["y_true"].astype(int), "y_prob": val_records["y_prob"], "split": "val"}).to_parquet(run_dir / "preds_val.parquet", index=False)
    pd.DataFrame({"record_id": test_records["record_id"], "y_true": test_records["y_true"].astype(int), "y_prob": test_records["y_prob"], "split": "test"}).to_parquet(run_dir / "preds.parquet", index=False)
    pd.DataFrame({"record_id": val_segments["record_id"], "segment_idx": val_segments["segment_idx"], "y_true": val_segments["y_true"], "y_prob": val_segments["y_prob"], "split": "val"}).to_parquet(run_dir / "preds_val_segments.parquet", index=False)
    pd.DataFrame({"record_id": test_segments["record_id"], "segment_idx": test_segments["segment_idx"], "y_true": test_segments["y_true"], "y_prob": test_segments["y_prob"], "split": "test"}).to_parquet(run_dir / "preds_segments.parquet", index=False)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plots(plots_dir, test_records["y_true"], test_records["y_prob"], thr)
    save_json(run_dir / "metrics_eval.json", {"val": vm, "test": tm, "threshold": thr})
    print({"val": vm, "test": tm, "threshold": thr})
