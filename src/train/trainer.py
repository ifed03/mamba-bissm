from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evaluate.evaluator import evaluate_record_level
from evaluate.plots import save_plots
from train.checkpointing import save_checkpoint
from train.losses import make_bce_loss
from train.lr_schedule import cosine_with_warmup
from train.metrics import choose_threshold_max_f1, compute_metrics
from utils.logging import save_json


def _run_epoch(model, loader, optimizer, criterion, device, scaler=None, clip_grad=1.0):
    model.train()
    losses = []
    for b in tqdm(loader, leave=False):
        x, y = b["x"].to(device), b["y"].to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logit, _ = model(x)
            loss = criterion(logit, y.to(logit.device))

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def _save_predictions(path: Path, record_ids, y_true, y_prob, split: str):
    df = pd.DataFrame(
        {
            "record_id": list(record_ids),
            "y_true": np.array(y_true).astype(int),
            "y_prob": np.array(y_prob),
            "split": split,
        }
    )
    df.to_parquet(path, index=False)


def _save_segment_predictions(path: Path, segment_outputs: dict, split: str):
    df = pd.DataFrame(
        {
            "record_id": list(segment_outputs["record_id"]),
            "segment_idx": np.array(segment_outputs["segment_idx"]).astype(int),
            "y_true": np.array(segment_outputs["y_true"]).astype(int),
            "y_prob": np.array(segment_outputs["y_prob"]),
            "split": split,
        }
    )
    df.to_parquet(path, index=False)


def train_model(model, train_loader, val_loader, test_loader, cfg, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_train = np.array(train_loader.dataset.sample_labels)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = make_bce_loss(float(pos_weight)).to(device)

    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    opt = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = cfg["training"]["epochs"] * len(train_loader)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * total_steps)
    sched = LambdaLR(opt, lambda s: cosine_with_warmup(s, total_steps, warmup_steps))
    amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None

    best_auc, best_epoch, bad = -1.0, -1, 0
    best_ckpt_path = checkpoints_dir / "best.ckpt"
    for epoch in range(cfg["training"]["epochs"]):
        tr_loss = _run_epoch(model, train_loader, opt, criterion, device, scaler, cfg["training"]["grad_clip"])
        sched.step()
        _, val_record_outputs, _, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
        thr = choose_threshold_max_f1(val_record_outputs["y_true"], val_record_outputs["y_prob"])
        vm = compute_metrics(val_record_outputs["y_true"], val_record_outputs["y_prob"], thr)
        score = vm["auroc"] if not np.isnan(vm["auroc"]) else vm["f1"]
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            save_checkpoint(best_ckpt_path, model, opt, epoch, best_auc)
            save_json(run_dir / "best_val_metrics.json", {"epoch": epoch, "train_loss": tr_loss, "threshold": thr, **vm})
        else:
            bad += 1
        if bad >= cfg["training"]["patience"]:
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, val_record_outputs, val_segment_outputs, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
    thr = choose_threshold_max_f1(val_record_outputs["y_true"], val_record_outputs["y_prob"])
    _, test_record_outputs, test_segment_outputs, feats = evaluate_record_level(model, test_loader, device, threshold=thr)

    _save_predictions(
        run_dir / "preds_val.parquet",
        val_record_outputs["record_id"],
        val_record_outputs["y_true"],
        val_record_outputs["y_prob"],
        "val",
    )
    _save_predictions(
        run_dir / "preds.parquet",
        test_record_outputs["record_id"],
        test_record_outputs["y_true"],
        test_record_outputs["y_prob"],
        "test",
    )
    _save_segment_predictions(run_dir / "preds_val_segments.parquet", val_segment_outputs, "val")
    _save_segment_predictions(run_dir / "preds_segments.parquet", test_segment_outputs, "test")
    np.savez(run_dir / "preds_test_features.npz", feats=feats)

    tm = compute_metrics(test_record_outputs["y_true"], test_record_outputs["y_prob"], thr)
    save_plots(plots_dir, test_record_outputs["y_true"], test_record_outputs["y_prob"], thr)
    out = {"best_epoch": best_epoch, "best_val_auroc": best_auc, "threshold": thr, "test": tm}
    save_json(run_dir / "metrics.json", out)
    return out
