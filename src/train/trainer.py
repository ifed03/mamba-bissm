from pathlib import Path
import time

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


TRAINING_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_auroc",
    "val_auprc",
    "val_f1",
    "val_accuracy",
    "val_sensitivity",
    "val_specificity",
    "learning_rate",
    "epoch_time_seconds",
    "best_checkpoint_this_epoch",
    "val_threshold",
    "best_val_metric",
    "best_val_metric_name",
    "num_bad_epochs",
    "train_time_seconds",
    "val_time_seconds",
]


def _select_best_val_metric(vm: dict) -> tuple[str, float]:
    if not np.isnan(vm["auroc"]):
        return "auroc", float(vm["auroc"])
    return "f1_fallback", float(vm["f1"])


def _run_epoch(model, loader, optimizer, criterion, device, scaler=None, clip_grad=1.0, scheduler=None, lr_history=None):
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
            if scheduler is not None:
                scheduler.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance the scheduler after a successful optimizer update; GradScaler lowers the scale when it skips.
            if scheduler is not None and scaler.get_scale() >= prev_scale:
                scheduler.step()
        if lr_history is not None:
            lr_history.append(optimizer.param_groups[0]["lr"])
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


def _save_training_history(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=TRAINING_HISTORY_COLUMNS).to_csv(path, index=False)


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

    best_metric, best_metric_name, best_epoch, bad = -1.0, "auroc", -1, 0
    best_ckpt_path = checkpoints_dir / "best.ckpt"
    lr_history = []
    epoch_times = []
    training_history = []
    train_start = time.perf_counter()
    for epoch in range(cfg["training"]["epochs"]):
        epoch_t0 = time.perf_counter()
        train_t0 = time.perf_counter()
        tr_loss = _run_epoch(
            model,
            train_loader,
            opt,
            criterion,
            device,
            scaler,
            cfg["training"]["grad_clip"],
            scheduler=sched,
            lr_history=lr_history,
        )
        train_time_seconds = time.perf_counter() - train_t0
        val_t0 = time.perf_counter()
        _, val_record_outputs, _, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
        thr = choose_threshold_max_f1(val_record_outputs["y_true"], val_record_outputs["y_prob"])
        vm = compute_metrics(val_record_outputs["y_true"], val_record_outputs["y_prob"], thr)
        val_time_seconds = time.perf_counter() - val_t0
        score_name, score = _select_best_val_metric(vm)
        best_checkpoint_this_epoch = False
        if score > best_metric:
            best_metric, best_metric_name, best_epoch, bad = score, score_name, epoch, 0
            best_checkpoint_this_epoch = True
            save_checkpoint(best_ckpt_path, model, opt, epoch, best_metric, best_metric_name=best_metric_name)
            save_json(
                run_dir / "best_val_metrics.json",
                {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "threshold": thr,
                    "best_val_metric_name": best_metric_name,
                    "best_val_metric": float(best_metric),
                    **vm,
                },
            )
        else:
            bad += 1
        epoch_time_seconds = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_time_seconds)
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "val_auroc": float(vm["auroc"]),
                "val_auprc": float(vm["auprc"]),
                "val_f1": float(vm["f1"]),
                "val_accuracy": float(vm["accuracy"]),
                "val_sensitivity": float(vm["sensitivity"]),
                "val_specificity": float(vm["specificity"]),
                "learning_rate": float(opt.param_groups[0]["lr"]),
                "epoch_time_seconds": float(epoch_time_seconds),
                "best_checkpoint_this_epoch": best_checkpoint_this_epoch,
                "val_threshold": float(thr),
                "best_val_metric": float(best_metric),
                "best_val_metric_name": best_metric_name,
                "num_bad_epochs": int(bad),
                "train_time_seconds": float(train_time_seconds),
                "val_time_seconds": float(val_time_seconds),
            }
        )
        if bad >= cfg["training"]["patience"]:
            break

    training_time_seconds = time.perf_counter() - train_start
    _save_training_history(run_dir / "training_history.csv", training_history)

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

    vm = compute_metrics(val_record_outputs["y_true"], val_record_outputs["y_prob"], thr)
    tm = compute_metrics(test_record_outputs["y_true"], test_record_outputs["y_prob"], thr)
    save_plots(plots_dir, test_record_outputs["y_true"], test_record_outputs["y_prob"], thr)
    pd.DataFrame({"update": np.arange(1, len(lr_history) + 1), "lr": lr_history}).to_csv(
        run_dir / "lr_history.csv", index=False
    )
    out = {
        "best_epoch": best_epoch,
        "best_val_metric_name": best_metric_name,
        "best_val_metric": float(best_metric),
        "best_val_auroc": float(best_metric),  # backward-compatible alias; may be F1 fallback
        "threshold": thr,
        "val": vm,
        "test": tm,
        "training_time_seconds": float(training_time_seconds),
        "mean_epoch_time_seconds": float(np.mean(epoch_times)) if epoch_times else 0.0,
    }
    save_json(run_dir / "metrics.json", out)
    return out
