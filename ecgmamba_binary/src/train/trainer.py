from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.evaluate.evaluator import predict
from src.evaluate.plots import save_plots
from src.train.checkpointing import save_checkpoint
from src.train.losses import make_bce_loss
from src.train.lr_schedule import cosine_with_warmup
from src.train.metrics import choose_threshold_max_f1, compute_metrics
from src.utils.logging import save_json


def _run_epoch(model, loader, optimizer, criterion, device, scaler=None, clip_grad=1.0):
    model.train()
    losses = []
    for b in tqdm(loader, leave=False):
        x, y = b["x"].to(device), b["y"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logit, _ = model(x)
            # Keep targets on the same device as model outputs to avoid CUDA/CPU mismatch errors.
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


def train_model(model, train_loader, val_loader, test_loader, cfg, run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_train = np.array([train_loader.dataset.labels[i] for i in train_loader.dataset.indices])
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = make_bce_loss(float(pos_weight))

    opt = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    total_steps = cfg["training"]["epochs"] * len(train_loader)
    warmup_steps = int(cfg["training"]["warmup_ratio"] * total_steps)
    sched = LambdaLR(opt, lambda s: cosine_with_warmup(s, total_steps, warmup_steps))
    amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None

    best_auc, best_epoch, bad = -1.0, -1, 0
    for epoch in range(cfg["training"]["epochs"]):
        tr_loss = _run_epoch(model, train_loader, opt, criterion, device, scaler, cfg["training"]["grad_clip"])
        sched.step()
        yv, pv, _ = predict(model, val_loader, device)
        thr = choose_threshold_max_f1(yv, pv)
        vm = compute_metrics(yv, pv, thr)
        if vm["auroc"] > best_auc:
            best_auc, best_epoch, bad = vm["auroc"], epoch, 0
            save_checkpoint(run_dir / "best.ckpt", model, opt, epoch, best_auc)
            np.savez(run_dir / "preds_val.npz", y=yv, p=pv)
            save_json(run_dir / "best_val_metrics.json", {"epoch": epoch, "train_loss": tr_loss, "threshold": thr, **vm})
        else:
            bad += 1
        if bad >= cfg["training"]["patience"]:
            break

    ckpt = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(ckpt["model"])
    yv, pv, _ = predict(model, val_loader, device)
    thr = choose_threshold_max_f1(yv, pv)
    tm, yt, pt = None, None, None
    yt, pt, feats = predict(model, test_loader, device)
    tm = compute_metrics(yt, pt, thr)
    np.savez(run_dir / "preds_test.npz", y=yt, p=pt, feats=feats)
    save_plots(run_dir, yt, pt, thr)
    out = {"best_epoch": best_epoch, "best_val_auroc": best_auc, "threshold": thr, "test": tm}
    save_json(run_dir / "metrics.json", out)
    return out
