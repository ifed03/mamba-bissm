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
    "learning_rate_start",
    "learning_rate_end",
    "learning_rate_mean",
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


def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, str, int)):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return [_json_scalar(v) for v in flat]
    if isinstance(value, np.ndarray):
        return [_json_scalar(v) for v in value.reshape(-1)]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return _json_scalar(value)


def _tensor_summary(tensor: torch.Tensor) -> dict:
    with torch.no_grad():
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = int(detached.numel())
        summary = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "finite_count": finite_count,
            "total_count": total_count,
        }
        if finite_count > 0:
            finite_values = detached[finite_mask].float()
            summary.update(
                {
                    "min": _json_scalar(finite_values.min()),
                    "max": _json_scalar(finite_values.max()),
                    "mean": _json_scalar(finite_values.mean()),
                    "std": _json_scalar(finite_values.std(unbiased=False)),
                }
            )
        else:
            summary.update({"min": None, "max": None, "mean": None, "std": None})
        return summary


def _target_summary(target: torch.Tensor) -> dict:
    summary = _tensor_summary(target)
    with torch.no_grad():
        unique = torch.unique(target.detach().cpu())
    summary["unique_values"] = [_json_scalar(v) for v in unique]
    return summary


def _parameter_finite_check(model) -> dict:
    checked = 0
    for name, parameter in model.named_parameters():
        checked += 1
        with torch.no_grad():
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                return {
                    "all_checked_parameters_finite": False,
                    "first_nonfinite_parameter": name,
                    "checked_parameter_count": checked,
                }
    return {
        "all_checked_parameters_finite": True,
        "first_nonfinite_parameter": None,
        "checked_parameter_count": checked,
        "message": "all checked parameters are finite",
    }


def _gradient_finite_check(model) -> dict:
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        checked += 1
        with torch.no_grad():
            if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
                return {
                    "checked": True,
                    "all_checked_gradients_finite": False,
                    "first_nonfinite_gradient": name,
                    "checked_gradient_count": checked,
                }
    if checked == 0:
        return {
            "checked": False,
            "reason": "no gradients had been computed when the nonfinite loss/logits were detected",
        }
    return {
        "checked": True,
        "all_checked_gradients_finite": True,
        "first_nonfinite_gradient": None,
        "checked_gradient_count": checked,
    }


def _write_nonfinite_diagnostics(
    *,
    run_dir: Path,
    epoch_index: int,
    batch_index: int,
    optimizer,
    amp_enabled: bool,
    batch: dict,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    loss: torch.Tensor,
    model,
) -> Path:
    path = run_dir / f"nonfinite_diagnostics_epoch{epoch_index}_batch{batch_index}.json"
    report = {
        "epoch_index": int(epoch_index),
        "batch_index": int(batch_index),
        "learning_rate": _json_scalar(optimizer.param_groups[0]["lr"]),
        "amp_enabled": bool(amp_enabled),
        "input": _tensor_summary(inputs),
        "target": _target_summary(targets),
        "logits_before_loss": _tensor_summary(logits),
        "loss_value": _json_scalar(loss.detach()),
        "model_parameter_finite_check": _parameter_finite_check(model),
        "gradient_finite_check": _gradient_finite_check(model),
        "batch_record_ids": _jsonable(batch["record_id"]) if "record_id" in batch else None,
        "batch_segment_indices": _jsonable(batch["segment_idx"]) if "segment_idx" in batch else None,
    }
    save_json(path, report)
    return path


def _run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    clip_grad=1.0,
    scheduler=None,
    lr_history=None,
    run_dir: Path | None = None,
    epoch_index: int = 0,
):
    model.train()
    losses = []
    for batch_index, b in enumerate(tqdm(loader, leave=False)):
        x, y = b["x"].to(device), b["y"].to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logit, _ = model(x)
            loss = criterion(logit, y.to(logit.device))

        loss_value = float(loss.detach().item())
        if (not bool(torch.isfinite(logit.detach()).all().item())) or (not bool(torch.isfinite(loss.detach()).item())):
            optimizer.zero_grad(set_to_none=True)
            if run_dir is None:
                raise ValueError(
                    f"nonfinite training loss/logits at epoch {epoch_index}, batch {batch_index}; "
                    "no run_dir was provided for diagnostics"
                )
            diagnostic_path = _write_nonfinite_diagnostics(
                run_dir=Path(run_dir),
                epoch_index=epoch_index,
                batch_index=batch_index,
                optimizer=optimizer,
                amp_enabled=scaler is not None,
                batch=b,
                inputs=x,
                targets=y,
                logits=logit,
                loss=loss,
                model=model,
            )
            print(
                f"Nonfinite training loss/logits at epoch {epoch_index}, batch {batch_index}; "
                f"diagnostics written to {diagnostic_path}"
            )
            raise ValueError(
                f"nonfinite training loss/logits at epoch {epoch_index}, batch {batch_index}; "
                f"diagnostics written to {diagnostic_path}"
            )

        update_applied = False
        if scaler is None:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if bool(torch.isfinite(grad_norm).item()):
                optimizer.step()
                update_applied = True
            else:
                optimizer.zero_grad(set_to_none=True)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if bool(torch.isfinite(grad_norm).item()):
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Only advance the scheduler after a successful optimizer update; GradScaler lowers the scale when it skips.
                update_applied = scaler.get_scale() >= prev_scale
            else:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        if scheduler is not None and update_applied:
            scheduler.step()
        if lr_history is not None and update_applied:
            lr_history.append(optimizer.param_groups[0]["lr"])
        losses.append(loss_value)

    finite_losses = [loss for loss in losses if np.isfinite(loss)]
    return float(np.mean(finite_losses)) if finite_losses else float("nan")


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
    for row in rows:
        epoch = row.get("epoch", "?")
        try:
            train_loss = float(row.get("train_loss"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"train_loss for epoch {epoch} must be numeric") from exc
        if not np.isfinite(train_loss):
            raise ValueError(f"train_loss for epoch {epoch} must be finite, got {train_loss}")
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
        epoch_lr_start = float(opt.param_groups[0]["lr"])
        epoch_lr_history_start = len(lr_history)
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
            run_dir=run_dir,
            epoch_index=epoch,
        )
        epoch_lrs = lr_history[epoch_lr_history_start:]
        epoch_lr_end = float(opt.param_groups[0]["lr"])
        epoch_lr_mean = float(np.mean(epoch_lrs)) if epoch_lrs else epoch_lr_end
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
                "learning_rate": epoch_lr_end,
                "learning_rate_start": epoch_lr_start,
                "learning_rate_end": epoch_lr_end,
                "learning_rate_mean": epoch_lr_mean,
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
    save_json(run_dir / "clean_validation_threshold.json", {
        "threshold": float(thr),
        "threshold_source": "clean_val",
        "checkpoint_source": "clean_val",
        "checkpoint": str(best_ckpt_path),
    })
    save_json(run_dir / "metrics.json", out)
    return out
