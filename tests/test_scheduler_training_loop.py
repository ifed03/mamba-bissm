import json
import math

import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from train.lr_schedule import cosine_with_warmup
from train.trainer import _run_epoch, _select_best_val_metric


class TinyBatchDataset(Dataset):
    def __init__(self, n_batches=4):
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        x = torch.tensor([[float(idx) + 1.0]])
        y = torch.tensor([[1.0 if idx % 2 == 0 else 0.0]])
        return {
            "x": x,
            "y": y,
            "record_id": f"tiny-{idx}",
            "segment_idx": torch.tensor(idx, dtype=torch.long),
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x), None


class RecordingAdamW(AdamW):
    def __init__(self, *args, order, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = order

    def step(self, *args, **kwargs):
        self.order.append("optimizer")
        return super().step(*args, **kwargs)


def _make_setup(total_steps=8, warmup_steps=2):
    model = TinyModel()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = LambdaLR(optimizer, lambda s: cosine_with_warmup(s, total_steps, warmup_steps))
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TinyBatchDataset(n_batches=4), batch_size=1)
    return model, optimizer, scheduler, criterion, loader


def test_cosine_with_warmup_shape():
    total_steps = 10
    warmup_steps = 3
    values = [cosine_with_warmup(s, total_steps, warmup_steps) for s in range(total_steps + 1)]

    assert values[0] == 0.0
    assert math.isclose(values[1], 1 / 3, rel_tol=1e-6)
    assert math.isclose(values[2], 2 / 3, rel_tol=1e-6)
    assert math.isclose(values[3], 1.0, rel_tol=1e-6)
    assert values[4] < values[3]
    assert values[-1] <= 1e-6


def test_scheduler_steps_per_optimizer_update_count():
    epochs = 3
    model, optimizer, scheduler, criterion, loader = _make_setup(total_steps=epochs * 4, warmup_steps=2)
    lr_history = []

    for _ in range(epochs):
        _run_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            scaler=None,
            clip_grad=1.0,
            scheduler=scheduler,
            lr_history=lr_history,
        )

    assert scheduler.last_epoch == epochs * len(loader)
    assert len(lr_history) == epochs * len(loader)
    assert scheduler.last_epoch != epochs


def test_scheduler_step_after_optimizer_step_order():
    order = []
    model = TinyModel()
    optimizer = RecordingAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, order=order)
    scheduler = LambdaLR(optimizer, lambda s: cosine_with_warmup(s, total_steps=4, warmup_steps=1))
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TinyBatchDataset(n_batches=4), batch_size=1)

    original_sched_step = scheduler.step

    def sched_step_wrapper(*args, **kwargs):
        order.append("scheduler")
        return original_sched_step(*args, **kwargs)

    scheduler.step = sched_step_wrapper

    _run_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device=torch.device("cpu"),
        scaler=None,
        clip_grad=1.0,
        scheduler=scheduler,
        lr_history=[],
    )

    assert len(order) == len(loader) * 2
    for i in range(0, len(order), 2):
        assert order[i] == "optimizer"
        assert order[i + 1] == "scheduler"


def test_lr_history_records_updated_param_group_learning_rates():
    total_steps = 8
    warmup_steps = 2
    model, optimizer, scheduler, criterion, loader = _make_setup(total_steps=total_steps, warmup_steps=warmup_steps)
    lr_history = []

    _run_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device=torch.device("cpu"),
        scaler=None,
        clip_grad=1.0,
        scheduler=scheduler,
        lr_history=lr_history,
    )

    expected = [1e-3 * cosine_with_warmup(step, total_steps, warmup_steps) for step in range(1, len(loader) + 1)]
    assert len(lr_history) == len(expected)
    assert len(set(lr_history)) > 1
    for actual, expected_lr in zip(lr_history, expected):
        assert math.isclose(actual, expected_lr, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(optimizer.param_groups[0]["lr"], lr_history[-1], rel_tol=1e-6, abs_tol=1e-12)


def test_lr_history_omits_batches_without_optimizer_update():
    order = []
    model = TinyModel()
    optimizer = RecordingAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, order=order)
    scheduler = LambdaLR(optimizer, lambda s: cosine_with_warmup(s, total_steps=1, warmup_steps=0))
    initial_lr = float(optimizer.param_groups[0]["lr"])
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TinyBatchDataset(n_batches=1), batch_size=1)
    lr_history = []
    hook = model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))

    try:
        loss = _run_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            scaler=None,
            clip_grad=1.0,
            scheduler=scheduler,
            lr_history=lr_history,
        )
    finally:
        hook.remove()

    assert math.isfinite(loss)
    assert order == []
    assert scheduler.last_epoch == 0
    assert lr_history == []
    assert math.isclose(optimizer.param_groups[0]["lr"], initial_lr, rel_tol=1e-9)


def test_run_epoch_with_amp_disabled_on_cpu_still_works():
    model, optimizer, scheduler, criterion, loader = _make_setup(total_steps=4, warmup_steps=1)
    loss = _run_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device=torch.device("cpu"),
        scaler=None,
        clip_grad=1.0,
        scheduler=scheduler,
        lr_history=[],
    )
    assert math.isfinite(loss)


def test_run_epoch_writes_diagnostics_and_raises_on_nonfinite_loss(tmp_path):
    order = []
    model = TinyModel()
    optimizer = RecordingAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, order=order)
    scheduler = LambdaLR(optimizer, lambda s: cosine_with_warmup(s, total_steps=2, warmup_steps=1))
    loader = DataLoader(TinyBatchDataset(n_batches=2), batch_size=1)
    bce = torch.nn.BCEWithLogitsLoss()
    calls = {"n": 0}

    def criterion(logit, target):
        calls["n"] += 1
        if calls["n"] == 1:
            return logit.sum() * torch.tensor(float("nan"), device=logit.device)
        return bce(logit, target)

    with pytest.raises(ValueError, match="nonfinite training loss/logits at epoch 3, batch 0"):
        _run_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            scaler=None,
            clip_grad=1.0,
            scheduler=scheduler,
            lr_history=[],
            run_dir=tmp_path,
            epoch_index=3,
        )

    diagnostic_path = tmp_path / "nonfinite_diagnostics_epoch3_batch0.json"
    assert diagnostic_path.exists()
    diagnostic = json.loads(diagnostic_path.read_text())
    for key in [
        "epoch_index",
        "batch_index",
        "learning_rate",
        "amp_enabled",
        "input",
        "target",
        "logits_before_loss",
        "loss_value",
        "model_parameter_finite_check",
        "gradient_finite_check",
        "batch_record_ids",
        "batch_segment_indices",
    ]:
        assert key in diagnostic
    assert diagnostic["epoch_index"] == 3
    assert diagnostic["batch_index"] == 0
    assert diagnostic["loss_value"] == "nan"
    assert diagnostic["amp_enabled"] is False
    assert diagnostic["input"]["finite_count"] == diagnostic["input"]["total_count"]
    assert diagnostic["target"]["unique_values"] == [1.0]
    assert diagnostic["logits_before_loss"]["finite_count"] == diagnostic["logits_before_loss"]["total_count"]
    assert diagnostic["model_parameter_finite_check"]["all_checked_parameters_finite"] is True
    assert diagnostic["gradient_finite_check"]["checked"] is False
    assert diagnostic["batch_record_ids"] == ["tiny-0"]
    assert diagnostic["batch_segment_indices"] == [0]
    assert order == []
    assert scheduler.last_epoch == 0


def test_best_metric_selection_uses_auroc_when_defined():
    metric_name, metric_value = _select_best_val_metric({"auroc": 0.83, "f1": 0.61})
    assert metric_name == "auroc"
    assert math.isclose(metric_value, 0.83, rel_tol=1e-9)


def test_best_metric_selection_uses_f1_fallback_when_auroc_nan():
    metric_name, metric_value = _select_best_val_metric({"auroc": float("nan"), "f1": 0.61})
    assert metric_name == "f1_fallback"
    assert math.isclose(metric_value, 0.61, rel_tol=1e-9)
