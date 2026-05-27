import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from train.lr_schedule import cosine_with_warmup
from train.trainer import _run_epoch


class TinyBatchDataset(Dataset):
    def __init__(self, n_batches=4):
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        x = torch.tensor([[float(idx) + 1.0]])
        y = torch.tensor([[1.0 if idx % 2 == 0 else 0.0]])
        return {"x": x, "y": y}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x), None


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
    model, optimizer, scheduler, criterion, loader = _make_setup(total_steps=4, warmup_steps=1)

    order = []
    original_opt_step = optimizer.step
    original_sched_step = scheduler.step

    def opt_step_wrapper(*args, **kwargs):
        order.append("optimizer")
        return original_opt_step(*args, **kwargs)

    def sched_step_wrapper(*args, **kwargs):
        order.append("scheduler")
        return original_sched_step(*args, **kwargs)

    optimizer.step = opt_step_wrapper
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
