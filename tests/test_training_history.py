import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from train import trainer
from train.trainer import TRAINING_HISTORY_COLUMNS, _save_training_history, train_model


REQUIRED_HISTORY_COLUMNS = [
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
]


class TinyRecordDataset(Dataset):
    def __init__(self):
        self.xs = [
            torch.tensor([[-1.0]], dtype=torch.float32),
            torch.tensor([[-0.5]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32),
        ]
        self.ys = [0.0, 0.0, 1.0, 1.0]
        self.record_ids = [f"r{i}" for i in range(len(self.ys))]
        self.sample_labels = [int(y) for y in self.ys]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return {
            "x": self.xs[idx],
            "y": torch.tensor([self.ys[idx]], dtype=torch.float32),
            "record_id": self.record_ids[idx],
            "segment_idx": torch.tensor(0, dtype=torch.long),
        }


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.zero_()

    def forward(self, x):
        features = x.reshape(x.shape[0], -1)
        return self.linear(features), features


def test_training_history_csv_created_with_required_multi_epoch_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "save_plots", lambda *args, **kwargs: None)
    torch.manual_seed(0)

    dataset = TinyRecordDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    cfg = {
        "training": {
            "epochs": 3,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "mixed_precision": False,
            "grad_clip": 1.0,
            "patience": 10,
        }
    }

    train_model(TinyClassifier(), loader, loader, loader, cfg, tmp_path)

    history_path = tmp_path / "training_history.csv"
    assert history_path.exists()

    history = pd.read_csv(history_path)
    assert len(history) == cfg["training"]["epochs"]
    assert list(history.columns) == TRAINING_HISTORY_COLUMNS
    assert set(REQUIRED_HISTORY_COLUMNS).issubset(history.columns)
    assert history["train_loss"].notna().all()
    assert history["train_loss"].map(math.isfinite).all()
    for column in ["learning_rate", "learning_rate_start", "learning_rate_end", "learning_rate_mean"]:
        assert history[column].map(math.isfinite).all()
        assert (history[column] >= 0).all()
    assert np.allclose(history["learning_rate"], history["learning_rate_end"])

    lr_history = pd.read_csv(tmp_path / "lr_history.csv")
    updates_per_epoch = len(loader)
    for epoch in range(cfg["training"]["epochs"]):
        epoch_lrs = lr_history["lr"].iloc[epoch * updates_per_epoch : (epoch + 1) * updates_per_epoch]
        assert len(epoch_lrs) == updates_per_epoch
        assert math.isclose(history.loc[epoch, "learning_rate_end"], epoch_lrs.iloc[-1], rel_tol=1e-6, abs_tol=1e-12)
        assert math.isclose(history.loc[epoch, "learning_rate_mean"], epoch_lrs.mean(), rel_tol=1e-6, abs_tol=1e-12)
    for epoch in range(1, cfg["training"]["epochs"]):
        assert math.isclose(
            history.loc[epoch, "learning_rate_start"],
            history.loc[epoch - 1, "learning_rate_end"],
            rel_tol=1e-6,
            abs_tol=1e-12,
        )
    assert history["best_checkpoint_this_epoch"].astype(bool).any()


def test_training_history_writer_rejects_nonfinite_train_loss(tmp_path):
    row = {column: 0.0 for column in TRAINING_HISTORY_COLUMNS}
    row.update(
        {
            "epoch": 2,
            "train_loss": float("nan"),
            "best_checkpoint_this_epoch": False,
            "best_val_metric_name": "auroc",
        }
    )

    history_path = tmp_path / "training_history.csv"
    with pytest.raises(ValueError, match="train_loss for epoch 2 must be finite"):
        _save_training_history(history_path, [row])

    assert not history_path.exists()
