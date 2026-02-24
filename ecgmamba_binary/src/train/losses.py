import torch
import torch.nn as nn


def make_bce_loss(pos_weight: float | None = None):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
