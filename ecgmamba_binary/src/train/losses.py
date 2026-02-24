import torch
import torch.nn as nn


def make_bce_loss(pos_weight: float | None = None, device: torch.device | None = None):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
