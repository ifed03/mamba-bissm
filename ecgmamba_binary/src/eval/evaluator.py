import numpy as np
import torch

from train.metrics import compute_metrics


def predict(model, loader, device):
    model.eval()
    ys, ps, feats = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            y = b["y"].cpu().numpy()
            logit, feat = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            ys.extend(y.tolist())
            ps.extend(prob.tolist())
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0) if feats else np.empty((0,))
    return np.array(ys), np.array(ps), feats


def evaluate(model, loader, device, threshold):
    y, p, _ = predict(model, loader, device)
    return compute_metrics(y, p, threshold), y, p
