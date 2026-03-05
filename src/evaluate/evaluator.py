import numpy as np
import torch

from train.metrics import compute_metrics


def predict_segments(model, loader, device):
    model.eval()
    outputs = {"record_id": [], "segment_idx": [], "y_true": [], "y_prob": []}
    feats = []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            y = b["y"].cpu().numpy().reshape(-1)
            logit, feat = model(x)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            record_ids = list(b["record_id"])
            segment_idxs = b["segment_idx"].cpu().numpy().reshape(-1)
            outputs["record_id"].extend(record_ids)
            outputs["segment_idx"].extend(segment_idxs.astype(int).tolist())
            outputs["y_true"].extend(y.astype(int).tolist())
            outputs["y_prob"].extend(prob.tolist())
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0) if feats else np.empty((0,))
    return outputs, feats


def aggregate_record_predictions(segment_outputs: dict):
    grouped = {}
    for record_id, y_true, y_prob in zip(
        segment_outputs["record_id"],
        segment_outputs["y_true"],
        segment_outputs["y_prob"],
        strict=True,
    ):
        if record_id not in grouped:
            grouped[record_id] = {"y_true": int(y_true), "y_prob": float(y_prob)}
            continue
        if grouped[record_id]["y_true"] != int(y_true):
            raise ValueError(f"Record {record_id!r} has inconsistent labels across segments")
        grouped[record_id]["y_prob"] = max(grouped[record_id]["y_prob"], float(y_prob))

    record_ids = list(grouped)
    y_true = np.array([grouped[record_id]["y_true"] for record_id in record_ids], dtype=int)
    y_prob = np.array([grouped[record_id]["y_prob"] for record_id in record_ids], dtype=float)
    return {"record_id": record_ids, "y_true": y_true, "y_prob": y_prob}


def evaluate_record_level(model, loader, device, threshold):
    segment_outputs, feats = predict_segments(model, loader, device)
    record_outputs = aggregate_record_predictions(segment_outputs)
    metrics = compute_metrics(record_outputs["y_true"], record_outputs["y_prob"], threshold)
    return metrics, record_outputs, segment_outputs, feats
