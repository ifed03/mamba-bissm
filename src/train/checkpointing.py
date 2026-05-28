from __future__ import annotations

from pathlib import Path

import torch


def infer_model_dims(state_dict: dict) -> tuple[int, int]:
    backbone_idxs = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith("backbone.")})
    if not backbone_idxs:
        raise ValueError("Cannot infer model size: checkpoint state_dict has no backbone.* keys.")
    if "pos.pe" not in state_dict:
        raise ValueError("Cannot infer model size: checkpoint state_dict is missing pos.pe.")
    return int(state_dict["pos.pe"].shape[-1]), len(backbone_idxs)


def validate_config_matches_state_dict(cfg: dict, state_dict: dict, *, checkpoint: str | Path | None = None) -> None:
    ckpt_d_model, ckpt_n_layers = infer_model_dims(state_dict)
    model_cfg = cfg.get("model") or {}
    cfg_d_model = int(model_cfg["d_model"])
    cfg_n_layers = int(model_cfg["n_layers"])
    if cfg_d_model == ckpt_d_model and cfg_n_layers == ckpt_n_layers:
        return
    loc = f" ({checkpoint})" if checkpoint is not None else ""
    raise ValueError(
        f"Config/checkpoint mismatch{loc}: config has model.d_model={cfg_d_model}, model.n_layers={cfg_n_layers} "
        f"but checkpoint has d_model={ckpt_d_model}, n_layers={ckpt_n_layers}. "
        "Use the config_resolved.yaml from the training run that produced this checkpoint."
    )


def save_checkpoint(path, model, optimizer, epoch, best_metric, best_metric_name=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if best_metric_name is not None:
        payload["best_metric_name"] = best_metric_name
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, cfg: dict | None = None):
    ckpt = torch.load(path, map_location="cpu")
    if cfg is not None:
        validate_config_matches_state_dict(cfg, ckpt["model"], checkpoint=path)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
