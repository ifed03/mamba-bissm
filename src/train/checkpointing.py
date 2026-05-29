from __future__ import annotations

from pathlib import Path

import torch


def infer_model_dims(state_dict: dict) -> tuple[int, int]:
    backbone_idxs: set[int] = set()
    for key in state_dict:
        if not key.startswith("backbone."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            backbone_idxs.add(int(parts[1]))
        elif len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit():
            backbone_idxs.add(int(parts[2]))
        elif len(parts) > 3 and parts[1] in {"fwd", "bwd"} and parts[2] == "layers" and parts[3].isdigit():
            backbone_idxs.add(int(parts[3]))
    if not backbone_idxs:
        raise ValueError("Cannot infer model size: checkpoint state_dict has no backbone layer keys.")
    if "pos.pe" not in state_dict:
        raise ValueError("Cannot infer model size: checkpoint state_dict is missing pos.pe.")
    return int(state_dict["pos.pe"].shape[-1]), len(backbone_idxs)


def validate_config_matches_state_dict(cfg: dict, state_dict: dict, *, checkpoint: str | Path | None = None) -> None:
    model_cfg = cfg.get("model") or {}
    model_name = str(model_cfg.get("name", "ecgmamba")).lower()
    backbone = str(model_cfg.get("backbone", "bissm" if model_name == "ecgmamba" else model_name)).lower()
    if model_name not in {"ecgmamba", "mamba", "bimamba"} or backbone not in {"bissm", "mamba", "bimamba"}:
        return

    ckpt_d_model, ckpt_n_layers = infer_model_dims(state_dict)
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
