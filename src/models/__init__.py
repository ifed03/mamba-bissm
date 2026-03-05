from .cnn_baseline import CNNBaseline
from .ecgmamba import ECGMamba
from .mamba_backbone import BiMambaBackbone, MambaBackbone


def build_model(cfg: dict):
    model_name = str(cfg["model"].get("name", "ecgmamba")).lower()
    if model_name in {"ecgmamba", "mamba", "bimamba"}:
        return ECGMamba(cfg)
    if model_name == "cnn":
        return CNNBaseline(cfg)
    raise ValueError(f"Unsupported model.name '{model_name}'.")


__all__ = [
    "CNNBaseline",
    "ECGMamba",
    "MambaBackbone",
    "BiMambaBackbone",
    "build_model",
]
