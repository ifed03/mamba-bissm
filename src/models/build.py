from .cnn_baseline import CNNBaseline
from .ecgmamba import ECGMamba
from .lstm_baseline import BiLSTMBaseline


def build_model(cfg: dict):
    model_name = str(cfg["model"].get("name", "ecgmamba")).lower()
    if model_name in {"ecgmamba", "mamba", "bimamba"}:
        return ECGMamba(cfg)
    if model_name in {"cnn1d", "cnn"}:
        return CNNBaseline(cfg)
    if model_name == "bilstm":
        return BiLSTMBaseline(cfg)
    raise ValueError(f"Unsupported model.name '{model_name}'.")
