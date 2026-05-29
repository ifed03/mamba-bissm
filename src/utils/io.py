from datetime import datetime
from pathlib import Path


def _format_seconds(value) -> str:
    return str(value).replace(".", "p")


def model_label_from_config(cfg: dict) -> str:
    """Return the canonical architecture label used in run/config names."""
    mcfg = cfg.get("model", {}) or {}
    model_name = str(mcfg.get("name", "model")).lower()

    if model_name == "cnn1d":
        channels = mcfg.get("cnn_channels", mcfg.get("channels", [32, 64, 128]))
        if isinstance(channels, int):
            channels = [channels]
        channels = [int(c) for c in channels]
        final_channels = channels[-1] if channels else "na"
        depth = len(channels) if channels else "na"
        kernel = mcfg.get("cnn_kernel_size", "na")
        return f"cnn1d_c{final_channels}_n{depth}_k{kernel}"

    if model_name == "bilstm":
        hidden = mcfg.get("hidden_size", "na")
        layers = mcfg.get("num_layers", "na")
        direction = "bi" if mcfg.get("bidirectional", True) else "uni"
        return f"bilstm_h{hidden}_n{layers}_{direction}"

    if model_name in {"ecgmamba", "mamba", "bimamba"}:
        default_backbone = model_name if model_name != "ecgmamba" else "bissm"
        backbone = str(mcfg.get("backbone", default_backbone)).lower()
        d_model = mcfg.get("d_model", "na")
        layers = mcfg.get("n_layers", "na")
        if backbone == "bissm":
            state = mcfg.get("state_dim", "na")
            return f"bissm_d{d_model}_n{layers}_s{state}"
        if backbone in {"mamba", "bimamba"}:
            state = mcfg.get("d_state", mcfg.get("state_dim", "na"))
            return f"{backbone}_d{d_model}_n{layers}_s{state}"
        if backbone == "bilstm":
            lstm_layers = mcfg.get("lstm_num_layers", layers)
            return f"ecgmamba_bilstm_d{d_model}_n{lstm_layers}"
        return f"ecgmamba_{backbone}_d{d_model}_n{layers}"

    return model_name


def build_run_name(cfg: dict, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocessing = cfg.get("preprocessing", {}) or {}
    fs_target = preprocessing.get("fs_target", "na")
    target_seconds = preprocessing.get("target_seconds", "na")
    windowing = preprocessing.get("windowing", {}) or {}
    seed = cfg.get("split", {}).get("seed", "na")
    win = _format_seconds(target_seconds)
    mode = "mil" if windowing.get("enabled", False) else "crop"
    model_label = model_label_from_config(cfg)
    return f"{model_label}_{mode}_fs{fs_target}_win{win}_seed{seed}__{ts}"


def make_run_dir(runs_dir: str, run_name: str | None = None, cfg: dict | None = None) -> Path:
    if run_name is None:
        run_name = build_run_name(cfg or {})
    path = Path(runs_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
