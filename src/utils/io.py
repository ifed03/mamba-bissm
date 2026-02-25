from pathlib import Path
from datetime import datetime


def build_run_name(cfg: dict, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = str(cfg.get("model", {}).get("name", "model"))
    fs_target = cfg.get("preprocessing", {}).get("fs_target", "na")
    target_seconds = cfg.get("preprocessing", {}).get("target_seconds", "na")
    seed = cfg.get("split", {}).get("seed", "na")
    win = str(target_seconds).replace(".", "p")
    return f"{model_name}_fs{fs_target}_win{win}_seed{seed}__{ts}"


def make_run_dir(runs_dir: str, run_name: str | None = None, cfg: dict | None = None) -> Path:
    if run_name is None:
        run_name = build_run_name(cfg or {})
    path = Path(runs_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
