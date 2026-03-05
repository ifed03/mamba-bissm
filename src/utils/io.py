from pathlib import Path
from datetime import datetime


def build_run_name(cfg: dict, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = str(cfg.get("model", {}).get("name", "model"))
    preprocessing = cfg.get("preprocessing", {})
    fs_target = preprocessing.get("fs_target", "na")
    target_seconds = preprocessing.get("target_seconds", "na")
    windowing = preprocessing.get("windowing", {})
    seed = cfg.get("split", {}).get("seed", "na")
    win = str(target_seconds).replace(".", "p")
    mode = "mil" if windowing.get("enabled", False) else "crop"
    return f"{model_name}_{mode}_fs{fs_target}_win{win}_seed{seed}__{ts}"


def make_run_dir(runs_dir: str, run_name: str | None = None, cfg: dict | None = None) -> Path:
    if run_name is None:
        run_name = build_run_name(cfg or {})
    path = Path(runs_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
