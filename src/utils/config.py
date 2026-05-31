from pathlib import Path

import yaml


CONFIG_SEARCH_DIRS = ("final_configs", "configs")


def resolve_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path

    if len(path.parts) >= 2 and path.parts[0] in CONFIG_SEARCH_DIRS:
        relative_name = Path(*path.parts[1:])
        for root in CONFIG_SEARCH_DIRS:
            candidate = Path(root) / relative_name
            if candidate.exists():
                return candidate

    if len(path.parts) == 1:
        for root in CONFIG_SEARCH_DIRS:
            candidate = Path(root) / path.name
            if candidate.exists():
                return candidate

    return path


def load_config(path: str | Path) -> dict:
    with open(resolve_config_path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(path: str | Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
