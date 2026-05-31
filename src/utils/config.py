from pathlib import Path

import yaml


CONFIG_SEARCH_DIRS = ("final_configs", "configs")


def _find_config(root: str, relative_name: Path) -> Path | None:
    candidate = Path(root) / relative_name
    if candidate.exists():
        return candidate

    matches = sorted(Path(root).rglob(relative_name.name))
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_config_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path

    if len(path.parts) >= 2 and path.parts[0] in CONFIG_SEARCH_DIRS:
        relative_name = Path(*path.parts[1:])
        for root in CONFIG_SEARCH_DIRS:
            found = _find_config(root, relative_name)
            if found is not None:
                return found

    if len(path.parts) == 1:
        for root in CONFIG_SEARCH_DIRS:
            found = _find_config(root, path)
            if found is not None:
                return found

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
