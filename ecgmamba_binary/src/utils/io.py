from pathlib import Path
from datetime import datetime


def make_run_dir(runs_dir: str, run_name: str | None = None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(runs_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
