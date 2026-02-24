#!/usr/bin/env python
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--results", default="runs/results.csv")
    args = parser.parse_args()

    rows = []
    for cfg in args.configs:
        run_name = Path(cfg).stem
        subprocess.run(["python", "scripts/train_model.py", "--config", cfg, "--run-name", run_name], check=True)
        metrics_path = Path("runs") / run_name / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "config": cfg,
                "auroc": metrics["test"]["auroc"],
                "f1": metrics["test"]["f1"],
                "acc": metrics["test"]["acc"],
            }
        )

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "auroc", "f1", "acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out}")
