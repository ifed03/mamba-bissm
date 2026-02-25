#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import subprocess
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--results", default="runs/results.csv")
    args = p.parse_args()

    rows = []
    for cfg in args.configs:
        run_name = Path(cfg).stem
        subprocess.run(["python", "scripts/train_model.py", "--config", cfg, "--run-name", run_name], check=True)
        metrics_path = Path("runs") / run_name / "metrics.json"
        import json

        m = json.loads(metrics_path.read_text())
        rows.append({"config": cfg, "auroc": m["test"]["auroc"], "f1": m["test"]["f1"], "acc": m["test"]["acc"]})

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "auroc", "f1", "acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {out}")
