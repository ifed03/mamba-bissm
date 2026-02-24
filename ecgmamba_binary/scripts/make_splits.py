#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import make_holdout_splits, make_kfold_splits, save_split
from src.utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kfold", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["split"]["seed"]
    data_path = cfg["paths"]["data_path"]
    splits_dir = Path(cfg["paths"]["splits_dir"])

    if args.kfold > 1:
        folds = make_kfold_splits(data_path, args.kfold, seed)
        for i, fold in enumerate(folds):
            save_split(splits_dir / f"kfold_{args.kfold}" / f"fold_{i}.json", fold)
    else:
        split = make_holdout_splits(data_path, seed, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"])
        save_split(splits_dir / f"seed_{seed}" / "split.json", split)
