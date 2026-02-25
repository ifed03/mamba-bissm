#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

from data.splits import make_holdout_splits, make_kfold_splits, save_split
from utils.config import load_config


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--kfold", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = cfg["split"]["seed"]
    data_path = cfg["paths"]["data_path"]
    splits_dir = Path(cfg["paths"]["splits_dir"])

    if args.kfold > 1:
        folds = make_kfold_splits(data_path, args.kfold, seed)
        for i, fold in enumerate(folds):
            save_split(splits_dir / f"kfold{args.kfold}_seed{seed}" / f"fold_{i}.json", fold)
    else:
        split = make_holdout_splits(data_path, seed, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"])
        save_split(splits_dir / f"holdout_seed{seed}" / "split.json", split)
