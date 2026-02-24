import json
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_labels(data_path: str):
    table = pq.read_table(data_path, columns=["label"])
    return table["label"].to_numpy().tolist()


def make_holdout_splits(data_path: str, seed: int, train_size: float = 0.8, val_size: float = 0.1):
    labels = load_labels(data_path)
    idx = list(range(len(labels)))
    train_idx, tmp_idx, y_train, y_tmp = train_test_split(idx, labels, train_size=train_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1 - train_size)
    val_idx, test_idx, _, _ = train_test_split(tmp_idx, y_tmp, train_size=rel_val, stratify=y_tmp, random_state=seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_kfold_splits(data_path: str, k: int, seed: int):
    labels = load_labels(data_path)
    idx = list(range(len(labels)))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(idx, labels):
        folds.append({"train": tr.tolist(), "test": te.tolist()})
    return folds


def save_split(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
