import json
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, train_test_split


def _available_columns(data_path: str) -> list[str]:
    return pq.ParquetFile(data_path).schema_arrow.names


def resolve_group_id_column(data_path: str, group_id_col: str = "record_id") -> str:
    columns = set(_available_columns(data_path))
    if group_id_col in columns:
        return group_id_col
    if "record_id" in columns:
        return "record_id"
    raise ValueError(f"Could not resolve a grouping column from {group_id_col!r} or 'record_id'")


def load_labels(data_path: str):
    table = pq.read_table(data_path, columns=["label"])
    return table["label"].to_numpy().tolist()


def load_group_labels(data_path: str, group_id_col: str = "record_id"):
    resolved_group_col = resolve_group_id_column(data_path, group_id_col)
    table = pq.read_table(data_path, columns=[resolved_group_col, "label"])
    group_ids = table[resolved_group_col].to_pylist()
    labels = table["label"].to_pylist()

    group_to_label = {}
    group_to_indices = {}
    for idx, (group_id, label) in enumerate(zip(group_ids, labels, strict=True)):
        if group_id in group_to_label and int(group_to_label[group_id]) != int(label):
            raise ValueError(f"Group {group_id!r} has inconsistent labels across rows")
        group_to_label[group_id] = int(label)
        group_to_indices.setdefault(group_id, []).append(idx)

    unique_groups = list(group_to_label)
    unique_labels = [group_to_label[group_id] for group_id in unique_groups]
    return resolved_group_col, unique_groups, unique_labels, group_to_indices


def _expand_group_indices(group_ids: list, group_to_indices: dict) -> list[int]:
    indices = []
    for group_id in group_ids:
        indices.extend(group_to_indices[group_id])
    return indices


def make_holdout_splits(data_path: str, seed: int, train_size: float = 0.8, val_size: float = 0.1, group_id_col: str = "record_id"):
    resolved_group_col, group_ids, group_labels, group_to_indices = load_group_labels(data_path, group_id_col)
    train_groups, tmp_groups, y_train, y_tmp = train_test_split(
        group_ids,
        group_labels,
        train_size=train_size,
        stratify=group_labels,
        random_state=seed,
    )
    rel_val = val_size / (1 - train_size)
    val_groups, test_groups, _, _ = train_test_split(
        tmp_groups,
        y_tmp,
        train_size=rel_val,
        stratify=y_tmp,
        random_state=seed,
    )
    return {
        "group_id_col": resolved_group_col,
        "train": _expand_group_indices(train_groups, group_to_indices),
        "val": _expand_group_indices(val_groups, group_to_indices),
        "test": _expand_group_indices(test_groups, group_to_indices),
    }


def make_kfold_splits(data_path: str, k: int, seed: int, group_id_col: str = "record_id"):
    resolved_group_col, group_ids, group_labels, group_to_indices = load_group_labels(data_path, group_id_col)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(group_ids, group_labels):
        train_groups = [group_ids[i] for i in tr.tolist()]
        test_groups = [group_ids[i] for i in te.tolist()]
        folds.append(
            {
                "group_id_col": resolved_group_col,
                "train": _expand_group_indices(train_groups, group_to_indices),
                "test": _expand_group_indices(test_groups, group_to_indices),
            }
        )
    return folds


def save_split(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
