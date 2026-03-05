import pandas as pd

from data.splits import make_holdout_splits


def test_holdout_split_keeps_patient_groups_together(tmp_path):
    rows = []
    for patient_idx in range(20):
        label = 0 if patient_idx < 10 else 1
        for record_idx in range(2):
            rows.append(
                {
                    "patient_id": f"patient_{patient_idx}",
                    "record_id": f"record_{patient_idx}_{record_idx}",
                    "x": [0.0, 1.0, 2.0],
                    "label": label,
                    "fs": 1,
                }
            )

    data_path = tmp_path / "toy.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    split = make_holdout_splits(str(data_path), seed=42, train_size=0.8, val_size=0.1, group_id_col="patient_id")
    index_to_patient = {idx: row["patient_id"] for idx, row in enumerate(rows)}

    split_patients = {
        split_name: {index_to_patient[idx] for idx in split_indices}
        for split_name, split_indices in split.items()
        if split_name in {"train", "val", "test"}
    }

    assert split["group_id_col"] == "patient_id"
    assert split_patients["train"].isdisjoint(split_patients["val"])
    assert split_patients["train"].isdisjoint(split_patients["test"])
    assert split_patients["val"].isdisjoint(split_patients["test"])

    for patient_id in {row["patient_id"] for row in rows}:
        membership = [patient_id in split_patients[split_name] for split_name in ("train", "val", "test")]
        assert sum(membership) == 1
