import numpy as np
import pandas as pd

from data.parquet_dataset import ParquetECGDataset


def test_windowed_dataset_expands_records_into_segments(tmp_path):
    data_path = tmp_path / "toy.parquet"
    pd.DataFrame(
        [
            {"record_id": "r1", "x": list(range(12)), "label": 1, "fs": 1},
            {"record_id": "r2", "x": list(range(5)), "label": 0, "fs": 1},
        ]
    ).to_parquet(data_path, index=False)

    ds = ParquetECGDataset(
        str(data_path),
        train=False,
        preprocess_cfg={
            "fs_target": 1,
            "target_seconds": 5.0,
            "normalize": "none",
            "windowing": {"enabled": True, "window_seconds": 5.0, "pad_remainder": True},
        },
    )

    assert len(ds) == 4
    assert ds.record_batches == [[0, 1, 2], [3]]
    assert ds.sample_record_ids == ["r1", "r1", "r1", "r2"]
    assert ds.sample_labels == [1, 1, 1, 0]

    final_window = ds[2]["x"].squeeze(0).numpy()
    assert np.array_equal(final_window, np.array([10, 11, 0, 0, 0], dtype=np.float32))
