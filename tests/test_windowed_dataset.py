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
            "windowing": {"enabled": True, "window_seconds": 5.0, "stride_seconds": 5.0, "pad_remainder": True},
        },
    )

    assert len(ds) == 4
    assert ds.record_batches == [[0, 1, 2], [3]]
    assert ds.sample_record_ids == ["r1", "r1", "r1", "r2"]
    assert ds.sample_labels == [1, 1, 1, 0]

    final_window = ds[2]["x"].squeeze(0).numpy()
    assert np.array_equal(final_window, np.array([10, 11, 0, 0, 0], dtype=np.float32))


def test_stride_windowing_drops_incomplete_remainder(tmp_path):
    data_path = tmp_path / "toy_stride.parquet"
    pd.DataFrame([{"record_id": "r1", "x": list(range(1050)), "label": 1, "fs": 100}]).to_parquet(data_path, index=False)

    ds = ParquetECGDataset(
        str(data_path),
        train=False,
        preprocess_cfg={
            "fs_target": 100,
            "target_seconds": 4.0,
            "normalize": "none",
            "windowing": {"enabled": True, "window_seconds": 4.0, "stride_seconds": 2.0, "pad_remainder": False},
        },
    )

    assert len(ds) == 4
    assert [ds[i]["segment_idx"].item() for i in range(len(ds))] == [0, 1, 2, 3]
    assert ds.record_num_segments == [4]
    assert ds._window_starts[0] == [0, 200, 400, 600]


def test_short_record_is_normalized_then_padded(tmp_path):
    data_path = tmp_path / "toy_short.parquet"
    sig = np.arange(1, 301, dtype=np.float32)
    pd.DataFrame([{"record_id": "r1", "x": sig.tolist(), "label": 1, "fs": 100}]).to_parquet(data_path, index=False)

    ds = ParquetECGDataset(
        str(data_path),
        train=False,
        preprocess_cfg={
            "fs_target": 100,
            "target_seconds": 4.0,
            "normalize": "zscore",
            "windowing": {"enabled": True, "window_seconds": 4.0, "stride_seconds": 2.0, "pad_remainder": False},
        },
    )
    out = ds[0]["x"].squeeze(0).numpy()
    assert len(ds) == 1
    assert out.shape[0] == 400
    assert np.array_equal(out[300:], np.zeros(100, dtype=np.float32))
    assert np.isclose(out[:300].mean(), 0.0, atol=1e-5)
    assert np.isclose(out[:300].std(), 1.0, atol=1e-5)
