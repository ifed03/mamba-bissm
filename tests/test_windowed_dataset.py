import numpy as np
import pandas as pd

from data.parquet_dataset import ParquetECGDataset


def _windowed_dataset(tmp_path, signal, *, window_seconds, stride_seconds, normalize="none", pad_remainder=True):
    data_path = tmp_path / "toy.parquet"
    pd.DataFrame([{"record_id": "r1", "x": list(signal), "label": 1, "fs": 1}]).to_parquet(data_path, index=False)

    return ParquetECGDataset(
        str(data_path),
        train=False,
        preprocess_cfg={
            "fs_target": 1,
            "target_seconds": window_seconds,
            "normalize": normalize,
            "windowing": {
                "enabled": True,
                "window_seconds": window_seconds,
                "stride_seconds": stride_seconds,
                "pad_remainder": pad_remainder,
            },
        },
    )


def test_windowed_dataset_drops_final_incomplete_window_with_equal_stride(tmp_path):
    ds = _windowed_dataset(tmp_path, range(12), window_seconds=5.0, stride_seconds=5.0, pad_remainder=True)

    assert len(ds) == 2
    assert ds._window_starts[0] == [0, 5]
    assert ds.record_batches == [[0, 1]]
    assert ds.sample_record_ids == ["r1", "r1"]
    assert ds.sample_labels == [1, 1]

    final_window = ds[1]["x"].squeeze(0).numpy()
    assert np.array_equal(final_window, np.array([5, 6, 7, 8, 9], dtype=np.float32))


def test_windowed_dataset_drops_final_incomplete_window_with_overlap(tmp_path):
    ds = _windowed_dataset(tmp_path, range(12), window_seconds=5.0, stride_seconds=2.0, pad_remainder=True)

    assert len(ds) == 4
    assert ds._window_starts[0] == [0, 2, 4, 6]
    assert [ds[i]["segment_idx"].item() for i in range(len(ds))] == [0, 1, 2, 3]
    assert ds.record_num_segments == [4]


def test_windowed_dataset_exact_length_record_has_one_unpadded_window(tmp_path):
    ds = _windowed_dataset(tmp_path, range(5), window_seconds=5.0, stride_seconds=2.0, pad_remainder=True)

    assert len(ds) == 1
    assert ds._window_starts[0] == [0]
    assert np.array_equal(ds[0]["x"].squeeze(0).numpy(), np.arange(5, dtype=np.float32))


def test_short_record_is_normalized_then_padded(tmp_path):
    ds = _windowed_dataset(tmp_path, [1, 2, 3, 4], window_seconds=5.0, stride_seconds=2.0, normalize="zscore")

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
