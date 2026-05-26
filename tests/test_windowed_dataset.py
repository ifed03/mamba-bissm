import numpy as np
import pandas as pd

from data.parquet_dataset import ParquetECGDataset


def _windowed_dataset(
    tmp_path,
    rows,
    *,
    fs_target,
    target_seconds,
    normalize,
    window_seconds,
    stride_seconds,
    pad_remainder=False,
):
    data_path = tmp_path / "toy.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    return ParquetECGDataset(
        str(data_path),
        train=False,
        preprocess_cfg={
            "fs_target": fs_target,
            "target_seconds": target_seconds,
            "normalize": normalize,
            "windowing": {
                "enabled": True,
                "window_seconds": window_seconds,
                "stride_seconds": stride_seconds,
                "pad_remainder": pad_remainder,
            },
        },
    )


def test_windowed_dataset_expands_records_into_segments_without_long_record_remainder_padding(tmp_path):
    ds = _windowed_dataset(
        tmp_path,
        [{"record_id": "r1", "x": list(range(12)), "label": 1, "fs": 1}],
        fs_target=1,
        target_seconds=5.0,
        normalize="none",
        window_seconds=5.0,
        stride_seconds=5.0,
    )

    assert ds.stride_seconds == 5.0
    assert ds.pad_remainder is False
    assert len(ds) == 2
    assert ds.record_num_segments == [2]
    assert ds._window_starts[0] == [0, 5]
    assert ds.record_batches == [[0, 1]]
    assert ds.sample_record_ids == ["r1", "r1"]
    assert ds.sample_labels == [1, 1]

    final_window = ds[1]["x"].squeeze(0).numpy()
    assert np.array_equal(final_window, np.array([5, 6, 7, 8, 9], dtype=np.float32))


def test_overlapping_long_record_drops_final_incomplete_window(tmp_path):
    ds = _windowed_dataset(
        tmp_path,
        [{"record_id": "r1", "x": list(range(12)), "label": 1, "fs": 1}],
        fs_target=1,
        target_seconds=5.0,
        normalize="none",
        window_seconds=5.0,
        stride_seconds=2.0,
    )

    assert ds.stride_seconds == 2.0
    assert ds.pad_remainder is False
    assert len(ds) == 4
    assert ds.record_num_segments == [4]
    assert ds._window_starts[0] == [0, 2, 4, 6]
    assert ds.record_batches == [[0, 1, 2, 3]]
    assert ds.sample_record_ids == ["r1", "r1", "r1", "r1"]
    assert ds.sample_labels == [1, 1, 1, 1]


def test_exact_length_record_produces_single_unpadded_window(tmp_path):
    ds = _windowed_dataset(
        tmp_path,
        [{"record_id": "r1", "x": list(range(5)), "label": 0, "fs": 1}],
        fs_target=1,
        target_seconds=5.0,
        normalize="none",
        window_seconds=5.0,
        stride_seconds=2.0,
    )

    assert len(ds) == 1
    assert ds.record_num_segments == [1]
    assert ds._window_starts[0] == [0]
    assert ds.record_batches == [[0]]
    assert ds.sample_record_ids == ["r1"]
    assert ds.sample_labels == [0]

    window = ds[0]["x"].squeeze(0).numpy()
    assert np.array_equal(window, np.array([0, 1, 2, 3, 4], dtype=np.float32))


def test_protocol_scale_stride_drops_final_incomplete_remainder(tmp_path):
    ds = _windowed_dataset(
        tmp_path,
        [{"record_id": "r1", "x": list(range(1050)), "label": 1, "fs": 100}],
        fs_target=100,
        target_seconds=4.0,
        normalize="none",
        window_seconds=4.0,
        stride_seconds=2.0,
    )

    assert ds.stride_seconds == 2.0
    assert ds.pad_remainder is False
    assert len(ds) == 4
    assert ds.record_num_segments == [4]
    assert ds._window_starts[0] == [0, 200, 400, 600]
    assert ds.record_batches == [[0, 1, 2, 3]]
    assert ds.sample_record_ids == ["r1", "r1", "r1", "r1"]
    assert ds.sample_labels == [1, 1, 1, 1]

    window_len = 400
    for i, start in enumerate(ds._window_starts[0]):
        window = ds[i]["x"].squeeze(0).numpy()
        expected = np.arange(start, start + window_len, dtype=np.float32)
        assert np.array_equal(window, expected)


def test_short_record_is_normalized_then_padded(tmp_path):
    sig = np.arange(1, 301, dtype=np.float32)
    ds = _windowed_dataset(
        tmp_path,
        [{"record_id": "r1", "x": sig.tolist(), "label": 1, "fs": 100}],
        fs_target=100,
        target_seconds=4.0,
        normalize="zscore",
        window_seconds=4.0,
        stride_seconds=2.0,
    )

    out = ds[0]["x"].squeeze(0).numpy()

    assert ds.stride_seconds == 2.0
    assert ds.pad_remainder is False
    assert len(ds) == 1
    assert ds.record_num_segments == [1]
    assert ds._window_starts[0] == [0]
    assert ds.record_batches == [[0]]
    assert ds.sample_record_ids == ["r1"]
    assert ds.sample_labels == [1]

    assert out.shape[0] == 400
    assert np.array_equal(out[300:], np.zeros(100, dtype=np.float32))
    assert np.isclose(out[:300].mean(), 0.0, atol=1e-5)
    assert np.isclose(out[:300].std(), 1.0, atol=1e-5)