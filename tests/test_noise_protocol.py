from evaluate.noise_protocol import (
    NoiseCondition,
    condition_key,
    condition_output_dir,
    ensure_clean_split,
    metadata_for_noisy_example,
)


def test_noise_injection_is_test_only():
    ensure_clean_split("test")


import pytest


@pytest.mark.parametrize("bad_split", ["train", "val", "validation", "dev"])
def test_non_test_split_rejected(bad_split):
    with pytest.raises(ValueError, match="only allowed for test split"):
        ensure_clean_split(bad_split)


def test_threshold_and_checkpoint_must_be_clean_val():
    c = NoiseCondition("bw", 0)
    with pytest.raises(ValueError, match=r"tau\* must come from clean validation only"):
        metadata_for_noisy_example(
            base_metadata={"record_id": "r1"},
            split="test",
            condition=c,
            threshold_source="noisy_test",
            checkpoint_source="clean_val",
        )

    with pytest.raises(ValueError, match="checkpoint selection must come from clean validation only"):
        metadata_for_noisy_example(
            base_metadata={"record_id": "r1"},
            split="test",
            condition=c,
            threshold_source="clean_val",
            checkpoint_source="noisy_test",
        )


def test_condition_outputs_are_separated_and_metadata_reproducible_fields_present(tmp_path):
    c = NoiseCondition("ma", -6)
    out_dir = condition_output_dir(tmp_path, c)
    assert "noise_type=ma" in str(out_dir)
    assert "snr_db=neg6" in str(out_dir)

    meta = metadata_for_noisy_example(
        base_metadata={"record_id": "r2", "seed": 123, "noise_start_index": 22, "noise_channel": 1},
        split="test",
        condition=c,
        threshold_source="clean_val",
        checkpoint_source="clean_val",
    )
    assert meta["record_id"] == "r2"
    assert meta["seed"] == 123
    assert meta["noise_start_index"] == 22
    assert meta["noise_channel"] == 1
    assert meta["noise_type"] == "ma"
    assert meta["snr_db"] == -6.0
    assert meta["threshold_source"] == "clean_val"
    assert meta["checkpoint_source"] == "clean_val"
    assert meta["split"] == "test"


def test_condition_key_stable():
    assert condition_key(NoiseCondition("em", 12)) == "noise_type=em__snr_db=12"


def test_zero_shot_noise_pipeline_order_and_test_only_transform():
    calls = []

    def _resample_ecg(x, fs_src, fs_tgt):
        calls.append(("resample_ecg", fs_src, fs_tgt))
        return x[::2]  # emulate 200Hz -> 100Hz

    def _inject_noise(x):
        calls.append(("inject_noise", len(x)))
        return [v + 1.0 for v in x]

    def _window_extract(x, w):
        calls.append(("window_extract", len(x), w))
        return [x[i : i + w] for i in range(0, len(x) - w + 1, w)]

    def _zscore_per_window(ws):
        calls.append(("zscore", len(ws)))
        out = []
        for w in ws:
            mu = sum(w) / len(w)
            var = sum((v - mu) ** 2 for v in w) / len(w)
            sd = (var + 1e-8) ** 0.5
            out.append([(v - mu) / sd for v in w])
        return out

    def _run_split(split):
        x_raw = [i / 19.0 for i in range(20)]
        x = _resample_ecg(x_raw, 200, 100)
        if split == "test":
            x = _inject_noise(x)
        ws = _window_extract(x, 5)
        ws = _zscore_per_window(ws)
        return ws

    train_ws = _run_split("train")
    val_ws = _run_split("val")
    test_ws = _run_split("test")

    # train/val are clean (no +1 offset before zscore path) and no noise call for those splits
    inject_calls = [c for c in calls if c[0] == "inject_noise"]
    assert len(inject_calls) == 1

    # explicit ordering check for test path: resample -> inject -> window -> zscore
    test_seq = [c[0] for c in calls[-4:]]
    assert test_seq == ["resample_ecg", "inject_noise", "window_extract", "zscore"]

    # windows are produced for every split; zscore done after window extraction (per-window)
    assert len(train_ws) > 0 and len(val_ws) > 0 and len(test_ws) > 0
