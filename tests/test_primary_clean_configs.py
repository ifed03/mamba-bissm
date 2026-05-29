from utils.config import load_config


def test_primary_clean_configs_window_shapes_and_stride():
    expected = {
        "configs/binary_ecgmamba_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_ecgmamba_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_ecgmamba_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_ecgmamba_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_bissm_d64_n2_s64_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_bissm_d64_n2_s64_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_bissm_d64_n2_s64_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_bissm_d64_n2_s64_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_bissm_d64_n4_s64_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_bissm_d64_n4_s64_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_bissm_d64_n4_s64_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_bissm_d64_n4_s64_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_bilstm_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_bilstm_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_bilstm_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_bilstm_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_bimamba_2layer_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_bimamba_2layer_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_bimamba_2layer_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_bimamba_2layer_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_mamba_d64_n2_s16_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_mamba_d64_n2_s16_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_mamba_d64_n2_s16_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_mamba_d64_n2_s16_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_mamba_d64_n4_s16_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_mamba_d64_n4_s16_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_mamba_d64_n4_s16_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_mamba_d64_n4_s16_100hz_win10s_stride2s.yaml": 10.0,
    }

    for path, seconds in expected.items():
        cfg = load_config(path)
        assert cfg["split"]["train_ratio"] == 0.7
        assert cfg["split"]["val_ratio"] == 0.1
        assert cfg["preprocessing"]["fs_target"] == 100
        assert cfg["preprocessing"]["target_seconds"] == seconds
        assert cfg["preprocessing"]["windowing"]["enabled"] is True
        assert cfg["preprocessing"]["windowing"]["window_seconds"] == seconds
        assert cfg["preprocessing"]["windowing"]["stride_seconds"] == 2.0
        assert cfg["preprocessing"]["windowing"]["pad_remainder"] is False
        assert "accumulation_steps" not in cfg["training"]
