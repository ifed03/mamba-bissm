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
        "configs/binary_cnn1d_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_cnn1d_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_cnn1d_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_cnn1d_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_bimamba_2layer_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_bimamba_2layer_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_bimamba_2layer_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_bimamba_2layer_100hz_win10s_stride2s.yaml": 10.0,
        "configs/binary_ecgmamba_bilstm_d64_n2_100hz_win4s_stride2s.yaml": 4.0,
        "configs/binary_ecgmamba_bilstm_d64_n2_100hz_win6s_stride2s.yaml": 6.0,
        "configs/binary_ecgmamba_bilstm_d64_n2_100hz_win8s_stride2s.yaml": 8.0,
        "configs/binary_ecgmamba_bilstm_d64_n2_100hz_win10s_stride2s.yaml": 10.0,
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


def test_controlled_ecgmamba_backbone_configs_share_frontend_settings():
    paths_by_backbone = {
        "bissm": "configs/binary_bissm_d64_n2_s64_100hz_win4s_stride2s.yaml",
        "mamba": "configs/binary_mamba_d64_n2_s16_100hz_win4s_stride2s.yaml",
        "bimamba": "configs/binary_bimamba_2layer_100hz_win4s_stride2s.yaml",
        "bilstm": "configs/binary_ecgmamba_bilstm_d64_n2_100hz_win4s_stride2s.yaml",
    }

    configs = {name: load_config(path) for name, path in paths_by_backbone.items()}
    reference = configs["bissm"]
    for name, cfg in configs.items():
        assert cfg["model"]["backbone"] == name
        assert cfg["model"]["d_model"] == reference["model"]["d_model"]
        assert cfg["model"]["dropout"] == reference["model"]["dropout"]
        assert cfg["model"]["use_encoder"] is True
        assert cfg["preprocessing"] == reference["preprocessing"]

    bilstm_model = configs["bilstm"]["model"]
    assert bilstm_model["name"] == "ecgmamba"
    assert bilstm_model["lstm_bidirectional"] is True
    assert bilstm_model["lstm_hidden_size"] == bilstm_model["d_model"] // 2
    assert bilstm_model["lstm_num_layers"] == bilstm_model["n_layers"]
    assert bilstm_model["lstm_dropout"] == bilstm_model["dropout"]
    assert bilstm_model["lstm_layernorm"] is True


def test_all_main_ecgmamba_bilstm_backbone_configs_are_bidirectional_and_half_width():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(
            f"configs/binary_ecgmamba_bilstm_d64_n2_100hz_win{sec}s_stride2s.yaml"
        )
        model_cfg = cfg["model"]
        assert model_cfg["name"] == "ecgmamba"
        assert model_cfg["backbone"] == "bilstm"
        assert model_cfg["lstm_bidirectional"] is True
        assert model_cfg["lstm_hidden_size"] == model_cfg["d_model"] // 2
        assert model_cfg["lstm_num_layers"] == model_cfg["n_layers"]


def test_all_main_cnn1d_external_baseline_configs_match_protocol():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_cnn1d_100hz_win{sec}s_stride2s.yaml")
        model_cfg = cfg["model"]
        assert model_cfg["name"] == "cnn1d"
        assert model_cfg["in_channels"] == 1
        assert model_cfg["cnn_channels"] == [64, 128, 256]
        assert model_cfg["cnn_kernel_size"] == 7
        assert model_cfg["cnn_stride"] == 2
        assert model_cfg["cnn_dropout"] == 0.1
        assert model_cfg["cnn_batchnorm"] is True
        assert cfg["preprocessing"]["fs_target"] == 100
        assert cfg["preprocessing"]["target_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["window_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["stride_seconds"] == 2.0
        assert cfg["preprocessing"]["windowing"]["pad_remainder"] is False
