from utils.io import build_run_name, model_label_from_config


def _base_cfg(model_cfg):
    return {
        "model": model_cfg,
        "preprocessing": {
            "fs_target": 100,
            "target_seconds": 4.0,
            "windowing": {"enabled": True},
        },
        "split": {"seed": 42},
    }


def test_cnn1d_model_label_includes_width_depth_and_kernel():
    cfg = _base_cfg(
        {
            "name": "cnn1d",
            "cnn_channels": [64, 128, 256],
            "cnn_kernel_size": 7,
        }
    )

    assert model_label_from_config(cfg) == "cnn1d_c256_n3_k7"
    assert (
        build_run_name(cfg, timestamp="20260529_120000")
        == "cnn1d_c256_n3_k7_mil_fs100_win4p0_seed42__20260529_120000"
    )


def test_standalone_bilstm_model_label_is_distinct_from_ecgmamba_bilstm_backbone():
    standalone = _base_cfg(
        {
            "name": "bilstm",
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
        }
    )
    backbone = _base_cfg(
        {
            "name": "ecgmamba",
            "backbone": "bilstm",
            "d_model": 64,
            "n_layers": 2,
            "lstm_num_layers": 2,
        }
    )

    assert model_label_from_config(standalone) == "bilstm_h128_n2_bi"
    assert model_label_from_config(backbone) == "ecgmamba_bilstm_d64_n2"


def test_ecgmamba_backbone_labels_match_clean_config_convention():
    assert (
        model_label_from_config(
            _base_cfg(
                {
                    "name": "ecgmamba",
                    "backbone": "bissm",
                    "d_model": 64,
                    "n_layers": 2,
                    "state_dim": 64,
                }
            )
        )
        == "bissm_d64_n2_s64"
    )
    assert (
        model_label_from_config(
            _base_cfg(
                {
                    "name": "mamba",
                    "backbone": "mamba",
                    "d_model": 64,
                    "n_layers": 2,
                    "d_state": 16,
                }
            )
        )
        == "mamba_d64_n2_s16"
    )
