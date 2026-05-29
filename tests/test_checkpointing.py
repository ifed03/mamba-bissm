import pytest

torch = pytest.importorskip("torch")

from train.checkpointing import infer_model_dims, validate_config_matches_state_dict


def test_checkpoint_shape_guard_skips_standalone_baselines_without_backbone_keys():
    validate_config_matches_state_dict(
        {"model": {"name": "bilstm", "hidden_size": 128, "num_layers": 2}},
        {"lstm.weight_ih_l0": torch.zeros(512, 1), "head.weight": torch.zeros(1, 256)},
    )


def test_checkpoint_shape_guard_still_checks_mamba_backbones():
    cfg = {"model": {"name": "mamba", "d_model": 64, "n_layers": 2}}
    state_dict = {
        "pos.pe": torch.zeros(1, 1, 128),
        "backbone.0.norm.weight": torch.zeros(128),
        "backbone.1.norm.weight": torch.zeros(128),
    }

    with pytest.raises(ValueError, match="Config/checkpoint mismatch"):
        validate_config_matches_state_dict(cfg, state_dict)


def test_checkpoint_shape_guard_infers_bimamba_layer_keys():
    state_dict = {
        "pos.pe": torch.zeros(1, 1, 128),
        "backbone.fwd.layers.0.A_log": torch.zeros(128),
        "backbone.fwd.layers.1.A_log": torch.zeros(128),
        "backbone.bwd.layers.0.A_log": torch.zeros(128),
        "backbone.bwd.layers.1.A_log": torch.zeros(128),
        "backbone.fuse.weight": torch.zeros(128, 256),
    }

    assert infer_model_dims(state_dict) == (128, 2)


def test_checkpoint_shape_guard_infers_mamba_layers_module_keys():
    state_dict = {
        "pos.pe": torch.zeros(1, 1, 128),
        "backbone.layers.0.A_log": torch.zeros(128),
        "backbone.layers.1.A_log": torch.zeros(128),
        "backbone.layers.2.A_log": torch.zeros(128),
        "backbone.layers.3.A_log": torch.zeros(128),
        "backbone.norm.weight": torch.zeros(128),
    }

    assert infer_model_dims(state_dict) == (128, 4)


def test_checkpoint_shape_guard_skips_ecgmamba_non_mamba_backbone():
    validate_config_matches_state_dict(
        {"model": {"name": "ecgmamba", "backbone": "bilstm", "d_model": 128, "n_layers": 2}},
        {"backbone.lstm.weight_ih_l0": torch.zeros(512, 1), "head.weight": torch.zeros(1, 256)},
    )
