from pathlib import Path

import pytest
import torch

from data.parquet_dataset import ParquetECGDataset
from evaluate.efficiency import efficiency_metadata_from_config
from models import build_model
from models.cnn_baseline import CNNBaseline
from models.lstm_baseline import BiLSTMBaseline
from utils.config import load_config


def _bilstm_cfg():
    return {
        "model": {
            "name": "bilstm",
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "dropout": 0.2,
            "pooling": "mean",
        }
    }


def test_build_model_bilstm_factory():
    model = build_model(_bilstm_cfg())
    assert isinstance(model, BiLSTMBaseline)


@pytest.mark.parametrize("seq_len", [400, 600, 800, 1000])
def test_bilstm_forward_shapes(seq_len):
    model = BiLSTMBaseline(_bilstm_cfg())
    x = torch.randn(2, 1, seq_len)
    logits, features = model(x)
    assert logits.shape == (2,)
    assert features.shape == (2, 256)


def test_bilstm_standard_lstm_no_sigmoid():
    model = BiLSTMBaseline(_bilstm_cfg())
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert model.__class__.__name__ != "PeepholeLSTM"
    assert not any(isinstance(m, torch.nn.Sigmoid) for m in model.modules())


def test_bilstm_backward_has_finite_nonzero_lstm_gradient():
    model = BiLSTMBaseline(_bilstm_cfg())
    x = torch.randn(2, 1, 400)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    logits, _ = model(x)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()

    grads = [p.grad for n, p in model.named_parameters() if "lstm" in n and p.grad is not None]
    assert grads
    assert any(torch.isfinite(g).all() and torch.count_nonzero(g).item() > 0 for g in grads)


def test_bilstm_parameter_count_expected_scale():
    model = BiLSTMBaseline(_bilstm_cfg())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == 529665
    assert trainable == 529665


def test_bilstm_configs_load_and_match_protocol():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_bilstm_100hz_win{sec}s_stride2s.yaml")
        assert cfg["model"]["name"] == "bilstm"
        assert cfg["model"]["hidden_size"] == 128
        assert cfg["model"]["num_layers"] == 2
        assert cfg["model"]["bidirectional"] is True
        assert cfg["model"]["dropout"] == 0.2
        assert cfg["model"]["pooling"] == "mean"
        assert cfg["preprocessing"]["fs_target"] == 100
        assert cfg["preprocessing"]["target_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["window_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["stride_seconds"] == 2.0
        assert cfg["preprocessing"]["windowing"]["pad_remainder"] is False
        assert cfg["split"]["train_ratio"] == 0.7
        assert cfg["split"]["val_ratio"] == 0.1


def test_efficiency_metadata_accepts_bilstm_config_shape():
    cfg = load_config("configs/binary_bilstm_100hz_win10s_stride2s.yaml")
    metadata = efficiency_metadata_from_config(cfg)
    assert metadata["input_length_samples"] == 1000


def test_padding_caveat_check_clean_data_protocol():
    cfg10 = load_config("configs/binary_bilstm_100hz_win10s_stride2s.yaml")
    data_path = Path(cfg10["paths"]["data_path"])
    if not data_path.exists():
        pytest.skip("Cloud environment missing clean-data parquet; padding caveat unverifiable here.")

    counts = {}
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_bilstm_100hz_win{sec}s_stride2s.yaml")
        ds = ParquetECGDataset(str(data_path), train=False, preprocess_cfg=cfg["preprocessing"])
        padded_windows = 0
        for i in range(len(ds)):
            x = ds[i]["x"].squeeze(0)
            if torch.any(x == 0):
                if torch.count_nonzero(x[-int(cfg["preprocessing"]["fs_target"]):] == 0) > 0:
                    padded_windows += 1
        counts[sec] = padded_windows

    assert counts[10] >= 1
    assert counts[4] == 0
    assert counts[6] == 0
    assert counts[8] == 0


def test_bilstm_dataloader_model_smoke_per_config():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_bilstm_100hz_win{sec}s_stride2s.yaml")
        model = build_model(cfg)
        x = torch.randn(2, 1, int(sec * 100))
        logits, features = model(x)
        assert logits.shape == (2,)
        assert features.shape == (2, 256)


def test_cnn1d_configs_load_and_match_protocol():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_cnn1d_c256_n3_k7_100hz_win{sec}s_stride2s.yaml")
        assert cfg["model"]["name"] == "cnn1d"
        assert cfg["model"]["in_channels"] == 1
        assert cfg["model"]["cnn_channels"] == [64, 128, 256]
        assert cfg["model"]["cnn_kernel_size"] == 7
        assert cfg["model"]["cnn_stride"] == 2
        assert cfg["model"]["cnn_dropout"] == 0.1
        assert cfg["model"]["cnn_batchnorm"] is True
        assert cfg["preprocessing"]["fs_target"] == 100
        assert cfg["preprocessing"]["target_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["window_seconds"] == float(sec)
        assert cfg["preprocessing"]["windowing"]["stride_seconds"] == 2.0
        assert cfg["preprocessing"]["windowing"]["pad_remainder"] is False
        assert cfg["split"]["train_ratio"] == 0.7
        assert cfg["split"]["val_ratio"] == 0.1


def test_cnn1d_dataloader_model_smoke_per_config():
    for sec in [4, 6, 8, 10]:
        cfg = load_config(f"configs/binary_cnn1d_c256_n3_k7_100hz_win{sec}s_stride2s.yaml")
        model = build_model(cfg)
        assert isinstance(model, CNNBaseline)
        x = torch.randn(2, 1, int(sec * 100))
        logits, features = model(x)
        assert logits.shape == (2,)
        assert features.shape == (2, 256)
