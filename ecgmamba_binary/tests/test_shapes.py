import torch

from models.cnn_baseline import CNNBaseline
from models.ecgmamba import ECGMamba


def test_model_shapes():
    x = torch.randn(2, 1, 1000)
    cnn = CNNBaseline({"model": {"channels": [16, 32, 64]}})
    y, f = cnn(x)
    assert y.shape == (2,)
    assert f.shape[0] == 2

    mcfg = {
        "model": {
            "d_model": 32,
            "n_layers": 2,
            "dropout": 0.1,
            "state_dim": 8,
            "kernel_size": 4,
            "expansion": 2,
            "ffn_hidden_mult": 2,
            "ffn_kernel_size": 1,
            "use_encoder": True,
            "use_layernorm": True,
            "use_ffn": True,
        }
    }
    mm = ECGMamba(mcfg)
    y2, f2 = mm(x)
    assert y2.shape == (2,)
    assert f2.shape[0] == 2
