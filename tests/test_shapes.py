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
    assert f2.shape == (2, mm.d_model)


def _small_ecgmamba_cfg():
    return {
        "model": {
            "name": "ecgmamba",
            "backbone": "bissm",
            "d_model": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "state_dim": 4,
            "kernel_size": 3,
            "expansion": 2,
            "ffn_hidden_mult": 2,
            "ffn_kernel_size": 1,
            "use_encoder": True,
            "use_layernorm": True,
            "use_ffn": True,
        }
    }


def test_ecgmamba_final_input_lengths_shape():
    model = ECGMamba(_small_ecgmamba_cfg())
    model.eval()

    for input_len, encoded_len in [(400, 100), (600, 150), (800, 200), (1000, 250)]:
        x = torch.randn(2, 1, input_len)
        with torch.no_grad():
            seq = model._to_sequence(x)
            logits, pooled = model(x)

        assert seq.shape == (2, encoded_len, model.d_model)
        assert logits.shape == (2,)
        assert pooled.shape == (2, model.d_model)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(pooled).all()


def test_ecgmamba_returns_raw_logits_not_probabilities():
    model = ECGMamba(_small_ecgmamba_cfg())
    model.eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(2.5)
        logits, _ = model(torch.randn(2, 1, 400))

    assert logits.shape == (2,)
    assert torch.all(logits > 1.0)


def _small_ecgmamba_bilstm_cfg():
    return {
        "model": {
            "name": "ecgmamba",
            "backbone": "bilstm",
            "d_model": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "lstm_dropout": 0.2,
            "lstm_bidirectional": True,
            "lstm_layernorm": True,
            "use_encoder": True,
            "use_layernorm": True,
        }
    }


def test_ecgmamba_bilstm_backbone_shapes_and_baseline_unchanged():
    from models.lstm_baseline import BiLSTMBaseline

    model = ECGMamba(_small_ecgmamba_bilstm_cfg())
    model.eval()
    x = torch.randn(2, 1, 400)

    with torch.no_grad():
        seq = model._to_sequence(x)
        backbone_out = model.backbone(model.pos(seq))
        logits, pooled = model(x)

    assert seq.shape == (2, 100, model.d_model)
    assert backbone_out.shape == (2, 100, model.d_model)
    assert logits.shape == (2,)
    assert pooled.shape == (2, model.d_model)
    assert model.backbone.lstm.input_size == model.d_model
    assert model.backbone.lstm.dropout == 0.0

    baseline = BiLSTMBaseline(
        {
            "model": {
                "name": "bilstm",
                "hidden_size": 8,
                "num_layers": 1,
                "bidirectional": True,
                "dropout": 0.0,
                "pooling": "mean",
            }
        }
    )
    baseline_logits, baseline_features = baseline(x)
    assert baseline.lstm.input_size == 1
    assert baseline_logits.shape == (2,)
    assert baseline_features.shape == (2, 16)


def _small_cnn1d_cfg():
    return {
        "model": {
            "name": "cnn1d",
            "in_channels": 1,
            "cnn_channels": [8, 16, 32],
            "cnn_kernel_size": 7,
            "cnn_stride": 2,
            "cnn_dropout": 0.1,
            "cnn_batchnorm": True,
        }
    }


def test_cnn1d_baseline_shapes_and_finite_outputs():
    model = CNNBaseline(_small_cnn1d_cfg())
    x = torch.randn(2, 1, 400)
    logit, features = model(x)

    assert logit.shape == (2,)
    assert features.ndim == 2
    assert features.shape == (2, 32)
    assert torch.isfinite(logit).all()
    assert torch.isfinite(features).all()


def test_cnn1d_baseline_accepts_unbatched_channel_input():
    model = CNNBaseline(_small_cnn1d_cfg())
    x = torch.randn(2, 400)
    logit, features = model(x)

    assert logit.shape == (2,)
    assert features.shape == (2, 32)


def test_cnn1d_baseline_variable_lengths():
    model = CNNBaseline(_small_cnn1d_cfg())
    model.eval()

    for seq_len in [400, 600, 1000]:
        x = torch.randn(2, 1, seq_len)
        with torch.no_grad():
            logit, features = model(x)

        assert logit.shape == (2,)
        assert features.shape == (2, 32)
        assert torch.isfinite(logit).all()
        assert torch.isfinite(features).all()


def test_cnn1d_baseline_backward_has_finite_gradient():
    model = CNNBaseline(_small_cnn1d_cfg())
    x = torch.randn(2, 1, 400)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    logit, _ = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(
        torch.isfinite(g).all() and torch.count_nonzero(g).item() > 0 for g in grads
    )


def test_build_model_routes_cnn1d_bilstm_and_ecgmamba_bilstm_backbone():
    from models import build_model
    from models.lstm_baseline import BiLSTMBaseline

    cnn = build_model(_small_cnn1d_cfg())
    assert isinstance(cnn, CNNBaseline)

    bilstm = build_model(
        {
            "model": {
                "name": "bilstm",
                "hidden_size": 8,
                "num_layers": 1,
                "bidirectional": True,
                "dropout": 0.0,
                "pooling": "mean",
            }
        }
    )
    assert isinstance(bilstm, BiLSTMBaseline)

    ecgmamba_bilstm = build_model(_small_ecgmamba_bilstm_cfg())
    assert isinstance(ecgmamba_bilstm, ECGMamba)
    assert not isinstance(ecgmamba_bilstm, BiLSTMBaseline)
    assert ecgmamba_bilstm.backbone_name == "bilstm"
