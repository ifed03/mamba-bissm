import torch

import pytest


def test_mamba_backbone_shape_and_backward():
    pytest.importorskip("mamba_ssm")
    from models.mamba_backbone import MambaBackbone

    model = MambaBackbone(d_model=64, n_layers=2).to("cuda")
    x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm fast path requires CUDA")
def test_bimamba_backbone_cuda_fast_path_forward_backward():
    pytest.importorskip("mamba_ssm")
    from models.mamba_backbone import BiMambaBackbone

    model = BiMambaBackbone(
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_fast_path=True,
    ).cuda()
    x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
    out = model(x)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm fast path requires CUDA")
def test_bimamba_build_model_forward():
    pytest.importorskip("mamba_ssm")
    from models import build_model

    cfg = {
        "model": {
            "name": "bimamba",
            "backbone": "bimamba",
            "d_model": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "d_state": 4,
            "d_conv": 3,
            "expand": 2,
            "use_fast_path": True,
            "use_encoder": True,
            "use_layernorm": True,
        }
    }
    model = build_model(cfg).to("cuda")
    model.eval()
    x = torch.randn(2, 1, 400, device="cuda")

    with torch.no_grad():
        logits, pooled = model(x)

    assert logits.shape == (2,)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(pooled).all()
