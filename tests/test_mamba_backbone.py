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


def test_bimamba_backbone_shape_and_backward():
    pytest.importorskip("mamba_ssm")
    from models.mamba_backbone import BiMambaBackbone

    model = BiMambaBackbone(d_model=64, n_layers=2).to("cuda")
    x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
