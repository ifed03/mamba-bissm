import torch

from models.layers.bissm import BiSSM


def test_bissm_output_shape():
    m = BiSSM(d_model=16, expansion=2, state_dim=8, kernel_size=4)
    x = torch.randn(2, 20, 16)
    y = m(x)
    assert y.shape == x.shape


def test_bissm_zero_input_stability():
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.zeros(1, 10, 8)
    y = m(x)
    assert torch.isfinite(y).all()
