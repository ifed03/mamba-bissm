import torch

from models.layers.bissm import BiSSM
from models.layers.conv_encoder import ConvEncoder
from models.layers.mamba_block import MambaBlock


def _reference_ssm_scan(module: BiSSM, u: torch.Tensor, B: torch.Tensor, C: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    # Independent implementation of the recurrence for regression checking.
    bsz, L, ED = u.shape
    A = -torch.nn.functional.softplus(module.A_raw)
    h = torch.zeros(bsz, ED, module.n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        ut = u[:, t, :]
        Bt = B[:, t, :].unsqueeze(1).expand(-1, ED, -1)
        Ct = C[:, t, :].unsqueeze(1).expand(-1, ED, -1)
        delt = delta[:, t, :].unsqueeze(-1)
        Abar = torch.exp(delt * A.unsqueeze(0))
        Bbar = ((Abar - 1.0) / (A.unsqueeze(0) + module.eps)) * Bt
        h = Abar * h + Bbar * ut.unsqueeze(-1)
        ys.append((Ct * h).sum(dim=-1))
    return torch.stack(ys, dim=1)


def test_conv_encoder_stage_shape():
    enc = ConvEncoder(in_ch=1, d_model=128)
    x = torch.randn(2, 1, 1000)
    y = enc(x)
    assert y.shape == (2, 128, 250)


def test_mamba_block_residual_identity_when_branches_zero():
    block = MambaBlock(
        d_model=16,
        dropout=0.0,
        expansion=2,
        state_dim=8,
        kernel_size=3,
        ffn_hidden_mult=2,
        ffn_kernel_size=1,
        use_layernorm=False,
        use_ffn=True,
    )
    for p in block.ssm.parameters():
        p.data.zero_()
    for p in block.ffn.parameters():
        p.data.zero_()

    x = torch.randn(2, 12, 16)
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_bissm_scan_matches_reference():
    torch.manual_seed(0)
    m = BiSSM(d_model=4, expansion=2, state_dim=3, kernel_size=3)
    bsz, L, ED, N = 2, 5, m.ed, m.n

    u = torch.randn(bsz, L, ED)
    B = torch.randn(bsz, L, N)
    C = torch.randn(bsz, L, N)
    delta = torch.nn.functional.softplus(torch.randn(bsz, L, ED))

    y_impl = m._ssm_scan(u, B, C, delta)
    y_ref = _reference_ssm_scan(m, u, B, C, delta)
    assert torch.allclose(y_impl, y_ref, atol=1e-6, rtol=1e-5)


def test_bissm_backward_is_finite():
    torch.manual_seed(1)
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.randn(2, 20, 8, requires_grad=True)

    y = m(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert m.A_raw.grad is not None
    assert torch.isfinite(m.A_raw.grad).all()


def test_bissm_large_input_numerical_stability():
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.randn(2, 30, 8) * 100.0
    y = m(x)
    assert torch.isfinite(y).all()
