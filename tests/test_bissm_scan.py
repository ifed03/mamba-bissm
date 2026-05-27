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


def test_bissm_uses_forward_and_backward_branches():
    torch.manual_seed(0)
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.randn(2, 12, 8)
    seen = {}

    def save_forward_input(_module, args):
        seen["forward"] = args[0].detach().clone()

    def save_backward_input(_module, args):
        seen["backward"] = args[0].detach().clone()

    hook_f = m.conv_f.register_forward_pre_hook(save_forward_input)
    hook_b = m.conv_b.register_forward_pre_hook(save_backward_input)
    try:
        y = m(x)
    finally:
        hook_f.remove()
        hook_b.remove()

    x_proj = m.x_proj(x).detach()
    assert y.shape == x.shape
    assert set(seen) == {"forward", "backward"}
    assert torch.allclose(seen["forward"], x_proj.transpose(1, 2))
    assert torch.allclose(seen["backward"], torch.flip(x_proj, dims=[1]).transpose(1, 2))


def test_bissm_backward_has_nonzero_parameter_gradient():
    torch.manual_seed(1)
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.randn(2, 16, 8, requires_grad=True)

    loss = m(x).square().mean()
    loss.backward()

    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)

