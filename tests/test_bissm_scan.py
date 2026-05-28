import torch

from models.layers.bissm import BiSSM


def _configure_identity_1x1_paths(m: BiSSM):
    assert m.ed == m.d_model
    with torch.no_grad():
        eye = torch.eye(m.d_model)
        m.x_proj.weight.copy_(eye)
        m.x_proj.bias.zero_()
        m.z_proj.weight.zero_()
        m.z_proj.bias.fill_(1.0)
        for conv in (m.conv_f, m.conv_b):
            conv.weight.zero_()
            conv.bias.zero_()
            idx = torch.arange(m.ed)
            conv.weight[idx, idx, 0] = 1.0
        m.out_proj.weight.copy_(eye)
        m.out_proj.bias.zero_()


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


def test_bissm_backward_ssm_scans_reversed_time_and_flips_output():
    m = BiSSM(d_model=2, expansion=1, state_dim=1, kernel_size=1)
    _configure_identity_1x1_paths(m)
    x = torch.arange(1, 11, dtype=torch.float32).view(1, 5, 2)
    scan_inputs = []

    def fake_scan(u, _B, _C, _delta):
        scan_inputs.append(u.detach().clone())
        if len(scan_inputs) == 1:
            return torch.zeros_like(u)
        steps = torch.arange(u.size(1), device=u.device, dtype=u.dtype).view(1, -1, 1)
        return steps.expand_as(u)

    original_scan = m._ssm_scan
    m._ssm_scan = fake_scan
    try:
        with torch.no_grad():
            y = m(x)
    finally:
        m._ssm_scan = original_scan

    expected_forward_scan_input = m.swish(x)
    expected_backward_scan_input = m.swish(torch.flip(x, dims=[1]))
    assert len(scan_inputs) == 2
    assert torch.allclose(scan_inputs[0], expected_forward_scan_input)
    assert torch.allclose(scan_inputs[1], expected_backward_scan_input)

    backward_scan_output = torch.arange(x.size(1), dtype=x.dtype).view(1, -1, 1).expand_as(x)
    gate = m.swish(torch.ones_like(x))
    expected_y = torch.flip(backward_scan_output, dims=[1]) * gate
    assert torch.allclose(y, expected_y)


def test_bissm_sums_forward_and_backward_branch_outputs():
    m = BiSSM(d_model=2, expansion=1, state_dim=1, kernel_size=1)
    _configure_identity_1x1_paths(m)
    x = torch.randn(1, 4, 2)
    scan_outputs = [
        torch.full((1, 4, 2), 2.0),
        torch.arange(8, dtype=torch.float32).view(1, 4, 2),
    ]
    calls = []

    def fake_scan(u, _B, _C, _delta):
        calls.append(u)
        return scan_outputs[len(calls) - 1].to(device=u.device, dtype=u.dtype)

    original_scan = m._ssm_scan
    m._ssm_scan = fake_scan
    try:
        with torch.no_grad():
            y = m(x)
    finally:
        m._ssm_scan = original_scan

    gate = m.swish(torch.ones_like(x))
    expected_y = (scan_outputs[0] + torch.flip(scan_outputs[1], dims=[1])) * gate
    assert len(calls) == 2
    assert torch.allclose(y, expected_y)


def test_bissm_backward_has_nonzero_parameter_gradient():
    torch.manual_seed(1)
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    x = torch.randn(2, 16, 8, requires_grad=True)

    loss = m(x).square().mean()
    loss.backward()

    def has_finite_nonzero_grad(prefixes):
        branch_grads = [
            p.grad for name, p in m.named_parameters() if name.startswith(prefixes)
        ]
        assert branch_grads
        for grad in branch_grads:
            assert grad is not None
            assert torch.isfinite(grad).all()
        return any(torch.count_nonzero(grad).item() > 0 for grad in branch_grads)

    assert has_finite_nonzero_grad(("conv_f.", "B_f.", "C_f.", "delta_f.", "delta_bias_f"))
    assert has_finite_nonzero_grad(("conv_b.", "B_b.", "C_b.", "delta_b.", "delta_bias_b"))


def test_bissm_parameter_count_regression():
    m = BiSSM(d_model=8, expansion=2, state_dim=4, kernel_size=3)
    assert sum(p.numel() for p in m.parameters()) == 2904
