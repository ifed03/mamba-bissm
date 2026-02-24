import torch
import torch.nn as nn
import torch.nn.functional as F

from .swish import Swish


class BiSSM(nn.Module):
    """Bi-directional selective SSM with per-channel state size N.

    Shapes:
    - input x: (B, L, D)
    - expanded ED = E*D
    - state h: (B, ED, N)
    """

    def __init__(self, d_model: int, expansion: int = 2, state_dim: int = 32, kernel_size: int = 4, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.ed = d_model * expansion
        self.n = state_dim
        self.eps = eps

        self.x_proj = nn.Linear(d_model, self.ed)
        self.z_proj = nn.Linear(d_model, self.ed)
        self.conv_f = nn.Conv1d(self.ed, self.ed, kernel_size=kernel_size, padding=kernel_size - 1, groups=1)
        self.conv_b = nn.Conv1d(self.ed, self.ed, kernel_size=kernel_size, padding=kernel_size - 1, groups=1)
        self.swish = Swish()

        self.B_f = nn.Linear(self.ed, self.n)
        self.C_f = nn.Linear(self.ed, self.n)
        self.delta_f = nn.Linear(self.ed, self.ed)
        self.delta_bias_f = nn.Parameter(torch.zeros(self.ed))

        self.B_b = nn.Linear(self.ed, self.n)
        self.C_b = nn.Linear(self.ed, self.n)
        self.delta_b = nn.Linear(self.ed, self.ed)
        self.delta_bias_b = nn.Parameter(torch.zeros(self.ed))

        self.A_raw = nn.Parameter(torch.randn(self.ed, self.n) * 0.02)
        self.out_proj = nn.Linear(self.ed, d_model)

    def _causal_trim(self, x, L):
        return x[..., :L]

    def _ssm_scan(self, u, B, C, delta):
        # u: (B,L,ED), B/C: (B,L,N), delta: (B,L,ED)
        bsz, L, ED = u.shape
        A = -F.softplus(self.A_raw)  # (ED,N)
        h = torch.zeros(bsz, ED, self.n, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            ut = u[:, t, :]  # (B,ED)
            Bt = B[:, t, :].unsqueeze(1).expand(-1, ED, -1)
            Ct = C[:, t, :].unsqueeze(1).expand(-1, ED, -1)
            delt = delta[:, t, :].unsqueeze(-1)
            Abar = torch.exp(delt * A.unsqueeze(0))
            Bbar = ((Abar - 1.0) / (A.unsqueeze(0) + self.eps)) * Bt
            h = Abar * h + Bbar * ut.unsqueeze(-1)
            y = (Ct * h).sum(dim=-1)
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x):
        B, L, _ = x.shape
        x_proj = self.x_proj(x)
        z_proj = self.z_proj(x)

        xf = self.swish(self._causal_trim(self.conv_f(x_proj.transpose(1, 2)), L).transpose(1, 2))
        xrev = torch.flip(x_proj, dims=[1])
        xb = self.swish(self._causal_trim(self.conv_b(xrev.transpose(1, 2)), L).transpose(1, 2))
        xb = torch.flip(xb, dims=[1])

        Bf, Cf = self.B_f(xf), self.C_f(xf)
        Df = F.softplus(self.delta_f(xf) + self.delta_bias_f)
        Bb, Cb = self.B_b(xb), self.C_b(xb)
        Db = F.softplus(self.delta_b(xb) + self.delta_bias_b)

        yf = self._ssm_scan(xf, Bf, Cf, Df)
        yb = self._ssm_scan(xb, Bb, Cb, Db)

        y = (yf + yb) * self.swish(z_proj)
        return self.out_proj(y)
