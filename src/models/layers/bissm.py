import torch
import torch.nn as nn
import torch.nn.functional as F

from .swish import Swish

# core custom sequence layer - bidirectional state-space modeling
# processes sequence forward in time and backward in time then combines both

# local conv first
# then long-range sequential state update
# in both directions
# then gating

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
        # project input into expanded feature space
        self.x_proj = nn.Linear(d_model, self.ed)
        # gating
        self.z_proj = nn.Linear(d_model, self.ed)
        # forward-direction convolution preprocessing
        self.conv_f = nn.Conv1d(self.ed, self.ed, kernel_size=kernel_size, padding=kernel_size - 1, groups=1)
        # backward-direction convolution preprocessing
        self.conv_b = nn.Conv1d(self.ed, self.ed, kernel_size=kernel_size, padding=kernel_size - 1, groups=1)
        # activation after convolutions and for later gating
        self.swish = Swish()

        # generate forward/backward B, C, foward step sizes (delta), add trainable delta bias
        self.B_f = nn.Linear(self.ed, self.n)
        self.C_f = nn.Linear(self.ed, self.n)
        self.delta_f = nn.Linear(self.ed, self.ed)
        self.delta_bias_f = nn.Parameter(torch.zeros(self.ed))

        self.B_b = nn.Linear(self.ed, self.n)
        self.C_b = nn.Linear(self.ed, self.n)
        self.delta_b = nn.Linear(self.ed, self.ed)
        self.delta_bias_b = nn.Parameter(torch.zeros(self.ed))

        # Learned base SSM dynamics matrix, before constraining it
        self.A_raw = nn.Parameter(torch.randn(self.ed, self.n) * 0.02)
        # Projects internal expanded features back to model width
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
            # Builds a discrete-time transition factor from continuous dynamics
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
        xb_rev = self.swish(self._causal_trim(self.conv_b(xrev.transpose(1, 2)), L).transpose(1, 2))

        Bf, Cf = self.B_f(xf), self.C_f(xf)
        Df = F.softplus(self.delta_f(xf) + self.delta_bias_f)
        Bb_rev, Cb_rev = self.B_b(xb_rev), self.C_b(xb_rev)
        Db_rev = F.softplus(self.delta_b(xb_rev) + self.delta_bias_b)

        yf = self._ssm_scan(xf, Bf, Cf, Df)
        yb_rev = self._ssm_scan(xb_rev, Bb_rev, Cb_rev, Db_rev)
        yb = torch.flip(yb_rev, dims=[1])

        y = (yf + yb) * self.swish(z_proj)
        return self.out_proj(y)
