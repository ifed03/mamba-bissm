import torch.nn as nn

from .bissm import BiSSM
from .ffn_conv import ConvFFN


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 2, state_dim: int = 32, kernel_size: int = 4, ffn_hidden_mult: int = 4, ffn_kernel_size: int = 1, use_layernorm: bool = True, use_ffn: bool = True):
        super().__init__()
        self.ssm = BiSSM(d_model, expansion=expansion, state_dim=state_dim, kernel_size=kernel_size)
        self.ffn = ConvFFN(d_model, hidden_mult=ffn_hidden_mult, kernel_size=ffn_kernel_size, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.use_ln = use_layernorm
        self.use_ffn = use_ffn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = x + self.drop(self.ssm(x))
        if self.use_ln:
            y = self.ln1(y)
        if self.use_ffn:
            z = y + self.drop(self.ffn(y))
            if self.use_ln:
                z = self.ln2(z)
            return z
        return y
