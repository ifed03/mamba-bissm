import torch.nn as nn

from .bissm import BiSSM
from .ffn_conv import ConvFFN


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 2, state_dim: int = 32, kernel_size: int = 4, ffn_hidden_mult: int = 4, ffn_kernel_size: int = 1, use_layernorm: bool = True, use_ffn: bool = True):
        super().__init__()
        # create main sequence mixing layer
        self.ssm = BiSSM(d_model, expansion=expansion, state_dim=state_dim, kernel_size=kernel_size)
        # create feed-forward sublayer
        self.ffn = ConvFFN(d_model, hidden_mult=ffn_hidden_mult, kernel_size=ffn_kernel_size, dropout=dropout)
        # dropout used after sublyaers
        self.drop = nn.Dropout(dropout)
        # store whether to apply LayerNorm or include FFN stage
        self.use_ln = use_layernorm
        self.use_ffn = use_ffn
        # Norms after SSM and FFN residual branches
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    # input shape (B, L, D)
    def forward(self, x):
        # output = input + transformed input 
        y = x + self.drop(self.ssm(x))
        if self.use_ln:
            y = self.ln1(y)
        if self.use_ffn:
            # residual connection around the FFN
            z = y + self.drop(self.ffn(y))
            if self.use_ln:
                z = self.ln2(z)
            return z
        return y
