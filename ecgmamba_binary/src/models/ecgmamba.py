import torch.nn as nn

from .layers.conv_encoder import ConvEncoder
from .layers.mamba_block import MambaBlock
from .layers.pos_encoding import SinusoidalPositionalEncoding


class ECGMamba(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        d_model = mcfg["d_model"]
        self.use_encoder = mcfg.get("use_encoder", True)
        self.encoder = ConvEncoder(1, d_model) if self.use_encoder else nn.Conv1d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                dropout=mcfg["dropout"],
                expansion=mcfg["expansion"],
                state_dim=mcfg["state_dim"],
                kernel_size=mcfg["kernel_size"],
                ffn_hidden_mult=mcfg["ffn_hidden_mult"],
                ffn_kernel_size=mcfg["ffn_kernel_size"],
                use_layernorm=mcfg.get("use_layernorm", True),
                use_ffn=mcfg.get("use_ffn", True),
            )
            for _ in range(mcfg["n_layers"])
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B,1,T)
        x = self.encoder(x).transpose(1, 2)  # (B,L,D)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1), pooled
