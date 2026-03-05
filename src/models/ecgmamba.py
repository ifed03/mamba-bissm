import torch
import torch.nn as nn

from .layers.conv_encoder import ConvEncoder
from .layers.mamba_block import MambaBlock
from .layers.pos_encoding import SinusoidalPositionalEncoding
from .mamba_backbone import BiMambaBackbone, MambaBackbone


class ECGMamba(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.d_model = mcfg["d_model"]
        self.use_encoder = mcfg.get("use_encoder", True)
        self.encoder = (
            ConvEncoder(1, self.d_model)
            if self.use_encoder
            else nn.Conv1d(1, self.d_model, kernel_size=3, stride=2, padding=1)
        )
        self.pos = SinusoidalPositionalEncoding(self.d_model)

        model_name = str(mcfg.get("name", "ecgmamba")).lower()
        backbone_cfg = mcfg.get("backbone")
        if backbone_cfg is not None:
            self.backbone_name = str(backbone_cfg).lower()
        elif model_name in {"mamba", "bimamba"}:
            self.backbone_name = model_name
        else:
            self.backbone_name = "bissm"
        if self.backbone_name not in {"bissm", "mamba", "bimamba"}:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Use one of: bissm, mamba, bimamba."
            )

        if self.backbone_name == "bissm":
            self.backbone = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=self.d_model,
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
                ]
            )
        else:
            mamba_kwargs = {
                "d_model": self.d_model,
                "n_layers": mcfg["n_layers"],
                "d_state": mcfg.get("d_state", mcfg.get("state_dim", 16)),
                "d_conv": mcfg.get("d_conv", mcfg.get("kernel_size", 4)),
                "expand": mcfg.get("expand", mcfg.get("expansion", 2)),
                "dropout": mcfg.get("dropout", 0.0),
                "norm": "layernorm" if mcfg.get("use_layernorm", True) else "none",
                "use_fast_path": mcfg.get("use_fast_path", True),
            }
            if self.backbone_name == "mamba":
                self.backbone = MambaBackbone(**mamba_kwargs)
            else:
                self.backbone = BiMambaBackbone(**mamba_kwargs)
        self.head = nn.Linear(self.d_model, 1)

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {tuple(x.shape)}")
        # Raw waveform path: (B, 1, T) -> (B, T', D)
        if x.size(1) == 1:
            return self.encoder(x).transpose(1, 2)
        # Already sequence-major features: (B, T, D)
        if x.size(-1) == self.d_model:
            return x
        # Channel-major features: (B, D, T) -> (B, T, D)
        if x.size(1) == self.d_model:
            return x.transpose(1, 2)
        raise ValueError(
            f"Could not adapt input shape {tuple(x.shape)} to (B, T, {self.d_model})."
        )

    def forward(self, x):
        x = self._to_sequence(x)
        x = self.pos(x)
        if self.backbone_name == "bissm":
            for blk in self.backbone:
                x = blk(x)
        else:
            x = self.backbone(x)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1), pooled
