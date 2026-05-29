import torch
import torch.nn as nn

from .layers.conv_encoder import ConvEncoder
from .layers.mamba_block import MambaBlock
from .layers.pos_encoding import SinusoidalPositionalEncoding
from .mamba_backbone import BiMambaBackbone, MambaBackbone


class BiLSTMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        layernorm: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_size = (
            int(hidden_size) if hidden_size is not None else max(1, self.d_model // 2)
        )
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=lstm_dropout,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.proj = nn.Linear(out_dim, self.d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model) if layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, D)
        y, _ = self.lstm(x)
        y = self.proj(y)
        y = self.dropout(y)
        return self.norm(y)


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
        if self.backbone_name not in {"bissm", "mamba", "bimamba", "bilstm"}:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}'. Use one of: bissm, mamba, bimamba, bilstm."
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
        elif self.backbone_name == "bilstm":
            self.backbone = BiLSTMBackbone(
                d_model=self.d_model,
                hidden_size=mcfg.get("lstm_hidden_size"),
                num_layers=mcfg.get("lstm_num_layers", mcfg.get("n_layers", 1)),
                dropout=mcfg.get("lstm_dropout", mcfg.get("dropout", 0.0)),
                bidirectional=mcfg.get("lstm_bidirectional", True),
                layernorm=mcfg.get("lstm_layernorm", mcfg.get("use_layernorm", True)),
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
