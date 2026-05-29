import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - exercised when dependency missing
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


def _make_norm(norm: str, d_model: int) -> nn.Module:
    norm_name = (norm or "layernorm").lower()
    if norm_name == "layernorm":
        return nn.LayerNorm(d_model)
    if norm_name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm '{norm}'. Use 'layernorm' or 'none'.")


def _make_mamba_layer(d_model: int, d_state: int, d_conv: int, expand: int, use_fast_path: bool) -> nn.Module:
    if Mamba is None:
        raise ImportError(
            "mamba-ssm is not installed. Install it with `pip install mamba-ssm`. "
            "For faster kernels you can also install `causal-conv1d`."
        ) from _MAMBA_IMPORT_ERROR
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=use_fast_path)


def _finite_activation(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class MambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm: str = "layernorm",
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _make_mamba_layer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=use_fast_path,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = _make_norm(norm, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        for block in self.layers:
            y = _finite_activation(block(x))
            x = _finite_activation(x + self.dropout(y))
        return _finite_activation(self.norm(x))


class BiMambaBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm: str = "layernorm",
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.fwd = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm=norm,
            use_fast_path=use_fast_path,
        )
        self.bwd = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm=norm,
            use_fast_path=use_fast_path,
        )
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.norm = _make_norm(norm, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y_f = self.fwd(x)
        x_rev = torch.flip(x, dims=[1])
        y_b = self.bwd(x_rev)
        y_b = torch.flip(y_b, dims=[1])
        y = torch.cat([y_f, y_b], dim=-1)
        y = _finite_activation(self.fuse(y))
        return _finite_activation(self.norm(y))
