import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """Standalone 1D-CNN baseline for raw ECG classification."""

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg.get("model", {})
        self.in_channels = int(mcfg.get("in_channels", 1))
        channels = mcfg.get("cnn_channels", mcfg.get("channels", [32, 64, 128]))
        if isinstance(channels, int):
            channels = [channels]
        self.channels = [int(c) for c in channels]
        if not self.channels:
            raise ValueError("CNNBaseline requires at least one output channel.")
        if any(c <= 0 for c in self.channels):
            raise ValueError("CNNBaseline cnn_channels must contain positive integers.")

        kernel_size = int(mcfg.get("cnn_kernel_size", 7))
        if kernel_size <= 0:
            raise ValueError("CNNBaseline cnn_kernel_size must be positive.")
        stride = int(mcfg.get("cnn_stride", 2))
        if stride <= 0:
            raise ValueError("CNNBaseline cnn_stride must be positive.")
        dropout = float(mcfg.get("cnn_dropout", mcfg.get("dropout", 0.1)))
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("CNNBaseline dropout must satisfy 0 <= dropout < 1.")
        use_batchnorm = bool(mcfg.get("cnn_batchnorm", True))
        padding = int(mcfg.get("cnn_padding", kernel_size // 2))

        layers: list[nn.Module] = []
        in_ch = self.in_channels
        for out_ch in self.channels:
            layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(self.channels[-1], 1)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, C, T) or (B, T), got {tuple(x.shape)}."
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), got shape {tuple(x.shape)}."
            )

        sequence = self.net(x)
        features = self.pool(sequence).squeeze(-1)
        logits = self.head(self.head_dropout(features))
        return logits.squeeze(-1), features


CNN1DBaseline = CNNBaseline
