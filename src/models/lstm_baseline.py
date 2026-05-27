import torch
import torch.nn as nn


class BiLSTMBaseline(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        mcfg = cfg.get("model", {})
        self.hidden_size = int(mcfg.get("hidden_size", 128))
        self.num_layers = int(mcfg.get("num_layers", 2))
        self.bidirectional = bool(mcfg.get("bidirectional", True))
        self.pooling = str(mcfg.get("pooling", "mean")).lower()
        if self.pooling != "mean":
            raise ValueError("BiLSTMBaseline supports only pooling='mean'.")

        dropout = float(mcfg.get("dropout", 0.2))
        lstm_dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=lstm_dropout,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        features = lstm_out.mean(dim=1)
        logits = self.head(self.dropout(features))
        return logits.squeeze(-1), features
