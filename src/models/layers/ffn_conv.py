import torch.nn as nn

# refines features channel-wise
# feed-forward sublayer implemented with 1D convolutions instead of linear layers

class ConvFFN(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, kernel_size: int = 1, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, d_model, kernel_size=kernel_size, padding=pad),
        )

    def forward(self, x):
        # x: (B, L, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)
