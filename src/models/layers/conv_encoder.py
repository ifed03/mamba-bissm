import torch.nn as nn

# Convert raw ECG signals into learned features

# defines convolutional feature extractor
class ConvEncoder(nn.Module):
    # 1 channel since single-lead ECG input
    # d_model is output feature width per timestep
    def __init__(self, in_ch: int = 1, d_model: int = 128):
        super().__init__()
        # creates pipeline of layers exectuted in order
        self.net = nn.Sequential(
            # input channels, output channels
            # large kernel captures local temporal patterns
            # stride roughly halves sequence length 
            # padding preserves alignment
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            # normalise channels to stabilise training
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    # input shape (B, C, T)
    def forward(self, x):
        return self.net(x)
