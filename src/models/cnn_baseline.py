import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        chs = cfg["model"].get("channels", [32, 64, 128])
        layers = []
        in_ch = 1
        for i, c in enumerate(chs):
            layers.extend([
                nn.Conv1d(in_ch, c, kernel_size=7 if i == 0 else 5, stride=2 if i < 2 else 1, padding=3 if i == 0 else 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
            ])
            in_ch = c
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(chs[-1], 1)

    def forward(self, x):
        feat = self.net(x)
        pooled = feat.mean(dim=-1)
        return self.head(pooled).squeeze(-1), pooled
