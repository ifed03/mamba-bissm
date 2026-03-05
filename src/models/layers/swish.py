import torch
import torch.nn as nn

# Swish activation function is smooth alternative to ReLU
# Used inside BiSSM

# Inheriting from nn.Module means it can be used inside bigger models
class Swish(nn.Module):
    def forward(self, x):
        # If x very negative, output near 0 but not hard-clipped
        # If x positive, then almost unchanged
        return x * torch.sigmoid(x)
