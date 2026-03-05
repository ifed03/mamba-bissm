import math

import torch
import torch.nn as nn

# adds explicity position information to each timstep
# Convolutions and SSMs process sequences, but this gives the model explicit awareness of timestep index.



# defines fixed sinusoidal position embeddings like original transformer
class SinusoidalPositionalEncoding(nn.Module):
    # d_model is feature width
    # max_len is max supported sequence length
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        # create table of positional encodings
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        # frequence scales for sinusoidal waves
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # even feature indices use sine values
        pe[:, 0::2] = torch.sin(pos * div)
        # odd feature indices use cosine values
        pe[:, 1::2] = torch.cos(pos * div)
        # Store  positional table inside module but not as trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    # expect input (B, L, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add positional encoding to first L positions to every item in batch
        return x + self.pe[:, : x.size(1)]
