# CNN1D baseline parameter-count preflight (2026-05-29)

This preflight compares the new standalone CNN1D external baseline against the existing standalone BiLSTM baseline and the controlled ECGMamba backbone variants before full training runs.

The main CNN1D configs were widened from `[32, 64, 128]` to `[64, 128, 256]` after the first pass because the narrower version had only 72,705 trainable parameters. The widened version remains simple, but is closer to the existing baselines for interpretation.

## Parameter counts

| Model/config family | Main config width/depth | Total parameters | Trainable parameters | Notes |
| --- | ---: | ---: | ---: | --- |
| standalone CNN1D | `cnn_channels: [64, 128, 256]` | 288,769 | 288,769 | External baseline; convolution-only temporal extractor. |
| standalone BiLSTM | hidden 128, 2 layers, bidirectional | 529,665 | 529,665 | Existing external baseline. |
| ECGMamba-BiSSM | `d_model: 64`, `n_layers: 2`, `state_dim: 64` | 551,425 | 551,425 | Controlled ECGMamba backbone comparison. |
| ECGMamba-Mamba | `d_model: 64`, `n_layers: 2`, `d_state: 16` | approx. 88,705 | approx. 88,705 | Estimated from the common `mamba-ssm` depthwise-conv parameterization; exact count should be confirmed in an environment with `mamba-ssm` installed. |
| ECGMamba-BiMamba | `d_model: 64`, `n_layers: 2`, `d_state: 16` | approx. 162,497 | approx. 162,497 | Estimated as two Mamba directions plus fusion/norm; exact count should be confirmed in an environment with `mamba-ssm` installed. |
| ECGMamba-BiLSTM-backbone | `d_model: 64`, hidden 32, 2 layers, bidirectional | 77,761 | 77,761 | Controlled ECGMamba backbone comparison, not the standalone BiLSTM baseline. |

## Interpretation

- The final CNN1D baseline is smaller than the standalone BiLSTM and ECGMamba-BiSSM models, but is no longer an order of magnitude smaller.
- It is larger than the ECGMamba-BiLSTM-backbone configuration because that controlled backbone uses hidden size 32 after the shared ECGMamba encoder.
- Mamba/BiMamba exact counts depend on the installed `mamba-ssm` implementation, so the table marks those values as approximate until the full training environment can instantiate those optional kernels.

## Reproducibility notes

- CNN1D count formula uses three Conv1d layers with bias, three BatchNorm1d layers, and one linear classifier head.
- BiLSTM counts follow PyTorch LSTM parameter shapes: `weight_ih`, `weight_hh`, `bias_ih`, and `bias_hh` per layer and direction.
- ECGMamba counts include the shared convolutional encoder and scalar classifier head.
