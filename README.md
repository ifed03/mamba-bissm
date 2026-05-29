# ECGMamba Binary Pipeline (Parquet, Single Lead)

End-to-end PyTorch pipeline for binary AF vs Normal classification from parquet ECG waveforms.

## Repository layout

```text
configs/
data/
  README.md
  raw/          # gitignored
  interim/      # gitignored
  processed/    # gitignored
runs/           # gitignored
scripts/
splits/         # tracked in git
src/
tests/
```

## Expected parquet schema

- `record_id`: string
- `x`: list<float> (single lead waveform)
- `label`: int64 (`0`=normal sinus rhythm, `1`=AF-present)
- `fs`: int64 source sampling rate (typically 500)

## Install

```bash
pip install -e .
```

Optional speedups:

```bash
pip install -e ".[speed]"
```

## Create deterministic splits

Holdout split (tracked convention: `splits/holdout_seed42/split.json`):

```bash
python scripts/make_splits.py --config configs/binary_ecgmamba_100hz.yaml
```

Optional K-fold split set (tracked convention: `splits/kfold10_seed42/fold_*.json`):

```bash
python scripts/make_splits.py --config configs/binary_ecgmamba_100hz.yaml --kfold 10
```

## Train

```bash
python scripts/train_model.py --config configs/binary_ecgmamba_100hz.yaml
```

Train with official Mamba backbone variants:

```bash
python scripts/train_model.py --config configs/mamba.yaml
python scripts/train_model.py --config configs/bimamba_d128_n8_s16_fastpath_amp_100hz_win10s.yaml
```

You can also pass a run name explicitly:

```bash
python scripts/train_model.py --config configs/binary_ecgmamba_100hz.yaml --run-name my_custom_run
```

Without `--run-name`, the pipeline creates a richer run name from config fields, e.g.:

```text
runs/bissm_d64_n2_s64_mil_fs100_win10p0_seed42__20260224_165428/
```

## Evaluate

```bash
python scripts/evaluate_model.py --config configs/binary_ecgmamba_100hz.yaml --ckpt runs/<run_name>/checkpoints/best.ckpt
```

## Zero-shot noise robustness sweep

The zero-shot noisy-test protocol reuses clean trained checkpoints and applies
NSTDB noise only to the test split. Thresholds remain sourced from clean
validation (`clean_val`); noisy test data is never used to tune thresholds.
Supported raw NSTDB noise labels are `bw`, `em`, and `ma`.

Dry-run every successful model and print the full evaluation matrix:

```bash
python scripts/sweep_zero_shot_noise.py --successful-runs successful_runs --dry-run
```

Run the full robustness sweep at 18, 6, 0, and -6 dB:

```bash
python scripts/sweep_zero_shot_noise.py \
  --successful-runs successful_runs \
  --noise-types bw em ma \
  --snrs 18 6 0 -6
```

By default, per-condition JSON files are written under:

```text
outputs/zero_shot_noise_sweep/<model_family>/<source_run_name>/zero-shot/
```

Consolidated summaries are written to:

```text
outputs/zero_shot_noise_sweep/zero_shot_noise_summary.csv
outputs/zero_shot_noise_sweep/zero_shot_noise_summary.json
```


## Noisy-input training/evaluation

The noisy-input protocol trains, validates, and tests with deterministic NSTDB
noise applied to every split. Noise is injected dynamically after ECG resampling
to 100 Hz and before window extraction and per-window normalization, preserving
the clean parquet dataset and the original train/validation/test membership.
The noisy validation split is used for both checkpoint selection and validation
threshold (`tau*`) selection; that noisy-validation `tau*` is then applied to
the matching noisy test condition. Supported NSTDB noise labels are `bw`, `em`,
and `ma`; the default SNR list is `24 18 12 6 0 -6`.

Dry-run the noisy-input matrix without writing outputs:

```bash
python scripts/train_model.py \
  --config configs/binary_ecgmamba_100hz.yaml \
  --noise-training-mode noisy-input \
  --noise-root data \
  --noise-types bw em ma \
  --snr-db 24 18 12 6 0 -6 \
  --base-seed 123 \
  --ecg-fs 100 \
  --output-root runs/noisy_input_training \
  --dry-run
```

Run one noisy-input condition:

```bash
python scripts/train_model.py \
  --config configs/binary_ecgmamba_100hz.yaml \
  --noise-training-mode noisy-input \
  --noise-root data \
  --noise-types bw \
  --snr-db 18 \
  --base-seed 123 \
  --ecg-fs 100 \
  --output-root runs/noisy_input_training
```

Per-condition run directories are named like
`noisy_input_training_<noise_type>_<snr>dB` and contain unambiguous files such
as `metrics_noisy-input-training_noise_type=bw__snr_db=18.json` and
`threshold_noisy-input-training_noise_type=bw__snr_db=18.json`.

## Sweep (baseline + ECGMamba + ablations via config toggles)

```bash
python scripts/sweep.py --configs configs/binary_cnn_baseline_100hz.yaml configs/binary_ecgmamba_100hz.yaml configs/binary_ecgmamba_500hz.yaml
```

Run the clean full-matrix preflight audit first. It checks controlled ECGMamba
parameter counts, intermediate tensor shapes, and saves the dry-run command
listing under `audits/`:

```bash
python scripts/audit_clean_full_matrix.py --batch-tag preflight_YYYYMMDD
```

Run the full clean-data 32-run AF/NSR matrix (controlled ECGMamba backbones,
depth sweeps, and standalone BiLSTM/CNN1D external baselines across 4/6/8/10s):

```bash
python scripts/run_clean_full_matrix.py
```


Specific sweep for:
- 2-layer BiSSM, state_dim 32 across 4/6/8/10s windows
- full-size ECGMamba BiSSM across 4/6/8/10s windows
- 4-layer BiSSM, state_dim 64 at 10s only

Direct command:

```bash
python scripts/sweep.py --results runs/sweeps/window_ablation_ecgmamba_mamba_bissm.csv --configs \
  configs/binary_bissm_d64_n2_s32_100hz_win4s_stride2s.yaml \
  configs/binary_bissm_d64_n2_s32_100hz_win6s_stride2s.yaml \
  configs/binary_bissm_d64_n2_s32_100hz_win8s_stride2s.yaml \
  configs/binary_bissm_d64_n2_s32_100hz_win10s_stride2s.yaml \
  configs/binary_ecgmamba_100hz_win4s_stride2s.yaml \
  configs/binary_ecgmamba_100hz_win6s_stride2s.yaml \
  configs/binary_ecgmamba_100hz_win8s_stride2s.yaml \
  configs/binary_ecgmamba_100hz_win10s_stride2s.yaml \
  configs/binary_bissm_d64_n4_s64_100hz_win10s_stride2s.yaml
```

## Preprocessing

1. `list<float>` -> float32 ndarray -> tensor
2. resample with `scipy.signal.resample_poly` to `fs_target` (default 100 Hz)
3. choose a segmentation strategy
   - default crop mode: fixed-length 10s crop (`target_len = fs_target * 10`)
     - train: random crop, val/test: center crop
     - zero right-pad when short
   - window mode: split each record into fixed-length windows and inherit the parent record label for every window
     - the primary clean-data protocol uses 4, 6, 8 and 10s windows with 2s stride
     - final incomplete windows are dropped, except when an entire record is shorter than the requested window length, in which case one window is retained and padded after normalization
4. normalization is applied before zero-padding in window mode (`none`, `zscore`, `robust`)

Output tensors are `(1, target_len)`.

## Metrics & thresholding

- AUROC
- F1/Accuracy/Sensitivity computed with threshold selected on validation set by maximizing F1
- The implementation uses independent window-level BCE training with inherited record labels. During validation and testing, window probabilities are grouped by record and max-pooled to obtain a record-level prediction. This is MIL-like at evaluation time but is not a full differentiable bag-level MIL objective.
- Same threshold applied to test set

## Optimizer & scheduler

- Training uses AdamW with linear warm-up followed by cosine decay.
- In mixed-precision training, the scheduler is stepped only after a successful optimizer update, so it is not advanced when GradScaler skips an update due to non-finite gradients.

## Splits

- Holdout and K-fold splits are group-aware.
- `split.group_id_col` controls the leakage boundary.
- If that column is absent in parquet, the pipeline falls back to `record_id`.

## Run artifact contract

Each run directory should contain:

- `config_resolved.yaml` (fully resolved config used for the run)
- `metrics.json` (machine-readable summary)
- `plots/` (ROC, PR, confusion matrix, score histogram)
- `checkpoints/best.ckpt`
- `preds.parquet` with record-level columns `record_id, y_true, y_prob, split` for test predictions
- `preds_val.parquet` with record-level columns `record_id, y_true, y_prob, split` for validation predictions
- `preds_segments.parquet` with window-level columns `record_id, segment_idx, y_true, y_prob, split` for test predictions when window mode is enabled
- `preds_val_segments.parquet` with window-level columns `record_id, segment_idx, y_true, y_prob, split` for validation predictions when window mode is enabled

`runs/` is intentionally ignored by git.
