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
python scripts/train_model.py --config configs/bimamba.yaml
```

You can also pass a run name explicitly:

```bash
python scripts/train_model.py --config configs/binary_ecgmamba_100hz.yaml --run-name my_custom_run
```

Without `--run-name`, the pipeline creates a richer run name from config fields, e.g.:

```text
runs/ecgmamba_fs100_win10p0_seed42__20260224_165428/
```

## Evaluate

```bash
python scripts/evaluate_model.py --config configs/binary_ecgmamba_100hz.yaml --ckpt runs/<run_name>/checkpoints/best.ckpt
```

## Sweep (baseline + ECGMamba + ablations via config toggles)

```bash
python scripts/sweep.py --configs configs/binary_cnn_baseline_100hz.yaml configs/binary_ecgmamba_100hz.yaml configs/binary_ecgmamba_500hz.yaml
```

## Makefile shortcuts

```bash
make train CONFIG=configs/binary_ecgmamba_100hz.yaml
make eval CONFIG=configs/binary_ecgmamba_100hz.yaml CKPT=runs/<run_name>/checkpoints/best.ckpt
make splits CONFIG=configs/binary_ecgmamba_100hz.yaml
make test
```

## Preprocessing

1. `list<float>` -> float32 ndarray -> tensor
2. resample with `scipy.signal.resample_poly` to `fs_target` (default 100 Hz)
3. choose a segmentation strategy
   - default crop mode: fixed-length 10s crop (`target_len = fs_target * 10`)
     - train: random crop, val/test: center crop
     - zero right-pad when short
   - MIL window mode: split each record into non-overlapping 10s windows, zero-pad only the final remainder window, and inherit the parent record label for every segment
4. normalization after crop/pad (`none`, `zscore`, `robust`)

Output tensors are `(1, target_len)`.

## Metrics & thresholding

- AUROC
- F1/Accuracy/Sensitivity computed with threshold selected on validation set by maximizing F1
- In MIL mode, validation/test metrics are computed at the record level after max-pooling segment probabilities (`OR` logic)
- Same threshold applied to test set

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
- `preds_segments.parquet` with segment-level columns `record_id, segment_idx, y_true, y_prob, split` for test predictions when MIL mode is enabled
- `preds_val_segments.parquet` with segment-level columns `record_id, segment_idx, y_true, y_prob, split` for validation predictions when MIL mode is enabled

`runs/` is intentionally ignored by git.
