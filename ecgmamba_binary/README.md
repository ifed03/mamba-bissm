# ECGMamba Binary Pipeline (Parquet, Single Lead)

End-to-end PyTorch pipeline that reproduces ECGMamba-style workflow for binary AF vs Normal classification from:
`data/cpsc_2018_labeled_single_lead.parquet`.

## Expected parquet schema
- `record_id`: string
- `x`: list<float> (single lead waveform)
- `label`: int64 (`0`=normal sinus rhythm, `1`=AF-present)
- `fs`: int64 source sampling rate (typically 500)

## Install
```bash
pip install -e .
```

## Create deterministic splits
```bash
python scripts/make_splits.py --config configs/binary_ecgmamba_100hz.yaml
```
Optional 10-fold:
```bash
python scripts/make_splits.py --config configs/binary_ecgmamba_100hz.yaml --kfold 10
```

## Train
```bash
python scripts/train.py --config configs/binary_ecgmamba_100hz.yaml
```

## Evaluate
```bash
python scripts/eval.py --config configs/binary_ecgmamba_100hz.yaml --ckpt runs/<run_name>/best.ckpt
```

## Sweep (baseline + ECGMamba + ablations via config toggles)
```bash
python scripts/sweep.py --configs configs/binary_cnn_baseline_100hz.yaml configs/binary_ecgmamba_100hz.yaml configs/binary_ecgmamba_500hz.yaml
```

## Preprocessing
1. list<float> -> float32 ndarray -> tensor
2. resample with `scipy.signal.resample_poly` to `fs_target` (default 100 Hz)
3. fixed-length 10s window (`target_len = fs_target * 10`)
   - train: random crop, val/test: center crop
   - zero right-pad when short
4. normalization after crop/pad (`none`, `zscore`, `robust`)

Output tensors are `(1, target_len)`.

## Metrics & thresholding
- AUROC
- F1/Accuracy computed with threshold selected on validation set by maximizing F1
- Same threshold applied to test set

## Outputs
Saved under `runs/<run_name>/`:
- `config.yaml`
- `best.ckpt`
- `metrics.json`
- `preds_val.npz`, `preds_test.npz`
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `score_hist.png`
