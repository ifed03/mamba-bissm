# Data directory contract

This project treats `data/` as an explicit data boundary:

- `raw/`: immutable source exports (for example waveform parquet files). **Git-ignored**.
- `interim/`: temporary conversion artifacts. **Git-ignored**.
- `processed/`: training-ready derived datasets. **Git-ignored**.

The default config currently points to `data/cpsc_2018_labeled_single_lead.parquet` for convenience in this repository.
For larger/private datasets, place them under `data/raw/` and update `paths.data_path` in your config.

## Required parquet schema

- `record_id`: string
- `x`: list<float> (single lead waveform)
- `label`: int64 (`0` = normal, `1` = AF)
- `fs`: int64 source sampling rate

## Reproducibility note

Train/val/test membership is versioned under `splits/` and should remain tracked in Git.
