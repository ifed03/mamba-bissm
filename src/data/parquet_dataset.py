from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .transforms import ECGPreprocessor, TransformConfig


class ParquetECGDataset(Dataset):
    def __init__(self, path: str, indices: list[int] | None = None, train: bool = True, preprocess_cfg: dict | None = None):
        table = pq.read_table(Path(path), columns=["record_id", "x", "label", "fs"])
        self.record_ids = table["record_id"].to_pylist()
        self.xs = table["x"].to_pylist()
        self.labels = table["label"].to_pylist()
        self.fs = table["fs"].to_pylist()
        self.indices = indices if indices is not None else list(range(len(self.labels)))

        preprocess_cfg = dict(preprocess_cfg or {})
        expected_fs = preprocess_cfg.pop("fs_source_expected", None)
        if expected_fs is not None and any(int(v) != int(expected_fs) for v in self.fs):
            raise ValueError(f"Found fs values not matching expected {expected_fs}")

        tcfg = TransformConfig(**preprocess_cfg, random_crop=train)
        self.preprocessor = ECGPreprocessor(tcfg)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.preprocessor(self.xs[i], int(self.fs[i]))
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return {"x": x, "y": y, "record_id": self.record_ids[i]}
