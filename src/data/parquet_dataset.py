import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from .transforms import ECGPreprocessor, TransformConfig


class RecordBatchSampler(Sampler[list[int]]):
    def __init__(self, record_batches: list[list[int]]):
        self.record_batches = record_batches

    def __iter__(self):
        yield from self.record_batches

    def __len__(self):
        return len(self.record_batches)


class ParquetECGDataset(Dataset):
    def __init__(self, path: str, indices: list[int] | None = None, train: bool = True, preprocess_cfg: dict | None = None):
        preprocess_cfg = dict(preprocess_cfg or {})
        windowing_cfg = dict(preprocess_cfg.pop("windowing", {}))
        expected_fs = preprocess_cfg.pop("fs_source_expected", None)

        tcfg = TransformConfig(**preprocess_cfg, random_crop=train)
        self.preprocessor = ECGPreprocessor(tcfg)
        self.windowing_enabled = bool(windowing_cfg.get("enabled", False))
        self.window_seconds = float(windowing_cfg.get("window_seconds", tcfg.target_seconds))
        self.pad_remainder = bool(windowing_cfg.get("pad_remainder", True))

        if self.windowing_enabled and not math.isclose(self.window_seconds, tcfg.target_seconds):
            raise ValueError("Windowed MIL expects preprocessing.target_seconds to match windowing.window_seconds")

        table = pq.read_table(Path(path), columns=["record_id", "x", "label", "fs"])
        all_record_ids = table["record_id"].to_pylist()
        all_xs = table["x"].to_pylist()
        all_labels = table["label"].to_pylist()
        all_fs = table["fs"].to_pylist()

        self.indices = indices if indices is not None else list(range(len(all_labels)))
        if expected_fs is not None and any(int(all_fs[i]) != int(expected_fs) for i in self.indices):
            raise ValueError(f"Found fs values not matching expected {expected_fs}")

        self.record_ids = []
        self.labels = []
        self.fs = []
        self._signals = []
        self._raw_signals = []
        self._samples = []
        self.sample_record_ids = []
        self.sample_labels = []
        self.record_num_segments = []
        self.record_batches = []

        by_record = defaultdict(list)
        window_len = self.preprocessor.target_length

        for dataset_idx, row_idx in enumerate(self.indices):
            record_id = all_record_ids[row_idx]
            label = int(all_labels[row_idx])
            fs = int(all_fs[row_idx])

            self.record_ids.append(record_id)
            self.labels.append(label)
            self.fs.append(fs)

            if self.windowing_enabled:
                signal = self.preprocessor.prepare_signal(all_xs[row_idx], fs)
                self._signals.append(signal)
                num_segments = max(1, math.ceil(len(signal) / window_len)) if self.pad_remainder else max(1, len(signal) // window_len)
                self.record_num_segments.append(num_segments)
                for segment_idx in range(num_segments):
                    sample_idx = len(self._samples)
                    self._samples.append((dataset_idx, segment_idx))
                    self.sample_record_ids.append(record_id)
                    self.sample_labels.append(label)
                    by_record[record_id].append(sample_idx)
            else:
                self._raw_signals.append(all_xs[row_idx])
                self.record_num_segments.append(1)
                sample_idx = len(self._samples)
                self._samples.append((dataset_idx, 0))
                self.sample_record_ids.append(record_id)
                self.sample_labels.append(label)
                by_record[record_id].append(sample_idx)

        self.record_batches = list(by_record.values())

    def __len__(self):
        return len(self._samples)

    def _segment_signal(self, dataset_idx: int, segment_idx: int):
        signal = self._signals[dataset_idx]
        window_len = self.preprocessor.target_length
        start = segment_idx * window_len
        stop = start + window_len
        segment = signal[start:stop]
        if len(segment) == 0:
            segment = signal[:window_len]
        return segment

    def __getitem__(self, idx: int):
        dataset_idx, segment_idx = self._samples[idx]
        label = self.labels[dataset_idx]

        if self.windowing_enabled:
            x = self.preprocessor.format_segment(self._segment_signal(dataset_idx, segment_idx))
            num_segments = self.record_num_segments[dataset_idx]
        else:
            x = self.preprocessor(self._raw_signals[dataset_idx], int(self.fs[dataset_idx]))
            num_segments = 1

        y = torch.tensor(float(label), dtype=torch.float32)
        return {
            "x": x,
            "y": y,
            "record_id": self.record_ids[dataset_idx],
            "segment_idx": torch.tensor(segment_idx, dtype=torch.int64),
            "num_segments": torch.tensor(num_segments, dtype=torch.int64),
        }
