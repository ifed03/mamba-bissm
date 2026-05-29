import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from .transforms import ECGPreprocessor, TransformConfig
from evaluate.noise_protocol import NoisyInputNoiseInjector, ZeroShotNoiseInjector, ensure_clean_split


class RecordBatchSampler(Sampler[list[int]]):
    def __init__(self, record_batches: list[list[int]]):
        self.record_batches = record_batches

    def __iter__(self):
        yield from self.record_batches

    def _warn_if_snr_deviates(self, meta: dict) -> None:
        requested = float(meta.get("snr_db", 0.0))
        measured = float(meta.get("measured_snr_db", requested))
        if abs(measured - requested) > self.noise_snr_tolerance_db:
            print(
                "WARNING: measured SNR deviates from requested SNR: "
                f"record_id={meta.get('record_id')}, split={meta.get('split')}, "
                f"requested={requested:g} dB, measured={measured:g} dB, "
                f"tolerance={self.noise_snr_tolerance_db:g} dB"
            )

    def __len__(self):
        return len(self.record_batches)


class ParquetECGDataset(Dataset):
    def __init__(
        self,
        path: str,
        indices: list[int] | None = None,
        train: bool = True,
        preprocess_cfg: dict | None = None,
        *,
        split_name: str | None = None,
        noise_cfg: dict | None = None,
    ):
        preprocess_cfg = dict(preprocess_cfg or {})
        windowing_cfg = dict(preprocess_cfg.pop("windowing", {}))
        expected_fs = preprocess_cfg.pop("fs_source_expected", None)

        tcfg = TransformConfig(**preprocess_cfg, random_crop=train)
        self.preprocessor = ECGPreprocessor(tcfg)
        self.windowing_enabled = bool(windowing_cfg.get("enabled", False))
        self.window_seconds = float(windowing_cfg.get("window_seconds", tcfg.target_seconds))
        self.stride_seconds = float(windowing_cfg.get("stride_seconds", self.window_seconds))
        self.pad_remainder = bool(windowing_cfg.get("pad_remainder", True))

        if self.windowing_enabled and not math.isclose(self.window_seconds, tcfg.target_seconds):
            raise ValueError("Windowed preprocessing expects preprocessing.target_seconds to match windowing.window_seconds")

        self.split_name = split_name
        self.noise_metadata = []
        self.noise_injector = None
        noise_cfg = dict(noise_cfg or {})
        self.noise_snr_tolerance_db = float(noise_cfg.get("snr_tolerance_db", 0.25)) if noise_cfg else 0.25
        if noise_cfg.get("enabled", False):
            noise_mode = noise_cfg.get("mode", "zero-shot")
            if noise_mode == "zero-shot":
                ensure_clean_split(split_name or "")
                injector_cls = ZeroShotNoiseInjector
            elif noise_mode == "noisy-input":
                injector_cls = NoisyInputNoiseInjector
            else:
                raise ValueError(f"Unknown noise injection mode {noise_mode!r}; expected 'zero-shot' or 'noisy-input'.")
            self.noise_injector = injector_cls(
                noise_type=noise_cfg["noise_type"],
                snr_db=noise_cfg["snr_db"],
                base_seed=noise_cfg.get("base_seed", 123),
                target_fs=noise_cfg.get("target_fs", tcfg.fs_target),
                noise_root=noise_cfg.get("noise_root", "data"),
                noise=noise_cfg.get("noise"),
                noise_fs=noise_cfg.get("noise_fs"),
            )

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
        self._window_starts = []
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
                if self.noise_injector is not None:
                    signal, meta = self.noise_injector.inject(signal, record_id=record_id, split=self.split_name or "test")
                    meta["original_label"] = label
                    meta["processing_order"] = "resample->noise->window->normalize"
                    self._warn_if_snr_deviates(meta)
                    self.noise_metadata.append(meta)
                self._signals.append(signal)
                stride_len = int(round(self.stride_seconds * self.preprocessor.cfg.fs_target))
                if stride_len <= 0:
                    raise ValueError("windowing.stride_seconds must produce a positive stride length")
                if len(signal) < window_len:
                    starts = [0]
                else:
                    starts = [start for start in range(0, len(signal) - window_len + 1, stride_len)]
                num_segments = len(starts)
                self._window_starts.append(starts)
                self.record_num_segments.append(num_segments)
                for segment_idx, _ in enumerate(starts):
                    sample_idx = len(self._samples)
                    self._samples.append((dataset_idx, segment_idx))
                    self.sample_record_ids.append(record_id)
                    self.sample_labels.append(label)
                    by_record[record_id].append(sample_idx)
            else:
                if self.noise_injector is not None:
                    signal = self.preprocessor.prepare_signal(all_xs[row_idx], fs)
                    signal, meta = self.noise_injector.inject(signal, record_id=record_id, split=self.split_name or "test")
                    meta["original_label"] = label
                    meta["processing_order"] = "resample->noise->crop_or_window->normalize"
                    self._warn_if_snr_deviates(meta)
                    self.noise_metadata.append(meta)
                    self._raw_signals.append(signal.tolist())
                else:
                    self._raw_signals.append(all_xs[row_idx])
                self.record_num_segments.append(1)
                sample_idx = len(self._samples)
                self._samples.append((dataset_idx, 0))
                self.sample_record_ids.append(record_id)
                self.sample_labels.append(label)
                by_record[record_id].append(sample_idx)

        self.record_batches = list(by_record.values())

    def _warn_if_snr_deviates(self, meta: dict) -> None:
        requested = float(meta.get("snr_db", 0.0))
        measured = float(meta.get("measured_snr_db", requested))
        if abs(measured - requested) > self.noise_snr_tolerance_db:
            print(
                "WARNING: measured SNR deviates from requested SNR: "
                f"record_id={meta.get('record_id')}, split={meta.get('split')}, "
                f"requested={requested:g} dB, measured={measured:g} dB, "
                f"tolerance={self.noise_snr_tolerance_db:g} dB"
            )

    def __len__(self):
        return len(self._samples)

    def _segment_signal(self, dataset_idx: int, segment_idx: int):
        signal = self._signals[dataset_idx]
        window_len = self.preprocessor.target_length
        start = self._window_starts[dataset_idx][segment_idx]
        stop = start + window_len
        segment = signal[start:stop]
        return segment

    def __getitem__(self, idx: int):
        dataset_idx, segment_idx = self._samples[idx]
        label = self.labels[dataset_idx]

        if self.windowing_enabled:
            x = self.preprocessor.format_segment(self._segment_signal(dataset_idx, segment_idx))
            num_segments = self.record_num_segments[dataset_idx]
        else:
            if self.noise_injector is not None:
                x = self.preprocessor.format_segment(self._raw_signals[dataset_idx])
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
