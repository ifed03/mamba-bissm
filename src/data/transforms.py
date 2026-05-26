from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import resample_poly


@dataclass
class TransformConfig:
    fs_target: int = 100
    target_seconds: float = 10.0
    normalize: str = "zscore"
    random_crop: bool = True


class ECGPreprocessor:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    @property
    def target_length(self) -> int:
        return int(self.cfg.target_seconds * self.cfg.fs_target)

    def _resample(self, x: np.ndarray, fs_source: int) -> np.ndarray:
        if fs_source == self.cfg.fs_target:
            return x.astype(np.float32)
        return resample_poly(x, up=self.cfg.fs_target, down=fs_source).astype(np.float32)

    def _crop_pad(self, x: np.ndarray, target_len: int | None = None) -> np.ndarray:
        target_len = self.target_length if target_len is None else int(target_len)
        if len(x) > target_len:
            start = np.random.randint(0, len(x) - target_len + 1) if self.cfg.random_crop else (len(x) - target_len) // 2
            x = x[start : start + target_len]
        elif len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)), mode="constant")
        return x.astype(np.float32)

    def _crop_only(self, x: np.ndarray, target_len: int | None = None) -> np.ndarray:
        target_len = self.target_length if target_len is None else int(target_len)
        if len(x) > target_len:
            start = np.random.randint(0, len(x) - target_len + 1) if self.cfg.random_crop else (len(x) - target_len) // 2
            x = x[start : start + target_len]
        return x.astype(np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.normalize == "none":
            return x
        if self.cfg.normalize == "zscore":
            return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)
        if self.cfg.normalize == "robust":
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-8
            return ((x - med) / mad).astype(np.float32)
        raise ValueError(f"Unknown normalization: {self.cfg.normalize}")

    def prepare_signal(self, x_list, fs_source: int) -> np.ndarray:
        x = np.asarray(x_list, dtype=np.float32)
        return self._resample(x, fs_source)

    def format_segment(self, x: np.ndarray, target_len: int | None = None) -> torch.Tensor:
        target_len = self.target_length if target_len is None else int(target_len)
        x = self._crop_only(x, target_len=target_len)
        x = self._normalize(x)
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)), mode="constant")
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    def __call__(self, x_list, fs_source: int) -> torch.Tensor:
        x = self.prepare_signal(x_list, fs_source)
        return self.format_segment(x)
