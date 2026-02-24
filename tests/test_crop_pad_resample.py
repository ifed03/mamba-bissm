import numpy as np

from data.transforms import ECGPreprocessor, TransformConfig


def test_resample_500_to_100():
    x = np.random.randn(5000).astype(np.float32)
    tr = ECGPreprocessor(TransformConfig(fs_target=100, target_seconds=10.0, normalize="none", random_crop=False))
    y = tr(x, fs_source=500)
    assert y.shape[-1] == 1000


def test_crop_pad_exact_len():
    tr = ECGPreprocessor(TransformConfig(fs_target=100, target_seconds=10.0, normalize="none", random_crop=False))
    short = tr(np.random.randn(100).astype(np.float32), fs_source=100)
    long = tr(np.random.randn(5000).astype(np.float32), fs_source=100)
    assert short.shape == (1, 1000)
    assert long.shape == (1, 1000)
