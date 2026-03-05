import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from .parquet_dataset import ParquetECGDataset, RecordBatchSampler


def make_dataloaders(cfg: dict, split: dict):
    data_path = cfg["paths"]["data_path"]
    pcfg = cfg["preprocessing"]
    train_ds = ParquetECGDataset(data_path, split["train"], train=True, preprocess_cfg=pcfg)
    val_ds = ParquetECGDataset(data_path, split["val"], train=False, preprocess_cfg=pcfg)
    test_ds = ParquetECGDataset(data_path, split["test"], train=False, preprocess_cfg=pcfg)

    batch_size = cfg["training"]["batch_size"]
    sampler = None
    if cfg["training"].get("weighted_sampler", False):
        labels = np.array(train_ds.sample_labels)
        class_count = np.bincount(labels.astype(int))
        weights = 1.0 / class_count
        sample_w = weights[labels.astype(int)]
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0)
    if val_ds.windowing_enabled:
        val_loader = DataLoader(val_ds, batch_sampler=RecordBatchSampler(val_ds.record_batches), num_workers=0)
        test_loader = DataLoader(test_ds, batch_sampler=RecordBatchSampler(test_ds.record_batches), num_workers=0)
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
