import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from .parquet_dataset import ParquetECGDataset, RecordBatchSampler


def make_dataloaders(cfg: dict, split: dict, test_noise_cfg: dict | None = None, noise_training_cfg: dict | None = None):
    data_path = cfg["paths"]["data_path"]
    pcfg = cfg["preprocessing"]
    train_noise_cfg = noise_training_cfg if noise_training_cfg is not None else None
    val_noise_cfg = noise_training_cfg if noise_training_cfg is not None else None
    test_noise = noise_training_cfg if noise_training_cfg is not None else test_noise_cfg
    train_ds = ParquetECGDataset(data_path, split["train"], train=True, preprocess_cfg=pcfg, split_name="train", noise_cfg=train_noise_cfg)
    val_ds = ParquetECGDataset(data_path, split["val"], train=False, preprocess_cfg=pcfg, split_name="val", noise_cfg=val_noise_cfg)
    test_ds = ParquetECGDataset(
        data_path,
        split["test"],
        train=False,
        preprocess_cfg=pcfg,
        split_name="test",
        noise_cfg=test_noise,
    )

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
