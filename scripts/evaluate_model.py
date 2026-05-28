#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)  # .insert(0, ...) puts it at the front of the search list

import argparse
import json

DEFAULT_NOISE_ROOT = Path("data")
DEFAULT_SNR_DB = [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]
VALID_NOISE_TYPES = {"bw", "em", "ma"}
REQUIRED_NSTDB_FILES = ("bw.hea", "bw.dat", "em.hea", "em.dat", "ma.hea", "ma.dat")


def _dry_validate_noise_type(noise_type: str) -> str:
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Invalid noise type {noise_type!r}. Expected one of {sorted(VALID_NOISE_TYPES)}.")
    return noise_type


def _dry_validate_nstdb_root(noise_root: str | Path) -> Path:
    root = Path(noise_root)
    if not root.exists():
        raise FileNotFoundError(f"NSTDB noise_root does not exist: {root}")
    missing = [name for name in REQUIRED_NSTDB_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"NSTDB noise_root {root} is missing required raw noise files: {', '.join(missing)}")
    return root


def _dry_condition_key(noise_type: str, snr_db: float) -> str:
    snr = f"{float(snr_db):g}".replace("-", "neg")
    return f"noise_type={noise_type}__snr_db={snr}"


def _dry_metrics_filename(noise_type: str, snr_db: float) -> str:
    return f"metrics_zero-shot_{_dry_condition_key(noise_type, snr_db)}.json"


def _write_predictions(run_dir: Path, val_records, test_records, val_segments, test_segments) -> None:
<<<<<<< codex/add-zero-shot-noise-evaluation-pathway-f6qe3i
    import pandas as pd

=======
>>>>>>> main
    pd.DataFrame(
        {"record_id": val_records["record_id"], "y_true": val_records["y_true"].astype(int), "y_prob": val_records["y_prob"], "split": "val"}
    ).to_parquet(run_dir / "preds_val.parquet", index=False)
    pd.DataFrame(
        {"record_id": test_records["record_id"], "y_true": test_records["y_true"].astype(int), "y_prob": test_records["y_prob"], "split": "test"}
    ).to_parquet(run_dir / "preds.parquet", index=False)
    pd.DataFrame(
        {"record_id": val_segments["record_id"], "segment_idx": val_segments["segment_idx"], "y_true": val_segments["y_true"], "y_prob": val_segments["y_prob"], "split": "val"}
    ).to_parquet(run_dir / "preds_val_segments.parquet", index=False)
    pd.DataFrame(
        {"record_id": test_segments["record_id"], "segment_idx": test_segments["segment_idx"], "y_true": test_segments["y_true"], "y_prob": test_segments["y_prob"], "split": "test"}
    ).to_parquet(run_dir / "preds_segments.parquet", index=False)


<<<<<<< codex/add-zero-shot-noise-evaluation-pathway-f6qe3i
def _load_model_from_checkpoint(cfg: dict, checkpoint: str, device):
    from models import build_model
    from train.checkpointing import load_checkpoint

=======
def _load_model_from_checkpoint(cfg: dict, checkpoint: str, device: torch.device):
>>>>>>> main
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is required and was not found: {checkpoint_path}")
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    return model


def _threshold_payload(threshold: float, *, checkpoint: str, split_path: str) -> dict:
    return {
        "threshold": float(threshold),
        "threshold_source": "clean_val",
        "checkpoint_source": "clean_val",
        "checkpoint": str(checkpoint),
        "split_path": str(split_path),
    }


def _format_summary(rows: list[dict]) -> str:
    def _sort_key(row):
        snr = row.get("snr_db", 0)
        try:
            snr_value = float(snr)
        except (TypeError, ValueError):
            snr_value = float("inf")
        return (str(row.get("noise_type", "")), snr_value)

    rows = sorted(rows, key=_sort_key)
    cols = ["noise_type", "snr_db", "auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity", "confusion_matrix"]
    lines = ["\t".join(cols)]
    for row in rows:
        metrics = row.get("metrics", row)
        values = []
        for col in cols:
            if col in row:
                values.append(str(row[col]))
            elif col in metrics:
                values.append(json.dumps(metrics[col]) if col == "confusion_matrix" else str(metrics[col]))
            else:
                values.append("")
        lines.append("\t".join(values))
    return "\n".join(lines)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", "--checkpoint", dest="checkpoint", required=False)
    p.add_argument("--split", default=None)
    p.add_argument("--noise-eval", choices=["none", "zero-shot"], default="none")
    p.add_argument("--noise-root", default=str(DEFAULT_NOISE_ROOT))
    p.add_argument("--noise-types", nargs="+", default=["bw", "em", "ma"])
    p.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--ecg-fs", type=float, default=100)
    p.add_argument("--clean-threshold-path", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.noise_eval == "zero-shot" and args.dry_run:
        if not args.checkpoint:
            raise ValueError("--checkpoint/--ckpt is required for zero-shot noise evaluation; refusing to retrain silently.")
        for noise_type in args.noise_types:
            _dry_validate_noise_type(noise_type)
        _dry_validate_nstdb_root(args.noise_root)
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is required and was not found: {checkpoint}")
        output_root = Path(args.output_root) if args.output_root else checkpoint.parent.parent / "zero_shot_noise_eval"
        print(f"Clean checkpoint: {checkpoint}")
        print(f"Noise root: {Path(args.noise_root)}")
        print(f"Conditions: noise_types={args.noise_types}, snr_db={args.snr_db}")
        for noise_type in args.noise_types:
            for snr_db in args.snr_db:
                print(f"DRY-RUN would evaluate noise_type={noise_type}, snr_db={snr_db:g}")
                print(f"DRY-RUN metric path: {output_root / _dry_metrics_filename(noise_type, snr_db)}")
        print("DRY-RUN complete: no writes performed and no evaluation run.")
        return

    import pandas as pd
    import torch

    from data.datamodule import make_dataloaders
    from data.splits import load_split
    from evaluate.evaluator import evaluate_record_level
    from evaluate.noise_protocol import (
        NoiseCondition,
        condition_key,
        load_clean_threshold,
        metrics_filename,
        validate_noise_type,
        validate_nstdb_root,
    )
    from evaluate.plots import save_plots
<<<<<<< codex/add-zero-shot-noise-evaluation-pathway-f6qe3i
=======
    from models import build_model
    from train.checkpointing import load_checkpoint
>>>>>>> main
    from train.metrics import choose_threshold_max_f1, compute_metrics
    from utils.config import load_config
    from utils.logging import save_json

    cfg = load_config(args.config)
    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"holdout_seed{cfg['split']['seed']}" / "split.json")
    split = load_split(split_path)

    if args.noise_eval == "zero-shot":
        if not args.checkpoint:
            raise ValueError("--checkpoint/--ckpt is required for zero-shot noise evaluation; refusing to retrain silently.")
        for noise_type in args.noise_types:
            validate_noise_type(noise_type)
        validate_nstdb_root(args.noise_root)
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is required and was not found: {checkpoint}")
        output_root = Path(args.output_root) if args.output_root else checkpoint.parent.parent / "zero_shot_noise_eval"
        print(f"Clean checkpoint: {checkpoint}")
        print(f"Noise root: {Path(args.noise_root)}")
        print(f"Conditions: noise_types={args.noise_types}, snr_db={args.snr_db}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model_from_checkpoint(cfg, str(checkpoint), device)
        _, val_loader, clean_test_loader = make_dataloaders(cfg, split)
        if args.clean_threshold_path:
            threshold = load_clean_threshold(args.clean_threshold_path)
            threshold_path = Path(args.clean_threshold_path)
        else:
            _, val_records, _, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
            threshold = choose_threshold_max_f1(val_records["y_true"], val_records["y_prob"])
            threshold_path = output_root / "clean_validation_threshold.json"
            save_json(threshold_path, _threshold_payload(threshold, checkpoint=str(checkpoint), split_path=split_path))
        print(f"Clean validation threshold tau*: {threshold} (source={threshold_path})")

        clean_metrics, _, _, _ = evaluate_record_level(model, clean_test_loader, device, threshold=threshold)
        save_json(output_root / "metrics_clean_baseline.json", {"eval": "clean-baseline", "threshold": threshold, "test": clean_metrics})

        summary_rows = [{"noise_type": "clean", "snr_db": "clean", "metrics": clean_metrics}]
        for noise_type in args.noise_types:
            for snr_db in args.snr_db:
                condition = NoiseCondition(noise_type, snr_db)
                print(f"Evaluating zero-shot noise: noise_type={noise_type}, snr_db={snr_db:g}")
                _, _, noisy_test_loader = make_dataloaders(
                    cfg,
                    split,
                    test_noise_cfg={
                        "enabled": True,
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "base_seed": args.base_seed,
                        "target_fs": args.ecg_fs,
                        "noise_root": args.noise_root,
                    },
                )
                test_count = len(noisy_test_loader.dataset.record_ids)
                print(f"Test examples processed: {test_count}")
                metrics, test_records, test_segments, _ = evaluate_record_level(model, noisy_test_loader, device, threshold=threshold)
                metric_path = output_root / metrics_filename(condition)
                payload = {
                    "eval": "zero-shot",
                    "condition": condition_key(condition),
                    "noise_type": noise_type,
                    "snr_db": float(snr_db),
                    "threshold": float(threshold),
                    "threshold_source": "clean_val",
                    "checkpoint": str(checkpoint),
                    "checkpoint_source": "clean_val",
                    "num_test_examples": int(test_count),
                    "test": metrics,
                    "noise_metadata": noisy_test_loader.dataset.noise_metadata,
                }
                save_json(metric_path, payload)
                pred_path = output_root / f"preds_{condition_key(condition)}.parquet"
                pd.DataFrame(
                    {"record_id": test_records["record_id"], "y_true": test_records["y_true"].astype(int), "y_prob": test_records["y_prob"], "split": "test"}
                ).to_parquet(pred_path, index=False)
                seg_path = output_root / f"preds_segments_{condition_key(condition)}.parquet"
                pd.DataFrame(
                    {"record_id": test_segments["record_id"], "segment_idx": test_segments["segment_idx"], "y_true": test_segments["y_true"], "y_prob": test_segments["y_prob"], "split": "test"}
                ).to_parquet(seg_path, index=False)
                print(f"Metric path: {metric_path}")
                summary_rows.append({"noise_type": noise_type, "snr_db": float(snr_db), "metrics": metrics})
        save_json(output_root / "robustness_summary.json", {"rows": summary_rows, "threshold": threshold})
        print("Final robustness summary table:")
        print(_format_summary(summary_rows))
        return

    if not args.checkpoint:
        raise ValueError("--ckpt/--checkpoint is required for clean evaluation.")
    _, val_loader, test_loader = make_dataloaders(cfg, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(cfg, args.checkpoint, device)

    _, val_records, val_segments, _ = evaluate_record_level(model, val_loader, device, threshold=0.5)
    thr = choose_threshold_max_f1(val_records["y_true"], val_records["y_prob"])
    tm, test_records, test_segments, _ = evaluate_record_level(model, test_loader, device, threshold=thr)
    vm = compute_metrics(val_records["y_true"], val_records["y_prob"], thr)
    ckpt_path = Path(args.checkpoint)
    run_dir = Path(args.output_root) if args.output_root else (ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent)

    _write_predictions(run_dir, val_records, test_records, val_segments, test_segments)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plots(plots_dir, test_records["y_true"], test_records["y_prob"], thr)
    save_json(run_dir / "clean_validation_threshold.json", _threshold_payload(thr, checkpoint=str(ckpt_path), split_path=split_path))
    save_json(run_dir / "metrics_eval.json", {"val": vm, "test": tm, "threshold": thr, "threshold_source": "clean_val"})
    print({"val": vm, "test": tm, "threshold": thr})


if __name__ == "__main__":
    main()
