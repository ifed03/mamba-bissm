#!/usr/bin/env python
import sys
from pathlib import Path

# Adds 'src' to path so 'data', 'models', and 'train' (the folder) are accessible
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path) # .insert(0, ...) puts it at the front of the search list

import argparse
import json

DEFAULT_SNR_DB = [24.0, 18.0, 12.0, 6.0, 0.0, -6.0]
VALID_NOISE_TYPES = {"bw", "em", "ma"}
REQUIRED_NSTDB_FILES = ("bw.hea", "bw.dat", "em.hea", "em.dat", "ma.hea", "ma.dat")


def _simple_load_config(path: str) -> dict:
    try:
        from utils.config import load_config

        return load_config(path)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in Path(path).read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value = raw_line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            scalar = value.strip().strip("'\"")
            if scalar.lower() in {"true", "false"}:
                parsed = scalar.lower() == "true"
            else:
                try:
                    parsed = int(scalar)
                except ValueError:
                    try:
                        parsed = float(scalar)
                    except ValueError:
                        parsed = scalar
            parent[key] = parsed
    return root


def _load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _validate_noise_type(noise_type: str) -> str:
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(f"Invalid noise type {noise_type!r}. Expected one of {sorted(VALID_NOISE_TYPES)}.")
    return noise_type


def _validate_nstdb_root(noise_root: str | Path) -> Path:
    root = Path(noise_root)
    if not root.exists():
        raise FileNotFoundError(f"NSTDB noise_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"NSTDB noise_root is not a directory: {root}")
    missing = [name for name in REQUIRED_NSTDB_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"NSTDB noise_root {root} is missing required raw noise files: {', '.join(missing)}")
    return root


def _condition_key(noise_type: str, snr_db: float) -> str:
    snr = f"{float(snr_db):g}".replace("-", "neg")
    return f"noise_type={noise_type}__snr_db={snr}"


def _noisy_input_condition_name(noise_type: str, snr_db: float) -> str:
    snr = f"{float(snr_db):g}".replace("-", "neg")
    return f"noisy_input_training_{noise_type}_{snr}dB"


def _noisy_input_metrics_filename(noise_type: str, snr_db: float) -> str:
    return f"metrics_noisy-input-training_{_condition_key(noise_type, snr_db)}.json"


def _noisy_input_threshold_filename(noise_type: str, snr_db: float) -> str:
    return f"threshold_noisy-input-training_{_condition_key(noise_type, snr_db)}.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--noise-training-mode", choices=["clean", "noisy-input"], default="clean")
    p.add_argument("--noise-root", default="data")
    p.add_argument("--noise-types", nargs="+", default=["bw", "em", "ma"])
    p.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--ecg-fs", type=float, default=100)
    p.add_argument("--output-root", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _split_counts(split: dict) -> dict:
    return {name: len(indices) for name, indices in split.items() if name in {"train", "val", "test"}}


def _dry_run(args, cfg: dict, split: dict, split_path: str) -> None:
    output_root = Path(args.output_root or cfg["paths"].get("runs_dir", "runs"))
    print("DRY-RUN noisy-input training/evaluation preflight")
    print(f"Input dataset path: {cfg['paths']['data_path']}")
    print(f"Input split path: {split_path}")
    print(f"Output root: {output_root}")
    print(f"Records per split: {_split_counts(split)}")
    print(f"Selected noise types: {args.noise_types}")
    print(f"Selected SNRs dB: {args.snr_db}")
    print(f"Sampling frequency: {args.ecg_fs:g} Hz")
    print("Data materialisation: dynamic injection in dataloaders; clean parquet is not overwritten")
    print("Checkpoint selection split: noisy validation")
    print("Threshold tau* selection split: noisy validation")
    for noise_type in args.noise_types:
        for snr_db in args.snr_db:
            run_name = args.run_name or _noisy_input_condition_name(noise_type, snr_db)
            out_dir = output_root / run_name
            print(f"DRY-RUN condition: {_condition_key(noise_type, snr_db)}")
            print(f"DRY-RUN output path: {out_dir}")
            print(f"DRY-RUN metrics path: {out_dir / _noisy_input_metrics_filename(noise_type, snr_db)}")
            print(f"DRY-RUN threshold path: {out_dir / _noisy_input_threshold_filename(noise_type, snr_db)}")
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(Path(cfg["paths"]["data_path"]), columns=["record_id", "label", "fs"])
        preview = table.slice(0, min(3, table.num_rows)).to_pydict()
        rows = [dict(zip(preview, values, strict=True)) for values in zip(*preview.values(), strict=True)] if preview else []
        print(f"First metadata-like input rows: {rows}")
    except ModuleNotFoundError:
        print("First metadata-like input rows: unavailable (pyarrow is not installed in this environment)")
    print("DRY-RUN complete: no writes performed and no training run.")


def _noise_cfg(args, noise_type: str, snr_db: float) -> dict:
    return {
        "enabled": True,
        "mode": "noisy-input",
        "noise_type": noise_type,
        "snr_db": float(snr_db),
        "base_seed": int(args.base_seed),
        "target_fs": float(args.ecg_fs),
        "noise_root": args.noise_root,
    }


def main():
    args = _parse_args()
    cfg = _simple_load_config(args.config)

    split_path = args.split or str(Path(cfg["paths"]["splits_dir"]) / f"holdout_seed{cfg['split']['seed']}" / "split.json")
    split = _load_split(split_path)

    if args.noise_training_mode == "noisy-input":
        for noise_type in args.noise_types:
            _validate_noise_type(noise_type)
        _validate_nstdb_root(args.noise_root)
        if args.dry_run:
            _dry_run(args, cfg, split, split_path)
            return

        output_root = Path(args.output_root or cfg["paths"].get("runs_dir", "runs"))
        metrics_by_condition = {}
        for noise_type in args.noise_types:
            for snr_db in args.snr_db:
                from evaluate.noise_protocol import NoiseCondition, condition_key, noisy_input_condition_name, noisy_input_metrics_filename

                condition = NoiseCondition(noise_type, snr_db)
                print(f"Starting noisy-input training/evaluation for {condition_key(condition)}")
                print("Progress: constructing noisy train/val/test dataloaders")
                from data.datamodule import make_dataloaders
                from models import build_model
                from train.trainer import train_model
                from utils.config import save_config
                from utils.io import make_run_dir
                from utils.seed import set_seed

                set_seed(cfg["split"]["seed"], cfg["training"].get("deterministic", True))
                train_loader, val_loader, test_loader = make_dataloaders(
                    cfg,
                    split,
                    noise_training_cfg=_noise_cfg(args, noise_type, snr_db),
                )
                for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                    print(
                        f"Prepared split={split_name}: records={len(loader.dataset.record_ids)}, "
                        f"segments={len(loader.dataset)}, noisy_records={len(loader.dataset.noise_metadata)}"
                    )
                run_cfg = dict(cfg)
                run_cfg["noise_training"] = {
                    "mode": "noisy-input",
                    "noise_type": noise_type,
                    "snr_db": float(snr_db),
                    "base_seed": int(args.base_seed),
                    "noise_root": str(args.noise_root),
                    "ecg_fs": float(args.ecg_fs),
                }
                model = build_model(cfg)
                run_name = args.run_name or noisy_input_condition_name(condition)
                if args.run_name and (len(args.noise_types) > 1 or len(args.snr_db) > 1):
                    run_name = f"{args.run_name}__{noisy_input_condition_name(condition)}"
                run_cfg["run_name"] = run_name
                run_dir = make_run_dir(str(output_root), run_name, cfg)
                save_config(run_dir / "config_resolved.yaml", run_cfg)
                metrics = train_model(model, train_loader, val_loader, test_loader, run_cfg, run_dir)
                metrics_by_condition[condition_key(condition)] = metrics
                print(
                    "Condition summary: "
                    f"output={run_dir}, metrics={run_dir / noisy_input_metrics_filename(condition)}, "
                    f"tau*={metrics['threshold']}, checkpoint={run_dir / 'checkpoints' / 'best.ckpt'}"
                )
        print(f"Noisy-input training/evaluation complete for {len(metrics_by_condition)} condition(s).")
        compact = {
            key: {
                "run_name": value.get("run_name"),
                "best_epoch": value.get("best_epoch"),
                "best_val_metric_name": value.get("best_val_metric_name"),
                "best_val_metric": value.get("best_val_metric"),
                "threshold": value.get("threshold"),
                "test": value.get("test"),
            }
            for key, value in metrics_by_condition.items()
        }
        print(compact)
        return

    if args.dry_run:
        print("DRY-RUN clean training preflight")
        print(f"Input dataset path: {cfg['paths']['data_path']}")
        print(f"Input split path: {split_path}")
        print(f"Output root: {Path(args.output_root or cfg['paths']['runs_dir'])}")
        print(f"Records per split: {_split_counts(split)}")
        print("DRY-RUN complete: no writes performed and no training run.")
        return

    from data.datamodule import make_dataloaders
    from models import build_model
    from train.trainer import train_model
    from utils.config import save_config
    from utils.io import make_run_dir
    from utils.seed import set_seed

    set_seed(cfg["split"]["seed"], cfg["training"].get("deterministic", True))
    train_loader, val_loader, test_loader = make_dataloaders(cfg, split)
    model = build_model(cfg)
    run_dir = make_run_dir(args.output_root or cfg["paths"]["runs_dir"], args.run_name, cfg)
    run_cfg = dict(cfg)
    if args.run_name:
        run_cfg["run_name"] = args.run_name
    save_config(run_dir / "config_resolved.yaml", run_cfg)
    metrics = train_model(model, train_loader, val_loader, test_loader, run_cfg, run_dir)
    print(metrics)


if __name__ == "__main__":
    main()
