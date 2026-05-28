.PHONY: train eval test splits sweep_window_ablation

CONFIG ?= configs/binary_ecgmamba_100hz.yaml
CKPT ?=

train:
	python scripts/train_model.py --config $(CONFIG)

eval:
	python scripts/evaluate_model.py --config $(CONFIG) --ckpt $(CKPT)

splits:
	python scripts/make_splits.py --config $(CONFIG)

test:
	pytest -q

sweep_window_ablation:
	python scripts/sweep.py --results runs/sweeps/window_ablation_ecgmamba_mamba_bissm.csv --configs \
		configs/binary_ecgmamba_reduced_100hz_win4s_stride2s.yaml \
		configs/binary_ecgmamba_reduced_100hz_win6s_stride2s.yaml \
		configs/binary_ecgmamba_reduced_100hz_win8s_stride2s.yaml \
		configs/binary_ecgmamba_reduced_100hz_win10s_stride2s.yaml \
		configs/binary_ecgmamba_100hz_win4s_stride2s.yaml \
		configs/binary_ecgmamba_100hz_win6s_stride2s.yaml \
		configs/binary_ecgmamba_100hz_win8s_stride2s.yaml \
		configs/binary_ecgmamba_100hz_win10s_stride2s.yaml \
		configs/binary_bissm_reduced4_100hz_win10s_stride2s.yaml
