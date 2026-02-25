.PHONY: train eval test splits

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
