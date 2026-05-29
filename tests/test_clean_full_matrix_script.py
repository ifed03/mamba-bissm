from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path("scripts/run_clean_full_matrix.py")
_SPEC = spec_from_file_location("run_clean_full_matrix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
clean_matrix_configs = _MODULE.clean_matrix_configs
controlled_ecgmamba_backbone_configs = _MODULE.controlled_ecgmamba_backbone_configs
depth_sweep_configs = _MODULE.depth_sweep_configs
external_baseline_configs = _MODULE.external_baseline_configs


def test_clean_matrix_has_32_unique_existing_configs():
    configs = clean_matrix_configs()
    assert len(configs) == 32
    assert len(set(configs)) == 32
    for cfg in configs:
        assert Path(cfg).exists(), f"missing expected config: {cfg}"


def test_clean_matrix_keeps_controlled_backbones_separate_from_external_baselines():
    controlled = controlled_ecgmamba_backbone_configs()
    depth_sweep = depth_sweep_configs()
    external = external_baseline_configs()

    assert len(controlled) == 16
    assert len(depth_sweep) == 8
    assert len(external) == 8
    assert not set(controlled) & set(depth_sweep)
    assert not set(controlled) & set(external)
    assert not set(depth_sweep) & set(external)
    assert all("binary_bilstm_" not in cfg for cfg in controlled)
    assert all("binary_ecgmamba_bilstm_" in cfg for cfg in controlled[-4:])
    assert all("d64_n4" in cfg for cfg in depth_sweep)
    assert all("binary_cnn1d_" not in cfg for cfg in controlled)
    assert all("binary_cnn1d_" not in cfg for cfg in depth_sweep)
    assert {f"configs/binary_cnn1d_100hz_win{w}s_stride2s.yaml" for w in (4, 6, 8, 10)} <= set(external)
    assert sum("binary_bilstm_" in cfg for cfg in external) == 4
    assert sum("binary_cnn1d_" in cfg for cfg in external) == 4
