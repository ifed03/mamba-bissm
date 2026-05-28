from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path("scripts/run_clean_full_matrix.py")
_SPEC = spec_from_file_location("run_clean_full_matrix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
clean_matrix_configs = _MODULE.clean_matrix_configs


def test_clean_matrix_has_20_unique_existing_configs():
    configs = clean_matrix_configs()
    assert len(configs) == 20
    assert len(set(configs)) == 20
    for cfg in configs:
        assert Path(cfg).exists(), f"missing expected config: {cfg}"
