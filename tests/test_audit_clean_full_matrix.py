from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SCRIPT_PATH = Path("scripts/audit_clean_full_matrix.py")
_SPEC = spec_from_file_location("audit_clean_full_matrix", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_audit_uses_the_controlled_backbone_matrix_only():
    configs = _MODULE.controlled_ecgmamba_backbone_configs()

    assert len(configs) == 16
    assert all("d64_n4" not in cfg for cfg in configs)
    assert all("configs/binary_bilstm_" not in cfg for cfg in configs)


def test_audit_backbone_labels_cover_controlled_variants():
    assert _MODULE.BACKBONE_LABELS == {
        "bissm": "BiSSM",
        "mamba": "Mamba",
        "bimamba": "BiMamba",
        "bilstm": "BiLSTM-backbone",
    }
