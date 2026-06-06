"""
Tests for esm2_mech.experiments.mechanism.mut_only_mlp.

Invariants:
- the module imports without ImportError (it previously imported three names
  that do not exist in mlp.py, so it crashed on import).
- load_variants_and_labels reads a variants JSON and returns row-aligned
  (variants, labels, genes) with labels collapsed to GOF/DN/LOF (HI/AR -> LOF).
"""

import json

import numpy as np


def test_module_imports():
    import esm2_mech.experiments.mechanism.mut_only_mlp as m

    assert hasattr(m, "load_variants_and_labels")
    assert hasattr(m, "load_wt_mut_mean_embeddings")
    assert hasattr(m, "main")


def test_load_variants_and_labels_collapses_and_aligns(tmp_path):
    from esm2_mech.experiments.mechanism.mut_only_mlp import load_variants_and_labels
    from esm2_mech.utils.constants import GOF, DN, LOF

    variants = [
        {"gene": "A", "mechanism": "GOF"},
        {"gene": "B", "mechanism": "HI"},   # -> LOF
        {"gene": "C", "mechanism": "AR"},   # -> LOF
        {"gene": "A", "mechanism": "DN"},
    ]
    path = tmp_path / "variants.json"
    with open(path, "w") as f:
        json.dump(variants, f)

    loaded, labels, genes = load_variants_and_labels(str(path))
    assert len(loaded) == len(labels) == len(genes) == 4
    assert list(labels) == [GOF, LOF, LOF, DN]
    assert list(genes) == ["A", "B", "C", "A"]
