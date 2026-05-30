"""
Tests for esm2_mech.utils.paths.

Invariants:
- All exported Path constants are Path objects (not strings)
- Embedding paths are under DATA_DIR / "embeddings" / ESM2_MODEL
- ESM3 embedding paths are under DATA_DIR / "embeddings" / ESM3_MODEL
- PREREQUISITE_FILES list is non-empty and all entries are Path objects
- check_prerequisites() returns False when files are absent, True when present
- PROJECT_ROOT is an ancestor of DATA_DIR
"""

import json
from pathlib import Path

import pytest

from esm2_mech.utils.paths import (
    DATA_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    CACHE_DIR,
    EMB_DIR,
    ESM3_EMB_DIR,
    VARIANTS_JSON,
    GENE_LIST_TSV,
    PFAM_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    ESM3_EMB_SEQ,
    PREREQUISITE_FILES,
    PROJECT_ROOT,
    check_prerequisites,
)
from esm2_mech.utils.constants import ESM2_MODEL, ESM3_MODEL


def test_all_path_constants_are_path_objects():
    paths = [
        DATA_DIR, RESULTS_DIR, REPORTS_DIR, CACHE_DIR,
        EMB_DIR, ESM3_EMB_DIR, VARIANTS_JSON, GENE_LIST_TSV,
        PFAM_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, ESM3_EMB_SEQ,
    ]
    for p in paths:
        assert isinstance(p, Path), f"{p!r} is not a Path"


def test_project_root_is_ancestor_of_data_dir():
    assert DATA_DIR.is_relative_to(PROJECT_ROOT)


def test_emb_dir_uses_esm2_model():
    assert ESM2_MODEL in str(EMB_DIR)


def test_esm3_emb_dir_uses_esm3_model():
    assert ESM3_MODEL in str(ESM3_EMB_DIR)


def test_cache_dir_under_data_dir():
    assert CACHE_DIR.is_relative_to(DATA_DIR)


def test_emb_wt_mean_under_emb_dir():
    assert EMB_WT_MEAN.is_relative_to(EMB_DIR)


def test_prerequisite_files_nonempty():
    assert len(PREREQUISITE_FILES) > 0


def test_prerequisite_files_are_paths():
    for p in PREREQUISITE_FILES:
        assert isinstance(p, Path)


def test_check_prerequisites_returns_false_when_missing(tmp_path, monkeypatch):
    # Point PREREQUISITE_FILES at a file that doesn't exist.
    import esm2_mech.utils.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PREREQUISITE_FILES", [tmp_path / "nonexistent.txt"])
    assert paths_mod.check_prerequisites() is False


def test_check_prerequisites_returns_true_when_all_present(tmp_path, monkeypatch):
    present = tmp_path / "file.txt"
    present.write_text("x")
    import esm2_mech.utils.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PREREQUISITE_FILES", [present])
    assert paths_mod.check_prerequisites() is True


def test_check_prerequisites_false_if_any_missing(tmp_path, monkeypatch):
    present = tmp_path / "present.txt"
    present.write_text("x")
    missing = tmp_path / "missing.txt"
    import esm2_mech.utils.paths as paths_mod
    monkeypatch.setattr(paths_mod, "PREREQUISITE_FILES", [present, missing])
    assert paths_mod.check_prerequisites() is False
