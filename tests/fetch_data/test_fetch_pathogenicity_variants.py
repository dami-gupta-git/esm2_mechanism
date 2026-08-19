"""Tests for the pathogenicity ClinVar fetch and cache contract."""

import json

import pytest

from esm2_mech.fetch_data import fetch_pathogenicity_variants as fetch_module
from esm2_mech.fetch_data.fetch_pathogenicity_variants import (
    StalePathogenicityCacheError,
    _deduplicate_protein_substitutions,
    _selection_params,
    fetch_phase,
)
from esm2_mech.utils.data import (
    validate_balanced_pathogenicity_variants,
    variants_fingerprint,
)


def _make_mechanism_variants(genes_and_uids):
    return [
        {"gene": gene, "uniprot_id": uid, "aa_wt": "A", "aa_mut": "V", "aa_pos": 1}
        for gene, uid in genes_and_uids
    ]


def _balanced_variants():
    return [
        {
            "gene": "BRAF",
            "label": "pathogenic",
            "aa_pos": 600,
            "aa_wt": "V",
            "aa_mut": "E",
            "clinvar_id": "1",
            "clinvar_ids": ["1"],
            "uniprot_id": "P15056",
        },
        {
            "gene": "BRAF",
            "label": "benign",
            "aa_pos": 601,
            "aa_wt": "V",
            "aa_mut": "A",
            "clinvar_id": "2",
            "clinvar_ids": ["2"],
            "uniprot_id": "P15056",
        },
    ]


def _current_metadata(mechanism, variants):
    return {
        "metadata_version": fetch_module._FETCH_METADATA_VERSION,
        "selection": _selection_params(mechanism, 20, 42),
        "clinvar_source": {
            "url": fetch_module.CLINVAR_URL,
            "assembly": fetch_module.CLINVAR_ASSEMBLY,
            "retrieved_at_utc": "2026-08-19T00:00:00+00:00",
            "compressed_sha256": "fixture",
            "compressed_bytes": 1,
            "last_modified": None,
        },
        "accounting": {
            "realised_design": validate_balanced_pathogenicity_variants(
                variants, require_unique_substitutions=True
            )
        },
        "variant_fingerprint": variants_fingerprint(variants),
    }


def _point_cache_at_tmp(monkeypatch, tmp_path, mechanism):
    variant_path = tmp_path / "clinvar_pathogenicity_variants.json"
    metadata_path = tmp_path / "clinvar_pathogenicity_variants.params.json"
    monkeypatch.setattr(fetch_module, "CLINVAR_PATHOGENICITY_VARIANTS_JSON", variant_path)
    monkeypatch.setattr(fetch_module, "CLINVAR_PATHOGENICITY_PARAMS_JSON", metadata_path)
    monkeypatch.setattr(fetch_module, "VARIANTS_JSON", tmp_path / "variants.json")
    monkeypatch.setattr(fetch_module, "load_variants", lambda _: mechanism)
    return variant_path, metadata_path


class TestFetchPhaseCacheValidation:
    def test_old_metadata_is_rejected_with_named_staleness_error(
        self, tmp_path, monkeypatch
    ):
        mechanism = _make_mechanism_variants([("BRAF", "P15056")])
        variant_path, metadata_path = _point_cache_at_tmp(
            monkeypatch, tmp_path, mechanism
        )
        variant_path.write_text(json.dumps(_balanced_variants()))
        metadata_path.write_text(
            json.dumps(
                {
                    "max_per_gene_per_class": 20,
                    "seed": 42,
                    "source_fingerprint": "old-schema",
                }
            )
        )

        with pytest.raises(StalePathogenicityCacheError, match="metadata_version"):
            fetch_phase(max_per_gene_per_class=20, seed=42)

    def test_matching_metadata_reuses_cache(self, tmp_path, monkeypatch):
        mechanism = _make_mechanism_variants([("BRAF", "P15056")])
        variants = _balanced_variants()
        variant_path, metadata_path = _point_cache_at_tmp(
            monkeypatch, tmp_path, mechanism
        )
        variant_path.write_text(json.dumps(variants))
        metadata_path.write_text(json.dumps(_current_metadata(mechanism, variants)))

        assert fetch_phase(max_per_gene_per_class=20, seed=42) == variants

    def test_partial_cache_is_rejected(self, tmp_path, monkeypatch):
        mechanism = _make_mechanism_variants([("BRAF", "P15056")])
        variant_path, _ = _point_cache_at_tmp(monkeypatch, tmp_path, mechanism)
        variant_path.write_text(json.dumps(_balanced_variants()))

        with pytest.raises(StalePathogenicityCacheError, match="only one"):
            fetch_phase(max_per_gene_per_class=20, seed=42)


class TestProteinSubstitutionDeduplication:
    def test_duplicate_records_are_merged_before_balance(self):
        variants = [
            {
                "gene": "BRAF",
                "aa_pos": 600,
                "aa_wt": "V",
                "aa_mut": "E",
                "label": "pathogenic",
                "clinvar_id": "20",
            },
            {
                "gene": "BRAF",
                "aa_pos": 600,
                "aa_wt": "V",
                "aa_mut": "E",
                "label": "pathogenic",
                "clinvar_id": "10",
            },
        ]

        deduplicated, accounting = _deduplicate_protein_substitutions(variants)

        assert len(deduplicated) == 1
        assert deduplicated[0]["clinvar_ids"] == ["10", "20"]
        assert accounting["n_duplicate_substitution_keys"] == 1
        assert accounting["n_duplicate_rows_removed"] == 1

    def test_conflicting_labels_abort(self):
        variants = [
            {
                "gene": "BRAF",
                "aa_pos": 600,
                "aa_wt": "V",
                "aa_mut": "E",
                "label": label,
                "clinvar_id": str(index),
            }
            for index, label in enumerate(("pathogenic", "benign"))
        ]

        with pytest.raises(RuntimeError, match="conflicting labels"):
            _deduplicate_protein_substitutions(variants)

    def test_unknown_label_aborts(self):
        variants = [
            {
                "gene": "BRAF",
                "aa_pos": 600,
                "aa_wt": "V",
                "aa_mut": "E",
                "label": "uncertain",
                "clinvar_id": "1",
            }
        ]

        with pytest.raises(ValueError, match="unexpected pathogenicity label"):
            _deduplicate_protein_substitutions(variants)
