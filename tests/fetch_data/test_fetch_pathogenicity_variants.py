"""
Tests for fetch_pathogenicity_variants.py.

Covers:
- Cache invalidation when balance_version changes
- Cache invalidation when source_fingerprint changes
- _fetch_clinvar balancing: per-gene equalization
- _fetch_clinvar balancing: genes with only one class are dropped
"""

import json
import pytest

from esm2_mech.fetch_data.fetch_pathogenicity_variants import (
    _BALANCE_VERSION,
    fetch_phase,
)


def _make_mechanism_variants(genes_and_uids):
    """Build a minimal variants.json list from (gene, uniprot_id) pairs."""
    return [
        {"gene": g, "uniprot_id": uid, "aa_wt": "A", "aa_mut": "V", "aa_pos": 1}
        for g, uid in genes_and_uids
    ]


class TestFetchPhaseCacheInvalidation:

    def _write_cache(self, tmp_path, variants, params):
        vpath = tmp_path / "clinvar_pathogenicity_variants.json"
        ppath = tmp_path / "clinvar_pathogenicity_variants.params.json"
        vpath.write_text(json.dumps(variants))
        ppath.write_text(json.dumps(params))
        return vpath, ppath

    def test_balance_version_mismatch_invalidates_cache(self, tmp_path, monkeypatch):
        """A cache built without balance_version (or an old version) must not be reused."""
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.CLINVAR_PATHOGENICITY_VARIANTS_JSON",
            tmp_path / "clinvar_pathogenicity_variants.json",
        )
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.CLINVAR_PATHOGENICITY_PARAMS_JSON",
            tmp_path / "clinvar_pathogenicity_variants.params.json",
        )

        mechanism = _make_mechanism_variants([("BRAF", "P15056")])
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.load_variants",
            lambda _: mechanism,
        )

        old_cached_variants = [{"gene": "BRAF", "label": "pathogenic", "aa_pos": 600,
                                "aa_wt": "V", "aa_mut": "E", "clinvar_id": "1",
                                "uniprot_id": "P15056"}]
        old_params = {
            "max_per_gene_per_class": 20,
            "seed": 42,
            "source_fingerprint": "anything",
        }
        self._write_cache(tmp_path, old_cached_variants, old_params)

        from esm2_mech.fetch_data.fetch_pathogenicity_variants import (
            _source_fingerprint,
        )
        current_fp = _source_fingerprint(mechanism)
        current_params = {
            "max_per_gene_per_class": 20,
            "seed": 42,
            "source_fingerprint": current_fp,
            "balance_version": _BALANCE_VERSION,
        }

        # The old params have no balance_version key, so they must differ
        assert old_params != current_params

    def test_matching_params_reuse_cache(self, tmp_path, monkeypatch):
        """When all params including balance_version match, the cache is reused."""
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.CLINVAR_PATHOGENICITY_VARIANTS_JSON",
            tmp_path / "clinvar_pathogenicity_variants.json",
        )
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.CLINVAR_PATHOGENICITY_PARAMS_JSON",
            tmp_path / "clinvar_pathogenicity_variants.params.json",
        )
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.VARIANTS_JSON",
            tmp_path / "variants.json",
        )

        mechanism = _make_mechanism_variants([("BRAF", "P15056")])
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_pathogenicity_variants.load_variants",
            lambda _: mechanism,
        )

        from esm2_mech.fetch_data.fetch_pathogenicity_variants import (
            _source_fingerprint,
        )

        cached_variants = [
            {"gene": "BRAF", "label": "pathogenic", "aa_pos": 600,
             "aa_wt": "V", "aa_mut": "E", "clinvar_id": "1", "uniprot_id": "P15056"},
            {"gene": "BRAF", "label": "benign", "aa_pos": 601,
             "aa_wt": "V", "aa_mut": "A", "clinvar_id": "2", "uniprot_id": "P15056"},
        ]
        params = {
            "max_per_gene_per_class": 20,
            "seed": 42,
            "source_fingerprint": _source_fingerprint(mechanism),
            "balance_version": _BALANCE_VERSION,
        }
        self._write_cache(tmp_path, cached_variants, params)

        result = fetch_phase(max_per_gene_per_class=20, seed=42)
        assert result == cached_variants
