"""Tests for the pathogenicity positive-control data and result contracts."""

import json

import numpy as np
import pytest

from esm2_mech.experiments.pathogenicity import pathogenicity_control as control
from esm2_mech.experiments.pathogenicity.pathogenicity_control import (
    CLAIM_2C_THRESHOLD,
    ExpectedPathogenicitySelection,
    _build_claim_2c,
    _rebalance_after_filter,
    _seed_params,
    _run_probe_with_contract,
    _validate_embedding_cache,
)
from esm2_mech.utils.data import embedding_fingerprint, pfam_fingerprint


def _variant(gene, label, pos=1, wt="A", mut="V", uid="P00000"):
    return {
        "gene": gene,
        "label": label,
        "aa_pos": pos,
        "aa_wt": wt,
        "aa_mut": mut,
        "uniprot_id": uid,
    }


def test_shared_probe_receives_classes_and_split_contract():
    captured = {}

    def fake_probe(features, labels, splits, **kwargs):
        captured.update(kwargs)
        return {"status": "success"}, None

    split_contract = {"status": "valid", "classes": [0, 1]}
    _run_probe_with_contract(
        fake_probe,
        np.ones((2, 1)),
        np.array([0, 1]),
        [(np.array([0]), np.array([1]))],
        [0, 1],
        split_contract,
        seed=0,
    )

    assert captured["classes"] == [0, 1]
    assert captured["split_contract"] is split_contract


class TestRebalanceAfterFilter:
    def test_uneven_filter_is_corrected(self):
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "benign", pos=2),
            _variant("BRAF", "benign", pos=3),
            _variant("BRAF", "benign", pos=4),
        ]
        indices = list(range(len(valid)))
        sequences = ["S"] * len(valid)
        positions = [0] * len(valid)

        _, output, _, _, _, accounting = _rebalance_after_filter(
            indices, valid, sequences, sequences, positions
        )

        labels = [variant["label"] for variant in output]
        assert labels.count("pathogenic") == labels.count("benign") == 1
        assert accounting["n_removed_by_postfilter_balance"] == 2

    def test_gene_with_only_one_class_is_dropped(self):
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "pathogenic", pos=2),
            _variant("TP53", "pathogenic", pos=1),
            _variant("TP53", "benign", pos=2),
        ]
        indices = list(range(len(valid)))
        sequences = ["S"] * len(valid)
        positions = [0] * len(valid)

        _, output, _, _, _, accounting = _rebalance_after_filter(
            indices, valid, sequences, sequences, positions
        )

        assert {variant["gene"] for variant in output} == {"TP53"}
        assert accounting["n_single_class_genes_dropped_postfilter"] == 1

    def test_already_balanced_input_is_unchanged(self):
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "benign", pos=2),
            _variant("TP53", "pathogenic", pos=3),
            _variant("TP53", "benign", pos=4),
        ]
        indices = list(range(len(valid)))
        sequences = ["S"] * len(valid)
        positions = [0] * len(valid)

        output_indices, output, _, _, _, accounting = _rebalance_after_filter(
            indices, valid, sequences, sequences, positions
        )

        assert output == valid
        assert output_indices == indices
        assert accounting["n_removed_by_postfilter_balance"] == 0

    def test_parallel_lists_stay_aligned(self):
        valid = [
            _variant("BRAF", "pathogenic", pos=10),
            _variant("BRAF", "benign", pos=20),
            _variant("BRAF", "benign", pos=30),
        ]
        indices = [100, 200, 300]
        wt_sequences = ["WT10", "WT20", "WT30"]
        mut_sequences = ["MUT10", "MUT20", "MUT30"]
        positions = [10, 20, 30]

        output = _rebalance_after_filter(
            indices, valid, wt_sequences, mut_sequences, positions
        )
        output_indices, variants, output_wt, output_mut, output_positions, _ = output

        assert len(output_indices) == len(variants) == 2
        assert output_indices == [100, 200]
        assert output_positions == [10, 20]
        assert output_wt == ["WT10", "WT20"]
        assert output_mut == ["MUT10", "MUT20"]


def _expected_selection(input_fingerprint="input-fingerprint"):
    variants = [
        _variant("BRAF", "pathogenic", pos=1),
        _variant("BRAF", "benign", pos=2),
    ]
    accounting = {
        "n_fetched_variants": 2,
        "filter_skips": {
            "no_uid": 0,
            "uid_not_in_seq_cache": 0,
            "apply_missense_none": 0,
        },
        "n_embeddable_before_rebalance": 2,
        "n_removed_by_postfilter_balance": 0,
        "n_single_class_genes_dropped_postfilter": 0,
        "n_scored_variants": 2,
        "realised_design": {
            "n_variants": 2,
            "n_genes": 1,
            "n_pathogenic": 1,
            "n_benign": 1,
            "per_gene_class_count_distribution": {"1": 1},
            "n_duplicate_substitution_keys": 0,
        },
    }
    return ExpectedPathogenicitySelection(
        valid_indices=[0, 1],
        variants=variants,
        wt_sequences=["AAAA", "AAAA"],
        mut_sequences=["VAAA", "AVAA"],
        positions=[0, 1],
        fingerprint="variant-fingerprint",
        embedding_input_fingerprint=input_fingerprint,
        accounting=accounting,
    )


def _write_embedding_cache(tmp_path, monkeypatch, expected, *, valid_indices=None):
    wt_path = tmp_path / "wt.npy"
    mut_path = tmp_path / "mut.npy"
    metadata_path = tmp_path / "meta.json"
    monkeypatch.setattr(control, "PATH_EMB_WT_MEAN", wt_path)
    monkeypatch.setattr(control, "PATH_EMB_MUT_MEAN", mut_path)
    monkeypatch.setattr(control, "PATH_EMB_META", metadata_path)

    wt = np.arange(6, dtype=float).reshape(2, 3)
    mut = wt + 1
    np.save(wt_path, wt)
    np.save(mut_path, mut)
    metadata = {
        "metadata_version": control._EMBEDDING_METADATA_VERSION,
        "valid_indices": expected.valid_indices if valid_indices is None else valid_indices,
        "n": expected.accounting["n_fetched_variants"],
        "n_valid": len(expected.variants),
        "fingerprint": expected.fingerprint,
        "embedding_input_fingerprint": expected.embedding_input_fingerprint,
        "model": control.ESM2_MODEL_650M,
        "fetch_variant_fingerprint": "fetch-fingerprint",
        "selection_accounting": expected.accounting,
        "embedding_fingerprint": embedding_fingerprint(wt, mut),
    }
    metadata_path.write_text(json.dumps(metadata))
    return wt, mut


class TestEmbeddingCacheValidation:
    def test_valid_cache_is_accepted(self, tmp_path, monkeypatch):
        expected = _expected_selection()
        wt, mut = _write_embedding_cache(tmp_path, monkeypatch, expected)

        loaded_wt, loaded_mut, _ = _validate_embedding_cache(
            expected,
            {"variant_fingerprint": "fetch-fingerprint"},
            control.ESM2_MODEL_650M,
        )

        assert np.array_equal(loaded_wt, wt)
        assert np.array_equal(loaded_mut, mut)

    def test_same_row_count_with_different_indices_is_rejected(
        self, tmp_path, monkeypatch
    ):
        expected = _expected_selection()
        _write_embedding_cache(
            tmp_path, monkeypatch, expected, valid_indices=[1, 0]
        )

        with pytest.raises(ValueError, match="selection contract"):
            _validate_embedding_cache(
                expected,
                {"variant_fingerprint": "fetch-fingerprint"},
                control.ESM2_MODEL_650M,
            )

    def test_changed_sequence_inputs_are_rejected(self, tmp_path, monkeypatch):
        cached_expected = _expected_selection("old-inputs")
        _write_embedding_cache(tmp_path, monkeypatch, cached_expected)
        current_expected = _expected_selection("new-inputs")

        with pytest.raises(ValueError, match="embedding_input_fingerprint"):
            _validate_embedding_cache(
                current_expected,
                {"variant_fingerprint": "fetch-fingerprint"},
                control.ESM2_MODEL_650M,
            )


class TestClaim2C:
    def _inference(self, ci):
        return {
            "point_estimate": 0.90,
            "ci": ci,
            "estimate_basis": "seed_0_mean_of_fold_aurocs",
            "resampling_unit": "pfam_family",
            "n_scored": 100,
            "n_excluded": 5,
        }

    def test_family_seed0_interval_adjudicates_claim(self):
        claim = _build_claim_2c(
            self._inference(
                {
                    "point": 0.90,
                    "ci_low": 0.87,
                    "ci_high": 0.93,
                    "ci_suppressed": False,
                }
            )
        )

        assert claim["split"] == "family"
        assert claim["seed"] == 0
        assert claim["threshold"] == CLAIM_2C_THRESHOLD
        assert claim["resampling_unit"] == "pfam_family"
        assert claim["verdict"] == "pass, established (CI excludes the threshold)"

    def test_missing_interval_does_not_adjudicate(self):
        claim = _build_claim_2c(self._inference(None))
        assert claim["verdict"] == "not adjudicated (CI unavailable)"


class TestProbeSeedParamsCompleteness:
    def test_pfam_change_changes_fingerprint(self):
        genes = ["BRAF", "TP53"]
        pfam_a = {"BRAF": "PF00069", "TP53": "PF00870"}
        pfam_b = {"BRAF": "PF00069", "TP53": "PF99999"}
        assert pfam_fingerprint(pfam_a, genes) != pfam_fingerprint(pfam_b, genes)

    def test_seed_params_cover_data_code_and_runtime(self, monkeypatch):
        monkeypatch.setattr(control, "_source_files_fingerprint", lambda: "source-fp")
        genes = np.array(["BRAF", "BRAF"])
        metadata = {
            "fingerprint": "variant-fp",
            "model": control.ESM2_MODEL_650M,
            "embedding_fingerprint": "embedding-fp",
        }
        fetch_metadata = {"variant_fingerprint": "fetch-fp"}

        params = _seed_params(
            0,
            True,
            1000,
            metadata,
            fetch_metadata,
            {"BRAF": "PF00069"},
            genes,
        )

        assert params["seed"] == 0
        assert params["variant_fingerprint"] == "variant-fp"
        assert params["embedding_fingerprint"] == "embedding-fp"
        assert params["analysis_source_fingerprint"] == "source-fp"
        assert params["runtime_versions"]["numpy"] == np.__version__
