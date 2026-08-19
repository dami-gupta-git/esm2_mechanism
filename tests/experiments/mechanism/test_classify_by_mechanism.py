"""
Tests for esm2_mech.experiments.mechanism.classify_by_mechanism.

Invariants:
- valid_variants is read straight from VALID_VARIANTS_JSON (no re-derived filter)
- when every embedding array's row count matches len(valid_variants), load_data
  returns arrays/lists all aligned to that length
- a row-count mismatch on ANY of the four embedding arrays raises ValueError
  naming that array, rather than silently indexing past a misaligned pairing
- embedding rows are rejected when their identity sidecar has the same row count
  but a different variant order
- the five-seed permutation distribution is preserved in the aggregate output
- the preregistered three-of-five rule is not evaluated from incomplete results
"""

import json

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import classify_by_mechanism
from esm2_mech.utils.constants import GOF, LOF


def _write_variants(path, n):
    variants = [
        {
            "uniprot_id": f"P{i:05d}",
            "aa_pos": i + 1,
            "aa_wt": "A",
            "aa_mut": "G",
            "gene": f"GENE{i}",
            "label_3class": GOF if i % 2 == 0 else LOF,
            "mechanism": "GOF" if i % 2 == 0 else "LOF",
            "foldx_ddg": None,
        }
        for i in range(n)
    ]
    with open(path, "w") as f:
        json.dump(variants, f)
    return variants


def _write_embeddings(path, n, dim=4):
    np.save(path, np.zeros((n, dim), dtype=np.float32))


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Point classify_by_mechanism's path constants at a scratch directory."""
    valid_variants_json = tmp_path / "valid_variants.json"
    embedded_variants_json = tmp_path / "embedded_variants.json"
    emb_paths = {
        name: tmp_path / f"{name}.npy"
        for name in ("EMB_WT_MEAN", "EMB_MUT_MEAN", "EMB_WT_POS", "EMB_MUT_POS")
    }
    monkeypatch.setattr(classify_by_mechanism, "VALID_VARIANTS_JSON", valid_variants_json)
    monkeypatch.setattr(
        classify_by_mechanism, "EMB_VALID_VARIANTS_JSON", embedded_variants_json
    )
    for name, path in emb_paths.items():
        monkeypatch.setattr(classify_by_mechanism, name, path)
    monkeypatch.setattr(
        classify_by_mechanism, "_load_alphamissense_scores",
        lambda variants: np.full(len(variants), np.nan),
    )
    return valid_variants_json, embedded_variants_json, emb_paths


class TestLoadData:

    def test_aligned_embeddings_loads_successfully(self, patched_paths):
        valid_variants_json, embedded_variants_json, emb_paths = patched_paths
        n = 6
        variants = _write_variants(valid_variants_json, n)
        embedded_variants_json.write_text(json.dumps(variants))
        for path in emb_paths.values():
            _write_embeddings(path, n)

        data = classify_by_mechanism.load_data()

        assert len(data["valid_variants"]) == n
        assert data["emb_wt_mean"].shape == (n, 4)
        assert data["emb_mut_mean"].shape == (n, 4)
        assert data["emb_wt_pos"].shape == (n, 4)
        assert data["emb_mut_pos"].shape == (n, 4)
        assert data["labels_3class"].shape == (n,)
        assert list(data["labels_3class"]) == [v["label_3class"] for v in variants]
        assert list(data["genes_arr"]) == [v["gene"] for v in variants]

    @pytest.mark.parametrize(
        "mismatched_name", ["EMB_WT_MEAN", "EMB_MUT_MEAN", "EMB_WT_POS", "EMB_MUT_POS"]
    )
    def test_row_mismatch_raises(self, patched_paths, mismatched_name):
        valid_variants_json, embedded_variants_json, emb_paths = patched_paths
        n = 6
        variants = _write_variants(valid_variants_json, n)
        embedded_variants_json.write_text(json.dumps(variants))
        for name, path in emb_paths.items():
            # One array gets fewer rows than the variant list -> misaligned.
            rows = n - 1 if name == mismatched_name else n
            _write_embeddings(path, rows)

        with pytest.raises(ValueError, match=mismatched_name):
            classify_by_mechanism.load_data()

    def test_same_count_reordered_variants_raise(self, patched_paths):
        valid_variants_json, embedded_variants_json, emb_paths = patched_paths
        variants = _write_variants(valid_variants_json, 6)
        embedded_variants_json.write_text(json.dumps(variants))
        valid_variants_json.write_text(json.dumps(list(reversed(variants))))
        for path in emb_paths.values():
            _write_embeddings(path, len(variants))

        with pytest.raises(ValueError, match="row order"):
            classify_by_mechanism.load_data()


def _permutation_seed_result(seed, wt_p, delta_p, *, include_wt=True):
    family_split = {
        "delta_mean": {
            "permutation": {
                "p_value": delta_p,
                "resolution_limited": seed == 0,
                "n_clusters_immovable": 3,
            }
        }
    }
    if include_wt:
        family_split["wt_only_mean"] = {
            "permutation": {
                "p_value": wt_p,
                "resolution_limited": False,
                "n_clusters_immovable": 3,
            }
        }
    return seed, f"seed{seed}.json", {"family_split": family_split}


class TestAggregatePermutationResults:

    def test_preserves_distribution_and_evaluates_three_of_five(self):
        wt_p_values = [0.01, 0.02, 0.03, 0.20, 0.40]
        delta_p_values = [0.001, 0.10, 0.20, 0.30, 0.40]
        seed_results = [
            _permutation_seed_result(seed, wt_p_values[seed], delta_p_values[seed])
            for seed in range(5)
        ]

        summary = classify_by_mechanism.aggregate_permutation_results(seed_results)

        wt_summary = summary["wt_only_mean"]
        assert [row["p_value"] for row in wt_summary["per_seed"]] == wt_p_values
        assert wt_summary["n_below_significance_threshold"] == 3
        assert wt_summary["preregistered_rule_evaluable"] is True
        assert wt_summary["meets_preregistered_three_of_five_rule"] is True

        delta_summary = summary["delta_mean"]
        assert delta_summary["resolution_limited_seeds"] == [0]
        assert delta_summary["n_below_significance_threshold"] == 1
        assert delta_summary["meets_preregistered_three_of_five_rule"] is False
        assert delta_summary["per_seed"][0]["n_clusters_immovable"] == 3

    def test_fewer_than_five_seeds_is_not_a_negative_result(self):
        seed_results = [
            _permutation_seed_result(seed, 0.01, 0.01)
            for seed in range(2)
        ]

        summary = classify_by_mechanism.aggregate_permutation_results(seed_results)

        assert summary["wt_only_mean"]["preregistered_rule_evaluable"] is False
        assert summary["wt_only_mean"]["meets_preregistered_three_of_five_rule"] is None

    def test_missing_feature_or_p_value_is_reported_and_not_evaluated(self):
        seed_results = [
            _permutation_seed_result(
                seed,
                None if seed == 3 else 0.01,
                0.01,
                include_wt=seed != 4,
            )
            for seed in range(5)
        ]

        summary = classify_by_mechanism.aggregate_permutation_results(seed_results)
        wt_summary = summary["wt_only_mean"]

        assert wt_summary["missing_seeds"] == [4]
        assert wt_summary["seeds_without_valid_p_value"] == [3]
        assert wt_summary["preregistered_rule_evaluable"] is False
        assert wt_summary["meets_preregistered_three_of_five_rule"] is None
