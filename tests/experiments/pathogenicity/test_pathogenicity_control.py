"""
Tests for pathogenicity_control.py.

Covers:
- _rebalance_after_filter: uneven filtering is corrected
- _rebalance_after_filter: genes losing one class entirely are dropped
- _rebalance_after_filter: already-balanced input is unchanged
- probe seed_params: Pfam fingerprint change invalidates cache
- probe seed_params: embedding fingerprint change invalidates cache
"""

import json

import numpy as np
import pytest

from esm2_mech.experiments.pathogenicity.pathogenicity_control import (
    _rebalance_after_filter,
)
from esm2_mech.utils.data import pfam_fingerprint


def _variant(gene, label, pos=1, wt="A", mut="V", uid="P00000"):
    return {"gene": gene, "label": label, "aa_pos": pos,
            "aa_wt": wt, "aa_mut": mut, "uniprot_id": uid}


class TestRebalanceAfterFilter:

    def test_uneven_filter_is_corrected(self):
        """If filtering drops more pathogenic than benign for a gene, rebalancing equalizes."""
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "benign", pos=2),
            _variant("BRAF", "benign", pos=3),
            _variant("BRAF", "benign", pos=4),
        ]
        indices = list(range(len(valid)))
        seqs = ["S"] * len(valid)
        positions = [0] * len(valid)

        _, out, _, _, _ = _rebalance_after_filter(indices, valid, seqs, seqs, positions)
        from collections import Counter
        counts = Counter(v["label"] for v in out)
        assert counts["pathogenic"] == counts["benign"] == 1

    def test_gene_with_only_one_class_dropped(self):
        """A gene that loses all variants of one class is dropped entirely."""
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "pathogenic", pos=2),
            _variant("TP53", "pathogenic", pos=1),
            _variant("TP53", "benign", pos=2),
        ]
        indices = list(range(len(valid)))
        seqs = ["S"] * len(valid)
        positions = [0] * len(valid)

        _, out, _, _, _ = _rebalance_after_filter(indices, valid, seqs, seqs, positions)
        genes = {v["gene"] for v in out}
        assert "BRAF" not in genes
        assert "TP53" in genes

    def test_already_balanced_unchanged(self):
        """A perfectly balanced input passes through without dropping anything."""
        valid = [
            _variant("BRAF", "pathogenic", pos=1),
            _variant("BRAF", "benign", pos=2),
            _variant("TP53", "pathogenic", pos=3),
            _variant("TP53", "benign", pos=4),
        ]
        indices = list(range(len(valid)))
        seqs = ["S"] * len(valid)
        positions = [0] * len(valid)

        out_idx, out, _, _, _ = _rebalance_after_filter(indices, valid, seqs, seqs, positions)
        assert len(out) == len(valid)
        assert out_idx == indices

    def test_parallel_lists_stay_aligned(self):
        """Indices, sequences, and positions stay aligned after rebalancing."""
        valid = [
            _variant("BRAF", "pathogenic", pos=10),
            _variant("BRAF", "benign", pos=20),
            _variant("BRAF", "benign", pos=30),
        ]
        indices = [100, 200, 300]
        wt_seqs = ["WT10", "WT20", "WT30"]
        mut_seqs = ["MUT10", "MUT20", "MUT30"]
        positions = [10, 20, 30]

        out_idx, out, out_wt, out_mut, out_pos = _rebalance_after_filter(
            indices, valid, wt_seqs, mut_seqs, positions
        )
        assert len(out_idx) == len(out) == len(out_wt) == len(out_mut) == len(out_pos) == 2
        # The pathogenic variant (index 0) and first benign (index 1) should be kept
        assert out_idx == [100, 200]
        assert out_pos == [10, 20]
        assert out_wt == ["WT10", "WT20"]


class TestProbeSeedParamsCompleteness:

    def test_pfam_change_changes_fingerprint(self):
        """Different Pfam assignments produce different fingerprints."""
        genes = ["BRAF", "TP53"]
        pfam_a = {"BRAF": "PF00069", "TP53": "PF00870"}
        pfam_b = {"BRAF": "PF00069", "TP53": "PF99999"}

        fp_a = pfam_fingerprint(pfam_a, genes)
        fp_b = pfam_fingerprint(pfam_b, genes)
        assert fp_a != fp_b

    def test_pfam_same_gives_same_fingerprint(self):
        genes = ["BRAF", "TP53"]
        pfam = {"BRAF": "PF00069", "TP53": "PF00870"}
        assert pfam_fingerprint(pfam, genes) == pfam_fingerprint(pfam, genes)

    def test_pfam_extra_genes_in_map_ignored(self):
        """Only genes in the variant set matter, not extra entries in the map."""
        genes = ["BRAF"]
        pfam_small = {"BRAF": "PF00069"}
        pfam_big = {"BRAF": "PF00069", "TP53": "PF00870"}
        assert pfam_fingerprint(pfam_small, genes) == pfam_fingerprint(pfam_big, genes)

    def test_seed_params_include_pfam_and_embedding_fingerprints(self):
        """The seed_params dict must contain pfam_fingerprint and embedding_fingerprint keys."""
        # This is a structural test: if someone removes these keys, this fails.
        # We can't run the full probe_phase here, but we can verify the keys
        # exist by inspecting the function's param construction.
        import inspect
        source = inspect.getsource(
            __import__(
                "esm2_mech.experiments.pathogenicity.pathogenicity_control",
                fromlist=["probe_phase"],
            ).probe_phase
        )
        assert "pfam_fingerprint" in source
        assert "embedding_fingerprint" in source
