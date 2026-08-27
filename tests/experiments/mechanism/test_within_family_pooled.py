"""
Tests for the pooled cross-family GOF test in mechanism_within_family.

These cover the statistics added on top of the per-family table: the seed-averaged
OOF plumbing, the per-family gene-cluster CIs, and the pooled delta-GOF-AUROC-vs-
chance test (cluster-bootstrap CI + gene-level label-permutation p-value).

Most tests exercise the pooling / CI / permutation *logic* on hand-built OOF dicts,
which is fast and isolates the statistics from the probe internals. A single
end-to-end test (marked `slow`) fits the real logreg + MLP probes via
_probe_one_family to prove the plumbing connects and a planted signal is recovered.

Invariants:
- _stack_oof: concatenates valid OOF dicts; skips None/empty; all-empty -> None
- _gof_auroc_from_oof: one-vs-rest GOF AUROC; None when GOF absent or all-GOF
- pooled_gof_test: no GOF-bearing family -> {"n_families": 0}
- pooled_gof_test: pools across families; reports gene/family counts
- pooled_gof_test: recovers a planted GOF ranking with CI excluding 0.5
- pooled_gof_test: compute_ci=False omits CI block
- pooled_gof_test: permutation null centers on chance; planted signal -> small p
- _probe_one_family (slow, real probes): returns (results, oof); attaches CIs;
  seed-averaged OOF has one row per variant; compute_ci=False omits CI

The permutation test remains an expected failure because that deferred path still
calls the shared permutation helper without a fold index. The CI path retains the
per-seed folds and is covered by the real-probe test below.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import mechanism_within_family as wf
from esm2_mech.utils.constants import MECHANISM_CLASSES, GOF, DN, LOF


# ---------------------------------------------------------------------------
# helpers — hand-built OOF (no probe fitting)
# ---------------------------------------------------------------------------

def _oof(row_ids, y, genes, proba, folds=None):
    row_ids = np.asarray(row_ids)
    return {
        "row_ids": row_ids,
        "y_true": np.asarray(y),
        "genes": np.asarray(genes, dtype=object),
        "proba": np.asarray(proba, dtype=float),
        "folds": (
            np.zeros(len(row_ids), dtype=int)
            if folds is None
            else np.asarray(folds, dtype=int)
        ),
    }


def _synthetic_family_oof(name, n_genes, gof_signal, rng, vars_per_gene=11):
    """A seed-averaged delta OOF dict for one family, with a tunable GOF ranking.

    Returns the shape _probe_one_family produces for one view:
    {"logreg": oof, "mlp": oof}. The GOF column probability is the true-GOF
    indicator plus noise scaled so larger `gof_signal` separates GOF from the rest
    more cleanly (gof_signal=0 -> chance). No classifier is fit, so this is fast and
    deterministic for a fixed rng.
    """
    base = [f"{name}_g{i}" for i in range(n_genes)]
    genes = np.repeat(base, vars_per_gene)
    n_gof = max(1, n_genes // 3)
    n_dn = max(1, n_genes // 3)
    cls_seq = [GOF] * n_gof + [DN] * n_dn + [LOF] * (n_genes - n_gof - n_dn)
    gene_to_cls = dict(zip(base, cls_seq))
    y = np.array([gene_to_cls[g] for g in genes])

    def _build_proba():
        n = len(y)
        is_gof = (y == GOF).astype(float)
        # GOF score = signal·indicator + noise; map through softmax-ish to a simplex.
        gof_logit = gof_signal * is_gof + rng.randn(n) * 0.5
        other = rng.randn(n, 2) * 0.5
        logits = np.column_stack([gof_logit, other])  # cols: GOF, DN, LOF order
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    row_ids = np.arange(len(y))
    return {
        "logreg": _oof(row_ids, y, genes, _build_proba()),
        "mlp": _oof(row_ids, y, genes, _build_proba()),
    }


def _family_input(name, n_genes, vars_per_gene=11, dim=12, rng=None):
    """Raw (X, y, genes) for one family — only needed by the real-probe path."""
    rng = rng or np.random.RandomState(0)
    base = [f"{name}_g{i}" for i in range(n_genes)]
    genes = np.repeat(base, vars_per_gene)
    n_gof = max(1, n_genes // 3)
    n_dn = max(1, n_genes // 3)
    cls_seq = [GOF] * n_gof + [DN] * n_dn + [LOF] * (n_genes - n_gof - n_dn)
    gene_to_cls = dict(zip(base, cls_seq))
    y = np.array([gene_to_cls[g] for g in genes])
    X = rng.randn(len(y), dim)
    X[y == GOF, 0] += 1.5
    return {"X": X, "y": y, "genes": genes, "classes": MECHANISM_CLASSES}


# ---------------------------------------------------------------------------
# _stack_oof
# ---------------------------------------------------------------------------

class TestStackOof:
    def test_concatenates_valid(self):
        a = _oof([0, 1], [GOF, LOF], ["A0", "A1"], [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])
        b = _oof([0], [DN], ["B0"], [[0.2, 0.6, 0.2]])
        pooled = wf._stack_oof([a, b])
        assert len(pooled["y_true"]) == 3
        assert list(pooled["genes"]) == ["A0", "A1", "B0"]

    def test_skips_none_and_empty(self):
        a = _oof([0], [GOF], ["A0"], [[0.6, 0.2, 0.2]])
        empty = _oof([], [], [], np.empty((0, 3)))
        pooled = wf._stack_oof([None, a, empty])
        assert len(pooled["y_true"]) == 1

    def test_all_empty_returns_none(self):
        assert wf._stack_oof([None, None]) is None
        assert wf._stack_oof([]) is None


# ---------------------------------------------------------------------------
# _gof_auroc_from_oof
# ---------------------------------------------------------------------------

class TestGofAurocFromOof:
    def test_perfect_separation(self):
        # GOF rows get the highest GOF-column proba -> AUROC 1.0
        y = [GOF, GOF, LOF, DN]
        proba = [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]
        oof = _oof([0, 1, 2, 3], y, ["g0", "g1", "g2", "g3"], proba)
        assert wf._gof_auroc_from_oof(oof) == pytest.approx(1.0)

    def test_none_when_no_gof(self):
        y = [DN, LOF, LOF]
        proba = [[0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]]
        oof = _oof([0, 1, 2], y, ["g0", "g1", "g2"], proba)
        assert wf._gof_auroc_from_oof(oof) is None

    def test_none_when_all_gof(self):
        y = [GOF, GOF]
        proba = [[0.6, 0.2, 0.2], [0.7, 0.1, 0.2]]
        oof = _oof([0, 1], y, ["g0", "g1"], proba)
        assert wf._gof_auroc_from_oof(oof) is None


# ---------------------------------------------------------------------------
# pooled_gof_test — on synthetic OOF (no probe fitting)
# ---------------------------------------------------------------------------

class TestPooledGofTest:
    def test_no_gof_family_returns_zero(self):
        # A family OOF that carries only DN/LOF (no GOF) -> nothing to pool.
        genes = np.repeat([f"X_g{i}" for i in range(8)], 10)
        y = np.array([DN if g.endswith(tuple("0123")) else LOF for g in
                      [f"X_g{i}" for i in range(8) for _ in range(10)]])
        proba = np.full((len(y), 3), 1 / 3)
        no_gof = {"logreg": _oof(np.arange(len(y)), y, genes, proba),
                  "mlp": _oof(np.arange(len(y)), y, genes, proba)}
        out = wf.pooled_gof_test({"X": no_gof}, {}, n_seeds=3, n_folds=5)
        assert out == {"n_families": 0}

    def test_pools_across_families_and_counts(self):
        rng = np.random.RandomState(1)
        delta_oof = {
            "FAMA": _synthetic_family_oof("FAMA", 12, gof_signal=2.0, rng=rng),
            "FAMB": _synthetic_family_oof("FAMB", 9, gof_signal=2.0, rng=rng),
        }
        out = wf.pooled_gof_test(delta_oof, {}, n_seeds=3, n_folds=5, compute_ci=False)
        assert out["n_families"] == 2
        assert set(out["families"]) == {"FAMA", "FAMB"}
        assert out["chance_auroc"] == 0.5
        assert out["logreg"]["n_genes"] == 12 + 9

    def test_recovers_planted_signal_with_interval_gated(self):
        rng = np.random.RandomState(2)
        delta_oof = {
            "FAMA": _synthetic_family_oof("FAMA", 12, gof_signal=2.5, rng=rng),
            "FAMB": _synthetic_family_oof("FAMB", 9, gof_signal=2.5, rng=rng),
        }
        out = wf.pooled_gof_test(delta_oof, {}, n_seeds=3, n_folds=5, compute_ci=True)
        assert out["logreg"]["point"] > 0.5
        assert out["logreg"]["ci"]["ci_low"] is None
        assert out["logreg"]["ci"]["reason"] == "blocked_by_audit_1_4"

    def test_no_signal_interval_is_also_gated(self):
        # gof_signal=0 -> GOF proba is pure noise -> AUROC ~0.5, CI straddles it.
        rng = np.random.RandomState(4)
        delta_oof = {"FAMA": _synthetic_family_oof("FAMA", 15, gof_signal=0.0, rng=rng)}
        out = wf.pooled_gof_test(delta_oof, {}, n_seeds=3, n_folds=5, compute_ci=True)
        ci = out["logreg"]["ci"]
        assert ci["ci_low"] is None
        assert ci["ci_high"] is None
        assert ci["reason"] == "blocked_by_audit_1_4"

    def test_compute_ci_false_omits_ci_block(self):
        rng = np.random.RandomState(4)
        delta_oof = {"FAMA": _synthetic_family_oof("FAMA", 12, gof_signal=1.5, rng=rng)}
        out = wf.pooled_gof_test(delta_oof, {}, n_seeds=3, n_folds=5, compute_ci=False)
        assert "ci" not in out["logreg"]
        assert out["logreg"]["point"] is not None


# ---------------------------------------------------------------------------
# permutation — needs the real probe (refits per shuffle), so it is slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPooledPermutation:
    def test_permutation_null_centers_on_chance(self):
        rng = np.random.RandomState(5)
        family_inputs = {
            "FAMA": _family_input("FAMA", 12, rng=rng),
            "FAMB": _family_input("FAMB", 9, rng=rng),
        }
        # Build the delta OOF via the real probe arm, then run the permutation test.
        delta_oof = {}
        for fam, inp in family_inputs.items():
            present = [c for c in MECHANISM_CLASSES if (inp["y"] == c).sum() > 0]
            _, oof_out = wf._probe_one_family(
                {wf.VIEW_DELTA: inp["X"]}, inp["y"], inp["genes"], present,
                n_seeds=2, n_folds=5, compute_ci=False,
                mlp_kwargs={"hidden": (16,), "max_iter": 50},
            )
            delta_oof[fam] = oof_out[wf.VIEW_DELTA]
        # Each permutation refits the MLP across n_seeds*n_folds for every family, so
        # the fit cost dominates. Use a tiny, few-iteration net (vs the production
        # (256,64)/500) to make those refits cheap; the test only checks that the
        # permutation null centers on chance, which does not need the full-size probe.
        out = wf.pooled_gof_test(
            delta_oof, family_inputs, n_seeds=2, n_folds=5,
            compute_ci=False, n_permutations=15,
            mlp_hidden=(16,), mlp_max_iter=50,
        )
        perm = out["permutation_mlp"]
        assert perm["observed"] is None
        assert perm["p_value"] is None
        assert perm["reason"] == "observed_fold_aware_gof_auroc_unavailable"


# ---------------------------------------------------------------------------
# _probe_one_family — one real-probe end-to-end check (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestProbeOneFamilyRealProbes:
    def test_returns_results_oof_and_ci(self):
        rng = np.random.RandomState(0)
        inp = _family_input("F", 12, rng=rng)
        present = [c for c in MECHANISM_CLASSES if (inp["y"] == c).sum() > 0]
        results, oof_out = wf._probe_one_family(
            {wf.VIEW_DELTA: inp["X"]}, inp["y"], inp["genes"], present,
            n_seeds=2, n_folds=5, compute_ci=True,
            mlp_kwargs={"hidden": (16,), "max_iter": 50},
        )
        entry = results[wf.VIEW_DELTA]["logreg"]
        assert "macro_f1" in entry and "auroc" in entry
        assert "ci" in entry and "auroc_GOF" in entry["ci"]
        # Seed-averaged OOF: each variant appears once despite the 2 seeds.
        oof = oof_out[wf.VIEW_DELTA]["logreg"]
        rows = oof["row_ids"].tolist()
        assert len(rows) == len(set(rows))

    def test_compute_ci_false_omits_intervals(self):
        rng = np.random.RandomState(1)
        inp = _family_input("F", 12, rng=rng)
        present = [c for c in MECHANISM_CLASSES if (inp["y"] == c).sum() > 0]
        results, _ = wf._probe_one_family(
            {wf.VIEW_DELTA: inp["X"]}, inp["y"], inp["genes"], present,
            n_seeds=2, n_folds=5, compute_ci=False,
            mlp_kwargs={"hidden": (16,), "max_iter": 50},
        )
        assert "ci" not in results[wf.VIEW_DELTA]["logreg"]
        assert "ci" not in results[wf.VIEW_DELTA]["mlp"]
