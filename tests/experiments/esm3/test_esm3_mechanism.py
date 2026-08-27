"""
Tests for esm2_mech.experiments.esm3.esm3_mechanism phase 3.

The ESM-3 macro-F1 is compared against an ESM-2 floor read from the matched
nonlinear-probe result (MLP, delta_mean, family-split), which is produced by
run_mlp_probe_cv in utils/probes.py. Any divergence between the two arms —
a different fold-skip condition, a different metric aggregation — makes the
M1/M2/M3 decision rules compare unlike quantities. These tests pin the ESM-3
arm to the same shared runner.

Invariants:
- phase 3 has no private MLP CV loop (_run_mlp) that can drift from the shared one
- phase 3 obtains its MLP metrics from run_mlp_probe_cv, so the reported
  mlp_f1_mean is that runner's per-fold mean, not a separately pooled F1
- phase 3 hands every fold of every split to the runner and drops none itself
- the logistic arm skips a fold only when train has < 2 classes
"""

import json

import numpy as np
import pytest

from esm2_mech.utils.constants import DN, GOF, LOF, MECHANISM_CLASSES

MODULE = "esm2_mech.experiments.esm3.esm3_mechanism"


def _synthetic_dataset(n_genes: int = 30, per_gene: int = 6, seed: int = 0):
    """Row-aligned variants/genes/pfam_map with all three mechanism classes."""
    rng = np.random.RandomState(seed)
    variants, genes = [], []
    classes = [GOF, DN, LOF]
    for gene_i in range(n_genes):
        gene = f"G{gene_i}"
        for row_i in range(per_gene):
            variants.append(
                {
                    "gene": gene,
                    "uniprot_id": f"U{gene_i}",
                    "mech3": classes[(gene_i + row_i) % len(classes)],
                }
            )
            genes.append(gene)
    n = len(variants)
    label_idx = np.array([MECHANISM_CLASSES.index(v["mech3"]) for v in variants])
    delta = rng.randn(n, 8) + label_idx[:, None] * 2.0
    pfam_map = {f"G{i}": f"PF{i % 5}" for i in range(n_genes)}
    return variants, np.array(genes), pfam_map, delta


@pytest.fixture
def phase3(monkeypatch, tmp_path):
    """Configure the module for a self-contained phase-3 run on synthetic data.

    Returns (module, delta) with the embedding files written and the ESM-2 floor
    pinned, so a test only has to stub the probe runner and call phase3_probes.
    """
    import importlib

    mod = importlib.import_module(MODULE)

    variants, genes, pfam_map, delta = _synthetic_dataset()
    monkeypatch.setattr(mod, "load_dataset", lambda: (variants, genes, pfam_map))
    monkeypatch.setattr(mod, "esm2_family_floor", lambda seeds: (0.3, "test"))

    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    out_dir = tmp_path / "out"
    seq_path = emb_dir / "seq_mean.npy"
    np.save(seq_path, delta)
    monkeypatch.setattr(mod, "EMB_SEQ", seq_path)
    # Only the seq condition exists, so seq_struct is skipped as a missing file.
    monkeypatch.setattr(mod, "EMB_SEQ_STRUCT", emb_dir / "seq_struct_mean.npy")
    monkeypatch.setattr(mod, "EMB_VALID_IDX", emb_dir / "valid_idx.npy")
    monkeypatch.setattr(mod, "STRUCT_META", emb_dir / "struct_meta.json")
    monkeypatch.setattr(mod, "OUT", out_dir)
    return mod, delta


def _summary(mod):
    with open(mod.OUT / "summary.json") as fh:
        return json.load(fh)


def test_no_private_mlp_loop():
    # A hand-copied CV loop is what let the ESM-3 arm drift from the ESM-2 floor
    # it is compared against (fold-skip at < 3 classes, pooled instead of
    # per-fold F1). The arm must use the shared runner, not its own copy.
    import importlib

    mod = importlib.import_module(MODULE)
    assert not hasattr(mod, "_run_mlp")
    assert hasattr(mod, "run_mlp_probe_cv")


def test_mlp_metrics_come_from_shared_runner(phase3, monkeypatch):
    # Stub the shared runner: whatever it reports must be what lands in the
    # summary. If phase 3 computes its own pooled F1, the stub value is ignored.
    mod, _delta = phase3
    agg = {
        "macro_f1_mean": 0.4242,
        "macro_f1_std": 0.01,
        f"auroc_{GOF}_mean": 0.61,
        f"auroc_{DN}_mean": 0.62,
        f"auroc_{LOF}_mean": 0.63,
        "n_folds": 5,
    }

    def stub(X, labels, splits, *args, **kwargs):
        return (agg, None) if kwargs.get("return_oof") else agg

    monkeypatch.setattr(mod, "run_mlp_probe_cv", stub)
    mod.phase3_probes(seeds=[0], compute_ci=False)

    family = _summary(mod)["results"]["seq"]["family_split"]
    assert family["mlp_f1_mean"] == pytest.approx(0.4242)
    assert family["mlp_gof_auroc_mean"] == pytest.approx(0.61)
    assert family["mlp_dn_auroc_mean"] == pytest.approx(0.62)
    assert family["mlp_lof_auroc_mean"] == pytest.approx(0.63)


def test_every_fold_is_handed_to_the_runner(phase3, monkeypatch):
    # Phase 3 must not filter folds itself — fold selection belongs to the shared
    # runner, so both arms drop the same folds.
    mod, _delta = phase3
    seen = []

    def stub(X, labels, splits, *args, **kwargs):
        seen.append(len(splits))
        agg = {"macro_f1_mean": 0.5, "macro_f1_std": 0.0, "n_folds": len(splits)}
        return (agg, None) if kwargs.get("return_oof") else agg

    monkeypatch.setattr(mod, "run_mlp_probe_cv", stub)
    mod.phase3_probes(seeds=[0], compute_ci=False)

    assert seen  # the runner was called
    assert all(n_folds == mod.N_FOLDS for n_folds in seen)


def test_logistic_arm_rejects_a_class_incomplete_split_before_fitting():
    import importlib

    mod = importlib.import_module(MODULE)
    rng = np.random.RandomState(0)
    n = 180
    labels = np.array([GOF, DN] * 80 + [LOF] * 20)
    X = rng.randn(n, 8)

    lof_rows = np.where(labels == LOF)[0]
    extra = np.where(labels != LOF)[0][:20]
    test_idx = np.concatenate([lof_rows, extra])
    train_idx = np.setdiff1d(np.arange(n), test_idx)
    from esm2_mech.utils.classification import validate_complete_classification_splits

    splits = [(train_idx, test_idx)]
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=1,
        eligible_rows=test_idx,
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=np.arange(n),
        held_out_unit="gene",
    )
    assert contract["status"] == "unscorable"
    assert mod._run_logreg_folds(X, labels, splits, contract, seed=0) is None
