"""
Pathogenicity positive control — phase 3: run probes.

Loads pathogenicity embeddings produced by embed_pathogenicity.py and runs
linear + MLP probes under gene-split and family-split CV to validate that the
ESM-2 pipeline can recover known pathogenicity signal.

Purpose: if AUROC >> 0.85 on gene-split, the pipeline is sound and a mechanism
null result is interpretable as a real absence of signal, not a pipeline failure.

Inputs:
  data/clinvar_pathogenicity_variants.json  (from pathogenicity_fetch.py)
  data/embeddings/esm2_t33_650M_UR50D/pathogenicity_wt_mean.npy  (from embed_pathogenicity.py)
  data/embeddings/esm2_t33_650M_UR50D/pathogenicity_mut_mean.npy
  data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json

Output: results/pathogenicity_control.json

Usage:
    python -m esm2_mech.experiments.pathogenicity.pathogenicity_probes
"""

import argparse
import functools
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.paths import (
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    PATH_EMB_META,
    PATH_EMB_MUT_MEAN,
    PATH_EMB_WT_MEAN,
    RESULTS_DIR,
    PFAM_JSON,
)
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)


def load_embeddings(variants):
    """Load pathogenicity embeddings. Raises FileNotFoundError if not found."""
    for path in [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file missing: {path}\n"
                "Run: python -m esm2_mech.embeddings.embed_pathogenicity"
            )
    try:
        with open(PATH_EMB_META) as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise FileNotFoundError(
            f"Corrupt embedding metadata {PATH_EMB_META} — delete it and re-run: "
            "python -m esm2_mech.embeddings.embed_pathogenicity"
        ) from exc

    print(f"  Loading pathogenicity embeddings: {PATH_EMB_WT_MEAN}")
    return (
        np.load(PATH_EMB_WT_MEAN),
        np.load(PATH_EMB_MUT_MEAN),
        [variants[i] for i in meta["valid_indices"]],
    )


def run_binary_probe(X, y, splits, probe_kind, seed=42):
    """probe_kind: 'logreg' or 'mlp'. y is binary 0/1."""
    aurocs, prs, f1s = [], [], []
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        ytr, yte = y[tr], y[te]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        if probe_kind == "logreg":
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(256,), max_iter=300, random_state=seed,
                early_stopping=True, validation_fraction=0.1,
            )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        pred = clf.predict(Xte)
        aurocs.append(float(roc_auc_score(yte, proba)))
        p_, r_, _ = precision_recall_curve(yte, proba)
        prs.append(float(auc(r_, p_)))
        f1s.append(float(f1_score(yte, pred, average="binary", zero_division=0)))
    if not aurocs:
        return {"note": "no usable folds"}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "pr_auc_mean": float(np.mean(prs)),
        "f1_mean": float(np.mean(f1s)),
        "n_folds": len(aurocs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(RESULTS_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"Phase 1 output not found: {CLINVAR_PATHOGENICITY_VARIANTS_JSON}\n"
            "Run first: python -m esm2_mech.experiments.pathogenicity.pathogenicity_fetch"
        )
    with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
        variants = json.load(f)

    wt_mean, mut_mean, valid = load_embeddings(variants)
    deltas = mut_mean - wt_mean
    print(f"  {len(valid)} embedded variants  ({Counter(v['label'] for v in valid)})")

    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in valid])
    genes = np.array([v["gene"] for v in valid])

    if not PFAM_JSON.exists():
        raise FileNotFoundError(f"{PFAM_JSON} not found — run fetch_data/fetch_annotations --step pfam first")
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    gs = gene_split_cv(genes, seed=args.seed)
    fs = family_split_cv(genes, pfam_map, seed=args.seed)
    print(f"  Gene-split folds: {len(gs)}  Family-split folds: {len(fs)}")

    features = {"delta_mean": deltas, "wt_only": wt_mean}
    results = {
        "n_variants": int(len(valid)),
        "n_genes": int(len(set(genes))),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((1 - y).sum()),
        "n_families": int(len({pfam_map.get(g) for g in genes} - {None})),
        "by_feature": {},
    }

    for fname, X in features.items():
        print(f"\n  --- {fname} ---")
        block = {}
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            for probe in ("logreg", "mlp"):
                if not splits:
                    continue
                r = run_binary_probe(X, y, splits, probe_kind=probe, seed=args.seed)
                key = f"{split_name}_{probe}"
                block[key] = r
                if "auroc_mean" in r:
                    print(
                        f"    {key:25s}  AUROC={r['auroc_mean']:.3f}  "
                        f"PR-AUC={r['pr_auc_mean']:.3f}  "
                        f"F1={r['f1_mean']:.3f}  (folds={r['n_folds']})"
                    )
        results["by_feature"][fname] = block

    import pathlib
    out_path = pathlib.Path(args.out_dir) / "pathogenicity_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    print("HEADLINE — pathogenicity positive control")
    print("=" * 60)

    def aur(k1, k2):
        return results["by_feature"][k1][k2].get("auroc_mean", float("nan"))

    print(f"delta_mean  gene-split   logreg AUROC: {aur('delta_mean','gene_split_logreg'):.3f}")
    print(f"delta_mean  gene-split   MLP    AUROC: {aur('delta_mean','gene_split_mlp'):.3f}")
    print(f"delta_mean  family-split logreg AUROC: {aur('delta_mean','family_split_logreg'):.3f}")
    print(f"delta_mean  family-split MLP    AUROC: {aur('delta_mean','family_split_mlp'):.3f}")
    print()
    print(f"wt_only     gene-split   logreg AUROC: {aur('wt_only','gene_split_logreg'):.3f}")
    print(f"wt_only     family-split logreg AUROC: {aur('wt_only','family_split_logreg'):.3f}")
    print()
    delta_gs = aur("delta_mean", "gene_split_logreg")
    if delta_gs >= 0.85:
        print("  ⇒ Pipeline PASSES positive control (delta AUROC ≥ 0.85 on gene-split).")
        print("    The mechanism null result is interpretable as a real absence of signal,")
        print("    not a pipeline failure.")
    elif delta_gs >= 0.70:
        print("  ⇒ Pipeline shows MODERATE pathogenicity signal (delta AUROC 0.70-0.85).")
        print("    The mechanism null is interpretable but weaker than published ESM-2 work.")
    else:
        print("  ⇒ Pipeline FAILS positive control (delta AUROC < 0.70).")
        print("    The mechanism null result is UNINTERPRETABLE — the pipeline cannot")
        print("    recover even known signal. Debug before drawing conclusions.")


if __name__ == "__main__":
    main()
