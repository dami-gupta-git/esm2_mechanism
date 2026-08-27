"""Perturbation pattern analysis.

Builds gene-level spatial features from per-residue delta embeddings and probes for mechanism signal.
"""

import functools, json, os, sys, numpy as np
from collections import Counter, defaultdict

from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
)

print = functools.partial(print, flush=True)
from esm2_mech.utils.splits import family_split_cv, gene_split_cv
from esm2_mech.utils.probes import run_logreg_cv
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.classification import validate_complete_classification_splits

DATA = str(_DATA_DIR)
OUT = str(_RESULTS_DIR / "perturbation_pattern")
os.makedirs(OUT, exist_ok=True)


def load_data():
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = (
                "LOF"
                if v.get("mechanism") in ("HI", "AR")
                else v.get("mechanism", "LOF")
            )

    wt_pos = np.load(EMB_WT_POS)
    mut_pos = np.load(EMB_MUT_POS)
    wt_mean = np.load(EMB_WT_MEAN)
    mut_mean = np.load(EMB_MUT_MEAN)

    delta_pos = mut_pos - wt_pos
    delta_mean = mut_mean - wt_mean

    return variants, delta_pos, delta_mean


def build_gene_features(variants, delta_pos, delta_mean):
    """Aggregate per-variant deltas into per-gene spatial features."""
    from sklearn.decomposition import PCA

    gene_data = defaultdict(list)
    for i, v in enumerate(variants):
        gene_data[v["gene"]].append(
            {
                "aa_pos": v["aa_pos"],
                "delta_pos": delta_pos[i],
                "delta_mean": delta_mean[i],
                "label": v["label_3class"],
            }
        )

    gene_list, X, labels = [], [], []
    feature_names = [
        "delta_mag_mean",
        "delta_mag_std",
        "delta_mag_cv",
        "pos_mean_norm",
        "pos_std_norm",
        "pc1_var_explained",
        "pc1_mean_proj",
        "n_variants_log",
        # mean-pooled delta (1280-dim) appended separately
    ]

    for gene, records in gene_data.items():
        label = Counter(r["label"] for r in records).most_common(1)[0][0]
        n = len(records)

        positions = np.array([r["aa_pos"] for r in records], dtype=float)
        d_pos_mat = np.array([r["delta_pos"] for r in records])
        d_mean_mat = np.array([r["delta_mean"] for r in records])

        mags = np.linalg.norm(d_pos_mat, axis=1)
        mag_mean = float(np.mean(mags))
        mag_std = float(np.std(mags))
        mag_cv = mag_std / (mag_mean + 1e-8)

        max_pos = float(np.max(positions))
        pos_mean_norm = float(np.mean(positions)) / (max_pos + 1e-8)
        pos_std_norm = float(np.std(positions)) / max_pos if len(positions) > 1 else 0.0

        if n >= 3:
            pca = PCA(n_components=1)
            pca.fit(d_pos_mat)
            pc1_var = float(pca.explained_variance_ratio_[0])
            pc1_vec = pca.components_[0]
            mean_delta_pos = d_pos_mat.mean(0)
            pc1_proj = float(np.dot(mean_delta_pos, pc1_vec))
        else:
            pc1_var = 0.0
            pc1_proj = 0.0

        gene_mean_delta = d_mean_mat.mean(0)

        scalar_feats = np.array(
            [
                mag_mean,
                mag_std,
                mag_cv,
                pos_mean_norm,
                pos_std_norm,
                pc1_var,
                pc1_proj,
                np.log1p(n),
            ],
            dtype=np.float32,
        )

        full_feat = np.concatenate([scalar_feats, gene_mean_delta.astype(np.float32)])

        gene_list.append(gene)
        X.append(full_feat)
        labels.append(label)

    gene_list = np.array(gene_list)
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)
    print(f"Built gene features: {len(gene_list)} genes, {X.shape[1]} features")
    print(f"  Scalar features: 8  |  Mean-pooled delta: 1280  |  Total: {X.shape[1]}")
    print(f"  Class distribution: {dict(Counter(labels))}")
    return gene_list, X, labels, len(scalar_feats)


def run_probe(X, labels, splits, groups, held_out_unit, seed=42):
    contract = validate_complete_classification_splits(
        splits, requested_folds=5,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=labels, classes=MECHANISM_CLASSES, groups=groups,
        held_out_unit=held_out_unit,
    )
    return run_logreg_cv(
        X, labels, splits, MECHANISM_CLASSES, contract, seed=seed
    )


def main():
    print("=== Loading data ===")
    variants, delta_pos, delta_mean = load_data()

    print("\n=== Building gene features ===")
    gene_list, X_full, labels, n_scalar = build_gene_features(
        variants, delta_pos, delta_mean
    )

    # Feature subsets
    X_scalar = X_full[:, :n_scalar]
    X_baseline = X_full[:, n_scalar:]
    X_combined = X_full

    with open(os.path.join(DATA, "pfam_families.json")) as f:
        pfam_map = json.load(f)

    all_results = {}
    for seed in range(5):
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list, seed=seed)
        fs = family_split_cv(gene_list, pfam_map, seed=seed)

        seed_res = {}
        split_specs = [
            ("gene_split", gs, gene_list, "gene"),
            (
                "family_split",
                fs,
                np.array([pfam_map.get(gene) for gene in gene_list], dtype=object),
                "family",
            ),
        ]
        for split_name, splits, groups, held_out_unit in split_specs:
            for feat_name, X in [
                ("baseline_delta_mean", X_baseline),
                ("scalar_pattern", X_scalar),
                ("combined", X_combined),
            ]:
                key = f"{feat_name}_{split_name}"
                r = run_probe(
                    X, labels, splits, groups, held_out_unit, seed=seed
                )
                seed_res[key] = r
                f1 = r.get("macro_f1_mean")
                gof = r.get("auroc_GOF_mean")
                if f1 is None:
                    print(f"  {key}: Unscorable")
                else:
                    gof_text = "NA" if gof is None else f"{gof:.3f}"
                    print(f"  {key}: F1={f1:.3f}  GOF={gof_text}")
        all_results[seed] = seed_res

    print("\n=== 5-SEED SUMMARY ===")
    summary = {}
    keys = list(all_results[0].keys())
    for key in keys:
        f1_vals = [
            all_results[s][key].get("macro_f1_mean") for s in range(5)
        ]
        gof_vals = [
            all_results[s][key].get("auroc_GOF_mean") for s in range(5)
        ]
        unavailable = any(value is None for value in f1_vals + gof_vals)
        summary[key] = {
            "status": "unavailable" if unavailable else "success",
            "macro_f1_mean": None if unavailable else float(np.mean(f1_vals)),
            "macro_f1_std": None if unavailable else float(np.std(f1_vals)),
            "auroc_GOF_mean": None if unavailable else float(np.mean(gof_vals)),
            "auroc_GOF_std": None if unavailable else float(np.std(gof_vals)),
        }
        if unavailable:
            print(f"  {key}: Unscorable")
            continue
        print(f"  {key}:")
        print(
            f'    F1  = {summary[key]["macro_f1_mean"]:.3f} ± {summary[key]["macro_f1_std"]:.3f}'
        )
        print(
            f'    GOF = {summary[key]["auroc_GOF_mean"]:.3f} ± {summary[key]["auroc_GOF_std"]:.3f}'
        )

    out = {"summary": summary, "per_seed": {str(s): all_results[s] for s in range(5)}}
    out_path = os.path.join(OUT, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults -> {out_path}")

    # Key comparison
    print("\n=== KEY COMPARISON (family-split) ===")
    baseline = summary["baseline_delta_mean_family_split"]
    scalar = summary["scalar_pattern_family_split"]
    combined = summary["combined_family_split"]
    if any(
        cell["status"] != "success" for cell in (baseline, scalar, combined)
    ):
        print("  Key comparison: Unscorable")
        return
    print(
        f'  Baseline (mean-pooled delta): F1={baseline["macro_f1_mean"]:.3f}  GOF={baseline["auroc_GOF_mean"]:.3f}'
    )
    print(
        f'  Scalar pattern features:      F1={scalar["macro_f1_mean"]:.3f}  GOF={scalar["auroc_GOF_mean"]:.3f}'
    )
    print(
        f'  Combined (scalar + delta):    F1={combined["macro_f1_mean"]:.3f}  GOF={combined["auroc_GOF_mean"]:.3f}'
    )

    delta_f1 = combined["macro_f1_mean"] - baseline["macro_f1_mean"]
    print(f"\n  Lift from adding perturbation pattern: ΔF1 = {delta_f1:+.3f}")
    if delta_f1 > 0.02:
        print("  => Perturbation pattern adds signal beyond mean-pooled delta")
    elif delta_f1 > 0:
        print("  => Small positive lift — marginal signal")
    else:
        print("  => No signal from perturbation pattern features")


if __name__ == "__main__":
    main()
