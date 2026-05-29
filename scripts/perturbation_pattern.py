"""
Amino acid perturbation pattern analysis.

Uses the existing per-residue delta embeddings (already cached) to build
gene-level spatial features from the pattern of observed variant deltas
across sequence positions, then tests whether these add mechanism signal
beyond the mean-pooled delta.

Features per gene (built from all observed variants in that gene):
  1. delta_mag_mean      — mean ||delta|| across positions
  2. delta_mag_std       — std of ||delta|| (how variable is perturbation size)
  3. delta_mag_cv        — coefficient of variation (std/mean) — concentration
  4. pos_mean_norm       — mean variant position / protein length (N vs C terminal)
  5. pos_std_norm        — std of variant positions / protein length (spread)
  6. pc1_var_explained   — variance explained by PC1 of per-gene delta matrix
                           (how much a single direction dominates)
  7. pc1_mean_delta      — projection of mean delta onto PC1
  8. n_variants          — number of observed variants (proxy for study depth)
  9. mean_pooled_delta   — the standard mean-pooled delta (baseline feature)

CV: 5-fold family-split, 5 seeds.
Probe: logistic regression (balanced class weights).

Outputs:
  results/perturbation_pattern/results.json
"""

import json, os, sys, numpy as np
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
EMB  = os.path.join(DATA, 'embeddings')
OUT  = os.path.join(ROOT, 'results', 'perturbation_pattern')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiseed_v1 import family_split_cv, gene_split_cv


def load_data():
    with open(os.path.join(DATA, 'geras_valid_variants.json')) as f:
        variants = json.load(f)
    for v in variants:
        if 'label_3class' not in v:
            v['label_3class'] = 'LOF' if v.get('mechanism') in ('HI','AR') else v.get('mechanism','LOF')

    wt_pos  = np.load(os.path.join(EMB, 'embeddings_wt_pos_esm2_t33_650M_UR50D.npy'))
    mut_pos = np.load(os.path.join(EMB, 'embeddings_mut_pos_esm2_t33_650M_UR50D.npy'))
    wt_mean = np.load(os.path.join(EMB, 'embeddings_wt_esm2_t33_650M_UR50D.npy'))
    mut_mean= np.load(os.path.join(EMB, 'embeddings_mut_esm2_t33_650M_UR50D.npy'))

    delta_pos  = mut_pos  - wt_pos    # per-residue delta at variant position
    delta_mean = mut_mean - wt_mean   # mean-pooled delta (baseline)

    return variants, delta_pos, delta_mean


def build_gene_features(variants, delta_pos, delta_mean):
    """Aggregate per-variant deltas into per-gene spatial pattern features."""
    from sklearn.decomposition import PCA

    # Group by gene
    gene_data = defaultdict(list)  # gene -> list of (aa_pos, delta_pos_vec, delta_mean_vec)
    for i, v in enumerate(variants):
        gene_data[v['gene']].append({
            'aa_pos': v['aa_pos'],
            'delta_pos': delta_pos[i],
            'delta_mean': delta_mean[i],
            'label': v['label_3class'],
        })

    gene_list, X, labels = [], [], []
    feature_names = [
        'delta_mag_mean', 'delta_mag_std', 'delta_mag_cv',
        'pos_mean_norm', 'pos_std_norm',
        'pc1_var_explained', 'pc1_mean_proj',
        'n_variants_log',
        # mean-pooled delta (1280-dim) appended separately
    ]

    for gene, records in gene_data.items():
        label = Counter(r['label'] for r in records).most_common(1)[0][0]
        n = len(records)

        positions  = np.array([r['aa_pos'] for r in records], dtype=float)
        d_pos_mat  = np.array([r['delta_pos'] for r in records])   # (n, 1280)
        d_mean_mat = np.array([r['delta_mean'] for r in records])  # (n, 1280)

        # Magnitudes of per-residue deltas
        mags = np.linalg.norm(d_pos_mat, axis=1)
        mag_mean = float(np.mean(mags))
        mag_std  = float(np.std(mags))
        mag_cv   = mag_std / (mag_mean + 1e-8)

        # Positional spread (normalise by max observed position as proxy for length)
        max_pos = float(np.max(positions))
        pos_mean_norm = float(np.mean(positions)) / (max_pos + 1e-8)
        pos_std_norm  = float(np.std(positions))  / max_pos if len(positions) > 1 else 0.0

        # PCA on per-residue delta matrix
        if n >= 3:
            pca = PCA(n_components=1)
            pca.fit(d_pos_mat)
            pc1_var = float(pca.explained_variance_ratio_[0])
            pc1_vec = pca.components_[0]
            mean_delta_pos = d_pos_mat.mean(0)
            pc1_proj = float(np.dot(mean_delta_pos, pc1_vec))
        else:
            pc1_var  = 0.0
            pc1_proj = 0.0

        # Mean-pooled delta for this gene (average over its variants)
        gene_mean_delta = d_mean_mat.mean(0)  # (1280,)

        scalar_feats = np.array([
            mag_mean, mag_std, mag_cv,
            pos_mean_norm, pos_std_norm,
            pc1_var, pc1_proj,
            np.log1p(n),
        ], dtype=np.float32)

        # Full feature: scalar features + mean-pooled delta
        full_feat = np.concatenate([scalar_feats, gene_mean_delta.astype(np.float32)])

        gene_list.append(gene)
        X.append(full_feat)
        labels.append(label)

    gene_list = np.array(gene_list)
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)
    print(f'Built gene features: {len(gene_list)} genes, {X.shape[1]} features')
    print(f'  Scalar features: 8  |  Mean-pooled delta: 1280  |  Total: {X.shape[1]}')
    from collections import Counter
    print(f'  Class distribution: {dict(Counter(labels))}')
    return gene_list, X, labels, len(scalar_feats)


def run_probe(X, labels, splits, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["GOF", "DN", "LOF"])
    y = le.transform(labels)
    classes = le.classes_
    fold_results = []

    for tr, te in splits:
        if len(set(y[tr])) < len(classes): continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0,
                                  class_weight='balanced', random_state=seed)
        clf.fit(Xtr, y[tr])
        raw_proba = clf.predict_proba(Xte)
        proba = np.zeros((len(Xte), len(classes)), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            proba[:, c] = raw_proba[:, ci]
        pred  = proba.argmax(axis=1)
        fm = {'macro_f1': float(f1_score(y[te], pred, average='macro', zero_division=0))}
        for i, cls in enumerate(classes):
            yb = (y[te] == i).astype(int)
            if yb.sum() > 0 and (1-yb).sum() > 0:
                fm[f'auroc_{cls}'] = float(roc_auc_score(yb, proba[:, i]))
        fold_results.append(fm)

    if not fold_results: return {}
    agg = {}
    for key in set().union(*[set(f) for f in fold_results]):
        vals = [f[key] for f in fold_results if key in f]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std']  = float(np.std(vals))
    return agg


def main():
    print('=== Loading data ===')
    variants, delta_pos, delta_mean = load_data()

    print('\n=== Building gene features ===')
    gene_list, X_full, labels, n_scalar = build_gene_features(
        variants, delta_pos, delta_mean)

    # Feature subsets
    X_scalar   = X_full[:, :n_scalar]          # 8 scalar features only
    X_baseline = X_full[:, n_scalar:]           # mean-pooled delta only (baseline)
    X_combined = X_full                         # scalar + mean-pooled delta

    with open(os.path.join(DATA, 'pfam_families.json')) as f:
        pfam_map = json.load(f)

    all_results = {}
    for seed in range(5):
        print(f'\n=== Seed {seed} ===')
        gs = gene_split_cv(gene_list, seed=seed)
        fs = family_split_cv(gene_list, pfam_map, seed=seed)

        seed_res = {}
        for split_name, splits in [('gene_split', gs), ('family_split', fs)]:
            for feat_name, X in [
                ('baseline_delta_mean', X_baseline),
                ('scalar_pattern',      X_scalar),
                ('combined',            X_combined),
            ]:
                key = f'{feat_name}_{split_name}'
                r = run_probe(X, labels, splits, seed=seed)
                seed_res[key] = r
                f1  = r.get('macro_f1_mean', float('nan'))
                gof = r.get('auroc_GOF_mean', float('nan'))
                print(f'  {key}: F1={f1:.3f}  GOF={gof:.3f}')
        all_results[seed] = seed_res

    # Summary across seeds
    print('\n=== 5-SEED SUMMARY ===')
    summary = {}
    keys = list(all_results[0].keys())
    for key in keys:
        f1_vals  = [all_results[s][key].get('macro_f1_mean', float('nan')) for s in range(5)]
        gof_vals = [all_results[s][key].get('auroc_GOF_mean', float('nan')) for s in range(5)]
        summary[key] = {
            'macro_f1_mean': float(np.nanmean(f1_vals)),
            'macro_f1_std':  float(np.nanstd(f1_vals)),
            'auroc_GOF_mean': float(np.nanmean(gof_vals)),
            'auroc_GOF_std':  float(np.nanstd(gof_vals)),
        }
        print(f'  {key}:')
        print(f'    F1  = {summary[key]["macro_f1_mean"]:.3f} ± {summary[key]["macro_f1_std"]:.3f}')
        print(f'    GOF = {summary[key]["auroc_GOF_mean"]:.3f} ± {summary[key]["auroc_GOF_std"]:.3f}')

    out = {'summary': summary, 'per_seed': {str(s): all_results[s] for s in range(5)}}
    out_path = os.path.join(OUT, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults -> {out_path}')

    # Key comparison
    print('\n=== KEY COMPARISON (family-split) ===')
    baseline = summary['baseline_delta_mean_family_split']
    scalar   = summary['scalar_pattern_family_split']
    combined = summary['combined_family_split']
    print(f'  Baseline (mean-pooled delta): F1={baseline["macro_f1_mean"]:.3f}  GOF={baseline["auroc_GOF_mean"]:.3f}')
    print(f'  Scalar pattern features:      F1={scalar["macro_f1_mean"]:.3f}  GOF={scalar["auroc_GOF_mean"]:.3f}')
    print(f'  Combined (scalar + delta):    F1={combined["macro_f1_mean"]:.3f}  GOF={combined["auroc_GOF_mean"]:.3f}')

    delta_f1 = combined['macro_f1_mean'] - baseline['macro_f1_mean']
    print(f'\n  Lift from adding perturbation pattern: ΔF1 = {delta_f1:+.3f}')
    if delta_f1 > 0.02:
        print('  => Perturbation pattern adds signal beyond mean-pooled delta')
    elif delta_f1 > 0:
        print('  => Small positive lift — marginal signal')
    else:
        print('  => No signal from perturbation pattern features')


if __name__ == '__main__':
    main()
