"""
ESM-2 delta-embedding mechanism geometry experiment.

Tests whether ESM-2 delta-embeddings (mutant - wildtype) organize by molecular
disease mechanism class (GOF / DN / LOF) after removing protein stability signal.

Data: merged_variants.json (built by fetch_data/fetch_variants.py --step merge)
  - Gerasimavicius et al. 2022 + ClinVar missense variants, gene-level mechanism labels
  - GOF, DN, HI, AR classes; FoldX ΔΔG provided
  - Primary: 3-class GOF / DN / LOF (HI+AR collapsed)

Pipeline:
  1. Load merged_variants.json
  2. Load embeddings from data/embeddings/<model>/ (built by embed_variants.py)
  3. Compute delta-embeddings (mutant - WT), mean-pooled and per-residue
  4. Fit stability nuisance subspace on Megascale data; validate transfer
  5. Linear probe (LR) with gene-split CV
  6. Baselines, negative controls, probe direction orthogonality analysis
"""

import argparse
import json
import os
import warnings
import functools

import numpy as np
print = functools.partial(print, flush=True)
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from esm2_mechanism.utils_probes import gene_split_cv, family_split_cv, run_logreg_cv
from esm2_mechanism.utils_sequences import (
    build_sequence_cache, fetch_pfam_families,
    apply_missense, window_sequence,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"

CLASSES_3 = ["GOF", "DN", "LOF"]

# Pre-registered thresholds
STABILITY_TRANSFER_RHO_THRESHOLD = 0.3  # Spearman ρ for Megascale→Gerasimavicius transfer
SCALE_INVARIANT_THRESHOLD = 0.03        # macro-F1 difference 650M vs 3B
SCALE_EMERGENT_THRESHOLD = 0.05
VARIANCE_ASYMMETRY_THRESHOLD = 0.30    # GOF ≥ 30% less variance explained than HI+AR
BENIGN_LEAK_THRESHOLD = 0.50           # benign AUROC as fraction of pathogenic AUROC


# ---------------------------------------------------------------------------
# ESM-2 embedding extraction
# ---------------------------------------------------------------------------

def get_esm2_embeddings_for_pairs(wt_seqs, mut_seqs, aa_positions,
                                   model_name=ESM2_MODEL_650M,
                                   device="cuda", batch_size=32):
    """
    Extract ESM-2 embeddings for WT and mutant sequences.
    WT and mutant are interleaved in the same batch to halve forward passes.
    Returns:
        wt_mean: (N, D) mean-pooled WT embeddings
        mut_mean: (N, D) mean-pooled mutant embeddings
        wt_pos: (N, D) per-residue embedding at variant position for WT
        mut_pos: (N, D) per-residue embedding at variant position for mutant
    """
    import torch
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    wt_mean_list, mut_mean_list = [], []
    wt_pos_list, mut_pos_list = [], []

    N = len(wt_seqs)
    for i in range(0, N, batch_size):
        pairs = list(zip(wt_seqs[i:i+batch_size],
                         mut_seqs[i:i+batch_size],
                         aa_positions[i:i+batch_size]))
        interleaved = []
        for j, (wt, mut, _) in enumerate(pairs):
            interleaved.append((f"wt{j}", wt))
            interleaved.append((f"mut{j}", mut))

        _, _, tokens = batch_converter(interleaved)
        tokens = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model(tokens, repr_layers=[n_layers])
        reps = out["representations"][n_layers].cpu().float()

        for j, (wt, mut, var_pos) in enumerate(pairs):
            wt_rep  = reps[2*j]
            mut_rep = reps[2*j + 1]

            wt_mean_list.append(wt_rep[1:len(wt)+1].mean(0).numpy())
            mut_mean_list.append(mut_rep[1:len(mut)+1].mean(0).numpy())

            var_idx_token = var_pos  # 1-indexed; BOS occupies token 0, so sequence tokens are 1..len(wt)
            if 0 < var_idx_token <= len(wt):
                wt_pos_list.append(wt_rep[var_idx_token].numpy())
                mut_pos_list.append(mut_rep[var_idx_token].numpy())
            else:
                wt_pos_list.append(wt_rep[1:len(wt)+1].mean(0).numpy())
                mut_pos_list.append(mut_rep[1:len(mut)+1].mean(0).numpy())

        if (i // batch_size) % 5 == 0:
            print(f"  Embedded {min(i + batch_size, N)}/{N} variant pairs")

    return (np.stack(wt_mean_list), np.stack(mut_mean_list),
            np.stack(wt_pos_list), np.stack(mut_pos_list))


# ---------------------------------------------------------------------------
# Stability nuisance subspace
# ---------------------------------------------------------------------------

def fit_stability_subspace_megascale(cache_dir, n_components=10,
                                      model_name=ESM2_MODEL_650M, device="cuda"):
    """
    Fit a stability nuisance subspace using the Megascale dataset
    (Tsuboyama et al. 2023, zenodo.org/records/7844779).

    Returns the projection matrix (n_components, D) or None if unavailable.
    """
    subspace_cache = os.path.join(cache_dir, f"stability_subspace_{model_name}.npy")
    if os.path.exists(subspace_cache):
        print("Loading cached stability subspace...")
        return np.load(subspace_cache)

    megascale_cache = os.path.join(cache_dir, "megascale_deltas.npy")
    megascale_ddg_cache = os.path.join(cache_dir, "megascale_ddg.npy")

    if not (os.path.exists(megascale_cache) and os.path.exists(megascale_ddg_cache)):
        print("Megascale delta embeddings not found — will fit subspace on Gerasimavicius data")
        return None

    print("Fitting stability subspace on Megascale delta embeddings...")
    deltas = np.load(megascale_cache)
    ddg = np.load(megascale_ddg_cache)

    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(ddg.reshape(-1, 1), deltas)
    coefs = np.array(reg.coef_).flatten()

    stability_dir = coefs / (np.linalg.norm(coefs) + 1e-10)

    deltas_res = deltas - deltas.dot(stability_dir)[:, None] * stability_dir
    pca = PCA(n_components=min(n_components - 1, deltas_res.shape[0] - 1, deltas_res.shape[1]))
    pca.fit(deltas_res)

    subspace = np.vstack([stability_dir.reshape(1, -1), pca.components_])
    subspace = subspace[:n_components]

    np.save(subspace_cache, subspace)
    return subspace


def fit_stability_subspace_direct(deltas, foldx_ddg, n_components=10, genes=None):
    """
    Fit stability subspace directly on Gerasimavicius variants using FoldX ΔΔG.
    Used as fallback when Megascale transfer fails.
    Returns projection matrix (n_components, D).
    """
    from sklearn.linear_model import Ridge

    valid = ~np.isnan(foldx_ddg)
    if valid.sum() < 50:
        print("  Too few variants with FoldX ΔΔG for direct subspace fit")
        return None

    ddg_valid = foldx_ddg[valid]
    deltas_valid = deltas[valid]
    genes_valid = genes[valid] if genes is not None else None

    if genes_valid is not None:
        unique_genes = np.unique(genes_valid)
        coefs_list = []
        for held_gene in unique_genes:
            mask = genes_valid != held_gene
            if mask.sum() < 20:
                continue
            reg = Ridge(alpha=1.0)
            reg.fit(ddg_valid[mask].reshape(-1, 1), deltas_valid[mask])
            coefs_list.append(reg.coef_)
        if coefs_list:
            coefs = np.mean(coefs_list, axis=0)
        else:
            reg = Ridge(alpha=1.0)
            reg.fit(ddg_valid.reshape(-1, 1), deltas_valid)
            coefs = reg.coef_
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(ddg_valid.reshape(-1, 1), deltas_valid)
        coefs = reg.coef_

    coefs = np.array(coefs).flatten()
    stability_dir = coefs / (np.linalg.norm(coefs) + 1e-10)

    deltas_res = deltas_valid - deltas_valid.dot(stability_dir)[:, None] * stability_dir
    n_comp = min(n_components - 1, deltas_res.shape[0] - 1, deltas_res.shape[1] - 1)
    if n_comp < 1:
        return stability_dir.reshape(1, -1)

    pca = PCA(n_components=n_comp)
    pca.fit(deltas_res)
    subspace = np.vstack([stability_dir.reshape(1, -1), pca.components_])
    return subspace[:n_components]


def validate_stability_transfer(subspace, deltas_geras, foldx_ddg_geras):
    """Validate Megascale-fit subspace against FoldX ΔΔG on Gerasimavicius variants.

    Returns Spearman ρ.
    """
    valid = ~np.isnan(foldx_ddg_geras)
    if valid.sum() < 20:
        return 0.0
    proj = deltas_geras[valid].dot(subspace[0])
    rho, _ = spearmanr(proj, foldx_ddg_geras[valid])
    return float(rho)


def select_stability_subspace(megascale_subspace, deltas, foldx_ddg, genes, n_components):
    """Choose between Megascale-transfer (Path A) and direct-fit (Path B) subspaces.

    Returns (subspace, stability_path, transfer_rho).
    """
    transfer_rho = float("nan")

    if megascale_subspace is not None:
        transfer_rho = validate_stability_transfer(megascale_subspace, deltas, foldx_ddg)
        print(f"  Megascale→Gerasimavicius transfer Spearman ρ = {transfer_rho:.3f}")
        print(f"  Pre-registered threshold: ρ > {STABILITY_TRANSFER_RHO_THRESHOLD}")

        if transfer_rho >= STABILITY_TRANSFER_RHO_THRESHOLD:
            print("  Path A: Megascale transfer PASSES — using Megascale subspace")
            return megascale_subspace, "A_megascale", transfer_rho

        print("  Path A: Megascale transfer FAILS — falling back to Path B")
    else:
        print("  Path B: No Megascale data — fitting subspace directly on Gerasimavicius")

    subspace = fit_stability_subspace_direct(deltas, foldx_ddg,
                                              n_components=n_components, genes=genes)
    return subspace, "B_direct", transfer_rho


def project_out_subspace(deltas, subspace):
    """Remove the subspace from delta embeddings."""
    if subspace is None:
        return deltas
    Q, _ = np.linalg.qr(subspace.T, mode='reduced')
    proj = deltas.dot(Q).dot(Q.T)
    return deltas - proj


def variance_explained_per_class(deltas, labels_3class, subspace):
    """
    Report fraction of variance explained by stability subspace per mechanism class.
    Pre-registered prediction: GOF has ≥ 30% less variance explained than HI+AR (LOF).
    """
    if subspace is None:
        return {}

    Q, _ = np.linalg.qr(subspace.T, mode='reduced')
    results = {}
    for cls in ["GOF", "DN", "LOF"]:
        mask = labels_3class == cls
        if mask.sum() < 5:
            continue
        d = deltas[mask]
        total_var = float(np.var(d, axis=0).sum())
        proj = d.dot(Q).dot(Q.T)
        proj_var = float(np.var(proj, axis=0).sum())
        results[cls] = proj_var / (total_var + 1e-10)

    if "GOF" in results and "LOF" in results:
        asymmetry = (results["LOF"] - results["GOF"]) / (results["LOF"] + 1e-10)
        results["gof_lof_asymmetry"] = float(asymmetry)
        results["asymmetry_prediction_holds"] = bool(asymmetry >= VARIANCE_ASYMMETRY_THRESHOLD)

    return results


# ---------------------------------------------------------------------------
# Probe direction orthogonality
# ---------------------------------------------------------------------------

def probe_direction_orthogonality(X, y, genes, stability_subspace,
                                   n_folds=5, seed=42, n_shuffle=50):
    """
    Fit pairwise LR probes (GOF-vs-DN, GOF-vs-LOF, DN-vs-LOF) and compute
    cosine similarity between probe weight vectors. Compare to shuffled-label null.
    """
    splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
    all_train_idx = np.unique(np.concatenate([tr for tr, _ in splits]))

    X_train = X[all_train_idx]
    y_train = y[all_train_idx]

    classes = sorted(set(y_train))
    pairs = [(c1, c2) for i, c1 in enumerate(classes) for c2 in classes[i+1:]]

    def fit_pairwise_probes(y_labels):
        weights = {}
        for c1, c2 in pairs:
            mask = np.isin(y_labels, [c1, c2])
            if mask.sum() < 10:
                continue
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
            try:
                clf.fit(X_train[mask], y_labels[mask])
                w = clf.coef_[0]
                weights[f"{c1}_vs_{c2}"] = w / (np.linalg.norm(w) + 1e-10)
            except Exception:
                pass
        return weights

    probe_weights = fit_pairwise_probes(y_train)

    pair_keys = list(probe_weights.keys())
    cosine_matrix = {}
    for i, k1 in enumerate(pair_keys):
        for j, k2 in enumerate(pair_keys):
            if i >= j:
                continue
            cosine_matrix[f"{k1}|{k2}"] = float(np.dot(probe_weights[k1], probe_weights[k2]))

    stability_cosines = {}
    if stability_subspace is not None:
        stab_dir = stability_subspace[0] / (np.linalg.norm(stability_subspace[0]) + 1e-10)
        for pair_key, w in probe_weights.items():
            stability_cosines[f"{pair_key}_vs_stability"] = float(np.dot(w, stab_dir))

    rng = np.random.RandomState(seed)
    null_cosines = []
    for _ in range(n_shuffle):
        y_shuf = rng.permutation(y_train)
        shuf_weights = fit_pairwise_probes(y_shuf)
        shuf_keys = list(shuf_weights.keys())
        for i, k1 in enumerate(shuf_keys):
            for j, k2 in enumerate(shuf_keys):
                if i >= j:
                    continue
                null_cosines.append(float(np.dot(shuf_weights[k1], shuf_weights[k2])))

    null_mean = float(np.mean(null_cosines)) if null_cosines else float("nan")
    null_std = float(np.std(null_cosines)) if null_cosines else float("nan")

    distinguishable = {}
    for pair, real_cos in cosine_matrix.items():
        if null_cosines and null_std > 0:
            z = (real_cos - null_mean) / null_std
            distinguishable[pair] = bool(abs(z) > 2.0)

    return {
        "cosine_matrix": cosine_matrix,
        "stability_cosines": stability_cosines,
        "null_cosine_mean": null_mean,
        "null_cosine_std": null_std,
        "distinguishable_from_null": distinguishable,
        "path": "A" if stability_subspace is not None else "B",
    }


# ---------------------------------------------------------------------------
# Baselines and negative controls
# ---------------------------------------------------------------------------

def run_baselines(embeddings_wt, deltas_mean, foldx_ddg, y, genes,
                  aa_wt_list, aa_mut_list, alphamissense_scores, seed=42):
    """Run four baselines under gene-split CV."""
    splits = gene_split_cv(genes, seed=seed)
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    aa_index = {a: i for i, a in enumerate(AA_ORDER)}

    n = len(aa_wt_list)
    onehot = np.zeros((n, 40), dtype=np.float32)
    for i, (wt, mut) in enumerate(zip(aa_wt_list, aa_mut_list)):
        wt_idx = aa_index.get(wt.upper())
        mut_idx = aa_index.get(mut.upper())
        if wt_idx is not None:
            onehot[i, wt_idx] = 1.0
        if mut_idx is not None:
            onehot[i, 20 + mut_idx] = 1.0

    # FoldX and AlphaMissense baselines are restricted to variants with observed values.
    # Missing values are not imputed — a variant with no FoldX ΔΔG is excluded from
    # that baseline entirely so the probe only sees real measurements.
    foldx_mask = ~np.isnan(foldx_ddg)
    if foldx_mask.sum() >= 20:
        ddg_splits = gene_split_cv(genes[foldx_mask], seed=seed)
        ddg_feat = foldx_ddg[foldx_mask].reshape(-1, 1)
        foldx_config = ("foldx_ddg_only", ddg_feat, y[foldx_mask], ddg_splits,
                        ddg_feat.std() > 0)
    else:
        foldx_config = ("foldx_ddg_only", None, None, None, False)

    if alphamissense_scores is not None:
        am_mask = ~np.isnan(alphamissense_scores)
        if am_mask.sum() >= 20:
            am_splits = gene_split_cv(genes[am_mask], seed=seed)
            am_feat = alphamissense_scores[am_mask].reshape(-1, 1)
            am_config = ("alphamissense", am_feat, y[am_mask], am_splits,
                         am_feat.std() > 0)
        else:
            am_config = ("alphamissense", None, None, None, False)
    else:
        am_config = ("alphamissense", None, None, None, False)

    # Full-data baselines use the shared splits computed above
    full_configs = [
        ("wt_only",   embeddings_wt, y, splits, True),
        ("onehot_aa", onehot,        y, splits, True),
    ]

    results = {}
    for name, X_bl, y_bl, spl, runnable in full_configs + [foldx_config, am_config]:
        if not runnable or X_bl is None:
            results[name] = {"note": f"{name} unavailable or zero-variance"}
            continue
        print(f"  Baseline: {name} (n={len(y_bl)})")
        results[name] = run_logreg_cv(X_bl, y_bl, spl, classes=CLASSES_3,
                                       seed=seed, label=name)
    return results


def run_negative_controls(deltas_mean, y, genes, seed=42):
    """Shuffle deltas across genes to verify signal collapses to chance."""
    rng = np.random.RandomState(seed)
    splits = gene_split_cv(genes, seed=seed)
    deltas_shuffled = deltas_mean[rng.permutation(len(deltas_mean))]
    print("  Negative control: shuffled deltas")
    return {
        "shuffled_delta": run_logreg_cv(deltas_shuffled, y, splits,
                                         classes=CLASSES_3, seed=seed,
                                         label="shuffled_delta"),
    }


# ---------------------------------------------------------------------------
# Phase helpers for run()
# ---------------------------------------------------------------------------

def _load_alphamissense_scores(variants: list, data_dir: str) -> np.ndarray:
    """Load AlphaMissense scores from fetch_data output (alphamissense_scores_full.json).

    Returns a float array of length len(variants), NaN where no score is available.
    Keys in the JSON are gene_aapos_aawt_aamut (same format used by fetch_annotations.py).
    """
    am_path = os.path.join(data_dir, "alphamissense_scores_full.json")
    if not os.path.exists(am_path):
        print(f"  WARNING: {am_path} not found — AlphaMissense scores unavailable")
        return np.full(len(variants), np.nan)
    try:
        with open(am_path) as f:
            am_scores = json.load(f)
    except json.JSONDecodeError:
        print(f"  WARNING: corrupt {am_path} — AlphaMissense scores unavailable")
        return np.full(len(variants), np.nan)

    scores = np.full(len(variants), np.nan)
    for idx, v in enumerate(variants):
        key = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        val = am_scores.get(key)
        if val is not None:
            scores[idx] = float(val)

    n_valid = int(np.sum(~np.isnan(scores)))
    print(f"  AlphaMissense scores: {n_valid}/{len(variants)} available")
    return scores


def _load_data(data_dir):
    """Phase 1: load and filter the merged variant dataset."""
    merged_path = os.path.join(data_dir, "merged_variants.json")
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"{merged_path} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    with open(merged_path) as f:
        variants = json.load(f)
    variants = [v for v in variants
                if v.get("uniprot_id") and v.get("aa_wt") and v.get("aa_mut") and v.get("aa_pos", 0) > 0]
    print(f"After filtering: {len(variants)} variants")
    return variants


def _prepare_sequences(variants, data_dir):
    """Phase 2: fetch sequences and build WT/mutant pairs."""
    seq_cache = build_sequence_cache(variants, data_dir)

    valid_variants, wt_seqs, mut_seqs, var_positions = [], [], [], []
    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_win, new_pos = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        var_positions.append(new_pos)
        valid_variants.append(v)

    print(f"Valid variant pairs: {len(valid_variants)}")
    if len(valid_variants) < 50:
        print("WARNING: Very few valid variants. Results may not be reliable.")
    return valid_variants, seq_cache, wt_seqs, mut_seqs, var_positions


def _load_or_extract_embeddings(wt_seqs, mut_seqs, var_positions,
                                 data_dir, model_name, device, batch_size):
    """Phase 3: load embeddings from data/embeddings/<model>/ or run ESM-2 forward pass."""
    emb_dir = os.path.join(data_dir, "embeddings", model_name)
    emb_cache = {
        "wt":      os.path.join(emb_dir, "embeddings_wt_mean.npy"),
        "mut":     os.path.join(emb_dir, "embeddings_mut_mean.npy"),
        "wt_pos":  os.path.join(emb_dir, "embeddings_wt_pos.npy"),
        "mut_pos": os.path.join(emb_dir, "embeddings_mut_pos.npy"),
    }
    if os.path.exists(emb_cache["wt"]) and os.path.exists(emb_cache["mut"]):
        print("\n=== Loading cached embeddings ===")
        for key in ("wt_pos", "mut_pos"):
            if not os.path.exists(emb_cache[key]):
                raise FileNotFoundError(
                    f"Mean embeddings found but positional embeddings missing: {emb_cache[key]}\n"
                    f"Re-run embed_variants.py to regenerate the full embedding set."
                )
        emb_wt_mean  = np.load(emb_cache["wt"])
        emb_mut_mean = np.load(emb_cache["mut"])
        emb_wt_pos   = np.load(emb_cache["wt_pos"])
        emb_mut_pos  = np.load(emb_cache["mut_pos"])
    else:
        print(f"\n=== Extracting ESM-2 embeddings ({model_name}) ===")
        os.makedirs(emb_dir, exist_ok=True)
        emb_wt_mean, emb_mut_mean, emb_wt_pos, emb_mut_pos = get_esm2_embeddings_for_pairs(
            wt_seqs, mut_seqs, var_positions,
            model_name=model_name, device=device, batch_size=batch_size,
        )
        np.save(emb_cache["wt"],      emb_wt_mean)
        np.save(emb_cache["mut"],     emb_mut_mean)
        np.save(emb_cache["wt_pos"],  emb_wt_pos)
        np.save(emb_cache["mut_pos"], emb_mut_pos)

    return emb_wt_mean, emb_mut_mean, emb_wt_pos, emb_mut_pos


def _run_primary_probes(deltas_mean_proj, deltas_mean,
                         deltas_pos_proj, deltas_pos,
                         y, genes, n_cv_folds, seed):
    """Phase 5: run the four primary probe variants."""
    splits = gene_split_cv(genes, n_folds=n_cv_folds, seed=seed)
    probe_configs = [
        ("mean_pooled_projected",   deltas_mean_proj),
        ("mean_pooled_unprojected", deltas_mean),
        ("per_residue_projected",   deltas_pos_proj),
        ("per_residue_unprojected", deltas_pos),
    ]
    results = {}
    for name, X in probe_configs:
        print(f"  {name}:")
        results[name] = run_logreg_cv(X, y, splits, classes=CLASSES_3,
                                       seed=seed, label=name)
    return results


def _run_secondary_probes(deltas_mean_proj, labels_4class, labels_3class,
                           genes, n_cv_folds, seed):
    """Phase 6: 4-class probe and HI-vs-AR probe."""
    results = {}
    classes_4 = ["GOF", "DN", "HI", "AR"]

    le4 = LabelEncoder().fit(classes_4)
    y4 = le4.transform(labels_4class)
    splits4 = gene_split_cv(genes, n_folds=n_cv_folds, seed=seed)
    print("  4-class (GOF/DN/HI/AR):")
    results["four_class"] = run_logreg_cv(deltas_mean_proj, y4, splits4,
                                           classes=classes_4, seed=seed, label="4class")

    hi_ar_mask = np.isin(labels_4class, ["HI", "AR"])
    if hi_ar_mask.sum() >= 20:
        le2 = LabelEncoder().fit(["AR", "HI"])
        y2 = le2.transform(labels_4class[hi_ar_mask])
        splits2 = gene_split_cv(genes[hi_ar_mask], n_folds=n_cv_folds, seed=seed)
        print("  HI vs AR (2-class):")
        results["hi_vs_ar"] = run_logreg_cv(deltas_mean_proj[hi_ar_mask], y2, splits2,
                                             classes=["AR", "HI"], seed=seed, label="hi_vs_ar")
    return results


def _run_family_cv(deltas_mean_proj, y, genes, valid_variants,
                   data_dir, n_cv_folds, seed):
    """Phase 7: gene-family-split CV using Pfam families."""
    pfam_map = fetch_pfam_families(valid_variants, data_dir)
    n_families = len(set(v for v in pfam_map.values() if v is not None))

    splits = family_split_cv(genes, pfam_map, n_folds=n_cv_folds, seed=seed)
    if not splits:
        print("  Family-split CV infeasible — skipping")
        return {}, pfam_map, n_families

    print(f"  Running family-split CV with {len(splits)} folds")
    results = run_logreg_cv(deltas_mean_proj, y, splits, classes=CLASSES_3,
                             seed=seed, label="family_cv")
    print(f"  Family-split macro-F1: {results.get('macro_f1_mean', float('nan')):.3f}")
    return results, pfam_map, n_families


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(out_dir, seed=0, model_name=ESM2_MODEL_650M, n_stability_components=10,
        n_cv_folds=5, batch_size=32):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {model_name}")

    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n=== Loading Gerasimavicius dataset ===")
    variants = _load_data(data_dir)

    # ------------------------------------------------------------------
    # 2. Prepare sequences
    # ------------------------------------------------------------------
    print("\n=== Fetching protein sequences ===")
    valid_variants, seq_cache, wt_seqs, mut_seqs, var_positions = _prepare_sequences(
        variants, data_dir
    )

    # ------------------------------------------------------------------
    # 3. Extract embeddings
    # ------------------------------------------------------------------
    emb_wt_mean, emb_mut_mean, emb_wt_pos, emb_mut_pos = _load_or_extract_embeddings(
        wt_seqs, mut_seqs, var_positions, data_dir, model_name, device, batch_size
    )

    deltas_mean = emb_mut_mean - emb_wt_mean
    deltas_pos  = emb_mut_pos  - emb_wt_pos
    print(f"Delta embedding shape: {deltas_mean.shape}")

    # Labels and metadata arrays
    le3 = LabelEncoder().fit(CLASSES_3)
    labels_3class  = np.array([v["label_3class"] for v in valid_variants])
    labels_4class  = np.array([v["mechanism"]    for v in valid_variants])
    y3             = le3.transform(labels_3class)
    genes_arr      = np.array([v["gene"]         for v in valid_variants])
    foldx_ddg      = np.array([v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan
                                for v in valid_variants])
    aa_wt_list  = [v["aa_wt"]  for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    print("\n=== Loading AlphaMissense scores ===")
    alphamissense_scores = _load_alphamissense_scores(valid_variants, data_dir)

    from collections import Counter
    print(f"3-class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    # ------------------------------------------------------------------
    # 4. Stability subspace
    # ------------------------------------------------------------------
    print("\n=== Stability subspace ===")
    megascale_subspace = fit_stability_subspace_megascale(
        data_dir, n_components=n_stability_components,
        model_name=model_name, device=device,
    )
    stability_subspace, stability_path, transfer_rho = select_stability_subspace(
        megascale_subspace, deltas_mean, foldx_ddg, genes_arr, n_stability_components
    )

    var_exp = variance_explained_per_class(deltas_mean, labels_3class, stability_subspace)
    print(f"  Variance explained by stability subspace: {var_exp}")

    deltas_mean_proj = project_out_subspace(deltas_mean, stability_subspace)
    deltas_pos_proj  = project_out_subspace(deltas_pos,  stability_subspace)

    # ------------------------------------------------------------------
    # 5. Primary probes
    # ------------------------------------------------------------------
    print("\n=== Primary linear probe (3-class: GOF/DN/LOF) ===")
    results_primary = _run_primary_probes(
        deltas_mean_proj, deltas_mean, deltas_pos_proj, deltas_pos,
        y3, genes_arr, n_cv_folds, seed,
    )

    # ------------------------------------------------------------------
    # 6. Secondary probes
    # ------------------------------------------------------------------
    print("\n=== Secondary probes ===")
    results_secondary = _run_secondary_probes(
        deltas_mean_proj, labels_4class, labels_3class, genes_arr, n_cv_folds, seed
    )

    # ------------------------------------------------------------------
    # 7. Baselines
    # ------------------------------------------------------------------
    print("\n=== Baselines ===")
    results_baselines = run_baselines(
        emb_wt_mean, deltas_mean_proj, foldx_ddg, y3, genes_arr,
        aa_wt_list, aa_mut_list, alphamissense_scores, seed=seed,
    )

    # ------------------------------------------------------------------
    # 8. Negative controls
    # ------------------------------------------------------------------
    print("\n=== Negative controls ===")
    results_negctrl = run_negative_controls(deltas_mean_proj, y3, genes_arr, seed=seed)

    # ------------------------------------------------------------------
    # 9. Gene-family-split CV
    # ------------------------------------------------------------------
    print("\n=== Gene-family-split CV ===")
    results_family_cv, pfam_map, pfam_n_families = _run_family_cv(
        deltas_mean_proj, y3, genes_arr, valid_variants, data_dir, n_cv_folds, seed
    )

    # ------------------------------------------------------------------
    # 10. Probe direction orthogonality
    # ------------------------------------------------------------------
    print("\n=== Probe direction orthogonality ===")
    subspace_for_ortho = stability_subspace if stability_path == "A_megascale" else None
    ortho_results = probe_direction_orthogonality(
        deltas_mean_proj, labels_3class, genes_arr,
        stability_subspace=subspace_for_ortho,
        n_folds=n_cv_folds, seed=seed,
    )
    print(f"  Cosine matrix: {ortho_results['cosine_matrix']}")
    print(f"  Null cosine mean: {ortho_results['null_cosine_mean']:.3f} ± {ortho_results['null_cosine_std']:.3f}")

    # ------------------------------------------------------------------
    # 11. Compile and save results
    # ------------------------------------------------------------------
    primary_mean_proj = results_primary["mean_pooled_projected"]
    primary_per_proj  = results_primary["per_residue_projected"]

    final_info = {
        "headline_macro_f1":        primary_mean_proj.get("macro_f1_mean",  float("nan")),
        "headline_auroc_GOF":       primary_mean_proj.get("auroc_GOF_mean", float("nan")),
        "headline_auroc_DN":        primary_mean_proj.get("auroc_DN_mean",  float("nan")),
        "headline_auroc_LOF":       primary_mean_proj.get("auroc_LOF_mean", float("nan")),
        "per_residue_macro_f1":     primary_per_proj.get("macro_f1_mean",   float("nan")),
        "per_residue_auroc_GOF":    primary_per_proj.get("auroc_GOF_mean",  float("nan")),
        "unprojected_macro_f1":     results_primary["mean_pooled_unprojected"].get("macro_f1_mean", float("nan")),
        "stability_path":           stability_path,
        "stability_transfer_rho":   transfer_rho,
        "variance_explained_GOF":   var_exp.get("GOF",   float("nan")),
        "variance_explained_DN":    var_exp.get("DN",    float("nan")),
        "variance_explained_LOF":   var_exp.get("LOF",   float("nan")),
        "variance_asymmetry_gof_lof":         var_exp.get("gof_lof_asymmetry",         float("nan")),
        "variance_asymmetry_prediction_holds": var_exp.get("asymmetry_prediction_holds", False),
        "baseline_wt_only_macro_f1":      results_baselines.get("wt_only",       {}).get("macro_f1_mean", float("nan")),
        "baseline_foldx_macro_f1":        results_baselines.get("foldx_ddg_only",{}).get("macro_f1_mean", float("nan")),
        "baseline_onehot_macro_f1":       results_baselines.get("onehot_aa",     {}).get("macro_f1_mean", float("nan")),
        "baseline_alphamissense_macro_f1":results_baselines.get("alphamissense", {}).get("macro_f1_mean", float("nan")),
        "neg_ctrl_shuffled_macro_f1":     results_negctrl.get("shuffled_delta",  {}).get("macro_f1_mean", float("nan")),
        "cosine_matrix":            ortho_results["cosine_matrix"],
        "null_cosine_mean":         ortho_results["null_cosine_mean"],
        "ortho_distinguishable_from_null": str(ortho_results["distinguishable_from_null"]),
        "family_cv_macro_f1":  results_family_cv.get("macro_f1_mean",  float("nan")),
        "family_cv_auroc_GOF": results_family_cv.get("auroc_GOF_mean", float("nan")),
        "family_cv_auroc_DN":  results_family_cv.get("auroc_DN_mean",  float("nan")),
        "family_cv_auroc_LOF": results_family_cv.get("auroc_LOF_mean", float("nan")),
        "orthogonality_path":  ortho_results.get("path", "unknown"),
        "n_variants": len(valid_variants),
        "n_genes":    len(set(genes_arr)),
        "n_GOF": int((labels_3class == "GOF").sum()),
        "n_DN":  int((labels_3class == "DN").sum()),
        "n_LOF": int((labels_3class == "LOF").sum()),
        "model":       model_name,
        "model_scale": "650M" if "650M" in model_name else "3B",
        "seed":        seed,
    }

    print("\n=== Final results ===")
    for k, v in final_info.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(os.path.join(out_dir, f"final_info_seed{seed}.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    detailed = {
        "primary":            results_primary,
        "secondary":          results_secondary,
        "baselines":          results_baselines,
        "negative_controls":  results_negctrl,
        "orthogonality":      ortho_results,
        "variance_explained": var_exp,
        "stability_path":     stability_path,
        "stability_transfer_rho": transfer_rho,
        "family_cv":          results_family_cv,
        "pfam_n_families":    pfam_n_families,
    }
    with open(os.path.join(out_dir, f"detailed_results_seed{seed}.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    return final_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    type=str, default="run_0")
    parser.add_argument("--model",      type=str, default=ESM2_MODEL_650M,
                        choices=[ESM2_MODEL_650M, ESM2_MODEL_3B])
    parser.add_argument("--seeds",      type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    final_infos_list = []
    for seed in args.seeds:
        print(f"\n{'='*60}\n=== Seed {seed} ===\n{'='*60}")
        fi = run(args.out_dir, seed=seed, model_name=args.model,
                 batch_size=args.batch_size)
        final_infos_list.append(fi)

    numeric_keys = [k for k, v in final_infos_list[0].items()
                    if isinstance(v, (int, float)) and not np.isnan(v)]
    final_infos = {
        "means":   {k: float(np.mean([d[k] for d in final_infos_list
                                       if isinstance(d.get(k), (int, float))]))
                    for k in numeric_keys},
        "stderrs": {k: float(np.std([d[k] for d in final_infos_list
                                      if isinstance(d.get(k), (int, float))]) /
                              max(len(args.seeds), 1)**0.5)
                    for k in numeric_keys},
        "final_info_list": final_infos_list,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, {"seeds": final_infos_list})

    print(f"\nDone. Results written to {args.out_dir}/final_info.json")
