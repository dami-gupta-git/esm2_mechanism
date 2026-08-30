"""ESM-2 delta-embedding mechanism geometry experiment (GOF/DN/LOF probing
with stability subspace removal).
"""

import hashlib
import json
import os
import warnings
import functools

import numpy as np

print = functools.partial(print, flush=True)
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.decomposition import PCA

from esm2_mech.experiments.stability.stability_data import load_stability_inputs
from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_logreg_cv
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.metrics import null_standard_score
from esm2_mech.utils.io import (
    atomic_write_json,
    load_json_or_discard,
    load_npy_or_discard,
    save_npy,
)
from esm2_mech.utils.embed import unpack_run_data
from esm2_mech.utils.sequences import (
    apply_missense,
    build_wt_mut_onehot,
    window_sequence,
)
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    STABILITY_SUBSPACE,
    STABILITY_SUBSPACE_PARAMS_JSON,
    RESULTS_DIR,
    VARIANTS_JSON,
    SEQUENCES_JSON,
    ALPHAMISSENSE_SCORES_JSON,
    PFAM_JSON,
    ESM2_MODEL,
)

warnings.filterwarnings("ignore")

ESM2_MODEL_3B = "esm2_t36_3B_UR50D"

CLASSES_3 = MECHANISM_CLASSES

STABILITY_TRANSFER_RHO_THRESHOLD = 0.3
SCALE_INVARIANT_THRESHOLD = 0.03
SCALE_EMERGENT_THRESHOLD = 0.05
VARIANCE_ASYMMETRY_THRESHOLD = 0.30
BENIGN_LEAK_THRESHOLD = 0.50


def _complete_contract(labels, groups, splits, classes, requested_folds, held_out_unit):
    return validate_complete_classification_splits(
        splits,
        requested_folds=requested_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=labels,
        classes=classes,
        groups=groups,
        held_out_unit=held_out_unit,
    )


def stability_subspace_fingerprint(ddg, n_components):
    """Content+mtime fingerprint of the Megascale inputs for cache validation."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(ddg, dtype=np.float64).tobytes())
    for path in (MEGASCALE_EMB_WT_MEAN, MEGASCALE_EMB_MUT_MEAN):
        stat = path.stat()
        digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return {
        "n_variants": int(len(ddg)),
        "n_components": int(n_components),
        "inputs_digest": digest.hexdigest(),
    }


def load_cached_stability_subspace(fingerprint):
    """Return the cached subspace if it matches `fingerprint`, else None."""
    if not os.path.exists(STABILITY_SUBSPACE):
        return None
    recorded = load_json_or_discard(STABILITY_SUBSPACE_PARAMS_JSON)
    if recorded is None:
        print(f"  {STABILITY_SUBSPACE} has no readable sidecar — refitting")
        return None
    if recorded != fingerprint:
        print(
            f"  {STABILITY_SUBSPACE} was fitted from different Megascale inputs "
            f"({recorded} != {fingerprint}) — refitting"
        )
        return None
    cached = load_npy_or_discard(STABILITY_SUBSPACE)
    if cached is None:
        print(f"  {STABILITY_SUBSPACE} unreadable — refitting")
        return None
    print("Loading cached stability subspace...")
    return cached


def fit_stability_subspace_megascale(n_components=10):
    """Fit stability subspace on Megascale data. Returns (n_components, D) or None."""
    try:
        variants = load_tsuboyama_variants()
        ddg = np.array([variant["ddg"] for variant in variants], dtype=np.float64)
        fingerprint = stability_subspace_fingerprint(ddg, n_components)
    except FileNotFoundError as exc:
        print(
            f"Megascale stability inputs unavailable ({exc}) — "
            f"will fit subspace on Gerasimavicius data"
        )
        return None

    cached = load_cached_stability_subspace(fingerprint)
    if cached is not None:
        return cached

    inputs = load_stability_inputs()

    print("Fitting stability subspace on Megascale delta embeddings...")
    deltas = inputs.delta_mean
    if len(inputs.ddg) != len(ddg):
        raise ValueError(
            f"ΔΔG row count changed between the cache-key read ({len(ddg)}) and "
            f"the full load ({len(inputs.ddg)}) — Megascale inputs are unstable."
        )

    from sklearn.linear_model import Ridge

    reg = Ridge(alpha=1.0)
    reg.fit(ddg.reshape(-1, 1), deltas)
    coefs = np.array(reg.coef_).flatten()

    stability_dir = coefs / (np.linalg.norm(coefs) + 1e-10)

    deltas_res = deltas - deltas.dot(stability_dir)[:, None] * stability_dir
    pca = PCA(
        n_components=min(n_components - 1, deltas_res.shape[0] - 1, deltas_res.shape[1])
    )
    pca.fit(deltas_res)

    subspace = np.vstack([stability_dir.reshape(1, -1), pca.components_])
    subspace = subspace[:n_components]

    # Array first, then sidecar: interrupt leaves a miss, not a stale-vouched array.
    save_npy(STABILITY_SUBSPACE, subspace)
    atomic_write_json(STABILITY_SUBSPACE_PARAMS_JSON, fingerprint, indent=2)
    return subspace


def fit_stability_subspace_direct(deltas, foldx_ddg, n_components=10, genes=None):
    """Fit stability subspace directly using FoldX ΔΔG. Fallback for Megascale."""
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
    """Spearman rho of Megascale subspace projection vs FoldX ΔΔG."""
    valid = ~np.isnan(foldx_ddg_geras)
    if valid.sum() < 20:
        return 0.0
    proj = deltas_geras[valid].dot(subspace[0])
    rho, _ = spearmanr(proj, foldx_ddg_geras[valid])
    return float(rho)


def select_stability_subspace(
    megascale_subspace, deltas, foldx_ddg, genes, n_components
):
    """Choose between Megascale-transfer (Path A) and direct-fit (Path B) subspaces.

    Returns (subspace, stability_path, transfer_rho).
    """
    transfer_rho = float("nan")

    if megascale_subspace is not None:
        transfer_rho = validate_stability_transfer(
            megascale_subspace, deltas, foldx_ddg
        )
        print(f"  Megascale→Gerasimavicius transfer Spearman ρ = {transfer_rho:.3f}")
        print(f"  Transfer threshold: ρ > {STABILITY_TRANSFER_RHO_THRESHOLD}")

        if transfer_rho >= STABILITY_TRANSFER_RHO_THRESHOLD:
            print("  Path A: Megascale transfer PASSES — using Megascale subspace")
            return megascale_subspace, "A_megascale", transfer_rho

        print("  Path A: Megascale transfer FAILS — falling back to Path B")
    else:
        print(
            "  Path B: No Megascale data — fitting subspace directly on Gerasimavicius"
        )

    subspace = fit_stability_subspace_direct(
        deltas, foldx_ddg, n_components=n_components, genes=genes
    )
    return subspace, "B_direct", transfer_rho


def project_out_subspace(deltas, subspace):
    """Remove the subspace from delta embeddings."""
    if subspace is None:
        return deltas
    Q, _ = np.linalg.qr(subspace.T, mode="reduced")
    proj = deltas.dot(Q).dot(Q.T)
    return deltas - proj


def standardize_once(deltas):
    """Standardize columns once, up front, returning (Z, scale).

    The probes that consume the projected deltas run with prescaled=True, so this
    is the only standardization applied. Doing it here rather than per fold is
    what makes the projection the last transform before the classifier: a per-fold
    StandardScaler applied after projection rescales each column independently and
    reintroduces variance along the removed directions.

    Constant columns get scale 1 (matching StandardScaler): centering already
    sends them to exactly zero, so no value is invented.
    """
    scale = deltas.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (deltas - deltas.mean(axis=0)) / scale, scale


def subspace_in_standardized_coords(subspace, scale):
    """Re-express a raw-space subspace in the standardized coordinates of `scale`.

    A row x maps to z = (x - mean) / scale, so x·v = z·(scale ⊙ v) + const. Removing
    direction (scale ⊙ v) from z therefore removes the raw stability coordinate x·v.
    Projecting the untransformed v out of z would remove a different direction and
    leave the stability signal partly intact.
    """
    if subspace is None:
        return None
    scaled = subspace * scale[None, :]
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(
            "subspace direction has zero norm in standardized coordinates — "
            "it lies entirely in constant feature columns"
        )
    return scaled / norms


def assert_subspace_removed(deltas_proj, subspace, name, tol=1e-8):
    """Verify no variance survives along `subspace` after projection.

    This is the invariant the whole projected-vs-unprojected comparison rests on;
    a silent failure here makes the "projected" arm test nothing.
    """
    if subspace is None:
        return 0.0
    Q, _ = np.linalg.qr(subspace.T, mode="reduced")
    residual_var = float(np.var(deltas_proj.dot(Q), axis=0).max())
    print(
        f"  {name}: max variance along stability subspace after projection = {residual_var:.3e}"
    )
    if residual_var > tol:
        raise ValueError(
            f"{name}: projection failed — {residual_var:.3e} variance remains along "
            f"the stability subspace (tolerance {tol:.0e})"
        )
    return residual_var


def variance_explained_per_class(deltas, labels_3class, subspace):
    """
    Report fraction of variance explained by stability subspace per mechanism class.
    Prediction: GOF has ≥ 30% less variance explained than HI+AR (LOF).
    """
    if subspace is None:
        return {}

    Q, _ = np.linalg.qr(subspace.T, mode="reduced")
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
        results["asymmetry_prediction_holds"] = bool(
            asymmetry >= VARIANCE_ASYMMETRY_THRESHOLD
        )

    return results


# ---------------------------------------------------------------------------
# Probe direction orthogonality
# ---------------------------------------------------------------------------


def probe_direction_orthogonality(
    X, y, genes, stability_subspace, n_folds=5, seed=42, n_shuffle=50
):
    """
    Fit pairwise LR probes (GOF-vs-DN, GOF-vs-LOF, DN-vs-LOF) and compute
    cosine similarity between probe weight vectors. Compare to shuffled-label null.
    """
    splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
    all_train_idx = np.unique(np.concatenate([tr for tr, _ in splits]))

    X_train = X[all_train_idx]
    y_train = y[all_train_idx]

    classes = sorted(set(y_train))
    pairs = [(c1, c2) for i, c1 in enumerate(classes) for c2 in classes[i + 1 :]]

    def fit_pairwise_probes(y_labels):
        weights = {}
        for c1, c2 in pairs:
            mask = np.isin(y_labels, [c1, c2])
            if mask.sum() < 10:
                continue
            clf = LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
            )
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
            cosine_matrix[f"{k1}|{k2}"] = float(
                np.dot(probe_weights[k1], probe_weights[k2])
            )

    stability_cosines = {}
    if stability_subspace is not None:
        stab_dir = stability_subspace[0] / (
            np.linalg.norm(stability_subspace[0]) + 1e-10
        )
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

    # Every shuffle contributes one cosine per real probe pair. A shuffled fit
    # that failed leaves the draw set short, which the declared count catches.
    expected_null_draws = n_shuffle * len(cosine_matrix)
    distinguishable = {}
    null_summaries = {}
    for pair, real_cos in cosine_matrix.items():
        null_summary = null_standard_score(
            real_cos, null_cosines, expected_draws=expected_null_draws
        )
        null_summaries[pair] = null_summary
        z_score = null_summary["z_score"]
        distinguishable[pair] = None if z_score is None else bool(abs(z_score) > 2.0)

    first_null_summary = next(iter(null_summaries.values()), None)

    return {
        "cosine_matrix": cosine_matrix,
        "stability_cosines": stability_cosines,
        "null_cosine_mean": (
            None if first_null_summary is None else first_null_summary["null_mean"]
        ),
        "null_cosine_std": (
            None if first_null_summary is None else first_null_summary["null_draw_std"]
        ),
        "null_score_summaries": null_summaries,
        "distinguishable_from_null": distinguishable,
        "path": "A" if stability_subspace is not None else "B",
    }


# ---------------------------------------------------------------------------
# Baselines and negative controls
# ---------------------------------------------------------------------------


def run_baselines(
    embeddings_wt,
    foldx_ddg,
    y,
    genes,
    aa_wt_list,
    aa_mut_list,
    alphamissense_scores,
    seed=42,
):
    """Run four baselines under gene-split CV."""
    splits = gene_split_cv(genes, seed=seed)
    onehot = build_wt_mut_onehot(aa_wt_list, aa_mut_list)

    # FoldX and AlphaMissense baselines are restricted to variants with observed values.
    # Missing values are not imputed — a variant with no FoldX ΔΔG is excluded from
    # that baseline entirely so the probe only sees real measurements.
    foldx_mask = ~np.isnan(foldx_ddg)
    if foldx_mask.sum() >= 20:
        ddg_splits = gene_split_cv(genes[foldx_mask], seed=seed)
        ddg_feat = foldx_ddg[foldx_mask].reshape(-1, 1)
        foldx_config = (
            "foldx_ddg_only",
            ddg_feat,
            y[foldx_mask],
            ddg_splits,
            genes[foldx_mask],
            ddg_feat.std() > 0,
        )
    else:
        foldx_config = ("foldx_ddg_only", None, None, None, None, False)

    if alphamissense_scores is not None:
        am_mask = ~np.isnan(alphamissense_scores)
        if am_mask.sum() >= 20:
            am_splits = gene_split_cv(genes[am_mask], seed=seed)
            am_feat = alphamissense_scores[am_mask].reshape(-1, 1)
            am_config = (
                "alphamissense",
                am_feat,
                y[am_mask],
                am_splits,
                genes[am_mask],
                am_feat.std() > 0,
            )
        else:
            am_config = ("alphamissense", None, None, None, None, False)
    else:
        am_config = ("alphamissense", None, None, None, None, False)

    # Full-data baselines use the shared splits computed above
    full_configs = [
        ("wt_only", embeddings_wt, y, splits, genes, True),
        ("onehot_aa", onehot, y, splits, genes, True),
    ]

    results = {}
    for name, X_bl, y_bl, spl, split_groups, runnable in full_configs + [
        foldx_config,
        am_config,
    ]:
        if not runnable or X_bl is None:
            results[name] = {"note": f"{name} unavailable or zero-variance"}
            continue
        print(f"  Baseline: {name} (n={len(y_bl)})")
        contract = _complete_contract(y_bl, split_groups, spl, CLASSES_3, 5, "gene")
        results[name] = run_logreg_cv(
            X_bl, y_bl, spl, CLASSES_3, contract, seed=seed, label=name
        )
    return results


def run_negative_controls(deltas_mean, y, genes, seed=42):
    """Shuffle deltas across genes to verify signal collapses to chance."""
    rng = np.random.RandomState(seed)
    splits = gene_split_cv(genes, seed=seed)
    deltas_shuffled = deltas_mean[rng.permutation(len(deltas_mean))]
    print("  Negative control: shuffled deltas")
    contract = _complete_contract(y, genes, splits, CLASSES_3, 5, "gene")
    return {
        "shuffled_delta": run_logreg_cv(
            deltas_shuffled,
            y,
            splits,
            CLASSES_3,
            contract,
            seed=seed,
            label="shuffled_delta",
            prescaled=True,
        ),
    }


# ---------------------------------------------------------------------------
# Phase helpers for run()
# ---------------------------------------------------------------------------


def _load_alphamissense_scores(variants: list) -> np.ndarray:
    """Load AlphaMissense scores from fetch_data output (alphamissense_scores_full.json).

    Returns a float array of length len(variants), NaN where no score is available.
    Keys in the JSON are gene_aapos_aawt_aamut (same format used by fetch_annotations.py).
    """
    if not ALPHAMISSENSE_SCORES_JSON.exists():
        print(
            f"  WARNING: {ALPHAMISSENSE_SCORES_JSON} not found — AlphaMissense scores unavailable"
        )
        return np.full(len(variants), np.nan)
    try:
        with open(ALPHAMISSENSE_SCORES_JSON) as f:
            am_scores = json.load(f)
    except json.JSONDecodeError:
        print(
            f"  WARNING: corrupt {ALPHAMISSENSE_SCORES_JSON} — AlphaMissense scores unavailable"
        )
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


def _load_data():
    """Phase 1: load and filter the merged variant dataset."""
    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VARIANTS_JSON} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    with open(VARIANTS_JSON) as f:
        variants = json.load(f)
    variants = [
        v
        for v in variants
        if v.get("uniprot_id")
        and v.get("aa_wt")
        and v.get("aa_mut")
        and v.get("aa_pos", 0) > 0
    ]
    print(f"After filtering: {len(variants)} variants")
    return variants


def _prepare_sequences(variants):
    """Phase 2: load cached sequences and build WT/mutant pairs."""
    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(
            f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first"
        )
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_variants, wt_seqs, mut_seqs, var_positions = [], [], [], []
    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
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


def _load_embeddings():
    """Phase 3: load Gerasimavicius embeddings. Raises FileNotFoundError if missing."""
    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file missing: {path}\n")
    print("\n=== Loading embeddings ===")
    return (
        np.load(EMB_WT_MEAN),
        np.load(EMB_MUT_MEAN),
        np.load(EMB_WT_POS),
        np.load(EMB_MUT_POS),
    )


def _run_primary_probes(
    deltas_mean_proj,
    deltas_mean,
    deltas_pos_proj,
    deltas_pos,
    y,
    genes,
    n_cv_folds,
    seed,
):
    """Phase 5: run the four primary probe variants.

    All four inputs are standardized up front by run(); the projected and
    unprojected arms therefore receive identical preprocessing and differ only
    by the projection. prescaled=True keeps the per-fold scaler out of the way.
    """
    splits = gene_split_cv(genes, n_folds=n_cv_folds, seed=seed)
    probe_configs = [
        ("mean_pooled_projected", deltas_mean_proj),
        ("mean_pooled_unprojected", deltas_mean),
        ("per_residue_projected", deltas_pos_proj),
        ("per_residue_unprojected", deltas_pos),
    ]
    results = {}
    contract = _complete_contract(y, genes, splits, CLASSES_3, n_cv_folds, "gene")
    for name, X in probe_configs:
        print(f"  {name}:")
        results[name] = run_logreg_cv(
            X, y, splits, CLASSES_3, contract, seed=seed, label=name, prescaled=True
        )
    return results


def _run_secondary_probes(
    deltas_mean_proj, labels_4class, labels_3class, genes, n_cv_folds, seed
):
    """Phase 6: 4-class probe and HI-vs-AR probe."""
    results = {}
    classes_4 = ["GOF", "DN", "HI", "AR"]

    # y4/y2 stay string labels — run_logreg_cv keys on `classes` (strings).
    y4 = np.asarray(labels_4class)
    splits4 = gene_split_cv(genes, n_folds=n_cv_folds, seed=seed)
    contract4 = _complete_contract(y4, genes, splits4, classes_4, n_cv_folds, "gene")
    print("  4-class (GOF/DN/HI/AR):")
    results["four_class"] = run_logreg_cv(
        deltas_mean_proj,
        y4,
        splits4,
        classes_4,
        contract4,
        seed=seed,
        label="4class",
        prescaled=True,
    )

    hi_ar_mask = np.isin(labels_4class, ["HI", "AR"])
    if hi_ar_mask.sum() >= 20:
        y2 = np.asarray(labels_4class[hi_ar_mask])
        splits2 = gene_split_cv(genes[hi_ar_mask], n_folds=n_cv_folds, seed=seed)
        contract2 = _complete_contract(
            y2,
            genes[hi_ar_mask],
            splits2,
            ["AR", "HI"],
            n_cv_folds,
            "gene",
        )
        print("  HI vs AR (2-class):")
        results["hi_vs_ar"] = run_logreg_cv(
            deltas_mean_proj[hi_ar_mask],
            y2,
            splits2,
            ["AR", "HI"],
            contract2,
            seed=seed,
            label="hi_vs_ar",
            prescaled=True,
        )
    return results


def _run_family_cv(deltas_mean_proj, y, genes, valid_variants, n_cv_folds, seed):
    """Phase 7: gene-family-split CV using Pfam families."""
    if not PFAM_JSON.exists():
        raise FileNotFoundError(
            f"{PFAM_JSON} not found — run fetch_data/fetch_annotations --step pfam first"
        )
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    n_families = len(set(v for v in pfam_map.values() if v is not None))

    splits = family_split_cv(genes, pfam_map, n_folds=n_cv_folds, seed=seed)
    print(f"  Running family-split CV with {len(splits)} folds")
    family_groups = np.array([pfam_map.get(gene) for gene in genes], dtype=object)
    contract = _complete_contract(
        y, family_groups, splits, CLASSES_3, n_cv_folds, "family"
    )
    results = run_logreg_cv(
        deltas_mean_proj,
        y,
        splits,
        CLASSES_3,
        contract,
        seed=seed,
        label="family_cv",
        prescaled=True,
    )
    if results["status"] == "success":
        print(f"  Family-split macro-F1: {results['macro_f1_mean']:.3f}")
    else:
        print(f"  Family-split: {results['status']}")
    return results, pfam_map, n_families


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run(
    data,
    out_dir,
    seed=0,
    n_stability_components=10,
    n_cv_folds=5,
):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1–3. Extract pre-loaded data
    # ------------------------------------------------------------------
    data = unpack_run_data(data)
    valid_variants = data["valid_variants"]
    emb_wt_mean = data["emb_wt_mean"]
    emb_mut_mean = data["emb_mut_mean"]
    emb_wt_pos = data["emb_wt_pos"]
    emb_mut_pos = data["emb_mut_pos"]
    labels_3class = data["labels_3class"]
    labels_4class = data["labels_4class"]
    genes_arr = data["genes_arr"]
    foldx_ddg = data["foldx_ddg"]
    aa_wt_list = data["aa_wt_list"]
    aa_mut_list = data["aa_mut_list"]
    alphamissense_scores = data["alphamissense_scores"]
    deltas_mean = data["deltas_mean"]
    deltas_pos = data["deltas_pos"]

    # y3 stays string labels — run_logreg_cv keys on `classes` (CLASSES_3, strings).
    y3 = np.asarray(labels_3class)

    from collections import Counter

    print(f"3-class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    # ------------------------------------------------------------------
    # 4. Stability subspace
    # ------------------------------------------------------------------
    print("\n=== Stability subspace ===")
    megascale_subspace = fit_stability_subspace_megascale(
        n_components=n_stability_components,
    )
    stability_subspace, stability_path, transfer_rho = select_stability_subspace(
        megascale_subspace, deltas_mean, foldx_ddg, genes_arr, n_stability_components
    )

    # Variance explained is a raw-space quantity — computed before standardization.
    var_exp = variance_explained_per_class(
        deltas_mean, labels_3class, stability_subspace
    )
    print(f"  Variance explained by stability subspace: {var_exp}")

    # Standardize once, here, and project afterwards. The probes downstream run
    # with prescaled=True: a per-fold StandardScaler applied to already-projected
    # data would rescale each column independently and put variance back along the
    # removed directions. Both arms of every projected-vs-unprojected comparison
    # get this same standardization, so the projection is the only difference.
    deltas_mean_std, mean_scale = standardize_once(deltas_mean)
    deltas_pos_std, pos_scale = standardize_once(deltas_pos)

    subspace_mean_std = subspace_in_standardized_coords(stability_subspace, mean_scale)
    subspace_pos_std = subspace_in_standardized_coords(stability_subspace, pos_scale)

    deltas_mean_proj = project_out_subspace(deltas_mean_std, subspace_mean_std)
    deltas_pos_proj = project_out_subspace(deltas_pos_std, subspace_pos_std)

    residual_var_mean = assert_subspace_removed(
        deltas_mean_proj, subspace_mean_std, "mean_pooled"
    )
    residual_var_pos = assert_subspace_removed(
        deltas_pos_proj, subspace_pos_std, "per_residue"
    )

    # ------------------------------------------------------------------
    # 5. Primary probes
    # ------------------------------------------------------------------
    print("\n=== Primary linear probe (3-class: GOF/DN/LOF) ===")
    results_primary = _run_primary_probes(
        deltas_mean_proj,
        deltas_mean_std,
        deltas_pos_proj,
        deltas_pos_std,
        y3,
        genes_arr,
        n_cv_folds,
        seed,
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
        emb_wt_mean,
        foldx_ddg,
        y3,
        genes_arr,
        aa_wt_list,
        aa_mut_list,
        alphamissense_scores,
        seed=seed,
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
        deltas_mean_proj, y3, genes_arr, valid_variants, n_cv_folds, seed
    )

    # ------------------------------------------------------------------
    # 10. Probe direction orthogonality
    # ------------------------------------------------------------------
    print("\n=== Probe direction orthogonality ===")
    # Standardized-coordinate subspace: the probe weights it is compared against are
    # fitted on the standardized+projected deltas, so the raw-space basis would be
    # a different direction in that space.
    subspace_for_ortho = subspace_mean_std if stability_path == "A_megascale" else None
    ortho_results = probe_direction_orthogonality(
        deltas_mean_proj,
        labels_3class,
        genes_arr,
        stability_subspace=subspace_for_ortho,
        n_folds=n_cv_folds,
        seed=seed,
    )
    print(f"  Cosine matrix: {ortho_results['cosine_matrix']}")
    null_mean = ortho_results["null_cosine_mean"]
    null_std = ortho_results["null_cosine_std"]
    if null_mean is None or null_std is None:
        print("  Null cosine summary: unavailable")
    else:
        print(f"  Null cosine mean: {null_mean:.3f} ± {null_std:.3f} null-draw SD")

    # ------------------------------------------------------------------
    # 11. Compile and save results
    # ------------------------------------------------------------------
    primary_mean_proj = results_primary["mean_pooled_projected"]
    primary_per_proj = results_primary["per_residue_projected"]

    final_info = {
        "headline_macro_f1": primary_mean_proj.get("macro_f1_mean", float("nan")),
        "headline_auroc_GOF": primary_mean_proj.get("auroc_GOF_mean", float("nan")),
        "headline_auroc_DN": primary_mean_proj.get("auroc_DN_mean", float("nan")),
        "headline_auroc_LOF": primary_mean_proj.get("auroc_LOF_mean", float("nan")),
        "per_residue_macro_f1": primary_per_proj.get("macro_f1_mean", float("nan")),
        "per_residue_auroc_GOF": primary_per_proj.get("auroc_GOF_mean", float("nan")),
        "unprojected_macro_f1": results_primary["mean_pooled_unprojected"].get(
            "macro_f1_mean", float("nan")
        ),
        "stability_path": stability_path,
        "stability_transfer_rho": transfer_rho,
        "stability_residual_var_mean_pooled": residual_var_mean,
        "stability_residual_var_per_residue": residual_var_pos,
        "variance_explained_GOF": var_exp.get("GOF", float("nan")),
        "variance_explained_DN": var_exp.get("DN", float("nan")),
        "variance_explained_LOF": var_exp.get("LOF", float("nan")),
        "variance_asymmetry_gof_lof": var_exp.get("gof_lof_asymmetry", float("nan")),
        "variance_asymmetry_prediction_holds": var_exp.get(
            "asymmetry_prediction_holds", False
        ),
        "baseline_wt_only_macro_f1": results_baselines.get("wt_only", {}).get(
            "macro_f1_mean", float("nan")
        ),
        "baseline_foldx_macro_f1": results_baselines.get("foldx_ddg_only", {}).get(
            "macro_f1_mean", float("nan")
        ),
        "baseline_onehot_macro_f1": results_baselines.get("onehot_aa", {}).get(
            "macro_f1_mean", float("nan")
        ),
        "baseline_alphamissense_macro_f1": results_baselines.get(
            "alphamissense", {}
        ).get("macro_f1_mean", float("nan")),
        "neg_ctrl_shuffled_macro_f1": results_negctrl.get("shuffled_delta", {}).get(
            "macro_f1_mean", float("nan")
        ),
        "cosine_matrix": ortho_results["cosine_matrix"],
        "null_cosine_mean": ortho_results["null_cosine_mean"],
        "ortho_distinguishable_from_null": str(
            ortho_results["distinguishable_from_null"]
        ),
        "family_cv_macro_f1": results_family_cv.get("macro_f1_mean", float("nan")),
        "family_cv_auroc_GOF": results_family_cv.get("auroc_GOF_mean", float("nan")),
        "family_cv_auroc_DN": results_family_cv.get("auroc_DN_mean", float("nan")),
        "family_cv_auroc_LOF": results_family_cv.get("auroc_LOF_mean", float("nan")),
        "orthogonality_path": ortho_results.get("path", "unknown"),
        "n_variants": len(valid_variants),
        "n_genes": len(set(genes_arr)),
        "n_GOF": int((labels_3class == "GOF").sum()),
        "n_DN": int((labels_3class == "DN").sum()),
        "n_LOF": int((labels_3class == "LOF").sum()),
        "model": ESM2_MODEL,
        "model_scale": "650M" if "650M" in ESM2_MODEL else "3B",
        "seed": seed,
    }

    print("\n=== Final results ===")
    for k, v in final_info.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(os.path.join(out_dir, f"final_info_seed{seed}.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    detailed = {
        "primary": results_primary,
        "secondary": results_secondary,
        "baselines": results_baselines,
        "negative_controls": results_negctrl,
        "orthogonality": ortho_results,
        "variance_explained": var_exp,
        "stability_path": stability_path,
        "stability_transfer_rho": transfer_rho,
        "family_cv": results_family_cv,
        "pfam_n_families": pfam_n_families,
    }
    with open(os.path.join(out_dir, f"detailed_results_seed{seed}.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    return final_info


# This module is intended to be run via mechanism_delta_cv.py, which loads data
# once and passes it to run(). Direct execution is not supported.
