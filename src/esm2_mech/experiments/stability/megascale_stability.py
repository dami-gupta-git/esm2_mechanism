"""
Megascale stability as a second ESM-2 positive control (linear/Ridge probe).

Dataset: full Tsuboyama 2023 point-mutant set, natural domains only
(Tsuboyama2023_Dataset2_Dataset3_20230416.csv). ~177k single-point missense
variants across ~181 natural domains with physical ΔΔG labels — no curation
circularity. De novo designs are excluded (no Pfam family); parsing/scope is in
tsuboyama_loader.py. Family-split uses real Pfam families assigned by HMMER
(build_domain_families.py); domains with no Pfam hit are excluded from
family-split only. See for_me/explain_stability.md for the full rationale.

Pre-registered hypotheses (docs/plans/plan_megascale_stability.md):
  H1: Spearman ρ ≥ 0.5 under random split (stability encoded)
  H2: ρ drops ≤ 0.05 under family-split CV (family-robust)
  H3: Stability projected out of mechanism delta_mean does not lift
      family-split mechanism F1 on the merged mechanism dataset.
      Protocol: train Ridge on stability → predict stability score for merged
      variants → compute residuals of delta_mean ⊥ predicted stability
      (OLS projection-out, one component) → re-run family-split logreg.
  H4: Per-domain ρ std ≤ 0.10 (tight per-stratum distribution)

Decision table — ordered by informativeness, not by prior probability.
LEAKY and HETEROGENEOUS are the high-value outcomes; ROBUST is expected:

  LEAKY:         random ρ ≥ 0.5, family-split Δ ≥ 0.10  → stability signal partly family-memorisation;
                 analogous to mechanism leakage; would reshape central claim
  HETEROGENEOUS: random ρ ≥ 0.5, Δ ≤ 0.05, per-domain std ≥ 0.15  → works on average, fails on some domains;
                 matches result_18 AM/ProteinGym pattern; curation vs physical label distinction is real
  ROBUST:        random ρ ≥ 0.5, Δ ≤ 0.05, per-domain std ≤ 0.10  → expected; strengthens positive-control claim
  WEAK:          random ρ 0.3–0.5  → partial signal
  NULL:          random ρ < 0.3  → very unexpected; would undermine central framing

Companion nonlinear probe (Ridge/MLP/RF/GBM): megascale_mlp.py.

Usage (embeddings must already be extracted on GPU):
  cd esm2_mechanism
  python -m esm2_mech.experiments.stability.megascale_stability

Outputs:
  data/megascale_tsuboyama_variants.json
  data/megascale_domain_families.json
  data/embeddings/<model>/megascale_{wt,mut}_{mean,pos}.npy
  results/<run>/megascale_stability/summary.json
  results/<run>/megascale_stability/per_protein_spearman.json
  results/<run>/megascale_stability/h3_stability_projection.json
"""

import functools
import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from esm2_mech.experiments.mechanism.loaders import load_merged
from esm2_mech.experiments.stability.stability_data import (
    load_stability_inputs,
    stability_splits,
)
from esm2_mech.utils.constants import N_SEEDS, N_FOLDS
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n, standardize
from esm2_mech.utils.splits import family_split_cv
from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    PFAM_JSON,
    ESM2_MODEL,
)

OUT = str(_RESULTS_DIR / "megascale_stability")

os.makedirs(OUT, exist_ok=True)
os.makedirs(str(_DATA_DIR / "embeddings" / ESM2_MODEL), exist_ok=True)


# ---------------------------------------------------------------------------
# Ridge regression probe
# ---------------------------------------------------------------------------


def run_regression_cv(X, y, splits, clf_fn, with_pearson=True):
    """Standardise-fit-predict a regressor over CV folds; return ρ/AUROC (+Pearson).

    Generic over the estimator (clf_fn returns a fresh estimator per fold) so the
    Ridge linear probe here and the RF/GBM/XGBoost probes in megascale_mlp.py share
    one implementation. Pearson r is only meaningful for the linear probe, so the
    nonlinear callers pass with_pearson=False.
    """
    rhos, pearsons, aurocs = [], [], []
    for tr, te in splits:
        Xtr, Xte = standardize(X[tr], X[te])
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        rhos.append(float(rho))
        aurocs.append(auroc_at_median(y[te], pred))
        if with_pearson:
            pearson, _ = pearsonr(y[te], pred)
            pearsons.append(float(pearson))
    if not rhos:
        return {}
    rho_mean, rho_std, n_rho = mean_std_n(rhos)
    au_mean, au_std, _ = mean_std_n(aurocs)
    out = {
        "spearman_mean": rho_mean,
        "spearman_std": rho_std,
        "auroc_mean": au_mean,
        "auroc_std": au_std,
        "n_folds": n_rho,
    }
    if with_pearson:
        pearson_mean, pearson_std, _ = mean_std_n(pearsons)
        out["pearson_mean"] = pearson_mean
        out["pearson_std"] = pearson_std
    return out


def run_ridge_with_auroc(X, y, splits):
    """Linear (Ridge, alpha=1.0) stability probe — thin wrapper over run_regression_cv."""
    return run_regression_cv(X, y, splits, lambda: Ridge(alpha=1.0), with_pearson=True)


# ---------------------------------------------------------------------------
# Per-protein Spearman distribution
# ---------------------------------------------------------------------------


def per_protein_spearman(X, y, proteins):
    """
    For each protein with ≥5 variants, fit Ridge on all others, predict on that protein.
    Analogous to result_17/18 per-stratum AUROC distributions.
    """
    unique = sorted(set(proteins))
    results = {}
    for prot in unique:
        mask = proteins == prot
        if mask.sum() < 5:
            continue
        tr = np.where(~mask)[0]
        te = np.where(mask)[0]
        if len(tr) < 10:
            continue
        Xtr, Xte = standardize(X[tr], X[te])
        clf = Ridge(alpha=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, pval = spearmanr(y[te], pred)
        results[prot] = {
            "spearman": float(rho),
            "p_value": float(pval),
            "n_variants": int(mask.sum()),
        }
    return results


# ---------------------------------------------------------------------------
# H3: stability projection out of mechanism
# ---------------------------------------------------------------------------


def run_h3_stability_projection(
    merged_delta_mean,
    merged_labels,
    merged_proteins,
    pfam_map,
    stability_variants,
    stability_delta_mean,
    stability_ddg,
    n_folds=5,
    n_seeds=5,
):
    """
    Pre-registered H3 protocol:
      1. Train Ridge on the stability set (wt_mean, mut_mean) -> ΔΔG.
      2. Use that Ridge to predict a stability score for each merged-dataset variant
         from its delta_mean embedding.
      3. Project stability score out of merged delta_mean via OLS (one component):
         residuals = delta_mean - (delta_mean @ v) * v  where v is the unit vector
         of the stability Ridge weights (normalised).
      4. Re-run family-split logistic regression on residuals, 5 seeds.
      5. Compare to baseline family-split F1 on raw delta_mean.

    Returns dict with baseline_f1, projected_f1, delta_f1, and per-seed values.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score

    # Fit stability Ridge on the stability (Tsuboyama) set
    sc_s = StandardScaler()
    X_s = sc_s.fit_transform(stability_delta_mean)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, stability_ddg)

    # Stability projection vector: unit-normalised Ridge weights (in sc_s feature space)
    stability_weights = ridge.coef_  # shape (D,)
    stability_dir = stability_weights / (
        np.linalg.norm(stability_weights) + 1e-12
    )  # unit vector in sc_s-scaled feature space

    # Project stability out of merged delta_mean — must scale first to match sc_s space.
    # Both arms live in this single sc_s-standardised space and are NOT re-standardised
    # per fold: a per-fold StandardScaler rescales each column by its own std, which —
    # because `residuals` is rank-deficient along the dense vector stability_dir —
    # reintroduces variance along that direction (the very one we removed), silently
    # defeating the test. The projection must be the LAST transform the classifier sees.
    merged_scaled = sc_s.transform(merged_delta_mean.astype(np.float64)).astype(
        np.float32
    )
    proj = merged_scaled @ stability_dir  # (N,) scalar stability score per variant
    residuals = merged_scaled - np.outer(proj, stability_dir)

    # Verify the projection actually removed the stability direction: the residuals
    # must have ~zero variance along stability_dir (CLAUDE.md: verify
    # var(X_final @ v) ≈ 0 after projecting a direction out). The whole H3 test
    # hinges on this, so assert it rather than trusting the algebra.
    var_before = float(np.var(merged_scaled.astype(np.float64) @ stability_dir))
    var_after = float(np.var(residuals.astype(np.float64) @ stability_dir))
    if var_after > 1e-6 * var_before + 1e-8:
        raise AssertionError(
            f"stability projection failed: var along stability_dir was {var_before:.3e} "
            f"before and {var_after:.3e} after projecting out — the removed direction "
            "leaked back in, so the projected arm still contains stability signal."
        )

    le = LabelEncoder()
    y = le.fit_transform(merged_labels)

    baseline_f1s, projected_f1s = [], []
    for seed in range(n_seeds):
        splits = family_split_cv(merged_proteins, pfam_map, n_folds=n_folds, seed=seed)
        for X, tag in [(merged_scaled, "baseline"), (residuals, "projected")]:
            fold_f1s = []
            for tr, te in splits:
                # No per-fold StandardScaler: X is already sc_s-standardised, and
                # re-standardising would undo the projection (see note above). Both
                # arms get identical handling so the only difference is the projection.
                clf = LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                )
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                fold_f1s.append(
                    float(f1_score(y[te], pred, average="macro", zero_division=0))
                )
            if tag == "baseline":
                baseline_f1s.append(float(np.mean(fold_f1s)))
            else:
                projected_f1s.append(float(np.mean(fold_f1s)))

    return {
        "baseline_f1_mean": float(np.mean(baseline_f1s)),
        "baseline_f1_std": float(np.std(baseline_f1s)),
        "projected_f1_mean": float(np.mean(projected_f1s)),
        "projected_f1_std": float(np.std(projected_f1s)),
        "delta_f1": float(np.mean(projected_f1s) - np.mean(baseline_f1s)),
        "h3_passes": float(np.mean(projected_f1s))
        <= float(np.mean(baseline_f1s)) + 0.01,
    }


# ---------------------------------------------------------------------------
# Decision rule — ordered by informativeness
# ---------------------------------------------------------------------------


def apply_decision_rule(random_rho, protein_rho, per_prot_std):
    """
    Ordered by informativeness (most surprising first), not by prior probability.
    ROBUST is the expected outcome; LEAKY and HETEROGENEOUS are the high-value findings.
    """
    delta = random_rho - protein_rho
    # Check in order of informativeness
    if random_rho >= 0.5 and delta >= 0.10:
        return "LEAKY"
    if random_rho >= 0.5 and delta <= 0.05 and per_prot_std >= 0.15:
        return "HETEROGENEOUS"
    if random_rho >= 0.5 and delta <= 0.05 and per_prot_std <= 0.10:
        return "ROBUST"
    if 0.3 <= random_rho < 0.5:
        return "WEAK"
    if random_rho < 0.3:
        return "NULL"
    return f"INTERMEDIATE (rho={random_rho:.3f}, delta={delta:.3f}, std={per_prot_std:.3f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ── 1. Shared inputs: variants, ΔΔG, Pfam family map, embedding deltas ─────
    # The linear probe needs the per-residue delta too (include_pos=True). Orphan
    # domains (no Pfam hit) are absent from family_map → excluded from family-split.
    inputs = load_stability_inputs(include_pos=True)
    variants = inputs.variants
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    delta_mean = inputs.delta_mean
    delta_pos = inputs.delta_pos
    n_families = inputs.n_families

    print(f"Embeddings: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")

    # ── 4. Multi-seed CV ──────────────────────────────────────────────────────
    # Three schemes: random (in-distribution), domain-holdout (never train+test on
    # the same domain), family-holdout (never train+test on related Pfam families).
    results_by_seed = []
    for seed in range(N_SEEDS):
        print(f"\n── Seed {seed} ──")

        splits_by_name = stability_splits(seed, len(variants), proteins, family_map)

        seed_result = {"seed": seed}

        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in splits_by_name.items():
                key = f"{feat_name}_{split_name}"
                res = run_ridge_with_auroc(X, ddg, splits)
                seed_result[key] = res
                if res:
                    print(
                        f"  {key}: ρ={res['spearman_mean']:.3f}±{res['spearman_std']:.3f}  "
                        f"AUROC={res['auroc_mean']:.3f}"
                    )

        results_by_seed.append(seed_result)

    # ── 5. Per-protein Spearman distribution ──────────────────────────────────
    print("\nPer-protein Spearman (leave-one-protein-out)...")
    per_prot = per_protein_spearman(delta_mean, ddg, proteins)
    prot_rhos = [entry["spearman"] for entry in per_prot.values()]
    # spearmanr returns NaN for a protein whose held-out ΔΔG or predictions are
    # constant. Aggregate with mean_std_n (NaN-filtering) and guard min/max on the
    # finite subset — a single NaN must not poison per_prot_std, which feeds the
    # HETEROGENEOUS branch of the verdict (CLAUDE.md: NaN-guard every reducer).
    per_prot_mean, per_prot_std, n_finite_prot = mean_std_n(prot_rhos)
    finite_rhos = [rho for rho in prot_rhos if np.isfinite(rho)]
    if finite_rhos:
        print(
            f"  Per-protein ρ: mean={per_prot_mean:.3f}  std={per_prot_std:.3f}  "
            f"min={min(finite_rhos):.3f}  max={max(finite_rhos):.3f}  "
            f"n={n_finite_prot} (of {len(prot_rhos)} proteins with ≥5 variants)"
        )
    else:
        print("  Per-protein ρ: no protein yielded a finite ρ — skipped")

    atomic_write_json(os.path.join(OUT, "per_protein_spearman.json"), per_prot)

    # ── 6. Aggregate across seeds ─────────────────────────────────────────────
    summary = {}
    all_keys = set()
    for seed_result in results_by_seed:
        for key_name, value in seed_result.items():
            if isinstance(value, dict):
                all_keys.add(key_name)

    for key in sorted(all_keys):
        vals_rho = [
            sr[key]["spearman_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        vals_auroc = [
            sr[key]["auroc_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        if not vals_rho:
            continue
        rho_mean, rho_std, n_seeds_used = mean_std_n(vals_rho)
        au_mean, au_std, _ = mean_std_n(vals_auroc)
        summary[key] = {
            "spearman_mean": rho_mean,
            "spearman_std": rho_std,
            "auroc_mean": au_mean,
            "auroc_std": au_std,
            "n_seeds": n_seeds_used,
        }

    summary["per_protein"] = {
        "spearman_mean": per_prot_mean,
        "spearman_std": per_prot_std,
        "n_proteins": len(prot_rhos),
        "n_proteins_finite": n_finite_prot,
    }

    # ── 7. H3 — stability projection out of mechanism ─────────────────────────
    # Use the canonical mechanism loader, which labels every variant GOF/DN/LOF
    # (raising on an unexpected mechanism rather than defaulting to LOF) and
    # asserts the embeddings are row-aligned to the variant list — no fallback
    # labels, no blind length truncation.
    h3_result = None
    if all(
        os.path.exists(path)
        for path in [VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, PFAM_JSON]
    ):
        print("\nRunning H3 stability projection test...")
        with open(PFAM_JSON) as handle:
            pfam_map = json.load(handle)

        merged_delta, merged_labels, merged_proteins = load_merged()

        h3_result = run_h3_stability_projection(
            merged_delta,
            merged_labels,
            merged_proteins,
            pfam_map,
            variants,
            delta_mean,
            ddg,
            n_folds=N_FOLDS,
            n_seeds=N_SEEDS,
        )
        print(
            f"  H3: baseline F1={h3_result['baseline_f1_mean']:.3f}  "
            f"projected F1={h3_result['projected_f1_mean']:.3f}  "
            f"Δ={h3_result['delta_f1']:+.3f}  "
            f"passes={'YES' if h3_result['h3_passes'] else 'NO (stability direction is informative)'}"
        )
        atomic_write_json(
            os.path.join(OUT, "h3_stability_projection.json"), h3_result
        )
    else:
        print("\nSkipping H3 (merged embeddings not found — run on pod with full data)")

    # ── 8. Decision rule ──────────────────────────────────────────────────────
    # Pre-registered H2 tests robustness under FAMILY-holdout (random − family Δ).
    dm_random = summary.get("delta_mean_random", {}).get("spearman_mean", float("nan"))
    dm_family = summary.get("delta_mean_family", {}).get("spearman_mean", float("nan"))

    verdict = apply_decision_rule(dm_random, dm_family, per_prot_std)
    summary["verdict"] = verdict
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_families"] = n_families
    summary["n_seeds"] = N_SEEDS
    summary["h3"] = h3_result

    print(f"\n{'='*60}")
    print(
        f"VERDICT: {verdict}  (ordered by informativeness: LEAKY > HETEROGENEOUS > ROBUST > WEAK > NULL)"
    )
    print(f"  delta_mean random ρ  : {dm_random:.3f}  (H1 threshold ≥ 0.5)")
    print(f"  delta_mean family ρ  : {dm_family:.3f}")
    print(f"  Δ (random − family)  : {dm_random - dm_family:.3f}  (LEAKY if Δ ≥ 0.10)")
    print(f"  per-domain ρ std     : {per_prot_std:.3f}  (HETEROGENEOUS if ≥ 0.15)")
    if h3_result:
        print(
            f"  H3 Δ mechanism F1    : {h3_result['delta_f1']:+.3f}  "
            f"(passes if ≤ +0.01 — stability projection doesn't help mechanism)"
        )
    print(f"{'='*60}")

    atomic_write_json(os.path.join(OUT, "summary.json"), summary)
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
