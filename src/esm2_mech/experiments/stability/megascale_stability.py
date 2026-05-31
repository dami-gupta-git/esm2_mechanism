"""
Megascale stability as a second ESM-2 positive control (result_21).

Dataset: S1724 benchmark (ThermoMutDB curated, 1,277 single-point missense
across 27 real PDB proteins). Physical ΔΔG labels — no curation circularity.
NOT the synthetic mini-protein portion of Megascale; all 27 proteins are
natural domains with Pfam coverage.

Pre-registered hypotheses (plan_megascale_stability.md):
  H1: Spearman ρ ≥ 0.5 under random split (stability encoded)
  H2: ρ drops ≤ 0.05 under protein-holdout CV (family-robust)
  H3: Stability projected out of mechanism delta_mean does not lift
      family-split mechanism F1 on merged Gerasimavicius dataset.
      Protocol: train Ridge on S1724 → predict stability score for merged
      variants → compute residuals of delta_mean ⊥ predicted stability
      (OLS projection-out, one component) → re-run family-split logreg.
  H4: Per-protein ρ std ≤ 0.10 (tight per-stratum distribution)

Decision table — ordered by informativeness, not by prior probability.
LEAKY and HETEROGENEOUS are the high-value outcomes; ROBUST is expected:

  LEAKY:         random ρ ≥ 0.5, protein-split Δ ≥ 0.10  → stability signal partly family-memorisation;
                 analogous to mechanism leakage; would reshape central claim
  HETEROGENEOUS: random ρ ≥ 0.5, Δ ≤ 0.05, per-prot std ≥ 0.15  → works on average, fails on some proteins;
                 matches result_18 AM/ProteinGym pattern; curation vs physical label distinction is real
  ROBUST:        random ρ ≥ 0.5, Δ ≤ 0.05, per-prot std ≤ 0.10  → expected; strengthens positive-control claim
  WEAK:          random ρ 0.3–0.5  → partial signal
  NULL:          random ρ < 0.3  → very unexpected; would undermine central framing

Usage (GPU required for embedding extraction):
  cd esm2_mechanism
  python scripts/megascale_stability.py

Outputs:
  data/megascale_variants.json
  data/embeddings/megascale_{wt,mut}_{mean,pos}.npy
  results/megascale_stability/summary.json
  results/megascale_stability/per_protein_spearman.json
  results/megascale_stability/h3_stability_projection.json
"""

import functools
import json
import os
import sys
import zipfile
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_POS,
    MEGASCALE_EMB_MUT_POS,
    ESM2_MODEL,
)

DATA = str(_DATA_DIR)
BM_ZIP = str(_DATA_DIR / "megascale" / "benchmarks.zip")
PFAM_JSON = str(_DATA_DIR / "pfam_families.json")
OUT = str(_RESULTS_DIR / "megascale_stability")

VARIANTS_CACHE = os.path.join(DATA, "megascale_variants.json")
WT_MEAN_EMB = MEGASCALE_EMB_WT_MEAN
MUT_MEAN_EMB = MEGASCALE_EMB_MUT_MEAN
WT_POS_EMB = MEGASCALE_EMB_WT_POS
MUT_POS_EMB = MEGASCALE_EMB_MUT_POS

N_SEEDS = 5
N_FOLDS = 5

os.makedirs(OUT, exist_ok=True)
os.makedirs(str(_DATA_DIR / "embeddings" / ESM2_MODEL), exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_s1724_variants():
    """
    Parse S1724 benchmark (ThermoMutDB curated, single-point missense).

    WT sequence: S1724_PDB_sequences.csv gabetrimmed column (keyed by PDB_chain e.g. '1AJ3_A').
    Mutation position: paper_seq_muts column, 1-indexed within gabetrimmed.
    Returns list of dicts: {protein, mutation_code, wt_seq, mut_seq, var_pos, ddg}
    """
    if os.path.exists(VARIANTS_CACHE):
        print("Loading cached S1724 variants...")
        with open(VARIANTS_CACHE) as f:
            return json.load(f)

    import re
    import pandas as pd

    mut_pat = re.compile(r"^([A-Z])(\d+)([A-Z])$")

    with zipfile.ZipFile(BM_ZIP) as z:
        with z.open("benchmarks/S1724_thermomutdb_cleaned_withseq.csv") as f:
            df = pd.read_csv(f)

    # Single-point missense with ΔΔG and parseable mutation
    # paper_seq stores the MUTANT sequence; paper_seq_muts gives the 1-indexed position
    # within paper_seq where the mutation was applied (paper_seq[var_pos-1] == aa_mut).
    # WT is reconstructed by reversing the mutation: replace aa_mut -> aa_wt at var_pos.
    df = df[
        (df["mutation_type"] == "Single")
        & df["ddg"].notna()
        & df["paper_seq"].notna()
        & df["paper_seq_muts"].notna()
    ].copy()

    variants = []
    skipped = {"bad_mut_pat": 0, "pos_out_of_range": 0, "mut_not_in_paper_seq": 0}
    for _, row in df.iterrows():
        protein = str(row["PDB_wild"]).strip()
        mut_str = str(row["paper_seq_muts"]).strip()
        m = mut_pat.match(mut_str)
        if not m:
            skipped["bad_mut_pat"] += 1
            continue
        aa_wt, pos_str, aa_mut = m.groups()
        var_pos = int(pos_str)  # 1-indexed in paper_seq

        paper_seq = str(row["paper_seq"]).strip()
        if var_pos < 1 or var_pos > len(paper_seq):
            skipped["pos_out_of_range"] += 1
            continue
        if paper_seq[var_pos - 1] != aa_mut:
            # paper_seq should have the mutant residue at var_pos
            skipped["mut_not_in_paper_seq"] += 1
            continue

        # Reconstruct WT by reversing the mutation
        wt_seq = paper_seq[: var_pos - 1] + aa_wt + paper_seq[var_pos:]
        mut_seq = paper_seq  # already the mutant

        variants.append(
            {
                "protein": protein,
                "mutation_code": mut_str,
                "wt_seq": wt_seq,
                "mut_seq": mut_seq,
                "var_pos": var_pos,
                "ddg": float(row["ddg"]),
            }
        )

    print(f"S1724: {len(variants)} single-point variants, skipped={skipped}")

    with open(VARIANTS_CACHE, "w") as f:
        json.dump(variants, f)
    print(f"Cached to {VARIANTS_CACHE}")
    return variants


# ---------------------------------------------------------------------------
# Protein-level clustering for family-split (MMseqs2 or sequence-identity)
# ---------------------------------------------------------------------------


def assign_protein_clusters(variants):
    """
    Group proteins by sequence identity for family-split CV.
    Uses exact-match on WT sequence as cluster ID (each unique WT = one protein domain).
    With only 33 proteins this is already a meaningful holdout.
    Returns: dict protein_id -> cluster_id (here: just protein_id itself, 1 cluster per protein)
    """
    # Each PDB_wild is one protein — cluster = protein identity.
    # For the family-split analogue we use MMseqs2 if available, else identity clustering.
    mmseqs_cache = os.path.join(DATA, "megascale_protein_clusters.json")
    if os.path.exists(mmseqs_cache):
        with open(mmseqs_cache) as f:
            return json.load(f)

    # Build unique WT sequences
    proteins = {}
    for v in variants:
        pid = v["protein"]
        if pid not in proteins:
            proteins[pid] = v["wt_seq"]

    # Try MMseqs2
    try:
        cluster_map = _run_mmseqs2(proteins)
    except Exception as e:
        print(f"MMseqs2 failed ({e}), falling back to identity clustering")
        cluster_map = {pid: pid for pid in proteins}

    with open(mmseqs_cache, "w") as f:
        json.dump(cluster_map, f)
    return cluster_map


def _run_mmseqs2(proteins, min_seq_id=0.20, coverage=0.20):
    """
    Run MMseqs2 easy-cluster on the 33 S1724 proteins.
    Returns dict: protein_id -> cluster_representative_id
    """
    import subprocess, tempfile, shutil

    if not shutil.which("mmseqs"):
        raise RuntimeError("mmseqs not found in PATH")

    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "seqs.fasta")
        with open(fasta, "w") as f:
            for pid, seq in proteins.items():
                f.write(f">{pid}\n{seq}\n")

        subprocess.run(
            [
                "mmseqs",
                "easy-cluster",
                fasta,
                os.path.join(tmp, "clust"),
                os.path.join(tmp, "mmseqs_tmp"),
                f"--min-seq-id",
                str(min_seq_id),
                "-c",
                str(coverage),
                "--cov-mode",
                "0",
                "-v",
                "0",
            ],
            check=True,
            capture_output=True,
        )

        tsv = os.path.join(tmp, "clust_cluster.tsv")
        cluster_map = {}
        with open(tsv) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                cluster_map[member] = rep

    return cluster_map


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CV splits
# ---------------------------------------------------------------------------


def random_split_cv(n, n_folds=5, seed=42):
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    splits = []
    for fold in np.array_split(idx, n_folds):
        te = fold
        tr = np.setdiff1d(idx, fold)
        splits.append((tr, te))
    return splits


def protein_split_cv(proteins, n_folds=5, seed=42):
    """Hold out whole proteins (analogous to gene-split)."""
    unique = np.array(sorted(set(proteins)))
    np.random.RandomState(seed).shuffle(unique)
    splits = []
    for fold_proteins in np.array_split(unique, n_folds):
        mask_te = np.isin(proteins, fold_proteins)
        tr = np.where(~mask_te)[0]
        te = np.where(mask_te)[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


def cluster_split_cv(proteins, cluster_map, n_folds=5, seed=42):
    """Hold out whole MMseqs2/identity clusters (analogous to family-split)."""
    prot_clusters = np.array([cluster_map.get(p, p) for p in proteins])
    unique_clusters = np.array(sorted(set(prot_clusters)))
    np.random.RandomState(seed).shuffle(unique_clusters)
    splits = []
    for fold_clusters in np.array_split(unique_clusters, n_folds):
        mask_te = np.isin(prot_clusters, fold_clusters)
        tr = np.where(~mask_te)[0]
        te = np.where(mask_te)[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


# ---------------------------------------------------------------------------
# Ridge regression probe
# ---------------------------------------------------------------------------


def auroc_at_median(y_true, y_pred):
    """Binary AUROC: above-median = positive."""
    med = np.median(y_true)
    binary = (y_true >= med).astype(int)
    if binary.sum() == 0 or (1 - binary).sum() == 0:
        return float("nan")
    return float(roc_auc_score(binary, y_pred))


def run_ridge_with_auroc(X, y, splits):
    rhos, rs, aurocs = [], [], []
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = Ridge(alpha=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        r, _ = pearsonr(y[te], pred)
        au = auroc_at_median(y[te], pred)
        rhos.append(float(rho))
        rs.append(float(r))
        aurocs.append(au)
    if not rhos:
        return {}
    return {
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std": float(np.std(rhos)),
        "pearson_mean": float(np.mean(rs)),
        "pearson_std": float(np.std(rs)),
        "auroc_mean": float(np.nanmean(aurocs)),
        "auroc_std": float(np.nanstd(aurocs)),
        "n_folds": len(rhos),
    }


# ---------------------------------------------------------------------------
# Per-protein Spearman distribution
# ---------------------------------------------------------------------------


def per_protein_spearman(X, y, proteins, use_delta_mean=True):
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
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
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
    s1724_variants,
    s1724_delta_mean,
    s1724_ddg,
    n_folds=5,
    n_seeds=5,
):
    """
    Pre-registered H3 protocol:
      1. Train Ridge on S1724 (wt_mean, mut_mean) -> ΔΔG.
      2. Use that Ridge to predict a stability score for each merged-dataset variant
         from its delta_mean embedding.
      3. Project stability score out of merged delta_mean via OLS (one component):
         residuals = delta_mean - (delta_mean @ v) * v  where v is the unit vector
         of the stability Ridge weights (normalised).
      4. Re-run family-split logistic regression on residuals, 5 seeds.
      5. Compare to baseline family-split F1 on raw delta_mean.

    Returns dict with baseline_f1, projected_f1, delta_f1, and per-seed values.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score
    from esm2_mech.utils.splits import family_split_cv

    # Fit stability Ridge on S1724
    sc_s = StandardScaler()
    X_s = sc_s.fit_transform(s1724_delta_mean)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, s1724_ddg)

    # Stability projection vector: unit-normalised Ridge weights (in sc_s feature space)
    w = ridge.coef_  # shape (D,)
    v = w / (np.linalg.norm(w) + 1e-12)  # unit vector in sc_s-scaled feature space

    # Project stability out of merged delta_mean — must scale first to match sc_s space
    merged_scaled = sc_s.transform(merged_delta_mean.astype(np.float64)).astype(
        np.float32
    )
    proj = merged_scaled @ v  # (N,) scalar stability score per variant
    residuals = merged_scaled - np.outer(proj, v)

    le = LabelEncoder()
    y = le.fit_transform(merged_labels)

    baseline_f1s, projected_f1s = [], []
    for seed in range(n_seeds):
        splits = family_split_cv(merged_proteins, pfam_map, n_folds=n_folds, seed=seed)
        for X, tag in [(merged_scaled, "baseline"), (residuals, "projected")]:
            fold_f1s = []
            for tr, te in splits:
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr].astype(np.float32))
                Xte = sc.transform(X[te].astype(np.float32))
                clf = LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=seed,
                )
                clf.fit(Xtr, y[tr])
                pred = clf.predict(Xte)
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
    # ── 1. Load variants ──────────────────────────────────────────────────────
    variants = load_s1724_variants()
    print(
        f"Loaded {len(variants)} S1724 variants across "
        f"{len(set(v['protein'] for v in variants))} proteins"
    )

    proteins = np.array([v["protein"] for v in variants])
    ddg = np.array([v["ddg"] for v in variants])

    # ── 2. Cluster assignment for family-split analogue ───────────────────────
    cluster_map = assign_protein_clusters(variants)
    n_clusters = len(set(cluster_map.values()))
    print(f"Protein clusters: {len(set(proteins))} proteins → {n_clusters} clusters")

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    for path in [WT_MEAN_EMB, MUT_MEAN_EMB, WT_POS_EMB, MUT_POS_EMB]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file missing: {path}\n"
                f"Run: python -m esm2_mech.embeddings.embed_variants --model {ESM2_MODEL}"
            )
    print("Loading embeddings...")
    wt_mean = np.load(WT_MEAN_EMB)
    mut_mean = np.load(MUT_MEAN_EMB)
    wt_pos = np.load(WT_POS_EMB)
    mut_pos = np.load(MUT_POS_EMB)

    delta_mean = mut_mean - wt_mean
    delta_pos = mut_pos - wt_pos

    print(f"Embeddings: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")

    # ── 4. Multi-seed CV ──────────────────────────────────────────────────────
    results_by_seed = []
    for seed in range(N_SEEDS):
        print(f"\n── Seed {seed} ──")

        splits_random = random_split_cv(len(variants), N_FOLDS, seed)
        splits_protein = protein_split_cv(proteins, N_FOLDS, seed)
        splits_cluster = cluster_split_cv(proteins, cluster_map, N_FOLDS, seed)

        seed_result = {"seed": seed}

        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in [
                ("random", splits_random),
                ("protein", splits_protein),
                ("cluster", splits_cluster),
            ]:
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
    prot_rhos = [v["spearman"] for v in per_prot.values()]
    per_prot_std = float(np.std(prot_rhos)) if prot_rhos else float("nan")
    per_prot_mean = float(np.mean(prot_rhos)) if prot_rhos else float("nan")

    print(
        f"  Per-protein ρ: mean={per_prot_mean:.3f}  std={per_prot_std:.3f}  "
        f"min={min(prot_rhos):.3f}  max={max(prot_rhos):.3f}  "
        f"n={len(prot_rhos)}"
    )

    with open(os.path.join(OUT, "per_protein_spearman.json"), "w") as f:
        json.dump(per_prot, f, indent=2)

    # ── 6. Aggregate across seeds ─────────────────────────────────────────────
    summary = {}
    all_keys = set()
    for sr in results_by_seed:
        for k, v in sr.items():
            if isinstance(v, dict):
                all_keys.add(k)

    for key in sorted(all_keys):
        vals_rho = [
            sr[key]["spearman_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        vals_auroc = [
            sr[key]["auroc_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        if not vals_rho:
            continue
        summary[key] = {
            "spearman_mean": float(np.mean(vals_rho)),
            "spearman_std": float(np.std(vals_rho)),
            "auroc_mean": float(np.nanmean(vals_auroc)),
            "auroc_std": float(np.nanstd(vals_auroc)),
        }

    summary["per_protein"] = {
        "spearman_mean": per_prot_mean,
        "spearman_std": per_prot_std,
        "n_proteins": len(prot_rhos),
    }

    # ── 7. H3 — stability projection out of mechanism ─────────────────────────
    h3_result = None
    merged_variants_path = VALID_VARIANTS_JSON
    merged_wt_path = EMB_WT_MEAN
    merged_mut_path = EMB_MUT_MEAN
    if all(
        os.path.exists(p)
        for p in [merged_variants_path, merged_wt_path, merged_mut_path, PFAM_JSON]
    ):
        print("\nRunning H3 stability projection test...")
        with open(merged_variants_path) as f:
            merged_variants = json.load(f)
        with open(PFAM_JSON) as f:
            pfam_map = json.load(f)
        merged_wt = np.load(merged_wt_path)
        merged_mut = np.load(merged_mut_path)
        merged_delta = merged_mut - merged_wt

        label_map = {"GOF": "GOF", "DN": "DN", "HI": "LOF", "AR": "LOF", "LOF": "LOF"}
        merged_labels = np.array(
            [
                label_map.get(v.get("mechanism", v.get("label", "")), "LOF")
                for v in merged_variants
            ]
        )
        merged_proteins = np.array(
            [v.get("gene", v.get("protein", "")) for v in merged_variants]
        )

        # Align lengths: merged embeddings may differ from variant list if some were dropped
        n_min = min(len(merged_delta), len(merged_labels))
        merged_delta = merged_delta[:n_min]
        merged_labels = merged_labels[:n_min]
        merged_proteins = merged_proteins[:n_min]

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
        with open(os.path.join(OUT, "h3_stability_projection.json"), "w") as f:
            json.dump(h3_result, f, indent=2)
    else:
        print("\nSkipping H3 (merged embeddings not found — run on pod with full data)")

    # ── 8. Decision rule ──────────────────────────────────────────────────────
    dm_random = summary.get("delta_mean_random", {}).get("spearman_mean", float("nan"))
    dm_protein = summary.get("delta_mean_protein", {}).get(
        "spearman_mean", float("nan")
    )

    verdict = apply_decision_rule(dm_random, dm_protein, per_prot_std)
    summary["verdict"] = verdict
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_clusters"] = n_clusters
    summary["n_seeds"] = N_SEEDS
    summary["h3"] = h3_result

    print(f"\n{'='*60}")
    print(
        f"VERDICT: {verdict}  (ordered by informativeness: LEAKY > HETEROGENEOUS > ROBUST > WEAK > NULL)"
    )
    print(f"  delta_mean random ρ  : {dm_random:.3f}  (H1 threshold ≥ 0.5)")
    print(f"  delta_mean protein ρ : {dm_protein:.3f}")
    print(f"  Δ (random − protein) : {dm_random - dm_protein:.3f}  (LEAKY if Δ ≥ 0.10)")
    print(f"  per-protein ρ std    : {per_prot_std:.3f}  (HETEROGENEOUS if ≥ 0.15)")
    if h3_result:
        print(
            f"  H3 Δ mechanism F1    : {h3_result['delta_f1']:+.3f}  "
            f"(passes if ≤ +0.01 — stability projection doesn't help mechanism)"
        )
    print(f"{'='*60}")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
