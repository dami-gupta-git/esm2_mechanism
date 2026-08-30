"""In-silico perturbation scan for mechanism prediction.

Mutates probe positions to Ala/Asp/Trp, extracts ESM-2 deltas, and computes per-gene scalar features.
"""

import argparse, json, os, sys, numpy as np
import functools

print = functools.partial(print, flush=True)
from collections import defaultdict
from pathlib import Path

from esm2_mech.utils.paths import (
    SCAN_FEATURES_NPY,
    SCAN_FEATURES_META_JSON,
    RESULTS_DIR as _RESULTS_DIR,
    SCAN_EMB_MUT,
    SCAN_EMB_WT,
    SCAN_PROBE_CACHE_JSON,
    SEQUENCES_EXTENDED_JSON,
    SEQUENCES_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.sequences import window_sequence, apply_missense

OUT = _RESULTS_DIR / "perturbation_scan"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_AAS = ["A", "D", "W"]  # Ala, Asp, Trp
PROBE_NAMES = ["ala", "asp", "trp"]
N_POSITIONS = 100  # evenly-spaced positions per gene
MIN_GENE_LEN = 10  # skip very short sequences


def load_sequences():
    """Merge sequences.json + extended cache."""
    with open(SEQUENCES_JSON) as f:
        seqs = json.load(f)
    if SEQUENCES_EXTENDED_JSON.exists():
        with open(SEQUENCES_EXTENDED_JSON) as f:
            seqs.update(json.load(f))
    print(f"Sequences loaded: {len(seqs)}")
    return seqs


def build_probe_list(seqs):
    """Sample N_POSITIONS per gene and create Ala/Asp/Trp probes."""
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)

    gene_to_uniprot = {}
    for v in variants:
        g = v.get("gene", "").upper()
        u = v.get("uniprot_id", "")
        if g and u:
            gene_to_uniprot[g] = u

    probes = []
    covered_genes = []
    missing_genes = []

    for gene, uniprot_id in sorted(gene_to_uniprot.items()):
        if uniprot_id not in seqs:
            missing_genes.append(gene)
            continue
        seq = seqs[uniprot_id]
        L = len(seq)
        if L < MIN_GENE_LEN:
            continue

        n_pos = min(N_POSITIONS, L)
        positions = np.linspace(1, L, n_pos, dtype=int).tolist()
        positions = sorted(set(positions))

        for pos in positions:
            wt_aa = seq[pos - 1]
            for probe_aa, probe_name in zip(PROBE_AAS, PROBE_NAMES):
                if probe_aa == wt_aa:
                    continue
                probes.append(
                    {
                        "gene": gene,
                        "uniprot_id": uniprot_id,
                        "aa_pos": int(pos),
                        "aa_wt": wt_aa,
                        "aa_mut": probe_aa,
                        "probe_name": probe_name,
                        "seq_len": L,
                    }
                )
        covered_genes.append(gene)

    print(f"Covered genes: {len(covered_genes)} / {len(gene_to_uniprot)}")
    print(f"Missing sequences: {len(missing_genes)}")
    print(f"Total probes: {len(probes)}")
    return probes, covered_genes


def _load_scan_embeddings():
    """Load scan embeddings. Raises FileNotFoundError if not found."""
    for path in [SCAN_EMB_WT, SCAN_EMB_MUT]:
        if not path.exists():
            raise FileNotFoundError(
                f"Scan embedding file missing: {path}\n"
                f"Run: python -m esm2_mech.embeddings.embed_scan"
            )
    print(f"Loading scan embeddings: {SCAN_EMB_WT}")
    return np.load(SCAN_EMB_WT), np.load(SCAN_EMB_MUT)


def _top_explained_variance_ratios(mat: np.ndarray, k: int = 2) -> np.ndarray:
    """Top-k PCA explained-variance ratios via torch SVD."""
    import torch

    centered = torch.from_numpy(np.ascontiguousarray(mat, dtype=np.float32))
    centered = centered - centered.mean(0, keepdim=True)
    svals = torch.linalg.svdvals(centered)
    variances = svals.square()
    total = variances.sum()
    ratios = (variances / total).cpu().numpy()
    return ratios[:k]


def compute_scan_features(probes, wt_emb, mut_emb, covered_genes, ablation=False):
    """Build per-gene feature vectors from probe embedding deltas."""
    deltas = mut_emb - wt_emb  # (n_probes, 1280)

    gene_probe_idx = defaultdict(list)
    for i, p in enumerate(probes):
        gene_probe_idx[p["gene"]].append(i)

    feature_names = [
        "scan_mag_mean",
        "scan_mag_cv",
        "scan_hotspot_frac",
        "scan_pc1_var",
        "scan_sub_variance",
    ]
    if ablation:
        feature_names += [
            "scan_mag_skew",
            "scan_hotspot_spacing_cv",
            "scan_top5_range",
            "scan_pc1_pc2_ratio",
        ]

    gene_list, X = [], []

    for gene in covered_genes:
        idxs = gene_probe_idx.get(gene, [])
        if len(idxs) < 3:
            continue

        gene_probes = [probes[i] for i in idxs]
        gene_deltas = deltas[idxs]  # (n, 1280)

        mags = np.linalg.norm(gene_deltas, axis=1)

        # --- Per-gene scan features ---
        mag_mean = float(np.mean(mags))
        mag_std = float(np.std(mags))
        mag_cv = mag_std / (mag_mean + 1e-8)

        threshold = mag_mean + mag_std
        hotspot_frac = float(np.mean(mags > threshold))

        # PC1 variance fraction
        if len(idxs) >= 4:
            ratios = _top_explained_variance_ratios(gene_deltas, k=2)
            pc1_var = float(ratios[0])
            pc2_var = float(ratios[1]) if len(ratios) > 1 else 0.0
        else:
            pc1_var, pc2_var = 0.0, 0.0

        # Per-position substitution variance
        # Group by position, compute variance of magnitudes across probes at that position
        pos_to_mags = defaultdict(list)
        for j, p in enumerate(gene_probes):
            pos_to_mags[p["aa_pos"]].append(mags[j])
        sub_vars = [np.var(v) for v in pos_to_mags.values() if len(v) >= 2]
        scan_sub_variance = float(np.mean(sub_vars)) if sub_vars else 0.0

        feats = [mag_mean, mag_cv, hotspot_frac, pc1_var, scan_sub_variance]

        # --- Ablation features ---
        if ablation:
            from scipy.stats import skew as scipy_skew

            mag_skew = float(scipy_skew(mags))

            hotspot_positions = np.where(mags > threshold)[0]
            if len(hotspot_positions) >= 2:
                spacings = np.diff(hotspot_positions)
                spacing_cv = float(np.std(spacings) / (np.mean(spacings) + 1e-8))
            else:
                spacing_cv = 0.0

            top5_idx = np.argsort(mags)[-5:]
            top5_positions = np.array([gene_probes[i]["aa_pos"] for i in top5_idx])
            seq_len = gene_probes[0]["seq_len"]
            top5_range = float(
                (top5_positions.max() - top5_positions.min()) / max(seq_len, 1)
            )

            pc1_pc2_ratio = float(pc1_var / (pc2_var + 1e-8))

            feats += [mag_skew, spacing_cv, top5_range, pc1_pc2_ratio]

        gene_list.append(gene)
        X.append(feats)

    gene_list = np.array(gene_list)
    X = np.array(X, dtype=np.float32)
    print(f"Gene features built: {len(gene_list)} genes × {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    return gene_list, X, feature_names


def save_features(gene_list, X, feature_names):
    np.save(SCAN_FEATURES_NPY, X)
    meta = {"genes": gene_list.tolist(), "feature_names": feature_names}
    with open(SCAN_FEATURES_META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved scan_features.npy ({X.shape})")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_phase",
        default="13",
        help="Which phases to run: '1', '3', '13' (default: all). "
             "For embedding extraction (phase 2) run: python -m esm2_mech.embeddings.embed_scan",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also compute ablation features (phase 3 only)",
    )
    args = parser.parse_args()

    phases = set(args.run_phase)

    seqs = load_sequences()

    probe_cache = SCAN_PROBE_CACHE_JSON
    if probe_cache.exists():
        print(f"Loading cached probe list from {probe_cache}")
        with open(probe_cache) as f:
            d = json.load(f)
        probes, covered_genes = d["probes"], d["covered_genes"]
    else:
        probes, covered_genes = build_probe_list(seqs)
        probe_cache.parent.mkdir(exist_ok=True)
        with open(probe_cache, "w") as f:
            json.dump({"probes": probes, "covered_genes": covered_genes}, f)
        print(f"Saved probe list to {probe_cache}")

    if "1" in phases:
        print(
            f"\n=== Phase 1 complete: {len(probes)} probes for {len(covered_genes)} genes ==="
        )

    if "3" in phases:
        wt_emb, mut_emb = _load_scan_embeddings()
        print(f"Loaded scan embeddings: {wt_emb.shape}")
        print("\n=== Phase 3: feature computation ===")
        gene_list, X, feature_names = compute_scan_features(
            probes, wt_emb, mut_emb, covered_genes, ablation=args.ablation
        )
        save_features(gene_list, X, feature_names)
        print(f"\nReady for probe runs. Next: python -m esm2_mech.experiments.perturbation.perturbation_probe")


if __name__ == "__main__":
    import traceback, signal

    def _sig_handler(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum} — exiting", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGHUP, _sig_handler)

    try:
        main()
    except Exception:
        print("\n[FATAL ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
