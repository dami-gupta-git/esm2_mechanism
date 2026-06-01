"""
Pathogenicity positive control — fetch, embed, and probe in one run.

This is the run-everything entry point for the result_6 control: the same ESM-2
delta embeddings that classify GOF/DN/LOF at chance should predict ClinVar
pathogenic-vs-benign well (published ESM-2 work: AUROC 0.88–0.94). If they do,
the mechanism null is a real absence of signal, not a broken pipeline; and the
gene-split / family-split gap being ~0 shows the pathogenicity signal is
per-variant biochemistry, not homology leakage.

Three phases run in sequence (designed to run on RunPod, which has both CPU and
GPU). Phase 1 caches its variant set (re-fetching if the fetch params change),
Phase 2 skips extraction when the cached embeddings match by content, and
Phase 3 always recomputes the probes (fast on CPU):

  Phase 1 (CPU, network) — fetch_phase()
      Download ClinVar variant_summary.txt.gz, keep balanced pathogenic/benign
      missense variants in the Gerasimavicius gene set (≤ max_per_gene_per_class
      each), restricted to the GRCh38 assembly (the only place a genome build is
      selected — see CLINVAR_ASSEMBLY), attach UniProt IDs.
      → data/clinvar_pathogenicity_variants.json

  Phase 2 (GPU) — embed_phase()
      Extract mean-pooled ESM-2 WT and mutant embeddings for those variants.
      → pathogenicity_{wt,mut}_mean.npy, pathogenicity_meta.json

  Phase 3 (CPU) — probe_phase()
      5-seed logreg + MLP probes on delta_mean and wt_only, under gene-split and
      family-split CV.  → results/<run>/pathogenicity_control.json

  Inputs : data/variants.json, data/cache/sequences.json, data/pfam_families.json
  Output : results/<run>/pathogenicity_control.json (paths.PATHOGENICITY_CONTROL_JSON)

Usage (on RunPod, inside tmux):
    python -m esm2_mech.experiments.pathogenicity.pathogenicity_control
        --model esm2_t33_650M_UR50D --batch_size 32
"""

from __future__ import annotations

import argparse
import functools
import gzip
import io
import json
import re
import urllib.request
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from esm2_mech.utils.data import load_variants, variants_fingerprint
from esm2_mech.utils.embed import get_esm2_embeddings_for_pairs
from esm2_mech.utils.io import atomic_write_json, save_npy
from esm2_mech.utils.paths import (
    CLINVAR_PATHOGENICITY_PARAMS_JSON,
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    EMB_DIR,
    PATH_EMB_META,
    PATH_EMB_MUT_MEAN,
    PATH_EMB_WT_MEAN,
    PATHOGENICITY_CONTROL_JSON,
    PATHOGENICITY_CONTROL_SEED_JSON,
    PFAM_JSON,
    RESULTS_DIR,
    SEQUENCES_JSON,
    VARIANTS_JSON,
)
from esm2_mech.utils.probes import run_logreg_binary_cv, run_mlp_binary_cv
from esm2_mech.utils.sequences import apply_missense, window_sequence
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)

# The pathogenicity embedding path constants (PATH_EMB_*) live under the 650M
# EMB_DIR, so this control runs on the 650M model only.
ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
N_SEEDS = 5

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# variant_summary.txt.gz lists each variant once per genome assembly (GRCh37 and
# GRCh38). Restrict to GRCh38 — the current standard build — so the same variant
# is not counted twice. This is the only place in the pipeline a genome build is
# selected; all other variant data is handled in protein coordinates (UniProt
# position + amino acid), which are assembly-independent.
CLINVAR_ASSEMBLY = "GRCh38"

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
HGVSP_PAT = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?=[^a-zA-Z]|$)")


# ===========================================================================
# Phase 1 — fetch ClinVar pathogenic/benign variants
# ===========================================================================
def _fetch_clinvar(target_genes, max_per_gene_per_class, seed):
    """Download and filter ClinVar to balanced P/B missense variants."""
    print("  Downloading ClinVar variant_summary.txt.gz (~150 MB compressed) ...")
    req = urllib.request.Request(CLINVAR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()

    print(f"  Downloaded {len(raw) / 1e6:.0f} MB, decompressing ...")
    gz = gzip.GzipFile(fileobj=io.BytesIO(raw))
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    header = text.readline().rstrip("\n").split("\t")
    col = {(h[1:] if h.startswith("#") else h): i for i, h in enumerate(header)}
    needed = ["Type", "GeneSymbol", "ClinicalSignificance", "Name", "VariationID", "Assembly"]
    for c in needed:
        if c not in col:
            raise RuntimeError(f"Missing ClinVar column: {c}")
    max_needed_idx = max(col[c] for c in needed)

    target_set = set(target_genes)
    by_gene_class = defaultdict(list)
    n_seen = 0
    try:
        for line in text:
            n_seen += 1
            parts = line.rstrip("\n").split("\t")
            # Only the read columns need to be present; guard their max index,
            # not the full header width (trailing empty fields may be omitted).
            if len(parts) <= max_needed_idx:
                continue
            if parts[col["Type"]] != "single nucleotide variant":
                continue
            if parts[col["Assembly"]] != CLINVAR_ASSEMBLY:
                continue
            gene = parts[col["GeneSymbol"]].upper()
            if gene not in target_set:
                continue
            sig_low = parts[col["ClinicalSignificance"]].strip().lower()
            if any(s in sig_low for s in ["conflict", "uncertain", "not provided", "other", "no assertion"]):
                continue
            if "pathogenic" in sig_low and "non-pathogenic" not in sig_low:
                label = "pathogenic"
            elif "benign" in sig_low:
                label = "benign"
            else:
                continue
            m = HGVSP_PAT.search(parts[col["Name"]])
            if not m:
                continue
            wt3, pos_s, mut3 = m.groups()
            if wt3 not in AA3 or mut3 not in AA3 or wt3 == mut3:
                continue
            by_gene_class[(gene, label)].append({
                "gene": gene,
                "aa_pos": int(pos_s),
                "aa_wt": AA3[wt3],
                "aa_mut": AA3[mut3],
                "label": label,
                "clinvar_id": parts[col["VariationID"]],
            })
    except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
        # A truncated download decompresses partway then fails. Do NOT return a
        # partial result that the caller would cache as success.
        raise RuntimeError(
            f"ClinVar download appears truncated after {n_seen} rows ({exc}). "
            "Re-run to retry; nothing was cached."
        ) from exc

    print(f"  Scanned {n_seen} ClinVar rows; matched {sum(len(v) for v in by_gene_class.values())} variants")

    rng = np.random.RandomState(seed)
    chosen = []
    for lst in by_gene_class.values():
        rng.shuffle(lst)
        chosen.extend(lst[:max_per_gene_per_class])
    return chosen


def _attach_uniprot_ids(variants, mechanism_variants):
    gene_to_uid = {
        v["gene"].upper(): v["uniprot_id"]
        for v in mechanism_variants
        if v.get("gene") and v.get("uniprot_id")
    }
    out = []
    for v in variants:
        uid = gene_to_uid.get(v["gene"].upper())
        if uid:
            v["uniprot_id"] = uid
            out.append(v)
    print(f"  {len(out)}/{len(variants)} variants mapped to a UniProt ID")
    return out


def fetch_phase(max_per_gene_per_class=20, seed=42):
    """Phase 1. Returns the variant list; caches to CLINVAR_PATHOGENICITY_VARIANTS_JSON."""
    print("=== Phase 1: fetch ClinVar pathogenic/benign variants ===")
    params = {"max_per_gene_per_class": max_per_gene_per_class, "seed": seed}
    if CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists():
        cached_params = None
        if CLINVAR_PATHOGENICITY_PARAMS_JSON.exists():
            try:
                with open(CLINVAR_PATHOGENICITY_PARAMS_JSON) as f:
                    cached_params = json.load(f)
            except json.JSONDecodeError:
                cached_params = None
        if cached_params != params:
            print(
                f"  Cache was built with {cached_params}, current params {params} — re-fetching"
            )
        else:
            try:
                with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
                    cached = json.load(f)
                print(f"  Loaded cached set: {len(cached)} variants ({Counter(v['label'] for v in cached)})")
                return cached
            except json.JSONDecodeError:
                print("  WARNING: corrupt cache — re-fetching")

    # Merged mechanism variants (Gerasimavicius + G2P) — defines the target gene set
    # and the gene→UniProt map for the ClinVar pathogenicity fetch.
    mechanism_variants = load_variants(VARIANTS_JSON)
    target_genes = sorted({v["gene"].upper() for v in mechanism_variants if v.get("gene")})
    print(f"  Target gene set: {len(target_genes)} genes")

    variants = _fetch_clinvar(target_genes, max_per_gene_per_class, seed)
    variants = _attach_uniprot_ids(variants, mechanism_variants)
    print(
        f"  Final set: {len(variants)} variants ({Counter(v['label'] for v in variants)}), "
        f"{len({v['gene'] for v in variants})} genes"
    )

    CLINVAR_PATHOGENICITY_VARIANTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CLINVAR_PATHOGENICITY_VARIANTS_JSON, variants)
    atomic_write_json(CLINVAR_PATHOGENICITY_PARAMS_JSON, params)
    return variants


# ===========================================================================
# Phase 2 — extract ESM-2 embeddings
# ===========================================================================
def _build_valid_pairs_indexed(variants, seq_cache):
    """Filter to embeddable variants; return original indices for re-alignment."""
    valid_indices, valid, wt_seqs, mut_seqs, positions = [], [], [], [], []
    skipped_no_uid = 0
    for idx, v in enumerate(variants):
        uid = v.get("uniprot_id")
        if not uid:
            skipped_no_uid += 1
            continue
        if uid not in seq_cache:
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_indices.append(idx)
        valid.append(v)
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)
    if skipped_no_uid:
        print(f"  WARNING: skipped {skipped_no_uid} variants with no uniprot_id")
    print(f"  Valid variant pairs: {len(valid)}")
    return valid_indices, valid, wt_seqs, mut_seqs, positions


def _embeddings_complete(valid_fingerprint, n_valid):
    """True only if cached embeddings match the current valid set by content.

    Compares a content fingerprint, not just counts, so a seed/cap change that
    happens to yield the same n_valid does not reuse stale embeddings.
    """
    if not all(p.exists() for p in [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]):
        return False
    try:
        with open(PATH_EMB_META) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        print(f"  WARNING: corrupt {PATH_EMB_META} — re-extracting")
        PATH_EMB_META.unlink()
        return False
    n_on_disk = np.load(PATH_EMB_WT_MEAN, mmap_mode="r").shape[0]
    return (
        n_on_disk == n_valid == meta.get("n_valid")
        and meta.get("fingerprint") == valid_fingerprint
    )


def embed_phase(variants, model, batch_size):
    """Phase 2. Extract and cache pathogenicity embeddings (GPU)."""
    print("\n=== Phase 2: extract ESM-2 embeddings ===")
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_indices, valid, wt_seqs, mut_seqs, positions = _build_valid_pairs_indexed(
        variants, seq_cache
    )
    valid_fingerprint = variants_fingerprint(valid)

    if _embeddings_complete(valid_fingerprint, len(valid)):
        print("  Embeddings already complete — skipping extraction.")
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Model: {model}")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        wt_seqs, mut_seqs, positions,
        valid_variants=valid, out_dir=None,
        model_name=model, device=device, batch_size=batch_size,
    )

    save_npy(str(PATH_EMB_WT_MEAN), wt_mean)
    save_npy(str(PATH_EMB_MUT_MEAN), mut_mean)
    atomic_write_json(
        PATH_EMB_META,
        {
            "valid_indices": valid_indices,
            "n": len(variants),
            "n_valid": len(valid),
            "fingerprint": valid_fingerprint,
            "model": model,
        },
    )
    print(f"  Saved {wt_mean.shape} -> {EMB_DIR}")


# ===========================================================================
# Phase 3 — 5-seed probes
# ===========================================================================
def probe_phase(variants, n_seeds, n_jobs=-1):
    """Phase 3. 5-seed logreg + MLP probes on delta_mean and wt_only."""
    print("\n=== Phase 3: probes ===")
    with open(PATH_EMB_META) as f:
        meta = json.load(f)
    wt_mean = np.load(PATH_EMB_WT_MEAN)
    mut_mean = np.load(PATH_EMB_MUT_MEAN)

    # valid_indices index into the variant list that produced the embeddings.
    valid = [variants[i] for i in meta["valid_indices"]]
    if not (len(valid) == wt_mean.shape[0] == mut_mean.shape[0]):
        raise ValueError(
            f"Row mismatch: {len(valid)} variants vs {wt_mean.shape[0]} embedding rows."
        )

    # Verify by content, not count: the embeddings must have been built from
    # exactly these variants in this order. A seed/cap change that yields a
    # colliding count would otherwise misalign labels/genes to embedding rows.
    if variants_fingerprint(valid) != meta.get("fingerprint"):
        raise ValueError(
            "Embedding fingerprint does not match the current variant set — the "
            f"embedding cache is stale. Delete {PATH_EMB_META.name} and the "
            "pathogenicity_*.npy files to re-extract."
        )

    delta = mut_mean - wt_mean
    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in valid])
    genes = np.array([v["gene"] for v in valid])
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    print(f"  {len(valid)} variants  pathogenic={int(y.sum())} benign={int((1 - y).sum())}  "
          f"{len(set(genes))} genes")

    features = {"delta_mean": delta, "wt_only": wt_mean}
    probes = {"logreg": run_logreg_binary_cv, "mlp": run_mlp_binary_cv}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run one seed at a time, writing that seed's per-cell AUROCs to its own file
    # as it completes (resume + progress visibility). Within a seed the 8 cells
    # (2 features x 2 probes x 2 splits) are independent, so they run in parallel.
    for seed in range(n_seeds):
        seed_path = Path(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed))
        if seed_path.exists():
            print(f"  seed {seed}: cached, skipping")
            continue

        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        cells = [
            (fname, pname, split_name, splits)
            for fname in features
            for pname in probes
            for split_name, splits in (("gene", gs), ("family", fs))
        ]

        def _run_cell(fname, pname, split_name, splits, seed=seed):
            res = probes[pname](features[fname], y, splits, seed=seed)
            return (fname, pname, split_name, res.get("auroc_mean", float("nan")))

        outcomes = Parallel(n_jobs=n_jobs)(
            delayed(_run_cell)(*c) for c in cells
        )

        seed_result = {}
        for fname, pname, split_name, auroc in outcomes:
            seed_result[f"{fname}_{pname}_{split_name}"] = auroc
        with open(seed_path, "w") as f:
            json.dump(seed_result, f, indent=2)
        summary = "  ".join(
            f"{k}={v:.3f}" for k, v in seed_result.items() if "mlp" in k
        )
        print(f"  seed {seed} done -> {seed_path.name}   {summary}")

    # Aggregate per-seed files into the final mean ± std result.
    per_cell = defaultdict(list)
    for seed in range(n_seeds):
        with open(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed)) as f:
            seed_result = json.load(f)
        for key, auroc in seed_result.items():
            per_cell[key].append(auroc)

    results = {
        "n_variants": int(len(valid)),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((1 - y).sum()),
        "n_genes": int(len(set(genes))),
        "n_seeds": n_seeds,
        "by_feature": {},
    }
    for fname in features:
        results["by_feature"][fname] = {}
        for pname in probes:
            for split_name in ("gene", "family"):
                key = f"{fname}_{pname}_{split_name}"
                vals = [v for v in per_cell[key] if not np.isnan(v)]
                mean = float(np.mean(vals)) if vals else float("nan")
                std = float(np.std(vals)) if vals else float("nan")
                results["by_feature"][fname][f"{pname}_{split_name}"] = {
                    "auroc_mean": mean,
                    "auroc_std": std,
                    "per_seed": per_cell[key],
                }

    with open(PATHOGENICITY_CONTROL_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Aggregated results written to {PATHOGENICITY_CONTROL_JSON}")
    return results


def _print_headline(results):
    print("\n" + "=" * 60)
    print("HEADLINE — pathogenicity positive control")
    print("=" * 60)

    def cell(feature, key):
        c = results["by_feature"][feature][key]
        return c["auroc_mean"], c["auroc_std"]

    for feature in ("delta_mean", "wt_only"):
        for key in ("logreg_gene", "logreg_family", "mlp_gene", "mlp_family"):
            mean, std = cell(feature, key)
            print(f"  {feature:11s} {key:14s} AUROC = {mean:.3f} ± {std:.3f}")
        print()

    d_mean, _ = cell("delta_mean", "mlp_gene")
    d_fam, _ = cell("delta_mean", "mlp_family")
    print(f"  delta_mean MLP gene→family Δ = {d_mean - d_fam:+.3f}")
    if d_mean >= 0.85:
        print("  ⇒ Pipeline PASSES positive control (delta MLP AUROC ≥ 0.85).")
        print("    The mechanism null is a real absence of signal, not a pipeline failure.")
    elif d_mean >= 0.70:
        print("  ⇒ MODERATE pathogenicity signal (0.70–0.85). Interpretable but weak.")
    else:
        print("  ⇒ Pipeline FAILS positive control (< 0.70). Mechanism null UNINTERPRETABLE.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of probe seeds (>=1)")
    parser.add_argument("--max_per_gene_per_class", type=int, default=20)
    parser.add_argument("--fetch_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1, help="parallel jobs for probes (-1 = all cores)")
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    variants = fetch_phase(
        max_per_gene_per_class=args.max_per_gene_per_class, seed=args.fetch_seed
    )
    embed_phase(variants, model=args.model, batch_size=args.batch_size)
    results = probe_phase(variants, n_seeds=args.seeds, n_jobs=args.n_jobs)
    _print_headline(results)


if __name__ == "__main__":
    main()
