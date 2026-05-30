"""
Positive control: predict ClinVar pathogenic vs benign with the same pipeline
used for mechanism classification.

Purpose: validate that the experiment.py pipeline (ESM-2 deltas, gene-split CV,
family-split CV, linear + MLP probes) CAN extract a signal that is known to be
present in ESM-2 embeddings. If pathogenicity AUROC >> 0.85 under gene-split
AND survives family-split, the pipeline is sound and the mechanism null result
(result_4.md) is interpretable. If pathogenicity also fails, the null is
uninterpretable (pipeline is broken).

Pipeline phases:
  1. CPU: fetch ClinVar variant_summary.txt.gz, filter to missense in
     Gerasimavicius gene set, balance pathogenic vs benign per gene.
  2. GPU (runpod): extract ESM-2 embeddings for WT+mutant sequences.
  3. CPU: linear + MLP probe under gene-split AND family-split CV.

Caches: each phase writes its output and the script skips ahead if found.

Usage:
    python pathogenicity_control.py --run_dir run_0 --model esm2_t33_650M_UR50D
"""

import argparse
import gzip
import io
import json
import os
import re
import time
import urllib.request

import numpy as np
from collections import Counter, defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from esm2_mechanism.utils_probes import gene_split_cv, family_split_cv
from esm2_mechanism.embeddings.esm2_mechanism import _load_data, ESM2_MODEL_650M
from esm2_mechanism.embeddings.embed_variants import get_esm2_embeddings_for_pairs
from esm2_mechanism.utils_sequences import build_sequence_cache, fetch_pfam_families, window_sequence, apply_missense

import functools
print = functools.partial(print, flush=True)


CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"


# 3-letter to 1-letter amino acid code (for parsing HGVSp like p.His77Tyr)
AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
HGVSP_PAT = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?=[^a-zA-Z]|$)")


# --------------------------------------------------------------------------- #
# Phase 1: ClinVar variant fetch + filtering                                  #
# --------------------------------------------------------------------------- #

def fetch_clinvar_variants(target_genes, cache_dir,
                            max_per_gene_per_class=20, seed=42):
    """
    Download ClinVar variant_summary.txt.gz, filter to:
      - Type == "single nucleotide variant"
      - GeneSymbol in target_genes
      - HGVSp parseable as a single missense
      - ClinicalSignificance in {Pathogenic/Likely_pathogenic, Benign/Likely_benign}
      - Per gene per class: cap at max_per_gene_per_class (random subsample)

    Returns list of dicts: {gene, uniprot_id (filled later), aa_pos, aa_wt,
                            aa_mut, label ("pathogenic"|"benign"), clinvar_id}
    """
    cache = os.path.join(cache_dir, "clinvar_pathogenicity_variants.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                data = json.load(f)
            print(f"  Loading cached ClinVar variants from {cache}")
            return data
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt cache {cache} — re-fetching")
            os.remove(cache)

    print(f"  Downloading ClinVar variant_summary.txt.gz "
          f"(this is ~150 MB compressed) ...")
    req = urllib.request.Request(CLINVAR_URL,
                                  headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()

    print(f"  Downloaded {len(raw)/1e6:.0f} MB, decompressing ...")
    gz = gzip.GzipFile(fileobj=io.BytesIO(raw))
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    header = text.readline().rstrip("\n").split("\t")
    col = {h.lstrip("#"): i for i, h in enumerate(header)}
    needed = ["Type", "GeneSymbol", "ClinicalSignificance",
              "Name", "VariationID", "Assembly"]
    for c in needed:
        if c not in col:
            raise RuntimeError(f"Missing ClinVar column: {c}")

    target_set = set(target_genes)
    by_gene_class = defaultdict(list)
    n_seen = 0
    for line in text:
        n_seen += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["Type"]] != "single nucleotide variant":
            continue
        # Take one assembly to avoid duplicates
        if parts[col["Assembly"]] != "GRCh38":
            continue
        gene = parts[col["GeneSymbol"]].upper()
        if gene not in target_set:
            continue
        sig = parts[col["ClinicalSignificance"]].strip()
        sig_low = sig.lower()
        if any(s in sig_low for s in ["conflict", "uncertain", "not provided",
                                       "other", "no assertion"]):
            continue
        if "pathogenic" in sig_low and "non-pathogenic" not in sig_low:
            label = "pathogenic"
        elif "benign" in sig_low:
            label = "benign"
        else:
            continue
        # Parse missense from Name: "NM_000059.4(BRCA2):c.5851C>T (p.Pro1951Ser)"
        name = parts[col["Name"]]
        m = HGVSP_PAT.search(name)
        if not m:
            continue
        wt3, pos_s, mut3 = m.groups()
        if wt3 not in AA3 or mut3 not in AA3:
            continue
        if wt3 == mut3:
            continue
        clinvar_id = parts[col["VariationID"]]
        by_gene_class[(gene, label)].append({
            "gene": gene,
            "aa_pos": int(pos_s),
            "aa_wt": AA3[wt3],
            "aa_mut": AA3[mut3],
            "label": label,
            "clinvar_id": clinvar_id,
        })

    print(f"  Scanned {n_seen} ClinVar rows; matched {sum(len(v) for v in by_gene_class.values())} variants")

    rng = np.random.RandomState(seed)
    chosen = []
    for (gene, label), lst in by_gene_class.items():
        rng.shuffle(lst)
        chosen.extend(lst[:max_per_gene_per_class])

    print(f"  After per-gene-per-class cap: {len(chosen)} variants "
          f"({sum(1 for v in chosen if v['label']=='pathogenic')} pathogenic, "
          f"{sum(1 for v in chosen if v['label']=='benign')} benign, "
          f"{len(set(v['gene'] for v in chosen))} genes)")

    tmp_cache = cache + ".tmp"
    with open(tmp_cache, "w") as f:
        json.dump(chosen, f)
    os.replace(tmp_cache, cache)
    return chosen


def attach_uniprot_ids(variants, gerasimavicius_variants):
    """ClinVar gives gene symbol; we need the same UniProt IDs the existing
    seq_cache is keyed by. Look them up from the Gerasimavicius variant table."""
    gene_to_uid = {}
    for v in gerasimavicius_variants:
        if v.get("gene") and v.get("uniprot_id"):
            gene_to_uid[v["gene"].upper()] = v["uniprot_id"]
    out = []
    for v in variants:
        uid = gene_to_uid.get(v["gene"].upper())
        if uid:
            v["uniprot_id"] = uid
            out.append(v)
    print(f"  {len(out)}/{len(variants)} variants mapped to UniProt IDs "
          f"present in the existing seq_cache")
    return out


# --------------------------------------------------------------------------- #
# Phase 2: embedding extraction (GPU)                                          #
# --------------------------------------------------------------------------- #

def build_wt_mut_pairs(variants, seq_cache):
    """Apply each missense to its WT sequence, with windowing for long proteins.

    Returns valid variants, sequences, positions, and the original indices into
    variants so callers can record which rows survived filtering.
    """
    valid = []
    valid_indices = []
    wt_seqs, mut_seqs, var_positions = [], [], []
    for idx, v in enumerate(variants):
        uid = v.get("uniprot_id")
        if uid not in seq_cache:
            continue
        wt_full = seq_cache[uid]
        wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        var_positions.append(new_pos)
        valid.append(v)
        valid_indices.append(idx)
    return valid, valid_indices, wt_seqs, mut_seqs, var_positions


def get_or_extract_embeddings(variants, seq_cache, data_dir, model_name,
                               batch_size=32):
    """Cache key is a SHA-ish suffix derived from variant count + model.
    If the cache exists and matches the variant count, reuse it."""
    emb_dir = os.path.join(data_dir, "embeddings", model_name)
    os.makedirs(emb_dir, exist_ok=True)
    suffix = f"pathogenicity_n{len(variants)}"
    wt_p   = os.path.join(emb_dir, f"emb_wt_mean_{suffix}.npy")
    mut_p  = os.path.join(emb_dir, f"emb_mut_mean_{suffix}.npy")
    meta_p = os.path.join(emb_dir, f"emb_meta_{suffix}.json")

    if os.path.exists(wt_p) and os.path.exists(mut_p) and os.path.exists(meta_p):
        try:
            with open(meta_p) as _f:
                meta = json.load(_f)
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt embedding metadata {meta_p} — re-extracting")
            os.remove(meta_p)
            meta = {}
        if meta.get("n") == len(variants):
            print(f"  Cached pathogenicity embeddings found: {wt_p}")
            return (np.load(wt_p), np.load(mut_p),
                    [variants[i] for i in meta["valid_indices"]])

    print(f"  Extracting ESM-2 embeddings — requires GPU.")
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    if device != "cuda":
        raise RuntimeError(
            "GPU not detected. ESM-2 embedding extraction requires CUDA. "
            "Run this script on runpod (or any CUDA host) for phase 2. "
            "Once cached, phase 3 (the probe) runs anywhere."
        )

    valid, valid_indices, wt_seqs, mut_seqs, var_positions = build_wt_mut_pairs(
        variants, seq_cache)
    print(f"  Embedding {len(valid)} WT/mut pairs at batch_size={batch_size} ...")

    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        wt_seqs, mut_seqs, var_positions,
        model_name=model_name, device=device, batch_size=batch_size,
    )
    np.save(wt_p, wt_mean)
    np.save(mut_p, mut_mean)
    tmp_meta = meta_p + ".tmp"
    with open(tmp_meta, "w") as _f:
        json.dump({"n": len(variants), "n_valid": len(valid),
                   "valid_indices": valid_indices,
                   "model": model_name}, _f)
    os.replace(tmp_meta, meta_p)
    print(f"  Cached: {wt_p}")
    return wt_mean, mut_mean, valid


# --------------------------------------------------------------------------- #
# Phase 3: probes                                                              #
# --------------------------------------------------------------------------- #


def run_binary_probe(X, y, splits, probe_kind, seed=42):
    """probe_kind: 'logreg' or 'mlp'. y is binary 0/1."""
    aurocs, prs, f1s = [], [], []
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr, yte = y[tr], y[te]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        if probe_kind == "logreg":
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                      random_state=seed)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                 random_state=seed, early_stopping=True,
                                 validation_fraction=0.1)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        pred = clf.predict(Xte)
        aurocs.append(float(roc_auc_score(yte, proba)))
        p_, r_, _ = precision_recall_curve(yte, proba)
        prs.append(float(auc(r_, p_)))
        f1s.append(float(f1_score(yte, pred, average="binary",
                                   zero_division=0)))
    if not aurocs:
        return {"note": "no usable folds"}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std":  float(np.std(aurocs)),
        "pr_auc_mean": float(np.mean(prs)),
        "f1_mean":     float(np.mean(f1s)),
        "n_folds": len(aurocs),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="run_0")
    p.add_argument("--model", default=ESM2_MODEL_650M)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_per_gene_per_class", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    data_dir = os.path.join(args.run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ----- Phase 1: data --------------------------------------------------- #
    print("\n=== Phase 1: ClinVar variant fetch ===")
    gerasimavicius = _load_data(data_dir)
    target_genes = sorted(set(v["gene"].upper() for v in gerasimavicius
                                if v.get("gene")))
    print(f"  Target gene set: {len(target_genes)} genes from Gerasimavicius")

    variants = fetch_clinvar_variants(
        target_genes, data_dir,
        max_per_gene_per_class=args.max_per_gene_per_class,
        seed=args.seed,
    )
    variants = attach_uniprot_ids(variants, gerasimavicius)

    print(f"\n  Final ClinVar set: {len(variants)} variants  "
          f"({Counter(v['label'] for v in variants)})  "
          f"{len(set(v['gene'] for v in variants))} genes")

    # ----- Phase 2: sequences + embeddings --------------------------------- #
    print("\n=== Phase 2: sequences + embeddings ===")
    seq_cache = build_sequence_cache(gerasimavicius, data_dir)
    wt_mean, mut_mean, valid = get_or_extract_embeddings(
        variants, seq_cache, data_dir, args.model,
        batch_size=args.batch_size,
    )
    deltas = mut_mean - wt_mean
    print(f"  Final: {len(valid)} embedded variants  "
          f"({Counter(v['label'] for v in valid)})")

    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in valid])
    genes = np.array([v["gene"] for v in valid])

    # ----- Phase 3: probes ------------------------------------------------- #
    print("\n=== Phase 3: probes ===")
    pfam_map = fetch_pfam_families(
        [{"gene": g, "uniprot_id": next((v["uniprot_id"] for v in valid
                                          if v["gene"] == g), None)}
         for g in set(genes)],
        data_dir,
    )
    gs = gene_split_cv(genes, seed=args.seed)
    fs = family_split_cv(genes, pfam_map, seed=args.seed)
    print(f"  Gene-split folds: {len(gs)}  Family-split folds: {len(fs)}")

    features = {
        "delta_mean": deltas,
        "wt_only":    wt_mean,
    }
    results = {
        "n_variants": int(len(valid)),
        "n_genes": int(len(set(genes))),
        "n_pathogenic": int(int(y.sum())),
        "n_benign": int(int((1-y).sum())),
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
                r = run_binary_probe(X, y, splits, probe_kind=probe,
                                      seed=args.seed)
                key = f"{split_name}_{probe}"
                block[key] = r
                if "auroc_mean" in r:
                    print(f"    {key:25s}  AUROC={r['auroc_mean']:.3f}  "
                          f"PR-AUC={r['pr_auc_mean']:.3f}  "
                          f"F1={r['f1_mean']:.3f}  (folds={r['n_folds']})")
        results["by_feature"][fname] = block

    out_path = os.path.join(args.run_dir, "pathogenicity_control.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # ----- Headline -------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("HEADLINE — pathogenicity positive control")
    print("=" * 60)
    def aur(k1, k2): return results["by_feature"][k1][k2].get("auroc_mean", float("nan"))
    print(f"delta_mean  gene-split   logreg AUROC: {aur('delta_mean','gene_split_logreg'):.3f}")
    print(f"delta_mean  gene-split   MLP    AUROC: {aur('delta_mean','gene_split_mlp'):.3f}")
    print(f"delta_mean  family-split logreg AUROC: {aur('delta_mean','family_split_logreg'):.3f}")
    print(f"delta_mean  family-split MLP    AUROC: {aur('delta_mean','family_split_mlp'):.3f}")
    print()
    print(f"wt_only     gene-split   logreg AUROC: {aur('wt_only','gene_split_logreg'):.3f}")
    print(f"wt_only     family-split logreg AUROC: {aur('wt_only','family_split_logreg'):.3f}")
    print()
    delta_gs = aur('delta_mean','gene_split_logreg')
    if delta_gs >= 0.85:
        print("  ⇒ Pipeline PASSES positive control "
              "(delta AUROC ≥ 0.85 on gene-split).")
        print("    The mechanism null result (result_4) is therefore "
              "interpretable as a real absence of mechanism signal,")
        print("    not a pipeline failure.")
    elif delta_gs >= 0.70:
        print("  ⇒ Pipeline shows MODERATE pathogenicity signal "
              "(delta AUROC 0.70-0.85).")
        print("    The mechanism null is interpretable but the pipeline "
              "is weaker than published ESM-2 pathogenicity work.")
    else:
        print("  ⇒ Pipeline FAILS positive control (delta AUROC < 0.70).")
        print("    The mechanism null result is UNINTERPRETABLE — the pipeline")
        print("    cannot recover even known signal. Debug before drawing")
        print("    conclusions from result_4.")


if __name__ == "__main__":
    main()
