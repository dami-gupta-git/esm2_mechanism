"""
GO term prediction smoke test.

Downloads goa_human.gaf, restricts to merged dataset genes,
picks the top N most common non-IEA GO terms, and runs a linear probe
under gene-split and family-split CV on frozen ESM-2 WT embeddings.

Goal: confirm leakage exists (gene-split AUROC >> family-split AUROC)
before committing to a full experiment.

Usage (local, no GPU needed):
  cd esm2_mechanism
  python scripts/go_smoke_test.py

Outputs:
  results/go_smoke/go_annotations.json
  results/go_smoke/smoke_results.json
"""

import gzip, io, json, os, sys, urllib.request
import numpy as np
from collections import defaultdict
import functools
print = functools.partial(print, flush=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB  = os.path.join(DATA, "embeddings")
OUT  = os.path.join(ROOT, "results", "go_smoke")
os.makedirs(OUT, exist_ok=True)

GOA_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"

# Evidence codes to exclude (electronic/predicted — keep only manual)
IEA_CODES = {"IEA", "ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR", "IRD", "RCA"}

MIN_POSITIVES = 20   # minimum annotated genes per GO term
MAX_POS_FRAC  = 0.5  # skip trivially common terms (>50% of genes positive)
TOP_N_TERMS   = 50   # number of GO terms to test in smoke run


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gene_set():
    """Load merged dataset gene list and UniProt IDs."""
    with open(os.path.join(DATA, "merged_valid_variants.json")) as f:
        variants = json.load(f)
    gene_to_uniprot = {}
    for v in variants:
        g = v.get("gene", "").upper()
        u = v.get("uniprot_id", "")
        if g and u:
            gene_to_uniprot[g] = u
    uniprot_to_gene = {u: g for g, u in gene_to_uniprot.items()}
    genes = sorted(set(gene_to_uniprot.keys()))
    print(f"Merged dataset: {len(genes)} unique genes, {len(gene_to_uniprot)} with UniProt IDs")
    return genes, gene_to_uniprot, uniprot_to_gene


def fetch_go_annotations(uniprot_to_gene, cache_path):
    """Download goa_human.gaf.gz and extract non-IEA annotations for our genes."""
    if os.path.exists(cache_path):
        print(f"Loading cached GO annotations from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"Downloading {GOA_URL} ...")
    req = urllib.request.Request(GOA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()
    print(f"Downloaded {len(raw)/1e6:.0f} MB, parsing ...")

    gz = gzip.GzipFile(fileobj=io.BytesIO(raw))
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")

    # gene -> {go_id -> namespace}
    gene_go = defaultdict(dict)
    n_lines = 0
    for line in text:
        if line.startswith("!"):
            continue
        n_lines += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue
        db_obj_id  = parts[1]   # UniProt accession
        go_id      = parts[4]   # GO:XXXXXXX
        evidence   = parts[6]   # evidence code
        namespace  = parts[8]   # P/F/C (BP/MF/CC)
        qualifier  = parts[3]   # e.g. "NOT"

        if "NOT" in qualifier:
            continue
        if evidence in IEA_CODES:
            continue
        if db_obj_id not in uniprot_to_gene:
            continue

        gene = uniprot_to_gene[db_obj_id]
        ns_map = {"P": "BP", "F": "MF", "C": "CC"}
        ns = ns_map.get(namespace, namespace)
        gene_go[gene][go_id] = ns

    print(f"Parsed {n_lines} lines. {len(gene_go)} genes with non-IEA annotations.")

    # Invert: go_id -> list of genes
    go_to_genes = defaultdict(list)
    go_namespace = {}
    for gene, gos in gene_go.items():
        for go_id, ns in gos.items():
            go_to_genes[go_id].append(gene)
            go_namespace[go_id] = ns

    result = {
        "gene_go": {g: list(gos.keys()) for g, gos in gene_go.items()},
        "go_to_genes": dict(go_to_genes),
        "go_namespace": go_namespace,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    print(f"Saved to {cache_path}")
    return result


def select_terms(go_data, all_genes, min_pos=MIN_POSITIVES,
                 max_frac=MAX_POS_FRAC, top_n=TOP_N_TERMS):
    """Pick testable GO terms: enough positives, not too common."""
    n_genes = len(all_genes)
    gene_set = set(all_genes)
    candidates = []
    for go_id, genes in go_data["go_to_genes"].items():
        pos_genes = [g for g in genes if g in gene_set]
        n_pos = len(pos_genes)
        if n_pos < min_pos:
            continue
        if n_pos / n_genes > max_frac:
            continue
        candidates.append((go_id, n_pos, go_data["go_namespace"].get(go_id, "?")))

    # Sort by n_pos descending, take top_n
    candidates.sort(key=lambda x: -x[1])
    selected = candidates[:top_n]
    print(f"Selected {len(selected)} GO terms (from {len(candidates)} candidates) "
          f"with {min_pos}–{int(max_frac*n_genes)} positives")
    for go_id, n_pos, ns in selected[:10]:
        print(f"  {go_id} [{ns}]: {n_pos} genes")
    if len(selected) > 10:
        print(f"  ... and {len(selected)-10} more")
    return selected


# ── Embeddings ────────────────────────────────────────────────────────────────

def load_gene_embeddings(genes, gene_to_uniprot):
    """Load per-gene mean WT embeddings from the merged dataset embeddings.
    The merged embeddings are per-variant; we average to get per-gene."""
    with open(os.path.join(DATA, "merged_valid_variants.json")) as f:
        variants = json.load(f)

    emb_wt = np.load(os.path.join(EMB, "merged_embeddings_wt_mean.npy"))
    assert len(emb_wt) == len(variants), "Embedding/variant count mismatch"

    # Average per gene
    gene_embs = defaultdict(list)
    for i, v in enumerate(variants):
        g = v.get("gene", "").upper()
        if g:
            gene_embs[g].append(emb_wt[i])

    gene_list = []
    X = []
    for g in genes:
        if g in gene_embs:
            gene_list.append(g)
            X.append(np.mean(gene_embs[g], axis=0))

    X = np.array(X, dtype=np.float32)
    print(f"Loaded embeddings for {len(gene_list)} / {len(genes)} genes  shape={X.shape}")
    return gene_list, X


# ── CV ────────────────────────────────────────────────────────────────────────

def gene_split_cv(genes, n_folds=5, seed=42):
    u = np.array(sorted(set(genes)))
    np.random.RandomState(seed).shuffle(u)
    splits = []
    for fold in np.array_split(u, n_folds):
        te = np.where(np.isin(genes, fold))[0]
        tr = np.where(~np.isin(genes, fold))[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


def family_split_cv(genes, pfam_map, n_folds=5, seed=42):
    g2p = {g: pfam_map[g] for g in np.unique(genes) if pfam_map.get(g)}
    fams = np.array(sorted(set(g2p.values())))
    np.random.RandomState(seed).shuffle(fams)
    n = len(genes)
    splits = []
    for fold_fams in np.array_split(fams, n_folds):
        fs = set(fold_fams)
        te = np.array([genes[i] in g2p and g2p[genes[i]] in fs  for i in range(n)])
        tr = np.array([genes[i] in g2p and g2p[genes[i]] not in fs for i in range(n)])
        if tr.sum() >= 10 and te.sum() >= 5:
            splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


def run_logreg_binary(X, y, splits, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for tr, te in splits:
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0,
                                  class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    if not aurocs:
        return None
    return float(np.mean(aurocs)), float(np.std(aurocs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Gene set
    all_genes, gene_to_uniprot, uniprot_to_gene = load_gene_set()

    # 2. GO annotations
    cache = os.path.join(OUT, "go_annotations.json")
    go_data = fetch_go_annotations(uniprot_to_gene, cache)

    # 3. Select testable terms
    selected_terms = select_terms(go_data, all_genes)
    if not selected_terms:
        print("No testable GO terms found. Check annotation coverage.")
        return

    # 4. Embeddings (gene-level mean pool)
    gene_list, X = load_gene_embeddings(all_genes, gene_to_uniprot)
    genes_arr = np.array(gene_list)

    # 5. Pfam map
    with open(os.path.join(DATA, "pfam_families.json")) as f:
        pfam_map = json.load(f)

    # 6. CV splits (same across all GO terms for efficiency)
    gs_splits = gene_split_cv(genes_arr, seed=42)
    fs_splits = family_split_cv(genes_arr, pfam_map, seed=42)
    print(f"\nGene-split folds: {len(gs_splits)}  Family-split folds: {len(fs_splits)}")

    # 7. Probe each GO term
    gene_set = set(gene_list)
    go_to_genes = go_data["go_to_genes"]
    go_ns = go_data["go_namespace"]

    results = []
    print(f"\nRunning probes on {len(selected_terms)} GO terms...")
    for i, (go_id, n_pos_all, ns) in enumerate(selected_terms):
        pos_genes = set(g for g in go_to_genes.get(go_id, []) if g in gene_set)
        y = np.array([1 if g in pos_genes else 0 for g in gene_list])
        n_pos = int(y.sum())
        if n_pos < MIN_POSITIVES or n_pos / len(gene_list) > MAX_POS_FRAC:
            continue

        gs = run_logreg_binary(X, y, gs_splits)
        fs = run_logreg_binary(X, y, fs_splits)
        if gs is None or fs is None:
            continue

        gs_auroc, gs_std = gs
        fs_auroc, fs_std = fs
        leakage = (gs_auroc - fs_auroc) / max(gs_auroc - 0.5, 1e-6)

        results.append({
            "go_id": go_id,
            "namespace": ns,
            "n_pos": n_pos,
            "gene_split_auroc": round(gs_auroc, 4),
            "family_split_auroc": round(fs_auroc, 4),
            "leakage_fraction": round(leakage, 4),
        })

        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{i+1}/{len(selected_terms)}] {go_id} [{ns}] "
                  f"n={n_pos}  gene={gs_auroc:.3f}  family={fs_auroc:.3f}  "
                  f"leakage={leakage:.1%}")

    # 8. Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Terms tested: {len(results)}")

    gs_aurocs = [r["gene_split_auroc"] for r in results]
    fs_aurocs = [r["family_split_auroc"] for r in results]
    leakages  = [r["leakage_fraction"] for r in results]

    print(f"Gene-split AUROC:   {np.mean(gs_aurocs):.3f} ± {np.std(gs_aurocs):.3f}")
    print(f"Family-split AUROC: {np.mean(fs_aurocs):.3f} ± {np.std(fs_aurocs):.3f}")
    print(f"Mean leakage:       {np.mean(leakages):.1%} ± {np.std(leakages):.1%}")

    # By namespace
    for ns in ["MF", "BP", "CC"]:
        ns_res = [r for r in results if r["namespace"] == ns]
        if ns_res:
            print(f"\n  {ns} ({len(ns_res)} terms):")
            print(f"    gene-split:   {np.mean([r['gene_split_auroc'] for r in ns_res]):.3f}")
            print(f"    family-split: {np.mean([r['family_split_auroc'] for r in ns_res]):.3f}")
            print(f"    leakage:      {np.mean([r['leakage_fraction'] for r in ns_res]):.1%}")

    # Top surviving terms (lowest leakage)
    results_sorted = sorted(results, key=lambda r: r["leakage_fraction"])
    print(f"\nTop 5 terms that SURVIVE family-split (lowest leakage):")
    for r in results_sorted[:5]:
        print(f"  {r['go_id']} [{r['namespace']}] n={r['n_pos']}  "
              f"gene={r['gene_split_auroc']:.3f}  family={r['family_split_auroc']:.3f}  "
              f"leakage={r['leakage_fraction']:.1%}")

    print(f"\nTop 5 terms that COLLAPSE (highest leakage):")
    for r in results_sorted[-5:]:
        print(f"  {r['go_id']} [{r['namespace']}] n={r['n_pos']}  "
              f"gene={r['gene_split_auroc']:.3f}  family={r['family_split_auroc']:.3f}  "
              f"leakage={r['leakage_fraction']:.1%}")

    out_path = os.path.join(OUT, "smoke_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {
            "n_terms": len(results),
            "gene_split_mean": float(np.mean(gs_aurocs)),
            "family_split_mean": float(np.mean(fs_aurocs)),
            "leakage_mean": float(np.mean(leakages)),
            "leakage_std": float(np.std(leakages)),
        }, "per_term": results}, f, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
