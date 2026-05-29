"""
Stage 0 pilot — Experiment 11
================================

Tests whether a minimal set of public gene-level features carries any
mechanism (GOF / DN / LOF) signal under family-split CV.  This is the
sanity-check / pipeline-bug gate that precedes the full Phase 1 data pull.

Features pulled here:
  1. pLI            (gnomAD v4 constraint)
  2. LOEUF          (gnomAD v4 constraint)
  3. mis_z          (gnomAD v4 constraint)
  4. paralog_count  (Ensembl Compara REST, paralogues type)

Models compared (all under 5-fold family-split CV, seed=0, gene-level):
  - majority baseline
  - logistic regression
  - tiny MLP (16 → 8 → 3)

Labels:  gene-level mechanism collapsed to 3 classes
         GOF → GOF, DN → DN, HI → LOF, LOF → LOF, AR → dropped

Decision rules (see docs/plan_experiment.md):
  V2-pilot family-split macro-F1 ≥ 0.40   → strong; proceed to full Phase 1
  V2-pilot ≈ majority (~0.31)             → weak; proceed but expectations down
  V2-pilot well below majority            → pipeline bug; debug

Runtime ≈ 20 minutes including downloads and paralog REST calls.  CPU only.

Outputs:
  data/cache/proteome_pilot/                   raw downloads + per-gene paralog cache
  data/gene_features_pilot.tsv                 human-readable per-gene feature table
  results/proteome_pilot/pilot_results.json    metrics, decision flag, settings

Usage:
  python scripts/proteome_pilot.py                  # full run
  python scripts/proteome_pilot.py --skip-paralogs  # use NaN paralogs (fast, no REST)
  python scripts/proteome_pilot.py --gerasimavicius # restrict to 948 Geras genes
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import functools
print = functools.partial(print, flush=True)
from utils_probes import family_split_indices

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("proteome_pilot")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache" / "proteome_pilot"
PARALOG_CACHE = CACHE_DIR / "paralogs"
RESULTS_DIR = PROJECT_DIR / "results" / "proteome_pilot"

for d in (CACHE_DIR, PARALOG_CACHE, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
GERASIMAVICIUS_GENE_LIST = DATA_DIR / "gerasimavicius_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"

GENE_FEATURES_TSV = DATA_DIR / "gene_features_pilot.tsv"
# Per-seed output filename; resolved at runtime once args.seed is known
def results_json_path(seed: int) -> Path:
    return RESULTS_DIR / f"pilot_results_seed{seed}.json"

# ---------------------------------------------------------------------------
# Data sources — pinned URLs
# ---------------------------------------------------------------------------
# gnomAD v4.1 constraint metrics (per gene, plain TSV ~30MB)
GNOMAD_CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "constraint/gnomad.v4.1.constraint_metrics.tsv"
)
GNOMAD_CACHE = CACHE_DIR / "gnomad_v4.1_constraint.tsv"

ENSEMBL_PARALOG_URL = (
    "https://rest.ensembl.org/homology/symbol/human/{gene}"
    "?type=paralogues;format=condensed"
)

# ---------------------------------------------------------------------------
# Label collapse: 5-way mechanism -> 3-class
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "GOF": "GOF",
    "DN": "DN",
    "HI": "LOF",
    "LOF": "LOF",
    # AR is recessive; not part of the dominant 3-way mechanism problem
    "AR": None,
    "": None,
}
CLASSES = ["GOF", "DN", "LOF"]


# ---------------------------------------------------------------------------
# Step 1 — Load gene list with mechanism labels
# ---------------------------------------------------------------------------
def load_gene_labels(tsv_path: Path) -> dict[str, str]:
    """Return {gene_symbol: 3class_label}.  Drops AR and unlabeled genes."""
    labels: dict[str, str] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["gene"].strip()
            mech = row["mechanism"].strip()
            collapsed = LABEL_MAP.get(mech)
            if collapsed is None:
                continue
            labels[gene] = collapsed
    log.info(f"Loaded {len(labels)} labeled genes from {tsv_path.name} "
             f"(GOF={sum(1 for v in labels.values() if v=='GOF')}, "
             f"DN={sum(1 for v in labels.values() if v=='DN')}, "
             f"LOF={sum(1 for v in labels.values() if v=='LOF')})")
    return labels


# ---------------------------------------------------------------------------
# Step 2 — Load Pfam families (for family-split CV)
# ---------------------------------------------------------------------------
def load_pfam_families() -> dict[str, str]:
    with open(PFAM_FAMILIES) as f:
        d = json.load(f)
    log.info(f"Loaded Pfam family assignments for {len(d)} genes")
    return d


# ---------------------------------------------------------------------------
# Step 3 — gnomAD constraint download + parse
# ---------------------------------------------------------------------------
def download_gnomad_constraint() -> Path:
    """Download gnomAD v4.1 constraint TSV to cache."""
    if GNOMAD_CACHE.exists() and GNOMAD_CACHE.stat().st_size > 1_000_000:
        log.info(f"gnomAD constraint cached: {GNOMAD_CACHE} "
                 f"({GNOMAD_CACHE.stat().st_size/1e6:.1f} MB)")
        return GNOMAD_CACHE
    log.info(f"Downloading gnomAD constraint from {GNOMAD_CONSTRAINT_URL}")
    t0 = time.time()
    urllib.request.urlretrieve(GNOMAD_CONSTRAINT_URL, GNOMAD_CACHE)
    log.info(f"  downloaded {GNOMAD_CACHE.stat().st_size/1e6:.1f} MB "
             f"in {time.time()-t0:.1f}s")
    return GNOMAD_CACHE


def parse_gnomad_constraint(path: Path) -> dict[str, dict[str, float]]:
    """
    Parse gnomAD constraint TSV.  Returns {gene_symbol: {pLI, LOEUF, mis_z}}.

    The v4 constraint TSV has one row per (gene, transcript, mane).
    We take the canonical/MANE_SELECT row per gene where available, otherwise
    the row with the highest lof.exp (most observation power).
    """
    log.info(f"Parsing gnomAD constraint from {path}")
    # Inspect header
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
    log.info(f"  gnomAD columns: {len(header)} (first 8: {header[:8]})")

    # Locate the columns we need.  The v4 schema names them as:
    #   gene, transcript, mane_select (bool),
    #   lof.pLI, lof.oe_ci.upper (= LOEUF), mis.z_score
    # Fall back to alternates if naming differs.
    def find_col(*candidates: str) -> Optional[int]:
        for c in candidates:
            if c in header:
                return header.index(c)
        return None

    idx_gene = find_col("gene")
    idx_mane = find_col("mane_select", "canonical")
    idx_lof_exp = find_col("lof.exp")
    idx_pli = find_col("lof.pLI", "pLI")
    idx_loeuf = find_col("lof.oe_ci.upper", "oe_lof_upper", "LOEUF")
    idx_misz = find_col("mis.z_score", "mis_z", "mis.z")

    missing = []
    if idx_gene is None: missing.append("gene")
    if idx_pli is None: missing.append("pLI")
    if idx_loeuf is None: missing.append("LOEUF")
    if idx_misz is None: missing.append("mis_z")
    if missing:
        log.error(f"gnomAD TSV missing required columns: {missing}.")
        log.error(f"  Full header: {header}")
        sys.exit(2)

    log.info(f"  using columns: gene={idx_gene}, pLI={idx_pli}, "
             f"LOEUF={idx_loeuf}, mis_z={idx_misz}, "
             f"mane={idx_mane}, lof.exp={idx_lof_exp}")

    by_gene: dict[str, dict] = {}
    n_rows = 0
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_gene, idx_pli, idx_loeuf, idx_misz):
                continue
            gene = parts[idx_gene].strip()
            if not gene or gene == "NA":
                continue

            def fnum(i: Optional[int]) -> Optional[float]:
                if i is None:
                    return None
                v = parts[i].strip()
                if v in ("", "NA", "nan", "NaN"):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            row = {
                "pLI": fnum(idx_pli),
                "LOEUF": fnum(idx_loeuf),
                "mis_z": fnum(idx_misz),
                "_mane": (parts[idx_mane].strip().lower() in ("true", "1", "yes"))
                         if idx_mane is not None else False,
                "_lof_exp": fnum(idx_lof_exp) or 0.0,
            }
            # Prefer MANE-select transcript; else keep the row with highest lof.exp
            prev = by_gene.get(gene)
            if prev is None:
                by_gene[gene] = row
            elif row["_mane"] and not prev["_mane"]:
                by_gene[gene] = row
            elif row["_mane"] == prev["_mane"] and row["_lof_exp"] > prev["_lof_exp"]:
                by_gene[gene] = row
            n_rows += 1
    log.info(f"  parsed {n_rows} rows -> {len(by_gene)} unique genes")
    return by_gene


# ---------------------------------------------------------------------------
# Step 4 — Ensembl paralog count
# ---------------------------------------------------------------------------
def fetch_paralog_count(gene: str, session=None) -> Optional[int]:
    """One Ensembl REST call per gene.  Resume-safe via per-gene cache file."""
    cache_file = PARALOG_CACHE / f"{gene}.json"
    if cache_file.exists():
        try:
            d = json.loads(cache_file.read_text())
            return d.get("paralog_count")
        except Exception:
            pass

    url = ENSEMBL_PARALOG_URL.format(gene=gene)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        data = json.loads(body)
        homologies = []
        for entry in data.get("data", []):
            homologies.extend(entry.get("homologies", []))
        # We asked the REST endpoint for ?type=paralogues, so everything in
        # `homologies` should already be a paralog.  Filter strictly on the
        # `type` field rather than `species` — the condensed format's `species`
        # field refers to the target species and can mislead.
        paralog_count = sum(
            1 for h in homologies
            if "paralog" in h.get("type", "").lower()
        )
        cache_file.write_text(json.dumps({"paralog_count": paralog_count}))
        return paralog_count
    except Exception as e:
        log.debug(f"paralog fetch failed for {gene}: {e}")
        cache_file.write_text(json.dumps({"paralog_count": None, "error": str(e)}))
        return None


def fetch_paralogs_for_genes(genes: list[str]) -> dict[str, Optional[int]]:
    """Rate-limited Ensembl REST loop with progress logging."""
    log.info(f"Fetching paralog counts for {len(genes)} genes "
             f"(rate-limited to 10 req/s; cached)")
    out: dict[str, Optional[int]] = {}
    n_cached = 0
    n_fetched = 0
    t0 = time.time()
    for i, g in enumerate(genes):
        cache_file = PARALOG_CACHE / f"{g}.json"
        was_cached = cache_file.exists()
        c = fetch_paralog_count(g)
        out[g] = c
        if was_cached:
            n_cached += 1
        else:
            n_fetched += 1
            time.sleep(0.1)  # 10 req/s
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            log.info(f"  {i+1}/{len(genes)} "
                     f"(cached={n_cached}, fetched={n_fetched}, "
                     f"elapsed={elapsed:.1f}s)")
    log.info(f"  done: {n_cached} cached + {n_fetched} fetched "
             f"in {time.time()-t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Step 5 — Build per-gene feature matrix
# ---------------------------------------------------------------------------
FEATURE_COLS = ["pLI", "LOEUF", "mis_z", "paralog_count"]


def build_feature_table(
    labels: dict[str, str],
    families: dict[str, str],
    gnomad: dict[str, dict],
    paralogs: dict[str, Optional[int]],
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        rows  : list of dicts, one per gene with label + family + raw features
        X     : (n, 2*n_features) array — features + binary missingness indicators
        y     : (n,) array of class indices
        groups: (n,) array of family ids for family-split CV
    """
    rows = []
    keep_genes = [g for g in labels if families.get(g) is not None]
    log.info(f"Genes with both label and Pfam family: {len(keep_genes)}/"
             f"{len(labels)} (dropped {len(labels)-len(keep_genes)} "
             f"without family assignment)")

    for gene in sorted(keep_genes):
        row = {
            "gene": gene,
            "label": labels[gene],
            "family": families[gene],
        }
        g = gnomad.get(gene, {})
        row["pLI"] = g.get("pLI")
        row["LOEUF"] = g.get("LOEUF")
        row["mis_z"] = g.get("mis_z")
        row["paralog_count"] = paralogs.get(gene)
        rows.append(row)

    # Coverage report
    for col in FEATURE_COLS:
        n_present = sum(1 for r in rows if r[col] is not None)
        log.info(f"  coverage[{col}]: {n_present}/{len(rows)} "
                 f"({100*n_present/len(rows):.1f}%)")

    # Build X with missingness indicators and median imputation
    X_raw = np.array(
        [[r[c] if r[c] is not None else np.nan for c in FEATURE_COLS]
         for r in rows],
        dtype=float,
    )
    miss = np.isnan(X_raw).astype(float)
    # Impute per-column median.  This is a model-input handling choice, not a
    # fabricated scientific value — the missingness indicator preserves the
    # information that the original measurement was unavailable.
    col_medians = np.nanmedian(X_raw, axis=0)
    for j in range(X_raw.shape[1]):
        if np.isnan(col_medians[j]):
            col_medians[j] = 0.0  # column entirely missing
        X_raw[np.isnan(X_raw[:, j]), j] = col_medians[j]
    X = np.hstack([X_raw, miss])

    y = np.array([CLASSES.index(r["label"]) for r in rows], dtype=int)
    groups = np.array([r["family"] for r in rows])
    log.info(f"Feature matrix: X={X.shape}, y={y.shape}, "
             f"groups={len(set(groups))} unique families")
    log.info(f"Class distribution: "
             + ", ".join(f"{c}={int((y==i).sum())}" for i, c in enumerate(CLASSES)))
    return rows, X, y, groups


# ---------------------------------------------------------------------------
# Step 6 — Family-split CV models
# ---------------------------------------------------------------------------

def evaluate_model(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_folds: int = 5, seed: int = 0) -> dict:
    """5-fold family-split CV.  Returns dict of metrics."""
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, test_idx in family_split_indices(groups, n_folds, seed):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # Per-fold scaler (no leakage across folds)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        m = clone(model)
        m.fit(Xtr_s, ytr)
        raw_proba = m.predict_proba(Xte_s)
        # Align to canonical CLASSES order in case a class is absent from training fold
        proba = np.zeros((len(Xte_s), len(CLASSES)), dtype=np.float32)
        for ci, c in enumerate(m.classes_):
            proba[:, c] = raw_proba[:, ci]
        pred = m.predict(Xte_s)
        y_true_all.append(yte)
        y_pred_all.append(pred)
        y_proba_all.append(proba)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all, axis=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    per_class_auroc = {}
    for i, cls in enumerate(CLASSES):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auroc[cls] = None
        else:
            per_class_auroc[cls] = float(roc_auc_score(y_bin, y_proba[:, i]))

    return {
        "macro_f1": float(macro_f1),
        "per_class_auroc": per_class_auroc,
        "n_test": int(len(y_true)),
        "class_distribution_test":
            {c: int((y_true == i).sum()) for i, c in enumerate(CLASSES)},
    }


def majority_baseline(y: np.ndarray, groups: np.ndarray,
                      n_folds: int = 5, seed: int = 0) -> dict:
    from sklearn.metrics import f1_score
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in family_split_indices(groups, n_folds, seed):
        ytr, yte = y[train_idx], y[test_idx]
        # Majority class in training fold
        maj = int(np.bincount(ytr, minlength=len(CLASSES)).argmax())
        y_true_all.append(yte)
        y_pred_all.append(np.full_like(yte, maj))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


# ---------------------------------------------------------------------------
# Step 7 — Save artifacts
# ---------------------------------------------------------------------------
def save_feature_table(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gene", "label", "family"] + FEATURE_COLS,
                                delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    log.info(f"Wrote feature table: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gerasimavicius", action="store_true",
                        help="Restrict to Gerasimavicius 948-gene subset.")
    parser.add_argument("--skip-paralogs", action="store_true",
                        help="Skip Ensembl REST calls; use NaN for paralog_count.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    # 1. Labels
    gene_list_path = (GERASIMAVICIUS_GENE_LIST if args.gerasimavicius
                      else MERGED_GENE_LIST)
    labels = load_gene_labels(gene_list_path)

    # 2. Families
    families = load_pfam_families()

    # 3. gnomAD constraint
    gnomad_path = download_gnomad_constraint()
    gnomad = parse_gnomad_constraint(gnomad_path)
    log.info(f"  gnomAD coverage of labeled genes: "
             f"{sum(1 for g in labels if g in gnomad)}/{len(labels)}")

    # 4. Paralogs
    if args.skip_paralogs:
        log.info("Skipping Ensembl paralog fetch (--skip-paralogs)")
        paralogs = {g: None for g in labels}
    else:
        # Only fetch for genes we'll actually use (have label + family)
        target = sorted(g for g in labels if g in families)
        paralogs = fetch_paralogs_for_genes(target)

    # 5. Feature table
    rows, X, y, groups = build_feature_table(labels, families, gnomad, paralogs)
    save_feature_table(rows, GENE_FEATURES_TSV)

    # 6. Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    log.info("=" * 60)
    log.info("Running 5-fold family-split CV")
    log.info("=" * 60)

    results = {
        "settings": {
            "gene_list": gene_list_path.name,
            "n_genes": int(len(y)),
            "n_families": int(len(set(groups))),
            "features": FEATURE_COLS,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "label_collapse": "GOF / DN / LOF (HI→LOF, AR dropped)",
        },
        "class_distribution": {
            c: int((y == i).sum()) for i, c in enumerate(CLASSES)
        },
        "models": {},
    }

    # Majority baseline
    maj = majority_baseline(y, groups, args.n_folds, args.seed)
    log.info(f"Majority baseline:  macro_f1 = {maj['macro_f1']:.4f}")
    results["models"]["majority"] = maj

    # Logistic regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=args.seed)
    lr_res = evaluate_model(lr, X, y, groups, args.n_folds, args.seed)
    log.info(f"Logistic regression: macro_f1 = {lr_res['macro_f1']:.4f}  "
             f"AUROC: " + ", ".join(
                 f"{c}={lr_res['per_class_auroc'][c]:.3f}"
                 if lr_res['per_class_auroc'][c] is not None else f"{c}=NA"
                 for c in CLASSES))
    results["models"]["logreg"] = lr_res

    # Tiny MLP
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                        early_stopping=True, validation_fraction=0.15,
                        random_state=args.seed)
    mlp_res = evaluate_model(mlp, X, y, groups, args.n_folds, args.seed)
    log.info(f"Tiny MLP (16,8):     macro_f1 = {mlp_res['macro_f1']:.4f}  "
             f"AUROC: " + ", ".join(
                 f"{c}={mlp_res['per_class_auroc'][c]:.3f}"
                 if mlp_res['per_class_auroc'][c] is not None else f"{c}=NA"
                 for c in CLASSES))
    results["models"]["mlp"] = mlp_res

    # 7. Decision flag
    # Branch order: BUG check fires first so that ANY underperformance vs the
    # trivial majority baseline gets flagged.  A working model has no reason
    # to be worse than majority; that signals a real problem (label bug,
    # leakage in the wrong direction, scaler misalignment, etc.).
    best_f1 = max(lr_res["macro_f1"], mlp_res["macro_f1"])
    maj_f1 = maj["macro_f1"]
    if best_f1 < maj_f1:
        decision = "PIPELINE_BUG: best model is below majority baseline — debug"
    elif best_f1 >= 0.40:
        decision = "STRONG_SIGNAL: proceed to full Phase 1"
    elif best_f1 >= maj_f1 + 0.02:
        decision = "WEAK_SIGNAL: proceed but expectations lowered"
    else:
        decision = "NULL: at majority baseline — proceed to full Phase 1 cautiously"
    results["decision"] = decision
    log.info("=" * 60)
    log.info(f"DECISION: {decision}")
    log.info(f"  best model macro_f1 = {best_f1:.4f}")
    log.info(f"  majority baseline   = {maj['macro_f1']:.4f}")

    out_path = results_json_path(args.seed)
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"Wrote results JSON: {out_path}")


if __name__ == "__main__":
    main()
