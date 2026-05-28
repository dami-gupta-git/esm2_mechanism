"""
ESM-2 delta-embedding mechanism geometry experiment.

Tests whether ESM-2 delta-embeddings (mutant - wildtype) organize by molecular
disease mechanism class (GOF / DN / LOF) after removing protein stability signal.

Data: Gerasimavicius et al. 2022 (NatComms 13:3895), OSF: 10.17605/OSF.IO/H62FQ
  - Pathogenic ClinVar missense variants, gene-level mechanism labels
  - GOF, DN, HI, AR classes; FoldX ΔΔG provided
  - Primary: 3-class GOF / DN / LOF (HI+AR collapsed)

Pipeline:
  1. Fetch Gerasimavicius dataset from OSF
  2. Fetch protein sequences from UniProt for each gene
  3. Extract ESM-2 650M embeddings for WT and mutant sequences
  4. Compute delta-embeddings (mutant - WT), mean-pooled and per-residue
  5. Fit stability nuisance subspace on Megascale data; validate transfer
  6. Linear probe (LR) with gene-split CV
  7. Baselines, negative controls, probe direction orthogonality analysis
"""

import argparse
import json
import os
import time
import warnings
import urllib.request
import urllib.error
import functools
from io import StringIO

import numpy as np
print = functools.partial(print, flush=True)
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSF_DATASET_URL = "https://osf.io/rct6d/download"  # Gerasimavicius et al. DiseaseMech_Stability_VEPS.xlsx
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"
MAX_SEQ_LEN = 1022  # ESM-2 token limit
WINDOW_HALF = 500   # for sequences > 1022 aa, window ±500 around variant position

# Pre-registered thresholds
STABILITY_TRANSFER_RHO_THRESHOLD = 0.3  # Spearman ρ for Megascale→Gerasimavicius transfer
SCALE_INVARIANT_THRESHOLD = 0.03        # macro-F1 difference 650M vs 3B
SCALE_EMERGENT_THRESHOLD = 0.05
VARIANCE_ASYMMETRY_THRESHOLD = 0.30    # GOF ≥ 30% less variance explained than HI+AR
BENIGN_LEAK_THRESHOLD = 0.50           # benign AUROC as fraction of pathogenic AUROC


# ---------------------------------------------------------------------------
# Data loading: Gerasimavicius et al. OSF dataset
# ---------------------------------------------------------------------------

def fetch_gerasimavicius_dataset(cache_dir):
    """
    Download and parse the Gerasimavicius et al. variant table from OSF.
    File: DiseaseMech_Stability_VEPS.xlsx, sheet: HGMD_four_class
    Columns used: Gene, Uniprot_id, Uniprot_variant, Gene_mechanism_label, raw_FoldX_Monomer
    Uniprot_variant format: e.g. "H77Y" (wt_aa + position + mut_aa)
    Returns list of dicts with keys: gene, uniprot_id, aa_pos, aa_wt, aa_mut,
    mechanism (GOF/DN/HI/AR), foldx_ddg.
    Falls back to a minimal synthetic dataset for testing if download fails.
    """
    cache_path = os.path.join(cache_dir, "gerasimavicius_variants.json")
    if os.path.exists(cache_path):
        print("Loading cached Gerasimavicius dataset...")
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    print("Downloading Gerasimavicius et al. dataset from OSF...")

    variants = []
    try:
        import io
        import re
        try:
            import openpyxl
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openpyxl"])
            import openpyxl

        req = urllib.request.Request(OSF_DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        ws = wb["ClinVar_gene_level"]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(h).strip() if h is not None else "" for h in rows[0]]
        col = {h: i for i, h in enumerate(header)}
        print(f"  Columns: {header[:10]}")

        variant_pat = re.compile(r"^([A-Z])(\d+)([A-Z])$")

        # Mechanism label normalisation for ClinVar_gene_level Disease_mechanism column
        mech_map = {
            "GOF": "GOF", "DN": "DN", "HI": "HI",
            "AR": "AR", "AR, HET": "AR", "AR, HOM": "AR",
        }

        for row in rows[1:]:
            try:
                # Only include ClinVar disease variants (not GNOMAD)
                row_class = str(row[col.get("Class", -1)] or "").strip().upper()
                if "CLINVAR" not in row_class:
                    continue

                gene = row[col["Gene"]]
                uniprot = row[col["Uniprot_id"]]
                variant_str = row[col["Uniprot_variant"]]
                mech_raw = row[col["Disease_mechanism"]]
                foldx_raw = row[col["raw_FoldX_Monomer"]] if "raw_FoldX_Monomer" in col else None

                if not all([gene, uniprot, variant_str, mech_raw]):
                    continue

                mech = mech_map.get(str(mech_raw).strip().upper())
                if mech is None:
                    continue

                m = variant_pat.match(str(variant_str).strip())
                if not m:
                    continue
                aa_wt, aa_pos_str, aa_mut = m.groups()
                aa_pos = int(aa_pos_str)

                foldx_ddg = None
                if foldx_raw is not None:
                    try:
                        foldx_ddg = float(foldx_raw)
                    except (ValueError, TypeError):
                        pass

                variants.append({
                    "gene": str(gene).upper(),
                    "uniprot_id": str(uniprot).strip(),
                    "aa_pos": aa_pos,
                    "aa_wt": aa_wt.upper(),
                    "aa_mut": aa_mut.upper(),
                    "mechanism": mech,
                    "foldx_ddg": foldx_ddg,
                    "clinvar_id": "",
                })
            except Exception:
                continue

        print(f"  Parsed {len(variants)} variants from OSF Excel (ClinVar_gene_level sheet)")

    except Exception as e:
        print(f"  OSF download failed: {e}")
        print("  Falling back to synthetic test dataset (small, for pipeline validation only)")
        variants = _make_synthetic_dataset()

    if not variants:
        print("  No variants parsed — using synthetic dataset")
        variants = _make_synthetic_dataset()

    with open(cache_path, "w") as f:
        json.dump(variants, f)

    _print_dataset_stats(variants)
    return variants


def _make_synthetic_dataset():
    """
    Minimal synthetic dataset for pipeline testing when OSF is unavailable.
    Uses real human proteins but fake variant positions/mechanisms.
    NOT for scientific use — only validates the pipeline runs end-to-end.
    """
    print("  WARNING: Using synthetic dataset. Results are not scientifically valid.")
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.RandomState(42)
    genes_mechanisms = [
        ("TP53", "P04637", "GOF"), ("KRAS", "P01116", "GOF"),
        ("EGFR", "P00533", "GOF"), ("BRAF", "P15056", "GOF"),
        ("PIK3CA", "P42336", "GOF"), ("MYC", "P01106", "GOF"),
        ("CTNNB1", "P35222", "GOF"), ("IDH1", "O75874", "GOF"),
        ("TP53", "P04637", "DN"), ("SMAD2", "Q15796", "DN"),
        ("SMAD3", "P84022", "DN"), ("SMAD4", "Q13485", "DN"),
        ("RUNX1", "Q01196", "DN"), ("PAX5", "Q02548", "DN"),
        ("WT1", "P19544", "DN"), ("SOX9", "P48436", "DN"),
        ("BRCA1", "P38398", "HI"), ("BRCA2", "P51587", "HI"),
        ("RB1", "P06400", "HI"), ("PTEN", "P60484", "HI"),
        ("VHL", "P40337", "AR"), ("CFTR", "P13569", "AR"),
        ("HEXA", "P06865", "AR"), ("MUTYH", "Q9UIF7", "AR"),
    ]
    variants = []
    for gene, uniprot, mech in genes_mechanisms:
        for _ in range(15):
            pos = int(rng.randint(1, 300))
            wt = rng.choice(aa_list)
            mut = rng.choice([a for a in aa_list if a != wt])
            variants.append({
                "gene": gene,
                "uniprot_id": uniprot,
                "aa_pos": pos,
                "aa_wt": wt,
                "aa_mut": mut,
                "mechanism": mech,
                "foldx_ddg": float(rng.randn()),
                "clinvar_id": f"SYNTH_{gene}_{pos}{wt}{mut}",
            })
    return variants


def _print_dataset_stats(variants):
    from collections import Counter
    mechs = Counter(v["mechanism"] for v in variants)
    genes = len(set(v["gene"] for v in variants))
    print(f"  Variants: {len(variants)} | Genes: {genes}")
    for m, n in sorted(mechs.items()):
        print(f"    {m}: {n}")


# ---------------------------------------------------------------------------
# Protein sequence fetching
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(uniprot_id, retries=3, delay=1.0):
    """Fetch canonical protein sequence from UniProt."""
    url = f"{UNIPROT_REST}/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                fasta = resp.read().decode()
            lines = fasta.strip().split("\n")
            seq = "".join(l for l in lines if not l.startswith(">"))
            return seq.upper() if seq else None
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def apply_missense(sequence, aa_pos, aa_wt, aa_mut):
    """
    Apply a missense mutation to a protein sequence.
    aa_pos is 1-indexed. Returns None if position is out of range or WT mismatch.
    """
    idx = aa_pos - 1
    if idx < 0 or idx >= len(sequence):
        return None
    if sequence[idx] != aa_wt:
        return None
    seq_list = list(sequence)
    seq_list[idx] = aa_mut
    return "".join(seq_list)


def window_sequence(sequence, aa_pos, window_half=WINDOW_HALF, max_len=MAX_SEQ_LEN):
    """
    For sequences longer than max_len, extract a window centered on aa_pos.
    Returns (windowed_seq, new_aa_pos) where new_aa_pos is the variant position
    in the windowed sequence (1-indexed).
    """
    if len(sequence) <= max_len:
        return sequence, aa_pos

    idx = aa_pos - 1  # 0-indexed
    start = max(0, idx - window_half)
    end = min(len(sequence), idx + window_half)
    # Ensure window is at most max_len
    if end - start > max_len:
        half = max_len // 2
        start = max(0, idx - half)
        end = min(len(sequence), start + max_len)

    windowed = sequence[start:end]
    new_pos = idx - start + 1  # 1-indexed in windowed sequence
    return windowed, new_pos


def build_sequence_cache(variants, cache_dir):
    """
    Fetch and cache protein sequences for all unique genes.
    Returns dict: uniprot_id -> canonical sequence.
    """
    cache_path = os.path.join(cache_dir, "sequences.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    sequences = {}
    unique_uniprots = {v["uniprot_id"] for v in variants if v["uniprot_id"]}
    print(f"Fetching sequences for {len(unique_uniprots)} UniProt IDs...")

    for i, uid in enumerate(sorted(unique_uniprots)):
        if i % 50 == 0:
            print(f"  {i}/{len(unique_uniprots)}")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            sequences[uid] = seq
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(sequences, f)

    print(f"  Fetched {len(sequences)}/{len(unique_uniprots)} sequences")
    return sequences


def fetch_alphamissense_scores(variants, cache_dir, retries=3, delay=1.0):
    """
    Fetch per-variant AlphaMissense pathogenicity scores via MyVariant.info
    (dbnsfp.alphamissense.score field). Returns numpy array of scores, NaN where unavailable.
    """
    cache_path = os.path.join(cache_dir, "alphamissense_scores.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
    else:
        cached = {}

    scores = np.full(len(variants), np.nan)
    to_fetch = []
    for i, v in enumerate(variants):
        key = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        if key in cached:
            val = cached[key]
            if val is not None:
                scores[i] = float(val)
        else:
            to_fetch.append((i, v, key))

    if to_fetch:
        print(f"  Fetching {len(to_fetch)} AlphaMissense scores from MyVariant.info...")
        base_url = "https://myvariant.info/v1/hg38/query"
        fetched = 0
        deadline = time.time() + 300  # 5 min max — AM is a non-critical baseline
        for i, v, key in to_fetch:
            if time.time() > deadline:
                print(f"  AlphaMissense fetch timeout — saving {fetched} fetched, continuing")
                break
            gene = v["gene"]
            aa_wt = v["aa_wt"]
            aa_mut = v["aa_mut"]
            aa_pos = v["aa_pos"]
            query = f"{gene} p.{aa_wt}{aa_pos}{aa_mut}"
            score = None
            for attempt in range(retries):
                try:
                    url = (f"{base_url}?q={urllib.request.quote(query)}"
                           f"&fields=dbnsfp.alphamissense&size=1")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode())
                    hits = data.get("hits", [])
                    if hits:
                        am = hits[0].get("dbnsfp", {}).get("alphamissense", {})
                        raw_score = am.get("score")
                        if isinstance(raw_score, list):
                            raw_score = raw_score[0] if raw_score else None
                        if raw_score is not None:
                            score = float(raw_score)
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(delay)
            cached[key] = score
            if score is not None:
                scores[i] = score
            fetched += 1
            if fetched % 500 == 0:
                print(f"    {fetched}/{len(to_fetch)}")
            time.sleep(0.2)

        with open(cache_path, "w") as f:
            json.dump(cached, f)

    n_valid = int(np.sum(~np.isnan(scores)))
    print(f"  AlphaMissense scores: {n_valid}/{len(variants)} available")
    return scores


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
    # Interleave WT and mutant: [wt0, mut0, wt1, mut1, ...]
    # batch_size refers to number of pairs; actual forward pass has 2*batch_size sequences
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

            var_idx_token = var_pos  # var_pos is 1-indexed; token index = var_pos (BOS at 0)
            if 0 < var_idx_token <= reps.shape[1] - 1:
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

    Returns the projection matrix (D x n_components) defining the subspace,
    or None if Megascale data is unavailable.
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

    # Regress each delta dimension on ΔΔG, collect regression coefficients
    # Then PCA on the coefficient matrix to get the stability subspace
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(ddg.reshape(-1, 1), deltas)
    coefs = np.array(reg.coef_).flatten()

    # The stability direction is the unit vector along the regression coefficient
    stability_dir = coefs / (np.linalg.norm(coefs) + 1e-10)

    # PCA on residuals after regressing out ΔΔG to find additional stability axes
    deltas_res = deltas - deltas.dot(stability_dir)[:, None] * stability_dir
    pca = PCA(n_components=min(n_components - 1, deltas_res.shape[0] - 1, deltas_res.shape[1]))
    pca.fit(deltas_res)

    # Subspace = stability direction + top PCA components of residuals
    subspace = np.vstack([stability_dir.reshape(1, -1), pca.components_])  # (n_components, D)
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
        # Leave-one-gene-out CV: fit on all-but-one-gene to get unbiased stability direction
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

    deltas_res = deltas_valid - deltas_valid.dot(stability_dir) [:, None] * stability_dir
    n_comp = min(n_components - 1, deltas_res.shape[0] - 1, deltas_res.shape[1] - 1)
    if n_comp < 1:
        return stability_dir.reshape(1, -1)

    pca = PCA(n_components=n_comp)
    pca.fit(deltas_res)
    subspace = np.vstack([stability_dir.reshape(1, -1), pca.components_])
    return subspace[:n_components]


def validate_stability_transfer(subspace, deltas_geras, foldx_ddg_geras):
    """
    Validate that the Megascale-fit subspace correlates with FoldX ΔΔG
    on Gerasimavicius variants. Returns Spearman ρ.
    """
    valid = ~np.isnan(foldx_ddg_geras)
    if valid.sum() < 20:
        return 0.0

    # Project deltas onto first (primary stability) direction of subspace
    proj = deltas_geras[valid].dot(subspace[0])
    rho, _ = spearmanr(proj, foldx_ddg_geras[valid])
    return float(rho)


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
        total_var = float(np.var(d))
        proj = d.dot(Q).dot(Q.T)
        proj_var = float(np.var(proj))
        results[cls] = proj_var / (total_var + 1e-10)

    if "GOF" in results and "LOF" in results:
        asymmetry = (results["LOF"] - results["GOF"]) / (results["LOF"] + 1e-10)
        results["gof_lof_asymmetry"] = float(asymmetry)
        results["asymmetry_prediction_holds"] = bool(asymmetry >= VARIANCE_ASYMMETRY_THRESHOLD)

    return results


# ---------------------------------------------------------------------------
# Gene-split cross-validation
# ---------------------------------------------------------------------------

def gene_split_cv(X, y, genes, n_folds=5, seed=42):
    """
    Gene-split CV: split unique genes into folds, ensuring no gene
    appears in both train and test. Returns list of (train_idx, test_idx).
    """
    unique_genes = np.array(sorted(set(genes)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_genes)

    gene_folds = np.array_split(unique_genes, n_folds)
    splits = []
    for fold_genes in gene_folds:
        fold_gene_set = set(fold_genes)
        test_mask = np.array([g in fold_gene_set for g in genes])
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    return splits


def fetch_pfam_families(variants, seq_cache, cache_dir):
    """
    Fetch primary Pfam family for each unique gene via UniProt.
    Returns dict: gene -> pfam_id (or None if not found).
    Genes lacking annotation are excluded from family-split CV.
    """
    cache_path = os.path.join(cache_dir, "pfam_families.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    pfam_map = {}
    unique_pairs = {(v["gene"], v["uniprot_id"]) for v in variants if v["uniprot_id"]}
    print(f"Fetching Pfam families for {len(unique_pairs)} genes...")

    for gene, uniprot_id in sorted(unique_pairs):
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        pfam_id = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "Pfam":
                    pfam_id = xref.get("id")
                    break
        except Exception:
            pass
        pfam_map[gene] = pfam_id
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(pfam_map, f)

    n_annotated = sum(1 for v in pfam_map.values() if v is not None)
    print(f"  Pfam annotations: {n_annotated}/{len(pfam_map)} genes")
    return pfam_map


def gene_family_split_cv(X, y, genes, pfam_map, n_folds=5, seed=42):
    """
    Gene-family-split CV: hold out entire Pfam families.
    Genes without Pfam annotation are dropped (not assigned to singleton groups).
    Returns list of (train_idx, test_idx), or empty list if < 10 Pfam families.
    """
    gene_to_pfam = {}
    for gene in np.unique(genes):
        pfam = pfam_map.get(gene)
        if pfam is not None:
            gene_to_pfam[gene] = pfam

    if not gene_to_pfam:
        print("  No Pfam annotations — family-split CV infeasible")
        return []

    # Only keep variants whose gene has a Pfam annotation
    annotated_mask = np.array([genes[i] in gene_to_pfam for i in range(len(genes))])
    if annotated_mask.sum() < 20:
        print("  Too few annotated variants — family-split CV infeasible")
        return []

    unique_families = sorted(set(gene_to_pfam.values()))
    if len(unique_families) < 10:
        print(f"  Only {len(unique_families)} Pfam families — family-split CV infeasible (need ≥ 10)")
        return []

    rng = np.random.RandomState(seed)
    family_arr = np.array(unique_families)
    rng.shuffle(family_arr)
    family_folds = np.array_split(family_arr, n_folds)

    splits = []
    for fold_families in family_folds:
        fold_family_set = set(fold_families)
        test_mask = np.array([
            genes[i] in gene_to_pfam and gene_to_pfam[genes[i]] in fold_family_set
            for i in range(len(genes))
        ])
        train_mask = np.array([
            genes[i] in gene_to_pfam and gene_to_pfam[genes[i]] not in fold_family_set
            for i in range(len(genes))
        ])
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    return splits


def run_linear_probe(X, y, genes, n_folds=5, seed=42):
    """
    Run linear probe (LR) with gene-split CV.
    Returns per-fold metrics and bootstrap CIs.
    """
    splits = gene_split_cv(X, y, genes, n_folds=n_folds, seed=seed)
    classes = sorted(set(y))

    fold_results = []
    all_test_idx = []
    all_pred_proba = []
    all_true = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(set(y_train)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  random_state=seed)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        fold_metrics = {}
        for i, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                fold_metrics[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
            else:
                fold_metrics[f"auroc_{cls}"] = float("nan")

        macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        fold_metrics["macro_f1"] = macro_f1

        for i, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
                fold_metrics[f"pr_auc_{cls}"] = float(auc(rec, prec))
            else:
                fold_metrics[f"pr_auc_{cls}"] = float("nan")

        fold_results.append(fold_metrics)
        all_true.extend(y_test.tolist())
        all_pred_proba.append(proba)

    if not fold_results:
        return {"error": "insufficient data for CV"}

    # Aggregate
    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    return agg


# ---------------------------------------------------------------------------
# Probe direction orthogonality
# ---------------------------------------------------------------------------

def probe_direction_orthogonality(X, y, genes, stability_subspace,
                                   n_folds=5, seed=42, n_shuffle=50):
    """
    Fit pairwise LR probes (GOF-vs-DN, GOF-vs-LOF, DN-vs-LOF) and compute
    cosine similarity between probe weight vectors. Compare to shuffled-label null.
    Path A (Megascale subspace): reports 4x4 matrix including stability direction.
    Path B (direct subspace or None): reports 3x3 pairwise inter-probe matrix only.
    """
    splits = gene_split_cv(X, y, genes, n_folds=n_folds, seed=seed)
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

    # Cosine similarity matrix between pairwise probe directions
    pair_keys = list(probe_weights.keys())
    cosine_matrix = {}
    for i, k1 in enumerate(pair_keys):
        for j, k2 in enumerate(pair_keys):
            if i >= j:
                continue
            cos = float(np.dot(probe_weights[k1], probe_weights[k2]))
            cosine_matrix[f"{k1}|{k2}"] = cos

    # Stability direction cosines (Path A only — Megascale subspace is not None)
    stability_cosines = {}
    if stability_subspace is not None:
        stab_dir = stability_subspace[0] / (np.linalg.norm(stability_subspace[0]) + 1e-10)
        for pair_key, w in probe_weights.items():
            cos = float(np.dot(w, stab_dir))
            stability_cosines[f"{pair_key}_vs_stability"] = cos

    # Shuffled-label null
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
# Baselines
# ---------------------------------------------------------------------------

def run_baselines(embeddings_wt, deltas_mean, foldx_ddg, labels, genes,
                  aa_wt_list, aa_mut_list, alphamissense_scores, seed=42):
    """
    Run four baselines under gene-split CV:
    1. WT-only ESM-2 embeddings (no delta)
    2. One-hot amino acid identity (40-dim: 20 WT + 20 mut)
    3. FoldX ΔΔG only (1-dim)
    4. AlphaMissense score (1-dim), if available
    """
    results = {}
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    aa_index = {a: i for i, a in enumerate(AA_ORDER)}

    # Baseline 1: WT-only embeddings
    print("  Baseline: WT-only embeddings")
    results["wt_only"] = run_linear_probe(embeddings_wt, labels, genes, seed=seed)

    # Baseline 2: one-hot amino acid identity
    n = len(aa_wt_list)
    onehot = np.zeros((n, 40), dtype=np.float32)
    for i, (wt, mut) in enumerate(zip(aa_wt_list, aa_mut_list)):
        wt_idx = aa_index.get(wt.upper())
        mut_idx = aa_index.get(mut.upper())
        if wt_idx is not None:
            onehot[i, wt_idx] = 1.0
        if mut_idx is not None:
            onehot[i, 20 + mut_idx] = 1.0
    print("  Baseline: one-hot amino acid identity")
    results["onehot_aa"] = run_linear_probe(onehot, labels, genes, seed=seed)

    # Baseline 3: FoldX ΔΔG only
    ddg_feat = np.nan_to_num(foldx_ddg, nan=0.0).reshape(-1, 1)
    if ddg_feat.std() > 0:
        print("  Baseline: FoldX ΔΔG only")
        results["foldx_ddg_only"] = run_linear_probe(ddg_feat, labels, genes, seed=seed)
    else:
        results["foldx_ddg_only"] = {"note": "no FoldX ΔΔG available"}

    # Baseline 4: AlphaMissense score
    if alphamissense_scores is not None and not np.all(np.isnan(alphamissense_scores)):
        am_feat = np.nan_to_num(alphamissense_scores, nan=0.0).reshape(-1, 1)
        if am_feat.std() > 0:
            print("  Baseline: AlphaMissense score")
            results["alphamissense"] = run_linear_probe(am_feat, labels, genes, seed=seed)
        else:
            results["alphamissense"] = {"note": "AlphaMissense scores have zero variance"}
    else:
        results["alphamissense"] = {"note": "AlphaMissense scores unavailable"}

    return results


def run_negative_controls(deltas_mean, labels, genes, seed=42):
    """
    Two negative controls:
    1. Shuffled delta: randomly reassign delta from a different gene
    2. Returns metrics — signal should be near chance
    """
    results = {}
    rng = np.random.RandomState(seed)

    # Control 1: shuffle deltas across genes (break variant-gene association)
    shuffle_idx = rng.permutation(len(deltas_mean))
    deltas_shuffled = deltas_mean[shuffle_idx]
    print("  Negative control: shuffled deltas")
    results["shuffled_delta"] = run_linear_probe(deltas_shuffled, labels, genes, seed=seed)

    return results


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
    variants = fetch_gerasimavicius_dataset(data_dir)

    # Build 3-class label: HI + AR → LOF
    for v in variants:
        v["label_3class"] = "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]

    # Filter to variants with UniProt ID and valid AA info
    variants = [v for v in variants if v["uniprot_id"] and v["aa_wt"] and v["aa_mut"] and v["aa_pos"] > 0]
    print(f"After filtering: {len(variants)} variants")

    # ------------------------------------------------------------------
    # 2. Fetch protein sequences
    # ------------------------------------------------------------------
    print("\n=== Fetching protein sequences ===")
    seq_cache = build_sequence_cache(variants, data_dir)

    # Prepare WT and mutant sequences
    valid_variants = []
    wt_seqs = []
    mut_seqs = []
    var_positions = []

    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_full = seq_cache[uid]

        # Window long sequences
        wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
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

    # ------------------------------------------------------------------
    # 3. Extract embeddings
    # ------------------------------------------------------------------
    emb_cache_wt = os.path.join(data_dir, f"embeddings_wt_{model_name}.npy")
    emb_cache_mut = os.path.join(data_dir, f"embeddings_mut_{model_name}.npy")
    emb_cache_wt_pos = os.path.join(data_dir, f"embeddings_wt_pos_{model_name}.npy")
    emb_cache_mut_pos = os.path.join(data_dir, f"embeddings_mut_pos_{model_name}.npy")

    if all(os.path.exists(p) for p in [emb_cache_wt, emb_cache_mut]):
        print("\n=== Loading cached embeddings ===")
        emb_wt_mean = np.load(emb_cache_wt)
        emb_mut_mean = np.load(emb_cache_mut)
        if os.path.exists(emb_cache_wt_pos):
            emb_wt_pos = np.load(emb_cache_wt_pos)
            emb_mut_pos = np.load(emb_cache_mut_pos)
        else:
            emb_wt_pos = emb_wt_mean
            emb_mut_pos = emb_mut_mean
    else:
        print(f"\n=== Extracting ESM-2 embeddings ({model_name}) ===")
        emb_wt_mean, emb_mut_mean, emb_wt_pos, emb_mut_pos = get_esm2_embeddings_for_pairs(
            wt_seqs, mut_seqs, var_positions,
            model_name=model_name, device=device, batch_size=batch_size
        )
        np.save(emb_cache_wt, emb_wt_mean)
        np.save(emb_cache_mut, emb_mut_mean)
        np.save(emb_cache_wt_pos, emb_wt_pos)
        np.save(emb_cache_mut_pos, emb_mut_pos)

    # Delta embeddings (co-primary: mean-pooled and per-residue)
    deltas_mean = emb_mut_mean - emb_wt_mean
    deltas_pos = emb_mut_pos - emb_wt_pos

    print(f"Delta embedding shape: {deltas_mean.shape}")

    # Labels and gene arrays
    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    labels_4class = np.array([v["mechanism"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    foldx_ddg = np.array([v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan
                           for v in valid_variants])
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    print("\n=== Fetching AlphaMissense scores ===")
    alphamissense_scores = fetch_alphamissense_scores(valid_variants, data_dir)

    from collections import Counter
    print(f"3-class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    # ------------------------------------------------------------------
    # 4. Stability subspace: fit and validate transfer
    # ------------------------------------------------------------------
    print("\n=== Stability subspace ===")
    megascale_subspace = fit_stability_subspace_megascale(
        data_dir, n_components=n_stability_components,
        model_name=model_name, device=device
    )

    stability_path = "A_megascale"
    transfer_rho = float("nan")

    if megascale_subspace is not None:
        transfer_rho = validate_stability_transfer(megascale_subspace, deltas_mean, foldx_ddg)
        print(f"  Megascale→Gerasimavicius transfer Spearman ρ = {transfer_rho:.3f}")
        print(f"  Pre-registered threshold: ρ > {STABILITY_TRANSFER_RHO_THRESHOLD}")

        if transfer_rho >= STABILITY_TRANSFER_RHO_THRESHOLD:
            stability_subspace = megascale_subspace
            stability_path = "A_megascale"
            print("  Path A: Megascale transfer PASSES — using Megascale subspace")
        else:
            print("  Path A: Megascale transfer FAILS — falling back to Path B")
            stability_subspace = fit_stability_subspace_direct(
                deltas_mean, foldx_ddg, n_components=n_stability_components, genes=genes_arr
            )
            stability_path = "B_direct"
    else:
        print("  Path B: No Megascale data — fitting subspace directly on Gerasimavicius")
        stability_subspace = fit_stability_subspace_direct(
            deltas_mean, foldx_ddg, n_components=n_stability_components, genes=genes_arr
        )
        stability_path = "B_direct"

    # Variance explained per class (pre-registered prediction)
    var_exp = variance_explained_per_class(deltas_mean, labels_3class, stability_subspace)
    print(f"  Variance explained by stability subspace: {var_exp}")

    # Project out stability subspace
    deltas_mean_proj = project_out_subspace(deltas_mean, stability_subspace)
    deltas_pos_proj = project_out_subspace(deltas_pos, stability_subspace)

    # ------------------------------------------------------------------
    # 5. Primary probe: 3-class GOF/DN/LOF
    # ------------------------------------------------------------------
    print("\n=== Primary linear probe (3-class: GOF/DN/LOF) ===")

    results_primary = {}

    # Mean-pooled delta, projected
    print("  Mean-pooled delta (stability-projected):")
    results_primary["mean_pooled_projected"] = run_linear_probe(
        deltas_mean_proj, labels_3class, genes_arr, n_folds=n_cv_folds, seed=seed
    )

    # Mean-pooled delta, unprojected
    print("  Mean-pooled delta (unprojected):")
    results_primary["mean_pooled_unprojected"] = run_linear_probe(
        deltas_mean, labels_3class, genes_arr, n_folds=n_cv_folds, seed=seed
    )

    # Per-residue delta, projected (co-primary)
    print("  Per-residue delta (stability-projected):")
    results_primary["per_residue_projected"] = run_linear_probe(
        deltas_pos_proj, labels_3class, genes_arr, n_folds=n_cv_folds, seed=seed
    )

    # Per-residue delta, unprojected
    print("  Per-residue delta (unprojected):")
    results_primary["per_residue_unprojected"] = run_linear_probe(
        deltas_pos, labels_3class, genes_arr, n_folds=n_cv_folds, seed=seed
    )

    # ------------------------------------------------------------------
    # 6. Secondary probe: 4-class and HI vs AR
    # ------------------------------------------------------------------
    print("\n=== Secondary probes ===")
    results_secondary = {}

    # 4-class
    print("  4-class (GOF/DN/HI/AR):")
    results_secondary["four_class"] = run_linear_probe(
        deltas_mean_proj, labels_4class, genes_arr, n_folds=n_cv_folds, seed=seed
    )

    # HI vs AR only
    hi_ar_mask = np.isin(labels_4class, ["HI", "AR"])
    if hi_ar_mask.sum() >= 20:
        print("  HI vs AR (2-class):")
        results_secondary["hi_vs_ar"] = run_linear_probe(
            deltas_mean_proj[hi_ar_mask], labels_4class[hi_ar_mask],
            genes_arr[hi_ar_mask], n_folds=n_cv_folds, seed=seed
        )

    # ------------------------------------------------------------------
    # 7. Baselines
    # ------------------------------------------------------------------
    print("\n=== Baselines ===")
    results_baselines = run_baselines(
        emb_wt_mean, deltas_mean_proj, foldx_ddg, labels_3class, genes_arr,
        aa_wt_list, aa_mut_list, alphamissense_scores, seed=seed
    )

    # ------------------------------------------------------------------
    # 8. Negative controls
    # ------------------------------------------------------------------
    print("\n=== Negative controls ===")
    results_negctrl = run_negative_controls(deltas_mean_proj, labels_3class, genes_arr, seed=seed)

    # ------------------------------------------------------------------
    # 8b. Gene-family-split CV (robustness check)
    # ------------------------------------------------------------------
    print("\n=== Gene-family-split CV ===")
    pfam_map = fetch_pfam_families(valid_variants, seq_cache, data_dir)
    family_splits = gene_family_split_cv(
        deltas_mean_proj, labels_3class, genes_arr, pfam_map,
        n_folds=n_cv_folds, seed=seed
    )
    results_family_cv = {}
    if family_splits:
        print(f"  Running family-split CV with {len(family_splits)} folds")
        family_fold_results = []
        for train_idx, test_idx in family_splits:
            X_tr, X_te = deltas_mean_proj[train_idx], deltas_mean_proj[test_idx]
            y_tr, y_te = labels_3class[train_idx], labels_3class[test_idx]
            if len(set(y_tr)) < 2:
                continue
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     random_state=seed)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            proba = clf.predict_proba(X_te)
            fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
            for i, cls in enumerate(clf.classes_):
                y_bin = (y_te == cls).astype(int)
                if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
            family_fold_results.append(fm)
        if family_fold_results:
            results_family_cv = {}
            for key in family_fold_results[0]:
                vals = [f[key] for f in family_fold_results if key in f and not np.isnan(f[key])]
                if vals:
                    results_family_cv[f"{key}_mean"] = float(np.mean(vals))
                    results_family_cv[f"{key}_std"] = float(np.std(vals))
        print(f"  Family-split macro-F1: {results_family_cv.get('macro_f1_mean', float('nan')):.3f}")
    else:
        print("  Family-split CV infeasible — skipping")

    # ------------------------------------------------------------------
    # 9. Probe direction orthogonality
    # ------------------------------------------------------------------
    print("\n=== Probe direction orthogonality ===")
    # Use Megascale subspace for path A cosines, or None for path B
    subspace_for_ortho = stability_subspace if stability_path == "A_megascale" else None
    ortho_results = probe_direction_orthogonality(
        deltas_mean_proj, labels_3class, genes_arr,
        stability_subspace=subspace_for_ortho,
        n_folds=n_cv_folds, seed=seed
    )
    print(f"  Cosine matrix: {ortho_results['cosine_matrix']}")
    print(f"  Null cosine mean: {ortho_results['null_cosine_mean']:.3f} ± {ortho_results['null_cosine_std']:.3f}")

    # ------------------------------------------------------------------
    # 10. Compile final_info
    # ------------------------------------------------------------------
    primary_mean_proj = results_primary["mean_pooled_projected"]
    primary_per_proj = results_primary["per_residue_projected"]

    # Pre-registered tiebreak: mean-pooled is headline
    headline_macro_f1 = primary_mean_proj.get("macro_f1_mean", float("nan"))
    headline_auroc_gof = primary_mean_proj.get("auroc_GOF_mean", float("nan"))
    headline_auroc_dn = primary_mean_proj.get("auroc_DN_mean", float("nan"))
    headline_auroc_lof = primary_mean_proj.get("auroc_LOF_mean", float("nan"))

    # Scale interpretation (pre-registered thresholds vs 650M)
    model_scale = "650M" if "650M" in model_name else "3B"

    final_info = {
        # Primary results (headline: mean-pooled, projected)
        "headline_macro_f1": headline_macro_f1,
        "headline_auroc_GOF": headline_auroc_gof,
        "headline_auroc_DN": headline_auroc_dn,
        "headline_auroc_LOF": headline_auroc_lof,
        # Per-residue (co-primary)
        "per_residue_macro_f1": primary_per_proj.get("macro_f1_mean", float("nan")),
        "per_residue_auroc_GOF": primary_per_proj.get("auroc_GOF_mean", float("nan")),
        # Unprojected (for comparison)
        "unprojected_macro_f1": results_primary["mean_pooled_unprojected"].get("macro_f1_mean", float("nan")),
        # Stability subspace info
        "stability_path": stability_path,
        "stability_transfer_rho": transfer_rho,
        "variance_explained_GOF": var_exp.get("GOF", float("nan")),
        "variance_explained_DN": var_exp.get("DN", float("nan")),
        "variance_explained_LOF": var_exp.get("LOF", float("nan")),
        "variance_asymmetry_gof_lof": var_exp.get("gof_lof_asymmetry", float("nan")),
        "variance_asymmetry_prediction_holds": var_exp.get("asymmetry_prediction_holds", False),
        # Baselines
        "baseline_wt_only_macro_f1": results_baselines.get("wt_only", {}).get("macro_f1_mean", float("nan")),
        "baseline_foldx_macro_f1": results_baselines.get("foldx_ddg_only", {}).get("macro_f1_mean", float("nan")),
        # Baselines (new)
        "baseline_onehot_macro_f1": results_baselines.get("onehot_aa", {}).get("macro_f1_mean", float("nan")),
        "baseline_alphamissense_macro_f1": results_baselines.get("alphamissense", {}).get("macro_f1_mean", float("nan")),
        # Negative controls
        "neg_ctrl_shuffled_macro_f1": results_negctrl.get("shuffled_delta", {}).get("macro_f1_mean", float("nan")),
        # Orthogonality
        "cosine_GOF_DN": ortho_results["cosine_matrix"].get("DN_vs_GOF", float("nan")),
        "cosine_GOF_LOF": ortho_results["cosine_matrix"].get("GOF_vs_LOF", float("nan")),
        "cosine_DN_LOF": ortho_results["cosine_matrix"].get("DN_vs_LOF", float("nan")),
        "null_cosine_mean": ortho_results["null_cosine_mean"],
        "ortho_distinguishable_from_null": str(ortho_results["distinguishable_from_null"]),
        # Gene-family-split CV
        "family_cv_macro_f1": results_family_cv.get("macro_f1_mean", float("nan")),
        "family_cv_auroc_GOF": results_family_cv.get("auroc_GOF_mean", float("nan")),
        "family_cv_auroc_DN": results_family_cv.get("auroc_DN_mean", float("nan")),
        "family_cv_auroc_LOF": results_family_cv.get("auroc_LOF_mean", float("nan")),
        # Stability path
        "orthogonality_path": ortho_results.get("path", "unknown"),
        # Dataset stats
        "n_variants": len(valid_variants),
        "n_genes": len(set(genes_arr)),
        "n_GOF": int((labels_3class == "GOF").sum()),
        "n_DN": int((labels_3class == "DN").sum()),
        "n_LOF": int((labels_3class == "LOF").sum()),
        "model": model_name,
        "model_scale": model_scale,
        "seed": seed,
    }

    print("\n=== Final results ===")
    for k, v in final_info.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save all results
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
        "pfam_n_families": len(set(v for v in pfam_map.values() if v is not None)),
    }
    with open(os.path.join(out_dir, f"detailed_results_seed{seed}.json"), "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    return final_info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--model", type=str, default=ESM2_MODEL_650M,
                        choices=[ESM2_MODEL_650M, ESM2_MODEL_3B])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    final_infos_list = []

    for seed in args.seeds:
        print(f"\n{'='*60}\n=== Seed {seed} ===\n{'='*60}")
        fi = run(args.out_dir, seed=seed, model_name=args.model,
                 batch_size=args.batch_size)
        all_results[f"seed{seed}_final_info"] = fi
        final_infos_list.append(fi)

    numeric_keys = [k for k, v in final_infos_list[0].items()
                    if isinstance(v, (int, float)) and not np.isnan(v)]
    final_infos = {
        "means": {k: float(np.mean([d[k] for d in final_infos_list
                                     if isinstance(d.get(k), (int, float))]))
                  for k in numeric_keys},
        "stderrs": {k: float(np.std([d[k] for d in final_infos_list
                                      if isinstance(d.get(k), (int, float))]) / max(len(args.seeds), 1))
                    for k in numeric_keys},
        "final_info_list": final_infos_list,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"\nDone. Results written to {args.out_dir}/final_info.json")
