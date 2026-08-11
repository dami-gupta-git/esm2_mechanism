"""
ESM-3 mechanism family-split experiment (plan_esm3_mechanism.md).

Tests whether ESM-3's structure tokens rescue the mechanism null from ESM-2.

Two embedding conditions (function tokens not implemented — dropped):
  seq        — sequence tokens only (fair scale comparison to ESM-2 650M)
  seq_struct — sequence + AlphaFold2 structure tokens

For each condition: delta = mean_pool(ESM-3(mut)) - mean_pool(ESM-3(wt))
PyTorch MLP + logistic probe, 5-fold gene-split + family-split, seeds 0-4,
3-class GOF/LOF/DN, on one of two datasets (--dataset):
  geras  — Gerasimavicius only (948 genes)
  merged — Gerasimavicius + G2P (1935 genes); matches the ESM-2 classifier report,
           the apples-to-apples comparison for the scale claim.

Decision rules (pre-registered in plan_esm3_mechanism.md):
  M1: ESM-3 seq_struct family-split macro-F1 > ESM-2 floor + 0.05
  M2: ESM-3 seq-only   family-split F1       > ESM-2 floor + 0.05  (scale alone rescues)
  M3: seq_struct − seq > 0.03  (structure adds signal beyond scale)
The ESM-2 floor is read at runtime from the matched MLP delta_mean family-split result
(nonlinear_results_seed*.json), not hardcoded — see esm2_family_floor().

Phases (each takes --dataset; outputs go to per-dataset subdirectories):
  --phase 1   CPU: download AF2 structures, cache coordinates
  --phase 2   GPU: extract ESM-3 embeddings for both conditions
  --phase 3   CPU: run probes, evaluate decision rules, write results

Usage:
  python3 -m esm2_mech.experiments.esm3.esm3_mechanism --phase 1 --dataset merged
  python3 -m esm2_mech.experiments.esm3.esm3_mechanism --phase 2 --dataset merged
  python3 -m esm2_mech.experiments.esm3.esm3_mechanism --phase 3 --dataset merged
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import (
    CACHE_DIR,
    DATA_DIR as DATA,
    ESM3_EMB_DIR,
    ESM3_MODEL,
    ESM3_STRUCT_TOKENS_JSON,
    NONLINEAR_RESULTS_SEED_JSON,
    RESULTS_DIR as _RESULTS_DIR,
    SEQUENCES_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    DELTA_MEAN_FEATURE, MECHANISM_CLASSES, N_FOLDS, N_SEEDS, SPLIT_FAMILY, nonlinear_key,
)
from esm2_mech.utils.bootstrap import (
    average_oof_over_seeds,
    bootstrap_mechanism_metrics,
    family_or_gene_clusters,
)

# The matched ESM-2 probe for the ESM-3 comparison: MLP, delta_mean, family-split.
MLP_DELTA_MEAN_FAMILY = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_FAMILY)
from esm2_mech.utils.io import atomic_write_json, save_npy
from esm2_mech.utils.sequences import apply_missense, window_sequence

AF2_DIR = CACHE_DIR / "af2_structures"

GERAS_VARIANTS = DATA / "gerasimavicius_variants.json"
PFAM_JSON = DATA / "pfam_families.json"

# The structure-token cache (phase 1) is dataset-independent: it is keyed by UniProt
# ID, so geras and merged share it and the merged run only fetches the extra proteins.
STRUCT_TOKENS = ESM3_STRUCT_TOKENS_JSON
AF2_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

# Per-dataset output paths. Resolved by configure_dataset() from --dataset before any
# phase runs, so geras and merged write to separate directories and never collide.
DATASET = None
OUT = None
EMB_DIR = None
EMB_SEQ = None
EMB_SEQ_STRUCT = None
EMB_VALID_IDX = None
STRUCT_META = None


def configure_dataset(dataset: str) -> None:
    """Set the module-level output paths for the chosen dataset (geras | merged).

    Embeddings go to data/embeddings/<model>/<dataset>/, results to
    results/<run>/esm3_mechanism/<dataset>/. Derived from the base path constants
    so both datasets stay under the canonical locations without colliding.
    """
    global DATASET, OUT, EMB_DIR, EMB_SEQ, EMB_SEQ_STRUCT, EMB_VALID_IDX, STRUCT_META
    DATASET = dataset
    EMB_DIR = ESM3_EMB_DIR / dataset
    EMB_SEQ = EMB_DIR / "seq_mean.npy"
    EMB_SEQ_STRUCT = EMB_DIR / "seq_struct_mean.npy"
    EMB_VALID_IDX = EMB_DIR / "valid_idx.npy"
    STRUCT_META = EMB_DIR / "struct_meta.json"
    OUT = _RESULTS_DIR / "esm3_mechanism" / dataset


# Probe config (matches the ESM-2 mechanism classifier exactly)
SEEDS = list(range(N_SEEDS))

# Decision rule margins (pre-registered in plan_esm3_mechanism.md)
M1_MARGIN = 0.05  # ESM-3 must beat the ESM-2 family-split floor by this much
M3_THRESHOLD = 0.03  # seq_struct − seq gap that counts as "structure adds signal"

# Fallback ESM-2 family-split floor, used only if the matched result file is absent.
# The live floor is read from NONLINEAR_RESULTS_SEED_JSON at runtime (esm2_family_floor).
ESM2_FLOOR_FALLBACK = 0.299


def esm2_family_floor(seeds: list[int] = SEEDS) -> tuple[float, str]:
    """Return (floor, source) for the ESM-2 family-split macro-F1 baseline.

    Reads the 5-seed mean of mlp_delta_mean_family from the run's nonlinear-probe
    result files — the matched ESM-2 probe (MLP, delta_mean, family-split) on the
    merged set, the like-for-like comparison to ESM-3 seq. Falls back to the
    pre-registered constant if no seed files are present (e.g. a geras-only checkout).
    """
    values = []
    for seed in seeds:
        path = Path(NONLINEAR_RESULTS_SEED_JSON.format(seed=seed))
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        entry = data.get(MLP_DELTA_MEAN_FAMILY)
        if entry and "macro_f1_mean" in entry:
            values.append(entry["macro_f1_mean"])
    if values:
        return float(np.mean(values)), f"nonlinear_results ({MLP_DELTA_MEAN_FAMILY}, {len(values)}-seed mean)"
    return ESM2_FLOOR_FALLBACK, "fallback constant (no nonlinear_results files found)"


# ── helpers ───────────────────────────────────────────────────────────────────


MECH_MAP = {"GOF": "GOF", "DN": "DN", "HI": "LOF", "AR": "LOF", "LOF": "LOF"}


def _mech3(variant: dict) -> str | None:
    """Collapse a variant's mechanism to GOF/DN/LOF, preferring a precomputed
    label_3class field (matches the ESM-2 classifier's _label_3class) and falling
    back to the HI/AR→LOF map. Returns None for any mechanism outside the 3 classes."""
    if "label_3class" in variant and variant["label_3class"] in ("GOF", "DN", "LOF"):
        return variant["label_3class"]
    return MECH_MAP.get(variant.get("mechanism"))


def _load_variants(variants_path: Path) -> tuple[list[dict], np.ndarray, dict]:
    """Load variants from a JSON file, attach mech3 label + wt_seq, drop variants
    with no cached sequence or no 3-class label. Returns (variants, genes, pfam_map)."""
    variants = json.loads(variants_path.read_text())
    pfam_map = json.loads(PFAM_JSON.read_text()) if PFAM_JSON.exists() else {}
    sequences = (
        json.loads(SEQUENCES_JSON.read_text()) if SEQUENCES_JSON.exists() else {}
    )
    kept = []
    skipped = 0
    for variant in variants:
        mech3 = _mech3(variant)
        if mech3 is None:
            continue
        uid = variant.get("uniprot_id", "")
        seq = sequences.get(uid)
        if not seq:
            skipped += 1
            continue
        variant = dict(variant)
        variant["mech3"] = mech3
        variant["wt_seq"] = seq
        kept.append(variant)
    if skipped:
        print(f"  {variants_path.name}: {skipped} variants skipped (no sequence in cache)")
    genes = np.array([variant["gene"] for variant in kept])
    return kept, genes, pfam_map


def load_dataset() -> tuple[list[dict], np.ndarray, dict]:
    """Load the dataset selected by configure_dataset() (geras | merged)."""
    if DATASET == "merged":
        return _load_variants(VALID_VARIANTS_JSON)
    return _load_variants(GERAS_VARIANTS)


# ── Phase 1: download AF2 structures and tokenise ────────────────────────────


def phase1_structure_tokens() -> None:
    """
    For each unique UniProt ID in the selected dataset, fetch its AF2 structure from
    EBI, tokenise with ESM3StructureTokenizer, cache tokens. The cache is keyed by
    UniProt ID and shared across datasets, so the merged run only fetches the proteins
    geras did not already cover. Genes without AF2 structures fall back to seq-only.
    """
    try:
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.protein_chain import ProteinChain
    except ImportError:
        print("ERROR: esm package not found. Install: pip install esm")
        sys.exit(1)

    AF2_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    variants, _, _ = load_dataset()
    uniprot_ids = sorted({v["uniprot_id"] for v in variants if v.get("uniprot_id")})
    print(f"Unique UniProt IDs: {len(uniprot_ids)}")

    if STRUCT_TOKENS.exists():
        try:
            cached = json.loads(STRUCT_TOKENS.read_text())
        except json.JSONDecodeError:
            print(
                f"WARNING: {STRUCT_TOKENS} is corrupt (partial write?); deleting and re-fetching"
            )
            STRUCT_TOKENS.unlink()
            cached = {}
        already = set(cached.keys())
        print(f"Resuming: {len(already)} already tokenised")
    else:
        cached = {}
        already = set()

    fallback = 0
    for i, uid in enumerate(uniprot_ids):
        if uid in already:
            continue

        # Download AF2 PDB
        pdb_path = AF2_DIR / f"{uid}.pdb"
        if not pdb_path.exists():
            url = AF2_API_URL.format(uniprot_id=uid)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    meta = json.loads(r.read())
                pdb_url = meta[0]["pdbUrl"]
                req2 = urllib.request.Request(
                    pdb_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req2, timeout=60) as r:
                    pdb_path.write_bytes(r.read())
            except Exception as e:
                print(
                    f"  [{i+1}/{len(uniprot_ids)}] {uid}: AF2 fetch failed ({e}), fallback to seq-only"
                )
                cached[uid] = None
                fallback += 1
                continue

        # Tokenise with ESM3StructureTokenizer
        try:
            chain = ProteinChain.from_pdb(str(pdb_path))
            protein = ESMProtein.from_protein_chain(chain)
            # Extract structure token ids as a list (one per residue)
            cached[uid] = (
                protein.coordinates.tolist()
                if protein.coordinates is not None
                else None
            )
            if cached[uid] is None:
                fallback += 1
        except Exception as e:
            print(
                f"  [{i+1}/{len(uniprot_ids)}] {uid}: tokenisation failed ({e}), fallback"
            )
            cached[uid] = None
            fallback += 1

        if (i + 1) % 50 == 0:
            atomic_write_json(STRUCT_TOKENS, cached)
            print(
                f"  Checkpoint: {i+1}/{len(uniprot_ids)}, {fallback} fallbacks so far"
            )

        print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: OK")

    atomic_write_json(STRUCT_TOKENS, cached)
    n_ok = sum(1 for v in cached.values() if v is not None)
    print(
        f"\nStructure tokens cached: {n_ok}/{len(uniprot_ids)} OK, {fallback} seq-only fallbacks"
    )
    print(f"Saved → {STRUCT_TOKENS}")


# ── Phase 2: GPU embedding extraction ────────────────────────────────────────


def phase2_extract_embeddings(batch_size: int = 4) -> None:
    """
    For each variant (wt and mut sequence) extract ESM-3 mean-pooled embeddings
    under two conditions: seq-only, seq+struct.
    Saves delta arrays (mut - wt) to EMB_SEQ, EMB_SEQ_STRUCT.
    """
    try:
        import torch
        from esm.sdk.api import ESMProtein
        from esm.pretrained import ESM3_sm_open_v0
    except ImportError:
        print("ERROR: esm package not found. Install: pip install esm")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase 2 requires a GPU.")

    print(f"Loading ESM-3 ({ESM3_MODEL}) on {device}...")
    model = ESM3_sm_open_v0(device=device)
    model.eval()

    variants, _, _ = load_dataset()
    n = len(variants)
    print(f"Variants to embed: {n}")

    # Load structure token cache (uid -> coordinates list or None)
    struct_cache: dict = {}
    if STRUCT_TOKENS.exists():
        struct_cache = json.loads(STRUCT_TOKENS.read_text())
        n_struct = sum(1 for v in struct_cache.values() if v is not None)
        print(
            f"Structure tokens loaded: {n_struct}/{len(struct_cache)} have structures"
        )

    # Two conditions only — function tokens not implemented
    conditions = {
        "seq": EMB_SEQ,
        "seq_struct": EMB_SEQ_STRUCT,
    }
    for cond, path in conditions.items():
        if path.exists():
            arr = np.load(str(path))
            print(f"  {cond}: cached ({arr.shape})")

    remaining_conds = [c for c, path in conditions.items() if not path.exists()]
    if not remaining_conds:
        print("All conditions already cached.")
        return

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoints if available
    wt_embs: dict[str, list] = {}
    mut_embs: dict[str, list] = {}
    valid_indices: list[int] = []
    resume_from = 0
    idx_ckpt = Path(str(EMB_VALID_IDX).replace(".npy", "_ckpt.npy"))
    if idx_ckpt.exists():
        valid_indices = list(np.load(str(idx_ckpt)).astype(int))
    for cond in remaining_conds:
        ckpt_wt = Path(str(conditions[cond]).replace(".npy", "_ckpt_wt.npy"))
        ckpt_mut = Path(str(conditions[cond]).replace(".npy", "_ckpt_mut.npy"))
        if ckpt_wt.exists() and ckpt_mut.exists():
            wt_embs[cond] = list(np.load(str(ckpt_wt)))
            mut_embs[cond] = list(np.load(str(ckpt_mut)))
            resume_from = max(
                resume_from, max(valid_indices) + 1 if valid_indices else 0
            )
        else:
            wt_embs[cond] = []
            mut_embs[cond] = []
    if resume_from > 0:
        # The three checkpoint files (valid_idx, wt, mut) are written separately and
        # are not atomic as a group: an interrupt between writes can leave them out of
        # sync, which would silently misalign valid_idx against the embedding rows in
        # phase 3. Fail loudly rather than continue from a corrupted checkpoint.
        for cond in remaining_conds:
            if not (len(wt_embs[cond]) == len(mut_embs[cond]) == len(valid_indices)):
                raise RuntimeError(
                    f"Checkpoint length mismatch for {cond}: "
                    f"wt={len(wt_embs[cond])} mut={len(mut_embs[cond])} "
                    f"valid_idx={len(valid_indices)} — checkpoint corrupted by a "
                    f"mid-write interrupt; delete the *_ckpt*.npy files and restart phase 2"
                )
        print(
            f"Resuming from checkpoint: {len(valid_indices)} variants done, next variant index {resume_from}"
        )

    struct_fallback_count = [0]  # mutable counter accessible in closure

    def get_struct_tokens(
        uid: str | None, wt_seq_full: str, win_start: int, win_len: int
    ) -> "torch.Tensor | None":
        """
        Return structure tokens for the windowed region [win_start, win_start+win_len).
        win_start is 0-indexed start of the window in the full sequence.
        Slices the AF2 coordinate array to match the windowed sequence.
        Returns None and falls back to seq-only if structure unavailable.
        """
        if uid is None or struct_cache.get(uid) is None:
            return None
        try:
            coords = torch.tensor(struct_cache[uid], dtype=torch.float32)
            L_full = coords.shape[0]
            # Slice coords to the window — if AF2 length doesn't match full seq, skip
            if L_full != len(wt_seq_full):
                struct_fallback_count[0] += 1
                return None
            coords_win = coords[win_start : win_start + win_len]
            if coords_win.shape[0] != win_len:
                struct_fallback_count[0] += 1
                return None
            protein = ESMProtein(sequence="A" * win_len, coordinates=coords_win)
            tensor = model.encode(protein)
            return tensor.structure
        except Exception:
            struct_fallback_count[0] += 1
            return None

    def embed_sequence(
        seq: str, struct_toks: "torch.Tensor | None", condition: str
    ) -> np.ndarray:
        """Run ESM-3 forward pass; return mean-pooled (D,) embedding (excluding BOS/EOS)."""
        protein = ESMProtein(sequence=seq)
        tensor = model.encode(protein)
        seq_tok = tensor.sequence.unsqueeze(0).to(device)  # (1, L+2)

        use_struct = condition == "seq_struct" and struct_toks is not None
        s_tok = struct_toks.unsqueeze(0).to(device) if use_struct else None

        with torch.inference_mode():
            out = model(sequence_tokens=seq_tok, structure_tokens=s_tok)

        # out.embeddings: (1, L+2, D) — exclude BOS (0) and EOS (-1)
        emb = out.embeddings[0, 1:-1].mean(dim=0).cpu().float().numpy()
        return emb

    import time

    t_start = time.time()
    n_struct_applied = 0

    for i, v in enumerate(variants):
        if i < resume_from:
            continue
        uid = v.get("uniprot_id")
        wt_seq = v["wt_seq"]
        pos = v["aa_pos"]

        wt_win, new_pos, win_start = window_sequence(wt_seq, pos)
        if new_pos < 1 or new_pos > len(wt_win):
            print(
                f"  SKIP variant {i}: pos={pos} new_pos={new_pos} out of range for seq len {len(wt_win)}"
            )
            continue
        # Apply the substitution through the shared helper so ESM-3 drops the exact
        # same variants ESM-2 does: apply_missense returns None on a WT-reference
        # mismatch (wrong isoform / off-by-one), preventing a delta computed on a
        # wrong wt/mut pair. Building the mutant by blind overwrite would silently
        # embed a corrupt pair and diverge from ESM-2's row set.
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            print(
                f"  SKIP variant {i}: WT mismatch at pos={pos} "
                f"(expected {v['aa_wt']}, window has {wt_win[new_pos - 1]})"
            )
            continue

        # win_start is the window offset returned by window_sequence itself, so the
        # coordinate slice below uses exactly the same offset as the sequence window
        # (previously a duplicated formula diverged by 11 residues for long proteins).
        struct_toks = get_struct_tokens(uid, wt_seq, win_start, len(wt_win))
        if struct_toks is not None:
            n_struct_applied += 1

        valid_indices.append(i)
        for cond in remaining_conds:
            wt_e = embed_sequence(wt_win, struct_toks, cond)
            mut_e = embed_sequence(mut_win, struct_toks, cond)
            wt_embs[cond].append(wt_e)
            mut_embs[cond].append(mut_e)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            done = (i + 1) - resume_from
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] {rate:.1f} var/s  ETA {eta/60:.0f} min")
            save_npy(
                str(EMB_VALID_IDX).replace(".npy", "_ckpt.npy"),
                np.array(valid_indices, dtype=np.int32),
            )
            for cond in remaining_conds:
                save_npy(
                    str(conditions[cond]).replace(".npy", "_ckpt_wt.npy"),
                    np.array(wt_embs[cond]),
                )
                save_npy(
                    str(conditions[cond]).replace(".npy", "_ckpt_mut.npy"),
                    np.array(mut_embs[cond]),
                )

    # Save valid variant indices for phase 3 label alignment
    valid_idx_arr = np.array(valid_indices, dtype=np.int32)
    save_npy(str(EMB_VALID_IDX), valid_idx_arr)
    n_skipped = n - len(valid_indices)
    print(f"\nVariants embedded: {len(valid_indices)}/{n}  skipped={n_skipped}")
    print(
        f"Structure applied: {n_struct_applied}/{len(valid_indices)} "
        f"({100*n_struct_applied/max(1,len(valid_indices)):.1f}%)"
    )
    print(f"Structure coord-length fallbacks: {struct_fallback_count[0]}")

    # Persist structure coverage for phase 3 / result_26 (counts reflect this run;
    # may undercount if phase 2 was resumed from a checkpoint).
    STRUCT_META.write_text(
        json.dumps(
            {
                "n_variants_total": n,
                "n_variants_embedded": len(valid_indices),
                "n_variants_skipped": n_skipped,
                "n_structure_applied": n_struct_applied,
                "structure_applied_frac": n_struct_applied / max(1, len(valid_indices)),
                "n_struct_coord_fallback": struct_fallback_count[0],
            },
            indent=2,
        )
    )

    for cond in remaining_conds:
        wt_arr = np.array(wt_embs[cond])
        mut_arr = np.array(mut_embs[cond])
        delta = mut_arr - wt_arr
        save_npy(str(conditions[cond]), delta)
        # Persist the raw wt and mut arrays alongside the delta so downstream
        # reports (mut-only probes, wt/mut geometry) can reuse them without re-embedding.
        wt_path = str(conditions[cond]).replace(".npy", "_wt.npy")
        mut_path = str(conditions[cond]).replace(".npy", "_mut.npy")
        save_npy(wt_path, wt_arr)
        save_npy(mut_path, mut_arr)
        print(
            f"  {cond}: delta {delta.shape} → {conditions[cond]}; "
            f"wt → {wt_path}; mut → {mut_path}"
        )
        for suffix in ("_ckpt_wt.npy", "_ckpt_mut.npy"):
            p = Path(str(conditions[cond]).replace(".npy", suffix))
            if p.exists():
                p.unlink()
    if idx_ckpt.exists():
        idx_ckpt.unlink()

    print("Phase 2 complete.")


# ── Phase 3: probes and decision rules ───────────────────────────────────────


def _run_mlp(
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    n_classes: int,
    seed: int,
    genes: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PyTorch MLP (256→64) with class-weighted cross-entropy and early stopping.
    Matches result_7's run_mlp_probe exactly.
    Returns (all_pred, all_true, all_proba, all_genes, all_rows) concatenated
    across folds — all_genes/all_rows (test-fold gene ids and original row
    indices) are for dependency-aware inference (cluster bootstrap); empty
    arrays when `genes` is None.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_pred, all_true, all_proba, all_genes, all_rows = [], [], [], [], []

    for fold_i, (tr, te) in enumerate(splits):
        X_tr, X_te = X[tr].astype(np.float32), X[te].astype(np.float32)
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr)) < 3:
            continue

        # 15% gene-disjoint validation split — matches run_mlp_probe in mlp.py
        rng = np.random.RandomState(seed + fold_i)
        if genes is not None:
            tr_genes = genes[tr]
            unique_tr_genes = np.array(sorted(set(tr_genes)))
            rng.shuffle(unique_tr_genes)
            n_val_genes = max(1, int(0.15 * len(unique_tr_genes)))
            val_gene_set = set(unique_tr_genes[:n_val_genes])
            val_mask = np.array([g in val_gene_set for g in tr_genes])
        else:
            idx = np.arange(len(y_tr))
            rng.shuffle(idx)
            n_val = max(1, int(0.15 * len(idx)))
            val_mask = np.zeros(len(y_tr), dtype=bool)
            val_mask[idx[:n_val]] = True
        fit_mask = ~val_mask
        X_fit, y_fit = X_tr[fit_mask], y_tr[fit_mask]
        X_val, y_val = X_tr[val_mask], y_tr[val_mask]
        if len(X_fit) < 10 or len(X_val) < 5:
            continue

        mu = X_fit.mean(0)
        std = X_fit.std(0) + 1e-8
        X_fit = (X_fit - mu) / std
        X_val = (X_val - mu) / std
        X_te_n = (X_te - mu) / std

        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        cw = torch.tensor(1.0 / (counts + 1e-8)).to(device)

        layers = []
        prev = X_fit.shape[1]
        for h in (256, 64):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        mlp = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
        crit = nn.CrossEntropyLoss(weight=cw)

        ds = TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit, dtype=torch.long))
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        best_val, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(100):
            mlp.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(mlp(xb.to(device)), yb.to(device)).backward()
                opt.step()
            mlp.eval()
            with torch.no_grad():
                vl = crit(
                    mlp(torch.tensor(X_val).to(device)),
                    torch.tensor(y_val, dtype=torch.long).to(device),
                ).item()
            if vl < best_val - 1e-4:
                best_val = vl
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= 10:
                    break

        if best_state:
            mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            proba = (
                torch.softmax(mlp(torch.tensor(X_te_n).to(device)), dim=1).cpu().numpy()
            )
        pred = proba.argmax(axis=1)
        all_pred.append(pred)
        all_true.append(y_te)
        all_proba.append(proba)
        if genes is not None:
            all_genes.append(genes[te])
            all_rows.append(np.asarray(te))

    if not all_pred:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(all_pred),
        np.concatenate(all_true),
        np.vstack(all_proba),
        np.concatenate(all_genes) if all_genes else np.array([]),
        np.concatenate(all_rows) if all_rows else np.array([]),
    )


def phase3_probes(
    seeds: list[int] = SEEDS,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> None:
    from esm2_mech.utils.splits import gene_split_cv, family_split_cv
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)

    variants, genes, pfam_map = load_dataset()
    y_labels = np.array([v["mech3"] for v in variants])
    label_set = MECHANISM_CLASSES
    y_all = np.array([label_set.index(label) for label in y_labels])
    n_classes = len(label_set)

    # Load valid indices saved by phase 2 for exact label alignment
    if EMB_VALID_IDX.exists():
        valid_idx = np.load(str(EMB_VALID_IDX))
        y = y_all[valid_idx]
        genes_valid = genes[valid_idx]
        print(f"Valid indices loaded: {len(valid_idx)}/{len(y_all)} variants embedded")
    else:
        y = y_all
        genes_valid = genes
        print("No valid index file found — assuming all variants embedded")

    conditions = {
        "seq": EMB_SEQ,
        "seq_struct": EMB_SEQ_STRUCT,
    }

    for cond, path in conditions.items():
        if path.exists():
            arr = np.load(str(path))
            print(
                f"  {cond}: {arr.shape[0]} variants embedded (of {len(variants)} total)"
            )

    results = {}

    for cond, path in conditions.items():
        if not path.exists():
            print(f"  SKIP {cond}: {path} not found")
            continue

        delta = np.load(str(path))
        if delta.shape[0] != len(y):
            raise RuntimeError(
                f"{cond}: delta rows {delta.shape[0]} != labels {len(y)} — valid index mismatch"
            )
        print(f"\n=== Condition: {cond}  shape={delta.shape} ===")
        y_cond = y
        genes_cond = genes_valid

        cond_results: dict = {"gene_split": {}, "family_split": {}}

        for cv_name, get_splits in [
            ("gene_split", lambda seed: gene_split_cv(genes_cond, N_FOLDS, seed)),
            (
                "family_split",
                lambda seed: family_split_cv(genes_cond, pfam_map, N_FOLDS, seed),
            ),
        ]:
            mlp_f1s, mlp_gof, mlp_dn, mlp_lof = [], [], [], []
            lr_f1s = []
            seed_oof_list = []

            for seed in seeds:
                splits = get_splits(seed)
                if not splits:
                    print(f"  {cv_name} seed={seed}: no valid splits, skip")
                    continue

                print(
                    f"  {cond} {cv_name} seed={seed}: training MLP "
                    f"({len(splits)} folds)..."
                )

                # PyTorch MLP — matches result_7
                pred, true, proba, oof_genes, oof_rows = _run_mlp(
                    delta, y_cond, splits, n_classes, seed, genes=genes_cond
                )
                if len(pred) == 0:
                    continue
                if compute_ci:
                    seed_oof_list.append({
                        "y_true": np.array([label_set[i] for i in true]),
                        "proba": proba,
                        "genes": oof_genes,
                        "row_ids": oof_rows,
                    })
                mlp_f1s.append(f1_score(true, pred, average="macro"))
                mlp_gof.append(
                    roc_auc_score(
                        (true == label_set.index("GOF")).astype(int),
                        proba[:, label_set.index("GOF")],
                    )
                )
                mlp_dn.append(
                    roc_auc_score(
                        (true == label_set.index("DN")).astype(int),
                        proba[:, label_set.index("DN")],
                    )
                )
                mlp_lof.append(
                    roc_auc_score(
                        (true == label_set.index("LOF")).astype(int),
                        proba[:, label_set.index("LOF")],
                    )
                )

                # Logistic regression
                scaler = StandardScaler()
                lr_preds, lr_true = [], []
                for tr, te in splits:
                    X_tr = scaler.fit_transform(delta[tr])
                    X_te = scaler.transform(delta[te])
                    clf = LogisticRegression(
                        max_iter=1000,
                        random_state=seed,
                        class_weight="balanced",
                        C=0.1,
                    )
                    clf.fit(X_tr, y_cond[tr])
                    lr_preds.append(clf.predict(X_te))
                    lr_true.append(y_cond[te])
                lr_f1s.append(
                    f1_score(
                        np.concatenate(lr_true),
                        np.concatenate(lr_preds),
                        average="macro",
                    )
                )

            if not mlp_f1s:
                continue

            r = {
                "mlp_f1_mean": float(np.mean(mlp_f1s)),
                "mlp_f1_std": float(np.std(mlp_f1s)),
                "mlp_gof_auroc_mean": float(np.mean(mlp_gof)),
                "mlp_dn_auroc_mean": float(np.mean(mlp_dn)),
                "mlp_lof_auroc_mean": float(np.mean(mlp_lof)),
                "lr_f1_mean": float(np.mean(lr_f1s)),
                "lr_f1_std": float(np.std(lr_f1s)),
                "n_seeds": len(mlp_f1s),
            }
            if compute_ci:
                # Each seed reshuffles the CV fold assignment, so its OOF cannot be
                # bootstrapped directly against another seed's — average_oof_over_seeds
                # collapses the per-seed OOF predictions to one proba-per-variant
                # first (matching classify_by_mechanism's cross-seed CI convention),
                # then the cluster bootstrap runs once over that combined OOF.
                combined_oof = average_oof_over_seeds(seed_oof_list)
                if combined_oof is not None:
                    clusters = family_or_gene_clusters(
                        combined_oof["genes"], pfam_map,
                        is_family_split=(cv_name == "family_split"),
                    )
                    r["ci"] = bootstrap_mechanism_metrics(
                        combined_oof["y_true"], combined_oof["proba"],
                        clusters, n_resamples=n_boot, seed=0,
                    )
            cond_results[cv_name] = r
            print(
                f"  {cv_name}: MLP F1={r['mlp_f1_mean']:.3f}±{r['mlp_f1_std']:.3f}  "
                f"GOF={r['mlp_gof_auroc_mean']:.3f}  DN={r['mlp_dn_auroc_mean']:.3f}  "
                f"LOF={r['mlp_lof_auroc_mean']:.3f}  "
                f"LR F1={r['lr_f1_mean']:.3f}"
            )

        results[cond] = cond_results

    # ── decision rules ─────────────────────────────────────────────────────
    esm2_floor, floor_source = esm2_family_floor(seeds)
    m1_threshold = esm2_floor + M1_MARGIN
    print(f"\n=== DECISION RULES ===")
    print(f"  ESM-2 family-split floor = {esm2_floor:.3f}  [{floor_source}]")
    print(f"  M1/M2 threshold = {m1_threshold:.3f}  (floor + {M1_MARGIN})")

    def get_f1(cond: str, cv: str) -> float:
        return results.get(cond, {}).get(cv, {}).get("mlp_f1_mean", float("nan"))

    ss_f1 = get_f1("seq_struct", "family_split")
    seq_f1 = get_f1("seq", "family_split")

    m1 = ss_f1 > m1_threshold if not np.isnan(ss_f1) else None
    m2 = seq_f1 > m1_threshold if not np.isnan(seq_f1) else None
    m3 = (
        (ss_f1 - seq_f1) > M3_THRESHOLD
        if not np.isnan(ss_f1) and not np.isnan(seq_f1)
        else None
    )

    def fmt(v, passed):
        s = f"{v:.3f}" if not np.isnan(v) else "N/A"
        return (
            f"{s} → {'PASS ✓' if passed else 'FAIL ✗' if passed is not None else 'N/A'}"
        )

    print(
        f"  M1: ESM-3 seq_struct family-split F1 > {m1_threshold:.3f} → {fmt(ss_f1, m1)}"
    )
    print(
        f"  M2: ESM-3 seq        family-split F1 > {m1_threshold:.3f} → {fmt(seq_f1, m2)}"
    )
    gap = (
        ss_f1 - seq_f1 if not np.isnan(ss_f1) and not np.isnan(seq_f1) else float("nan")
    )
    print(
        f"  M3: seq_struct − seq > {M3_THRESHOLD:.3f}                 → {fmt(gap, m3)}"
    )

    if m1 is False:
        print(
            "\n  Interpretation: NULL CONFIRMED — mechanism not recoverable from ESM-3 "
            "sequence or sequence+structure representations (function tokens not tested)."
        )
    elif m1 and m2 is False and m3:
        print(
            "\n  Interpretation: STRUCTURE RESCUES MECHANISM — structure tokens are the operative ingredient."
        )
    elif m1 and m2:
        print(
            "\n  Interpretation: SCALE SUFFICES — model scale (not structure tokens) accounts for the lift."
        )
    elif m1 and m3 is False:
        print(
            "\n  Interpretation: ESM-3 better than ESM-2 but structure tokens not the reason."
        )

    struct_meta = json.loads(STRUCT_META.read_text()) if STRUCT_META.exists() else None

    summary = {
        "esm2_baseline_family_split_f1": esm2_floor,
        "esm2_baseline_source": floor_source,
        "m1_threshold": m1_threshold,
        "conditions": "seq, seq_struct (function tokens not implemented)",
        "structure_coverage": struct_meta,
        "results": results,
        "decision_rules": {
            "M1": {
                "criterion": f"seq_struct family-split F1 > {m1_threshold:.3f}",
                "value": ss_f1,
                "passed": m1,
            },
            "M2": {
                "criterion": f"seq family-split F1 > {m1_threshold:.3f}",
                "value": seq_f1,
                "passed": m2,
            },
            "M3": {
                "criterion": f"seq_struct − seq > {M3_THRESHOLD:.3f}",
                "value": float(gap) if not np.isnan(gap) else None,
                "passed": m3,
            },
        },
        "model": ESM3_MODEL,
        "dataset": DATASET,
        "n_folds": N_FOLDS,
        "seeds": seeds,
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote → {OUT}/summary.json")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        required=True,
        choices=["1", "2", "3"],
        help="1=structure tokens (CPU), 2=embeddings (GPU), 3=probes (CPU)",
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--dataset",
        choices=["geras", "merged"],
        default="geras",
        help="geras=Gerasimavicius only (948 genes); merged=Gerasimavicius+G2P (matches ESM-2 classifier)",
    )
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help="number of probe seeds for phase 3; runs 0..seeds-1 (>=1)")
    ap.add_argument("--no_ci", action="store_true",
                    help="phase 3 only: skip cluster-bootstrap CIs")
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")

    configure_dataset(args.dataset)
    print(f"Dataset: {args.dataset}  →  embeddings {EMB_DIR}, results {OUT}")

    if args.phase == "1":
        print("=== Phase 1: AF2 structure download + ESM-3 tokenisation ===")
        phase1_structure_tokens()
    elif args.phase == "2":
        print("=== Phase 2: ESM-3 embedding extraction (GPU) ===")
        phase2_extract_embeddings(batch_size=args.batch_size)
    elif args.phase == "3":
        print("=== Phase 3: probes + decision rules ===")
        phase3_probes(
            seeds=list(range(args.seeds)),
            compute_ci=not args.no_ci, n_boot=args.n_boot,
        )


if __name__ == "__main__":
    main()
