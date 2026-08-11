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
for every seed (nonlinear_results_seed*.json), never hardcoded or defaulted — a missing
or partial baseline raises rather than moving the gate. See esm2_family_floor().

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
import math
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import (
    CACHE_DIR,
    DATA_DIR as DATA,
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
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
    DELTA_MEAN_FEATURE, DN, GOF, HTTP_USER_AGENT, LOF, MECHANISM_CLASSES,
    MIN_TRAIN_CLASSES, N_FOLDS, N_SEEDS, SPLIT_FAMILY, nonlinear_key,
)
from esm2_mech.fetch_data.uniprot_fetch import TransientFetchError, fetch_with_retries
from esm2_mech.utils.bootstrap import (
    average_oof_over_seeds,
    bootstrap_mechanism_metrics,
    family_or_gene_clusters,
    paired_cluster_bootstrap_diff,
)

# The matched ESM-2 probe for the ESM-3 comparison: MLP, delta_mean, family-split.
MLP_DELTA_MEAN_FAMILY = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_FAMILY)
from esm2_mech.utils.io import (
    atomic_write_json,
    atomic_write_text,
    load_json_or_discard,
    save_npy,
)
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import run_mlp_probe_cv
from esm2_mech.utils.sequences import apply_missense, window_sequence

AF2_DIR = CACHE_DIR / "af2_structures"

# Failures that are properties of the environment, not of the structure being
# tokenised. These must never be cached as "no structure" — the next run would
# skip a protein AF2 may well have modelled. (KeyboardInterrupt/SystemExit derive
# from BaseException and are already outside `except Exception`.)
INFRASTRUCTURE_ERRORS = (MemoryError, ImportError, OSError, RecursionError)

GERAS_VARIANTS = DATA / "gerasimavicius_variants.json"
PFAM_JSON = DATA / "pfam_families.json"

# The structure-token cache (phase 1) is dataset-independent: it is keyed by UniProt
# ID, so geras and merged share it and the merged run only fetches the extra proteins.
STRUCT_TOKENS = ESM3_STRUCT_TOKENS_JSON
AF2_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

# Inverse regularisation strength for the logistic arm. Deliberately stronger
# regularisation than the shared logreg probe's default, since the ESM-3 delta
# matrices are high-dimensional relative to the variant count.
LOGREG_C = 0.1

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

# The matched ESM-2 arm, run inside this module on the ESM-3 variant subset under the
# same folds and seeds. Distinct from `esm2_family_floor`, which is the pre-registered
# full-merged-set floor the M1/M2 thresholds are pinned to.
ESM2_COND = "esm2_delta_mean"


def esm2_matched_delta(n_variants: int, valid_idx: np.ndarray | None) -> np.ndarray:
    """ESM-2 delta_mean rows for the ESM-3 variant subset, for the paired arms.

    Only defined for --dataset merged: there load_dataset() reads VALID_VARIANTS_JSON,
    the same file and row order the ESM-2 embeddings were extracted from, so `valid_idx`
    (phase 2's index into that variant list) selects the matching embedding rows. The
    geras variant list has its own ordering and cannot be indexed this way.
    """
    if DATASET != "merged":
        raise ValueError(
            f"ESM-2 matched arm requires --dataset merged; got {DATASET!r} "
            "(the geras variant list is not row-aligned to the ESM-2 embeddings)"
        )
    wt = np.load(str(EMB_WT_MEAN))
    mut = np.load(str(EMB_MUT_MEAN))
    if wt.shape != mut.shape:
        raise RuntimeError(
            f"ESM-2 embeddings shape mismatch: wt {wt.shape} vs mut {mut.shape}"
        )
    if wt.shape[0] != n_variants:
        raise RuntimeError(
            f"ESM-2 embeddings have {wt.shape[0]} rows but the variant list has "
            f"{n_variants} — the ESM-2 arm would not be row-aligned to the ESM-3 arms"
        )
    delta = mut - wt
    return delta if valid_idx is None else delta[valid_idx]


def paired_family_split_diff(
    oof_a: dict | None,
    oof_b: dict | None,
    pfam_map: dict,
    n_boot: int,
    label: str,
) -> dict | None:
    """Paired family-cluster bootstrap CI on macro-F1(A) − macro-F1(B).

    Both arms were scored under the same family-split folds over the same variants, so
    the difference is a same-fold paired statistic: one family resample per replicate,
    the identical drawn rows handed to both arms. Rows scored in only one arm (a fold
    skipped there but not here) are dropped, since the pairing is undefined on them.
    """
    from sklearn.metrics import f1_score

    if oof_a is None or oof_b is None:
        print(f"  [paired] {label}: skipped — an arm has no combined OOF")
        return None

    rows_a = {int(row): pos for pos, row in enumerate(oof_a["row_ids"])}
    rows_b = {int(row): pos for pos, row in enumerate(oof_b["row_ids"])}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        print(f"  [paired] {label}: skipped — the arms share no scored variants")
        return None
    dropped = (len(rows_a) - len(shared)) + (len(rows_b) - len(shared))
    if dropped:
        print(
            f"  [paired] {label}: {len(shared)} shared variants, "
            f"{dropped} arm-specific rows dropped"
        )

    idx_a = np.array([rows_a[row] for row in shared], dtype=int)
    idx_b = np.array([rows_b[row] for row in shared], dtype=int)

    y_a = np.asarray(oof_a["y_true"])[idx_a]
    y_b = np.asarray(oof_b["y_true"])[idx_b]
    if not np.array_equal(y_a, y_b):
        raise RuntimeError(
            f"{label}: the two arms disagree on the label of a shared variant — "
            "the OOF row spaces are not the same variants"
        )

    def _predictions(oof: dict, idx: np.ndarray) -> np.ndarray:
        proba = np.asarray(oof["proba"])[idx]
        return np.array([MECHANISM_CLASSES[col] for col in proba.argmax(axis=1)])

    pred_a = _predictions(oof_a, idx_a)
    pred_b = _predictions(oof_b, idx_b)

    def _macro_f1(pred: np.ndarray):
        def _fn(rows: np.ndarray) -> float:
            return float(
                f1_score(y_a[rows], pred[rows], average="macro", zero_division=0)
            )
        return _fn

    clusters = family_or_gene_clusters(
        np.asarray(oof_a["genes"], dtype=object)[idx_a], pfam_map, is_family_split=True
    )
    out = paired_cluster_bootstrap_diff(
        clusters,
        _macro_f1(pred_a),
        _macro_f1(pred_b),
        n_resamples=n_boot,
        seed=0,
    )
    out["n_shared"] = len(shared)
    out["n_dropped"] = dropped
    return out


def adjudicate_diff(passed: bool | None, diff_ci: dict | None, threshold: float) -> str:
    """R7.1 verdict for a gate, reading its point estimate against its paired CI.

    The point estimate decides pass/fail; the CI decides whether that reading is
    established or underpowered. A pass whose CI spans zero is reported as consistent
    in direction but not distinguishable, never as an established effect.
    """
    if passed is None:
        return "not adjudicated (no point estimate)"
    if diff_ci is None or diff_ci.get("ci_suppressed") or diff_ci.get("ci_low") is None:
        return f"{'pass' if passed else 'fail'}, no CI"
    ci_low, ci_high = diff_ci["ci_low"], diff_ci["ci_high"]
    if passed:
        if ci_low > 0:
            return "pass, established (CI excludes zero)"
        return "pass on point estimate, not distinguishable (CI spans zero)"
    if ci_high > threshold:
        return "fail, underpowered (CI spans the pre-registered threshold)"
    return "fail, established (CI excludes the pre-registered threshold)"


def esm2_family_floor(seeds: list[int] = SEEDS) -> tuple[float, str]:
    """Return (floor, source) for the ESM-2 family-split macro-F1 baseline.

    Reads the mean of mlp_delta_mean_family over ALL of `seeds` from the run's
    nonlinear-probe result files — the matched ESM-2 probe (MLP, delta_mean,
    family-split) on the merged set, the like-for-like comparison to ESM-3 seq.
    The M1/M2 thresholds are this floor plus a margin, so a floor averaged over a
    different seed set than the ESM-3 arm uses, or a floor substituted from any
    other source, silently moves the gate. Every requested seed must be present
    with a finite value; otherwise this raises.
    """
    values = []
    for seed in seeds:
        path = Path(NONLINEAR_RESULTS_SEED_JSON.format(seed=seed))
        with open(path) as fh:
            data = json.load(fh)
        entry = data.get(MLP_DELTA_MEAN_FAMILY)
        if not entry or "macro_f1_mean" not in entry:
            raise KeyError(f"{MLP_DELTA_MEAN_FAMILY}.macro_f1_mean missing from {path}")
        value = entry["macro_f1_mean"]
        if value is None or not math.isfinite(value):
            raise ValueError(
                f"{MLP_DELTA_MEAN_FAMILY}.macro_f1_mean is {value!r} in {path}"
            )
        values.append(float(value))
    return float(np.mean(values)), (
        f"nonlinear_results ({MLP_DELTA_MEAN_FAMILY}, {len(values)}-seed mean, "
        f"seeds={list(seeds)})"
    )


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
    # No empty-dict fallback: every variant needs a cached sequence, so an absent
    # cache would drop all of them into `skipped` and return an empty set that
    # looks like a legitimate "no variants matched" result.
    sequences = json.loads(SEQUENCES_JSON.read_text())
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

    # The shared loader, not a local try/except: it also catches UnicodeDecodeError,
    # which is what a truncated or zero-byte cache raises during the read, before
    # json parsing starts. It deletes the bad file so this run re-fetches.
    cached = load_json_or_discard(STRUCT_TOKENS) or {}
    already = set(cached.keys())
    print(f"Resuming: {len(already)} already tokenised")

    transient = 0
    for i, uid in enumerate(uniprot_ids):
        if uid in already:
            continue

        # Download AF2 PDB. A transient failure must NOT be written to the cache:
        # `already` is keyed on cached.keys(), so a cached None would permanently
        # downgrade this protein to seq-only and bias the M3 contrast.
        pdb_path = AF2_DIR / f"{uid}.pdb"
        if not pdb_path.exists():
            url = AF2_API_URL.format(uniprot_id=uid)
            try:
                meta_body = fetch_with_retries(
                    url,
                    headers={"User-Agent": HTTP_USER_AGENT},
                    timeout=30,
                    label=f"{uid} AF2 metadata",
                )
                if meta_body is None:
                    # HTTP 404 — AF2 has no model for this accession. Real result.
                    print(
                        f"  [{i+1}/{len(uniprot_ids)}] {uid}: no AF2 model (404), seq-only"
                    )
                    cached[uid] = None
                    continue
                meta = json.loads(meta_body)
                pdb_url = meta[0]["pdbUrl"]
                pdb_body = fetch_with_retries(
                    pdb_url,
                    headers={"User-Agent": HTTP_USER_AGENT},
                    timeout=60,
                    label=f"{uid} AF2 PDB",
                )
                if pdb_body is None:
                    raise TransientFetchError(
                        f"{uid}: metadata listed {pdb_url} but it returned 404"
                    )
                # Atomic write: a partial download must never leave a file that
                # pdb_path.exists() would treat as a complete structure next run.
                atomic_write_text(pdb_path, pdb_body)
            except TransientFetchError as exc:
                print(
                    f"  [{i+1}/{len(uniprot_ids)}] {uid}: transient AF2 fetch failure ({exc}); "
                    "not cached, will retry next run"
                )
                transient += 1
                continue
            except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
                # Malformed/unexpected metadata body — most likely a truncated or
                # error-page response, so treat it as transient rather than caching
                # "no structure" for a protein AF2 may well have.
                print(
                    f"  [{i+1}/{len(uniprot_ids)}] {uid}: unusable AF2 metadata ({exc}); "
                    "not cached, will retry next run"
                )
                transient += 1
                continue

        # Tokenise with ESM3StructureTokenizer. The PDB on disk is known complete
        # (atomic write), so a parse failure here is a property of the structure,
        # not of the network — it is a definitive result and safe to cache.
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
                print(
                    f"  [{i+1}/{len(uniprot_ids)}] {uid}: no coordinates in AF2 model, seq-only"
                )
            else:
                print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: OK")
        except INFRASTRUCTURE_ERRORS:
            # Not a property of the structure — an out-of-memory, a missing
            # dependency or a disk error says nothing about whether AF2 modelled
            # this protein. Caching None here would permanently mark it seq-only
            # for a reason that will not reproduce, so let it propagate.
            raise
        except Exception as exc:
            # The PDB on disk is known complete (atomic write), so a parse/shape
            # failure is a property of the structure: a definitive result, cacheable.
            print(
                f"  [{i+1}/{len(uniprot_ids)}] {uid}: tokenisation failed ({exc}), seq-only"
            )
            cached[uid] = None

        if (i + 1) % 50 == 0:
            atomic_write_json(STRUCT_TOKENS, cached)
            n_fallback = sum(1 for value in cached.values() if value is None)
            print(
                f"  Checkpoint: {i+1}/{len(uniprot_ids)}, {n_fallback} seq-only so far, "
                f"{transient} transient failures pending retry"
            )

    atomic_write_json(STRUCT_TOKENS, cached)
    n_ok = sum(1 for value in cached.values() if value is not None)
    n_fallback = sum(1 for value in cached.values() if value is None)
    n_unresolved = len(uniprot_ids) - len(cached)
    print(
        f"\nStructure tokens cached: {n_ok}/{len(uniprot_ids)} OK, "
        f"{n_fallback} seq-only fallbacks, {n_unresolved} unresolved "
        f"({transient} transient failures this run — rerun phase 1 to retry)"
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


def _run_logreg_folds(
    X: np.ndarray,
    y: np.ndarray,
    splits: list,
    seed: int,
) -> dict | None:
    """Logistic-regression CV over `splits`, returning per-fold-averaged macro-F1.

    Kept separate from the shared run_logreg_cv because this arm is deliberately
    regularised at C=LOGREG_C rather than the shared probe's default.
    A fold is skipped only when its train split has fewer than MIN_TRAIN_CLASSES
    classes (a classifier needs two to fit) — the same condition run_mlp_probe_cv
    applies, so both arms of this experiment are averaged over the same fold set.
    Returns None when no fold was scorable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    fold_f1s = []
    for fold_i, (tr, te) in enumerate(splits):
        if len(set(y[tr].tolist())) < MIN_TRAIN_CLASSES:
            print(f"    [logreg] Fold {fold_i+1}: skipped (< {MIN_TRAIN_CLASSES} classes in train)")
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            C=LOGREG_C,
        )
        clf.fit(X_tr, y[tr])
        fold_f1s.append(
            float(f1_score(y[te], clf.predict(X_te), average="macro", zero_division=0))
        )
    if not fold_f1s:
        return None
    return {
        "macro_f1_mean": float(np.mean(fold_f1s)),
        "macro_f1_std": float(np.std(fold_f1s)),
        "n_folds": len(fold_f1s),
    }


def phase3_probes(
    seeds: list[int] = SEEDS,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> None:
    from esm2_mech.utils.splits import gene_split_cv, family_split_cv

    OUT.mkdir(parents=True, exist_ok=True)

    variants, genes, pfam_map = load_dataset()
    y_labels = np.array([v["mech3"] for v in variants])
    label_set = MECHANISM_CLASSES
    y_all = np.array([label_set.index(label) for label in y_labels])

    # Load valid indices saved by phase 2 for exact label alignment
    if EMB_VALID_IDX.exists():
        valid_idx = np.load(str(EMB_VALID_IDX))
        y = y_all[valid_idx]
        labels_valid = y_labels[valid_idx]
        genes_valid = genes[valid_idx]
        print(f"Valid indices loaded: {len(valid_idx)}/{len(y_all)} variants embedded")
    else:
        y = y_all
        labels_valid = y_labels
        genes_valid = genes
        valid_idx = None
        print("No valid index file found — assuming all variants embedded")

    conditions = {
        "seq": EMB_SEQ,
        "seq_struct": EMB_SEQ_STRUCT,
    }

    cond_arrays: dict[str, np.ndarray] = {}
    for cond, path in conditions.items():
        if not path.exists():
            print(f"  SKIP {cond}: {path} not found")
            continue
        arr = np.load(str(path))
        print(f"  {cond}: {arr.shape[0]} variants embedded (of {len(variants)} total)")
        cond_arrays[cond] = arr

    # The ESM-2 arm the scale claim is a difference against, run here on the same
    # variants, folds, seeds and probe as the ESM-3 arms so M1/M2/M3 can be tested as
    # paired differences rather than two independently-scored point estimates.
    if DATASET == "merged":
        cond_arrays[ESM2_COND] = esm2_matched_delta(len(y_all), valid_idx)
        print(f"  {ESM2_COND}: {cond_arrays[ESM2_COND].shape[0]} variants (matched ESM-2)")
    else:
        print(
            f"  SKIP {ESM2_COND}: --dataset {DATASET} is not row-aligned to the ESM-2 "
            "embeddings; M1/M2 have no paired arm on this dataset"
        )

    results = {}
    oof_by_arm: dict[tuple[str, str], dict] = {}

    for cond, delta in cond_arrays.items():
        if delta.shape[0] != len(y):
            raise RuntimeError(
                f"{cond}: delta rows {delta.shape[0]} != labels {len(y)} — valid index mismatch"
            )
        print(f"\n=== Condition: {cond}  shape={delta.shape} ===")
        y_cond = y
        labels_cond = labels_valid
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

                # The shared runner behind the ESM-2 family-split floor this arm is
                # compared against (M1/M2/M3). Using it — rather than a local copy —
                # is what keeps the fold-skip condition, the standardization and the
                # per-fold metric aggregation identical across the two arms.
                agg, oof = run_mlp_probe_cv(
                    delta,
                    labels_cond,
                    splits,
                    seed=seed,
                    genes=genes_cond,
                    label=f"{cond}_{cv_name}_seed{seed}",
                    return_oof=True,
                )
                if not agg:
                    continue
                if compute_ci and oof is not None:
                    seed_oof_list.append(oof)
                mlp_f1s.append(agg["macro_f1_mean"])
                mlp_gof.append(agg.get(f"auroc_{GOF}_mean", float("nan")))
                mlp_dn.append(agg.get(f"auroc_{DN}_mean", float("nan")))
                mlp_lof.append(agg.get(f"auroc_{LOF}_mean", float("nan")))

                # Logistic regression, over the same fold set.
                lr_agg = _run_logreg_folds(delta, y_cond, splits, seed)
                if lr_agg is not None:
                    lr_f1s.append(lr_agg["macro_f1_mean"])

            if not mlp_f1s:
                continue

            # NaN-safe across seeds: a class absent from a whole seed's test folds
            # leaves that seed's AUROC undefined, which must not poison the mean.
            f1_mean, f1_std, n_seeds_scored = mean_std_n(mlp_f1s)
            lr_mean, lr_std, _ = mean_std_n(lr_f1s)
            r = {
                "mlp_f1_mean": f1_mean,
                "mlp_f1_std": f1_std,
                "mlp_gof_auroc_mean": mean_std_n(mlp_gof)[0],
                "mlp_dn_auroc_mean": mean_std_n(mlp_dn)[0],
                "mlp_lof_auroc_mean": mean_std_n(mlp_lof)[0],
                "lr_f1_mean": lr_mean,
                "lr_f1_std": lr_std,
                "n_seeds": n_seeds_scored,
            }
            if compute_ci:
                # Each seed reshuffles the CV fold assignment, so its OOF cannot be
                # bootstrapped directly against another seed's — average_oof_over_seeds
                # collapses the per-seed OOF predictions to one proba-per-variant
                # first (matching classify_by_mechanism's cross-seed CI convention),
                # then the cluster bootstrap runs once over that combined OOF.
                combined_oof = average_oof_over_seeds(seed_oof_list)
                if combined_oof is not None:
                    oof_by_arm[(cond, cv_name)] = combined_oof
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

    # The matched ESM-2 arm scores the same variants the ESM-3 arms do; the gate's
    # floor is the full merged set. They are two populations, so a divergence beyond
    # the matched arm's own seed spread means the gate and its CI are not describing
    # the same comparison, and the summary records the flag rather than reconciling it.
    matched_f1 = get_f1(ESM2_COND, "family_split")
    matched_std = (
        results.get(ESM2_COND, {}).get("family_split", {}).get("mlp_f1_std", float("nan"))
    )
    baseline_divergence = None
    if not np.isnan(matched_f1):
        baseline_divergence = float(matched_f1 - esm2_floor)
        print(
            f"  ESM-2 matched-subset floor = {matched_f1:.3f}±{matched_std:.3f}  "
            f"(gate uses the full-set floor {esm2_floor:.3f}; "
            f"difference {baseline_divergence:+.3f})"
        )
        if not np.isnan(matched_std) and abs(baseline_divergence) > matched_std:
            print(
                "  WARNING: the matched-subset ESM-2 floor differs from the "
                "pre-registered full-set floor by more than one seed of spread. The "
                "M1/M2 thresholds are pinned to the full set while the paired CIs are "
                "computed on the subset."
            )

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

    # ── paired differences (C5's instrument, and the same for M1/M3) ────────
    diffs: dict[str, dict] = {}
    if compute_ci:
        print("\n=== PAIRED DIFFERENCES (family-split, family-cluster bootstrap) ===")
        contrasts = [
            ("M1", "seq_struct", ESM2_COND, M1_MARGIN),
            ("M2", "seq", ESM2_COND, M1_MARGIN),
            ("M3", "seq_struct", "seq", M3_THRESHOLD),
        ]
        for gate, arm_a, arm_b, threshold in contrasts:
            label = f"{gate}: {arm_a} − {arm_b}"
            diff = paired_family_split_diff(
                oof_by_arm.get((arm_a, "family_split")),
                oof_by_arm.get((arm_b, "family_split")),
                pfam_map,
                n_boot,
                label,
            )
            if diff is None:
                continue
            diffs[gate] = diff
            if diff.get("ci_low") is None:
                print(f"  {label}: diff={diff['point_diff']:+.4f}  CI suppressed")
            else:
                print(
                    f"  {label}: diff={diff['point_diff']:+.4f}  "
                    f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
                    f"(threshold {threshold:.3f}, {diff['n_clusters']} families)"
                )
    else:
        print("\n  Paired differences skipped (--no_ci)")

    verdicts = {
        "M1": adjudicate_diff(m1, diffs.get("M1"), M1_MARGIN),
        "M2": adjudicate_diff(m2, diffs.get("M2"), M1_MARGIN),
        "M3": adjudicate_diff(m3, diffs.get("M3"), M3_THRESHOLD),
    }
    for gate, verdict in verdicts.items():
        print(f"  {gate} verdict: {verdict}")

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
        "esm2_matched_subset_family_split_f1": (
            float(matched_f1) if not np.isnan(matched_f1) else None
        ),
        "esm2_matched_minus_full_set_floor": baseline_divergence,
        "m1_threshold": m1_threshold,
        "conditions": "seq, seq_struct (function tokens not implemented)",
        "structure_coverage": struct_meta,
        "results": results,
        "decision_rules": {
            "M1": {
                "criterion": f"seq_struct family-split F1 > {m1_threshold:.3f}",
                "value": ss_f1,
                "passed": m1,
                "paired_diff": diffs.get("M1"),
                "paired_diff_arms": f"seq_struct − {ESM2_COND}",
                "paired_threshold": M1_MARGIN,
                "verdict": verdicts["M1"],
            },
            "M2": {
                "criterion": f"seq family-split F1 > {m1_threshold:.3f}",
                "value": seq_f1,
                "passed": m2,
                "paired_diff": diffs.get("M2"),
                "paired_diff_arms": f"seq − {ESM2_COND}",
                "paired_threshold": M1_MARGIN,
                "verdict": verdicts["M2"],
            },
            "M3": {
                "criterion": f"seq_struct − seq > {M3_THRESHOLD:.3f}",
                "value": float(gap) if not np.isnan(gap) else None,
                "passed": m3,
                "paired_diff": diffs.get("M3"),
                "paired_diff_arms": "seq_struct − seq",
                "paired_threshold": M3_THRESHOLD,
                "verdict": verdicts["M3"],
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
