"""
ESM-3 mechanism family-split experiment (plan_esm3_mechanism.md).

Tests whether ESM-3's structure tokens rescue the mechanism null from ESM-2.

Three embedding conditions:
  seq       — sequence tokens only (fair scale comparison to ESM-2 650M)
  seq_struct — sequence + AlphaFold2 structure tokens
  full      — sequence + structure + function tokens (where available)

For each condition: delta = mean_pool(ESM-3(mut)) - mean_pool(ESM-3(wt))
Same probe/CV/seeds as result_7: MLP + logistic, 5-fold gene-split +
family-split, seeds 0-4 on Gerasimavicius (948 genes, 3-class GOF/LOF/DN).

Decision rules (pre-registered in plan_esm3_mechanism.md):
  M1: ESM-3 full family-split macro-F1 > 0.414  (ESM-2 0.364 + 0.05)
  M2: ESM-3 seq-only family-split F1  > 0.414  (scale alone rescues)
  M3: ESM-3 full - ESM-3 seq-only    > 0.03   (structure adds signal)

Phases:
  --phase 1   CPU: download AF2 structures, tokenise, cache structure tokens
  --phase 2   GPU: extract ESM-3 embeddings for all three conditions
  --phase 3   CPU: run probes, evaluate decision rules, write results

Usage:
  python3 scripts/esm3_mechanism.py --phase 1        # local, CPU
  python3 scripts/esm3_mechanism.py --phase 2        # RunPod H100
  python3 scripts/esm3_mechanism.py --phase 3        # local, CPU
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

SCRIPTS = Path(__file__).resolve().parent
ROOT    = SCRIPTS.parent
DATA    = ROOT / "data"
EMB     = DATA / "embeddings"
AF2_DIR = DATA / "cache" / "af2_structures"
OUT     = ROOT / "results" / "esm3_mechanism"

GERAS_VARIANTS = DATA / "gerasimavicius_variants.json"
SEQUENCES_JSON = DATA / "sequences.json"
PFAM_JSON      = DATA / "pfam_families.json"

# ESM-3 embedding cache files (phase 2 output)
EMB_SEQ        = EMB / "esm3_geras_seq_mean.npy"
EMB_SEQ_STRUCT = EMB / "esm3_geras_seq_struct_mean.npy"
EMB_FULL       = EMB / "esm3_geras_full_mean.npy"

# AF2 structure token cache (phase 1 output)
STRUCT_TOKENS  = DATA / "cache" / "esm3_struct_tokens.json"

ESM3_MODEL  = "esm3-sm-open-v1"   # 1.4B, open weights
AF2_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

# Match result_7 exactly
N_FOLDS = 5
SEEDS   = [0, 1, 2, 3, 4]

# Decision rule thresholds (pre-registered)
M1_THRESHOLD = 0.414   # ESM-2 family-split 0.364 + 0.05
M3_THRESHOLD = 0.03    # full - seq-only gap


# ── helpers ───────────────────────────────────────────────────────────────────

def window_sequence(seq: str, pos_1indexed: int, window: int = 1022) -> tuple[str, int]:
    """Centre a window on pos; clamp to sequence ends. Returns (windowed, new_pos_1indexed)."""
    L = len(seq)
    if L <= window:
        return seq, pos_1indexed
    half = window // 2
    start = max(0, pos_1indexed - 1 - half)
    end   = start + window
    if end > L:
        end   = L
        start = max(0, end - window)
    return seq[start:end], pos_1indexed - start


def load_geras() -> tuple[list[dict], np.ndarray, dict]:
    """Load Gerasimavicius variants, collapse HI+AR → LOF, attach wt_seq, return (variants, genes, pfam_map)."""
    variants = json.loads(GERAS_VARIANTS.read_text())
    pfam_map = json.loads(PFAM_JSON.read_text()) if PFAM_JSON.exists() else {}
    sequences = json.loads(SEQUENCES_JSON.read_text()) if SEQUENCES_JSON.exists() else {}
    mech_map = {"GOF": "GOF", "DN": "DN", "HI": "LOF", "AR": "LOF", "LOF": "LOF"}
    kept = []
    skipped = 0
    for v in variants:
        if v.get("mechanism") not in mech_map:
            continue
        uid = v.get("uniprot_id", "")
        seq = sequences.get(uid)
        if not seq:
            skipped += 1
            continue
        v = dict(v)
        v["mech3"]  = mech_map[v["mechanism"]]
        v["wt_seq"] = seq
        kept.append(v)
    if skipped:
        print(f"  load_geras: {skipped} variants skipped (no sequence in cache)")
    genes = np.array([v["gene"] for v in kept])
    return kept, genes, pfam_map


# ── Phase 1: download AF2 structures and tokenise ────────────────────────────

def phase1_structure_tokens() -> None:
    """
    For each unique UniProt ID in Gerasimavicius, fetch AF2 structure from EBI,
    tokenise with ESM3StructureTokenizer, cache tokens.
    Genes without AF2 structures fall back to seq-only in phase 2.
    """
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.protein_chain import ProteinChain
    except ImportError:
        print("ERROR: esm package not found. Install: pip install esm")
        sys.exit(1)

    AF2_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    variants, _, _ = load_geras()
    uniprot_ids = sorted({v["uniprot_id"] for v in variants if v.get("uniprot_id")})
    print(f"Unique UniProt IDs: {len(uniprot_ids)}")

    if STRUCT_TOKENS.exists():
        cached = json.loads(STRUCT_TOKENS.read_text())
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
                req2 = urllib.request.Request(pdb_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req2, timeout=60) as r:
                    pdb_path.write_bytes(r.read())
            except Exception as e:
                print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: AF2 fetch failed ({e}), fallback to seq-only")
                cached[uid] = None
                fallback += 1
                continue

        # Tokenise with ESM3StructureTokenizer
        try:
            chain = ProteinChain.from_pdb(str(pdb_path))
            protein = ESMProtein.from_protein_chain(chain)
            # Extract structure token ids as a list (one per residue)
            cached[uid] = protein.coordinates.tolist() if protein.coordinates is not None else None
            if cached[uid] is None:
                fallback += 1
        except Exception as e:
            print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: tokenisation failed ({e}), fallback")
            cached[uid] = None
            fallback += 1

        if (i + 1) % 50 == 0:
            STRUCT_TOKENS.write_text(json.dumps(cached))
            print(f"  Checkpoint: {i+1}/{len(uniprot_ids)}, {fallback} fallbacks so far")

        print(f"  [{i+1}/{len(uniprot_ids)}] {uid}: OK")

    STRUCT_TOKENS.write_text(json.dumps(cached))
    n_ok = sum(1 for v in cached.values() if v is not None)
    print(f"\nStructure tokens cached: {n_ok}/{len(uniprot_ids)} OK, {fallback} seq-only fallbacks")
    print(f"Saved → {STRUCT_TOKENS}")


# ── Phase 2: GPU embedding extraction ────────────────────────────────────────

def phase2_extract_embeddings(batch_size: int = 4) -> None:
    """
    For each variant (wt and mut sequence) extract ESM-3 mean-pooled embeddings
    under three conditions: seq-only, seq+struct, full.
    Saves delta arrays (mut - wt) to EMB_SEQ, EMB_SEQ_STRUCT, EMB_FULL.
    """
    try:
        import torch
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, GenerationConfig
        from esm.pretrained import ESM3_sm_open_v0
    except ImportError:
        print("ERROR: esm package not found. Install: pip install esm")
        sys.exit(1)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase 2 requires a GPU.")

    print(f"Loading ESM-3 ({ESM3_MODEL}) on {device}...")
    model = ESM3_sm_open_v0(device=device)
    model.eval()

    variants, _, _ = load_geras()
    n = len(variants)
    print(f"Variants to embed: {n}")

    # Load structure token cache (uid -> coordinates list or None)
    struct_cache: dict = {}
    if STRUCT_TOKENS.exists():
        struct_cache = json.loads(STRUCT_TOKENS.read_text())
        n_struct = sum(1 for v in struct_cache.values() if v is not None)
        print(f"Structure tokens loaded: {n_struct}/{len(struct_cache)} have structures")

    # Check which conditions are already done
    conditions = {
        "seq":        EMB_SEQ,
        "seq_struct": EMB_SEQ_STRUCT,
        "full":       EMB_FULL,
    }
    for cond, path in conditions.items():
        if path.exists():
            arr = np.load(str(path))
            print(f"  {cond}: cached ({arr.shape})")

    remaining_conds = [c for c, path in conditions.items() if not path.exists()]
    if not remaining_conds:
        print("All conditions already cached.")
        return

    EMB.mkdir(parents=True, exist_ok=True)

    wt_embs  = {c: [] for c in remaining_conds}
    mut_embs = {c: [] for c in remaining_conds}

    def get_struct_tokens(uid: str | None, seq_len: int) -> "torch.Tensor | None":
        """Return structure tokens tensor (L+2,) for uid, or None if unavailable."""
        if uid is None or struct_cache.get(uid) is None:
            return None
        try:
            coords = torch.tensor(struct_cache[uid], dtype=torch.float32)
            # coords shape: (L, 37, 3) from AF2 PDB via ProteinChain
            # Trim/pad to match seq_len
            L = coords.shape[0]
            if L != seq_len:
                return None
            # Tokenise via the structure encoder
            protein = ESMProtein(sequence="A" * seq_len, coordinates=coords)
            tensor = model.encode(protein)
            return tensor.structure  # (L+2,) or None
        except Exception:
            return None

    def embed_sequence(seq: str, struct_toks: "torch.Tensor | None",
                       condition: str) -> np.ndarray:
        """Run ESM-3 forward pass; return mean-pooled (D,) embedding (excluding BOS/EOS)."""
        protein = ESMProtein(sequence=seq)
        tensor  = model.encode(protein)
        seq_tok = tensor.sequence.unsqueeze(0).to(device)   # (1, L+2)

        use_struct = condition in ("seq_struct", "full") and struct_toks is not None
        s_tok = struct_toks.unsqueeze(0).to(device) if use_struct else None

        with torch.inference_mode():
            out = model(sequence_tokens=seq_tok, structure_tokens=s_tok)

        # out.embeddings: (1, L+2, D) — exclude BOS (0) and EOS (-1)
        emb = out.embeddings[0, 1:-1].mean(dim=0).cpu().float().numpy()
        return emb

    for i, v in enumerate(variants):
        uid    = v.get("uniprot_id")
        wt_seq = v["wt_seq"]
        pos    = v["aa_pos"]

        wt_win, new_pos = window_sequence(wt_seq, pos)
        mut_win = list(wt_win)
        mut_win[new_pos - 1] = v["aa_mut"]
        mut_win = "".join(mut_win)

        # Get structure tokens once per variant (same for wt and mut — same position context)
        struct_toks = get_struct_tokens(uid, len(wt_win))

        for cond in remaining_conds:
            wt_e  = embed_sequence(wt_win,  struct_toks, cond)
            mut_e = embed_sequence(mut_win, struct_toks, cond)
            wt_embs[cond].append(wt_e)
            mut_embs[cond].append(mut_e)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] variants embedded")
            for cond in remaining_conds:
                np.save(str(conditions[cond]).replace(".npy", "_ckpt_wt.npy"),
                        np.array(wt_embs[cond]))
                np.save(str(conditions[cond]).replace(".npy", "_ckpt_mut.npy"),
                        np.array(mut_embs[cond]))

    for cond in remaining_conds:
        wt_arr  = np.array(wt_embs[cond])
        mut_arr = np.array(mut_embs[cond])
        delta   = mut_arr - wt_arr
        np.save(str(conditions[cond]), delta)
        print(f"  {cond}: delta saved → {conditions[cond]}  shape={delta.shape}")
        for suffix in ("_ckpt_wt.npy", "_ckpt_mut.npy"):
            p = Path(str(conditions[cond]).replace(".npy", suffix))
            if p.exists():
                p.unlink()

    print("Phase 2 complete.")


# ── Phase 3: probes and decision rules ───────────────────────────────────────

def phase3_probes() -> None:
    sys.path.insert(0, str(SCRIPTS))
    from utils_probes import gene_split_cv, family_split_cv
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)

    variants, genes, pfam_map = load_geras()
    y_labels = np.array([v["mech3"] for v in variants])
    label_set = ["GOF", "DN", "LOF"]
    y = np.array([label_set.index(l) for l in y_labels])

    conditions = {
        "seq":        EMB_SEQ,
        "seq_struct": EMB_SEQ_STRUCT,
        "full":       EMB_FULL,
    }

    results = {}

    for cond, path in conditions.items():
        if not path.exists():
            print(f"  SKIP {cond}: {path} not found")
            continue

        delta = np.load(str(path))
        print(f"\n=== Condition: {cond}  shape={delta.shape} ===")

        cond_results = {"gene_split": {}, "family_split": {}}

        for cv_name, get_splits in [
            ("gene_split",   lambda seed: gene_split_cv(genes, N_FOLDS, seed)),
            ("family_split", lambda seed: family_split_cv(genes, pfam_map, N_FOLDS, seed)),
        ]:
            mlp_f1s, mlp_gof, mlp_dn = [], [], []
            lr_f1s = []

            for seed in SEEDS:
                splits = get_splits(seed)
                if not splits:
                    print(f"  {cv_name} seed={seed}: no valid splits, skip")
                    continue

                scaler = StandardScaler()

                # MLP
                mlp_preds, mlp_true = [], []
                mlp_probs_gof, mlp_probs_dn = [], []
                for tr, te in splits:
                    X_tr = scaler.fit_transform(delta[tr])
                    X_te = scaler.transform(delta[te])
                    clf = MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        max_iter=300,
                        random_state=seed,
                        class_weight={i: 1.0 / np.mean(y[tr] == i) for i in range(3)},
                    )
                    clf.fit(X_tr, y[tr])
                    pred  = clf.predict(X_te)
                    proba = clf.predict_proba(X_te)
                    mlp_preds.append(pred)
                    mlp_true.append(y[te])
                    mlp_probs_gof.append(proba[:, label_set.index("GOF")])
                    mlp_probs_dn.append(proba[:, label_set.index("DN")])

                all_pred = np.concatenate(mlp_preds)
                all_true = np.concatenate(mlp_true)
                all_gof  = np.concatenate(mlp_probs_gof)
                all_dn   = np.concatenate(mlp_probs_dn)

                f1 = f1_score(all_true, all_pred, average="macro")
                gof_auc = roc_auc_score((all_true == label_set.index("GOF")).astype(int), all_gof)
                dn_auc  = roc_auc_score((all_true == label_set.index("DN")).astype(int),  all_dn)
                mlp_f1s.append(f1)
                mlp_gof.append(gof_auc)
                mlp_dn.append(dn_auc)

                # Logistic regression
                lr_preds, lr_true = [], []
                for tr, te in splits:
                    X_tr = scaler.fit_transform(delta[tr])
                    X_te = scaler.transform(delta[te])
                    clf = LogisticRegression(
                        max_iter=1000, random_state=seed,
                        class_weight="balanced", C=0.1,
                    )
                    clf.fit(X_tr, y[tr])
                    lr_preds.append(clf.predict(X_te))
                    lr_true.append(y[te])
                lr_f1s.append(f1_score(np.concatenate(lr_true), np.concatenate(lr_preds), average="macro"))

            if not mlp_f1s:
                continue

            r = {
                "mlp_f1_mean":  float(np.mean(mlp_f1s)),
                "mlp_f1_std":   float(np.std(mlp_f1s)),
                "mlp_gof_auroc_mean": float(np.mean(mlp_gof)),
                "mlp_dn_auroc_mean":  float(np.mean(mlp_dn)),
                "lr_f1_mean":   float(np.mean(lr_f1s)),
                "lr_f1_std":    float(np.std(lr_f1s)),
                "n_seeds":      len(mlp_f1s),
            }
            cond_results[cv_name] = r
            print(f"  {cv_name}: MLP F1={r['mlp_f1_mean']:.3f}±{r['mlp_f1_std']:.3f}  "
                  f"GOF={r['mlp_gof_auroc_mean']:.3f}  DN={r['mlp_dn_auroc_mean']:.3f}  "
                  f"LR F1={r['lr_f1_mean']:.3f}")

        results[cond] = cond_results

    # ── decision rules ─────────────────────────────────────────────────────
    print("\n=== DECISION RULES ===")

    esm2_family_f1 = 0.364   # result_7 MLP 5-seed mean

    def get_f1(cond, cv):
        return results.get(cond, {}).get(cv, {}).get("mlp_f1_mean", float("nan"))

    full_f1 = get_f1("full", "family_split")
    seq_f1  = get_f1("seq",  "family_split")
    ss_f1   = get_f1("seq_struct", "family_split")

    m1 = full_f1 > M1_THRESHOLD if not np.isnan(full_f1) else None
    m2 = seq_f1  > M1_THRESHOLD if not np.isnan(seq_f1)  else None
    m3 = (full_f1 - seq_f1) > M3_THRESHOLD if not np.isnan(full_f1) and not np.isnan(seq_f1) else None

    print(f"  ESM-2 baseline (result_7): family-split MLP F1 = {esm2_family_f1:.3f}")
    print(f"  M1: ESM-3 full   family-split F1 > {M1_THRESHOLD:.3f} → {full_f1:.3f} → {'PASS ✓' if m1 else 'FAIL ✗' if m1 is not None else 'N/A'}")
    print(f"  M2: ESM-3 seq    family-split F1 > {M1_THRESHOLD:.3f} → {seq_f1:.3f}  → {'PASS ✓' if m2 else 'FAIL ✗' if m2 is not None else 'N/A'}")
    print(f"  M3: full - seq   > {M3_THRESHOLD:.3f}               → {full_f1 - seq_f1:.3f}  → {'PASS ✓' if m3 else 'FAIL ✗' if m3 is not None else 'N/A'}")

    if m1 is False:
        print("\n  Interpretation: NULL CONFIRMED — mechanism is not recoverable from ESM-3 regardless of modality.")
    elif m1 and m2 is False and m3:
        print("\n  Interpretation: STRUCTURE RESCUES MECHANISM — structure tokens are the operative ingredient.")
    elif m1 and m2:
        print("\n  Interpretation: SCALE SUFFICES — model scale (not structure tokens) accounts for the lift.")
    elif m1 and m3 is False:
        print("\n  Interpretation: ESM-3 better but structure tokens not the reason.")

    summary = {
        "esm2_baseline_family_split_f1": esm2_family_f1,
        "results": results,
        "decision_rules": {
            "M1": {"criterion": f"full family-split F1 > {M1_THRESHOLD}", "value": full_f1, "passed": m1},
            "M2": {"criterion": f"seq family-split F1 > {M1_THRESHOLD}",  "value": seq_f1,  "passed": m2},
            "M3": {"criterion": f"full - seq > {M3_THRESHOLD}",           "value": float(full_f1 - seq_f1) if not np.isnan(full_f1) and not np.isnan(seq_f1) else None, "passed": m3},
        },
        "model": ESM3_MODEL,
        "dataset": "gerasimavicius",
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote → {OUT}/summary.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["1", "2", "3"],
                    help="1=structure tokens (CPU), 2=embeddings (GPU), 3=probes (CPU)")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    if args.phase == "1":
        print("=== Phase 1: AF2 structure download + ESM-3 tokenisation ===")
        phase1_structure_tokens()
    elif args.phase == "2":
        print("=== Phase 2: ESM-3 embedding extraction (GPU) ===")
        phase2_extract_embeddings(batch_size=args.batch_size)
    elif args.phase == "3":
        print("=== Phase 3: probes + decision rules ===")
        phase3_probes()


if __name__ == "__main__":
    main()
