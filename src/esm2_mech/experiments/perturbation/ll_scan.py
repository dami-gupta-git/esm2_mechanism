"""
Log-likelihood scan for mechanism prediction (result_22).

For each gene, masks each of the 100 evenly-spaced probe positions in turn and
reads log P(aa | context) for all 20 amino acids directly from ESM-2's output
logits. This is the masked language model's native scoring — more principled
than the embedding L2 distance used in result_20.

Computes 5 pre-registered scalar features per gene from ΔLL scores:
  1. ll_wt_mean        — mean log P(wt_aa) across positions (conservation)
  2. ll_delta_mean     — mean ΔLL = mean(log P(wt) - log P(probe)) across Ala/Asp/Trp
  3. ll_delta_cv       — CV of ΔLL across positions (hotspot concentration)
  4. ll_hotspot_frac   — fraction of positions with ΔLL > mean + 1σ
  5. ll_top_entropy    — mean entropy of the 20-AA distribution at top-10 ΔLL positions

Speed: ~198k forward passes (1 per position, all 20 AAs scored simultaneously),
~3x fewer than the embedding scan (result_20).

Usage:
  # Phase 2 (GPU required):
  python3 scripts/ll_scan.py --run_phase 2

  # Phase 3 only (CPU, after scores cached):
  python3 scripts/ll_scan.py --run_phase 3

  # All phases:
  python3 scripts/ll_scan.py
"""

import argparse, functools, json, os, sys, numpy as np
from collections import defaultdict
from pathlib import Path

from esm2_mech.utils.paths import EMB_MUT_MEAN, EMB_WT_MEAN, LL_CKPT_JSON, RESULTS_DIR as _RESULTS_DIR, SCAN_PROBE_CACHE_JSON, SEQUENCES_EXTENDED_JSON, SEQUENCES_JSON, VALID_VARIANTS_JSON

print = functools.partial(print, flush=True)
from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M
from esm2_mech.utils.sequences import window_sequence

OUT = _RESULTS_DIR / "ll_scan"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_AAS = ["A", "D", "W"]  # Ala, Asp, Trp — same as result_20
CHECKPOINT_EVERY = 50  # genes between saves
MIN_POSITIONS = 3


# ── Phase 1: load probe list ──────────────────────────────────────────────────


def load_probe_list():
    """Reuse the same probe positions from result_20."""
    probe_cache = SCAN_PROBE_CACHE_JSON
    with open(probe_cache) as f:
        d = json.load(f)
    probes = d["probes"]
    covered_genes = d["covered_genes"]

    # Build gene → unique positions mapping (deduplicate across Ala/Asp/Trp probes)
    gene_positions = defaultdict(dict)  # gene → {aa_pos: (wt_aa, seq, uniprot_id)}
    for p in probes:
        gene = p["gene"]
        pos = p["aa_pos"]
        if pos not in gene_positions[gene]:
            gene_positions[gene][pos] = {
                "wt_aa": p["aa_wt"],
                "uniprot_id": p["uniprot_id"],
                "seq_len": p["seq_len"],
            }

    print(
        f"Genes: {len(covered_genes)}  Total unique positions: {sum(len(v) for v in gene_positions.values())}"
    )
    return covered_genes, gene_positions


def load_sequences():
    with open(SEQUENCES_JSON) as f:
        seqs = json.load(f)
    if SEQUENCES_EXTENDED_JSON.exists():
        with open(SEQUENCES_EXTENDED_JSON) as f:
            seqs.update(json.load(f))
    return seqs


# ── Phase 2: GPU log-likelihood extraction ────────────────────────────────────


def extract_ll_scores(covered_genes, gene_positions, seqs, batch_size=32):
    """
    For each gene, mask each probe position in turn and record:
      - log P(wt_aa | context)
      - log P(probe_aa | context) for each of Ala, Asp, Trp

    Returns dict: gene → list of dicts with keys:
      aa_pos, wt_aa, ll_wt, ll_ala, ll_asp, ll_trp
    """
    import torch
    import esm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Run on RunPod.")

    ckpt_path = LL_CKPT_JSON
    out_path = DATA / "ll_scores.json"

    if out_path.exists():
        print(f"Cached LL scores found: {out_path}")
        with open(out_path) as f:
            return json.load(f)

    # Resume from checkpoint
    done_genes = set()
    all_scores = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        done_genes = set(ckpt["done_genes"])
        all_scores = ckpt["scores"]
        print(
            f"Resuming from checkpoint: {len(done_genes)}/{len(covered_genes)} genes done"
        )

    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL_650M)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    # Map AA char → token index
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
    probe_aas = ["A", "D", "W"]

    remaining = [g for g in covered_genes if g not in done_genes]
    print(f"Extracting LL scores for {len(remaining)} genes on {device}...")

    for gene_num, gene in enumerate(remaining):
        positions = gene_positions[gene]
        uniprot_id = next(iter(positions.values()))["uniprot_id"]
        if uniprot_id not in seqs:
            continue
        seq = seqs[uniprot_id]

        gene_scores = []
        pos_list = sorted(positions.keys())

        # Process positions in batches — each position needs one masked sequence
        for batch_start in range(0, len(pos_list), batch_size):
            batch_pos = pos_list[batch_start : batch_start + batch_size]
            batch_data = []

            for pos in batch_pos:
                wt_win, new_pos = window_sequence(seq, pos)
                # Build masked sequence: replace new_pos (1-indexed) with mask token
                masked = list(wt_win)
                masked[new_pos - 1] = "<mask>"
                masked_seq = "".join(masked)
                batch_data.append((f"{gene}_{pos}", masked_seq))

            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.inference_mode():
                out = model(tokens)
            logits = out["logits"].cpu().float()  # (B, L+2, vocab)

            for i, pos in enumerate(batch_pos):
                wt_win, new_pos = window_sequence(seq, pos)
                wt_aa = positions[pos]["wt_aa"]

                # Token index for the masked position (BOS at 0, so token i = seq position i)
                tok_idx = new_pos  # new_pos is 1-indexed, matches token position with BOS offset

                log_probs = torch.log_softmax(logits[i, tok_idx], dim=-1).numpy()

                ll_wt = (
                    float(log_probs[aa_to_idx[wt_aa]])
                    if wt_aa in aa_to_idx
                    else float("nan")
                )
                ll_ala = float(log_probs[aa_to_idx["A"]])
                ll_asp = float(log_probs[aa_to_idx["D"]])
                ll_trp = float(log_probs[aa_to_idx["W"]])

                # Full 20-AA distribution for entropy computation
                aa_order = "ACDEFGHIKLMNPQRSTVWY"
                full_probs = [
                    float(np.exp(log_probs[aa_to_idx[aa]])) for aa in aa_order
                ]

                gene_scores.append(
                    {
                        "aa_pos": pos,
                        "wt_aa": wt_aa,
                        "ll_wt": ll_wt,
                        "ll_ala": ll_ala,
                        "ll_asp": ll_asp,
                        "ll_trp": ll_trp,
                        "full_probs": full_probs,
                    }
                )

        all_scores[gene] = gene_scores
        done_genes.add(gene)

        if (gene_num + 1) % CHECKPOINT_EVERY == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"done_genes": list(done_genes), "scores": all_scores}, f)
            print(f"  Checkpoint: {len(done_genes)}/{len(covered_genes)} genes")

    # Save final output, remove checkpoint
    with open(out_path, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved LL scores: {out_path}")
    if ckpt_path.exists():
        ckpt_path.unlink()

    return all_scores


# ── Phase 3: feature computation ─────────────────────────────────────────────


def compute_ll_features(covered_genes, all_scores):
    """
    Build 5 pre-registered scalar features per gene from LL scores.
    Returns: gene_list (array), X (n_genes × 5), feature_names (list)
    """
    feature_names = [
        "ll_wt_mean",
        "ll_delta_mean",
        "ll_delta_cv",
        "ll_hotspot_frac",
        "ll_top_entropy",
    ]

    gene_list, X = [], []
    aa_order = "ACDEFGHIKLMNPQRSTVWY"

    for gene in covered_genes:
        scores = all_scores.get(gene, [])
        if len(scores) < MIN_POSITIONS:
            continue

        # Per-position ΔLL = mean(log P(wt) - log P(probe)) across Ala/Asp/Trp
        ll_wt_vals = np.array([s["ll_wt"] for s in scores])
        delta_vals = np.array(
            [
                s["ll_wt"] - np.mean([s["ll_ala"], s["ll_asp"], s["ll_trp"]])
                for s in scores
            ]
        )

        ll_wt_mean = float(np.nanmean(ll_wt_vals))

        delta_mean = float(np.nanmean(delta_vals))
        delta_std = float(np.nanstd(delta_vals))
        delta_cv = delta_std / (abs(delta_mean) + 1e-8)

        threshold = delta_mean + delta_std
        hotspot_frac = float(np.mean(delta_vals > threshold))

        # Entropy at top-10 ΔLL positions
        top10_idx = np.argsort(delta_vals)[-10:]
        entropies = []
        for idx in top10_idx:
            probs = np.array(scores[idx]["full_probs"])
            probs = probs / (probs.sum() + 1e-12)
            ent = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropies.append(ent)
        ll_top_entropy = float(np.mean(entropies))

        gene_list.append(gene)
        X.append([ll_wt_mean, delta_mean, delta_cv, hotspot_frac, ll_top_entropy])

    gene_list = np.array(gene_list)
    X = np.array(X, dtype=np.float32)
    print(f"Gene features built: {len(gene_list)} genes × {X.shape[1]} features")
    print(f"Feature names: {feature_names}")
    return gene_list, X, feature_names


def save_features(gene_list, X, feature_names):
    np.save(DATA / "ll_features.npy", X)
    meta = {"genes": gene_list.tolist(), "feature_names": feature_names}
    with open(DATA / "ll_features_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved ll_features.npy ({X.shape})")


# ── Phase 4: probe runs ───────────────────────────────────────────────────────


def run_probe_analysis():
    """
    Logistic regression probe: ll-only, ll+delta, ll+scan, ll+scan+delta.
    Decision rules (pre-registered, thresholds from result_20):
      G1: ll-only family-split F1 > 0.282   (scan-only 0.272 + 0.01)
      G2: ll+delta family-split F1 > 0.385  (scan+delta 0.375 + 0.01)
      G3: ll+scan family-split F1 > max(ll-only, scan-only) + 0.02
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score, roc_auc_score
    from collections import Counter

    from esm2_mech.utils.splits import gene_split_cv, family_split_cv

    DECISION_RULES = {
        "G1": ("ll_only_family_split", "macro_f1_mean", 0.282),
        "G2": ("ll_delta_family_split", "macro_f1_mean", 0.385),
    }

    print("=== Loading data ===")
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = (
                "LOF"
                if v.get("mechanism") in ("HI", "AR")
                else v.get("mechanism", "LOF")
            )

    gene_labels = defaultdict(list)
    for v in variants:
        gene_labels[v["gene"].upper()].append(v["label_3class"])
    gene_list_all = np.array(sorted(gene_labels.keys()))
    labels_all = np.array(
        [Counter(gene_labels[g]).most_common(1)[0][0] for g in gene_list_all]
    )

    # Load LL features
    ll_X = np.load(DATA / "ll_features.npy")
    with open(DATA / "ll_features_meta.json") as f:
        ll_meta = json.load(f)
    ll_genes = np.array(ll_meta["genes"])
    ll_idx = {g: i for i, g in enumerate(ll_genes)}
    ll_mask = np.array([g in ll_idx for g in gene_list_all])
    gene_list = gene_list_all[ll_mask]
    labels = labels_all[ll_mask]
    ll_X_aligned = np.array([ll_X[ll_idx[g]] for g in gene_list])
    print(f"Genes with LL features: {len(gene_list)}  Classes: {dict(Counter(labels))}")

    # Load mean-pooled delta — use the same variants file as the label map above
    with open(VALID_VARIANTS_JSON) as f:
        mvv = json.load(f)
    wt_emb = np.load(EMB_WT_MEAN)
    mut_emb = np.load(EMB_MUT_MEAN)
    delta = mut_emb - wt_emb
    gene_delta = defaultdict(list)
    for i, v in enumerate(mvv):
        gene_delta[v["gene"].upper()].append(delta[i])
    delta_X = np.array(
        [np.mean(gene_delta[g], axis=0) for g in gene_list], dtype=np.float32
    )

    # Load scan features (result_20)
    scan_path = DATA / "scan_features.npy"
    if scan_path.exists():
        scan_X_all = np.load(scan_path)
        with open(DATA / "scan_features_meta.json") as f:
            scan_meta = json.load(f)
        scan_idx = {g: i for i, g in enumerate(scan_meta["genes"])}
        scan_X = np.array(
            [scan_X_all[scan_idx[g]] for g in gene_list if g in scan_idx],
            dtype=np.float32,
        )
        scan_mask = np.array([g in scan_idx for g in gene_list])
    else:
        scan_X = None
        scan_mask = None
        print("  scan_features.npy not found — skipping ll+scan combo")

    # Feature combos
    combos = {"ll_only": ll_X_aligned, "ll_delta": np.hstack([ll_X_aligned, delta_X])}
    if scan_X is not None and scan_mask.all():
        combos["ll_scan"] = np.hstack([ll_X_aligned, scan_X])
        combos["ll_scan_delta"] = np.hstack([ll_X_aligned, scan_X, delta_X])

    with open(DATA / "pfam_families.json") as f:
        pfam_map = json.load(f)

    def run_probe(X, labels, splits, seed):
        le = LabelEncoder()
        le.fit(["GOF", "DN", "LOF"])
        y = le.transform(labels)
        classes = le.classes_
        fold_results = []
        for tr, te in splits:
            if len(set(y[tr])) < len(classes):
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
            )
            clf.fit(Xtr, y[tr])
            raw_proba = clf.predict_proba(Xte)
            proba = np.zeros((len(Xte), len(classes)), dtype=np.float32)
            for ci, c in enumerate(clf.classes_):
                proba[:, c] = raw_proba[:, ci]
            pred = proba.argmax(axis=1)
            fm = {
                "macro_f1": float(
                    f1_score(y[te], pred, average="macro", zero_division=0)
                )
            }
            for i, cls in enumerate(classes):
                yb = (y[te] == i).astype(int)
                if yb.sum() > 0 and (1 - yb).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(yb, proba[:, i]))
            fold_results.append(fm)
        if not fold_results:
            return {}
        agg = {}
        for key in set().union(*[set(f) for f in fold_results]):
            vals = [f[key] for f in fold_results if key in f]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        return agg

    all_results = {}
    for seed in range(5):
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list, seed=seed)
        fs = family_split_cv(gene_list, pfam_map, seed=seed)
        seed_res = {}
        for combo_name, X in combos.items():
            for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
                key = f"{combo_name}_{split_name}"
                r = run_probe(X, labels, splits, seed=seed)
                seed_res[key] = r
                f1 = r.get("macro_f1_mean", float("nan"))
                gof = r.get("auroc_GOF_mean", float("nan"))
                print(f"  {key}: F1={f1:.3f}  GOF={gof:.3f}")
        all_results[seed] = seed_res

    print("\n=== 5-SEED SUMMARY ===")
    summary = {}
    for key in all_results[0].keys():
        f1_vals = [
            all_results[s][key].get("macro_f1_mean", float("nan")) for s in range(5)
        ]
        gof_vals = [
            all_results[s][key].get("auroc_GOF_mean", float("nan")) for s in range(5)
        ]
        summary[key] = {
            "macro_f1_mean": float(np.nanmean(f1_vals)),
            "macro_f1_std": float(np.nanstd(f1_vals)),
            "auroc_GOF_mean": float(np.nanmean(gof_vals)),
            "auroc_GOF_std": float(np.nanstd(gof_vals)),
        }
        print(
            f"  {key}: F1={summary[key]['macro_f1_mean']:.3f}±{summary[key]['macro_f1_std']:.3f}"
            f"  GOF={summary[key]['auroc_GOF_mean']:.3f}±{summary[key]['auroc_GOF_std']:.3f}"
        )

    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        val = summary.get(key, {}).get(metric, float("nan"))
        passed = val > threshold
        gate_results[gate] = {"value": val, "threshold": threshold, "passed": passed}
        status = "PASS ✓" if passed else "FAIL ✗"
        print(
            f"  {gate}: {key} {metric} = {val:.3f} (threshold {threshold:.3f}) → {status}"
        )

    # G3: complementarity
    ll_only_f1 = summary.get("ll_only_family_split", {}).get(
        "macro_f1_mean", float("nan")
    )
    scan_only_f1 = 0.272  # from result_20
    g3_threshold = max(ll_only_f1, scan_only_f1) + 0.02
    ll_scan_f1 = summary.get("ll_scan_family_split", {}).get(
        "macro_f1_mean", float("nan")
    )
    g3_passed = ll_scan_f1 > g3_threshold
    gate_results["G3"] = {
        "value": ll_scan_f1,
        "threshold": g3_threshold,
        "passed": g3_passed,
    }
    status = "PASS ✓" if g3_passed else "FAIL ✗"
    print(
        f"  G3: ll_scan_family_split macro_f1_mean = {ll_scan_f1:.3f} (threshold {g3_threshold:.3f}) → {status}"
    )

    out = {
        "summary": summary,
        "gate_results": gate_results,
        "per_seed": {str(s): all_results[s] for s in range(5)},
        "n_genes": int(len(gene_list)),
        "feature_combos": list(combos.keys()),
    }
    out_path = OUT / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_phase",
        default="23",
        help="Phases to run: '2', '3', '23' (default: 23). Phase 2=GPU extraction, 3=features+probe",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    phases = set(args.run_phase)

    covered_genes, gene_positions = load_probe_list()
    seqs = load_sequences()

    if "2" in phases:
        print("\n=== Phase 2: log-likelihood extraction ===")
        all_scores = extract_ll_scores(
            covered_genes, gene_positions, seqs, batch_size=args.batch_size
        )
    elif "3" in phases:
        out_path = DATA / "ll_scores.json"
        if not out_path.exists():
            print("ERROR: ll_scores.json not found. Run phase 2 first (requires GPU).")
            sys.exit(1)
        with open(out_path) as f:
            all_scores = json.load(f)
        print(f"Loaded LL scores for {len(all_scores)} genes")

    if "3" in phases:
        print("\n=== Phase 3: feature computation ===")
        gene_list, X, feature_names = compute_ll_features(covered_genes, all_scores)
        save_features(gene_list, X, feature_names)

        print("\n=== Phase 4: probe runs ===")
        run_probe_analysis()


if __name__ == "__main__":
    main()
