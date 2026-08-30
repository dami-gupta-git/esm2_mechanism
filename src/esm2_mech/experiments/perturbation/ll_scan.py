"""Log-likelihood scan for mechanism prediction (result_22).

Masks probe positions, reads ESM-2 logits for all 20 AAs, and computes 5 scalar features per gene.
"""

import argparse, functools, json, os, sys, numpy as np
from collections import defaultdict
from pathlib import Path

from esm2_mech.utils.paths import EMB_MUT_MEAN, EMB_WT_MEAN, LL_CKPT_JSON, RESULTS_DIR as _RESULTS_DIR, SCAN_FEATURES_META_JSON, SCAN_FEATURES_NPY, SCAN_PROBE_CACHE_JSON, SEQUENCES_EXTENDED_JSON, SEQUENCES_JSON, VALID_VARIANTS_JSON
from esm2_mech.utils.constants import AA_ORDER, MECHANISM_CLASSES, N_SEEDS
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.seed_aggregation import aggregate_result_contract, seed_result_contract, seed_count
from esm2_mech.experiments.perturbation.seed_summary import (
    aggregate_probe_results,
    read_probe_metric,
)
from esm2_mech.utils.embed import load_esm2_model, load_gene_delta, masked_aa_log_probs
from esm2_mech.utils.probes import run_logreg_cv
from esm2_mech.utils.classification import validate_complete_classification_splits

print = functools.partial(print, flush=True)
from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M
from esm2_mech.utils.sequences import window_sequence

OUT = _RESULTS_DIR / "ll_scan"
OUT.mkdir(parents=True, exist_ok=True)

PROBE_AAS = ["A", "D", "W"]  # Ala, Asp, Trp — same as result_20
CHECKPOINT_EVERY = 50  # genes between saves
MIN_POSITIONS = 3


def load_probe_list():
    """Reuse the same probe positions from result_20."""
    probe_cache = SCAN_PROBE_CACHE_JSON
    with open(probe_cache) as f:
        d = json.load(f)
    probes = d["probes"]
    covered_genes = d["covered_genes"]

    gene_positions = defaultdict(dict)
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


def extract_ll_scores(covered_genes, gene_positions, seqs, batch_size=32):
    """Mask each probe position and record log P for wt and probe AAs."""
    import torch

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

    model, alphabet = load_esm2_model(ESM2_MODEL_650M, device=device)

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

        items = []
        for pos in pos_list:
            wt_win, new_pos, _ = window_sequence(seq, pos)
            masked = list(wt_win)
            masked[new_pos - 1] = "<mask>"
            # tok_idx = new_pos: BOS occupies token 0, so token i = seq position i.
            items.append((pos, "".join(masked), new_pos))

        log_probs_by_pos = masked_aa_log_probs(
            model, alphabet, device, items, AA_ORDER, batch_size=batch_size
        )

        for pos in pos_list:
            wt_aa = positions[pos]["wt_aa"]
            log_probs = log_probs_by_pos[pos]

            ll_wt = (
                float(log_probs[AA_ORDER.index(wt_aa)])
                if wt_aa in AA_ORDER
                else float("nan")
            )
            ll_ala = float(log_probs[AA_ORDER.index("A")])
            ll_asp = float(log_probs[AA_ORDER.index("D")])
            ll_trp = float(log_probs[AA_ORDER.index("W")])
            full_probs = [float(np.exp(lp)) for lp in log_probs]

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

    with open(out_path, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved LL scores: {out_path}")
    if ckpt_path.exists():
        ckpt_path.unlink()

    return all_scores


def compute_ll_features(covered_genes, all_scores):
    """Build the 5 scalar features per gene from LL scores."""
    feature_names = [
        "ll_wt_mean",
        "ll_delta_mean",
        "ll_delta_cv",
        "ll_hotspot_frac",
        "ll_top_entropy",
    ]

    gene_list, X = [], []
    aa_order = AA_ORDER

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


def _show_metric(metric):
    if not metric.available:
        return f"unavailable ({metric.message})"
    if metric.spread is None:
        return f"{metric.value:.3f}"
    return f"{metric.value:.3f}±{metric.spread:.3f} seed SD"


def run_probe_analysis(n_seeds=N_SEEDS):
    """Logistic regression probe across ll-only, ll+delta, ll+scan, ll+scan+delta."""
    from collections import Counter

    from esm2_mech.utils.splits import gene_split_cv, family_split_cv

    DECISION_RULES = {
        "G1": ("ll_only_family_split", "macro_f1", 0.282),
        "G2": ("ll_delta_family_split", "macro_f1", 0.385),
    }
    requested_seeds = tuple(range(n_seeds))

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

    gene_delta = load_gene_delta(VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN)
    delta_X = np.array(
        [np.mean(gene_delta[g], axis=0) for g in gene_list], dtype=np.float32
    )

    scan_path = SCAN_FEATURES_NPY
    if scan_path.exists():
        scan_X_all = np.load(scan_path)
        with open(SCAN_FEATURES_META_JSON) as f:
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

    combos = {"ll_only": ll_X_aligned, "ll_delta": np.hstack([ll_X_aligned, delta_X])}
    if scan_X is not None and scan_mask.all():
        combos["ll_scan"] = np.hstack([ll_X_aligned, scan_X])
        combos["ll_scan_delta"] = np.hstack([ll_X_aligned, scan_X, delta_X])
    # Declared before any arm is skipped, so an arm whose features are absent is
    # reported unavailable rather than dropping out of the summary unnoticed.
    declared_arms = [
        f"{combo_name}_{split_name}"
        for combo_name in ("ll_only", "ll_delta", "ll_scan", "ll_scan_delta")
        for split_name in ("gene_split", "family_split")
    ]

    with open(DATA / "pfam_families.json") as f:
        pfam_map = json.load(f)

    def run_probe(X, labels, splits, groups, held_out_unit, seed):
        contract = validate_complete_classification_splits(
            splits, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=labels, classes=MECHANISM_CLASSES, groups=groups,
            held_out_unit=held_out_unit,
        )
        return run_logreg_cv(
            X, labels, splits, MECHANISM_CLASSES, contract, seed=seed
        )

    all_results = {}
    for seed in requested_seeds:
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list, seed=seed)
        fs = family_split_cv(gene_list, pfam_map, seed=seed)
        seed_res = {}
        for combo_name, X in combos.items():
            split_specs = [
                ("gene_split", gs, gene_list, "gene"),
                (
                    "family_split", fs,
                    np.array([pfam_map.get(gene) for gene in gene_list], dtype=object),
                    "family",
                ),
            ]
            for split_name, splits, groups, held_out_unit in split_specs:
                key = f"{combo_name}_{split_name}"
                r = run_probe(X, labels, splits, groups, held_out_unit, seed=seed)
                seed_res[key] = r
                f1 = r.get("macro_f1_mean")
                gof = r.get("auroc_GOF_mean")
                f1_text = "unavailable" if f1 is None else f"{f1:.3f}"
                gof_text = "unavailable" if gof is None else f"{gof:.3f}"
                print(f"  {key}: F1={f1_text}  GOF={gof_text}")
        all_results[seed] = {
            **seed_result_contract(seed),
            "results": seed_res,
        }

    print(f"\n=== {n_seeds}-SEED SUMMARY ===")
    summary = aggregate_probe_results(requested_seeds, all_results, declared_arms)
    for key in summary:
        f1 = read_probe_metric(summary, key, "macro_f1")
        gof = read_probe_metric(summary, key, "auroc_GOF")
        print(f"  {key}: F1={_show_metric(f1)}  GOF={_show_metric(gof)}")

    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        result = read_probe_metric(summary, key, metric)
        val = result.value
        passed = None if not result.available else val > threshold
        gate_results[gate] = {
            "metric": summary[key][metric],
            "value": val,
            "threshold": threshold,
            "passed": passed,
        }
        if passed is None:
            print(f"  {gate}: {key} {metric} = Unscorable")
            continue
        status = "PASS ✓" if passed else "FAIL ✗"
        print(
            f"  {gate}: {key} {metric} = {val:.3f} (threshold {threshold:.3f}) → {status}"
        )

    # G3: complementarity
    g3_threshold = None
    ll_scan_result = (
        read_probe_metric(summary, "ll_scan_family_split", "macro_f1")
        if "ll_scan_family_split" in summary
        else None
    )
    ll_scan_f1 = None if ll_scan_result is None else ll_scan_result.value
    g3_passed = None
    gate_results["G3"] = {
        "value": ll_scan_f1,
        "threshold": g3_threshold,
        "passed": g3_passed,
        "reason": "current run has no scan-only comparator arm",
    }
    if g3_passed is None:
        print("  G3: ll_scan_family_split macro_f1_mean = Unscorable")
    else:
        status = "PASS ✓" if g3_passed else "FAIL ✗"
        print(
            f"  G3: ll_scan_family_split macro_f1_mean = {ll_scan_f1:.3f} "
            f"(threshold {g3_threshold:.3f}) → {status}"
        )

    out = {
        **aggregate_result_contract(),
        "summary": summary,
        "gate_results": gate_results,
        "per_seed": {str(seed): all_results[seed] for seed in requested_seeds},
        "n_genes": int(len(gene_list)),
        "feature_combos": list(combos.keys()),
    }
    out_path = OUT / "probe_results.json"
    write_result_json(out_path, out, seeds=list(requested_seeds))
    print(f"\nResults → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_phase",
        default="23",
        help="Phases to run: '2', '3', '23' (default: 23). Phase 2=GPU extraction, 3=features+probe",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=seed_count, default=N_SEEDS)
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
        run_probe_analysis(n_seeds=args.seeds)


if __name__ == "__main__":
    main()
