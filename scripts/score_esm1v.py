"""
Score ClinVar pathogenic/benign variants with ESM-1v masked-marginal ΔLL.

For each variant:
  ΔLL = mean_over_checkpoints( log P(mut_aa | context) − log P(wt_aa | context) )

using ESM-1v checkpoints 1 and 2 (esm1v_t33_650M_UR90S_1/2).

Inputs:
  data/pathogenicity_valid_variants.json  (17,236 variants)
  data/merged_valid_variants.json         (gene -> uniprot_id mapping)
  data/sequences.json                     (uniprot_id -> sequence)

Output:
  data/esm1v_scores_full.json             (variant_key -> ΔLL, same format as
                                           alphamissense_scores_full.json)

Requires GPU. Checkpoints are saved every --checkpoint-every genes.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CHECKPOINTS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
]
CHECKPOINT_EVERY = 50   # genes between saves
WINDOW = 1022           # ESM-1v max tokens (1024 minus BOS/EOS)


def build_gene_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (gene->uniprot, uniprot->sequence)."""
    merged = json.load(open(DATA / "merged_valid_variants.json"))
    g2u: dict[str, str] = {}
    for r in merged:
        g2u.setdefault(r["gene"], r["uniprot_id"])
    seqs: dict[str, str] = json.load(open(DATA / "sequences.json"))
    return g2u, seqs


def window_sequence(seq: str, pos_1indexed: int) -> tuple[str, int]:
    """Centre a window of WINDOW residues on pos; return (windowed_seq, new_pos_1indexed)."""
    L = len(seq)
    if L <= WINDOW:
        return seq, pos_1indexed
    half = WINDOW // 2
    start = max(0, pos_1indexed - 1 - half)
    end = start + WINDOW
    if end > L:
        end = L
        start = max(0, end - WINDOW)
    return seq[start:end], pos_1indexed - start


def score_variants_single_model(
    model,
    alphabet,
    device: str,
    gene_variants: dict[str, list[dict]],
    g2u: dict[str, str],
    seqs: dict[str, str],
    batch_size: int,
) -> dict[str, float]:
    """
    Score all variants for one model checkpoint.
    Returns {variant_key: delta_ll}.
    """
    import torch

    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    scores: dict[str, float] = {}
    for gene, variants in gene_variants.items():
        uniprot = g2u.get(gene)
        seq = seqs.get(uniprot, "") if uniprot else ""
        if not seq:
            for v in variants:
                scores[v["key"]] = float("nan")
            continue

        # Group by position — one forward pass per unique position.
        from collections import defaultdict
        pos_to_variants: dict[int, list[dict]] = defaultdict(list)
        for v in variants:
            pos_to_variants[v["pos"]].append(v)

        pos_list = sorted(pos_to_variants.keys())
        for batch_start in range(0, len(pos_list), batch_size):
            batch_pos = pos_list[batch_start : batch_start + batch_size]
            batch_data = []
            batch_meta = []

            for pos in batch_pos:
                win_seq, new_pos = window_sequence(seq, pos)
                masked = list(win_seq)
                masked[new_pos - 1] = "<mask>"
                batch_data.append((f"{gene}_{pos}", "".join(masked)))
                batch_meta.append((pos, new_pos))

            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.inference_mode():
                out = model(tokens, repr_layers=[], return_contacts=False)
            logits = out["logits"].cpu().float()  # (B, L+2, vocab)

            for i, (pos, new_pos) in enumerate(batch_meta):
                tok_idx = new_pos  # 1-indexed; BOS at position 0, so correct
                log_probs = torch.log_softmax(logits[i, tok_idx], dim=-1).numpy()
                for v in pos_to_variants[pos]:
                    wt_aa, mut_aa = v["wt_aa"], v["mut_aa"]
                    if wt_aa in aa_to_idx and mut_aa in aa_to_idx:
                        delta = (
                            float(log_probs[aa_to_idx[mut_aa]])
                            - float(log_probs[aa_to_idx[wt_aa]])
                        )
                    else:
                        delta = float("nan")
                    scores[v["key"]] = delta

    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "esm1v_scores_full.json")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="positions per GPU batch")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                    help="genes between incremental saves")
    args = ap.parse_args()

    try:
        import torch
        import esm as esm_lib
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Run on RunPod or equivalent.")

    g2u, seqs = build_gene_maps()

    variants_raw: list[dict] = json.load(open(DATA / "pathogenicity_valid_variants.json"))
    print(f"variants: {len(variants_raw):,}")

    # Index by gene.
    from collections import defaultdict
    gene_variants: dict[str, list[dict]] = defaultdict(list)
    for v in variants_raw:
        key = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        gene_variants[v["gene"]].append({
            "key": key,
            "pos": v["aa_pos"],
            "wt_aa": v["aa_wt"],
            "mut_aa": v["aa_mut"],
            "label": v["label"],
        })
    print(f"genes: {len(gene_variants):,}")

    ckpt_path = DATA / "esm1v_scores_ckpt.json"
    done_genes: set[str] = set()
    # Accumulate per-checkpoint scores: {ckpt_name: {vkey: delta_ll}}
    per_ckpt: dict[str, dict[str, float]] = {c: {} for c in CHECKPOINTS}

    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        done_genes = set(ckpt["done_genes"])
        per_ckpt = ckpt["per_ckpt"]
        print(f"Resuming from checkpoint: {len(done_genes)}/{len(gene_variants)} genes done")

    remaining_genes = [g for g in gene_variants if g not in done_genes]
    print(f"Genes remaining: {len(remaining_genes):,}")

    for ckpt_name in CHECKPOINTS:
        print(f"\nLoading {ckpt_name}...")
        model, alphabet = esm_lib.pretrained.load_model_and_alphabet(ckpt_name)
        model = model.to(device).eval()
        print(f"  model loaded on {device}")

        gene_batch: list[str] = []
        for i, gene in enumerate(remaining_genes):
            gene_batch.append(gene)

            if len(gene_batch) >= args.checkpoint_every or i == len(remaining_genes) - 1:
                batch_gene_variants = {g: gene_variants[g] for g in gene_batch}
                new_scores = score_variants_single_model(
                    model, alphabet, device,
                    batch_gene_variants, g2u, seqs,
                    args.batch_size,
                )
                per_ckpt[ckpt_name].update(new_scores)
                n_ok = sum(1 for v in new_scores.values() if not np.isnan(v))
                print(f"  [{i+1}/{len(remaining_genes)}] batch of {len(gene_batch)} genes "
                      f"scored ({n_ok}/{len(new_scores)} non-nan)")
                gene_batch = []

        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Average across checkpoints; mark gene as done after both models ran.
    all_vkeys = set()
    for c in CHECKPOINTS:
        all_vkeys.update(per_ckpt[c].keys())

    averaged: dict[str, float] = {}
    for vkey in all_vkeys:
        vals = [per_ckpt[c][vkey] for c in CHECKPOINTS if vkey in per_ckpt[c]]
        finite = [v for v in vals if not np.isnan(v)]
        averaged[vkey] = float(np.mean(finite)) if finite else float("nan")

    n_scored = sum(1 for v in averaged.values() if not np.isnan(v))
    print(f"\nFinal: {n_scored:,}/{len(averaged):,} variants with finite ΔLL")

    with open(args.out, "w") as f:
        json.dump(averaged, f, indent=2, sort_keys=True)
    print(f"Wrote {args.out}")

    if ckpt_path.exists():
        ckpt_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
