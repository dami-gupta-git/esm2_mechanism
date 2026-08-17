"""Score ClinVar pathogenic/benign variants with ESM-1v masked-marginal delta-LL."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.utils.constants import AA_ORDER
from esm2_mech.utils.eval import vkey
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import DATA_DIR as DATA, VALID_VARIANTS_JSON
from esm2_mech.utils.sequences import window_sequence

CHECKPOINTS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
]
CHECKPOINT_EVERY = 50  # genes between saves


def build_gene_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (gene -> uniprot, uniprot -> sequence)."""
    with open(VALID_VARIANTS_JSON) as _f:
        merged = json.load(_f)
    g2u: dict[str, str] = {}
    for r in merged:
        g2u.setdefault(r["gene"], r["uniprot_id"])
    with open(DATA / "sequences.json") as _f:
        seqs: dict[str, str] = json.load(_f)
    return g2u, seqs


def score_variants_single_model(
    model,
    alphabet,
    device: str,
    gene_variants: dict[str, list[dict]],
    g2u: dict[str, str],
    seqs: dict[str, str],
    batch_size: int,
) -> tuple[dict[str, float], set[str]]:
    """Score all variants for one model checkpoint."""
    import torch

    batch_converter = alphabet.get_batch_converter()
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA_ORDER}

    scores: dict[str, float] = {}
    genes_without_sequence: set[str] = set()
    for gene, variants in gene_variants.items():
        uniprot = g2u.get(gene)
        seq = seqs.get(uniprot, "") if uniprot else ""
        if not seq:
            print(f"  WARNING: no sequence for gene {gene!r} (uniprot={uniprot!r}) — skipping")
            genes_without_sequence.add(gene)
            continue

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
                win_seq, new_pos, _ = window_sequence(seq, pos)
                masked = list(win_seq)
                masked[new_pos - 1] = "<mask>"
                batch_data.append((f"{gene}_{pos}", "".join(masked)))
                batch_meta.append((pos, new_pos))

            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.inference_mode():
                out = model(tokens, repr_layers=[], return_contacts=False)
            logits = out["logits"].cpu().float()

            for i, (pos, new_pos) in enumerate(batch_meta):
                tok_idx = new_pos
                log_probs = torch.log_softmax(logits[i, tok_idx], dim=-1).numpy()
                for v in pos_to_variants[pos]:
                    wt_aa, mut_aa = v["wt_aa"], v["mut_aa"]
                    if wt_aa in aa_to_idx and mut_aa in aa_to_idx:
                        delta = float(log_probs[aa_to_idx[mut_aa]]) - float(
                            log_probs[aa_to_idx[wt_aa]]
                        )
                    else:
                        delta = float("nan")
                    scores[v["key"]] = delta

    return scores, genes_without_sequence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "esm1v_scores_full.json")
    ap.add_argument(
        "--batch-size", type=int, default=16, help="positions per GPU batch"
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="genes between incremental saves",
    )
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

    with open(DATA / "pathogenicity_valid_variants.json") as _f:
        variants_raw: list[dict] = json.load(_f)
    print(f"variants: {len(variants_raw):,}")

    from collections import defaultdict

    gene_variants: dict[str, list[dict]] = defaultdict(list)
    for v in variants_raw:
        key = vkey(v)
        gene_variants[v["gene"]].append(
            {
                "key": key,
                "pos": v["aa_pos"],
                "wt_aa": v["aa_wt"],
                "mut_aa": v["aa_mut"],
                "label": v["label"],
            }
        )
    print(f"genes: {len(gene_variants):,}")

    ckpt_path = DATA / "esm1v_scores_ckpt.json"
    per_ckpt: dict[str, dict[str, float]] = {c: {} for c in CHECKPOINTS}
    ckpt_done: dict[str, set[str]] = {c: set() for c in CHECKPOINTS}

    if ckpt_path.exists():
        try:
            with open(ckpt_path) as _f:
                saved = json.load(_f)
        except json.JSONDecodeError:
            print(f"WARNING: corrupt checkpoint at {ckpt_path} — discarding and restarting")
            ckpt_path.unlink()
            saved = None
        if saved is not None:
            per_ckpt = saved.get("per_ckpt", per_ckpt)
            for c in CHECKPOINTS:
                ckpt_done[c] = set(saved.get("ckpt_done", {}).get(c, []))
            done_genes = set.intersection(*[ckpt_done[c] for c in CHECKPOINTS])
            print(
                f"Resuming from checkpoint: {len(done_genes)}/{len(gene_variants)} genes fully done"
            )
        else:
            done_genes = set()
    else:
        done_genes = set()

    remaining_genes = [g for g in gene_variants if g not in done_genes]
    print(f"Genes remaining: {len(remaining_genes):,}")

    for ckpt_name in CHECKPOINTS:
        print(f"\nLoading {ckpt_name}...")
        model, alphabet = esm_lib.pretrained.load_model_and_alphabet(ckpt_name)
        model = model.to(device).eval()
        print(f"  model loaded on {device}")

        genes_for_ckpt = [g for g in remaining_genes if g not in ckpt_done[ckpt_name]]

        gene_batch: list[str] = []
        for i, gene in enumerate(genes_for_ckpt):
            gene_batch.append(gene)

            if len(gene_batch) >= args.checkpoint_every or i == len(genes_for_ckpt) - 1:
                batch_gene_variants = {g: gene_variants[g] for g in gene_batch}
                new_scores, no_seq_genes = score_variants_single_model(
                    model,
                    alphabet,
                    device,
                    batch_gene_variants,
                    g2u,
                    seqs,
                    args.batch_size,
                )
                per_ckpt[ckpt_name].update(new_scores)
                # Genes without a sequence are left out of ckpt_done so they are retried if sequences.json is updated.
                ckpt_done[ckpt_name].update(g for g in gene_batch if g not in no_seq_genes)
                n_ok = sum(1 for v in new_scores.values() if not np.isnan(v))
                print(
                    f"  [{i+1}/{len(genes_for_ckpt)}] batch of {len(gene_batch)} genes "
                    f"scored ({n_ok}/{len(new_scores)} non-nan)"
                )
                gene_batch = []

                all_fully_done = set.intersection(*[ckpt_done[c] for c in CHECKPOINTS])
                ckpt_state = {
                    "done_genes": list(all_fully_done),
                    "per_ckpt": per_ckpt,
                    "ckpt_done": {c: list(ckpt_done[c]) for c in CHECKPOINTS},
                }
                atomic_write_json(ckpt_path, ckpt_state)
                print(
                    f"  Checkpoint saved ({len(all_fully_done)} genes fully done)",
                    flush=True,
                )

        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

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

    atomic_write_json(args.out, averaged, indent=2, sort_keys=True)
    print(f"Wrote {args.out}")

    if ckpt_path.exists():
        ckpt_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
