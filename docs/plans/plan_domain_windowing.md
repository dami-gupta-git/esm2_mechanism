# Plan: Domain-aware windowing for long proteins

## The problem

ESM-2 has a 1022-residue token limit (`MAX_SEQ_LEN` in `src/esm2_mech/utils/constants.py`). Any protein longer than that gets truncated to a ±511-residue window centred on the mutated position (`window_sequence` in `src/esm2_mech/utils/sequences.py`). The window is chosen purely by sequence position — it has no awareness of protein structure, so it can cut through the middle of the domain the mutation sits in, or drop a domain the mutation acts on entirely.

This is not a corner case: 2,769 of 10,233 variants (27%) and 143 of 948 genes (15%) exceed 1022 residues and get windowed. The affected genes include some of the largest and most domain-dense proteins in the dataset (KMT2D, RYR2, HERC2, PKD1, ANK2, CUBN, RELN — repeat-domain and multi-domain proteins), so this is concentrated exactly where domain structure matters most.

Proteins ≤1022 residues (the remaining 85% of genes) are unaffected and pass through unwindowed; this plan only touches the 143 affected genes.

## Why Pfam alone doesn't fix it

The cached `data/pfam_families.json` stores a single Pfam family ID per gene, not per-domain residue coordinates. It's sufficient for family-level grouping (used in family-split CV) but says nothing about where a domain starts or ends in the sequence, so it can't anchor a window. All 143 affected genes have a Pfam ID recorded, but that coverage number is misleading for this purpose — it doesn't mean domain boundaries are known.

## Step 1 — Fetch domain coordinates

Add a fetcher under `src/esm2_mech/fetch_data/` (following the caching pattern in `fetch_alphamissense.py`) that queries the InterPro REST API per UniProt ID and returns each domain's start/end residue and source database (Pfam, SMART, etc.). Only fetch for the 143 genes that exceed 1022 residues.

Output: `data/cache/interpro_domains.json`, keyed by UniProt ID → list of `(start, end, domain_id)`.

## Step 2 — Domain-aware window selection

Extend `window_sequence` with a domain-aware path:

- Find the domain (if any) whose span contains `aa_pos`.
- If found and the domain fits within `MAX_SEQ_LEN`, centre the window on the domain span (full domain plus symmetric padding to fill the budget) instead of a flat ±511 offset.
- If the domain itself exceeds `MAX_SEQ_LEN`, or the mutation falls outside any annotated domain (linker/disordered region), fall back to the current position-centred window — there's no domain boundary to respect in either case.

## Step 3 — Track provenance

Add a per-variant flag recording whether the domain-aware path or the fallback was used. Needed to compare results between the two groups later, and required by this project's rule against silently blending differently-derived values.

## Step 4 — Re-run affected experiments

Re-extract ESM-2 embeddings only for the 143 affected genes, then re-run Result 1 (and any downstream result that depends on it) restricted to that subset, comparing against the current position-centred windowing before deciding whether to redo the full pipeline run.

## Known gap — not covered by this plan

Mutations that fall in disordered linker regions between domains have no domain to anchor a window to; they will still fall back to plain position-centred windowing. Before starting Step 1, check what fraction of the 143 genes' mutations actually land inside an annotated domain — if it's small, the InterPro fetch buys less than expected and that should be weighed before committing to it.
