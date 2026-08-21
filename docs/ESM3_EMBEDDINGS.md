# Generating the ESM-3 embeddings

The ESM-3 scale-and-structure experiment is a run6-era result and is not part of the run_biorxiv
pipeline. Its outputs are in `results/run6/esm3_mechanism/` and its report is
`reports/run6/report_esm3_mechanism.md`. Regenerating the embeddings is only necessary if that
experiment is revisited.

There is no separate embedding script. The experiment module
`src/esm2_mech/experiments/esm3/esm3_mechanism.py` runs in three phases selected with `--phase`,
and phase 2 is the one that writes the arrays.

## Prerequisites

**The `esm` SDK, not `fair-esm`.** Phases 1 and 2 import from `esm.sdk.api` and `esm.pretrained`,
which come from the EvolutionaryScale ESM-3 SDK published as `esm`. The pinned version is
`esm==3.2.1.post1`. This is a different package from `fair-esm`, which is what `pyproject.toml`
declares and what the current `.venv` has installed, and the two claim the same top-level `esm`
module name. Installing the ESM-3 SDK into the existing environment therefore breaks ESM-2
embedding generation; use a separate environment for this experiment. Both phases exit with
`ERROR: esm package not found` if the SDK is absent.

**A HuggingFace token with the model licence accepted.** The weights are gated behind the
EvolutionaryScale non-commercial licence on the `esm3-sm-open-v1` model repository. Accept it on
HuggingFace, then export the token as `HF_TOKEN` before running. No code in this repository reads
the variable; it is consumed by the SDK's own download path.

**A GPU for phase 2.** It raises `Phase 2 requires a GPU.` when CUDA is unavailable. Run it on
RunPod inside tmux and copy the dataset directory back afterwards. Phases 1 and 3 are CPU work,
and phase 1 additionally needs network access.

## Commands

    python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 1 --dataset merged
    python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 2 --dataset merged
    python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 3 --dataset merged --seeds 5

`--dataset` selects the variant cohort and takes `geras` or `merged`, defaulting to `geras`. Use
`merged`; see the note on the contaminated geras set below. Phase 3 also accepts `--seeds`,
`--no_ci`, and `--n_boot`, matching the other experiments. A `--batch_size` flag is accepted but
has no effect in phase 2.

Phase 1 downloads AlphaFold2 models from the EBI prediction API into `data/cache/af2_structures/`
and caches the per-residue backbone coordinates it parses out of them in
`data/cache/esm3_struct_tokens.json`, keyed by UniProt ID and shared across both datasets. Neither
of those caches is currently present, so phase 1 has to be run in full before phase 2. Transient
fetch failures are deliberately not cached, so a network problem during phase 1 cannot permanently
mark a protein as structure-free.

Phase 2 loads the model, tokenises the cached coordinates for each variant's sequence window, and
runs each variant twice: sequence only, and sequence with structure tokens. A variant whose
coordinates are missing, whose AlphaFold length disagrees with the full sequence length, or whose
window slice is too short falls back to sequence-only and is counted in the fallback total rather
than dropped. Function tokens are not implemented. Progress is checkpointed every 100 variants and
the checkpoint files are removed on completion, so leftover `*_ckpt_*.npy` files mean the phase
did not finish.

## Outputs

Arrays are written to `data/embeddings/esm3-sm-open-v1/<dataset>/`, keyed by the model name rather
than by run. Each dataset directory holds `seq_mean.npy` and `seq_struct_mean.npy`, which are the
mutant-minus-wildtype deltas, the four raw `_wt` and `_mut` arrays behind them, `valid_idx.npy`
giving the rows of the cohort that were embedded, and `struct_meta.json` recording how many
variants were embedded, skipped, given structure tokens, or fell back to coordinates.

Phase 3 writes its probe results under the current run, so it lands in
`results/<RUN_NAME>/esm3_mechanism/<dataset>/summary.json`.

## The geras set is contaminated

Both dataset directories already exist on disk, but only `merged` is usable. The `geras` arrays
were produced before phase 2 checked that the wildtype residue matched the reference, and 93
variants were built on the wrong wildtype/mutant pair. That directory carries its own
`KNOWN_ISSUES.md` recording this. It also predates the addition of the raw `_wt` and `_mut` arrays,
so it holds only the two delta arrays. The check has since been fixed; the clean, matched set is
`merged`, covering 17,826 variants with structure tokens applied to 94.5 percent of them.
