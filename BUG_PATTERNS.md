## Code quality

Bugs that have been found recently that you should not repeat

- Wrong row index for feature matrices e.g.
badonyi_mechanism, badonyi_leakage_analysis, mmseqs_cluster_holdout, perturbation_probe, proteome_mechanism, per_gene_ablation, clinical_utility, enzyme_classification all indexed *_features_aligned.npy by gene_list.tsv, but those matrices are written in gene_universe.tsv order (a shorter, differently-ordered filtered set). Every gene was reading another gene's features. 
- Fabricated values for missing data e.g. broadcast/broadcast_gene_features,build_proteome_features and build_badonyi_features median-imputed before CV.
- Int-encoded y handed to string-keyed helpers e.g. LabelEncoder-encoded labels reached run_logreg_cv/run_mlp_cv/compute_metrics, which compare against string class labels
- Projection undone by re-standardization. e.g.(mechanism_delta_probe) — per-fold StandardScaler after projecting out the stability subspace reintroduced variance along it. 
- Caching transient failures (esm3_mechanism phase 1) — any AF2 fetch exception cached None, permanently downgrading a protein to seq-only
- Hardcoded reference numbers (clan_holdout) — compared against literals 0.352/0.387 from old runs
- NaN-poisoned means / mismatched fold sets — esm3_mechanism used plain np.mean over per-seed AUROCs (now mean_std_n), and its MLP and LogReg arms skipped folds on different conditions
- A guard that degrades instead of raising, hiding the bug behind it. The row-index bug above survived for months because `broadcast()` clamped with `if r < matrix.shape[0]`: only the overflow rows became NaN, and the rest quietly read the wrong gene. Same shape elsewhere — esm3_mechanism's `json.loads(SEQUENCES_JSON) if SEQUENCES_JSON.exists() else {}` returned an empty variant set rather than failing, and fit_stability_subspace_megascale read two .npy files nothing ever wrote, so it always fell through to the Path B fit and `stability_path: "A_megascale"` was unreachable. A bounds check, existence fallback or bare `except` that substitutes a plausible value converts a loud failure into a silently wrong number.
- Refactor that leaves callers behind. Every item above with more than one file in it is this: a shared helper or canonical input changed and only some call sites followed (gene_universe.tsv became canonical and no consumer was repointed; run_logreg_cv/run_mlp_cv moved to string labels and four callers kept passing ints). After changing a shared contract, grep for every reader of the old one — including tests — and fix them in the same change.
- A `seed` argument that does not seed every RNG the run uses. `run_mlp_probe_cv`, `megascale_mlp`, `contrastive_mechanism` and `proteome_mechanism` all took a seed but never called `torch.manual_seed`, so it controlled only the numpy validation split while torch weight init, dropout masks and `DataLoader(shuffle=True)` stayed unseeded. Two runs of the same code on the same data gave family-split macro-F1 0.380 and 0.415, and the reported across-seed std (±0.010) measured only the val-split variation, not the real spread. Any torch training loop must call `torch.manual_seed(seed + fold_i)` before the model is constructed and pass an explicit `torch.Generator` to every `DataLoader`.

## What catches these

Assert the invariant at the point it must hold, so a violation crashes instead of producing a number. This is what the fixes above added, and what their absence cost:

- Row alignment: `len(gene_to_row) == matrix.shape[0]`, raising and naming both files. Nothing asserted this, which is why the index bug lasted months.
- NaN reaching a model that cannot consume it: `require_no_nan` in utils/probes.py, on every NaN-intolerant runner.
- A projection actually removing its subspace: `assert_subspace_removed` (var along the removed directions ≈ 0).
- Embeddings row-aligned to their variant list: the explicit row-count check in io.load_variants_and_delta.

Prefer an assertion over a comment saying the invariant holds; prefer a crash over a fallback whenever the fallback would be indistinguishable from a real result.