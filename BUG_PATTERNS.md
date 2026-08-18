Bugs already found in this codebase. Do not reintroduce them. Prefer an assertion that crashes over a fallback that could pass as a real result.

## Patterns

- Feature matrix indexed by the wrong gene list — matrix and index came from files in different orders. Assert row counts match, naming both files.
- Imputing missing values (e.g. median) before cross-validation instead of restricting to observed rows.
- Integer-encoded labels reaching a helper that compares against string class names.
- Re-standardizing after projecting out a subspace, reintroducing the variance just removed.
- Caching a fetch exception as `None`, permanently treating a transient failure as a real result.
- Comparing against hardcoded numbers copied from an old run instead of recomputing.
- Averaging per-seed metrics with plain `np.mean` when a fold can return NaN.
- A bounds check or missing-file fallback that silently substitutes a plausible value instead of raising — this is how the row-index bug above went unnoticed for months.
- A shared contract (helper, canonical file) changed but not every caller/test was updated. Grep all readers after any such change.
- A `seed` argument that seeds numpy but not torch, so weight init/dropout/shuffling stay random and reported seed variance is meaningless.
- A cache keyed on a subset of the parameters that produced the file. Any cache check must compare a fingerprint of every input that affects the file's content — data, model, and config flags — not just a few flags or file presence. A row-count check alone is not a fingerprint — two different variant lists can have the same length; use a content hash.
- Capping each class independently per group without equalizing both classes within each group. If gene X has 20 pathogenic and 5 benign variants, an independent cap at 20 ships 20/5 — the probe can learn group-level label prevalence as a shortcut. After capping, downsample both classes within each group to the minority count, and drop groups with only one class.
- Fitting StandardScaler (or any preprocessing) on both train and test data before splitting. The scaler's mean and variance leak test-set statistics into the training features. Fit on training rows only, then transform both train and test with that scaler. This applies to any cross-validation or transfer-evaluation loop — the scaler must be re-fit inside each fold.
- Fitting PCA (or any dimensionality reduction) on the full dataset before cross-validation splitting. The principal components capture test-set variance and leak it into training features. PCA must be fitted inside each fold on training rows only, then used to transform both train and test.
- Mixing two definitions of the same metric across a report. Per-fold averaging (mean of fold-level F1s) and pooled out-of-fold scoring (one F1 call on all concatenated OOF predictions) give different numbers on the same data. A bootstrap CI computed on pooled OOF predictions must be paired with a pooled point estimate, not a fold-mean one.
- Holding a nuisance constant fixed across bootstrap resamples when resampling changes the quantity it depends on. A majority-class baseline, for example, depends on class proportions; resampling families shifts those proportions, so the baseline must be recomputed inside each replicate.
- A cache keyed on data parameters but not on the algorithm version. When the processing logic changes (e.g. a new balancing step), old caches built without that logic look valid because no parameter changed. Include a version counter in the cache params and bump it on algorithm changes.
- Balancing a dataset and then filtering it without rebalancing. If one class loses more rows during validation/embedding filtering, the balance is broken. Any filter applied after balancing must rebalance the surviving subset per group.
- A result cache that omits inputs affecting the result through an indirect path: the Pfam family map affects CV fold assignments, and embedding content can change while the model name stays the same. Every input that changes the output — even if it enters through a different file — must be fingerprinted in the cache key.

- Computing cosine similarity between coefficient vectors fitted under different StandardScalers. Each scaler defines its own coordinate system; coefficients from scaler A and scaler B are not in the same space, so their dot product is not a valid cosine in the original embedding space. Fit one shared scaler when the coefficients will be compared.
- Deciding a gate on one metric definition (fold-mean F1) while attaching a CI computed from a different definition (pooled-OOF F1). The CI does not bound the quantity the gate is judging. Use the same pooled-OOF computation for both the point estimate and the CI.
- Reporting a pass/fail verdict on a difference (A beats B by margin X) without a paired bootstrap CI on the difference. Without the paired interval, the preregistered decision rule requiring a CI that excludes zero cannot be applied. Always compute the paired CI when the decision rule requires one.

## Standing rules

- File I/O: use a context manager; `newline=""` for csv; skip headers explicitly; guard row length before indexing; never write `None` into a TSV field (use a `_missing` column instead).
- Caching: write atomically via the shared JSON helper; load inside `try/except` that deletes and re-fetches on corruption; write only after all fetches succeed; treat a definitive 404 as a real cacheable `None`, distinct from a transient error.
- NumPy: never compare an array to `None` with `==`/`!=`.
- Python: bind loop variables into lambdas via default args; no hardcoded dataset-size assertions; check `math.isfinite` after parsing floats; route every skip path to a named counted bucket.
- Use the shared gene→row builder and atomic JSON writer, not local copies. When building a tuple-keyed lookup, collect collisions instead of overwriting silently.
- Call `parser.parse_args()` with no argument, not `parse_args([])`.
