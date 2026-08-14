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

## Standing rules

- File I/O: use a context manager; `newline=""` for csv; skip headers explicitly; guard row length before indexing; never write `None` into a TSV field (use a `_missing` column instead).
- Caching: write atomically via the shared JSON helper; load inside `try/except` that deletes and re-fetches on corruption; write only after all fetches succeed; treat a definitive 404 as a real cacheable `None`, distinct from a transient error.
- NumPy: never compare an array to `None` with `==`/`!=`.
- Python: bind loop variables into lambdas via default args; no hardcoded dataset-size assertions; check `math.isfinite` after parsing floats; route every skip path to a named counted bucket.
- Use the shared gene→row builder and atomic JSON writer, not local copies. When building a tuple-keyed lookup, collect collisions instead of overwriting silently.
- Call `parser.parse_args()` with no argument, not `parse_args([])`.
