# Running the tests

The suite is plain pytest with no fixtures that reach outside the repository. No test needs a GPU,
a network connection, or any embedding array or result file to be present. Every test that touches
a data or results path redirects it to a temporary directory first, so the suite can run on a clean
checkout.

## Setup

pytest is in the `dev` dependency group rather than the runtime dependencies, so an editable
install alone does not provide it:

    uv sync

If you are using a plain virtual environment instead, install the package editable and add pytest:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    pip install pytest

## Commands

    pytest                  # the whole suite
    pytest -m "not slow"    # everything except the real-probe fits
    pytest -m slow          # only the real-probe fits

`testpaths` is set to `tests`, so a bare `pytest` picks up the suite without an argument. Under uv,
prefix any of these with `uv run`.

## The slow marker

Three test classes are marked `slow` because they fit real logistic-regression and MLP probes
rather than stubs: the pooled within-family permutation test and the real-probe class in
`tests/experiments/mechanism/test_within_family_pooled.py`, and the end-to-end panel test in
`tests/experiments/mechanism/test_homology_partition_panel.py`. Everything else runs in seconds.
Deselecting them is the fast pre-commit loop; the runbook's step 0.7 precondition expects the full
suite, including them, to pass before a run starts.

## Layout

Tests mirror the package: `tests/utils/` covers the shared probe, split, metric, bootstrap, and
path helpers; `tests/fetch_data/` covers cohort construction and every external fetch, with all
HTTP mocked; `tests/experiments/` has one directory per experiment family, matching
`src/esm2_mech/experiments/`; and `tests/scripts/` covers the run-comparison tool, which is why
`pythonpath` includes the repository root as well as `src`.

The homology-partition panel module is not part of the current pipeline, so its test file begins
with an `importorskip` and is skipped rather than failed when the module is absent. Several classes
in that file and in the pooled within-family file are marked `xfail` under strict mode, so a
behaviour that starts passing again is reported as a failure rather than passing silently.
