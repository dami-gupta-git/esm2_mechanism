"""Repository-level guards for the shared model-seed contract."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
PRODUCTION_ROOTS = (ROOT / "src" / "esm2_mech", ROOT / "scripts")

MEAN_STD_ALLOWLIST = {
    (
        "src/esm2_mech/experiments/mechanism/mechanism_within_family.py",
        "_within_family_majority_reference",
    ),
    ("src/esm2_mech/experiments/mechanism/cascade_mechanism.py", "run_arm"),
    ("src/esm2_mech/experiments/stability/megascale_stability.py", "run_regression_cv"),
    ("src/esm2_mech/experiments/stability/megascale_stability.py", "main"),
    ("src/esm2_mech/experiments/stability/megascale_mlp.py", "run_mlp_regression"),
}

DIRECT_SEED_REDUCER_ALLOWLIST = {
    ("src/esm2_mech/utils/seed_aggregation.py", "aggregate_seed_values"),
    (
        "src/esm2_mech/utils/seed_aggregation.py",
        "aggregate_seed_confusion_matrices",
    ),
    ("src/esm2_mech/experiments/geometry/axis_analysis.py", "_seed_summary"),
    # The one within-seed fold/partition summary the geometry producers share.
    ("src/esm2_mech/utils/metrics.py", "within_seed_summary"),
}


def _production_files():
    for root in PRODUCTION_ROOTS:
        yield from root.rglob("*.py")


class _CallFinder(ast.NodeVisitor):
    def __init__(self):
        self.function = None
        self.calls = []

    def visit_FunctionDef(self, node):
        parent = self.function
        self.function = node.name
        self.generic_visit(node)
        self.function = parent

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "mean_std_n":
            self.calls.append((self.function, node.lineno))
        self.generic_visit(node)


def test_mean_std_n_is_only_used_for_declared_non_seed_units():
    found = set()
    for path in _production_files():
        finder = _CallFinder()
        finder.visit(ast.parse(path.read_text(), filename=str(path)))
        relative = str(path.relative_to(ROOT))
        found.update((relative, function) for function, _line in finder.calls)
    assert found == MEAN_STD_ALLOWLIST


def test_named_seed_reducers_do_not_reimplement_mean_or_spread():
    found = set()
    statistical_methods = {"average", "mean", "nanmean", "nanstd", "std"}
    for path in _production_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = function.name.lower()
            if "seed" not in name or not any(
                word in name for word in ("aggregate", "combine", "pool", "summar")
            ):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in statistical_methods
                for call in ast.walk(function)
            ):
                found.add((str(path.relative_to(ROOT)), function.name))
    assert found == DIRECT_SEED_REDUCER_ALLOWLIST


SEED_RECORD_FACTORIES = {"make_seed_record", "make_seed_payload_record"}

# The shared module builds and interprets the aggregate itself; the run-comparison
# script rebuilds one from flattened keys purely to hand it to the shared reader.
COMPLETENESS_POLICY_ALLOWLIST = {
    "src/esm2_mech/utils/seed_aggregation.py",
    "scripts/compare_runs.py",
}


def _seed_from_position_hits(tree) -> list[int]:
    """Lines where a seed record takes its identity from an enumerate() counter."""
    hits = []
    for node in ast.walk(tree):
        iterators = []
        if isinstance(node, (ast.For, ast.AsyncFor)):
            iterators = [node.iter]
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            iterators = [generator.iter for generator in node.generators]
        if not any(
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "enumerate"
            for iterator in iterators
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in SEED_RECORD_FACTORIES
            ):
                hits.append(inner.lineno)
    return hits


def test_a_seed_never_takes_its_identity_from_a_list_position():
    """See BUG_PATTERNS.md: counting positions is correct only until a seed is skipped.

    One skipped seed shifts every later value onto the wrong seed while the
    completeness check still sees a full set, so nothing raises.
    """
    offenders = []
    for path in _production_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders += [
            f"{path.relative_to(ROOT)}:{line}"
            for line in _seed_from_position_hits(tree)
        ]
    assert offenders == []


def test_no_consumer_decides_seed_completeness_for_itself():
    """Availability is the shared readers' judgement, never a caller's.

    A consumer that counts contributing seeds or reads the stored state is a
    second copy of the rule, free to drift from the one the producers apply.
    """
    offenders = []
    for path in _production_files():
        relative = str(path.relative_to(ROOT))
        if relative in COMPLETENESS_POLICY_ALLOWLIST:
            continue
        source = path.read_text()
        for marker in ('"contributing_seeds"', '["state"]', 'get("state")'):
            if marker in source:
                offenders.append(f"{relative}: {marker}")
    assert offenders == []


def test_removed_partial_oof_and_compatibility_schema_do_not_return():
    source = "\n".join(path.read_text() for path in _production_files())
    assert "average_oof_over_seeds" not in source
    assert "SEED_MEAN_SUFFIX" not in source
    assert "ci_low_seed_aggregate" not in source
    assert "ci_high_seed_aggregate" not in source


def test_publication_figures_never_average_interval_bounds_across_seeds():
    """A figure may draw an interval, but never one built by pooling seeds.

    An interval is a within-seed resampling quantity, so averaging its bounds
    across seeds would describe neither the resampling uncertainty nor the
    spread between seeds.
    """
    source = (ROOT / "src/esm2_mech/figures/manuscript_figures.py").read_text()
    assert "ci_low_seed_aggregate" not in source
    assert "ci_high_seed_aggregate" not in source


def test_wt_sensitivity_consumers_do_not_read_removed_interval_votes():
    paths = (
        ROOT
        / "src"
        / "esm2_mech"
        / "experiments"
        / "mechanism"
        / "wt_identity_sensitivity.py",
        ROOT
        / "src"
        / "esm2_mech"
        / "experiments"
        / "mechanism"
        / "wt_window_average_sensitivity.py",
    )
    source = "\n".join(path.read_text() for path in paths)
    assert "meets_split_gap_interval_rule" not in source
    assert 'split_gap_summary["seed_vote"]' not in source
