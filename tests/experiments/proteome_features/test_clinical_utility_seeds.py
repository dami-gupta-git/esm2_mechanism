"""The clinical-utility entry point must honour the seed it was asked for.

`--seed N` used to be routed by seed count: one seed selected a descriptive
report that takes no seed at all and therefore ran at the module default, so the
requested seed was silently discarded and the output carried no record of which
seed produced it. These tests pin the two properties that made that possible.
"""

import ast
import inspect
from pathlib import Path

import pytest

from esm2_mech.experiments.proteome_features import clinical_utility


SOURCE = Path(inspect.getfile(clinical_utility)).read_text()


def _entry_block() -> ast.If:
    """The `if __name__ == "__main__":` block, which argparse drives."""
    tree = ast.parse(SOURCE)
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", None) == "__name__"
        ):
            return node
    raise AssertionError("no __main__ block found")


def test_the_report_takes_the_seed_it_runs_at():
    """A report that takes no seed silently describes the module default."""
    parameters = inspect.signature(clinical_utility.main).parameters
    assert "seed" in parameters
    assert parameters["seed"].default is inspect.Parameter.empty, (
        "seed must be required, so the report cannot fall back to the module "
        "default and describe a different seed from the one requested"
    )


def test_the_report_applies_that_seed():
    """Declaring the seed is not enough; it has to reach the estimators."""
    body = next(
        node
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assigns_random_state = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RANDOM_STATE"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "seed"
        for node in ast.walk(body)
    )
    assert assigns_random_state, "main must apply its seed, not just accept it"


def test_the_entry_point_does_not_branch_on_how_many_seeds_were_asked_for():
    """One seed and five seeds take the same path through the seed contract.

    The old branch sent a single seed to the descriptive report instead, so it
    produced no per-seed file and never went through the shared contract.
    """
    for node in ast.walk(_entry_block()):
        if not isinstance(node, ast.If):
            continue
        test_source = ast.unparse(node.test)
        assert "len(seeds)" not in test_source, (
            f"entry point still branches on the seed count: {test_source}"
        )


def test_every_requested_seed_reaches_the_seeded_run():
    """The per-seed path is driven by the requested seeds, unconditionally."""
    loops = [
        node
        for node in ast.walk(_entry_block())
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "seeds"
    ]
    assert loops, "no loop over the requested seeds in the entry point"
    calls_run_seed = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_seed"
        for loop in loops
        for node in ast.walk(loop)
    )
    assert calls_run_seed


def test_the_report_does_not_reload_data_the_entry_point_already_loaded():
    """Both paths used to read the same four files independently."""
    body = next(
        node
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    loaders = {"read_csv", "load"}
    reloads = [
        ast.unparse(node)
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in loaders
    ]
    assert not reloads, f"main reloads data it is given: {reloads}"
