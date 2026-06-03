"""
Tests for esm2_mech.experiments.stability.tsuboyama_loader.

Invariants:
- _is_bare_natural_domain: accepts a bare PDB id ("1BK2.pdb"), rejects suffixed
  (_MUT background) names, designs, and empty strings
- load_tsuboyama_variants: keeps single-point missense from bare natural domains
- load_tsuboyama_variants: drops designs, _MUT-background rows, indels, multi-muts
- load_tsuboyama_variants: drops ddG_ML == "-" / "" / unparseable (never imputes)
- load_tsuboyama_variants: drops rows failing the wt/mut residue sanity check
- load_tsuboyama_variants: drops out-of-range positions
- load_tsuboyama_variants: variant dict has the S1724-compatible shape
- load_tsuboyama_variants: reads the cache verbatim when it already exists
- load_tsuboyama_variants: writes the parsed variants to the cache path
- load_tsuboyama_variants: conflicting wt sequences for one domain raise ValueError
- load_tsuboyama_variants: a repeated identical wt sequence is not a collision
- load_tsuboyama_variants: corrupt cache is deleted and re-parsed from the CSV
"""

import csv
import json

import pytest

from esm2_mech.experiments.stability.tsuboyama_loader import (
    _is_bare_natural_domain,
    load_tsuboyama_variants,
)


# ---------------------------------------------------------------------------
# _is_bare_natural_domain
# ---------------------------------------------------------------------------

class TestIsBareNaturalDomain:

    @pytest.mark.parametrize("name", ["1BK2.pdb", "1A0N.pdb", "2lhc.pdb"])
    def test_accepts_bare_pdb_id(self, name):
        assert _is_bare_natural_domain(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "1BK2.pdb_L7S",   # measured against an L7S mutant background
            "EA|run2_0001",   # de novo design
            "HHH",            # design
            "XX|run1",        # design
            "1BK2",           # no .pdb suffix -> regex rejects
            "",               # empty
        ],
    )
    def test_rejects_non_bare_or_design(self, name):
        assert _is_bare_natural_domain(name) is False


# ---------------------------------------------------------------------------
# load_tsuboyama_variants
# ---------------------------------------------------------------------------

# Column order is irrelevant (DictReader), but the loader reads these four keys.
_FIELDS = ["WT_name", "mut_type", "aa_seq", "ddG_ML"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(wt_name, mut_type, aa_seq, ddg):
    return {"WT_name": wt_name, "mut_type": mut_type, "aa_seq": aa_seq, "ddG_ML": ddg}


def _load(tmp_path, rows):
    csv_path = tmp_path / "tsuboyama.csv"
    cache_path = tmp_path / "variants.json"
    _write_csv(csv_path, rows)
    return load_tsuboyama_variants(csv_path=csv_path, cache_path=cache_path), cache_path


class TestLoadTsuboyamaVariants:

    # A 4-residue reference domain so positions 1..4 are valid.
    WT = "ACDE"

    def _domain_rows(self):
        return [_row("1BK2.pdb", "wt", self.WT, "-")]

    def test_keeps_valid_single_substitution(self, tmp_path):
        # A1G at pos 1: wt residue A matches, mutant seq carries G at pos 1.
        rows = self._domain_rows() + [
            _row("1BK2.pdb", "A1G", "GCDE", "1.5"),
        ]
        variants, _ = _load(tmp_path, rows)
        assert len(variants) == 1
        var = variants[0]
        assert var["protein"] == "1BK2.pdb"
        assert var["mutation_code"] == "A1G"
        assert var["wt_seq"] == self.WT
        assert var["mut_seq"] == "GCDE"
        assert var["var_pos"] == 1
        assert var["ddg"] == pytest.approx(1.5)

    def test_variant_dict_shape(self, tmp_path):
        rows = self._domain_rows() + [_row("1BK2.pdb", "A1G", "GCDE", "1.5")]
        variants, _ = _load(tmp_path, rows)
        assert set(variants[0]) == {
            "protein", "mutation_code", "wt_seq", "mut_seq", "var_pos", "ddg",
        }

    def test_drops_design_and_mut_background(self, tmp_path):
        rows = self._domain_rows() + [
            _row("HHH", "A1G", "GCDE", "1.5"),            # design
            _row("1BK2.pdb_L7S", "A1G", "GCDE", "1.5"),  # _MUT background
        ]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_drops_indels_and_multimutants(self, tmp_path):
        rows = self._domain_rows() + [
            _row("1BK2.pdb", "insA5", "ACDEA", "1.0"),  # insertion
            _row("1BK2.pdb", "delA1", "CDE", "1.0"),    # deletion
            _row("1BK2.pdb", "A1G_C2T", "GTDE", "1.0"), # multi-mutant
        ]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    @pytest.mark.parametrize("ddg", ["-", "", "NaN", "n/a"])
    def test_drops_missing_or_unparseable_ddg(self, tmp_path, ddg):
        # "-" / "" are the documented missing markers; "NaN"/"n/a" are unparseable
        # junk. None is imputed — all four rows must be dropped.
        rows = self._domain_rows() + [_row("1BK2.pdb", "A1G", "GCDE", ddg)]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_drops_wt_residue_mismatch(self, tmp_path):
        # Mutation code claims wt residue C at pos 1, but the reference has A.
        rows = self._domain_rows() + [_row("1BK2.pdb", "C1G", "GCDE", "1.0")]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_drops_mut_residue_mismatch(self, tmp_path):
        # Code says ->G at pos 1, but mut_seq carries T there.
        rows = self._domain_rows() + [_row("1BK2.pdb", "A1G", "TCDE", "1.0")]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_drops_out_of_range_position(self, tmp_path):
        # pos 9 exceeds the 4-residue reference.
        rows = self._domain_rows() + [_row("1BK2.pdb", "A9G", "GCDE", "1.0")]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_drops_substitution_without_wt_sequence(self, tmp_path):
        # Substitution row for a bare domain that has no "wt" reference row.
        rows = [_row("1ABC.pdb", "A1G", "GCDE", "1.0")]
        variants, _ = _load(tmp_path, rows)
        assert variants == []

    def test_writes_cache(self, tmp_path):
        rows = self._domain_rows() + [_row("1BK2.pdb", "A1G", "GCDE", "1.5")]
        variants, cache_path = _load(tmp_path, rows)
        assert cache_path.exists()
        with open(cache_path) as handle:
            assert json.load(handle) == variants

    def test_reads_existing_cache_verbatim(self, tmp_path):
        # When the cache exists the CSV is ignored entirely — return it as-is.
        csv_path = tmp_path / "tsuboyama.csv"
        cache_path = tmp_path / "variants.json"
        _write_csv(csv_path, self._domain_rows() + [_row("1BK2.pdb", "A1G", "GCDE", "1.5")])
        sentinel = [{"protein": "CACHED", "ddg": 9.9}]
        with open(cache_path, "w") as handle:
            json.dump(sentinel, handle)
        result = load_tsuboyama_variants(csv_path=csv_path, cache_path=cache_path)
        assert result == sentinel

    def test_conflicting_wt_sequences_raise(self, tmp_path):
        # Two "wt" rows for the same domain with DIFFERENT sequences: the
        # reference is ambiguous, so the loader refuses rather than guessing.
        rows = [
            _row("1BK2.pdb", "wt", "ACDE", "-"),
            _row("1BK2.pdb", "wt", "WCDE", "-"),   # conflicting reference
            _row("1BK2.pdb", "A1G", "GCDE", "1.0"),
        ]
        with pytest.raises(ValueError):
            _load(tmp_path, rows)

    def test_repeated_identical_wt_is_not_a_collision(self, tmp_path):
        # The same wt sequence appearing twice is a duplicate, not a conflict —
        # it must not raise, and the variant parses normally.
        rows = [
            _row("1BK2.pdb", "wt", "ACDE", "-"),
            _row("1BK2.pdb", "wt", "ACDE", "-"),   # identical; benign
            _row("1BK2.pdb", "A1G", "GCDE", "1.0"),
        ]
        variants, _ = _load(tmp_path, rows)
        assert len(variants) == 1
        assert variants[0]["wt_seq"] == "ACDE"

    def test_corrupt_cache_is_deleted_and_reparsed(self, tmp_path):
        # A truncated/corrupt cache must not crash the loader: it is deleted and
        # the CSV re-parsed, then the freshly-parsed variants overwrite the cache.
        csv_path = tmp_path / "tsuboyama.csv"
        cache_path = tmp_path / "variants.json"
        _write_csv(csv_path, self._domain_rows() + [_row("1BK2.pdb", "A1G", "GCDE", "1.5")])
        with open(cache_path, "w") as handle:
            handle.write('[{"protein": "1BK2.pdb",')  # truncated JSON

        result = load_tsuboyama_variants(csv_path=csv_path, cache_path=cache_path)

        assert len(result) == 1
        assert result[0]["mutation_code"] == "A1G"
        # The corrupt file was replaced with valid JSON matching the parse.
        with open(cache_path) as handle:
            assert json.load(handle) == result
