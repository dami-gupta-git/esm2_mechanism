"""
Tests for esm2_mech.experiments.stability.build_domain_families.

hmmscan itself is never invoked: tests that exercise run_hmmscan monkeypatch
shutil.which / os.path.exists / subprocess.run, with the fake subprocess.run
writing a synthetic --tblout file so the real Pfam database and binary are not
needed.

Invariants:
- _wt_sequences: one WT sequence per domain, first occurrence wins
- run_hmmscan: best (lowest E-value) Pfam hit per domain is chosen
- run_hmmscan: Pfam accession version suffix is stripped (PF00018.24 -> PF00018)
- run_hmmscan: domains with no hit are absent from the map (orphans)
- run_hmmscan: comment/blank tblout lines are ignored
- run_hmmscan: missing hmmscan binary / db / pressed index raise clearly
- build_family_map: writes the {domain: PfamID} map to the cache path
- build_family_map: orphan domains are excluded from the written map
"""

import json
import os

import pytest

import esm2_mech.experiments.stability.build_domain_families as bdf
from esm2_mech.experiments.stability.build_domain_families import (
    _wt_sequences,
    run_hmmscan,
    build_family_map,
)


# ---------------------------------------------------------------------------
# _wt_sequences
# ---------------------------------------------------------------------------

class TestWtSequences:

    def test_one_sequence_per_domain_first_wins(self):
        variants = [
            {"protein": "1BK2.pdb", "wt_seq": "ACDE"},
            {"protein": "1BK2.pdb", "wt_seq": "WXYZ"},   # later row ignored
            {"protein": "1A0N.pdb", "wt_seq": "MKLP"},
        ]
        assert _wt_sequences(variants) == {"1BK2.pdb": "ACDE", "1A0N.pdb": "MKLP"}

    def test_empty(self):
        assert _wt_sequences([]) == {}


# ---------------------------------------------------------------------------
# run_hmmscan — fake-subprocess harness
# ---------------------------------------------------------------------------

def _install_fake_hmmscan(monkeypatch, tblout_text, db_present=True, pressed=True):
    """Patch the external dependencies of run_hmmscan.

    subprocess.run writes tblout_text to the --tblout path the function passes,
    standing in for a real hmmscan invocation.
    """
    monkeypatch.setattr(bdf.shutil, "which", lambda name: "/usr/bin/hmmscan")

    real_exists = os.path.exists

    def fake_exists(path):
        if path.endswith(".h3m"):
            return pressed
        if str(path).endswith("Pfam-A.hmm") or "hmm_db" in str(path):
            return db_present
        return real_exists(path)

    monkeypatch.setattr(bdf.os.path, "exists", fake_exists)

    def fake_run(cmd, check=False, **kwargs):
        tblout_path = cmd[cmd.index("--tblout") + 1]
        with open(tblout_path, "w") as handle:
            handle.write(tblout_text)
        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(bdf.subprocess, "run", fake_run)


# A minimal --tblout: real hmmscan emits 18+ whitespace columns; run_hmmscan only
# reads col 2 (accession), col 3 (query/domain), col 5 (full-seq E-value).
def _tbl_line(pfam_name, pfam_acc, domain, evalue):
    cols = [pfam_name, pfam_acc, domain, "-", str(evalue), "120.0"] + ["-"] * 12
    return " ".join(cols)


class TestRunHmmscan:

    def test_best_evalue_hit_wins_and_version_stripped(self, monkeypatch, tmp_path):
        tblout = "\n".join([
            "# a comment header line",
            _tbl_line("SH3_1", "PF00018.24", "1BK2.pdb", "1e-2"),
            _tbl_line("SH3_2", "PF14604.9", "1BK2.pdb", "1e-30"),   # better hit
            _tbl_line("Fn3", "PF00041.25", "1A0N.pdb", "3e-12"),
            "",
        ]) + "\n"
        _install_fake_hmmscan(monkeypatch, tblout)
        result = run_hmmscan({"1BK2.pdb": "AAAA", "1A0N.pdb": "CCCC"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))
        assert result == {"1BK2.pdb": "PF14604", "1A0N.pdb": "PF00041"}

    def test_domain_with_no_hit_is_absent(self, monkeypatch, tmp_path):
        tblout = _tbl_line("SH3", "PF00018.24", "1BK2.pdb", "1e-30") + "\n"
        _install_fake_hmmscan(monkeypatch, tblout)
        # 1A0N.pdb has no line -> orphan -> absent from the map.
        result = run_hmmscan({"1BK2.pdb": "AAAA", "1A0N.pdb": "CCCC"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))
        assert result == {"1BK2.pdb": "PF00018"}
        assert "1A0N.pdb" not in result

    def test_only_comments_yields_empty_map(self, monkeypatch, tmp_path):
        _install_fake_hmmscan(monkeypatch, "# header only\n#\n\n")
        result = run_hmmscan({"1BK2.pdb": "AAAA"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))
        assert result == {}

    def test_missing_binary_raises(self, monkeypatch, tmp_path):
        _install_fake_hmmscan(monkeypatch, "")
        monkeypatch.setattr(bdf.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError):
            run_hmmscan({"1BK2.pdb": "AAAA"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))

    def test_missing_db_raises(self, monkeypatch, tmp_path):
        _install_fake_hmmscan(monkeypatch, "", db_present=False)
        with pytest.raises(FileNotFoundError):
            run_hmmscan({"1BK2.pdb": "AAAA"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))

    def test_unpressed_db_raises(self, monkeypatch, tmp_path):
        _install_fake_hmmscan(monkeypatch, "", pressed=False)
        with pytest.raises(FileNotFoundError):
            run_hmmscan({"1BK2.pdb": "AAAA"}, hmm_db=str(tmp_path / "Pfam-A.hmm"))


# ---------------------------------------------------------------------------
# build_family_map
# ---------------------------------------------------------------------------

class TestBuildFamilyMap:

    def test_writes_map_and_excludes_orphans(self, monkeypatch, tmp_path):
        variants = [
            {"protein": "1BK2.pdb", "wt_seq": "AAAA"},
            {"protein": "1A0N.pdb", "wt_seq": "CCCC"},  # will be an orphan
        ]
        # Stub the hmmscan layer: only 1BK2.pdb gets a family.
        monkeypatch.setattr(bdf, "run_hmmscan", lambda wt_seqs: {"1BK2.pdb": "PF00018"})
        out_path = tmp_path / "domain_families.json"

        family_map = build_family_map(variants=variants, out_path=out_path)

        assert family_map == {"1BK2.pdb": "PF00018"}
        assert "1A0N.pdb" not in family_map
        with open(out_path) as handle:
            assert json.load(handle) == {"1BK2.pdb": "PF00018"}
