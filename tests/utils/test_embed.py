"""
Tests for utils/embed.py pure-logic functions.

Covers:
- unpack_run_data: deltas_mean = emb_mut_mean - emb_wt_mean
- unpack_run_data: deltas_pos  = emb_mut_pos  - emb_wt_pos
- unpack_run_data: original keys are preserved in returned dict
- unpack_run_data: zero delta when WT and mutant embeddings are identical
- load_gene_delta: groups delta vectors by gene (upper-cased)
- load_gene_delta: genes not in variants produce no entry
- load_gene_delta: multiple variants for the same gene accumulate all deltas
- load_gene_delta: delta vector equals mut - wt at that row index
- _flush_checkpoint: writes all four .npy arrays plus valid_variants.json
- _flush_checkpoint: atomic — no .tmp file left after success
- _flush_checkpoint: written arrays round-trip with correct shape/values
- _flush_checkpoint: valid_variants.json holds the provided slice
"""

import json

import numpy as np
import pytest

from esm2_mech.utils.embed import _flush_checkpoint, load_gene_delta, unpack_run_data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _data(n=4, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "emb_wt_mean":  rng.random((n, d)).astype(np.float32),
        "emb_mut_mean": rng.random((n, d)).astype(np.float32),
        "emb_wt_pos":   rng.random((n, d)).astype(np.float32),
        "emb_mut_pos":  rng.random((n, d)).astype(np.float32),
    }


def _write_npy(path, arr):
    np.save(path, arr)
    return path


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# unpack_run_data
# ---------------------------------------------------------------------------

class TestUnpackRunData:

    def test_deltas_mean_is_mut_minus_wt(self):
        data = _data()
        result = unpack_run_data(data)
        expected = data["emb_mut_mean"] - data["emb_wt_mean"]
        np.testing.assert_allclose(result["deltas_mean"], expected)

    def test_deltas_pos_is_mut_minus_wt(self):
        data = _data()
        result = unpack_run_data(data)
        expected = data["emb_mut_pos"] - data["emb_wt_pos"]
        np.testing.assert_allclose(result["deltas_pos"], expected)

    def test_original_keys_preserved(self):
        data = _data()
        result = unpack_run_data(data)
        for key in ("emb_wt_mean", "emb_mut_mean", "emb_wt_pos", "emb_mut_pos"):
            assert key in result

    def test_zero_delta_when_identical(self):
        n, d = 3, 5
        arr = np.ones((n, d), dtype=np.float32)
        data = {
            "emb_wt_mean": arr,
            "emb_mut_mean": arr,
            "emb_wt_pos": arr,
            "emb_mut_pos": arr,
        }
        result = unpack_run_data(data)
        np.testing.assert_array_equal(result["deltas_mean"], np.zeros((n, d)))
        np.testing.assert_array_equal(result["deltas_pos"], np.zeros((n, d)))

    def test_output_shape_matches_input(self):
        n, d = 7, 16
        data = _data(n=n, d=d)
        result = unpack_run_data(data)
        assert result["deltas_mean"].shape == (n, d)
        assert result["deltas_pos"].shape == (n, d)

    def test_extra_keys_passed_through(self):
        data = _data()
        data["variants"] = [{"gene": "BRCA1"}]
        result = unpack_run_data(data)
        assert result["variants"] == [{"gene": "BRCA1"}]


# ---------------------------------------------------------------------------
# load_gene_delta
# ---------------------------------------------------------------------------

class TestLoadGeneDelta:

    def _setup(self, tmp_path, variants, wt, mut):
        vpath = _write_json(tmp_path / "variants.json", variants)
        wt_path = _write_npy(tmp_path / "wt.npy", wt)
        mut_path = _write_npy(tmp_path / "mut.npy", mut)
        return vpath, wt_path, mut_path

    def test_delta_equals_mut_minus_wt(self, tmp_path):
        wt = np.array([[1.0, 2.0]], dtype=np.float32)
        mut = np.array([[4.0, 6.0]], dtype=np.float32)
        variants = [{"gene": "BRCA1"}]
        vpath, wt_path, mut_path = self._setup(tmp_path, variants, wt, mut)
        result = load_gene_delta(vpath, wt_path, mut_path)
        np.testing.assert_allclose(result["BRCA1"][0], [3.0, 4.0])

    def test_gene_names_uppercased(self, tmp_path):
        wt = np.array([[1.0]], dtype=np.float32)
        mut = np.array([[2.0]], dtype=np.float32)
        variants = [{"gene": "brca1"}]
        vpath, wt_path, mut_path = self._setup(tmp_path, variants, wt, mut)
        result = load_gene_delta(vpath, wt_path, mut_path)
        assert "BRCA1" in result
        assert "brca1" not in result

    def test_multiple_variants_same_gene_accumulate(self, tmp_path):
        wt = np.array([[1.0], [2.0]], dtype=np.float32)
        mut = np.array([[3.0], [5.0]], dtype=np.float32)
        variants = [{"gene": "BRCA1"}, {"gene": "BRCA1"}]
        vpath, wt_path, mut_path = self._setup(tmp_path, variants, wt, mut)
        result = load_gene_delta(vpath, wt_path, mut_path)
        assert len(result["BRCA1"]) == 2
        np.testing.assert_allclose(result["BRCA1"][0], [2.0])
        np.testing.assert_allclose(result["BRCA1"][1], [3.0])

    def test_different_genes_get_separate_entries(self, tmp_path):
        wt = np.array([[1.0], [2.0]], dtype=np.float32)
        mut = np.array([[3.0], [6.0]], dtype=np.float32)
        variants = [{"gene": "BRCA1"}, {"gene": "TP53"}]
        vpath, wt_path, mut_path = self._setup(tmp_path, variants, wt, mut)
        result = load_gene_delta(vpath, wt_path, mut_path)
        assert "BRCA1" in result
        assert "TP53" in result
        np.testing.assert_allclose(result["BRCA1"][0], [2.0])
        np.testing.assert_allclose(result["TP53"][0], [4.0])

    def test_each_delta_aligned_to_correct_row(self, tmp_path):
        d = 4
        rng = np.random.default_rng(42)
        wt = rng.random((3, d)).astype(np.float32)
        mut = rng.random((3, d)).astype(np.float32)
        variants = [{"gene": "A"}, {"gene": "B"}, {"gene": "A"}]
        vpath, wt_path, mut_path = self._setup(tmp_path, variants, wt, mut)
        result = load_gene_delta(vpath, wt_path, mut_path)
        np.testing.assert_allclose(result["A"][0], mut[0] - wt[0])
        np.testing.assert_allclose(result["B"][0], mut[1] - wt[1])
        np.testing.assert_allclose(result["A"][1], mut[2] - wt[2])


# ---------------------------------------------------------------------------
# _flush_checkpoint
# ---------------------------------------------------------------------------

class TestFlushCheckpoint:

    ARRAY_NAMES = [
        "embeddings_wt_mean",
        "embeddings_mut_mean",
        "embeddings_wt_pos",
        "embeddings_mut_pos",
    ]

    def _flush(self, tmp_path, n=3, d=4, seed=0):
        rng = np.random.default_rng(seed)
        wt_mean = [rng.random(d).astype(np.float32) for _ in range(n)]
        mut_mean = [rng.random(d).astype(np.float32) for _ in range(n)]
        wt_pos = [rng.random(d).astype(np.float32) for _ in range(n)]
        mut_pos = [rng.random(d).astype(np.float32) for _ in range(n)]
        valid = [{"gene": f"G{i}"} for i in range(n)]
        _flush_checkpoint(
            str(tmp_path), wt_mean, mut_mean, wt_pos, mut_pos, valid, n
        )
        return wt_mean, mut_mean, wt_pos, mut_pos, valid

    def test_all_arrays_written(self, tmp_path):
        self._flush(tmp_path)
        for name in self.ARRAY_NAMES:
            assert (tmp_path / f"{name}.npy").exists()

    def test_valid_variants_json_written(self, tmp_path):
        self._flush(tmp_path)
        assert (tmp_path / "valid_variants.json").exists()

    def test_no_tmp_file_left(self, tmp_path):
        self._flush(tmp_path)
        assert not (tmp_path / "valid_variants.json.tmp").exists()
        for name in self.ARRAY_NAMES:
            assert not (tmp_path / f"{name}.npy.tmp").exists()

    def test_array_shape_and_values(self, tmp_path):
        wt_mean, mut_mean, wt_pos, mut_pos, _ = self._flush(tmp_path, n=3, d=4)
        loaded = np.load(tmp_path / "embeddings_wt_mean.npy")
        assert loaded.shape == (3, 4)
        np.testing.assert_allclose(loaded, np.stack(wt_mean))
        np.testing.assert_allclose(
            np.load(tmp_path / "embeddings_mut_pos.npy"), np.stack(mut_pos)
        )

    def test_valid_variants_content(self, tmp_path):
        _, _, _, _, valid = self._flush(tmp_path, n=3)
        written = json.loads((tmp_path / "valid_variants.json").read_text())
        assert written == valid
