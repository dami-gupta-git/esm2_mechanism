"""
Tests for esm2_mech.utils.io.

Invariants:
- save_npy writes a valid .npy file readable by np.load
- save_npy is atomic: no .tmp file left on disk after success
- save_npy round-trips values exactly
- atomic_write_json writes valid JSON readable by json.load
- atomic_write_json is atomic: no .tmp file left on disk after success
- atomic_write_json round-trips dicts, lists, and nested structures
- load_json_or_discard returns None for a missing file
- load_json_or_discard round-trips valid JSON
- load_json_or_discard discards (deletes + returns None) a JSON-corrupt file
- load_json_or_discard discards a file with invalid UTF-8 bytes (UnicodeDecodeError)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from esm2_mech.utils.io import save_npy, atomic_write_json, load_json_or_discard


class TestSaveNpy:

    def test_roundtrip_1d(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        path = tmp_path / "arr.npy"
        save_npy(path, arr)
        np.testing.assert_array_equal(np.load(path), arr)

    def test_roundtrip_2d(self, tmp_path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "arr.npy"
        save_npy(path, arr)
        np.testing.assert_array_equal(np.load(path), arr)

    def test_no_tmp_file_left(self, tmp_path):
        arr = np.zeros(5)
        path = tmp_path / "arr.npy"
        save_npy(path, arr)
        assert not (tmp_path / "arr.npy.tmp").exists()

    def test_output_file_exists(self, tmp_path):
        path = tmp_path / "arr.npy"
        save_npy(path, np.ones(3))
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "arr.npy"
        save_npy(path, np.array([1.0]))
        save_npy(path, np.array([9.0]))
        assert np.load(path)[0] == pytest.approx(9.0)

    def test_preserves_dtype(self, tmp_path):
        arr = np.array([1, 2, 3], dtype=np.int16)
        path = tmp_path / "arr.npy"
        save_npy(path, arr)
        assert np.load(path).dtype == np.int16


class TestAtomicWriteJson:

    def test_roundtrip_dict(self, tmp_path):
        data = {"a": 1, "b": [1, 2, 3], "c": None}
        path = tmp_path / "out.json"
        atomic_write_json(path, data)
        assert json.loads(path.read_text()) == data

    def test_roundtrip_list(self, tmp_path):
        data = [1, "two", 3.0]
        path = tmp_path / "out.json"
        atomic_write_json(path, data)
        assert json.loads(path.read_text()) == data

    def test_no_tmp_file_left(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"x": 1})
        assert not (tmp_path / "out.json.tmp").exists()

    def test_output_file_exists(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {})
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}

    def test_nested_structure(self, tmp_path):
        data = {"outer": {"inner": [1, 2, {"deep": True}]}}
        path = tmp_path / "out.json"
        atomic_write_json(path, data)
        assert json.loads(path.read_text()) == data


class TestLoadJsonOrDiscard:

    def test_missing_returns_none(self, tmp_path):
        assert load_json_or_discard(tmp_path / "nope.json") is None

    def test_valid_roundtrip(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"a": 1}))
        assert load_json_or_discard(path) == {"a": 1}

    def test_json_corrupt_is_discarded(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"a": 1')  # truncated JSON
        assert load_json_or_discard(path) is None
        assert not path.exists()  # deleted so next run re-fetches

    def test_invalid_utf8_is_discarded(self, tmp_path):
        # A truncated write can leave invalid UTF-8, which raises
        # UnicodeDecodeError (not JSONDecodeError) during the read. It must be
        # treated as corruption: deleted + None, not propagated.
        path = tmp_path / "bytes.json"
        path.write_bytes(b'\xff\xfe{"a": 1')
        assert load_json_or_discard(path) is None
        assert not path.exists()
