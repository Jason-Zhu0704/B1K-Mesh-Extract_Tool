"""Tests for b1k_extractor (no OmniGibson required — unit tests only)."""

import numpy as np
import pytest
from b1k_extractor.extractor import _triangulate


def test_triangulate_triangles():
    fi = np.array([0, 1, 2,  3, 4, 5])
    fc = np.array([3, 3])
    tris = _triangulate(fi, fc)
    assert tris.shape == (2, 3)
    assert list(tris[0]) == [0, 1, 2]
    assert list(tris[1]) == [3, 4, 5]


def test_triangulate_quad():
    # quad [0,1,2,3] → fan tris [0,1,2] and [0,2,3]
    fi = np.array([0, 1, 2, 3])
    fc = np.array([4])
    tris = _triangulate(fi, fc)
    assert tris.shape == (2, 3)
    assert list(tris[0]) == [0, 1, 2]
    assert list(tris[1]) == [0, 2, 3]


def test_triangulate_pentagon():
    fi = np.array([0, 1, 2, 3, 4])
    fc = np.array([5])
    tris = _triangulate(fi, fc)
    assert tris.shape == (3, 3)


def test_triangulate_empty():
    fi = np.array([], dtype=np.int32)
    fc = np.array([], dtype=np.int32)
    tris = _triangulate(fi, fc)
    assert tris.shape[1] == 3
    assert len(tris) == 0


def test_extraction_result_repr():
    """ExtractionResult repr should not raise."""
    import trimesh
    from b1k_extractor.extractor import ExtractionResult
    cube = trimesh.creation.box()
    r = ExtractionResult("bed", "fnzxxr", visual=cube, collision_hull=cube)
    assert "bed/fnzxxr" in repr(r)


def test_extraction_result_extents():
    import trimesh
    from b1k_extractor.extractor import ExtractionResult
    cube = trimesh.creation.box(extents=[2.0, 1.5, 0.7])
    r = ExtractionResult("bed", "x", visual=cube, collision_hull=cube)
    ext = r.visual_extents
    assert ext is not None
    np.testing.assert_allclose(ext, [2.0, 1.5, 0.7], atol=1e-6)


def test_index_roundtrip(tmp_path):
    from b1k_extractor.index import load_index, save_index, update_index_entry
    idx = {}
    update_index_entry(idx, "bed", "fnzxxr",
                       written_files={"visual.obj": tmp_path / "v.obj"},
                       metadata={"bbox_size": [2.0, 1.5, 0.7]})
    idx_path = tmp_path / "index.json"
    save_index(idx, idx_path)
    loaded = load_index(idx_path)
    assert "bed-fnzxxr" in loaded
    assert loaded["bed-fnzxxr"]["bbox_size"] == [2.0, 1.5, 0.7]
