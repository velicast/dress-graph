"""
Tests for the DRESS Python (pybind11) bindings.

Run from the repo root after building:
    pip install ./python
    pytest tests/python/
"""

import pytest
import numpy as np


# ── import guard ─────────────────────────────────────────────────────

dress = pytest.importorskip("dress", reason="dress extension not built")


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def triangle():
    """Unweighted triangle: 0-1, 1-2, 0-2."""
    g = dress.DRESS(3, [0, 1, 0], [1, 2, 2])
    return g


@pytest.fixture
def path():
    """Unweighted path: 0-1-2-3."""
    g = dress.DRESS(4, [0, 1, 2], [1, 2, 3])
    return g


# ── construction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_unweighted(self, triangle):
        assert triangle.n_vertices == 3
        assert triangle.n_edges == 3
        assert triangle.variant == dress.UNDIRECTED

    def test_weighted(self):
        g = dress.DRESS(3, [0, 1, 0], [1, 2, 2],
                        weights=[1.0, 2.0, 3.0])
        assert g.n_edges == 3

    def test_repr(self, triangle):
        r = repr(triangle)
        assert "n_vertices=3" in r
        assert "n_edges=3" in r

    def test_all_variants(self):
        for v in (dress.UNDIRECTED, dress.DIRECTED, dress.FORWARD, dress.BACKWARD):
            g = dress.DRESS(3, [0, 1, 0], [1, 2, 2], v)
            assert g.variant == v
            assert g.n_edges == 3

    def test_precompute_intercepts(self):
        g = dress.DRESS(3, [0, 1, 0], [1, 2, 2],
                        precompute_intercepts=True)
        assert g.n_edges == 3


# ── fitting ──────────────────────────────────────────────────────────

class TestFit:
    def test_triangle_convergence(self, triangle):
        result = triangle.fit(100, 1e-8)
        assert result.iterations > 0
        assert result.delta >= 0.0

    def test_triangle_equal_dress(self, triangle):
        triangle.fit(100, 1e-8)
        d0 = triangle.edge_dress(0)
        for e in range(triangle.n_edges):
            assert abs(triangle.edge_dress(e) - d0) < 1e-6, \
                "all edges in a triangle should have equal dress"

    def test_path_positive_dress(self, path):
        path.fit(100, 1e-6)
        for e in range(path.n_edges):
            d = path.edge_dress(e)
            assert d > 0.0, "path dress should be positive (self-loop term)"
            assert d < 2.0, "path dress should be well below 2"

    def test_path_symmetry(self, path):
        path.fit(100, 1e-6)
        # Endpoint edges (0-1 and 2-3) should be symmetric
        assert abs(path.edge_dress(0) - path.edge_dress(2)) < 1e-10

    def test_fit_result_repr(self, triangle):
        result = triangle.fit(100, 1e-8)
        r = repr(result)
        assert "iterations=" in r
        assert "delta=" in r

    def test_weighted_fit(self):
        g = dress.DRESS(3, [0, 1, 0], [1, 2, 2],
                        weights=[1.0, 2.0, 3.0])
        result = g.fit(100, 1e-6)
        assert result.iterations > 0


# ── per-element accessors ────────────────────────────────────────────

class TestAccessors:
    def test_edge_source_target(self, triangle):
        sources = [triangle.edge_source(e) for e in range(3)]
        targets = [triangle.edge_target(e) for e in range(3)]
        assert sources == [0, 1, 0]
        assert targets == [1, 2, 2]

    def test_edge_weight_unweighted(self, triangle):
        triangle.fit(1, 1.0)
        # Unweighted undirected edges get weight 2.0 (doubled)
        for e in range(3):
            assert triangle.edge_weight(e) == pytest.approx(2.0)

    def test_node_dress_after_fit(self, triangle):
        triangle.fit(100, 1e-8)
        for u in range(3):
            nd = triangle.node_dress(u)
            assert nd > 0.0, "node dress should be positive after fitting"


# ── NumPy view properties ────────────────────────────────────────────

class TestNumpyViews:
    def test_sources_array(self, triangle):
        s = triangle.sources
        assert isinstance(s, np.ndarray)
        assert s.dtype == np.int32
        assert len(s) == 3
        np.testing.assert_array_equal(s, [0, 1, 0])

    def test_targets_array(self, triangle):
        t = triangle.targets
        assert isinstance(t, np.ndarray)
        assert t.dtype == np.int32
        assert len(t) == 3
        np.testing.assert_array_equal(t, [1, 2, 2])

    def test_weights_array(self, triangle):
        triangle.fit(1, 1.0)
        w = triangle.weights
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64
        assert len(w) == 3

    def test_dress_values_array(self, triangle):
        triangle.fit(100, 1e-8)
        dv = triangle.dress_values
        assert isinstance(dv, np.ndarray)
        assert dv.dtype == np.float64
        assert len(dv) == 3
        # All dress values should be equal in a triangle
        np.testing.assert_allclose(dv, dv[0], atol=1e-6)

    def test_node_dress_values_array(self, triangle):
        triangle.fit(100, 1e-8)
        nd = triangle.node_dress_values
        assert isinstance(nd, np.ndarray)
        assert nd.dtype == np.float64
        assert len(nd) == 3
        # All nodes in a triangle should have equal dress norm
        np.testing.assert_allclose(nd, nd[0], atol=1e-6)

    def test_views_are_zero_copy(self, triangle):
        """Accessing the same property twice should return views of the same memory."""
        triangle.fit(1, 1.0)
        a = triangle.dress_values
        b = triangle.dress_values
        assert np.shares_memory(a, b)


# ── dress_fit top-level function ─────────────────────────────────────

class TestDressFit:
    """Test the primary dress_fit() functional API."""

    def test_basic(self):
        r = dress.dress_fit(3, [0, 1, 0], [1, 2, 2])
        assert isinstance(r, dress.DRESSResult)
        assert len(r.edge_dress) == 3
        assert len(r.node_dress) == 3
        assert len(r.sources) == 3
        assert len(r.targets) == 3
        assert len(r.edge_weight) == 3
        assert r.iterations > 0

    def test_triangle_value(self):
        r = dress.dress_fit(3, [0, 1, 0], [1, 2, 2],
                            max_iterations=500, epsilon=1e-12)
        assert abs(r.edge_dress[0] - 2.0) < 1e-6

    def test_weighted(self):
        r = dress.dress_fit(3, [0, 1, 0], [1, 2, 2],
                            weights=[1.0, 2.0, 3.0])
        assert r.iterations > 0

    def test_directed(self):
        r = dress.dress_fit(3, [0, 1, 0], [1, 2, 2],
                            variant=dress.DIRECTED)
        assert r.iterations >= 0

    def test_repr(self):
        r = dress.dress_fit(3, [0, 1, 0], [1, 2, 2])
        assert "DRESSResult" in repr(r)
        assert "edges=3" in repr(r)
