"""
Tests for the pure-Python DRESS implementation (dress.core).

Run from the repo root:
    pytest tests/python/test_dress_core.py -v
"""

import math
import pytest

from dress.core import DRESS, DRESSResult, Variant, UNDIRECTED, DIRECTED, FORWARD, BACKWARD


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def triangle():
    """Unweighted triangle: 0-1, 1-2, 0-2."""
    return DRESS(3, [0, 1, 0], [1, 2, 2])


@pytest.fixture
def path():
    """Unweighted path: 0-1-2-3."""
    return DRESS(4, [0, 1, 2], [1, 2, 3])


@pytest.fixture
def k4():
    """Complete graph K4."""
    return DRESS(4, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])


@pytest.fixture
def star():
    """Star graph: centre=0, leaves=1,2,3,4."""
    return DRESS(5, [0, 0, 0, 0], [1, 2, 3, 4])


# ── construction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_unweighted(self, triangle):
        r = triangle.fit()
        assert len(r.sources) == 3
        assert len(r.targets) == 3
        assert len(r.edge_dress) == 3

    def test_weighted(self):
        g = DRESS(3, [0, 1, 0], [1, 2, 2], weights=[1.0, 2.0, 3.0])
        r = g.fit()
        assert len(r.edge_dress) == 3

    def test_all_variants(self):
        for v in (UNDIRECTED, DIRECTED, FORWARD, BACKWARD):
            g = DRESS(3, [0, 1, 0], [1, 2, 2], variant=v)
            r = g.fit()
            assert len(r.edge_dress) == 3

    def test_bad_lengths(self):
        with pytest.raises(ValueError):
            DRESS(3, [0, 1], [1])

    def test_bad_weights_length(self):
        with pytest.raises(ValueError):
            DRESS(3, [0, 1], [1, 2], weights=[1.0])

    def test_repr(self, triangle):
        r = triangle.fit()
        s = repr(r)
        assert "DRESSResult" in s
        assert "edges=3" in s
        assert "iterations=" in s


# ── fitting ──────────────────────────────────────────────────────────

class TestFit:
    def test_triangle_convergence(self, triangle):
        r = triangle.fit(max_iterations=100, epsilon=1e-8)
        assert r.iterations > 0
        assert r.delta >= 0.0

    def test_triangle_equal_dress(self, triangle):
        r = triangle.fit(max_iterations=100, epsilon=1e-8)
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6, \
                "all edges in a triangle should have equal dress"

    def test_path_positive_dress(self, path):
        r = path.fit()
        for d in r.edge_dress:
            assert d > 0.0, "path dress should be positive"
            assert d < 2.0, "path dress should be below 2"

    def test_path_symmetry(self, path):
        """Endpoint edges (0-1 and 2-3) should be equal by symmetry."""
        r = path.fit()
        assert abs(r.edge_dress[0] - r.edge_dress[2]) < 1e-10

    def test_k4_equal_dress(self, k4):
        """All edges in K4 should have equal dress by symmetry."""
        r = k4.fit()
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6

    def test_k4_node_norms_equal(self, k4):
        """All node norms in K4 should be equal."""
        r = k4.fit()
        n0 = r.node_dress[0]
        for n in r.node_dress:
            assert abs(n - n0) < 1e-6

    def test_star_dress(self, star):
        """Star edges should all have equal dress (by symmetry)."""
        r = star.fit()
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6

    def test_weighted_fit(self):
        g = DRESS(3, [0, 1, 0], [1, 2, 2], weights=[1.0, 2.0, 3.0])
        r = g.fit()
        assert r.iterations > 0

    def test_single_edge(self):
        """Graph with one edge should converge immediately."""
        g = DRESS(2, [0], [1])
        r = g.fit()
        for d in r.edge_dress:
            assert d > 0.0

    def test_convergence_epsilon(self, triangle):
        """Small epsilon should yield small delta."""
        r = triangle.fit(max_iterations=500, epsilon=1e-12)
        assert r.delta < 1e-12


# ── dress values bounded ─────────────────────────────────────────────

class TestBounds:
    def test_dress_in_0_2(self, triangle):
        r = triangle.fit()
        for d in r.edge_dress:
            assert 0.0 <= d <= 2.0 + 1e-9

    def test_path_dress_in_0_2(self, path):
        r = path.fit()
        for d in r.edge_dress:
            assert 0.0 <= d <= 2.0 + 1e-9

    def test_node_dress_positive(self, triangle):
        r = triangle.fit()
        for n in r.node_dress:
            assert n >= 2.0, "node norm >= 2 (self-loop contributes 4.0 under sqrt)"


# ── determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_result_twice(self, triangle):
        g1 = DRESS(3, [0, 1, 0], [1, 2, 2])
        r1 = g1.fit()
        g2 = DRESS(3, [0, 1, 0], [1, 2, 2])
        r2 = g2.fit()
        for a, b in zip(r1.edge_dress, r2.edge_dress):
            assert a == b


# ── edge weight ──────────────────────────────────────────────────────

class TestEdgeWeight:
    def test_unweighted_undirected_weight(self, triangle):
        """Undirected unweighted edges have combined weight 2.0."""
        r = triangle.fit()
        for w in r.edge_weight:
            assert w == pytest.approx(2.0)


# ── directed variants ────────────────────────────────────────────────

class TestDirectedVariants:
    def _make_directed(self, variant):
        # 0->1, 1->2, 0->2
        return DRESS(3, [0, 1, 0], [1, 2, 2], variant=variant)

    def test_forward_fits(self):
        g = self._make_directed(FORWARD)
        r = g.fit()
        assert r.iterations >= 0

    def test_backward_fits(self):
        g = self._make_directed(BACKWARD)
        r = g.fit()
        assert r.iterations >= 0

    def test_directed_fits(self):
        g = self._make_directed(DIRECTED)
        r = g.fit()
        assert r.iterations >= 0

    def test_directed_weight_reciprocal(self):
        """Reciprocal edges: combined weight = w(u->v) + w(v->u)."""
        g = DRESS(2, [0, 1], [1, 0], weights=[3.0, 5.0], variant=DIRECTED)
        r = g.fit()
        assert r.edge_weight[0] == pytest.approx(8.0)

    def test_forward_weight(self):
        """Forward variant: combined weight = w(u->v)."""
        g = DRESS(2, [0], [1], weights=[3.0], variant=FORWARD)
        r = g.fit()
        assert r.edge_weight[0] == pytest.approx(3.0)

    def test_backward_weight(self):
        """Backward variant: combined weight = w(u->v) used as w(v<-u)."""
        g = DRESS(2, [0], [1], weights=[3.0], variant=BACKWARD)
        r = g.fit()
        assert r.edge_weight[0] == pytest.approx(3.0)


# ── cross-validation with C binding values ────────────────────────

class TestCrossValidation:
    """Compare pure Python results with known values from the C implementation."""

    def test_triangle_dress_value(self):
        """Triangle dress converges to 2.0 (all pairs maximally similar)."""
        g = DRESS(3, [0, 1, 0], [1, 2, 2])
        r = g.fit(max_iterations=500, epsilon=1e-12)
        d = r.edge_dress[0]
        assert abs(d - 2.0) < 1e-6, f"triangle dress = {d}"

    def test_path_endpoint_edges_larger(self):
        """In a 4-path, endpoint edges (0-1, 2-3) have higher dress
        than the middle edge (1-2) because higher-degree nodes have
        larger norms that dilute the middle edge's similarity."""
        g = DRESS(4, [0, 1, 2], [1, 2, 3])
        r = g.fit(max_iterations=500, epsilon=1e-12)
        # edge 0: 0-1, edge 1: 1-2, edge 2: 2-3
        assert r.edge_dress[0] > r.edge_dress[1], \
            "endpoint edge should have higher dress than middle"
