"""
Tests for the ∇^k-DRESS NetworkX wrapper.

Run from the repo root:
    pip install ./python
    pytest tests/python/test_nabla_dress_networkx.py -v
"""

import pytest

dress = pytest.importorskip("dress", reason="dress package not installed")
nx = pytest.importorskip("networkx", reason="networkx not installed")

from dress.networkx import nabla_dress_graph  # noqa: E402
from dress.core import NablaDRESSResult       # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def triangle():
    """Unweighted triangle: 0-1, 1-2, 0-2."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return G


@pytest.fixture
def k4():
    """Complete graph K4."""
    return nx.complete_graph(4)


@pytest.fixture
def path4():
    """Path graph 0-1-2-3."""
    return nx.path_graph(4)


# ── ∇^0 tests ────────────────────────────────────────────────────────

class TestNabla0:
    def test_triangle_returns_result(self, triangle):
        result = nabla_dress_graph(triangle, k=0)
        assert isinstance(result, NablaDRESSResult)
        assert result.hist_size > 0
        assert len(result.histogram) == result.hist_size

    def test_triangle_sum_equals_edges(self, triangle):
        result = nabla_dress_graph(triangle, k=0)
        total = sum(result.histogram)
        assert total == 3, f"expected 3 edge values, got {total}"

    def test_triangle_top_bin(self, triangle):
        """All edges in a triangle converge to 2.0 → top bin."""
        result = nabla_dress_graph(triangle, k=0)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 3

    def test_k4_sum(self, k4):
        result = nabla_dress_graph(k4, k=0)
        total = sum(result.histogram)
        assert total == 6

    def test_k4_top_bin(self, k4):
        result = nabla_dress_graph(k4, k=0)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 6


# ── ∇^1 tests ────────────────────────────────────────────────────────

class TestNabla1:
    def test_triangle_sum(self, triangle):
        """C(3,1)=3 subsets × 3 edges each = 9 edge values."""
        result = nabla_dress_graph(triangle, k=1)
        total = sum(result.histogram)
        assert total == 9

    def test_k4_sum(self, k4):
        """C(4,1)=4 subsets × 6 edges each = 24 edge values."""
        result = nabla_dress_graph(k4, k=1)
        total = sum(result.histogram)
        assert total == 24

    def test_path_sum(self, path4):
        """C(4,1)=4 subsets × 3 edges each = 12 edge values."""
        result = nabla_dress_graph(path4, k=1)
        total = sum(result.histogram)
        assert total == 12


# ── ∇^2 on K4 ────────────────────────────────────────────────────────

class TestNabla2:
    def test_k4_sum(self, k4):
        """C(4,2)=6 subsets × 6 edges each = 36 values total."""
        result = nabla_dress_graph(k4, k=2)
        total = sum(result.histogram)
        assert total == 36

    def test_triangle_sum(self, triangle):
        """C(3,2)=3 subsets × 3 edges each = 9 values."""
        result = nabla_dress_graph(triangle, k=2)
        total = sum(result.histogram)
        assert total == 9


# ── edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        result = nabla_dress_graph(G, k=0)
        assert sum(result.histogram) == 0

    def test_k_exceeds_n(self, triangle):
        """k >= N → no valid subsets → empty histogram."""
        result = nabla_dress_graph(triangle, k=3)
        assert sum(result.histogram) == 0

    def test_path_nabla0(self, path4):
        result = nabla_dress_graph(path4, k=0)
        total = sum(result.histogram)
        assert total == 3  # 3 edges

    def test_hist_size_formula(self, triangle):
        eps = 0.1
        result = nabla_dress_graph(triangle, k=0, epsilon=eps)
        expected = int(2.0 / eps) + 1  # = 21
        assert result.hist_size == expected


# ── nabla_weight parameter ────────────────────────────────────────────

class TestNablaWeight:
    def test_different_weights_differ(self, triangle):
        """Different nabla_weight values produce different histograms."""
        r1 = nabla_dress_graph(triangle, k=1, nabla_weight=2.0)
        r2 = nabla_dress_graph(triangle, k=1, nabla_weight=3.0)
        assert r1.histogram != r2.histogram

    def test_weight_1_same_as_nabla0_scaled(self, triangle):
        """nabla_weight=1.0 with k=1 → marking has no effect."""
        r0 = nabla_dress_graph(triangle, k=0)
        r1 = nabla_dress_graph(triangle, k=1, nabla_weight=1.0)
        # Each subset is identical to base DRESS → total = C(3,1) * 3 = 9
        assert sum(r1.histogram) == 9
        for a, b in zip(r0.histogram, r1.histogram):
            assert b == 3 * a


# ── weighted graph ────────────────────────────────────────────────────

class TestWeighted:
    def test_weighted_triangle(self):
        """Weighted triangle passes weights through to nabla."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=0.5)
        G.add_edge(0, 2, weight=0.5)
        result = nabla_dress_graph(G, k=0)
        total = sum(result.histogram)
        assert total == 3


# ── string-labeled nodes ──────────────────────────────────────────────

class TestStringNodes:
    def test_karate(self):
        """Works with NetworkX's built-in karate club (integer nodes)."""
        G = nx.karate_club_graph()
        result = nabla_dress_graph(G, k=0)
        total = sum(result.histogram)
        assert total == G.number_of_edges()

    def test_string_labels(self):
        """Works with arbitrary string node labels."""
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        result = nabla_dress_graph(G, k=0)
        total = sum(result.histogram)
        assert total == 3
