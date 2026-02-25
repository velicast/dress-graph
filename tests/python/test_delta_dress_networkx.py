"""
Tests for the Δ^k-DRESS NetworkX wrapper.

Run from the repo root:
    pip install ./python
    pytest tests/python/test_delta_dress_networkx.py -v
"""

import pytest

dress = pytest.importorskip("dress", reason="dress package not installed")
nx = pytest.importorskip("networkx", reason="networkx not installed")

from dress.networkx import delta_dress_graph  # noqa: E402
from dress.core import DeltaDRESSResult       # noqa: E402


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


# ── Δ^0 tests ────────────────────────────────────────────────────────

class TestDelta0:
    def test_triangle_returns_result(self, triangle):
        result = delta_dress_graph(triangle, k=0)
        assert isinstance(result, DeltaDRESSResult)
        assert result.hist_size > 0
        assert len(result.histogram) == result.hist_size

    def test_triangle_sum_equals_edges(self, triangle):
        result = delta_dress_graph(triangle, k=0)
        total = sum(result.histogram)
        assert total == 3, f"expected 3 edge values, got {total}"

    def test_triangle_top_bin(self, triangle):
        """All edges in a triangle converge to 2.0 → top bin."""
        result = delta_dress_graph(triangle, k=0)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 3

    def test_k4_sum(self, k4):
        result = delta_dress_graph(k4, k=0)
        total = sum(result.histogram)
        assert total == 6

    def test_k4_top_bin(self, k4):
        result = delta_dress_graph(k4, k=0)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 6


# ── Δ^1 tests ────────────────────────────────────────────────────────

class TestDelta1:
    def test_triangle_sum(self, triangle):
        """C(3,1)=3 subsets, each leaves 1 edge → 3 edge values."""
        result = delta_dress_graph(triangle, k=1)
        total = sum(result.histogram)
        assert total == 3

    def test_triangle_all_top_bin(self, triangle):
        """Each subgraph of triangle minus 1 vertex is a single edge (dress=2)."""
        result = delta_dress_graph(triangle, k=1)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 3

    def test_k4_sum(self, k4):
        """C(4,1)=4 subsets of K4, each gives K3 (3 edges) → 12 values."""
        result = delta_dress_graph(k4, k=1)
        total = sum(result.histogram)
        assert total == 12

    def test_k4_top_bin(self, k4):
        """All K3 edges converge to 2.0."""
        result = delta_dress_graph(k4, k=1)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 12


# ── Δ^2 on K4 ────────────────────────────────────────────────────────

class TestDelta2:
    def test_k4_sum(self, k4):
        """C(4,2)=6 subsets, each leaves 1 edge → 6 values total."""
        result = delta_dress_graph(k4, k=2)
        total = sum(result.histogram)
        assert total == 6

    def test_k4_top_bin(self, k4):
        result = delta_dress_graph(k4, k=2)
        top = result.histogram[-1] + result.histogram[-2]
        assert top == 6


# ── edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        result = delta_dress_graph(G, k=0)
        assert sum(result.histogram) == 0

    def test_k_exceeds_n(self, triangle):
        """k >= N → no valid subsets → empty histogram."""
        result = delta_dress_graph(triangle, k=3)
        assert sum(result.histogram) == 0

    def test_path_delta0(self, path4):
        result = delta_dress_graph(path4, k=0)
        total = sum(result.histogram)
        assert total == 3  # 3 edges

    def test_hist_size_formula(self, triangle):
        eps = 0.1
        result = delta_dress_graph(triangle, k=0, epsilon=eps)
        expected = int(2.0 / eps) + 1  # = 21
        assert result.hist_size == expected


# ── string-labeled nodes ──────────────────────────────────────────────

class TestStringNodes:
    def test_karate(self):
        """Works with NetworkX's built-in karate club (integer nodes)."""
        G = nx.karate_club_graph()
        result = delta_dress_graph(G, k=0)
        total = sum(result.histogram)
        assert total == G.number_of_edges()

    def test_string_labels(self):
        """Works with arbitrary string node labels."""
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        result = delta_dress_graph(G, k=0)
        total = sum(result.histogram)
        assert total == 3
