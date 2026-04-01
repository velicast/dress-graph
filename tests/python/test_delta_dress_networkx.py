"""
Tests for the Δ^k-DRESS NetworkX wrapper.

Run from the repo root:
    pip install ./python
    pytest tests/python/test_delta_dress_networkx.py -v
"""

import pytest
import math

dress = pytest.importorskip("dress", reason="dress package not installed")
nx = pytest.importorskip("networkx", reason="networkx not installed")

from dress.networkx import delta_fit as delta_dress_graph  # noqa: E402
from dress.core import DeltaDRESSResult       # noqa: E402

EPS = 1e-3


def get_count_near(hist, value, tol=0.1):
    """Sum counts for keys in [value-tol, value+tol]."""
    total = 0
    for hist_value, count in hist:
        if abs(hist_value - value) < tol:
            total += count
    return total


def hist_total(hist):
    return sum(count for _, count in hist)


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
        # Histogram should be non-empty for a graph with edges
        assert len(result.histogram) > 0

    def test_triangle_sum_equals_edges(self, triangle):
        result = delta_dress_graph(triangle, k=0)
        total = hist_total(result.histogram)
        assert total == 3, f"expected 3 edge values, got {total}"

    def test_triangle_top_bin(self, triangle):
        """All edges in a triangle converge to 2.0."""
        result = delta_dress_graph(triangle, k=0)
        # Check for values near 2.0
        count = get_count_near(result.histogram, 2.0)
        assert count == 3

    def test_k4_sum(self, k4):
        result = delta_dress_graph(k4, k=0)
        total = hist_total(result.histogram)
        assert total == 6

    def test_k4_top_bin(self, k4):
        result = delta_dress_graph(k4, k=0)
        count = get_count_near(result.histogram, 2.0)
        assert count == 6


# ── Δ^1 tests ────────────────────────────────────────────────────────

class TestDelta1:
    def test_triangle_sum(self, triangle):
        """C(3,1)=3 subsets, each leaves 1 edge → 3 edge values."""
        result = delta_dress_graph(triangle, k=1)
        total = hist_total(result.histogram)
        assert total == 3

    def test_triangle_all_top_bin(self, triangle):
        """Each subgraph of triangle minus 1 vertex is a single edge (dress=2)."""
        result = delta_dress_graph(triangle, k=1)
        count = get_count_near(result.histogram, 2.0)
        assert count == 3

    def test_k4_sum(self, k4):
        """C(4,1)=4 subsets of K4, each gives K3 (3 edges) → 12 values."""
        result = delta_dress_graph(k4, k=1)
        total = hist_total(result.histogram)
        assert total == 12

    def test_k4_top_bin(self, k4):
        """All K3 edges converge to 2.0."""
        result = delta_dress_graph(k4, k=1)
        count = get_count_near(result.histogram, 2.0)
        assert count == 12


# ── Δ^2 on K4 ────────────────────────────────────────────────────────

class TestDelta2:
    def test_k4_sum(self, k4):
        """C(4,2)=6 subsets, each leaves 1 edge → 6 values total."""
        result = delta_dress_graph(k4, k=2)
        total = hist_total(result.histogram)
        assert total == 6

    def test_k4_top_bin(self, k4):
        result = delta_dress_graph(k4, k=2)
        count = get_count_near(result.histogram, 2.0)
        assert count == 6


# ── edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        result = delta_dress_graph(G, k=0)
        assert hist_total(result.histogram) == 0

    def test_k_exceeds_n(self, triangle):
        """k >= N → no valid subsets → empty histogram."""
        result = delta_dress_graph(triangle, k=3)
        assert hist_total(result.histogram) == 0

    def test_path_delta0(self, path4):
        result = delta_dress_graph(path4, k=0)
        total = hist_total(result.histogram)
        assert total == 3  # 3 edges

    def test_weighted_high_values(self):
        """Weighted triangle with non-uniform weights → values > 2.0."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=10.0)
        G.add_edge(0, 2, weight=1.0)
        result = delta_dress_graph(G, k=0, epsilon=1e-3)
        
        # We expect some values to be larger than 2.0 because of the weight 10.0
        # Just check that we have keys > 2.001
        high_values = [hist_value for hist_value, _ in result.histogram if hist_value > 2.001]
        assert len(high_values) > 0
        assert hist_total(result.histogram) == 3


# ── string-labeled nodes ──────────────────────────────────────────────

class TestMultisets:
    def test_disabled_by_default(self, triangle):
        r = delta_dress_graph(triangle, k=0, epsilon=EPS)
        assert r.multisets is None

    def test_delta0_triangle(self, triangle):
        r = delta_dress_graph(triangle, k=0, epsilon=EPS, keep_multisets=True)
        assert r.num_subgraphs == 1
        import numpy as np
        ms = np.asarray(r.multisets)
        assert ms.shape == (1, 3)
        for v in ms.flat:
            assert abs(v - 2.0) < EPS

    def test_delta1_k3_nan(self, triangle):
        r = delta_dress_graph(triangle, k=1, epsilon=EPS, keep_multisets=True)
        assert r.num_subgraphs == 3
        import numpy as np
        ms = np.asarray(r.multisets)
        assert ms.shape == (3, 3)
        for row_i in range(3):
            # Each row should have exactly 2 NaNs (edges connected to removed vertex)
            nans = np.isnan(ms[row_i]).sum()
            assert nans == 2


class TestStringNodes:
    def test_karate(self):
        """Works with NetworkX's built-in karate club (integer nodes)."""
        G = nx.karate_club_graph()
        result = delta_dress_graph(G, k=0)
        total = hist_total(result.histogram)
        assert total == G.number_of_edges()

    def test_string_labels(self):
        """Works with arbitrary string vertex labels."""
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        result = delta_dress_graph(G, k=0)
        total = hist_total(result.histogram)
        # Triangle
        assert total == 3
