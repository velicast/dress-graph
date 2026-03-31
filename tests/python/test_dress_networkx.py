"""
Tests for the DRESS NetworkX wrapper (dress_graph).

Run from the repo root:
    pip install ./python
    pytest tests/python/test_dress_networkx.py -v
"""

import pytest

dress = pytest.importorskip("dress", reason="dress package not installed")
nx = pytest.importorskip("networkx", reason="networkx not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

from dress.networkx import fit as dress_graph       # noqa: E402
from dress.networkx import NxDRESS            # noqa: E402
from dress.core import DRESSResult           # noqa: E402


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


@pytest.fixture
def single_edge():
    """Graph with a single edge."""
    G = nx.Graph()
    G.add_edge(0, 1)
    return G


# ── return type and basic structure ──────────────────────────────────

class TestReturnType:
    def test_returns_dress_result(self, triangle):
        result = dress_graph(triangle)
        assert isinstance(result, DRESSResult)

    def test_has_required_fields(self, triangle):
        result = dress_graph(triangle)
        assert hasattr(result, "sources")
        assert hasattr(result, "targets")
        assert hasattr(result, "edge_dress")
        assert hasattr(result, "edge_weight")
        assert hasattr(result, "node_dress")
        assert hasattr(result, "iterations")
        assert hasattr(result, "delta")

    def test_edge_count_matches(self, triangle):
        result = dress_graph(triangle)
        assert len(result.edge_dress) == triangle.number_of_edges()
        assert len(result.sources) == triangle.number_of_edges()
        assert len(result.targets) == triangle.number_of_edges()
        assert len(result.edge_weight) == triangle.number_of_edges()

    def test_node_count_matches(self, triangle):
        result = dress_graph(triangle)
        assert len(result.node_dress) == triangle.number_of_nodes()


# ── value bounds ─────────────────────────────────────────────────────

class TestBounds:
    def test_edge_values_bounded(self, k4):
        result = dress_graph(k4)
        for v in result.edge_dress:
            assert -1e-9 <= v <= 2.0 + 1e-9, f"edge value {v} out of [0, 2]"

    def test_node_norms_positive(self, k4):
        result = dress_graph(k4)
        for v in result.node_dress:
            assert v > 0.0, f"node norm {v} should be positive"


# ── convergence ──────────────────────────────────────────────────────

class TestConvergence:
    def test_converges_within_max_iter(self, triangle):
        result = dress_graph(triangle, max_iterations=100, epsilon=1e-6)
        assert result.iterations < 100
        assert result.delta < 1e-6

    def test_k4_converges(self, k4):
        result = dress_graph(k4, max_iterations=100, epsilon=1e-6)
        assert result.iterations < 100


# ── known values ─────────────────────────────────────────────────────

class TestKnownValues:
    def test_triangle_all_edges_equal_2(self, triangle):
        """All edges in a triangle converge to 2.0 (complete graph)."""
        result = dress_graph(triangle)
        for v in result.edge_dress:
            assert abs(v - 2.0) < 1e-4, f"expected ~2.0, got {v}"

    def test_k4_all_edges_equal(self, k4):
        """All edges in K4 converge to the same value (vertex-transitive)."""
        result = dress_graph(k4)
        vals = result.edge_dress
        assert max(vals) - min(vals) < 1e-6

    def test_single_edge_converges_to_2(self, single_edge):
        """A single edge converges to 2.0 (self-similarity)."""
        result = dress_graph(single_edge)
        assert abs(result.edge_dress[0] - 2.0) < 1e-4

    def test_path_bridge_lower_than_leaf(self, path4):
        """In a path, the middle edge (bridge) scores differently from leaf edges."""
        result = dress_graph(path4)
        vals = sorted(result.edge_dress)
        # Path has 3 edges: two leaf edges and one central bridge;
        # they should not all be the same value
        assert max(vals) - min(vals) > 0.01


# ── set_attributes ───────────────────────────────────────────────────

class TestSetAttributes:
    def test_dress_edge_attribute_written(self, triangle):
        dress_graph(triangle, set_attributes=True)
        for u, v in triangle.edges():
            assert "dress" in triangle[u][v], f"edge ({u},{v}) missing 'dress'"

    def test_dress_norm_node_attribute_written(self, triangle):
        dress_graph(triangle, set_attributes=True)
        for n in triangle.nodes():
            assert "dress_norm" in triangle.nodes[n], f"node {n} missing 'dress_norm'"

    def test_edge_values_match_result(self, k4):
        result = dress_graph(k4, set_attributes=True)
        attr_vals = sorted(k4[u][v]["dress"] for u, v in k4.edges())
        result_vals = sorted(result.edge_dress)
        for a, r in zip(attr_vals, result_vals):
            assert abs(a - r) < 1e-10

    def test_without_set_attributes_no_change(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        dress_graph(G, set_attributes=False)
        for u, v in G.edges():
            assert "dress" not in G[u][v]


# ── weighted graphs ──────────────────────────────────────────────────

class TestWeighted:
    def test_weighted_triangle(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(0, 2, weight=1.0)
        result = dress_graph(G)
        assert isinstance(result, DRESSResult)
        assert len(result.edge_dress) == 3
        # Weighted DRESS values can exceed 2.0 — the [0,2] bound only
        # holds for unweighted graphs.  Just sanity-check positivity.
        for v in result.edge_dress:
            assert v > 0.0

    def test_unweighted_default(self, triangle):
        """When no weight attribute exists, edge_weight holds variant weights."""
        result = dress_graph(triangle)
        # Variant weights are set by the DRESS engine (2.0 for undirected)
        for w in result.edge_weight:
            assert w > 0.0


# ── string and mixed labels ──────────────────────────────────────────

class TestLabels:
    def test_string_labels(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        result = dress_graph(G)
        assert len(result.edge_dress) == 3
        for v in result.edge_dress:
            assert abs(v - 2.0) < 1e-4

    def test_string_labels_set_attributes(self):
        G = nx.Graph()
        G.add_edges_from([("x", "y"), ("y", "z")])
        dress_graph(G, set_attributes=True)
        for u, v in G.edges():
            assert "dress" in G[u][v]
        for n in G.nodes():
            assert "dress_norm" in G.nodes[n]

    def test_karate_club(self):
        G = nx.karate_club_graph()
        result = dress_graph(G)
        assert len(result.edge_dress) == G.number_of_edges()
        assert result.iterations < 100


# ── determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_result_on_repeated_calls(self, k4):
        r1 = dress_graph(k4)
        r2 = dress_graph(k4)
        assert r1.edge_dress == r2.edge_dress
        assert r1.node_dress == r2.node_dress
        assert r1.iterations == r2.iterations


# ── isomorphism distinguishing ───────────────────────────────────────

class TestIsomorphism:
    def test_prism_vs_k33(self):
        """Prism and K3,3 are both 3-regular on 6 nodes but non-isomorphic."""
        prism = nx.Graph()
        prism.add_edges_from([
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (0, 3), (1, 4), (2, 5),
        ])
        k33 = nx.complete_bipartite_graph(3, 3)

        r_prism = dress_graph(prism)
        r_k33 = dress_graph(k33)

        fp_prism = sorted(round(v, 6) for v in r_prism.edge_dress)
        fp_k33 = sorted(round(v, 6) for v in r_k33.edge_dress)
        assert fp_prism != fp_k33, "DRESS should distinguish Prism from K3,3"

    def test_isomorphic_graphs_same_fingerprint(self):
        """Two isomorphic graphs must produce the same sorted fingerprint."""
        G1 = nx.cycle_graph(5)
        # Relabeled copy
        mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        G2 = nx.relabel_nodes(G1, mapping)

        r1 = dress_graph(G1)
        r2 = dress_graph(G2)

        fp1 = sorted(round(v, 6) for v in r1.edge_dress)
        fp2 = sorted(round(v, 6) for v in r2.edge_dress)
        assert fp1 == fp2


# ── NxDRESS persistent object ────────────────────────────────────────

class TestNxDRESS:
    def test_basic_lifecycle(self, triangle):
        dg = NxDRESS(triangle)
        fr = dg.fit()
        assert fr.iterations < 100
        assert fr.delta < 1e-6
        r = dg.result()
        assert isinstance(r, DRESSResult)
        assert len(r.edge_dress) == 3
        dg.close()

    def test_get_existing_edge(self, triangle):
        dg = NxDRESS(triangle)
        dg.fit()
        # All edges in a triangle converge to ~2.0
        val = dg.get(0, 1)
        assert abs(val - 2.0) < 1e-4
        dg.close()

    def test_get_virtual_edge(self):
        """Query a non-existent edge on a path graph."""
        G = nx.path_graph(4)  # 0-1-2-3
        dg = NxDRESS(G)
        dg.fit()
        # 0-3 is a virtual edge (not in graph)
        val = dg.get(0, 3)
        assert val > 0.0
        assert val < 2.0
        dg.close()

    def test_get_string_labels(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        dg = NxDRESS(G)
        dg.fit()
        val = dg.get("a", "b")
        assert abs(val - 2.0) < 1e-4
        dg.close()

    def test_context_manager(self, triangle):
        with NxDRESS(triangle) as dg:
            dg.fit()
            val = dg.get(0, 1)
            assert abs(val - 2.0) < 1e-4

    def test_result_iterations_and_delta(self, triangle):
        dg = NxDRESS(triangle)
        fr = dg.fit()
        r = dg.result()
        assert r.iterations == fr.iterations
        assert r.delta == fr.delta
        dg.close()

    def test_repr(self, triangle):
        dg = NxDRESS(triangle)
        s = repr(dg)
        assert "NxDRESS" in s
        assert "n_vertices=3" in s
        dg.close()

    def test_nodes_property(self):
        G = nx.Graph()
        G.add_edges_from([("x", "y"), ("y", "z")])
        dg = NxDRESS(G)
        assert set(dg.nodes) == {"x", "y", "z"}
        dg.close()

    def test_repeated_fit(self, triangle):
        """Fitting twice should work and re-converge."""
        dg = NxDRESS(triangle)
        dg.fit()
        r1 = dg.result()
        dg.fit()
        r2 = dg.result()
        for a, b in zip(r1.edge_dress, r2.edge_dress):
            assert abs(a - b) < 1e-6
        dg.close()
