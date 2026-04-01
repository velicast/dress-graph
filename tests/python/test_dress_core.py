"""
Tests for the pure-Python DRESS implementation (dress.core).

Run from the repo root:
    pytest tests/python/test_dress_core.py -v
"""

import math
import pytest

from dress.core import fit, DRESS, DRESSResult, Variant
from dress.core import UNDIRECTED, DIRECTED, FORWARD, BACKWARD


# ── helpers ──────────────────────────────────────────────────────────

def _fit(n, src, tgt, **kwargs):
    """Short-hand: fit returning a DRESSResult."""
    return fit(n, src, tgt, **kwargs)


# ── construction / validation ────────────────────────────────────────

class TestConstruction:
    def test_unweighted(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2])
        assert len(r.sources) == 3
        assert len(r.targets) == 3
        assert len(r.edge_dress) == 3

    def test_weighted(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], weights=[1.0, 2.0, 3.0])
        assert len(r.edge_dress) == 3

    def test_vertex_weights_default(self):
        """Passing all 1.0 vertex weights should match default behavior."""
        src = [0, 1, 0, 1]
        tgt = [1, 2, 2, 3]
        n = 4

        # 1. Default (implicit All-1 vertex weights)
        r1 = _fit(n, src, tgt)

        # 2. Explicit All-1 vertex weights
        nw = [1.0] * n
        r2 = _fit(n, src, tgt, vertex_weights=nw)

        for d1, d2 in zip(r1.edge_dress, r2.edge_dress):
            assert abs(d1 - d2) < 1e-12, "Explicit vertex_weights=1.0 differs from default"

    def test_all_variants(self):
        for v in (UNDIRECTED, DIRECTED, FORWARD, BACKWARD):
            r = _fit(3, [0, 1, 0], [1, 2, 2], variant=v)
            assert len(r.edge_dress) == 3

    def test_bad_lengths(self):
        with pytest.raises(ValueError):
            _fit(3, [0, 1], [1])

    def test_bad_weights_length(self):
        with pytest.raises(ValueError):
            _fit(3, [0, 1], [1, 2], weights=[1.0])

    def test_repr(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2])
        s = repr(r)
        assert "DRESSResult" in s
        assert "edges=3" in s
        assert "iterations=" in s


# ── fitting ──────────────────────────────────────────────────────────

class TestFit:
    def test_triangle_convergence(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], max_iterations=100, epsilon=1e-8)
        assert r.iterations > 0
        assert r.delta >= 0.0

    def test_triangle_equal_dress(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], max_iterations=100, epsilon=1e-8)
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6, \
                "all edges in a triangle should have equal dress"

    def test_path_positive_dress(self):
        r = _fit(4, [0, 1, 2], [1, 2, 3])
        for d in r.edge_dress:
            assert d > 0.0, "path dress should be positive"
            assert d < 2.0, "path dress should be below 2"

    def test_path_symmetry(self):
        """Endpoint edges (0-1 and 2-3) should be equal by symmetry."""
        r = _fit(4, [0, 1, 2], [1, 2, 3])
        assert abs(r.edge_dress[0] - r.edge_dress[2]) < 1e-10

    def test_k4_equal_dress(self):
        """All edges in K4 should have equal dress by symmetry."""
        r = _fit(4, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6

    def test_k4_node_norms_equal(self):
        """All vertex norms in K4 should be equal."""
        r = _fit(4, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])
        n0 = r.vertex_dress[0]
        for n in r.vertex_dress:
            assert abs(n - n0) < 1e-6

    def test_star_dress(self):
        """Star edges should all have equal dress (by symmetry)."""
        r = _fit(5, [0, 0, 0, 0], [1, 2, 3, 4])
        d0 = r.edge_dress[0]
        for d in r.edge_dress:
            assert abs(d - d0) < 1e-6

    def test_weighted_fit(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], weights=[1.0, 2.0, 3.0])
        assert r.iterations > 0

    def test_single_edge(self):
        """Graph with one edge should converge immediately."""
        r = _fit(2, [0], [1])
        for d in r.edge_dress:
            assert d > 0.0

    def test_convergence_epsilon(self):
        """Small epsilon should yield small delta."""
        r = _fit(3, [0, 1, 0], [1, 2, 2],
                 max_iterations=500, epsilon=1e-12)
        assert r.delta < 1e-12


# ── dress values bounded ─────────────────────────────────────────────

class TestBounds:
    def test_dress_in_0_2(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2])
        for d in r.edge_dress:
            assert 0.0 <= d <= 2.0 + 1e-9

    def test_path_dress_in_0_2(self):
        r = _fit(4, [0, 1, 2], [1, 2, 3])
        for d in r.edge_dress:
            assert 0.0 <= d <= 2.0 + 1e-9

    def test_vertex_dress_positive(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2])
        for n in r.vertex_dress:
            assert n >= 2.0, "vertex norm >= 2 (self-loop contributes 4.0 under sqrt)"


# ── determinism ──────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_result_twice(self):
        r1 = _fit(3, [0, 1, 0], [1, 2, 2])
        r2 = _fit(3, [0, 1, 0], [1, 2, 2])
        for a, b in zip(r1.edge_dress, r2.edge_dress):
            assert a == b


# ── edge weight ──────────────────────────────────────────────────────

class TestEdgeWeight:
    def test_unweighted_undirected_weight(self):
        """Undirected unweighted edges have combined weight 2.0."""
        r = _fit(3, [0, 1, 0], [1, 2, 2])
        for w in r.edge_weight:
            assert w == pytest.approx(2.0)


# ── directed variants ────────────────────────────────────────────────

class TestDirectedVariants:
    def test_forward_fits(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], variant=FORWARD)
        assert r.iterations >= 0

    def test_backward_fits(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], variant=BACKWARD)
        assert r.iterations >= 0

    def test_directed_fits(self):
        r = _fit(3, [0, 1, 0], [1, 2, 2], variant=DIRECTED)
        assert r.iterations >= 0

    def test_directed_weight_reciprocal(self):
        """Reciprocal edges: combined weight = w(u->v) + w(v->u)."""
        r = _fit(2, [0, 1], [1, 0], weights=[3.0, 5.0], variant=DIRECTED)
        assert r.edge_weight[0] == pytest.approx(8.0)

    def test_forward_weight(self):
        """Forward variant: combined weight = w(u->v)."""
        r = _fit(2, [0], [1], weights=[3.0], variant=FORWARD)
        assert r.edge_weight[0] == pytest.approx(3.0)

    def test_backward_weight(self):
        """Backward variant: combined weight = w(u->v) used as w(v<-u)."""
        r = _fit(2, [0], [1], weights=[3.0], variant=BACKWARD)
        assert r.edge_weight[0] == pytest.approx(3.0)


# ── cross-validation with C binding values ────────────────────────

class TestCrossValidation:
    """Compare pure Python results with known values from the C implementation."""

    def test_triangle_dress_value(self):
        """Triangle dress converges to 2.0 (all pairs maximally similar)."""
        r = _fit(3, [0, 1, 0], [1, 2, 2],
                 max_iterations=500, epsilon=1e-12)
        assert abs(r.edge_dress[0] - 2.0) < 1e-6, \
            f"triangle dress = {r.edge_dress[0]}"

    def test_path_endpoint_edges_larger(self):
        """In a 4-path, endpoint edges (0-1, 2-3) have higher dress
        than the middle edge (1-2) because higher-degree nodes have
        larger norms that dilute the middle edge's similarity."""
        r = _fit(4, [0, 1, 2], [1, 2, 3],
                 max_iterations=500, epsilon=1e-12)
        # edge 0: 0-1, edge 1: 1-2, edge 2: 2-3
        assert r.edge_dress[0] > r.edge_dress[1], \
            "endpoint edge should have higher dress than middle"


# ── helpers for relabeling tests ──────────────────────────────────

import struct


def _permute_edges(src, dst, perm):
    """Apply vertex permutation perm[old] = new to an edge list."""
    return [perm[s] for s in src], [perm[d] for d in dst]


def _fit_dress(n, src, dst, **kwargs):
    """Build a DRESS object, fit it, and return it."""
    g = DRESS(n, list(src), list(dst), **kwargs)
    g.fit(max_iterations=200, epsilon=1e-12)
    return g


def _sorted_fingerprint(values):
    """Sort a list of doubles and return packed bytes for bitwise comparison."""
    s = sorted(values)
    return struct.pack(f'{len(s)}d', *s)


def _assert_fingerprint_equal(g1, g2, label=""):
    """Sorted edge_dress and vertex_dress must be bitwise identical."""
    assert _sorted_fingerprint(g1._edge_dress) == \
           _sorted_fingerprint(g2._edge_dress), \
        f"edge fingerprint mismatch: {label}"
    assert _sorted_fingerprint(g1._vertex_dress) == \
           _sorted_fingerprint(g2._vertex_dress), \
        f"vertex fingerprint mismatch: {label}"


# ── label-independence (sort+KBN) ────────────────────────────────

class TestLabelIndependence:
    """The sort+KBN implementation guarantees that sorted DRESS arrays
    are bitwise identical for isomorphic graphs (the product)."""

    def test_relabel_petersen(self):
        """Petersen graph under a non-trivial vertex permutation."""
        src = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
        dst = [1, 4, 5, 2, 6, 3, 7, 4, 8, 9, 7, 8, 9, 9, 5]
        N = 10

        g1 = _fit_dress(N, src, dst)

        perm = [7, 2, 5, 0, 9, 1, 8, 4, 3, 6]
        ps, pd = _permute_edges(src, dst, perm)
        g2 = _fit_dress(N, ps, pd)

        _assert_fingerprint_equal(g1, g2, "Petersen relabeled")

    def test_relabel_weighted(self):
        """Weighted house graph under relabeling."""
        src = [0, 0, 1, 2, 2, 3]
        dst = [1, 3, 2, 3, 4, 4]
        wts = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0]
        N = 5

        g1 = _fit_dress(N, src, dst, weights=wts)

        perm = [3, 0, 4, 1, 2]
        ps, pd = _permute_edges(src, dst, perm)
        g2 = _fit_dress(N, ps, pd, weights=wts)

        _assert_fingerprint_equal(g1, g2, "weighted house relabeled")

    def test_edge_reorder(self):
        """Same graph with edges in reversed order."""
        src1 = [0, 1, 2, 3, 1]
        dst1 = [1, 2, 3, 4, 3]
        src2 = [1, 3, 2, 1, 0]
        dst2 = [3, 4, 3, 2, 1]
        N = 5

        g1 = _fit_dress(N, src1, dst1)
        g2 = _fit_dress(N, src2, dst2)

        _assert_fingerprint_equal(g1, g2, "edge reorder")

    def test_relabel_directed(self):
        """Directed cycle under relabeling for all directed variants."""
        src = [0, 1, 2]
        dst = [1, 2, 0]
        N = 3
        perm = [2, 0, 1]

        for v in (DIRECTED, FORWARD, BACKWARD):
            g1 = _fit_dress(N, src, dst, variant=v)

            ps, pd = _permute_edges(src, dst, perm)
            g2 = _fit_dress(N, ps, pd, variant=v)

            _assert_fingerprint_equal(g1, g2, f"directed variant={v}")


# ── dress_get tests ──────────────────────────────────────────────

class TestDressGet:
    """Test the DRESS.get() query API."""

    def test_existing_edge_returns_edge_dress(self):
        """get(u,v) for an existing edge should return the fitted value."""
        g = _fit_dress(4, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])  # K4
        for e in range(6):
            u, v = g._U[e], g._V[e]
            assert g.get(u, v) == g._edge_dress[e]
            assert g.get(v, u) == g._edge_dress[e]

    def test_virtual_edge_positive(self):
        """Virtual edge in a path should have positive dress."""
        g = _fit_dress(4, [0, 1, 2], [1, 2, 3])
        d03 = g.get(0, 3)
        assert d03 > 0.0
        assert d03 < 2.0

    def test_virtual_edge_symmetric(self):
        """get(u,v) == get(v,u) for virtual edges."""
        g = _fit_dress(4, [0, 1, 2], [1, 2, 3])
        assert g.get(0, 3) == pytest.approx(g.get(3, 0), abs=1e-12)

    def test_virtual_edge_common_neighbor_higher(self):
        """Virtual edge with a common neighbor should have higher dress."""
        g = _fit_dress(4, [0, 1, 2], [1, 2, 3])
        d02 = g.get(0, 2)  # common neighbor: 1
        d03 = g.get(0, 3)  # no common neighbors
        assert d02 > d03

    def test_virtual_edge_relabel_invariance(self):
        """Sorted virtual-edge fingerprint for C5 must be bitwise identical."""
        src = [0, 1, 2, 3, 4]
        dst = [1, 2, 3, 4, 0]
        N, E = 5, 5

        g1 = _fit_dress(N, src, dst)

        perm = [4, 3, 2, 1, 0]
        ps, pd = _permute_edges(src, dst, perm)
        g2 = _fit_dress(N, ps, pd)

        # Collect all virtual-edge dress values, sort, memcmp
        v1, v2 = [], []
        for u in range(N):
            for v in range(u + 1, N):
                diff = v - u
                if diff == 1 or diff == N - 1:
                    continue  # skip existing edges in C5
                v1.append(g1.get(u, v))
                v2.append(g2.get(u, v))

        assert _sorted_fingerprint(v1) == _sorted_fingerprint(v2), \
            "virtual edge sorted fingerprint must be bitwise identical"
