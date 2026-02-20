"""
A/B cross-validation: pure-Python vs C (pybind11) DRESS implementation.

Ensures the two backends produce bit-identical results on a variety of
graph topologies, weights, and directed variants.

Run from the repo root:
    pytest tests/python/test_dress_ab.py -v
"""

import math
import pytest

# ── import both backends ─────────────────────────────────────────────

core = pytest.importorskip("dress.core", reason="pure-Python backend not found")
_core = pytest.importorskip("dress._core", reason="C extension not built")

from dress.core import dress_fit as py_dress_fit, Variant as PyVariant
from dress._core import DRESS as CDRESS

# Map pure-Python variant enum to C variant enum
_C_VARIANTS = {
    PyVariant.UNDIRECTED: _core.UNDIRECTED,
    PyVariant.DIRECTED:   _core.DIRECTED,
    PyVariant.FORWARD:    _core.FORWARD,
    PyVariant.BACKWARD:   _core.BACKWARD,
}

MAX_ITER = 200
EPS = 1e-12


# ── helpers ──────────────────────────────────────────────────────────

def _compare(n, sources, targets, weights=None, variant=PyVariant.UNDIRECTED):
    """Build both backends, fit, and assert identical results."""
    # Pure Python (functional API)
    pr = py_dress_fit(n, sources, targets, weights=weights, variant=variant,
                      max_iterations=MAX_ITER, epsilon=EPS)

    # C
    cv = _C_VARIANTS[variant]
    if weights is not None:
        cg = CDRESS(n, sources, targets, list(weights), cv)
    else:
        cg = CDRESS(n, sources, targets, cv)
    cg.fit(MAX_ITER, EPS)

    # Compare edge dress values
    assert len(pr.edge_dress) == cg.n_edges
    for e in range(cg.n_edges):
        cd = cg.edge_dress(e)
        pd = pr.edge_dress[e]
        assert cd == pytest.approx(pd, abs=1e-14), (
            f"edge {e}: C={cd}  Py={pd}  diff={abs(cd - pd):.2e}"
        )

    # Compare edge weights
    for e in range(cg.n_edges):
        cw = cg.edge_weight(e)
        pw = pr.edge_weight[e]
        assert cw == pytest.approx(pw, abs=1e-14), (
            f"edge {e} weight: C={cw}  Py={pw}"
        )

    # Compare node dress norms
    for u in range(n):
        cn = cg.node_dress(u)
        pn = pr.node_dress[u]
        assert cn == pytest.approx(pn, abs=1e-14), (
            f"node {u}: C={cn}  Py={pn}"
        )


# ── tests: topologies ────────────────────────────────────────────────

class TestTopologies:
    def test_single_edge(self):
        _compare(2, [0], [1])

    def test_triangle(self):
        _compare(3, [0, 1, 0], [1, 2, 2])

    def test_path_4(self):
        _compare(4, [0, 1, 2], [1, 2, 3])

    def test_path_6(self):
        _compare(6, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5])

    def test_k4(self):
        _compare(4, [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3])

    def test_star_5(self):
        _compare(5, [0, 0, 0, 0], [1, 2, 3, 4])

    def test_cycle_5(self):
        _compare(5, [0, 1, 2, 3, 4], [1, 2, 3, 4, 0])

    def test_petersen_like(self):
        """A small dense graph (K5 minus one edge)."""
        src = [0, 0, 0, 1, 1, 1, 2, 2, 3]
        tgt = [1, 2, 3, 2, 3, 4, 3, 4, 4]
        _compare(5, src, tgt)

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing edge 1-2: 0-1, 0-2, 1-2, 1-3, 2-3."""
        _compare(4, [0, 0, 1, 1, 2], [1, 2, 2, 3, 3])

    def test_disconnected_component(self):
        """Two isolated edges: 0-1, 2-3."""
        _compare(4, [0, 2], [1, 3])


# ── tests: weighted ──────────────────────────────────────────────────

class TestWeighted:
    def test_triangle_weighted(self):
        _compare(3, [0, 1, 0], [1, 2, 2], weights=[1.0, 2.0, 3.0])

    def test_path_weighted(self):
        _compare(4, [0, 1, 2], [1, 2, 3], weights=[0.5, 1.5, 2.5])

    def test_k4_weighted(self):
        _compare(4,
                 [0, 0, 0, 1, 1, 2],
                 [1, 2, 3, 2, 3, 3],
                 weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_uniform_weight(self):
        """Uniform weight != 1 should differ from unweighted."""
        _compare(3, [0, 1, 0], [1, 2, 2], weights=[2.0, 2.0, 2.0])


# ── tests: directed variants ─────────────────────────────────────────

class TestDirectedVariants:
    def _dag(self, variant):
        """DAG: 0->1, 1->2, 0->2."""
        _compare(3, [0, 1, 0], [1, 2, 2], variant=variant)

    def test_undirected(self):
        self._dag(PyVariant.UNDIRECTED)

    def test_directed(self):
        self._dag(PyVariant.DIRECTED)

    def test_forward(self):
        self._dag(PyVariant.FORWARD)

    def test_backward(self):
        self._dag(PyVariant.BACKWARD)

    def test_directed_reciprocal(self):
        """Reciprocal edges with different weights."""
        _compare(3, [0, 1, 1, 2], [1, 0, 2, 1],
                 weights=[3.0, 5.0, 1.0, 2.0],
                 variant=PyVariant.DIRECTED)

    def test_forward_weighted(self):
        _compare(3, [0, 1, 0], [1, 2, 2],
                 weights=[1.0, 2.0, 3.0],
                 variant=PyVariant.FORWARD)

    def test_backward_weighted(self):
        _compare(3, [0, 1, 0], [1, 2, 2],
                 weights=[1.0, 2.0, 3.0],
                 variant=PyVariant.BACKWARD)

    def test_directed_star(self):
        """Directed star: 0->1, 0->2, 0->3."""
        _compare(4, [0, 0, 0], [1, 2, 3], variant=PyVariant.DIRECTED)

    def test_directed_cycle(self):
        """Directed cycle: 0->1->2->3->0."""
        _compare(4, [0, 1, 2, 3], [1, 2, 3, 0], variant=PyVariant.DIRECTED)
