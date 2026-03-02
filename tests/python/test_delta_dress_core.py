"""
Tests for the pure-Python Δ^k-DRESS implementation (dress.core.delta_dress_fit).

Run from the repo root:
    pytest tests/python/test_delta_dress_core.py -v
"""

import pytest

from dress.core import delta_dress_fit, DeltaDRESSResult, Variant
from dress.core import UNDIRECTED


# ── helpers ──────────────────────────────────────────────────────────

K3_SRC = [0, 1, 0]
K3_TGT = [1, 2, 2]
K4_SRC = [0, 0, 0, 1, 1, 2]
K4_TGT = [1, 2, 3, 2, 3, 3]
P4_SRC = [0, 1, 2]
P4_TGT = [1, 2, 3]

EPS = 1e-3


def _total(r):
    return sum(r.histogram)


# ── histogram size ───────────────────────────────────────────────────

class TestHistSize:
    def test_eps_1e3(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-3)
        assert r.hist_size == 2001

    def test_eps_1e6(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-6)
        assert r.hist_size == 2000001

    def test_result_type(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert isinstance(r, DeltaDRESSResult)
        assert len(r.histogram) == r.hist_size

    def test_repr(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        s = repr(r)
        assert "DeltaDRESSResult" in s
        assert "hist_size=" in s

    def test_weighted_hist_size(self):
        """Weighted K3 with non-uniform weights → adaptive dmax > 2.0."""
        r = delta_dress_fit(3, K3_SRC, K3_TGT,
                            weights=[1.0, 10.0, 1.0],
                            k=0, epsilon=1e-3)
        assert r.hist_size > 2001, f"expected > 2001, got {r.hist_size}"
        assert len(r.histogram) == r.hist_size
        assert _total(r) == 3


# ── Δ^0 — full graph ────────────────────────────────────────────────

class TestDelta0:
    def test_k3_total(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_k3_single_bin(self):
        """K3 is vertex-transitive → all edges in same bin."""
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero == 1

    def test_k3_top_bin(self):
        """K3 edges have dress ≈ 2.0 → top bin(s)."""
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        # Allow for float imprecision: bin 1999 or 2000
        assert r.histogram[-1] + r.histogram[-2] == 3

    def test_k4_total(self):
        r = delta_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 6

    def test_k4_all_equal(self):
        """K4 is vertex-transitive → all 6 edges in same bin."""
        r = delta_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero == 1


# ── Δ^1 ─────────────────────────────────────────────────────────────

class TestDelta1:
    def test_k3_total(self):
        """C(3,1)=3 subgraphs × 1 edge each = 3."""
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS)
        assert _total(r) == 3

    def test_k4_total(self):
        """C(4,1)=4 subgraphs × 3 edges each = 12."""
        r = delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS)
        assert _total(r) == 12

    def test_k4_all_at_top(self):
        """Each K4\v is K3 → all 12 edges near 2.0."""
        r = delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS)
        assert r.histogram[-1] + r.histogram[-2] == 12


# ── Δ^2 ─────────────────────────────────────────────────────────────

class TestDelta2:
    def test_k3_zero(self):
        """Removing 2 of 3 vertices → 0 edges."""
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=2, epsilon=EPS)
        assert _total(r) == 0

    def test_k4_total(self):
        """C(4,2)=6 subgraphs × 1 edge each = 6."""
        r = delta_dress_fit(4, K4_SRC, K4_TGT, k=2, epsilon=EPS)
        assert _total(r) == 6


# ── edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_k_eq_N(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=3, epsilon=EPS)
        assert _total(r) == 0

    def test_k_gt_N(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=10, epsilon=EPS)
        assert _total(r) == 0

    def test_single_edge_delta0(self):
        r = delta_dress_fit(2, [0], [1], k=0, epsilon=EPS)
        assert _total(r) == 1

    def test_single_edge_delta1(self):
        """Removing either endpoint of a single-edge graph → 0 edges."""
        r = delta_dress_fit(2, [0], [1], k=1, epsilon=EPS)
        assert _total(r) == 0


# ── path graph ───────────────────────────────────────────────────────

class TestPath:
    def test_delta0_total(self):
        r = delta_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_delta0_not_all_equal(self):
        """P4 is NOT vertex-transitive → at least 2 distinct bins."""
        r = delta_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero >= 2

    def test_delta1_total(self):
        """C(4,1)=4 subsets of P4.
        Remove 0 → P3(1-2-3) → 2 edges; remove 1 → K1+K2 → 1 edge;
        remove 2 → K2+K1 → 1 edge; remove 3 → P3(0-1-2) → 2 edges.
        Total = 2+1+1+2 = 6."""
        r = delta_dress_fit(4, P4_SRC, P4_TGT, k=1, epsilon=EPS)
        assert _total(r) == 6


# ── precompute flag ──────────────────────────────────────────────────

class TestMultisets:
    def test_disabled_by_default(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert r.multisets is None

    def test_delta0_k3_shape(self):
        import math
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS,
                            keep_multisets=True)
        assert r.num_subgraphs == 1
        assert len(r.multisets) == 1
        assert len(r.multisets[0]) == 3

    def test_delta0_k3_values(self):
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS,
                            keep_multisets=True)
        for row in r.multisets:
            for v in row:
                assert abs(v - 2.0) < EPS, f"expected ~2.0, got {v}"

    def test_delta1_k3_nan_pattern(self):
        import math
        r = delta_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS,
                            keep_multisets=True)
        assert r.num_subgraphs == 3
        assert len(r.multisets) == 3
        for row in r.multisets:
            nans = sum(1 for v in row if math.isnan(v))
            assert nans == 2, f"expected 2 NaN per row, got {nans}"
            vals = [v for v in row if not math.isnan(v)]
            assert abs(vals[0] - 2.0) < EPS


class TestPrecompute:
    def test_identical_results(self):
        r1 = delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                             precompute=False)
        r2 = delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                             precompute=True)
        assert r1.hist_size == r2.hist_size
        assert r1.histogram == r2.histogram


# ── dispatch from __init__.py ────────────────────────────────────────

class TestInitDispatch:
    def test_top_level_import(self):
        from dress import delta_dress_fit as top_delta
        r = top_delta(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert isinstance(r, DeltaDRESSResult)
        assert _total(r) == 3
