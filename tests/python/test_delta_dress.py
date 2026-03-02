"""
Tests for the Δ^k-DRESS Python (pybind11) bindings.

Run from the repo root after building:
    pip install ./python
    pytest tests/python/test_delta_dress.py -v
"""

import pytest
import numpy as np

# ── import guard ─────────────────────────────────────────────────────

dress = pytest.importorskip("dress", reason="dress extension not built")

# ── helpers ──────────────────────────────────────────────────────────

K3_SRC = [0, 1, 0]
K3_TGT = [1, 2, 2]
K4_SRC = [0, 0, 0, 1, 1, 2]
K4_TGT = [1, 2, 3, 2, 3, 3]
P4_SRC = [0, 1, 2]
P4_TGT = [1, 2, 3]

EPS = 1e-3


def _total(r):
    return int(np.sum(r.histogram)) if hasattr(r.histogram, '__array__') else sum(r.histogram)


# ── histogram size ───────────────────────────────────────────────────

class TestHistSize:
    def test_eps_1e3(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-3)
        assert r.hist_size == 2001

    def test_eps_1e6(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-6)
        assert r.hist_size == 2000001

    def test_histogram_length(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert len(r.histogram) == r.hist_size

    def test_weighted_hist_size(self):
        """Weighted K3 with non-uniform weights → adaptive dmax > 2.0."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT,
                                  weights=[1.0, 10.0, 1.0],
                                  k=0, epsilon=1e-3)
        assert r.hist_size > 2001, f"expected > 2001, got {r.hist_size}"
        assert len(r.histogram) == r.hist_size
        assert _total(r) == 3


# ── Δ^0 — full graph ────────────────────────────────────────────────

class TestDelta0:
    def test_k3_total(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_k3_single_bin(self):
        """K3 is vertex-transitive → all edges in same bin."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero == 1

    def test_k3_top_bin(self):
        """K3 edges have dress = 2.0 → top bin."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert r.histogram[r.hist_size - 1] == 3

    def test_k4_total(self):
        r = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 6

    def test_k4_all_at_top(self):
        r = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        assert r.histogram[r.hist_size - 1] == 6


# ── Δ^1 ─────────────────────────────────────────────────────────────

class TestDelta1:
    def test_k3_total(self):
        """C(3,1)=3 subgraphs × 1 edge each = 3."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS)
        assert _total(r) == 3

    def test_k4_total(self):
        """C(4,1)=4 subgraphs × 3 edges each = 12."""
        r = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS)
        assert _total(r) == 12

    def test_k4_all_at_top(self):
        """Each K4\\v is K3 → all 12 edges at 2.0."""
        r = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS)
        assert r.histogram[r.hist_size - 1] == 12


# ── Δ^2 ─────────────────────────────────────────────────────────────

class TestDelta2:
    def test_k3_zero(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=2, epsilon=EPS)
        assert _total(r) == 0

    def test_k4_total(self):
        """C(4,2)=6 subgraphs × 1 edge each = 6."""
        r = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=2, epsilon=EPS)
        assert _total(r) == 6


# ── edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_k_eq_N(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=3, epsilon=EPS)
        assert _total(r) == 0

    def test_k_gt_N(self):
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=10, epsilon=EPS)
        assert _total(r) == 0

    def test_single_edge_delta0(self):
        r = dress.delta_dress_fit(2, [0], [1], k=0, epsilon=EPS)
        assert _total(r) == 1


# ── path graph ───────────────────────────────────────────────────────

class TestPath:
    def test_delta0_total(self):
        r = dress.delta_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_delta0_not_all_equal(self):
        """P4 is NOT vertex-transitive → at least 2 distinct bins."""
        r = dress.delta_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero >= 2


# ── multisets ────────────────────────────────────────────────────────

class TestMultisets:
    def test_disabled_by_default(self):
        """When keep_multisets is False (default), multisets is None."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert r.multisets is None

    def test_delta0_k3_shape(self):
        """Δ^0 K3: C(3,0)=1 subgraph, 3 edges → (1,3) matrix."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS,
                                  keep_multisets=True)
        assert r.num_subgraphs == 1
        ms = np.asarray(r.multisets)
        assert ms.shape == (1, 3)

    def test_delta0_k3_values(self):
        """Δ^0 K3: all edges converge to 2.0 in the full graph."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS,
                                  keep_multisets=True)
        ms = np.asarray(r.multisets)
        for v in ms.flat:
            assert abs(v - 2.0) < EPS, f"expected 2.0, got {v}"

    def test_delta1_k3_shape(self):
        """Δ^1 K3: C(3,1)=3 subgraphs, 3 edges → (3,3) matrix."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS,
                                  keep_multisets=True)
        assert r.num_subgraphs == 3
        ms = np.asarray(r.multisets)
        assert ms.shape == (3, 3)

    def test_delta1_k3_nan_pattern(self):
        """Δ^1 K3: each subgraph removes 1 vertex → 2 NaN, 1 non-NaN per row."""
        r = dress.delta_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS,
                                  keep_multisets=True)
        ms = np.asarray(r.multisets)
        for row in range(ms.shape[0]):
            nans = sum(1 for v in ms[row] if np.isnan(v))
            assert nans == 2, f"row {row}: expected 2 NaN, got {nans}"
            vals = [v for v in ms[row] if not np.isnan(v)]
            assert len(vals) == 1
            assert abs(vals[0] - 2.0) < EPS


# ── precompute flag ──────────────────────────────────────────────────

class TestPrecompute:
    def test_identical_results(self):
        r1 = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                                   precompute=False)
        r2 = dress.delta_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                                   precompute=True)
        assert r1.hist_size == r2.hist_size
        # Compare histograms element-by-element
        for a, b in zip(r1.histogram, r2.histogram):
            assert a == b
