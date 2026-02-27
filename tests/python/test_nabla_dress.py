"""
Tests for the ∇^k-DRESS Python (pybind11) bindings.

Run from the repo root after building:
    pip install ./python
    pytest tests/python/test_nabla_dress.py -v
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
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-3)
        assert r.hist_size == 2001

    def test_eps_1e6(self):
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=1e-6)
        assert r.hist_size == 2000001

    def test_histogram_length(self):
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert len(r.histogram) == r.hist_size


# ── ∇^0 — no individualization (identical to base DRESS) ────────────

class TestNabla0:
    def test_k3_total(self):
        """∇^0 on K3: 1 round × 3 edges = 3 values."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_k3_single_bin(self):
        """K3 is vertex-transitive → all edges in same bin."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero == 1

    def test_k3_top_bin(self):
        """K3 edges have dress = 2.0 → top bin."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        assert r.histogram[r.hist_size - 1] == 3

    def test_k4_total(self):
        r = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 6

    def test_k4_all_at_top(self):
        r = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=0, epsilon=EPS)
        assert r.histogram[r.hist_size - 1] == 6


# ── ∇^1 — mark one vertex at a time ─────────────────────────────────

class TestNabla1:
    def test_k3_total(self):
        """C(3,1)=3 subsets × 3 edges each = 9 values."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS)
        assert _total(r) == 9

    def test_k4_total(self):
        """C(4,1)=4 subsets × 6 edges each = 24 values."""
        r = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS)
        assert _total(r) == 24

    def test_p4_total(self):
        """C(4,1)=4 subsets × 3 edges each = 12 values."""
        r = dress.nabla_dress_fit(4, P4_SRC, P4_TGT, k=1, epsilon=EPS)
        assert _total(r) == 12


# ── ∇^2 ─────────────────────────────────────────────────────────────

class TestNabla2:
    def test_k3_total(self):
        """C(3,2)=3 subsets × 3 edges each = 9 values."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=2, epsilon=EPS)
        assert _total(r) == 9

    def test_k4_total(self):
        """C(4,2)=6 subsets × 6 edges each = 36 values."""
        r = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=2, epsilon=EPS)
        assert _total(r) == 36


# ── edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_k_eq_N(self):
        """k == N → code returns empty histogram."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=3, epsilon=EPS)
        assert _total(r) == 0

    def test_k_gt_N(self):
        """k > N → no valid subsets → empty histogram."""
        r = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=10, epsilon=EPS)
        assert _total(r) == 0

    def test_single_edge_nabla0(self):
        r = dress.nabla_dress_fit(2, [0], [1], k=0, epsilon=EPS)
        assert _total(r) == 1


# ── path graph ───────────────────────────────────────────────────────

class TestPath:
    def test_nabla0_total(self):
        r = dress.nabla_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        assert _total(r) == 3

    def test_nabla0_not_all_equal(self):
        """P4 is NOT vertex-transitive → at least 2 distinct bins."""
        r = dress.nabla_dress_fit(4, P4_SRC, P4_TGT, k=0, epsilon=EPS)
        nonzero = sum(1 for v in r.histogram if v > 0)
        assert nonzero >= 2


# ── nabla_weight parameter ───────────────────────────────────────────

class TestNablaWeight:
    def test_different_weights_differ(self):
        """Different nabla_weight values should produce different histograms."""
        r1 = dress.nabla_dress_fit(4, P4_SRC, P4_TGT, k=1, epsilon=EPS,
                                   nabla_weight=2.0)
        r2 = dress.nabla_dress_fit(4, P4_SRC, P4_TGT, k=1, epsilon=EPS,
                                   nabla_weight=3.0)
        assert r1.histogram != r2.histogram

    def test_weight_1_same_as_nabla0(self):
        """nabla_weight=1.0 with k=1 should match ∇^0 × C(N,1) repetitions."""
        r0 = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=0, epsilon=EPS)
        r1 = dress.nabla_dress_fit(3, K3_SRC, K3_TGT, k=1, epsilon=EPS,
                                   nabla_weight=1.0)
        # With nabla_weight=1.0, marking has no effect → each subset is
        # identical to base DRESS.  Total = C(3,1) * 3 = 9.
        assert _total(r1) == 9
        # Each bin should be exactly 3× the ∇^0 bin
        for a, b in zip(r0.histogram, r1.histogram):
            assert b == 3 * a


# ── precompute flag ──────────────────────────────────────────────────

class TestPrecompute:
    def test_identical_results(self):
        r1 = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                                   precompute=False)
        r2 = dress.nabla_dress_fit(4, K4_SRC, K4_TGT, k=1, epsilon=EPS,
                                   precompute=True)
        assert r1.hist_size == r2.hist_size
        for a, b in zip(r1.histogram, r2.histogram):
            assert a == b
