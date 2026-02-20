"""
Pure-Python DRESS implementation.

No compiled dependencies required. Only the Python standard library and
(optionally) NumPy for array output.

Usage::

    from dress.core import DRESS, UNDIRECTED

    g = DRESS(4, [0, 1, 2, 0], [1, 2, 3, 3])
    result = g.fit()
    print(result.edge_dress)   # per-edge similarity values
    print(result.node_dress)   # per-node norms
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence

__all__ = [
    "DRESS",
    "DRESSResult",
    "Variant",
    "UNDIRECTED",
    "DIRECTED",
    "FORWARD",
    "BACKWARD",
]


# -- variant enum ----------------------------------------------------------

class Variant(IntEnum):
    UNDIRECTED = 0
    DIRECTED = 1
    FORWARD = 2
    BACKWARD = 3


UNDIRECTED = Variant.UNDIRECTED
DIRECTED = Variant.DIRECTED
FORWARD = Variant.FORWARD
BACKWARD = Variant.BACKWARD


# -- result dataclass -------------------------------------------------------

@dataclass
class DRESSResult:
    """Result of :meth:`DRESS.fit`."""

    sources: List[int]
    targets: List[int]
    edge_dress: List[float]
    edge_weight: List[float]
    node_dress: List[float]
    iterations: int
    delta: float

    def __repr__(self) -> str:
        return (
            f"DRESSResult(edges={len(self.sources)}, "
            f"iterations={self.iterations}, delta={self.delta:.6e})"
        )


# -- helpers ----------------------------------------------------------------

def _sorted_merge_common(
    at: List[int],
    ae: List[int],
    a_start: int,
    a_end: int,
    bt: List[int],
    be: List[int],
    b_start: int,
    b_end: int,
) -> List[tuple]:
    """Sorted-merge walk returning (edge_idx_u, edge_idx_v) for common neighbours."""
    result = []
    i, j = a_start, b_start
    while i < a_end and j < b_end:
        x, y = at[i], bt[j]
        if x == y:
            result.append((ae[i], be[j]))
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return result


def _binary_search(targets: List[int], start: int, end: int, value: int) -> int:
    """Binary search for *value* in targets[start:end]. Returns index or -1."""
    lo, hi = start, end
    while lo < hi:
        mid = (lo + hi) >> 1
        v = targets[mid]
        if v == value:
            return mid
        elif v < value:
            lo = mid + 1
        else:
            hi = mid
    return -1


# -- main class -------------------------------------------------------------

class DRESS:
    """Pure-Python DRESS graph.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights. ``None`` for unweighted (all weights = 1).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    """

    def __init__(
        self,
        n_vertices: int,
        sources: Sequence[int],
        targets: Sequence[int],
        weights: Optional[Sequence[float]] = None,
        variant: Variant = UNDIRECTED,
        precompute_intercepts: bool = False,
    ) -> None:
        E = len(sources)
        if len(targets) != E:
            raise ValueError("sources and targets must have equal length")
        if weights is not None and len(weights) != E:
            raise ValueError("weights must have the same length as sources")

        self._N = n_vertices
        self._E = E
        self._U = list(sources)
        self._V = list(targets)
        self._variant = Variant(variant)
        self._precompute_intercepts = precompute_intercepts

        w_in = [1.0] * E if weights is None else [float(w) for w in weights]

        # Build variant adjacency (CSR)
        self._build_adjacency(w_in)

        # Initialise dress values to 1.0
        self._edge_dress: List[float] = [1.0] * E
        self._edge_dress_next: List[float] = [1.0] * E
        self._node_dress: List[float] = [0.0] * n_vertices

    # -- public properties --------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return self._N

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self._E

    @property
    def variant(self) -> Variant:
        """DRESS variant used."""
        return self._variant

    def __repr__(self) -> str:
        return (
            f"DRESS(n_vertices={self._N}, n_edges={self._E}, "
            f"variant={self._variant.name})"
        )

    # -- per-element accessors (pybind11-compatible API) --------------------

    def edge_source(self, e: int) -> int:
        """Source vertex of edge *e*."""
        return self._U[e]

    def edge_target(self, e: int) -> int:
        """Target vertex of edge *e*."""
        return self._V[e]

    def edge_weight(self, e: int) -> float:
        """Variant weight of edge *e* (available after adjacency build)."""
        return self._edge_weight[e]

    def edge_dress(self, e: int) -> float:
        """DRESS value of edge *e* (call :meth:`fit` first)."""
        return self._edge_dress[e]

    def node_dress(self, u: int) -> float:
        """Node DRESS norm of vertex *u* (call :meth:`fit` first)."""
        return self._node_dress[u]

    # -- NumPy array properties ---------------------------------------------

    @property
    def sources(self):
        """Edge source array as ``numpy.ndarray[int32]``."""
        import numpy as np
        if not hasattr(self, '_np_sources'):
            self._np_sources = np.array(self._U, dtype=np.int32)
        return self._np_sources

    @property
    def targets(self):
        """Edge target array as ``numpy.ndarray[int32]``."""
        import numpy as np
        if not hasattr(self, '_np_targets'):
            self._np_targets = np.array(self._V, dtype=np.int32)
        return self._np_targets

    @property
    def weights(self):
        """Variant edge-weight array as ``numpy.ndarray[float64]``."""
        import numpy as np
        if not hasattr(self, '_np_weights'):
            self._np_weights = np.array(self._edge_weight, dtype=np.float64)
        return self._np_weights

    @property
    def dress_values(self):
        """DRESS edge values as ``numpy.ndarray[float64]``."""
        import numpy as np
        if not hasattr(self, '_np_dress'):
            self._np_dress = np.array(self._edge_dress, dtype=np.float64)
        else:
            # Update in-place so views stay valid
            for i, v in enumerate(self._edge_dress):
                self._np_dress[i] = v
        return self._np_dress

    @property
    def node_dress_values(self):
        """Node DRESS norms as ``numpy.ndarray[float64]``."""
        import numpy as np
        if not hasattr(self, '_np_node_dress'):
            self._np_node_dress = np.array(self._node_dress, dtype=np.float64)
        else:
            for i, v in enumerate(self._node_dress):
                self._np_node_dress[i] = v
        return self._np_node_dress

    # -- adjacency construction --------------------------------------------

    def _build_adjacency(self, w_in: List[float]) -> None:
        N, E = self._N, self._E
        U, V = self._U, self._V
        variant = self._variant

        # Step 1: build raw adjacency (input-level)
        raw_deg = [0] * N
        for i in range(E):
            if variant == Variant.UNDIRECTED:
                raw_deg[U[i]] += 1
                raw_deg[V[i]] += 1
            else:
                raw_deg[U[i]] += 1

        raw_offset = [0] * (N + 1)
        for i in range(N):
            raw_offset[i + 1] = raw_offset[i] + raw_deg[i]

        S_raw = raw_offset[N]
        raw_target = [0] * S_raw
        raw_eidx = [0] * S_raw
        raw_weight = [0.0] * S_raw
        cnt = [0] * N

        for i in range(E):
            u, v, w = U[i], V[i], w_in[i]
            if variant == Variant.UNDIRECTED:
                p = raw_offset[u] + cnt[u]; cnt[u] += 1
                raw_target[p] = v; raw_eidx[p] = i; raw_weight[p] = w
                p = raw_offset[v] + cnt[v]; cnt[v] += 1
                raw_target[p] = u; raw_eidx[p] = i; raw_weight[p] = w
            else:
                p = raw_offset[u] + cnt[u]; cnt[u] += 1
                raw_target[p] = v; raw_eidx[p] = i; raw_weight[p] = w

        # Sort each node's segment by target id
        for u in range(N):
            s, e = raw_offset[u], raw_offset[u + 1]
            if e - s > 1:
                idxs = list(range(s, e))
                idxs.sort(key=lambda k: raw_target[k])
                raw_target[s:e] = [raw_target[k] for k in idxs]
                raw_eidx[s:e] = [raw_eidx[k] for k in idxs]
                raw_weight[s:e] = [raw_weight[k] for k in idxs]

        # Step 2: build variant adjacency
        var_deg = [0] * N
        self._edge_weight = [0.0] * E

        if variant == Variant.UNDIRECTED:
            for i in range(E):
                var_deg[U[i]] += 1
                var_deg[V[i]] += 1
        elif variant == Variant.DIRECTED:
            in_deg = [0] * N
            for i in range(E):
                var_deg[U[i]] += 1
                if _binary_search(raw_target, raw_offset[V[i]], raw_offset[V[i] + 1], U[i]) < 0:
                    in_deg[V[i]] += 1
            for i in range(N):
                var_deg[i] += in_deg[i]
        elif variant == Variant.FORWARD:
            for i in range(E):
                var_deg[U[i]] += 1
        elif variant == Variant.BACKWARD:
            for i in range(E):
                var_deg[V[i]] += 1

        adj_offset = [0] * (N + 1)
        for i in range(N):
            adj_offset[i + 1] = adj_offset[i] + var_deg[i]

        S = adj_offset[N]
        adj_target = [0] * S
        adj_eidx = [0] * S
        cnt = [0] * N

        for u in range(N):
            rs, re = raw_offset[u], raw_offset[u + 1]
            for k in range(rs, re):
                v = raw_target[k]
                eid = raw_eidx[k]
                w = raw_weight[k]

                if variant == Variant.UNDIRECTED:
                    p = adj_offset[u] + cnt[u]; cnt[u] += 1
                    adj_target[p] = v; adj_eidx[p] = eid
                    self._edge_weight[eid] = 2.0 * w

                elif variant == Variant.DIRECTED:
                    recip = _binary_search(raw_target, raw_offset[v], raw_offset[v + 1], u)
                    cw = w + (raw_weight[recip] if recip >= 0 else 0.0)
                    p = adj_offset[u] + cnt[u]; cnt[u] += 1
                    adj_target[p] = v; adj_eidx[p] = eid
                    self._edge_weight[eid] = cw
                    if recip < 0:
                        p = adj_offset[v] + cnt[v]; cnt[v] += 1
                        adj_target[p] = u; adj_eidx[p] = eid
                        # weight already set above

                elif variant == Variant.FORWARD:
                    p = adj_offset[u] + cnt[u]; cnt[u] += 1
                    adj_target[p] = v; adj_eidx[p] = eid
                    self._edge_weight[eid] = w

                elif variant == Variant.BACKWARD:
                    p = adj_offset[v] + cnt[v]; cnt[v] += 1
                    adj_target[p] = u; adj_eidx[p] = eid
                    self._edge_weight[eid] = w

        # Sort each node's variant segment by target id
        for u in range(N):
            s, e = adj_offset[u], adj_offset[u + 1]
            if e - s > 1:
                idxs = list(range(s, e))
                idxs.sort(key=lambda k: adj_target[k])
                adj_target[s:e] = [adj_target[k] for k in idxs]
                adj_eidx[s:e] = [adj_eidx[k] for k in idxs]

        self._adj_offset = adj_offset
        self._adj_target = adj_target
        self._adj_eidx = adj_eidx

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
    ) -> DRESSResult:
        """Run iterative fixed-point fitting.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations.
        epsilon : float
            Convergence threshold (max per-edge change).

        Returns
        -------
        DRESSResult
        """
        N, E = self._N, self._E
        adj_offset = self._adj_offset
        adj_target = self._adj_target
        adj_eidx = self._adj_eidx
        ew = self._edge_weight
        ed = self._edge_dress
        ed_next = self._edge_dress_next
        nd = self._node_dress

        final_delta = 0.0
        final_iter = max_iterations

        for iteration in range(max_iterations):
            max_delta = 0.0

            # Phase 1: compute node norms
            for u in range(N):
                s = 4.0  # self-loop: w=2, d=2 -> w*d = 4
                base = adj_offset[u]
                end = adj_offset[u + 1]
                for k in range(base, end):
                    ei = adj_eidx[k]
                    s += ew[ei] * ed[ei]
                nd[u] = math.sqrt(s)

            # Phase 2: compute next dress values
            for e in range(E):
                u, v = self._U[e], self._V[e]
                numerator = 0.0

                # Sorted-merge walk for common neighbours
                iu, iu_end = adj_offset[u], adj_offset[u + 1]
                iv, iv_end = adj_offset[v], adj_offset[v + 1]
                i, j = iu, iv
                while i < iu_end and j < iv_end:
                    x, y = adj_target[i], adj_target[j]
                    if x == y:
                        eu = adj_eidx[i]
                        ev = adj_eidx[j]
                        numerator += ew[eu] * ed[eu] + ew[ev] * ed[ev]
                        i += 1
                        j += 1
                    elif x < y:
                        i += 1
                    else:
                        j += 1

                # Self-loop + edge's own contribution
                #
                # UNDIRECTED/DIRECTED: both u∈N[v] and v∈N[u], so both
                # self-loops cross:  (4 + w·d) + (w·d + 4) = 8 + 2·w·d.
                #
                # FORWARD/BACKWARD: only one direction exists, so only
                # one self-loop crosses: 4 + w·d.
                uv = ew[e] * ed[e]
                if self._variant in (Variant.FORWARD, Variant.BACKWARD):
                    numerator += 4.0 + uv
                else:
                    numerator += 8.0 + 2.0 * uv

                denom = nd[u] * nd[v]
                ed_next[e] = numerator / denom if denom > 0.0 else 0.0

                d = abs(ed[e] - ed_next[e])
                if d > max_delta:
                    max_delta = d

            # Phase 3: swap buffers
            self._edge_dress, self._edge_dress_next = ed_next, ed
            ed, ed_next = self._edge_dress, self._edge_dress_next

            final_delta = max_delta

            # Phase 4: convergence check
            if max_delta < epsilon:
                final_iter = iteration
                break

        self._node_dress = nd

        # Invalidate numpy caches so they re-sync on next access
        if hasattr(self, '_np_dress'):
            for i, v in enumerate(ed):
                self._np_dress[i] = v
        if hasattr(self, '_np_node_dress'):
            for i, v in enumerate(nd):
                self._np_node_dress[i] = v

        return DRESSResult(
            sources=list(self._U),
            targets=list(self._V),
            edge_dress=list(ed),
            edge_weight=list(ew),
            node_dress=list(nd),
            iterations=final_iter,
            delta=final_delta,
        )
