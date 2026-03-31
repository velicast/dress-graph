"""
Pure-Python DRESS implementation.

No compiled dependencies required. Only the Python standard library and
(optionally) NumPy for array output.

Usage::

    from dress import fit

    result = fit(4, [0, 1, 2, 0], [1, 2, 3, 3])
    result.edge_dress     # per-edge similarity values
    result.node_dress     # per-node norms
    result.iterations     # number of iterations
    result.delta          # final convergence delta
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "fit",
    "delta_fit",
    "nabla_fit",
    "DRESS",
    "DRESSResult",
    "DeltaDRESSResult",
    "FitResult",
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


# -- result dataclasses -----------------------------------------------------

@dataclass
class FitResult:
    """Lightweight result of :meth:`DRESS.fit` (matches C extension API)."""

    iterations: int
    delta: float

    def __repr__(self) -> str:
        return (
            f"FitResult(iterations={self.iterations}, "
            f"delta={self.delta:.6e})"
        )


@dataclass
class DeltaDRESSResult:
    """Result of :func:`dress_delta_fit`."""

    histogram: List[Tuple[float, int]]
    multisets: object = None     # Optional 2D array (C(N,k) x E), NaN = removed
    num_subgraphs: int = 0

    def __repr__(self) -> str:
        total = sum(count for _, count in self.histogram)
        return (
            f"DeltaDRESSResult(histogram_entries={len(self.histogram)}, "
            f"total_count={total})"
        )


@dataclass
class NablaDRESSResult:
    """Result of :func:`dress_nabla_fit`."""

    histogram: List[Tuple[float, int]]
    multisets: object = None     # Optional 2D array (P(N,k) x E)
    num_tuples: int = 0

    def __repr__(self) -> str:
        total = sum(count for _, count in self.histogram)
        return (
            f"NablaDRESSResult(histogram_entries={len(self.histogram)}, "
            f"total_count={total})"
        )


def _flatten_histogram(hist: Dict[float, int]) -> List[Tuple[float, int]]:
    return sorted(hist.items())


@dataclass
class DRESSResult:
    """Extended result with copies of edge/node data (legacy API)."""

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


def _binom(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


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
    node_weights : sequence of float, optional
        Per-vertex weights. ``None`` for unit weights (all = 1).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    """

    def __init__(
        self,
        n_vertices: int,
        sources: Sequence[int],
        targets: Sequence[int],
        weights_or_variant=None,
        variant_or_precompute=UNDIRECTED,
        precompute_intercepts: bool = False,
        *,
        weights: Optional[Sequence[float]] = None,
        node_weights: Optional[Sequence[float]] = None,
        variant: Optional[Variant] = None,
    ) -> None:
        # Support both calling conventions:
        #   DRESS(n, src, dst, weights, variant, precompute)   -- weighted positional
        #   DRESS(n, src, dst, variant, precompute)            -- unweighted positional (C-extension style)
        #   DRESS(n, src, dst, weights=..., variant=...)       -- keyword style
        if weights is not None or variant is not None:
            # Keyword-style call
            _weights = weights
            _variant = Variant(variant) if variant is not None else UNDIRECTED
        elif isinstance(weights_or_variant, (Variant, int)) and not isinstance(weights_or_variant, bool):
            # Unweighted positional: DRESS(n, src, dst, variant, precompute)
            _weights = None
            _variant = Variant(weights_or_variant)
            if isinstance(variant_or_precompute, bool):
                precompute_intercepts = variant_or_precompute
        else:
            # Weighted positional: DRESS(n, src, dst, weights, variant, precompute)
            _weights = weights_or_variant
            _variant = Variant(variant_or_precompute)

        E = len(sources)
        if len(targets) != E:
            raise ValueError("sources and targets must have equal length")
        if _weights is not None and len(_weights) != E:
            raise ValueError("weights must have the same length as sources")

        self._N = n_vertices
        self._E = E
        self._U = list(sources)
        self._V = list(targets)
        self._variant = Variant(_variant)
        self._precompute_intercepts = precompute_intercepts

        w_in = [1.0] * E if _weights is None else [float(w) for w in _weights]
        self._weights_input = _weights  # original caller weights (None = unweighted)
        self._node_weights = [float(x) for x in node_weights] if node_weights is not None else None

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
    ) -> FitResult:
        """Run iterative fixed-point fitting.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations.
        epsilon : float
            Convergence threshold (max per-edge change).

        Returns
        -------
        FitResult
            Lightweight result with ``iterations`` and ``delta``.
            Edge and node values are available on the graph object
            via ``dress_values``, ``node_dress_values``, etc.
        """
        N, E = self._N, self._E
        adj_offset = self._adj_offset
        adj_target = self._adj_target
        adj_eidx = self._adj_eidx
        ew = self._edge_weight
        ed = self._edge_dress
        ed_next = self._edge_dress_next
        nd = self._node_dress
        nw = self._node_weights

        final_delta = 0.0
        final_iter = max_iterations

        for iteration in range(max_iterations):
            max_delta = 0.0

            # Phase 1: compute node norms (sort+KBN for bitwise reproducibility)
            for u in range(N):
                base = adj_offset[u]
                end = adj_offset[u + 1]
                nw_u = nw[u] if nw is not None else 1.0
                terms = [4.0 * nw_u]
                for k in range(base, end):
                    ei = adj_eidx[k]
                    terms.append(ew[ei] * ed[ei])
                terms.sort()
                # KBN compensated sum
                s = terms[0]
                comp = 0.0
                for i in range(1, len(terms)):
                    t = s + terms[i]
                    if abs(s) >= abs(terms[i]):
                        comp += (s - t) + terms[i]
                    else:
                        comp += (terms[i] - t) + s
                    s = t
                nd[u] = math.sqrt(s + comp)

            # Phase 2: compute next dress values
            for e in range(E):
                u, v = self._U[e], self._V[e]

                # Collect numerator terms for sort+KBN
                terms = []
                iu, iu_end = adj_offset[u], adj_offset[u + 1]
                iv, iv_end = adj_offset[v], adj_offset[v + 1]
                i, j = iu, iv
                while i < iu_end and j < iv_end:
                    x, y = adj_target[i], adj_target[j]
                    if x == y:
                        eu = adj_eidx[i]
                        ev = adj_eidx[j]
                        terms.append(ew[eu] * ed[eu] + ew[ev] * ed[ev])
                        i += 1
                        j += 1
                    elif x < y:
                        i += 1
                    else:
                        j += 1

                # Self-loop + edge's own contribution
                uv = ew[e] * ed[e]
                nw_u = nw[u] if nw is not None else 1.0
                nw_v = nw[v] if nw is not None else 1.0
                if self._variant in (Variant.FORWARD, Variant.BACKWARD):
                    terms.append(4.0 * nw_u + uv)
                else:
                    terms.append(4.0 * nw_u + 4.0 * nw_v + 2.0 * uv)

                # Sort + KBN sum
                terms.sort()
                numerator = terms[0]
                comp = 0.0
                for i in range(1, len(terms)):
                    t = numerator + terms[i]
                    if abs(numerator) >= abs(terms[i]):
                        comp += (numerator - t) + terms[i]
                    else:
                        comp += (terms[i] - t) + numerator
                    numerator = t
                numerator += comp

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

        return FitResult(
            iterations=final_iter,
            delta=final_delta,
        )

    # -- virtual-edge query -------------------------------------------------

    def get(
        self,
        u: int,
        v: int,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        edge_weight: float = 1.0,
    ) -> float:
        """Query the DRESS value for any vertex pair (u, v).

        If edge (u,v) exists, returns its converged dress value.
        If not (virtual edge), estimates it via local fixed-point
        iteration against the frozen steady state.

        The graph must have been fitted (:meth:`fit`) before calling this.

        Parameters
        ----------
        u, v : int
            Vertex ids (0-based).
        max_iterations : int
            Max local iterations for virtual edges (default 100).
        epsilon : float
            Convergence threshold (default 1e-6).
        edge_weight : float
            Hypothetical weight of the virtual edge (default 1.0).

        Returns
        -------
        float
            DRESS similarity value.
        """
        adj_offset = self._adj_offset
        adj_target = self._adj_target
        adj_eidx = self._adj_eidx
        ew = self._edge_weight
        ed = self._edge_dress
        nd = self._node_dress

        # Binary search for existing edge
        lo, hi = adj_offset[u], adj_offset[u + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            nb = adj_target[mid]
            if nb == v:
                return ed[adj_eidx[mid]]
            elif nb < v:
                lo = mid + 1
            else:
                hi = mid

        # Virtual edge: merge-join neighbour lists (sort+KBN for consistency)
        terms = []
        iu, iu_end = adj_offset[u], adj_offset[u + 1]
        iv, iv_end = adj_offset[v], adj_offset[v + 1]
        i, j = iu, iv
        while i < iu_end and j < iv_end:
            x, y = adj_target[i], adj_target[j]
            if x == y:
                eu = adj_eidx[i]
                ev = adj_eidx[j]
                terms.append(ew[eu] * ed[eu] + ew[ev] * ed[ev])
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1

        nw_u = self._node_weights[u] if self._node_weights is not None else 1.0
        nw_v = self._node_weights[v] if self._node_weights is not None else 1.0
        if self._variant in (Variant.FORWARD, Variant.BACKWARD):
            self_loop = 4.0 * nw_u
        else:
            self_loop = 4.0 * nw_u + 4.0 * nw_v
        terms.append(self_loop)

        # Sort + KBN sum
        terms.sort()
        s = terms[0]
        comp = 0.0
        for i in range(1, len(terms)):
            t = s + terms[i]
            if abs(s) >= abs(terms[i]):
                comp += (s - t) + terms[i]
            else:
                comp += (terms[i] - t) + s
            s = t
        A = s + comp

        cross_factor = 1.0 if self._variant in (Variant.FORWARD, Variant.BACKWARD) else 2.0

        cw = 2.0 * edge_weight if self._variant == Variant.UNDIRECTED else edge_weight

        Du2 = nd[u] * nd[u]
        Dv2 = nd[v] * nd[v]

        if Du2 <= 0.0 or Dv2 <= 0.0:
            return 0.0

        d_uv = 1.0
        for _ in range(max_iterations):
            wd = cw * d_uv
            denom = math.sqrt(Du2 + wd) * math.sqrt(Dv2 + wd)
            if denom <= 0.0:
                return 0.0
            next_val = (A + cross_factor * wd) / denom
            if abs(next_val - d_uv) <= epsilon:
                d_uv = next_val
                break
            d_uv = next_val

        return d_uv

    # -- hardware backends ------------------------------------------------

    def delta_fit(self, k: int = 0, max_iterations: int = 100,
                  epsilon: float = 1e-6,
                  n_samples: int = 0, seed: int = 0,
                  keep_multisets: bool = False,
                  compute_histogram: bool = True):
        """Compute the Δ^k-DRESS histogram.

        Exhaustively removes all k-vertex subsets from the graph, runs
        DRESS on each resulting subgraph, and accumulates every converged
        edge value into a single histogram.

        Parameters
        ----------
        k : int
            Deletion depth (0 = original graph).
        max_iterations : int
            Maximum DRESS iterations per subgraph (default 100).
        epsilon : float
            Convergence threshold (default 1e-6).
        keep_multisets : bool
            If True, return per-subgraph DRESS values in a 2D array
            of shape ``(C(N,k), E)``; NaN marks removed edges.
        compute_histogram : bool
            If False, skip histogram construction (default True).

        Returns
        -------
        DeltaDRESSResult
        """
        return delta_fit(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            n_samples=n_samples,
            seed=seed,
            keep_multisets=keep_multisets,
            compute_histogram=compute_histogram,
        )

    def fit_cuda(self, max_iterations: int = 100, epsilon: float = 1e-6):
        """Run DRESS fitting on the GPU.  Requires ``libdress_cuda.so``."""
        from dress.cuda import fit as _cuda_fit
        result = _cuda_fit(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
        self._edge_dress = list(result.edge_dress)
        self._node_dress = list(result.node_dress)
        if hasattr(self, '_np_dress'):
            del self._np_dress
        if hasattr(self, '_np_node_dress'):
            del self._np_node_dress
        return FitResult(iterations=result.iterations, delta=result.delta)

    def delta_fit_cuda(self, k: int = 0, max_iterations: int = 100,
                       epsilon: float = 1e-6, keep_multisets: bool = False,
                       n_samples: int = 0, seed: int = 0,
                       compute_histogram: bool = True):
        """Δ^k-DRESS histogram on the GPU.  Requires ``libdress_cuda.so``."""
        from dress.cuda import delta_fit as _cuda_delta
        return _cuda_delta(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples,
            seed=seed,
            compute_histogram=compute_histogram,
        )

    def delta_fit_mpi(self, k: int = 0, max_iterations: int = 100,
                      epsilon: float = 1e-6, keep_multisets: bool = False,
                      n_samples: int = 0, seed: int = 0,
                      compute_histogram: bool = True,
                      comm=None):
        """Δ^k-DRESS histogram distributed over MPI (CPU).
        Requires ``libdress.so`` built with ``-DDRESS_MPI=ON``."""
        from dress.mpi import delta_fit as _mpi_delta
        return _mpi_delta(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            keep_multisets=keep_multisets,
            comm=comm,
            n_samples=n_samples,
            seed=seed,
            compute_histogram=compute_histogram,
        )

    def delta_fit_mpi_cuda(self, k: int = 0, max_iterations: int = 100,
                           epsilon: float = 1e-6, keep_multisets: bool = False,
                           n_samples: int = 0, seed: int = 0,
                           compute_histogram: bool = True,
                           comm=None):
        """Δ^k-DRESS histogram distributed over MPI with GPU acceleration.
        Requires ``libdress.so`` built with ``-DDRESS_MPI=ON`` + CUDA."""
        from dress.mpi.cuda import delta_fit as _mpi_cuda_delta
        return _mpi_cuda_delta(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            keep_multisets=keep_multisets,
            comm=comm,
            n_samples=n_samples,
            seed=seed,
            compute_histogram=compute_histogram,
        )

    def fit_omp(self, max_iterations: int = 100, epsilon: float = 1e-6):
        """Run DRESS fitting with OpenMP edge-parallelism."""
        from dress.omp import fit as _omp_fit
        result = _omp_fit(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
        self._edge_dress = list(result.edge_dress)
        self._node_dress = list(result.node_dress)
        if hasattr(self, '_np_dress'):
            del self._np_dress
        if hasattr(self, '_np_node_dress'):
            del self._np_node_dress
        return FitResult(iterations=result.iterations, delta=result.delta)

    def delta_fit_omp(self, k: int = 0, max_iterations: int = 100,
                      epsilon: float = 1e-6, keep_multisets: bool = False,
                      n_samples: int = 0, seed: int = 0,
                      compute_histogram: bool = True):
        """Δ^k-DRESS histogram with OpenMP subgraph-parallelism."""
        from dress.omp import delta_fit as _omp_delta
        return _omp_delta(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples,
            seed=seed,
            compute_histogram=compute_histogram,
        )

    def delta_fit_mpi_omp(self, k: int = 0, max_iterations: int = 100,
                          epsilon: float = 1e-6, keep_multisets: bool = False,
                          n_samples: int = 0, seed: int = 0,
                          compute_histogram: bool = True,
                          comm=None):
        """Δ^k-DRESS histogram distributed over MPI with OpenMP parallelism
        within each rank."""
        from dress.mpi.omp import delta_fit as _mpi_omp_delta
        return _mpi_omp_delta(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            keep_multisets=keep_multisets,
            comm=comm,
            n_samples=n_samples,
            seed=seed,
            compute_histogram=compute_histogram,
        )

    # -- nabla fitting (∇^k-DRESS) -----------------------------------------

    def nabla_fit(self, k: int = 0, max_iterations: int = 100,
                  epsilon: float = 1e-6,
                  n_samples: int = 0, seed: int = 0,
                  keep_multisets: bool = False,
                  compute_histogram: bool = True):
        """Compute the ∇^k-DRESS histogram (CPU, sequential).

        Enumerates all P(N,k) ordered k-tuples, marks each with
        generic injective node weights, runs DRESS on each marked
        graph, and accumulates edge values into a histogram.

        Returns
        -------
        NablaDRESSResult
        """
        return nabla_fit(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k,
            variant=self._variant,
            max_iterations=max_iterations,
            epsilon=epsilon,
            n_samples=n_samples,
            seed=seed,
            keep_multisets=keep_multisets,
            compute_histogram=compute_histogram,
        )

    def nabla_fit_cuda(self, k: int = 0, max_iterations: int = 100,
                       epsilon: float = 1e-6,
                       n_samples: int = 0, seed: int = 0,
                       keep_multisets: bool = False,
                       compute_histogram: bool = True):
        """∇^k-DRESS on the GPU."""
        from dress.cuda import nabla_fit as _cuda_nabla
        return _cuda_nabla(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k, variant=self._variant,
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit_omp(self, k: int = 0, max_iterations: int = 100,
                      epsilon: float = 1e-6,
                      n_samples: int = 0, seed: int = 0,
                      keep_multisets: bool = False,
                      compute_histogram: bool = True):
        """∇^k-DRESS with OpenMP tuple-parallelism."""
        from dress.omp import nabla_fit as _omp_nabla
        return _omp_nabla(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k, variant=self._variant,
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit_mpi(self, k: int = 0, max_iterations: int = 100,
                      epsilon: float = 1e-6,
                      n_samples: int = 0, seed: int = 0,
                      keep_multisets: bool = False,
                      compute_histogram: bool = True,
                      comm=None):
        """∇^k-DRESS distributed over MPI (CPU)."""
        from dress.mpi import nabla_fit as _mpi_nabla
        return _mpi_nabla(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k, variant=self._variant,
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit_mpi_cuda(self, k: int = 0, max_iterations: int = 100,
                           epsilon: float = 1e-6,
                           n_samples: int = 0, seed: int = 0,
                           keep_multisets: bool = False,
                           compute_histogram: bool = True,
                           comm=None):
        """∇^k-DRESS distributed over MPI with GPU acceleration."""
        from dress.mpi.cuda import nabla_fit as _mpi_cuda_nabla
        return _mpi_cuda_nabla(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k, variant=self._variant,
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit_mpi_omp(self, k: int = 0, max_iterations: int = 100,
                          epsilon: float = 1e-6,
                          n_samples: int = 0, seed: int = 0,
                          keep_multisets: bool = False,
                          compute_histogram: bool = True,
                          comm=None):
        """∇^k-DRESS distributed over MPI with OpenMP parallelism."""
        from dress.mpi.omp import nabla_fit as _mpi_omp_nabla
        return _mpi_omp_nabla(
            self._N, self._U, self._V,
            weights=self._weights_input,
            node_weights=self._node_weights,
            k=k, variant=self._variant,
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )


# -- top-level convenience function -----------------------------------------

def fit(
    n_vertices: int,
    sources: Sequence[int],
    targets: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    node_weights: Optional[Sequence[float]] = None,
    variant: Variant = UNDIRECTED,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute_intercepts: bool = False,
) -> DRESSResult:
    """Compute DRESS similarity for a graph and return all results.

    This is the primary Python API.  It mirrors the functional style used
    by the Rust, Go, Julia, R, MATLAB, and JavaScript bindings.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted, i.e. all 1).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    max_iterations : int
        Maximum number of fix-point iterations (default 100).
    epsilon : float
        Convergence threshold (default 1e-6).
    precompute_intercepts : bool
        Pre-compute common-neighbour index (default ``False``).

    Returns
    -------
    DRESSResult
        Dataclass with fields ``sources``, ``targets``, ``edge_dress``,
        ``edge_weight``, ``node_dress``, ``iterations``, and ``delta``.
    """
    g = DRESS(n_vertices, sources, targets, weights=weights,
              node_weights=node_weights, variant=variant,
              precompute_intercepts=precompute_intercepts)
    fr = g.fit(max_iterations=max_iterations, epsilon=epsilon)
    return DRESSResult(
        sources=list(g._U),
        targets=list(g._V),
        edge_dress=list(g._edge_dress),
        edge_weight=list(g._edge_weight),
        node_dress=list(g._node_dress),
        iterations=fr.iterations,
        delta=fr.delta,
    )



def delta_fit(
    n_vertices: int,
    sources: Sequence[int],
    targets: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    node_weights: Optional[Sequence[float]] = None,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    n_samples: int = 0,
    seed: int = 0,
    precompute: bool = False,
    keep_multisets: bool = False,
    compute_histogram: bool = True,
) -> DeltaDRESSResult:
    """Compute the Δ^k-DRESS histogram.

    Exhaustively removes all k-vertex subsets from the graph, runs DRESS
    on each resulting subgraph, and accumulates every converged edge value
    into a single histogram binned by *epsilon*.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted, i.e. all 1).
    k : int
        Deletion depth — number of vertices removed per subset.
        ``k=0`` runs DRESS on the original graph (Δ^0).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    max_iterations : int
        Maximum DRESS iterations per subgraph (default 100).
    epsilon : float
        Convergence threshold and histogram bin width (default 1e-6).
    precompute : bool
        Pre-compute common-neighbour index (default ``False``).
    keep_multisets : bool
        If ``True``, return per-subgraph DRESS values in a 2-D list
        of shape ``(C(N,k), E)``.  ``NaN`` marks removed edges.

    Returns
    -------
    DeltaDRESSResult
        Dataclass with ``histogram`` (dict: value -> count).
        If *keep_multisets* is ``True``, also
        ``multisets`` (list of lists) and ``num_subgraphs``.
    """
    N = n_vertices
    E = len(sources)

    src = list(sources)
    tgt = list(targets)
    wgt: Optional[List[float]] = list(weights) if weights is not None else None
    nwgt: Optional[List[float]] = list(node_weights) if node_weights is not None else None

    # Build the full graph once.
    full_g = DRESS(N, list(src), list(tgt),
                   weights=list(wgt) if wgt is not None else None,
                   node_weights=list(nwgt) if nwgt is not None else None,
                   variant=variant)
    hist: Dict[float, int] = {}
    do_hist = compute_histogram

    # Compute C(N, k) and optionally allocate multisets buffer.
    cnk: int = 1 if k == 0 else _binom(N, k)
    wants_ms = bool(keep_multisets)
    ms: Optional[List[List[float]]] = None
    if wants_ms:
        ms = [[float('nan')] * E for _ in range(cnk)]

    def _accumulate_histogram(g: DRESS) -> None:
        """Accumulate converged edge dress values into *hist*."""
        for e in range(g.n_edges):
            d = g._edge_dress[e]
            hist[d] = hist.get(d, 0) + 1

    def _fill_multiset_row(g: DRESS, edge_map: List[int], row: int) -> None:
        """Fill one row of the multisets matrix.

        multisets[row][e] = sub.edge_dress[edge_map[e]], or NaN if
        edge_map[e] == -1 (edge was removed).
        """
        assert ms is not None
        for e in range(E):
            if edge_map[e] >= 0:
                ms[row][e] = g._edge_dress[edge_map[e]]
            # else: already NaN from initialisation

    # ── k = 0: Δ^0 — full graph ────────────────────────────────
    if k == 0:
        # Reuse the already-constructed full graph.
        full_g.fit(max_iterations=max_iterations, epsilon=epsilon)
        if do_hist:
            _accumulate_histogram(full_g)

        if wants_ms:
            # k=0: single subgraph (s=0), identity edge map.
            assert ms is not None
            for e in range(E):
                ms[0][e] = full_g._edge_dress[e]

        return DeltaDRESSResult(
            histogram=_flatten_histogram(hist),
            multisets=ms, num_subgraphs=cnk,
        )

    # ── k >= N: no valid deletion subsets ───────────────────────
    if k >= N:
        return DeltaDRESSResult(
            histogram=_flatten_histogram(hist),
            multisets=ms, num_subgraphs=cnk,
        )

    # ── Sampling setup ─────────────────────────────────────────
    import random as _random

    use_sampling = n_samples > 0 and n_samples < cnk
    picks: Optional[List[int]] = None
    if use_sampling:
        rng = _random.Random(seed)
        picks = sorted(rng.sample(range(cnk), n_samples))

    pick_idx = 0

    # ── k >= 1: iterative DFS over C(N, k) combinations ────────
    combo = [0] * k
    combo[0] = -1
    depth = 0
    s = 0               # subgraph counter (for multiset rows)

    while depth >= 0:
        combo[depth] += 1

        # Upper bound: ensure room for remaining slots.
        if combo[depth] > N - k + depth:
            depth -= 1
            continue

        if depth == k - 1:
            # Sampling: skip combos not in picks list.
            if use_sampling:
                assert picks is not None
                if pick_idx >= len(picks):
                    break
                if s != picks[pick_idx]:
                    s += 1
                    continue
                pick_idx += 1

            # Complete k-subset: combo[0..k-1]
            deleted = set(combo[:k])

            # Build vertex remapping
            node_map = [-1] * N
            new_id = 0
            for v in range(N):
                if v not in deleted:
                    node_map[v] = new_id
                    new_id += 1
            sub_n = new_id

            # Build subgraph edge list (and edge map)
            sub_src: List[int] = []
            sub_tgt: List[int] = []
            sub_wgt_list: Optional[List[float]] = [] if wgt is not None else None
            edge_map: List[int] = [-1] * E
            sub_e_idx = 0
            for e in range(E):
                mu = node_map[src[e]]
                mv = node_map[tgt[e]]
                if mu >= 0 and mv >= 0:
                    sub_src.append(mu)
                    sub_tgt.append(mv)
                    if sub_wgt_list is not None:
                        sub_wgt_list.append(wgt[e])
                    edge_map[e] = sub_e_idx
                    sub_e_idx += 1

            if sub_src:
                sub_nwgt = None
                if nwgt is not None:
                    sub_nwgt = [nwgt[v] for v in range(N) if v not in deleted]
                g = DRESS(sub_n, sub_src, sub_tgt,
                          weights=sub_wgt_list if sub_wgt_list else None,
                          node_weights=sub_nwgt,
                          variant=variant)
                g.fit(max_iterations=max_iterations, epsilon=epsilon)
                if do_hist:
                    _accumulate_histogram(g)
                if wants_ms:
                    _fill_multiset_row(g, edge_map, s)
            # else: zero-edge subgraph — ms[s] already all NaN

            s += 1
        else:
            # Descend: seed next depth from current value
            depth += 1
            combo[depth] = combo[depth - 1]  # incremented at top

    return DeltaDRESSResult(
        histogram=_flatten_histogram(hist),
        multisets=ms, num_subgraphs=cnk,
    )


def nabla_fit(
    n_vertices: int,
    sources: Sequence[int],
    targets: Sequence[int],
    weights: Optional[Sequence[float]] = None,
    node_weights: Optional[Sequence[float]] = None,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    n_samples: int = 0,
    seed: int = 0,
    precompute: bool = False,
    keep_multisets: bool = False,
    compute_histogram: bool = True,
) -> NablaDRESSResult:
    """Compute the ∇^k-DRESS histogram (pure Python).

    Enumerates all P(N,k) ordered k-tuples, marks each with
    generic injective node weights (sqrt of successive primes),
    runs DRESS on each marked graph, and accumulates edge values.

    Returns
    -------
    NablaDRESSResult
    """
    import math as _math

    N = n_vertices
    E = len(sources)
    src = list(sources)
    tgt = list(targets)
    wgt: Optional[List[float]] = list(weights) if weights is not None else None
    nwgt: Optional[List[float]] = list(node_weights) if node_weights is not None else None

    # P(N, k)
    pnk = 1
    for i in range(k):
        pnk *= (N - i)

    # Generic marker weights: sqrt(prime(i+1))
    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    marker_weights = [_math.sqrt(float(_PRIMES[i])) if i < len(_PRIMES)
                      else _math.sqrt(float(2 * i + 3))
                      for i in range(k)]

    do_hist = compute_histogram
    wants_ms = keep_multisets

    hist: Dict[float, int] = {}
    ms = None
    if wants_ms:
        ms = [float('nan')] * (pnk * E)

    def _accumulate_histogram(g: DRESS) -> None:
        for e_idx in range(g.n_edges):
            v = g.edge_dress(e_idx)
            hist[v] = hist.get(v, 0) + 1

    def _fill_multiset_row(g: DRESS, row_idx: int) -> None:
        base = row_idx * E
        for e_idx in range(g.n_edges):
            ms[base + e_idx] = g.edge_dress(e_idx)

    if k == 0:
        g = DRESS(N, src, tgt, weights=wgt, node_weights=nwgt, variant=variant)
        g.fit(max_iterations=max_iterations, epsilon=epsilon)
        if do_hist:
            _accumulate_histogram(g)
        if wants_ms:
            _fill_multiset_row(g, 0)
        return NablaDRESSResult(
            histogram=_flatten_histogram(hist),
            multisets=ms, num_tuples=1,
        )

    # Iterative DFS over ordered k-tuples (without repetition)
    tuple_buf = [0] * k
    used = [False] * N
    depth = 0
    tuple_buf[0] = -1
    s = 0

    while depth >= 0:
        tuple_buf[depth] += 1

        # Un-use previous vertex
        if tuple_buf[depth] > 0 and tuple_buf[depth] <= N:
            prev = tuple_buf[depth] - 1
            if prev < N and used[prev]:
                used[prev] = False

        # Skip used vertices
        while tuple_buf[depth] < N and used[tuple_buf[depth]]:
            tuple_buf[depth] += 1

        if tuple_buf[depth] >= N:
            depth -= 1
            if depth >= 0:
                used[tuple_buf[depth]] = False
            continue

        used[tuple_buf[depth]] = True

        if depth == k - 1:
            # Complete tuple found — build marked graph
            marked_nw = [1.0] * N
            if nwgt is not None:
                marked_nw = list(nwgt)
            for i in range(k):
                marked_nw[tuple_buf[i]] = marker_weights[i]

            g = DRESS(N, src, tgt, weights=wgt, node_weights=marked_nw, variant=variant)
            g.fit(max_iterations=max_iterations, epsilon=epsilon)
            if do_hist:
                _accumulate_histogram(g)
            if wants_ms:
                _fill_multiset_row(g, s)

            s += 1
            used[tuple_buf[depth]] = False
        else:
            depth += 1
            tuple_buf[depth] = -1

    return NablaDRESSResult(
        histogram=_flatten_histogram(hist),
        multisets=ms, num_tuples=pnk,
    )

