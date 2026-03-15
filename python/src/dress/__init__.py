"""
dress — Python bindings for the DRESS graph library.

Uses the compiled C extension (``dress._core``) when available, otherwise
falls back to the pure-Python implementation (``dress.core``).

The :class:`DRESS` class exposed here has the **same API** regardless of
which backend is active — including hardware-accelerated methods
(``fit_cuda``, ``delta_fit_cuda``, ``delta_fit_mpi``, ``delta_fit_mpi_cuda``).
"""

from dress.core import (
    DRESSResult,
    DeltaDRESSResult,
    FitResult,
    Variant,
    UNDIRECTED,
    DIRECTED,
    FORWARD,
    BACKWARD,
)

# ---------------------------------------------------------------------------
#  Backend detection
# ---------------------------------------------------------------------------

try:
    import dress._core as _core  # noqa: F401

    _BACKEND = "c"
except ImportError:
    _core = None
    _BACKEND = "python"


# ---------------------------------------------------------------------------
#  Unified DRESS class
# ---------------------------------------------------------------------------

class DRESS:
    """DRESS graph — unified API over all backends.

    Switch backends by changing the import::

        from dress import DRESS           # CPU
        from dress.cuda import DRESS      # CUDA
        from dress.mpi import DRESS       # MPI CPU
        from dress.mpi.cuda import DRESS  # MPI + CUDA

    All share the same methods: ``fit()``, ``delta_fit()``, ``get()``.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights.  ``None`` for unweighted (all weights = 1).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    precompute_intercepts : bool
        Pre-compute common-neighbour index (default ``False``).
    """

    _force_python_impl = False

    def __init__(
        self,
        n_vertices,
        sources,
        targets,
        weights_or_variant=None,
        variant_or_precompute=UNDIRECTED,
        precompute_intercepts=False,
        *,
        weights=None,
        variant=None,
    ):
        # Resolve the flexible positional / keyword calling conventions:
        #   DRESS(n, s, t, weights, variant, precompute)   -- weighted positional
        #   DRESS(n, s, t, variant, precompute)            -- unweighted positional
        #   DRESS(n, s, t, weights=..., variant=...)       -- keyword
        if weights is not None or variant is not None:
            _weights = weights
            _variant = variant if variant is not None else UNDIRECTED
            if isinstance(variant_or_precompute, bool):
                precompute_intercepts = variant_or_precompute
        elif isinstance(weights_or_variant, (int, Variant)) and not isinstance(weights_or_variant, bool):
            _weights = None
            _variant = weights_or_variant
            if isinstance(variant_or_precompute, bool):
                precompute_intercepts = variant_or_precompute
        elif weights_or_variant is not None:
            _weights = weights_or_variant
            _variant = variant_or_precompute
        else:
            _weights = None
            _variant = variant_or_precompute

        self._n_v = int(n_vertices)
        self._src = list(sources)
        self._tgt = list(targets)
        self._wgt = list(_weights) if _weights is not None else None
        self._var = Variant(int(_variant))
        self._precompute = bool(precompute_intercepts)

        # Build the backend-specific graph object.
        # Subclasses (cuda, mpi, …) set _force_python_impl = True so that
        # hardware fit results can be written back for get() to work.
        if _BACKEND == "c" and not self._force_python_impl:
            cv = _core.Variant(int(self._var))
            if self._wgt is not None:
                self._impl = _core.DRESS(
                    self._n_v, self._src, self._tgt,
                    self._wgt, cv, self._precompute,
                )
            else:
                self._impl = _core.DRESS(
                    self._n_v, self._src, self._tgt,
                    cv, self._precompute,
                )
        else:
            from dress.core import DRESS as _PyDRESS
            self._impl = _PyDRESS(
                self._n_v, self._src, self._tgt,
                weights=self._wgt, variant=self._var,
                precompute_intercepts=self._precompute,
            )

    # -- properties --------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices."""
        return self._impl.n_vertices

    @property
    def n_edges(self):
        """Number of edges."""
        return self._impl.n_edges

    @property
    def variant(self):
        """Graph variant."""
        return self._var

    # -- per-element accessors ---------------------------------------------

    def edge_source(self, e):
        """Source vertex of edge *e*."""
        return self._impl.edge_source(e)

    def edge_target(self, e):
        """Target vertex of edge *e*."""
        return self._impl.edge_target(e)

    def edge_weight(self, e):
        """Combined weight of edge *e*."""
        return self._impl.edge_weight(e)

    def edge_dress(self, e):
        """DRESS value of edge *e* (call :meth:`fit` first)."""
        return self._impl.edge_dress(e)

    def node_dress(self, u):
        """Node DRESS norm of vertex *u* (call :meth:`fit` first)."""
        return self._impl.node_dress(u)

    # -- NumPy array properties (zero-copy for C, lazy for Python) ---------

    @property
    def sources(self):
        """Edge source array (NumPy)."""
        return self._impl.sources

    @property
    def targets(self):
        """Edge target array (NumPy)."""
        return self._impl.targets

    @property
    def weights(self):
        """Combined edge weight array (NumPy)."""
        return self._impl.weights

    @property
    def dress_values(self):
        """Per-edge DRESS similarity array (NumPy)."""
        return self._impl.dress_values

    @property
    def node_dress_values(self):
        """Per-node DRESS norm array (NumPy)."""
        return self._impl.node_dress_values

    # -- fitting -----------------------------------------------------------

    def fit(self, max_iterations=100, epsilon=1e-6):
        """Run iterative fixed-point DRESS fitting.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations (default 100).
        epsilon : float
            Convergence threshold (default 1e-6).

        Returns
        -------
        FitResult
            With ``iterations`` and ``delta`` fields.
        """
        fr = self._impl.fit(max_iterations, epsilon)
        if _BACKEND == "c":
            return FitResult(iterations=fr.iterations, delta=fr.delta)
        return fr

    def delta_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  keep_multisets=False, offset=0, stride=1):
        """Compute the Δ^k-DRESS histogram.

        Exhaustively removes all k-vertex subsets, runs DRESS on each
        subgraph, and accumulates edge values into a histogram.

        Parameters
        ----------
        k : int
            Deletion depth (default 0 = original graph).
        max_iterations : int
            Max DRESS iterations per subgraph (default 100).
        epsilon : float
            Convergence threshold and histogram bin width (default 1e-6).
        keep_multisets : bool
            If True, return per-subgraph DRESS values (default False).
        offset : int
            Process only subgraphs where ``index % stride == offset``.
        stride : int
            Total number of strides (default 1 = all).

        Returns
        -------
        DeltaDRESSResult
        """
        if _BACKEND == "c":
            dr = self._impl.delta_fit(
                k, max_iterations, epsilon,
                keep_multisets, offset, stride,
            )
            ms = None
            ns = 0
            if keep_multisets and dr.multisets is not None:
                ms = dr.multisets
                ns = dr.num_subgraphs
            return DeltaDRESSResult(
                histogram=list(dr.histogram),
                hist_size=dr.hist_size,
                multisets=ms,
                num_subgraphs=ns,
            )
        else:
            return self._impl.delta_fit(
                k=k, max_iterations=max_iterations, epsilon=epsilon,
                keep_multisets=keep_multisets, offset=offset, stride=stride,
            )

    def get(self, u, v, max_iterations=100, epsilon=1e-6, edge_weight=1.0):
        """Query the DRESS value for any vertex pair *(u, v)*.

        If the edge exists, returns its converged value.  Otherwise
        estimates it via local fixed-point iteration (virtual edge).

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
        """
        return self._impl.get(u, v, max_iterations, epsilon, edge_weight)

    # -- result syncing (used by hardware subclasses) ----------------------

    def _sync_hardware_fit(self, result):
        """Copy hardware-backend fit results into the pure-Python impl.

        After this, ``edge_dress()``, ``node_dress()``, ``get()`` etc.
        all reflect the hardware-computed values.
        """
        self._impl._edge_dress = list(result.edge_dress)
        self._impl._node_dress = list(result.node_dress)
        for attr in ('_np_dress', '_np_node_dress'):
            if hasattr(self._impl, attr):
                delattr(self._impl, attr)

    # -- hardware-accelerated convenience (suffix API) ---------------------

    def fit_cuda(self, max_iterations=100, epsilon=1e-6):
        """Run DRESS fitting on the GPU.  Requires ``libdress_cuda.so``."""
        from dress.cuda import dress_fit as _cuda_fit
        result = _cuda_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
        )
        if self._force_python_impl or _BACKEND == "python":
            self._sync_hardware_fit(result)
        return FitResult(iterations=result.iterations, delta=result.delta)

    def delta_fit_cuda(self, k=0, max_iterations=100, epsilon=1e-6,
                       keep_multisets=False, offset=0, stride=1):
        """Δ^k-DRESS histogram on the GPU."""
        from dress.cuda import delta_dress_fit as _cuda_delta
        return _cuda_delta(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets,
            offset=offset, stride=stride,
        )

    def delta_fit_mpi(self, k=0, max_iterations=100, epsilon=1e-6,
                      keep_multisets=False, comm=None):
        """Δ^k-DRESS histogram distributed over MPI (CPU)."""
        from dress.mpi import delta_dress_fit as _mpi_delta
        return _mpi_delta(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
        )

    def delta_fit_mpi_cuda(self, k=0, max_iterations=100, epsilon=1e-6,
                           keep_multisets=False, comm=None):
        """Δ^k-DRESS histogram distributed over MPI with GPU acceleration."""
        from dress.mpi.cuda import delta_dress_fit as _mpi_cuda_delta
        return _mpi_cuda_delta(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
        )

    # -- repr --------------------------------------------------------------

    def __repr__(self):
        return (
            f"DRESS(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"variant={self._var.name})"
        )


# ---------------------------------------------------------------------------
#  Module-level convenience functions
# ---------------------------------------------------------------------------

def dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute_intercepts=False,
):
    """Compute DRESS similarity for a graph and return all results.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted).
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
    """
    g = DRESS(n_vertices, sources, targets, weights=weights,
              variant=variant, precompute_intercepts=precompute_intercepts)
    fr = g.fit(max_iterations=max_iterations, epsilon=epsilon)
    E = g.n_edges
    return DRESSResult(
        sources=[g.edge_source(e) for e in range(E)],
        targets=[g.edge_target(e) for e in range(E)],
        edge_dress=[g.edge_dress(e) for e in range(E)],
        edge_weight=[g.edge_weight(e) for e in range(E)],
        node_dress=[g.node_dress(u) for u in range(g.n_vertices)],
        iterations=fr.iterations,
        delta=fr.delta,
    )


def delta_dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    offset=0,
    stride=1,
):
    """Compute the Δ^k-DRESS histogram.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted).
    k : int
        Deletion depth (default 0).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    max_iterations : int
        Max DRESS iterations per subgraph (default 100).
    epsilon : float
        Convergence threshold and histogram bin width (default 1e-6).
    precompute : bool
        Pre-compute common-neighbour index (default ``False``).
    keep_multisets : bool
        If True, return per-subgraph DRESS values (default ``False``).
    offset : int
        Process only subgraphs where ``index % stride == offset``.
    stride : int
        Total number of strides (default 1 = all).

    Returns
    -------
    DeltaDRESSResult
    """
    g = DRESS(n_vertices, sources, targets, weights=weights,
              variant=variant, precompute_intercepts=precompute)
    return g.delta_fit(
        k=k, max_iterations=max_iterations, epsilon=epsilon,
        keep_multisets=keep_multisets, offset=offset, stride=stride,
    )
