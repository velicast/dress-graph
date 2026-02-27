"""
dress - Python bindings for the DRESS graph library.

Uses the compiled C extension (``dress._core``) when available, otherwise
falls back to the pure-Python implementation (``dress.core``).
"""

from dress.core import DRESSResult, DeltaDRESSResult, NablaDRESSResult

try:
    from dress._core import DRESS as _DRESS_cls  # noqa: F401
    from dress._core import UNDIRECTED, DIRECTED, FORWARD, BACKWARD

    _BACKEND = "c"
except ImportError:
    from dress.core import DRESS as _DRESS_cls  # noqa: F401
    from dress.core import UNDIRECTED, DIRECTED, FORWARD, BACKWARD

    _BACKEND = "python"

# Re-export the Variant enum from whichever backend is active
try:
    from dress._core import Variant
except ImportError:
    from dress.core import Variant

# Re-export
DRESS = _DRESS_cls


def dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
):
    """Compute DRESS similarity for a graph and return all results.

    This is the primary Python API.  It creates the internal DRESS
    object, runs the iterative fitting, and returns a single
    :class:`DRESSResult` containing every output array.

    Works identically whether the C extension or the pure-Python
    backend is active.

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

    Returns
    -------
    DRESSResult
        Dataclass with fields ``sources``, ``targets``, ``edge_dress``,
        ``edge_weight``, ``node_dress``, ``iterations``, and ``delta``.
    """
    if _BACKEND == "c":
        import dress._core as _core
        # Convert variant to C extension's Variant type
        _cv = _core.Variant(int(variant))
        # C extension: positional constructor
        if weights is not None:
            g = _DRESS_cls(n_vertices, list(sources), list(targets),
                           list(weights), _cv)
        else:
            g = _DRESS_cls(n_vertices, list(sources), list(targets), _cv)
        fr = g.fit(max_iterations, epsilon)
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
    else:
        # Pure-Python: use the dress_fit function directly
        from dress.core import dress_fit as _py_dress_fit
        return _py_dress_fit(
            n_vertices, sources, targets,
            weights=weights, variant=variant,
            max_iterations=max_iterations, epsilon=epsilon,
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
):
    """Compute the Δ^k-DRESS histogram.

    Exhaustively removes all k-vertex subsets from the graph, runs DRESS
    on each resulting subgraph, and accumulates every converged edge value
    into a single histogram binned by *epsilon*.

    Works identically whether the C extension or the pure-Python backend
    is active.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted, i.e. all 1).
    k : int
        Deletion depth — vertices removed per subset (default 0).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    max_iterations : int
        Maximum DRESS iterations per subgraph (default 100).
    epsilon : float
        Convergence threshold and histogram bin width (default 1e-6).
    precompute : bool
        Pre-compute common-neighbour index (default ``False``).
    keep_multisets : bool
        If True, return per-subgraph DRESS values in a 2D array of shape
        (C(N,k), E).  NaN marks removed edges (default ``False``).

    Returns
    -------
    DeltaDRESSResult
        Dataclass with ``histogram`` (list of int) and ``hist_size``.
        If *keep_multisets* is True, also ``multisets`` and ``num_subgraphs``.
    """
    if _BACKEND == "c":
        import dress._core as _core
        _cv = _core.Variant(int(variant))
        if weights is not None:
            g = _DRESS_cls(n_vertices, list(sources), list(targets),
                           list(weights), _cv, precompute)
        else:
            g = _DRESS_cls(n_vertices, list(sources), list(targets), _cv,
                           precompute)
        dr = g.delta_fit(k, max_iterations, epsilon, keep_multisets)
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
        from dress.core import delta_dress_fit as _py_delta
        return _py_delta(
            n_vertices, sources, targets,
            weights=weights, k=k, variant=variant,
            max_iterations=max_iterations, epsilon=epsilon,
            precompute=precompute,
        )


def nabla_dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    k=0,
    nabla_weight=2.0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
):
    """Compute the ∇^k-DRESS histogram.

    Exhaustively individualizes all k-vertex subsets by multiplying
    incident edge weights by *nabla_weight*, runs DRESS on the same
    topology, and accumulates every converged edge value into a single
    histogram binned by *epsilon*.

    Works identically whether the C extension or the pure-Python
    backend is active.

    Parameters
    ----------
    n_vertices : int
        Number of vertices (0-indexed).
    sources, targets : sequence of int
        Edge endpoint arrays (same length).
    weights : sequence of float, optional
        Per-edge weights (``None`` for unweighted, i.e. all 1).
    k : int
        Individualization depth — vertices marked per subset (default 0).
    nabla_weight : float
        Multiplicative factor for incident edges (default 2.0).
    variant : Variant
        ``UNDIRECTED`` (default), ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    max_iterations : int
        Maximum DRESS iterations per round (default 100).
    epsilon : float
        Convergence threshold and histogram bin width (default 1e-6).
    precompute : bool
        Pre-compute common-neighbour index (default ``False``).
    keep_multisets : bool
        If True, return per-subset DRESS values in a 2D array of shape
        (C(N,k), E) (default ``False``).

    Returns
    -------
    NablaDRESSResult
        Dataclass with ``histogram`` (list of int) and ``hist_size``.
        If *keep_multisets* is True, also ``multisets`` and ``num_subsets``.
    """
    if _BACKEND == "c":
        import dress._core as _core
        _cv = _core.Variant(int(variant))
        if weights is not None:
            g = _DRESS_cls(n_vertices, list(sources), list(targets),
                           list(weights), _cv, precompute)
        else:
            g = _DRESS_cls(n_vertices, list(sources), list(targets), _cv,
                           precompute)
        nr = g.nabla_fit(k, max_iterations, epsilon, nabla_weight,
                         keep_multisets)
        ms = None
        ns = 0
        if keep_multisets and nr.multisets is not None:
            ms = nr.multisets
            ns = nr.num_subsets
        return NablaDRESSResult(
            histogram=list(nr.histogram),
            hist_size=nr.hist_size,
            multisets=ms,
            num_subsets=ns,
        )
    else:
        from dress.core import nabla_dress_fit as _py_nabla
        return _py_nabla(
            n_vertices, sources, targets,
            weights=weights, k=k, nabla_weight=nabla_weight,
            variant=variant, max_iterations=max_iterations,
            epsilon=epsilon, precompute=precompute,
        )
