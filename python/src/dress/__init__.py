"""
dress - Python bindings for the DRESS (Diffusive Recursive Structural
Similarity) graph library.

Uses the compiled C extension (``dress._core``) when available, otherwise
falls back to the pure-Python implementation (``dress.core``).
"""

from dress.core import DRESSResult

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
