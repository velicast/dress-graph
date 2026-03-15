"""
NetworkX helpers for DRESS.

Usage::

    import networkx as nx

    # CPU (default — uses C extension or pure-Python fallback)
    from dress.networkx import dress_graph, delta_dress_graph

    result = dress_graph(G)
    delta  = delta_dress_graph(G, k=1)

    # GPU (requires libdress_cuda.so) — same API, different import
    from dress.cuda.networkx import dress_graph, delta_dress_graph

    result = dress_graph(G)
    delta  = delta_dress_graph(G, k=1)

    # Set attributes on the graph directly
    dress_graph(G, set_attributes=True)
"""

from __future__ import annotations

from dress.core import DRESSResult, DeltaDRESSResult, Variant, UNDIRECTED

__all__ = ["dress_graph", "delta_dress_graph", "NxDRESS"]


# ---------------------------------------------------------------------------
# Shared helpers (also used by dress.cuda.networkx)
# ---------------------------------------------------------------------------

def _extract_edges(G, weight_attr="weight"):
    """Convert a NetworkX graph to 0-indexed edge arrays.

    Returns (n_vertices, sources, targets, weights_or_None, node_list).
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    sources = []
    targets = []
    weights = []
    has_weights = False

    for u, v, data in G.edges(data=True):
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
        w = data.get(weight_attr, 1.0)
        weights.append(float(w))
        if w != 1.0:
            has_weights = True

    return len(nodes), sources, targets, (weights if has_weights else None), nodes


def _set_graph_attributes(G, result, nodes):
    """Write DRESS results back onto a NetworkX graph."""
    for i, (s, t) in enumerate(zip(result.sources, result.targets)):
        u_label = nodes[s]
        v_label = nodes[t]
        if G.has_edge(u_label, v_label):
            G[u_label][v_label]["dress"] = result.edge_dress[i]
    for i, n in enumerate(nodes):
        G.nodes[n]["dress_norm"] = result.node_dress[i]


def _dress_graph_impl(
    fit_fn,
    G,
    *,
    variant=UNDIRECTED,
    weight="weight",
    max_iterations=100,
    epsilon=1e-6,
    precompute_intercepts=False,
    set_attributes=False,
):
    """Core implementation shared by CPU and CUDA wrappers."""
    n_vertices, sources, targets, weights, nodes = _extract_edges(G, weight)

    result = fit_fn(
        n_vertices, sources, targets,
        weights=weights, variant=variant,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute_intercepts=precompute_intercepts,
    )

    if set_attributes:
        _set_graph_attributes(G, result, nodes)

    return result


def _delta_dress_graph_impl(
    delta_fn,
    G,
    *,
    k=0,
    variant=UNDIRECTED,
    weight="weight",
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    **kwargs,
):
    """Core implementation shared by CPU, CUDA, and MPI wrappers."""
    n_vertices, sources, targets, weights, _nodes = _extract_edges(G, weight)

    return delta_fn(
        n_vertices, sources, targets,
        weights=weights, k=k, variant=variant,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public CPU API
# ---------------------------------------------------------------------------

def dress_graph(
    G,
    *,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute_intercepts: bool = False,
    set_attributes: bool = False,
) -> DRESSResult:
    """Compute DRESS similarity on a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    variant : Variant
        ``UNDIRECTED``, ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    weight : str
        Edge attribute name for weights. If the attribute is missing the
        edge is treated as unweighted (weight = 1).
    max_iterations : int
        Maximum fitting iterations.
    epsilon : float
        Convergence threshold.
    precompute_intercepts : bool
        Pre-compute common-neighbour index (default ``False``).
    set_attributes : bool
        If ``True``, write ``"dress"`` edge attributes and ``"dress_norm"``
        node attributes back onto *G*.

    Returns
    -------
    DRESSResult
    """
    from dress import dress_fit
    return _dress_graph_impl(
        dress_fit, G,
        variant=variant, weight=weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute_intercepts=precompute_intercepts,
        set_attributes=set_attributes,
    )


def delta_dress_graph(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
    keep_multisets: bool = False,
) -> DeltaDRESSResult:
    """Compute the Δ^k-DRESS histogram on a NetworkX graph.

    Exhaustively removes all k-vertex subsets from the graph, runs DRESS
    on each resulting subgraph, and accumulates every converged edge value
    into a single histogram binned by *epsilon*.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    k : int
        Deletion depth — number of vertices removed per subset.
        ``k=0`` runs DRESS on the original graph (Δ^0).
    variant : Variant
        ``UNDIRECTED``, ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    weight : str
        Edge attribute name for weights. If the attribute is missing the
        edge is treated as unweighted (weight = 1).
    max_iterations : int
        Maximum DRESS fitting iterations per subgraph.
    epsilon : float
        Convergence threshold and histogram bin width.
    precompute : bool
        If ``True``, precompute common-neighbour index per subgraph.
    keep_multisets : bool
        If ``True``, return per-subgraph DRESS values in a 2-D array
        of shape ``(C(N,k), E)``. ``NaN`` marks removed edges.

    Returns
    -------
    DeltaDRESSResult
        Dataclass with ``histogram`` (list of int, length ``hist_size``)
        and ``hist_size``. If *keep_multisets* is ``True``, also
        ``multisets`` and ``num_subgraphs``.
    """
    from dress import delta_dress_fit
    return _delta_dress_graph_impl(
        delta_dress_fit, G,
        k=k, variant=variant, weight=weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
    )


# ---------------------------------------------------------------------------
# Persistent NetworkX NxDRESS
# ---------------------------------------------------------------------------

class NxDRESS:
    """Persistent DRESS graph backed by a NetworkX graph.

    Translates NetworkX node labels to 0-based indices automatically,
    so :meth:`get` accepts the original node labels.

    The backend is determined by which module you import from::

        from dress.networkx import NxDRESS           # CPU
        from dress.cuda.networkx import NxDRESS      # CUDA
        from dress.mpi.networkx import NxDRESS       # MPI
        from dress.mpi.cuda.networkx import NxDRESS  # MPI+CUDA

    All share the same API: ``fit()``, ``delta_fit()``, ``get()``,
    ``result()``.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    variant : Variant
        ``UNDIRECTED``, ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    weight : str
        Edge attribute name for weights (default ``"weight"``).
    precompute_intercepts : bool
        Pre-compute common-neighbour index (default ``False``).
    """

    # Subclasses / factory functions override this to inject a different backend.
    _dress_cls = None

    def __init__(
        self,
        G,
        *,
        variant: Variant = UNDIRECTED,
        weight: str = "weight",
        precompute_intercepts: bool = False,
    ) -> None:
        n_vertices, sources, targets, weights, nodes = _extract_edges(G, weight)
        self._nodes = nodes
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        cls = self._dress_cls
        if cls is None:
            from dress import DRESS as cls
        self._dress = cls(
            n_vertices, sources, targets,
            weights=weights, variant=variant,
            precompute_intercepts=precompute_intercepts,
        )
        self._closed = False

    def fit(
        self,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
    ):
        """Run iterative fitting.

        Returns
        -------
        FitResult
            With ``iterations`` and ``delta`` attributes.
        """
        fr = self._dress.fit(max_iterations=max_iterations, epsilon=epsilon)
        self._last_fit = fr
        return fr

    def delta_fit(
        self,
        k: int = 0,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        keep_multisets: bool = False,
        **kwargs,
    ):
        """Compute the Δ^k-DRESS histogram.

        Accepts the same parameters as :meth:`dress.DRESS.delta_fit`
        plus any backend-specific keyword arguments (e.g. ``comm``).

        Returns
        -------
        DeltaDRESSResult
        """
        return self._dress.delta_fit(
            k=k, max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, **kwargs,
        )

    def get(
        self,
        u,
        v,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        edge_weight: float = 1.0,
    ) -> float:
        """Query the DRESS similarity for any node pair.

        Accepts original NetworkX node labels (strings, ints, etc.).
        """
        ui = self._node_to_idx[u]
        vi = self._node_to_idx[v]
        return self._dress.get(ui, vi, max_iterations=max_iterations,
                               epsilon=epsilon, edge_weight=edge_weight)

    def result(self) -> DRESSResult:
        """Extract current results as a :class:`DRESSResult`."""
        g = self._dress
        E = g.n_edges
        fr = getattr(self, '_last_fit', None)
        return DRESSResult(
            sources=[g.edge_source(e) for e in range(E)],
            targets=[g.edge_target(e) for e in range(E)],
            edge_dress=[g.edge_dress(e) for e in range(E)],
            edge_weight=[g.edge_weight(e) for e in range(E)],
            node_dress=[g.node_dress(u) for u in range(g.n_vertices)],
            iterations=fr.iterations if fr else 0,
            delta=fr.delta if fr else 0.0,
        )

    def close(self) -> None:
        """Release the underlying graph (idempotent)."""
        self._closed = True

    @property
    def nodes(self):
        """Node label list (same order as 0-based indices)."""
        return self._nodes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"NxDRESS(n_vertices={self._dress.n_vertices}, "
            f"n_edges={self._dress.n_edges}, "
            f"variant={self._dress.variant.name})"
        )

