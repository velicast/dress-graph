"""
NetworkX helpers for the pure-Python DRESS implementation.

Usage::

    import networkx as nx
    from dress.networkx import dress_graph

    G = nx.karate_club_graph()
    result = dress_graph(G, max_iterations=100, epsilon=1e-6)

    # result is a DRESSResult with .edge_dress, .node_dress, etc.
    # or attach values directly to the graph:
    nx_result = dress_graph(G, set_attributes=True)
"""

from __future__ import annotations

from typing import Optional

from dress.core import DRESS, DRESSResult, DeltaDRESSResult, Variant, UNDIRECTED

__all__ = ["dress_graph", "delta_dress_graph"]


def dress_graph(
    G,
    *,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
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
    set_attributes : bool
        If ``True``, write ``"dress"`` edge attributes and ``"dress_norm"``
        node attributes back onto *G*.

    Returns
    -------
    DRESSResult
    """
    # Map node labels to 0-based integers
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_vertices = len(nodes)

    sources = []
    targets = []
    weights = []
    has_weights = False

    for u, v, data in G.edges(data=True):
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
        w = data.get(weight, 1.0)
        weights.append(float(w))
        if w != 1.0:
            has_weights = True

    g = DRESS(
        n_vertices,
        sources,
        targets,
        weights=weights if has_weights else None,
        variant=variant,
    )
    result = g.fit(max_iterations=max_iterations, epsilon=epsilon)

    if set_attributes:
        # Write edge attributes
        for i, (u, v) in enumerate(zip(result.sources, result.targets)):
            u_label = nodes[u]
            v_label = nodes[v]
            if G.has_edge(u_label, v_label):
                G[u_label][v_label]["dress"] = result.edge_dress[i]

        # Write node attributes
        for i, n in enumerate(nodes):
            G.nodes[n]["dress_norm"] = result.node_dress[i]

    return result


def delta_dress_graph(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
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
        Edge attribute name for weights (currently unused — delta_dress_fit
        operates on unweighted topology only).
    max_iterations : int
        Maximum DRESS fitting iterations per subgraph.
    epsilon : float
        Convergence threshold and histogram bin width.
    precompute : bool
        If ``True``, precompute common-neighbour index per subgraph.

    Returns
    -------
    DeltaDRESSResult
        Dataclass with ``histogram`` (list of int, length ``hist_size``)
        and ``hist_size``.
    """
    from dress import delta_dress_fit as _delta_dress_fit

    # Map node labels to 0-based integers
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_vertices = len(nodes)

    sources = []
    targets = []
    for u, v in G.edges():
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])

    return _delta_dress_fit(
        n_vertices,
        sources,
        targets,
        k=k,
        variant=variant,
        max_iterations=max_iterations,
        epsilon=epsilon,
        precompute=precompute,
    )
