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

from dress.core import DRESS, DRESSResult, Variant, UNDIRECTED

__all__ = ["dress_graph"]


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
