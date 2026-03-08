"""
GPU-accelerated NetworkX helpers for DRESS.

Drop-in replacement for ``dress.networkx`` — just change the import::

    from dress.cuda.networkx import dress_graph, delta_dress_graph, NxDRESS

    result = dress_graph(G)
    delta  = delta_dress_graph(G, k=1)
"""

from __future__ import annotations

from dress.core import DRESSResult, DeltaDRESSResult, Variant, UNDIRECTED
from dress.networkx import (
    _dress_graph_impl,
    _delta_dress_graph_impl,
    NxDRESS,
)

__all__ = ["dress_graph", "delta_dress_graph", "NxDRESS"]


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
    """Compute DRESS similarity on a NetworkX graph (CUDA).

    Same interface as :func:`dress.networkx.dress_graph` but uses the
    GPU-accelerated backend.
    """
    from dress.cuda import dress_fit
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
    """Compute the Δ^k-DRESS histogram on a NetworkX graph (CUDA).

    Same interface as :func:`dress.networkx.delta_dress_graph` but uses
    the GPU-accelerated backend.
    """
    from dress.cuda import delta_dress_fit
    return _delta_dress_graph_impl(
        delta_dress_fit, G,
        k=k, variant=variant, weight=weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
    )
