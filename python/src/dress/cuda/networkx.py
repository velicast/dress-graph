"""
GPU-accelerated NetworkX helpers for DRESS.

Drop-in replacement for ``dress.networkx`` — just change the import::

    from dress.cuda.networkx import fit, delta_fit, NxDRESS

    result = fit(G)
    delta  = delta_fit(G, k=1)

    with NxDRESS(G) as dg:   # fit() / delta_fit() run on GPU
        dg.fit()
        print(dg.get("Alice", "Bob"))
"""

from __future__ import annotations

from dress.core import DRESSResult, DeltaDRESSResult, Variant, UNDIRECTED
from dress.networkx import (
    _dress_graph_impl,
    _delta_dress_graph_impl,
    NxDRESS as _BaseNxDRESS,
)

__all__ = ["fit", "delta_fit", "nabla_fit", "NxDRESS"]


def fit(
    G,
    *,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    node_weight: str = "node_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute_intercepts: bool = False,
    set_attributes: bool = False,
) -> DRESSResult:
    """Compute DRESS similarity on a NetworkX graph (CUDA).

    Same interface as :func:`dress.networkx.fit` but uses the
    GPU-accelerated backend.
    """
    from dress.cuda import fit
    return _dress_graph_impl(
        fit, G,
        variant=variant, weight=weight, node_weight=node_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute_intercepts=precompute_intercepts,
        set_attributes=set_attributes,
    )


def delta_fit(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    node_weight: str = "node_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
    keep_multisets: bool = False,
    n_samples: int = 0,
    seed: int = 0,
    compute_histogram: bool = True,
) -> DeltaDRESSResult:
    """Compute the Δ^k-DRESS histogram on a NetworkX graph (CUDA).

    Same interface as :func:`dress.networkx.delta_fit` but uses
    the GPU-accelerated backend.
    """
    from dress.cuda import delta_fit
    return _delta_dress_graph_impl(
        delta_fit, G,
        k=k, variant=variant, weight=weight, node_weight=node_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        n_samples=n_samples, seed=seed,
        compute_histogram=compute_histogram,
    )

def nabla_fit(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    node_weight: str = "node_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
    keep_multisets: bool = False,
    n_samples: int = 0,
    seed: int = 0,
    compute_histogram: bool = True,
):
    """Compute the nabla^k-DRESS histogram on a NetworkX graph (CUDA)."""
    from dress.cuda import nabla_fit
    return _delta_dress_graph_impl(
        nabla_fit, G,
        k=k, variant=variant, weight=weight, node_weight=node_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        n_samples=n_samples, seed=seed,
        compute_histogram=compute_histogram,
    )

class NxDRESS(_BaseNxDRESS):
    """GPU-accelerated ``NxDRESS`` — ``fit`` / ``delta_fit`` run on CUDA."""

    @property
    def _dress_cls(self):
        from dress.cuda import DRESS
        return DRESS
