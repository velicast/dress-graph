"""MPI+OMP NetworkX helpers for DRESS.

Drop-in replacement for ``dress.networkx`` — just change the import::

    from dress.mpi.omp.networkx import dress_graph, delta_fit, NxDRESS

    delta = delta_fit(G, k=1)

    with NxDRESS(G) as dg:
        dg.fit()              # OMP edge-parallel
        dg.delta_fit(k=3)    # MPI + OMP subgraph-parallel
"""

from __future__ import annotations

from dress.core import DRESSResult, DeltaDRESSResult, Variant, UNDIRECTED
from dress.networkx import (
    _dress_graph_impl,
    _delta_dress_graph_impl,
    NxDRESS as _BaseNxDRESS,
)

__all__ = ["dress_graph", "delta_fit", "nabla_fit", "NxDRESS"]


def delta_fit(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    vertex_weight: str = "vertex_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
    keep_multisets: bool = False,
    comm=None,
    n_samples: int = 0,
    seed: int = 0,
    compute_histogram: bool = True,
) -> DeltaDRESSResult:
    """Compute the Δ^k-DRESS histogram on a NetworkX graph (MPI+OMP).

    MPI distributes subgraphs across ranks.  Within each rank, OpenMP
    threads further parallelise the subgraph slice.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).
    All other parameters are identical to :func:`dress.networkx.delta_fit`.
    """
    from dress.mpi.omp import delta_fit
    return _delta_dress_graph_impl(
        delta_fit, G,
        k=k, variant=variant, weight=weight, vertex_weight=vertex_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        comm=comm,
        n_samples=n_samples, seed=seed,
        compute_histogram=compute_histogram,
    )


def dress_graph(
    G,
    *,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    vertex_weight: str = "vertex_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute_intercepts: bool = False,
    set_attributes: bool = False,
) -> DRESSResult:
    """Compute DRESS similarity on a NetworkX graph (OMP edge-parallel).

    ``fit()`` uses OpenMP — MPI+OMP only accelerates ``delta_fit``.
    Same interface as :func:`dress.networkx.dress_graph`.
    """
    from dress.omp import fit
    return _dress_graph_impl(
        fit, G,
        variant=variant, weight=weight, vertex_weight=vertex_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute_intercepts=precompute_intercepts,
        set_attributes=set_attributes,
    )

def nabla_fit(
    G,
    *,
    k: int = 0,
    variant: Variant = UNDIRECTED,
    weight: str = "weight",
    vertex_weight: str = "vertex_weight",
    max_iterations: int = 100,
    epsilon: float = 1e-6,
    precompute: bool = False,
    keep_multisets: bool = False,
    comm=None,
    n_samples: int = 0,
    seed: int = 0,
    compute_histogram: bool = True,
):
    """Compute the nabla^k-DRESS histogram on a NetworkX graph (MPI+OMP)."""
    from dress.mpi.omp import nabla_fit
    return _delta_dress_graph_impl(
        nabla_fit, G,
        k=k, variant=variant, weight=weight, vertex_weight=vertex_weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        comm=comm,
        n_samples=n_samples, seed=seed,
        compute_histogram=compute_histogram,
    )

class NxDRESS(_BaseNxDRESS):
    """MPI+OMP ``NxDRESS`` — ``fit`` uses OMP, ``delta_fit`` uses MPI+OMP."""

    @property
    def _dress_cls(self):
        from dress.mpi.omp import DRESS
        return DRESS
