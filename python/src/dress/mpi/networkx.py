"""MPI-distributed NetworkX helpers for DRESS.

Drop-in replacement for ``dress.networkx`` — just change the import::

    from dress.mpi.networkx import dress_graph, delta_fit, NxDRESS

    delta = delta_fit(G, k=1)

    with NxDRESS(G) as dg:   # delta_fit() distributed over MPI
        dg.fit()              # CPU (single graph)
        dg.delta_fit(k=3)    # MPI-distributed
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
    """Compute the Δ^k-DRESS histogram on a NetworkX graph (MPI, CPU).

    Same interface as :func:`dress.networkx.delta_fit` but
    distributes subgraph processing across MPI ranks.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    k : int
        Deletion depth — number of vertices removed per subset.
    variant : Variant
        ``UNDIRECTED``, ``DIRECTED``, ``FORWARD``, or ``BACKWARD``.
    weight : str
        Edge attribute name for weights.
    max_iterations : int
        Maximum DRESS fitting iterations per subgraph.
    epsilon : float
        Convergence threshold and histogram bin width.
    precompute : bool
        If ``True``, precompute common-neighbour index per subgraph.
    keep_multisets : bool
        If ``True``, return per-subgraph DRESS values.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).

    Returns
    -------
    DeltaDRESSResult
    """
    from dress.mpi import delta_fit
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
    """Compute DRESS similarity on a NetworkX graph (CPU).

    ``fit()`` runs on CPU — MPI only accelerates ``delta_fit``.
    Same interface as :func:`dress.networkx.dress_graph`.
    """
    from dress import fit
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
    """Compute the nabla^k-DRESS histogram on a NetworkX graph (MPI)."""
    from dress.mpi import nabla_fit
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
    """MPI-distributed ``NxDRESS`` — ``delta_fit`` distributed over MPI."""

    @property
    def _dress_cls(self):
        from dress.mpi import DRESS
        return DRESS
