"""MPI+CUDA NetworkX helpers for DRESS.

Drop-in replacement for ``dress.networkx.delta_dress_graph`` — just
change the import::

    from dress.mpi.cuda.networkx import delta_dress_graph

    delta = delta_dress_graph(G, k=1)

Note: only ``delta_dress_graph`` is available (MPI backends do not
expose a single-graph ``dress_fit``).
"""

from __future__ import annotations

from dress.core import DeltaDRESSResult, Variant, UNDIRECTED
from dress.networkx import _delta_dress_graph_impl

__all__ = ["delta_dress_graph"]


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
    comm=None,
) -> DeltaDRESSResult:
    """Compute the Δ^k-DRESS histogram on a NetworkX graph (MPI+CUDA).

    Same interface as :func:`dress.networkx.delta_dress_graph` but
    distributes subgraph processing across MPI ranks with
    GPU-accelerated DRESS on each rank.

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
    from dress.mpi.cuda import delta_dress_fit
    return _delta_dress_graph_impl(
        delta_dress_fit, G,
        k=k, variant=variant, weight=weight,
        max_iterations=max_iterations, epsilon=epsilon,
        precompute=precompute,
        keep_multisets=keep_multisets,
        comm=comm,
    )
