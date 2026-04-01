"""``dress.mpi`` — MPI-distributed DRESS (CPU backend).

Switch to the MPI backend by changing the import::

    from dress.mpi import DRESS

    g = DRESS(N, sources, targets)
    g.fit()           # CPU (single graph)
    g.delta_fit(k=3)  # MPI-distributed across ranks
    g.get(0, 1)       # CPU

Requires ``libdress.so`` built with ``-DDRESS_MPI=ON`` and ``mpi4py``.
All MPI logic (stride partitioning + Allreduce) runs in C.
The wrapper passes the Fortran communicator handle via ``comm.py2f()``.

Module-level functions are also available::

    from dress.mpi import delta_fit
"""

import ctypes
import os
import numpy as np

from dress import UNDIRECTED
from dress.core import DeltaDRESSResult, NablaDRESSResult
from dress._ctypes_helpers import _DressGraph, _DressHistPair, _p_dress_graph_t, _malloc_array

# ---------------------------------------------------------------------------
#  Lazy-load libdress.so (with MPI support)
# ---------------------------------------------------------------------------

_lib = None


def _get_lib():
    """Load libdress.so (with MPI) on first use."""
    global _lib
    if _lib is not None:
        return _lib

    _here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_here, '..', '..', '..', '..', 'build', 'libdress', 'libdress.so'),
        os.path.join(_here, '..', '..', '..', '..', 'build', 'libdress.so'),
        'libdress.so',
    ]
    lib = None
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError(
            "libdress.so (with MPI) not found. Build with:\n"
            "  cmake -DDRESS_MPI=ON -S libdress -B build && cmake --build build"
        )

    try:
        lib.dress_delta_fit_mpi_fcomm
    except AttributeError:
        raise RuntimeError(
            "libdress.so found but MPI _fcomm API not available. "
            "Rebuild with -DDRESS_MPI=ON."
        )

    lib.dress_init_graph.restype = _p_dress_graph_t
    lib.dress_init_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.dress_delta_fit_mpi_fcomm.restype = ctypes.POINTER(_DressHistPair)
    lib.dress_delta_fit_mpi_fcomm.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ]

    lib.dress_nabla_fit_mpi_fcomm.restype = ctypes.POINTER(_DressHistPair)
    lib.dress_nabla_fit_mpi_fcomm.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ]

    lib.dress_free_graph.restype = None
    lib.dress_free_graph.argtypes = [_p_dress_graph_t]

    _lib = lib
    return _lib


def _decode_histogram(hist_ptr, size, lib):
    if not hist_ptr or size == 0:
        return []
    pairs = np.ctypeslib.as_array(hist_ptr, shape=(size,)).copy()
    lib.free(hist_ptr)
    return [(float(pair["value"]), int(pair["count"])) for pair in pairs]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def delta_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    vertex_weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    comm=None,
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """MPI-distributed Δ^k-DRESS (CPU backend).

    All MPI logic (stride partitioning + Allreduce) runs in C.
    Uses ``comm.py2f()`` to pass the Fortran MPI handle.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).
    All other parameters are identical to :func:`dress.dress_delta_fit`.
    """
    from mpi4py import MPI as MPI4PY
    if comm is None:
        comm = MPI4PY.COMM_WORLD
    comm_f = comm.py2f()

    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    p_U = _malloc_array([int(x) for x in sources], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in targets], ctypes.c_int)

    if weights is not None:
        p_W = _malloc_array([float(x) for x in weights], ctypes.c_double)
    else:
        p_W = ctypes.POINTER(ctypes.c_double)()

    if vertex_weights is not None:
        p_NW = _malloc_array([float(x) for x in vertex_weights], ctypes.c_double)
    else:
        p_NW = ctypes.POINTER(ctypes.c_double)()

    g = lib.dress_init_graph(n_vertices, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_subgraphs = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.dress_delta_fit_mpi_fcomm(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs),
        ctypes.c_int(comm_f))

    size = hist_size.value
    ns = num_subgraphs.value
    histogram = _decode_histogram(hist_ptr, size, lib)

    ms = None
    if keep_multisets and multisets_ptr and ns > 0:
        total = ns * E
        flat = np.ctypeslib.as_array(multisets_ptr, shape=(total,)).copy()
        ms = flat.reshape(ns, E)
        lib.free(multisets_ptr)

    lib.dress_free_graph(g)

    return DeltaDRESSResult(
        histogram=histogram,
        multisets=ms,
        num_subgraphs=ns,
    )


def nabla_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    vertex_weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    comm=None,
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """MPI-distributed ∇^k-DRESS (CPU backend).

    All MPI logic (stride partitioning + Allreduce) runs in C.
    Uses ``comm.py2f()`` to pass the Fortran MPI handle.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).
    All other parameters are identical to :func:`dress.dress_nabla_fit`.
    """
    from mpi4py import MPI as MPI4PY
    if comm is None:
        comm = MPI4PY.COMM_WORLD
    comm_f = comm.py2f()

    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    p_U = _malloc_array([int(x) for x in sources], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in targets], ctypes.c_int)

    if weights is not None:
        p_W = _malloc_array([float(x) for x in weights], ctypes.c_double)
    else:
        p_W = ctypes.POINTER(ctypes.c_double)()

    if vertex_weights is not None:
        p_NW = _malloc_array([float(x) for x in vertex_weights], ctypes.c_double)
    else:
        p_NW = ctypes.POINTER(ctypes.c_double)()

    g = lib.dress_init_graph(n_vertices, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_tuples = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.dress_nabla_fit_mpi_fcomm(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_tuples),
        ctypes.c_int(comm_f))

    size = hist_size.value
    ns = num_tuples.value
    histogram = _decode_histogram(hist_ptr, size, lib)

    ms = None
    if keep_multisets and multisets_ptr and ns > 0:
        total = ns * E
        flat = np.ctypeslib.as_array(multisets_ptr, shape=(total,)).copy()
        ms = flat.reshape(ns, E)
        lib.free(multisets_ptr)

    lib.dress_free_graph(g)

    return NablaDRESSResult(
        histogram=histogram,
        multisets=ms,
        num_tuples=ns,
    )


# ---------------------------------------------------------------------------
#  DRESS class — same API as dress.DRESS, delta_fit runs over MPI
# ---------------------------------------------------------------------------

from dress import DRESS as _BaseDRESS  # noqa: E402


class DRESS(_BaseDRESS):
    """MPI-distributed DRESS — same API, ``delta_fit`` runs over MPI.

    ``fit()`` and ``get()`` run on CPU (single graph, no distribution
    benefit).  ``delta_fit()`` distributes the C(N,k) subgraph
    enumeration across MPI ranks.

    Usage::

        from dress.mpi import DRESS

        g = DRESS(4, [0, 1, 2, 0], [1, 2, 3, 3])
        g.fit()               # CPU
        dr = g.delta_fit(k=3) # MPI-distributed
    """

    _force_python_impl = True

    def delta_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  n_samples=0, seed=0,
                  keep_multisets=False, compute_histogram=True,
                  comm=None):
        return delta_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, vertex_weights=self._nwgt,
            k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  n_samples=0, seed=0,
                  keep_multisets=False, compute_histogram=True,
                  comm=None):
        return nabla_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, vertex_weights=self._nwgt,
            k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def __repr__(self):
        return (
            f"DRESS(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"variant={self._var.name}, backend=mpi)"
        )
