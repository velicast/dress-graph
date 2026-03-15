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

    from dress.mpi import delta_dress_fit
"""

import ctypes
import os
import numpy as np

from dress import UNDIRECTED
from dress.core import DeltaDRESSResult
from dress._ctypes_helpers import _DressGraph, _p_dress_graph_t, _malloc_array

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
        lib.delta_dress_fit_mpi_fcomm
    except AttributeError:
        raise RuntimeError(
            "libdress.so found but MPI _fcomm API not available. "
            "Rebuild with -DDRESS_MPI=ON."
        )

    lib.init_dress_graph.restype = _p_dress_graph_t
    lib.init_dress_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.delta_dress_fit_mpi_fcomm.restype = ctypes.POINTER(ctypes.c_int64)
    lib.delta_dress_fit_mpi_fcomm.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ]

    lib.free_dress_graph.restype = None
    lib.free_dress_graph.argtypes = [_p_dress_graph_t]

    _lib = lib
    return _lib


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def delta_dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    comm=None,
):
    """MPI-distributed Δ^k-DRESS (CPU backend).

    All MPI logic (stride partitioning + Allreduce) runs in C.
    Uses ``comm.py2f()`` to pass the Fortran MPI handle.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).
    All other parameters are identical to :func:`dress.delta_dress_fit`.
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

    g = lib.init_dress_graph(n_vertices, E, p_U, p_V, p_W,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_subgraphs = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.delta_dress_fit_mpi_fcomm(
        g, k, max_iterations, ctypes.c_double(epsilon),
        ctypes.byref(hist_size),
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs),
        ctypes.c_int(comm_f))

    size = hist_size.value
    ns = num_subgraphs.value
    histogram = []
    if hist_ptr and size > 0:
        histogram = np.ctypeslib.as_array(hist_ptr, shape=(size,)).copy().tolist()
        lib.free(hist_ptr)

    ms = None
    if keep_multisets and multisets_ptr and ns > 0:
        total = ns * E
        flat = np.ctypeslib.as_array(multisets_ptr, shape=(total,)).copy()
        ms = flat.reshape(ns, E)
        lib.free(multisets_ptr)

    lib.free_dress_graph(g)

    return DeltaDRESSResult(
        histogram=histogram,
        hist_size=size,
        multisets=ms,
        num_subgraphs=ns,
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
                  keep_multisets=False, comm=None):
        return delta_dress_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, comm=comm,
        )

    def __repr__(self):
        return (
            f"DRESS(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"variant={self._var.name}, backend=mpi)"
        )
