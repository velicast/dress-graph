"""``dress.omp`` — OpenMP-parallel DRESS fitting.

Switch to the OpenMP backend by changing the import::

    from dress.omp import DRESS

    g = DRESS(N, sources, targets)
    g.fit()           # OpenMP edge-parallel
    g.delta_fit(k=3)  # OpenMP subgraph-parallel

The OMP functions live in the main ``libdress.so`` (built with OpenMP).

Module-level functions are also available::

    from dress.omp import fit, delta_fit
"""

import ctypes
import os
import numpy as np

from dress import UNDIRECTED
from dress.core import DRESSResult, DeltaDRESSResult, NablaDRESSResult, FitResult
from dress._ctypes_helpers import (
    _DressGraph, _DressHistPair, _p_dress_graph_t, _malloc_array,
)

# ---------------------------------------------------------------------------
#  Lazy-load libdress.so (with OpenMP support)
# ---------------------------------------------------------------------------

_lib = None


def _get_lib():
    """Load libdress.so (with OpenMP) on first use."""
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
            "libdress.so not found. Build with:\n"
            "  ./build.sh c"
        )

    try:
        lib.dress_fit_omp
    except AttributeError:
        raise RuntimeError(
            "libdress.so found but OMP API not available. "
            "Rebuild with an OpenMP-capable compiler."
        )

    lib.dress_init_graph.restype = _p_dress_graph_t
    lib.dress_init_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.dress_fit_omp.restype = None
    lib.dress_fit_omp.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ]

    lib.dress_delta_fit_omp.restype = ctypes.POINTER(_DressHistPair)
    lib.dress_delta_fit_omp.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.dress_delta_fit_omp_strided.restype = ctypes.POINTER(_DressHistPair)
    lib.dress_delta_fit_omp_strided.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.dress_nabla_fit_omp.restype = ctypes.POINTER(_DressHistPair)
    lib.dress_nabla_fit_omp.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
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

def fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    node_weights=None,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute_intercepts=False,
):
    """OpenMP-parallel DRESS fitting — drop-in replacement for ``dress.fit``.

    Parallelises the per-edge and per-node loops within each iteration.
    """
    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    p_U = _malloc_array([int(x) for x in sources], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in targets], ctypes.c_int)
    p_W = (_malloc_array([float(x) for x in weights], ctypes.c_double)
           if weights is not None else ctypes.POINTER(ctypes.c_double)())
    p_NW = (_malloc_array([float(x) for x in node_weights], ctypes.c_double)
            if node_weights is not None else ctypes.POINTER(ctypes.c_double)())

    g = lib.dress_init_graph(n_vertices, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute_intercepts)))

    iterations = ctypes.c_int(0)
    delta = ctypes.c_double(0.0)
    lib.dress_fit_omp(g, max_iterations, ctypes.c_double(epsilon),
                      ctypes.byref(iterations), ctypes.byref(delta))

    edge_dress = np.ctypeslib.as_array(g.contents.edge_dress, shape=(E,)).copy()
    node_dress = np.ctypeslib.as_array(g.contents.node_dress, shape=(n_vertices,)).copy()
    edge_weight = np.ctypeslib.as_array(g.contents.edge_weight, shape=(E,)).copy()

    iters = iterations.value
    delta_val = delta.value
    lib.dress_free_graph(g)

    return DRESSResult(
        sources=sources,
        targets=targets,
        edge_dress=edge_dress.tolist(),
        edge_weight=edge_weight.tolist(),
        node_dress=node_dress.tolist(),
        iterations=iters,
        delta=delta_val,
    )


def delta_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    node_weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    offset=0,
    stride=1,
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """OpenMP-parallel Δ^k-DRESS — drop-in replacement for ``dress.delta_fit``.

    Parallelises the outer subgraph loop: each OMP thread processes a
    strided slice of C(N,k) subgraphs, fitting each one sequentially.
    """
    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    p_U = _malloc_array([int(x) for x in sources], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in targets], ctypes.c_int)
    p_W = (_malloc_array([float(x) for x in weights], ctypes.c_double)
           if weights is not None else ctypes.POINTER(ctypes.c_double)())
    p_NW = (_malloc_array([float(x) for x in node_weights], ctypes.c_double)
            if node_weights is not None else ctypes.POINTER(ctypes.c_double)())

    g = lib.dress_init_graph(n_vertices, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_subgraphs = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.dress_delta_fit_omp_strided(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs),
        offset, stride)

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
    node_weights=None,
    k=0,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute=False,
    keep_multisets=False,
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """OpenMP-parallel ∇^k-DRESS — drop-in replacement for ``dress.nabla_fit``.

    Parallelises the outer tuple loop: each OMP thread processes a
    slice of P(N,k) tuples, fitting each one sequentially.
    """
    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    p_U = _malloc_array([int(x) for x in sources], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in targets], ctypes.c_int)
    p_W = (_malloc_array([float(x) for x in weights], ctypes.c_double)
           if weights is not None else ctypes.POINTER(ctypes.c_double)())
    p_NW = (_malloc_array([float(x) for x in node_weights], ctypes.c_double)
            if node_weights is not None else ctypes.POINTER(ctypes.c_double)())

    g = lib.dress_init_graph(n_vertices, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_tuples = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.dress_nabla_fit_omp(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_tuples))

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
#  DRESS class — same API as dress.DRESS, fit/delta_fit run with OpenMP
# ---------------------------------------------------------------------------

from dress import DRESS as _BaseDRESS  # noqa: E402


class DRESS(_BaseDRESS):
    """OpenMP-parallel DRESS — same API, ``fit``/``delta_fit`` use OpenMP.

    ``fit()`` parallelises edges within each iteration (large single graph).
    ``delta_fit()`` parallelises the outer subgraph loop (many small subgraphs).

    Usage::

        from dress.omp import DRESS

        g = DRESS(4, [0, 1, 2, 0], [1, 2, 3, 3])
        fr = g.fit()          # OMP edge-parallel
        dr = g.delta_fit(k=2) # OMP subgraph-parallel
        val = g.get(0, 2)     # CPU
    """

    _force_python_impl = True

    def fit(self, max_iterations=100, epsilon=1e-6):
        result = fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, node_weights=self._nwgt,
            variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
        )
        self._sync_hardware_fit(result)
        return FitResult(iterations=result.iterations, delta=result.delta)

    def delta_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  n_samples=0, seed=0,
                  keep_multisets=False, compute_histogram=True,
                  offset=0, stride=1):
        return delta_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, node_weights=self._nwgt,
            k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets, offset=offset, stride=stride,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def nabla_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  n_samples=0, seed=0,
                  keep_multisets=False, compute_histogram=True):
        return nabla_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, node_weights=self._nwgt,
            k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def __repr__(self):
        return (
            f"DRESS(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"variant={self._var.name}, backend=omp)"
        )
