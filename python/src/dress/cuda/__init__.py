"""
dress.cuda — GPU-accelerated DRESS fitting (drop-in replacement for dress).

The simplest usage mirrors the CPU API exactly — just change the import:

    from dress.cuda import dress_fit, delta_dress_fit   # GPU
    # from dress import dress_fit, delta_dress_fit      # CPU (same API)

    result = dress_fit(N, sources, targets)
    delta  = delta_dress_fit(N, sources, targets, k=2)

NetworkX helpers are also available:

    from dress.cuda.networkx import dress_graph, delta_dress_graph

Lower-level helpers are also available for advanced use:

    from dress.cuda import dress_cuda, dress_fit_cuda, delta_dress_fit_cuda

Importing this module never fails — CUDA availability is checked lazily
on first call.  Use ``dress.cuda.is_available()`` to probe.
"""

import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer

# ---------------------------------------------------------------------------
#  Type definitions matching dress.h
# ---------------------------------------------------------------------------

class _DressGraph(ctypes.Structure):
    """Mirrors dress_graph_t — only needed so ctypes knows it's a struct."""
    _fields_ = [
        ("variant",                ctypes.c_int),
        ("N",                      ctypes.c_int),
        ("E",                      ctypes.c_int),
        ("U",                      ctypes.POINTER(ctypes.c_int)),
        ("V",                      ctypes.POINTER(ctypes.c_int)),
        ("adj_offset",             ctypes.POINTER(ctypes.c_int)),
        ("adj_target",             ctypes.POINTER(ctypes.c_int)),
        ("adj_edge_idx",           ctypes.POINTER(ctypes.c_int)),
        ("W",                      ctypes.POINTER(ctypes.c_double)),
        ("edge_weight",            ctypes.POINTER(ctypes.c_double)),
        ("edge_dress",             ctypes.POINTER(ctypes.c_double)),
        ("edge_dress_next",        ctypes.POINTER(ctypes.c_double)),
        ("node_dress",             ctypes.POINTER(ctypes.c_double)),
        ("precompute_intercepts",  ctypes.c_int),
        ("intercept_offset",       ctypes.POINTER(ctypes.c_int)),
        ("intercept_edge_ux",      ctypes.POINTER(ctypes.c_int)),
        ("intercept_edge_vx",      ctypes.POINTER(ctypes.c_int)),
    ]

_p_dress_graph_t = ctypes.POINTER(_DressGraph)

# ---------------------------------------------------------------------------
#  Lazy-load shared library (so importing dress.cuda never crashes)
# ---------------------------------------------------------------------------

_lib = None


def _get_lib():
    """Load libdress_cuda.so on first use. Raises RuntimeError if unavailable."""
    global _lib
    if _lib is not None:
        return _lib

    _here = os.path.dirname(os.path.abspath(__file__))
    _cuda_lib_path = os.path.join(
        _here, '..', '..', '..', '..', 'libdress', 'src', 'cuda', 'libdress_cuda.so')
    if not os.path.isfile(_cuda_lib_path):
        _cuda_lib_path = 'libdress_cuda.so'

    try:
        lib = ctypes.CDLL(_cuda_lib_path)
    except OSError as e:
        raise RuntimeError(
            "CUDA shared library (libdress_cuda.so) not found. "
            "Build it with `make` in libdress/src/cuda/, or ensure "
            "it is on LD_LIBRARY_PATH."
        ) from e

    # Bind C function signatures
    lib.init_dress_graph.restype  = _p_dress_graph_t
    lib.init_dress_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.dress_fit.restype  = None
    lib.dress_fit.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ]

    lib.dress_fit_cuda.restype  = None
    lib.dress_fit_cuda.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ]

    lib.delta_dress_fit_cuda.restype  = ctypes.POINTER(ctypes.c_int64)
    lib.delta_dress_fit_cuda.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.free_dress_graph.restype  = None
    lib.free_dress_graph.argtypes = [_p_dress_graph_t]

    _lib = lib
    return _lib


def is_available():
    """Return True if the CUDA library can be loaded."""
    try:
        _get_lib()
        return True
    except RuntimeError:
        return False

# ---------------------------------------------------------------------------
#  Python API
# ---------------------------------------------------------------------------

# Variant constants (matching dress_variant_t enum)
UNDIRECTED = 0
DIRECTED   = 1
FORWARD    = 2
BACKWARD   = 3


def _make_c_array(arr, ctype):
    """Convert a Python list/numpy array to a heap-allocated ctypes array.
    
    The C library takes ownership, so we use malloc-compatible allocation
    via (ctype * n).from_buffer_copy(), which ctypes manages.
    NOTE: init_dress_graph takes ownership of U, V, W — they are freed by
    free_dress_graph. We must allocate with ctypes so the C free() works.
    """
    n = len(arr)
    c_arr = (ctype * n)(*arr)
    # Convert to a pointer that the C library can free() — use ctypes malloc
    ptr = ctypes.cast(ctypes.pointer(c_arr), ctypes.POINTER(ctype))
    return ptr, c_arr  # keep c_arr alive


def dress_cuda(N, E, U, V, W=None, variant=UNDIRECTED,
               precompute_intercepts=0, max_iterations=100, epsilon=1e-10):
    """
    Compute DRESS edge and node values using GPU acceleration.

    Parameters
    ----------
    N : int — number of nodes
    E : int — number of edges
    U : array-like[int] — source node for each edge (length E)
    V : array-like[int] — target node for each edge (length E)
    W : array-like[float] or None — edge weights (None = unweighted)
    variant : int — 0=UNDIRECTED, 1=DIRECTED, 2=FORWARD, 3=BACKWARD
    precompute_intercepts : int — 1 to precompute intercepts (faster for dense)
    max_iterations : int
    epsilon : float

    Returns
    -------
    edge_dress : np.ndarray[float64] of shape (E,)
    node_dress : np.ndarray[float64] of shape (N,)
    iterations : int
    delta : float
    """
    # Allocate C arrays that init_dress_graph will take ownership of.
    # We use ctypes's malloc so free() in C works correctly.
    c_U = (ctypes.c_int * E)(*[int(x) for x in U])
    c_V = (ctypes.c_int * E)(*[int(x) for x in V])
    p_U = ctypes.cast(c_U, ctypes.POINTER(ctypes.c_int))
    p_V = ctypes.cast(c_V, ctypes.POINTER(ctypes.c_int))

    if W is not None:
        c_W = (ctypes.c_double * E)(*[float(x) for x in W])
        p_W = ctypes.cast(c_W, ctypes.POINTER(ctypes.c_double))
    else:
        p_W = ctypes.POINTER(ctypes.c_double)()  # NULL

    lib = _get_lib()

    # Build graph on CPU
    g = lib.init_dress_graph(N, E, p_U, p_V, p_W,
                             ctypes.c_int(variant),
                             ctypes.c_int(precompute_intercepts))

    # Run GPU fit
    iterations = ctypes.c_int(0)
    delta      = ctypes.c_double(0.0)
    lib.dress_fit_cuda(g, max_iterations, ctypes.c_double(epsilon),
                       ctypes.byref(iterations), ctypes.byref(delta))

    # Extract results
    edge_dress  = np.ctypeslib.as_array(g.contents.edge_dress,  shape=(E,)).copy()
    node_dress  = np.ctypeslib.as_array(g.contents.node_dress,  shape=(N,)).copy()
    edge_weight = np.ctypeslib.as_array(g.contents.edge_weight, shape=(E,)).copy()

    iters = iterations.value
    delta_val = delta.value

    # Free C memory
    lib.free_dress_graph(g)

    return edge_dress, node_dress, edge_weight, iters, delta_val


def dress_fit_cuda(g_ptr, max_iterations=100, epsilon=1e-10):
    """
    Run GPU-accelerated fit on an existing dress_graph_t pointer.

    Parameters
    ----------
    g_ptr : ctypes pointer to dress_graph_t
    max_iterations : int
    epsilon : float

    Returns
    -------
    iterations : int
    delta : float
    """
    lib = _get_lib()
    iterations = ctypes.c_int(0)
    delta      = ctypes.c_double(0.0)
    lib.dress_fit_cuda(g_ptr, max_iterations, ctypes.c_double(epsilon),
                       ctypes.byref(iterations), ctypes.byref(delta))
    return iterations.value, delta.value


def delta_dress_fit_cuda(g_ptr, k, max_iterations=100, epsilon=1e-10,
                          keep_multisets=False):
    """
    GPU-accelerated Δ^k-DRESS histogram computation.

    Parameters
    ----------
    g_ptr : ctypes pointer to dress_graph_t
    k : int — subgraph size
    max_iterations : int
    epsilon : float
    keep_multisets : bool

    Returns
    -------
    histogram : np.ndarray[int64]
    num_subgraphs : int
    """
    lib = _get_lib()
    hist_size = ctypes.c_int(0)
    num_subgraphs = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.delta_dress_fit_cuda(
        g_ptr, k, max_iterations, ctypes.c_double(epsilon),
        ctypes.byref(hist_size),
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs))

    size = hist_size.value
    if not hist_ptr or size == 0:
        return np.array([], dtype=np.int64), 0

    histogram = np.ctypeslib.as_array(hist_ptr, shape=(size,)).copy()
    lib.free(hist_ptr)

    return histogram, num_subgraphs.value


# ---------------------------------------------------------------------------
#  Unified API — same signatures as dress.dress_fit / dress.delta_dress_fit
# ---------------------------------------------------------------------------

def dress_fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute_intercepts=False,
):
    """GPU-accelerated DRESS fitting — drop-in replacement for ``dress.dress_fit``.

    Same parameters and return type as :func:`dress.dress_fit`, but the
    iterative fitting loop runs on the GPU.
    """
    from dress.core import DRESSResult

    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    edge_dress, node_dress, edge_weight, iters, delta = dress_cuda(
        N=n_vertices,
        E=E,
        U=sources,
        V=targets,
        W=list(weights) if weights is not None else None,
        variant=int(variant),
        precompute_intercepts=int(precompute_intercepts),
        max_iterations=max_iterations,
        epsilon=epsilon,
    )

    return DRESSResult(
        sources=sources,
        targets=targets,
        edge_dress=edge_dress.tolist(),
        edge_weight=edge_weight.tolist(),
        node_dress=node_dress.tolist(),
        iterations=iters,
        delta=delta,
    )


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
):
    """GPU-accelerated Δ^k-DRESS — drop-in replacement for ``dress.delta_dress_fit``.

    Same parameters and return type as :func:`dress.delta_dress_fit`, but
    each subgraph fitting runs on the GPU.
    """
    from dress.core import DeltaDRESSResult

    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    # Build the graph via ctypes
    c_U = (ctypes.c_int * E)(*[int(x) for x in sources])
    c_V = (ctypes.c_int * E)(*[int(x) for x in targets])
    p_U = ctypes.cast(c_U, ctypes.POINTER(ctypes.c_int))
    p_V = ctypes.cast(c_V, ctypes.POINTER(ctypes.c_int))

    if weights is not None:
        c_W = (ctypes.c_double * E)(*[float(x) for x in weights])
        p_W = ctypes.cast(c_W, ctypes.POINTER(ctypes.c_double))
    else:
        p_W = ctypes.POINTER(ctypes.c_double)()

    g = lib.init_dress_graph(n_vertices, E, p_U, p_V, p_W,
                             ctypes.c_int(int(variant)),
                             ctypes.c_int(int(precompute)))

    hist_size = ctypes.c_int(0)
    num_subgraphs = ctypes.c_int64(0)
    multisets_ptr = ctypes.POINTER(ctypes.c_double)()

    hist_ptr = lib.delta_dress_fit_cuda(
        g, k, max_iterations, ctypes.c_double(epsilon),
        ctypes.byref(hist_size),
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs))

    size = hist_size.value
    ns = num_subgraphs.value
    if not hist_ptr or size == 0:
        histogram = []
    else:
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
