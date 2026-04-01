"""
dress.cuda — GPU-accelerated DRESS fitting.

Switch to the GPU backend by changing the import::

    from dress.cuda import DRESS        # GPU
    # from dress import DRESS            # CPU (same API)

    g = DRESS(N, sources, targets)
    g.fit()           # runs on GPU
    g.delta_fit(k=2)  # runs on GPU
    g.get(0, 1)       # runs on CPU

Module-level functions are also available::

    from dress.cuda import fit, delta_fit

NetworkX helpers::

    from dress.cuda.networkx import dress_graph, delta_dress_graph

Importing this module never fails — CUDA availability is checked lazily
on first call.  Use ``dress.cuda.is_available()`` to probe.
"""

import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer

from dress._ctypes_helpers import _DressGraph, _DressHistPair, _p_dress_graph_t, _malloc_array

# ---------------------------------------------------------------------------
#  Lazy-load shared library (so importing dress.cuda never crashes)
# ---------------------------------------------------------------------------

_lib = None

# Path constants for auto-build
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_HERE)                       # dress/
_VENDORED = os.path.join(_PKG_DIR, '_vendored')           # dress/_vendored/
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..', '..'))
# Prefer vendored sources (pip install) over repo-relative (editable install)
if os.path.isdir(_VENDORED):
    _LIB_DIR = _VENDORED
else:
    _LIB_DIR = os.path.join(_ROOT, 'libdress')
_LOCAL_SO = os.path.join(_LIB_DIR, 'src', 'cuda', 'libdress_cuda.so')


def _sources_newer_than(so_path):
    """Return True if any vendored source is newer than the .so."""
    if not os.path.isfile(so_path):
        return True
    # If dress_cuda.cu is absent we are in a pre-built wheel; skip stale check
    cuda_cu = os.path.join(_LIB_DIR, 'src', 'cuda', 'dress_cuda.cu')
    if not os.path.isfile(cuda_cu):
        return False
    so_mtime = os.path.getmtime(so_path)
    src = os.path.join(_LIB_DIR, 'src')
    inc = os.path.join(_LIB_DIR, 'include')
    for d in [src, inc]:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.getmtime(os.path.join(root, f)) > so_mtime:
                    return True
    return False


def _build_cuda_so():
    """Build a self-contained libdress_cuda.so from source (cudart_static baked in)."""
    import shutil
    import subprocess

    nvcc = shutil.which('nvcc')
    if nvcc is None:
        raise RuntimeError("nvcc not found — cannot auto-build CUDA library.")

    inc = os.path.join(_LIB_DIR, 'include')
    src = os.path.join(_LIB_DIR, 'src')
    cuda_dir = os.path.join(src, 'cuda')

    cuda_cu = os.path.join(cuda_dir, 'dress_cuda.cu')
    cuda_obj = os.path.join(cuda_dir, 'dress_cuda.o')

    # Compile CUDA kernel with nvcc (recompile if sources changed)
    if not os.path.isfile(cuda_obj) or _sources_newer_than(cuda_obj):
        subprocess.check_call([
            nvcc, '-O2', '-Xcompiler', '-fPIC', f'-I{inc}',
            '-c', cuda_cu, '-o', cuda_obj,
        ])

    cc = os.environ.get('CC', 'gcc')
    c_srcs = [
        os.path.join(src, 'dress.c'),
        os.path.join(src, 'dress_histogram.c'),
        os.path.join(src, 'delta_dress.c'),
        os.path.join(src, 'delta_dress_impl.c'),
        os.path.join(src, 'omp', 'dress_omp.c'),
        os.path.join(src, 'omp', 'delta_dress_omp.c'),
        os.path.join(cuda_dir, 'delta_dress_cuda.c'),
    ]
    subprocess.check_call([
        cc, '-shared', '-fPIC', '-O3', '-fopenmp', '-DDRESS_CUDA',
        f'-I{inc}', f'-I{src}',
        '-o', _LOCAL_SO,
        *c_srcs, cuda_obj,
        '-lcudart_static', '-lm', '-ldl', '-lrt', '-lpthread',
    ])


def _get_lib():
    """Load libdress_cuda.so on first use, auto-building if needed."""
    global _lib
    if _lib is not None:
        return _lib

    # Try pre-built .so first, then system path
    lib = None
    for path in [_LOCAL_SO, 'libdress_cuda.so']:
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError:
            continue

    # Stale check: rebuild if sources are newer than .so
    if lib is not None and path == _LOCAL_SO and _sources_newer_than(_LOCAL_SO):
        lib = None

    # Auto-build from source if nvcc is available
    if lib is None:
        _build_cuda_so()
        lib = ctypes.CDLL(_LOCAL_SO)

    # Bind C function signatures
    lib.dress_init_graph.restype  = _p_dress_graph_t
    lib.dress_init_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
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

    lib.dress_delta_fit_cuda.restype  = ctypes.POINTER(_DressHistPair)
    lib.dress_delta_fit_cuda.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.dress_delta_fit_cuda_strided.restype  = ctypes.POINTER(_DressHistPair)
    lib.dress_delta_fit_cuda_strided.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
    ]

    lib.dress_nabla_fit_cuda.restype  = ctypes.POINTER(_DressHistPair)
    lib.dress_nabla_fit_cuda.argtypes = [
        _p_dress_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_int64),
    ]

    lib.dress_free_graph.restype  = None
    lib.dress_free_graph.argtypes = [_p_dress_graph_t]

    _lib = lib
    return _lib


def _decode_histogram(hist_ptr, size):
    if not hist_ptr or size == 0:
        return []
    pairs = np.ctypeslib.as_array(hist_ptr, shape=(size,)).copy()
    lib = _get_lib()
    lib.free(hist_ptr)
    return [(float(pair["value"]), int(pair["count"])) for pair in pairs]


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





def dress_cuda(N, E, U, V, W=None, NW=None, variant=UNDIRECTED,
               precompute_intercepts=0, max_iterations=100, epsilon=1e-10):
    """
    Compute DRESS edge and vertex values using GPU acceleration.

    Parameters
    ----------
    N : int — number of nodes
    E : int — number of edges
    U : array-like[int] — source vertex for each edge (length E)
    V : array-like[int] — target vertex for each edge (length E)
    W : array-like[float] or None — edge weights (None = unweighted)
    NW : array-like[float] or None — vertex weights (None = all 1.0)
    variant : int — 0=UNDIRECTED, 1=DIRECTED, 2=FORWARD, 3=BACKWARD
    precompute_intercepts : int — 1 to precompute intercepts (faster for dense)
    max_iterations : int
    epsilon : float

    Returns
    -------
    edge_dress : np.ndarray[float64] of shape (E,)
    vertex_dress : np.ndarray[float64] of shape (N,)
    iterations : int
    delta : float
    """
    # Allocate C arrays with malloc — dress_init_graph takes ownership
    # and dress_free_graph calls free() on them.
    p_U = _malloc_array([int(x) for x in U], ctypes.c_int)
    p_V = _malloc_array([int(x) for x in V], ctypes.c_int)

    if W is not None:
        p_W = _malloc_array([float(x) for x in W], ctypes.c_double)
    else:
        p_W = ctypes.POINTER(ctypes.c_double)()  # NULL

    if NW is not None:
        p_NW = _malloc_array([float(x) for x in NW], ctypes.c_double)
    else:
        p_NW = ctypes.POINTER(ctypes.c_double)()  # NULL

    lib = _get_lib()

    # Build graph on CPU
    g = lib.dress_init_graph(N, E, p_U, p_V, p_W, p_NW,
                             ctypes.c_int(variant),
                             ctypes.c_int(precompute_intercepts))

    # Run GPU fit
    iterations = ctypes.c_int(0)
    delta      = ctypes.c_double(0.0)
    lib.dress_fit_cuda(g, max_iterations, ctypes.c_double(epsilon),
                       ctypes.byref(iterations), ctypes.byref(delta))

    # Extract results
    edge_dress  = np.ctypeslib.as_array(g.contents.edge_dress,  shape=(E,)).copy()
    vertex_dress  = np.ctypeslib.as_array(g.contents.vertex_dress,  shape=(N,)).copy()
    edge_weight = np.ctypeslib.as_array(g.contents.edge_weight, shape=(E,)).copy()

    iters = iterations.value
    delta_val = delta.value

    # Free C memory
    lib.dress_free_graph(g)

    return edge_dress, vertex_dress, edge_weight, iters, delta_val


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


def dress_delta_fit_cuda(g_ptr, k, max_iterations=100, epsilon=1e-10,
                          n_samples=0, seed=0, keep_multisets=False):
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

    hist_ptr = lib.dress_delta_fit_cuda(
        g_ptr, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
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
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """GPU-accelerated ∇^k-DRESS — drop-in replacement for ``dress.nabla_fit``.

    Same parameters and return type as :func:`dress.dress_nabla_fit`, but
    each tuple fitting runs on the GPU.
    """
    from dress.core import NablaDRESSResult

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

    hist_ptr = lib.dress_nabla_fit_cuda(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_tuples))

    size = hist_size.value
    ns = num_tuples.value
    histogram = _decode_histogram(hist_ptr, size)

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
#  Unified API — same signatures as dress.dress_fit / dress.dress_delta_fit
# ---------------------------------------------------------------------------

def fit(
    n_vertices,
    sources,
    targets,
    weights=None,
    vertex_weights=None,
    variant=UNDIRECTED,
    max_iterations=100,
    epsilon=1e-6,
    precompute_intercepts=False,
):
    """GPU-accelerated DRESS fitting — drop-in replacement for ``dress.fit``.

    Same parameters and return type as :func:`dress.dress_fit`, but the
    iterative fitting loop runs on the GPU.
    """
    from dress.core import DRESSResult

    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    edge_dress, vertex_dress, edge_weight, iters, delta = dress_cuda(
        N=n_vertices,
        E=E,
        U=sources,
        V=targets,
        W=list(weights) if weights is not None else None,
        NW=list(vertex_weights) if vertex_weights is not None else None,
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
        vertex_dress=vertex_dress.tolist(),
        iterations=iters,
        delta=delta,
    )


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
    offset=0,
    stride=1,
    n_samples=0,
    seed=0,
    compute_histogram=True,
):
    """GPU-accelerated Δ^k-DRESS — drop-in replacement for ``dress.delta_fit``.

    Same parameters and return type as :func:`dress.dress_delta_fit`, but
    each subgraph fitting runs on the GPU.

    Parameters *offset* and *stride* select a subset of the C(N,k) subgraphs
    (process only those where ``index % stride == offset``).
    """
    from dress.core import DeltaDRESSResult

    lib = _get_lib()
    sources = list(sources)
    targets = list(targets)
    E = len(sources)

    # Build the graph via ctypes — use malloc so dress_free_graph works
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

    hist_ptr = lib.dress_delta_fit_cuda_strided(
        g, k, max_iterations, ctypes.c_double(epsilon),
        n_samples, ctypes.c_uint(seed),
        ctypes.byref(hist_size) if compute_histogram else None,
        1 if keep_multisets else 0,
        ctypes.byref(multisets_ptr),
        ctypes.byref(num_subgraphs),
        offset, stride)

    size = hist_size.value
    ns = num_subgraphs.value
    histogram = _decode_histogram(hist_ptr, size)

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


# ---------------------------------------------------------------------------
#  DRESS class — same API as dress.DRESS, fit/delta_fit run on GPU
# ---------------------------------------------------------------------------

from dress import DRESS as _BaseDRESS  # noqa: E402
from dress.core import FitResult as _FitResult  # noqa: E402


class DRESS(_BaseDRESS):
    """GPU-accelerated DRESS — same API, ``fit``/``delta_fit`` run on CUDA.

    Usage::

        from dress.cuda import DRESS

        g = DRESS(4, [0, 1, 2, 0], [1, 2, 3, 3])
        fr = g.fit()          # GPU
        dr = g.delta_fit(k=2) # GPU
        val = g.get(0, 2)     # CPU (uses converged state)
    """

    _force_python_impl = True

    def fit(self, max_iterations=100, epsilon=1e-6):
        result = fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, vertex_weights=self._nwgt,
            variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
        )
        self._sync_hardware_fit(result)
        return _FitResult(iterations=result.iterations, delta=result.delta)

    def delta_fit(self, k=0, max_iterations=100, epsilon=1e-6,
                  n_samples=0, seed=0,
                  keep_multisets=False, compute_histogram=True,
                  offset=0, stride=1):
        return delta_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, vertex_weights=self._nwgt,
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
            weights=self._wgt, vertex_weights=self._nwgt,
            k=k, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
            keep_multisets=keep_multisets,
            n_samples=n_samples, seed=seed,
            compute_histogram=compute_histogram,
        )

    def __repr__(self):
        return (
            f"DRESS(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"variant={self._var.name}, backend=cuda)"
        )
