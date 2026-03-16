"""``dress.mpi.cuda`` — MPI-distributed DRESS (CUDA backend).

Switch to the MPI+CUDA backend by changing the import::

    from dress.mpi.cuda import DRESS

    g = DRESS(N, sources, targets)
    g.fit()           # GPU (single graph)
    g.delta_fit(k=3)  # MPI-distributed, each rank uses GPU
    g.get(0, 1)       # CPU

Requires ``libdress.so`` built with ``-DDRESS_MPI=ON``,
``libdress_cuda.so``, and ``mpi4py``.  All MPI logic (stride
partitioning + Allreduce) runs in C.  The wrapper passes the
Fortran communicator handle via ``comm.py2f()``.

Module-level functions are also available::

    from dress.mpi.cuda import delta_dress_fit
"""

import ctypes
import os
import numpy as np

from dress import UNDIRECTED
from dress.core import DeltaDRESSResult
from dress._ctypes_helpers import _DressGraph, _p_dress_graph_t, _malloc_array

# ---------------------------------------------------------------------------
#  Lazy-load libdress.so (with MPI + CUDA support)
# ---------------------------------------------------------------------------

_lib = None
_cuda_lib = None

# Path constants for auto-build
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))  # dress/
_VENDORED = os.path.join(_PKG_DIR, '_vendored')              # dress/_vendored/
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
# Prefer vendored sources (pip install) over repo-relative (editable install)
if os.path.isdir(_VENDORED):
    _LIB_DIR = _VENDORED
else:
    _LIB_DIR = os.path.join(_ROOT, 'libdress')
_LOCAL_SO = os.path.join(_LIB_DIR, 'src', 'cuda', 'libdress_mpi_cuda.so')


def _sources_newer_than(so_path):
    """Return True if any vendored source is newer than the .so."""
    if not os.path.isfile(so_path):
        return True
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
    """Build a self-contained libdress_mpi_cuda.so from source (cudart_static + MPI baked in)."""
    import shutil
    import subprocess

    nvcc = shutil.which('nvcc')
    if nvcc is None:
        raise RuntimeError("nvcc not found — cannot auto-build CUDA library.")
    mpicc = shutil.which('mpicc')
    if mpicc is None:
        raise RuntimeError("mpicc not found — cannot auto-build MPI+CUDA library.")

    inc = os.path.join(_LIB_DIR, 'include')
    src = os.path.join(_LIB_DIR, 'src')
    cuda_dir = os.path.join(src, 'cuda')

    cuda_cu = os.path.join(cuda_dir, 'dress_cuda.cu')
    cuda_obj = os.path.join(cuda_dir, 'dress_mpi_cuda.o')

    # Compile CUDA kernel with nvcc (recompile if sources changed)
    if not os.path.isfile(cuda_obj) or _sources_newer_than(cuda_obj):
        subprocess.check_call([
            nvcc, '-O2', '-Xcompiler', '-fPIC', f'-I{inc}',
            '-c', cuda_cu, '-o', cuda_obj,
        ])

    # Get MPI flags
    mpi_cflags = subprocess.check_output(
        [mpicc, '--showme:compile'], text=True).strip().split()
    mpi_ldflags = subprocess.check_output(
        [mpicc, '--showme:link'], text=True).strip().split()

    cc = os.environ.get('CC', 'gcc')
    c_srcs = [
        os.path.join(src, 'dress.c'),
        os.path.join(src, 'delta_dress.c'),
        os.path.join(src, 'delta_dress_impl.c'),
        os.path.join(cuda_dir, 'delta_dress_cuda.c'),
        os.path.join(src, 'mpi', 'dress_mpi.c'),
    ]
    subprocess.check_call([
        cc, '-shared', '-fPIC', '-O3', '-fopenmp', '-DDRESS_CUDA',
        f'-I{inc}', f'-I{src}',
        *mpi_cflags,
        '-o', _LOCAL_SO,
        *c_srcs, cuda_obj,
        '-lcudart_static', '-lm', '-ldl', '-lrt', '-lpthread',
        *mpi_ldflags,
    ])


def _has_mpi_symbols(lib):
    """Check if a loaded .so contains the MPI+CUDA entry point."""
    try:
        lib.delta_dress_fit_mpi_cuda_fcomm
        return True
    except AttributeError:
        return False


def _get_lib():
    """Load libdress_mpi_cuda.so on first use, auto-building if needed."""
    global _lib, _cuda_lib
    if _lib is not None:
        return _lib

    # Try pre-built .so first, then system-level libdress_cuda.so
    _cuda_lib = None
    for path in [_LOCAL_SO, 'libdress_mpi_cuda.so', 'libdress_cuda.so']:
        try:
            candidate = ctypes.CDLL(path)
            if _has_mpi_symbols(candidate):
                _cuda_lib = candidate
                break
        except OSError:
            continue

    # Stale check: rebuild if sources are newer than .so
    if _cuda_lib is not None and path == _LOCAL_SO and _sources_newer_than(_LOCAL_SO):
        _cuda_lib = None

    # Auto-build from source if tools available
    if _cuda_lib is None:
        _build_cuda_so()
        _cuda_lib = ctypes.CDLL(_LOCAL_SO)

    # The .so contains everything (CPU + CUDA + MPI), so lib == _cuda_lib
    lib = _cuda_lib

    lib.init_dress_graph.restype = _p_dress_graph_t
    lib.init_dress_graph.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
    ]

    _cuda_lib.delta_dress_fit_mpi_cuda_fcomm.restype = ctypes.POINTER(ctypes.c_int64)
    _cuda_lib.delta_dress_fit_mpi_cuda_fcomm.argtypes = [
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
    """MPI-distributed Δ^k-DRESS (CUDA backend).

    All MPI logic (stride partitioning + Allreduce) runs in C.
    Each rank runs GPU-accelerated DRESS on its stride of subgraphs.
    Uses ``comm.py2f()`` to pass the Fortran MPI handle.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        MPI communicator (default: ``MPI.COMM_WORLD``).
    All other parameters are identical to :func:`dress.cuda.delta_dress_fit`.
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

    hist_ptr = _cuda_lib.delta_dress_fit_mpi_cuda_fcomm(
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
#  DRESS class — same API as dress.DRESS, fit uses CUDA, delta_fit uses MPI+CUDA
# ---------------------------------------------------------------------------

from dress import DRESS as _BaseDRESS  # noqa: E402
from dress.core import FitResult as _FitResult  # noqa: E402


class DRESS(_BaseDRESS):
    """MPI+CUDA DRESS — same API, ``fit`` on GPU, ``delta_fit`` MPI+GPU.

    ``fit()`` runs on the GPU (single graph).  ``delta_fit()`` distributes
    C(N,k) subgraph enumeration across MPI ranks, each using the GPU.
    ``get()`` runs on CPU.

    Usage::

        from dress.mpi.cuda import DRESS

        g = DRESS(4, [0, 1, 2, 0], [1, 2, 3, 3])
        g.fit()               # GPU
        dr = g.delta_fit(k=3) # MPI + GPU
    """

    _force_python_impl = True

    def fit(self, max_iterations=100, epsilon=1e-6):
        from dress.cuda import dress_fit as _cuda_fit
        result = _cuda_fit(
            self._n_v, self._src, self._tgt,
            weights=self._wgt, variant=int(self._var),
            max_iterations=max_iterations, epsilon=epsilon,
        )
        self._sync_hardware_fit(result)
        return _FitResult(iterations=result.iterations, delta=result.delta)

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
            f"variant={self._var.name}, backend=mpi.cuda)"
        )
