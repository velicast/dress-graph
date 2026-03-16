"""Shared ctypes helpers used by dress.cuda, dress.mpi, and dress.mpi.cuda."""

import ctypes


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
        ("max_degree",             ctypes.c_int),
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

# libc handle for malloc-compatible allocation
_libc = ctypes.CDLL(None)
_libc.malloc.restype = ctypes.c_void_p
_libc.malloc.argtypes = [ctypes.c_size_t]


def _malloc_array(arr, ctype):
    """Allocate a C malloc'd array and copy data into it.

    The C library (free_dress_graph) will call free() on these pointers,
    so they MUST be allocated with C malloc — not Python ctypes arrays.
    """
    n = len(arr)
    nbytes = n * ctypes.sizeof(ctype)
    raw = _libc.malloc(nbytes)
    if not raw:
        raise MemoryError(f"malloc({nbytes}) failed")
    ptr = ctypes.cast(raw, ctypes.POINTER(ctype))
    tmp = (ctype * n)(*arr)
    ctypes.memmove(ptr, tmp, nbytes)
    return ptr
