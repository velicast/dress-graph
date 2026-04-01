"""
    DRESS.CUDA

GPU-accelerated DRESS — same API as the CPU functions, different import.

# Usage (import-based switching)

```julia
# CPU
using DRESS
result = fit(4, [0,1,2,0], [1,2,3,3])

# CUDA — same call, different import
using DRESS.CUDA
result = fit(4, [0,1,2,0], [1,2,3,3])
```
"""
module CUDA

using Libdl

# Re-export types and constants from the parent module.
using ..DRESS: DRESSResult, DeltaDRESSResult, NablaDRESSResult, HistogramEntry, UNDIRECTED, DIRECTED, FORWARD, BACKWARD, _copy_histogram
import ..DRESS: dress_build as _cpu_build
export fit, delta_fit, nabla_fit, DRESSResult, DeltaDRESSResult, NablaDRESSResult,
       UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── locate CUDA shared library ───────────────────────────────────────

const _PKG_DIR  = dirname(@__DIR__)
const _LIB_DIR  = normpath(joinpath(_PKG_DIR, "..", "libdress"))
const _CUDA_SO  = "libdress_cuda" * (Sys.iswindows() ? ".dll" :
                                      Sys.isapple()   ? ".dylib" : ".so")
const _CUDA_PATH = joinpath(_PKG_DIR, _CUDA_SO)

# Also need the CPU library for dress_init_graph / dress_free_graph.
const _CPU_SO   = "libdress" * (Sys.iswindows() ? ".dll" :
                                 Sys.isapple()   ? ".dylib" : ".so")
const _CPU_PATH = joinpath(_PKG_DIR, _CPU_SO)

# Pre-compiled CUDA kernel object (built from dress_cuda.cu by nvcc on first use)
const _CUDA_OBJ = joinpath(_PKG_DIR, "dress_cuda.o")
const _CUDA_CU  = joinpath(_LIB_DIR, "src", "cuda", "dress_cuda.cu")

"""
    dress_cuda_build()

Compile `libdress_cuda.so` from vendored C sources + the CUDA kernel.
Called automatically on first use if the `.so` is missing and `nvcc` is
available.  The CUDA kernel (`dress_cuda.cu`) is compiled by `nvcc` into
an object file, then everything is linked together with `cudart_static`.
"""
function dress_cuda_build()
    src       = joinpath(_LIB_DIR, "src", "dress.c")
    src_delta = joinpath(_LIB_DIR, "src", "delta_dress.c")
    src_impl  = joinpath(_LIB_DIR, "src", "delta_dress_impl.c")
    src_hist  = joinpath(_LIB_DIR, "src", "dress_histogram.c")
    src_cuda  = joinpath(_LIB_DIR, "src", "cuda", "delta_dress_cuda.c")
    inc       = joinpath(_LIB_DIR, "include")
    src_dir   = joinpath(_LIB_DIR, "src")

    for f in (src, src_delta, src_impl, src_hist, src_cuda)
        isfile(f) || error("Cannot find $f")
    end

    # Compile CUDA kernel with nvcc if the object doesn't exist yet
    if !isfile(_CUDA_OBJ)
        isfile(_CUDA_CU) || error("Cannot find CUDA kernel source at $_CUDA_CU")
        nvcc = get(ENV, "NVCC", "nvcc")
        nvcc_cmd = `$nvcc -O2 -Xcompiler -fPIC -I$inc -c $_CUDA_CU -o $_CUDA_OBJ`
        @info "Compiling CUDA kernel…" nvcc_cmd
        run(nvcc_cmd)
    end

    cc = get(ENV, "CC", "gcc")
    cmd = `$cc -shared -fPIC -O3 -fopenmp -DDRESS_CUDA -I$inc -I$src_dir
           -o $_CUDA_PATH
            $src $src_delta $src_impl $src_hist $src_cuda $_CUDA_OBJ
           -lcudart_static -lm -ldl -lrt -lpthread`
    @info "Building DRESS CUDA shared library…" cmd
    run(cmd)
    @info "Built $_CUDA_PATH"
end

# ── library handles and function pointers ────────────────────────────

const _LIB_CPU    = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_CUDA   = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA   = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA_STRIDED = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA   = Ref{Ptr{Cvoid}}(C_NULL)

function _try_dlopen(name::String, local_path::String; build_fn::Union{Nothing,Function}=nothing)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    if build_fn !== nothing && !isfile(local_path)
        build_fn()
    end
    isfile(local_path) || error("Library $name not found.")
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_CUDA[] != C_NULL && return

    # Load CPU library (init/free live there)
    if _LIB_CPU[] == C_NULL
        _LIB_CPU[] = _try_dlopen(_CPU_SO, _CPU_PATH; build_fn=_cpu_build)
    end
    _FN_INIT[] = dlsym(_LIB_CPU[], :dress_init_graph)
    _FN_FREE[] = dlsym(_LIB_CPU[], :dress_free_graph)

    # Load CUDA library — auto-build from sources if nvcc is available
    has_nvcc = Sys.which("nvcc") !== nothing
    _LIB_CUDA[] = _try_dlopen(_CUDA_SO, _CUDA_PATH;
                              build_fn = has_nvcc ? dress_cuda_build : nothing)

    _FN_FIT[]   = dlsym(_LIB_CUDA[], :dress_fit_cuda)
    _FN_DELTA[] = dlsym(_LIB_CUDA[], :dress_delta_fit_cuda)
    _FN_DELTA_STRIDED[] = dlsym(_LIB_CUDA[], :dress_delta_fit_cuda_strided)
    _FN_NABLA[] = dlsym(_LIB_CUDA[], :dress_nabla_fit_cuda)
end

# ── fit (CUDA) ─────────────────────────────────────────────────

"""
    fit(N, sources, targets; kwargs...) → DRESSResult

GPU-accelerated DRESS fitting.  Same signature as `DRESS.fit()`.
"""
function fit(N::Integer,
                   sources::AbstractVector{<:Integer},
                   targets::AbstractVector{<:Integer};
                   weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   vertex_weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   variant::Integer   = UNDIRECTED,
                   max_iterations::Integer = 100,
                   epsilon::Real      = 1e-6,
                   precompute_intercepts::Bool = true)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))
    if weights !== nothing
        length(weights) == E || throw(ArgumentError("weights must have the same length as sources"))
    end
    if vertex_weights !== nothing
        length(vertex_weights) == N || throw(ArgumentError("vertex_weights must have length N"))
    end

    _ensure_lib()

    U_c = Libc.malloc(E * sizeof(Cint))
    V_c = Libc.malloc(E * sizeof(Cint))

    u_arr = unsafe_wrap(Array, Ptr{Cint}(U_c), E)
    v_arr = unsafe_wrap(Array, Ptr{Cint}(V_c), E)
    u_arr .= Cint.(sources)
    v_arr .= Cint.(targets)

    W_c = C_NULL
    if weights !== nothing
        W_c = Libc.malloc(E * sizeof(Cdouble))
        w_arr = unsafe_wrap(Array, Ptr{Cdouble}(W_c), E)
        w_arr .= Cdouble.(weights)
    end

    NW_c = C_NULL
    if vertex_weights !== nothing
        NW_c = Libc.malloc(N * sizeof(Cdouble))
        nw_arr = unsafe_wrap(Array, Ptr{Cdouble}(NW_c), N)
        nw_arr .= Cdouble.(vertex_weights)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), Ptr{Cdouble}(W_c), Ptr{Cdouble}(NW_c),
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("dress_init_graph returned NULL")

    iters_ref = Ref{Cint}(0)
    delta_ref = Ref{Cdouble}(0.0)
    ccall(_FN_FIT[], Cvoid,
          (Ptr{Cvoid}, Cint, Cdouble, Ptr{Cint}, Ptr{Cdouble}),
          g, Cint(max_iterations), Cdouble(epsilon),
          iters_ref, delta_ref)

    # Read struct fields (LP64 offsets — same as CPU)
    edge_weight_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 72))
    edge_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 80))
    node_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 96))
    nw_ptr          = unsafe_load(Ptr{Ptr{Cdouble}}(g + 104))

    ew = copy(unsafe_wrap(Array, edge_weight_ptr, E))
    ed = copy(unsafe_wrap(Array, edge_dress_ptr,  E))
    nd = copy(unsafe_wrap(Array, node_dress_ptr,  Cint(N)))
    src_out = copy(unsafe_wrap(Array, Ptr{Cint}(U_c), E))
    tgt_out = copy(unsafe_wrap(Array, Ptr{Cint}(V_c), E))

    nw_out = nothing
    if nw_ptr != C_NULL
        nw_out = copy(unsafe_wrap(Array, nw_ptr, Cint(N)))
    end

    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DRESSResult(src_out, tgt_out, ew, ed, nd,
                       Int(iters_ref[]), Float64(delta_ref[]), nw_out)
end

# ── delta_fit (CUDA) ───────────────────────────────────────────

"""
    delta_fit(N, sources, targets; kwargs...) → DeltaDRESSResult

GPU-accelerated Δ^k-DRESS.  Same signature as `DRESS.delta_fit()`.
"""
function delta_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         vertex_weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         k::Integer         = 0,
                         variant::Integer   = UNDIRECTED,
                         max_iterations::Integer = 100,
                         epsilon::Real      = 1e-6,
                         n_samples::Integer = 0,
                         seed::Integer      = 0,
                         precompute::Bool   = false,
                         keep_multisets::Bool = false,
                         compute_histogram::Bool = true)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))

    _ensure_lib()

    U_c = Libc.malloc(E * sizeof(Cint))
    V_c = Libc.malloc(E * sizeof(Cint))

    u_arr = unsafe_wrap(Array, Ptr{Cint}(U_c), E)
    v_arr = unsafe_wrap(Array, Ptr{Cint}(V_c), E)
    u_arr .= Cint.(sources)
    v_arr .= Cint.(targets)

    W_c = if weights !== nothing
        w_ptr = Libc.malloc(E * sizeof(Cdouble))
        w_arr = unsafe_wrap(Array, Ptr{Cdouble}(w_ptr), E)
        w_arr .= Cdouble.(weights)
        Ptr{Cdouble}(w_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    NW_c = if vertex_weights !== nothing
        nw_ptr = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N) .= Cdouble.(vertex_weights)
        Ptr{Cdouble}(nw_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c, NW_c,
              Cint(variant), Cint(precompute))

    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)
    h_ptr = ccall(_FN_DELTA_STRIDED[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref,
                  Cint(0), Cint(1))

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    ms_mat = nothing
    nsub = Int(nsub_ref[])
    if keep_multisets
        ms_p = ms_ref[]
        if ms_p != Ptr{Cdouble}(C_NULL) && nsub > 0
            flat = copy(unsafe_wrap(Array, ms_p, nsub * E))
            ms_mat = permutedims(reshape(flat, E, nsub))
            Libc.free(ms_p)
        else
            ms_mat = Matrix{Float64}(undef, 0, 0)
        end
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DeltaDRESSResult(histogram, ms_mat, nsub)
end

# ── nabla_fit (CUDA) ───────────────────────────────────────────

"""
    nabla_fit(N, sources, targets; kwargs...) → NablaDRESSResult

GPU-accelerated ∇^k-DRESS.  Same signature as `DRESS.nabla_fit()`.
"""
function nabla_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         vertex_weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         k::Integer         = 0,
                         variant::Integer   = UNDIRECTED,
                         max_iterations::Integer = 100,
                         epsilon::Real      = 1e-6,
                         n_samples::Integer = 0,
                         seed::Integer      = 0,
                         precompute::Bool   = false,
                         keep_multisets::Bool = false,
                         compute_histogram::Bool = true)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))

    _ensure_lib()

    U_c = Libc.malloc(E * sizeof(Cint))
    V_c = Libc.malloc(E * sizeof(Cint))

    u_arr = unsafe_wrap(Array, Ptr{Cint}(U_c), E)
    v_arr = unsafe_wrap(Array, Ptr{Cint}(V_c), E)
    u_arr .= Cint.(sources)
    v_arr .= Cint.(targets)

    W_c = if weights !== nothing
        w_ptr = Libc.malloc(E * sizeof(Cdouble))
        w_arr = unsafe_wrap(Array, Ptr{Cdouble}(w_ptr), E)
        w_arr .= Cdouble.(weights)
        Ptr{Cdouble}(w_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    NW_c = if vertex_weights !== nothing
        nw_ptr = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N) .= Cdouble.(vertex_weights)
        Ptr{Cdouble}(nw_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c, NW_c,
              Cint(variant), Cint(precompute))

    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    ntup_ref  = Ref{Int64}(0)
    h_ptr = ccall(_FN_NABLA[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  ntup_ref)

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    ms_mat = nothing
    ntup = Int(ntup_ref[])
    if keep_multisets
        ms_p = ms_ref[]
        if ms_p != Ptr{Cdouble}(C_NULL) && ntup > 0
            flat = copy(unsafe_wrap(Array, ms_p, ntup * E))
            ms_mat = permutedims(reshape(flat, E, ntup))
            Libc.free(ms_p)
        else
            ms_mat = Matrix{Float64}(undef, 0, 0)
        end
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return NablaDRESSResult(histogram, ms_mat, ntup)
end

# ── persistent DressGraph (CUDA) ─────────────────────────────────────

"""
    DressGraph

A persistent DRESS graph whose `fit!` runs on the GPU via CUDA.
Same API as `DRESS.DressGraph`, different import.

```julia
using DRESS.CUDA
g = DressGraph(4, [0,1,2,0], [1,2,3,3])
fit!(g)
d = get(g, 0, 2)
close!(g)
```
"""
mutable struct DressGraph
    ptr      :: Ptr{Cvoid}
    n        :: Int
    e        :: Int
    sources  :: Vector{Int32}
    targets  :: Vector{Int32}
end

export DressGraph, fit!, delta_fit!, nabla_fit!, close!

"""
    DressGraph(N, sources, targets; weights=nothing, vertex_weights=nothing,
               variant=UNDIRECTED, precompute_intercepts=false)

Create a persistent CUDA DRESS graph object.
"""
function DressGraph(N::Integer,
                    sources::AbstractVector{<:Integer},
                    targets::AbstractVector{<:Integer};
                    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                    vertex_weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                    variant::Integer   = UNDIRECTED,
                    precompute_intercepts::Bool = false)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources/targets length mismatch"))
    if weights !== nothing
        length(weights) == E || throw(ArgumentError("weights length mismatch"))
    end
    if vertex_weights !== nothing
        length(vertex_weights) == N || throw(ArgumentError("vertex_weights length mismatch"))
    end

    _ensure_lib()

    U_c = Libc.malloc(E * sizeof(Cint))
    V_c = Libc.malloc(E * sizeof(Cint))
    unsafe_wrap(Array, Ptr{Cint}(U_c), E) .= Cint.(sources)
    unsafe_wrap(Array, Ptr{Cint}(V_c), E) .= Cint.(targets)

    W_c = if weights !== nothing
        w_ptr = Libc.malloc(E * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(w_ptr), E) .= Cdouble.(weights)
        Ptr{Cdouble}(w_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    NW_c = if vertex_weights !== nothing
        nw_ptr = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N) .= Cdouble.(vertex_weights)
        Ptr{Cdouble}(nw_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c, NW_c,
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("dress_init_graph returned NULL")

    obj = DressGraph(g, Int(N), E, Int32.(sources), Int32.(targets))
    finalizer(obj) do o
        close!(o)
    end
    obj
end

"""
    fit!(g::DressGraph; max_iterations=100, epsilon=1e-6) → (iterations, delta)

Fit the DRESS model on the GPU via CUDA.
"""
function fit!(g::DressGraph; max_iterations::Integer=100, epsilon::Real=1e-6)
    g.ptr == C_NULL && error("DressGraph already closed")
    iters_ref = Ref{Cint}(0)
    delta_ref = Ref{Cdouble}(0.0)
    ccall(_FN_FIT[], Cvoid,
          (Ptr{Cvoid}, Cint, Cdouble, Ptr{Cint}, Ptr{Cdouble}),
          g.ptr, Cint(max_iterations), Cdouble(epsilon),
          iters_ref, delta_ref)
    (Int(iters_ref[]), Float64(delta_ref[]))
end

# get — uses CPU dress_get (single-pair too small for GPU)
const _FN_GET = Ref{Ptr{Cvoid}}(C_NULL)

function _ensure_get()
    _FN_GET[] != C_NULL && return
    _ensure_lib()
    # dress_get lives in the CPU library
    _FN_GET[] = dlsym(_LIB_CPU[], :dress_get)
end

"""
    Base.get(g::DressGraph, u, v; max_iterations=100, epsilon=1e-6, edge_weight=1.0)

Query the DRESS value for an edge (existing or virtual).  Runs on the CPU.
"""
function Base.get(g::DressGraph, u::Integer, v::Integer;
                  max_iterations::Integer=100, epsilon::Real=1e-6,
                  edge_weight::Real=1.0)
    g.ptr == C_NULL && error("DressGraph already closed")
    _ensure_get()
    ccall(_FN_GET[], Cdouble,
          (Ptr{Cvoid}, Cint, Cint, Cint, Cdouble, Cdouble),
          g.ptr, Cint(u), Cint(v), Cint(max_iterations),
          Cdouble(epsilon), Cdouble(edge_weight))
end

"""
    result(g::DressGraph) → DRESSResult

Extract a snapshot of the current DRESS results without freeing.
"""
function result(g::DressGraph)
    g.ptr == C_NULL && error("DressGraph already closed")
    ew = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 72)), g.e))
    ed = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 80)), g.e))
    nd = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 96)), g.n))

    nw_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 104))
    nw = nothing
    if nw_ptr != C_NULL
        nw = copy(unsafe_wrap(Array, nw_ptr, g.n))
    end

    DRESSResult(copy(g.sources), copy(g.targets), ew, ed, nd, 0, 0.0, nw)
end

export result

"""
    delta_fit!(g::DressGraph, k; max_iterations=100, epsilon=1e-6,
              n_samples=0, seed=0,
              keep_multisets=false, compute_histogram=true) → DeltaDRESSResult

GPU-accelerated Δ^k-DRESS on a persistent graph.
"""
function delta_fit!(g::DressGraph, k::Integer;
                    max_iterations::Integer=100,
                    epsilon::Real=1e-6,
                    n_samples::Integer=0,
                    seed::Integer=0,
                    keep_multisets::Bool=false,
                    compute_histogram::Bool=true)
    g.ptr == C_NULL && error("DressGraph already closed")
    _ensure_lib()

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_DELTA_STRIDED[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint,
                   Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint, Cint),
                  g.ptr, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref,
                  Cint(0), Cint(1))

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    ms_mat = nothing
    ns = Int(nsub_ref[])
    E = g.e
    if keep_multisets
        ms_p = ms_ref[]
        if ms_p != Ptr{Cdouble}(C_NULL) && ns > 0
            flat = copy(unsafe_wrap(Array, ms_p, ns * E))
            ms_mat = permutedims(reshape(flat, E, ns))
            Libc.free(ms_p)
        else
            ms_mat = Matrix{Float64}(undef, 0, 0)
        end
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end

    return DeltaDRESSResult(histogram, ms_mat, ns)
end

"""
    nabla_fit!(g::DressGraph, k; max_iterations=100, epsilon=1e-6,
               keep_multisets=false) → NablaDRESSResult

GPU-accelerated ∇^k-DRESS on a persistent graph.
"""
function nabla_fit!(g::DressGraph, k::Integer;
                    max_iterations::Integer=100,
                    epsilon::Real=1e-6,
                    n_samples::Integer=0,
                    seed::Integer=0,
                    keep_multisets::Bool=false,
                    compute_histogram::Bool=true)
    g.ptr == C_NULL && error("DressGraph already closed")
    _ensure_lib()

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    ntup_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_NABLA[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g.ptr, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  ntup_ref)

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    ms_mat = nothing
    ntup = Int(ntup_ref[])
    E = g.e
    if keep_multisets
        ms_p = ms_ref[]
        if ms_p != Ptr{Cdouble}(C_NULL) && ntup > 0
            flat = copy(unsafe_wrap(Array, ms_p, ntup * E))
            ms_mat = permutedims(reshape(flat, E, ntup))
            Libc.free(ms_p)
        else
            ms_mat = Matrix{Float64}(undef, 0, 0)
        end
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end

    return NablaDRESSResult(histogram, ms_mat, ntup)
end

"""
    close!(g::DressGraph)

Explicitly free the underlying C graph.
"""
function close!(g::DressGraph)
    if g.ptr != C_NULL
        ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g.ptr)
        g.ptr = C_NULL
    end
    nothing
end

function Base.show(io::IO, g::DressGraph)
    state = g.ptr == C_NULL ? "closed" : "open"
    print(io, "DressGraph{CUDA}(N=$(g.n), E=$(g.e), $state)")
end

end # module CUDA
