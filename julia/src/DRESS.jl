"""
    DRESS

Julia package for the DRESS algorithm.

# Quick start

```julia
using DRESS

result = fit(4, [0,1,2,0], [1,2,3,3])

result.edge_dress          # per-edge similarity
result.vertex_dress          # per-vertex aggregated similarity
result.iterations          # iterations until convergence
result.delta               # final max change
```
"""
module DRESS

using Libdl

export fit, delta_fit, nabla_fit, DRESSResult, DeltaDRESSResult, NablaDRESSResult, HistogramEntry, DressGraph, UNDIRECTED, DIRECTED, FORWARD, BACKWARD
export fit!, close!, delta_fit!, nabla_fit!

# ── variant constants ────────────────────────────────────────────────
const UNDIRECTED = Cint(0)
const DIRECTED   = Cint(1)
const FORWARD    = Cint(2)
const BACKWARD   = Cint(3)

# ── locate / build shared library ────────────────────────────────────

const _PKG_DIR  = dirname(@__DIR__)

# Two possible source layouts:
#   1. vendor/  inside the Julia package (standalone / published)
#   2. ../libdress/  sibling directory (monorepo development)
const _VENDOR_DIR = joinpath(_PKG_DIR, "vendor")
const _LIB_DIR    = normpath(joinpath(_PKG_DIR, "..", "libdress"))
const _SO_NAME  = "libdress" * (Sys.iswindows() ? ".dll" :
                                Sys.isapple()   ? ".dylib" : ".so")
const _SO_PATH  = joinpath(_PKG_DIR, _SO_NAME)

function _find_sources()
    # Try vendored sources first (standalone package)
    vendor_src = joinpath(_VENDOR_DIR, "src", "dress.c")
    if isfile(vendor_src)
        return joinpath(_VENDOR_DIR, "src"), joinpath(_VENDOR_DIR, "include")
    end
    # Fall back to monorepo layout
    mono_src = joinpath(_LIB_DIR, "src", "dress.c")
    if isfile(mono_src)
        return joinpath(_LIB_DIR, "src"), joinpath(_LIB_DIR, "include")
    end
    error("Cannot find DRESS C sources.  Expected either:\n" *
          "  $vendor_src  (standalone package)\n" *
          "  $mono_src  (monorepo)")
end

"""
    dress_build()

Compile the shared library from C sources into `julia/libdress.so`.
Called automatically on first use if the `.so` is missing.
"""
function dress_build()
    src_dir, inc_dir = _find_sources()
    src = joinpath(src_dir, "dress.c")
    src_delta = joinpath(src_dir, "delta_dress.c")
    src_impl = joinpath(src_dir, "delta_dress_impl.c")
    src_nabla = joinpath(src_dir, "nabla_dress.c")
    src_nabla_impl = joinpath(src_dir, "nabla_dress_impl.c")
    src_hist = joinpath(src_dir, "dress_histogram.c")
    src_omp = joinpath(src_dir, "omp", "dress_omp.c")
    src_omp_delta = joinpath(src_dir, "omp", "delta_dress_omp.c")
    src_omp_nabla = joinpath(src_dir, "omp", "nabla_dress_omp.c")
    isfile(src) || error("Cannot find dress.c at $src")
    isfile(src_delta) || error("Cannot find delta_dress.c at $src_delta")
    isfile(src_impl) || error("Cannot find delta_dress_impl.c at $src_impl")
    isfile(src_nabla) || error("Cannot find nabla_dress.c at $src_nabla")
    isfile(src_nabla_impl) || error("Cannot find nabla_dress_impl.c at $src_nabla_impl")
    isfile(src_hist) || error("Cannot find dress_histogram.c at $src_hist")
    cc = get(ENV, "CC", "gcc")
    omp_srcs = String[]
    if isfile(src_omp) && isfile(src_omp_delta) && isfile(src_omp_nabla)
        push!(omp_srcs, src_omp, src_omp_delta, src_omp_nabla)
    end
    cmd = `$cc -shared -fPIC -O3 -fopenmp -I$inc_dir -o $_SO_PATH $src $src_delta $src_impl $src_nabla $src_nabla_impl $src_hist $omp_srcs -lm`
    @info "Building DRESS shared library…" cmd
    run(cmd)
    @info "Built $_SO_PATH"
end

# ── library handle and function pointers ─────────────────────────────

const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_GET     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA   = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA_STRIDED = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA   = Ref{Ptr{Cvoid}}(C_NULL)

function _try_dlopen(name::String, local_path::String)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    isfile(local_path) || dress_build()
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_HANDLE[] != C_NULL && return
    _LIB_HANDLE[] = _try_dlopen(_SO_NAME, _SO_PATH)
    _FN_INIT[]    = dlsym(_LIB_HANDLE[], :dress_init_graph)
    _FN_FIT[]     = dlsym(_LIB_HANDLE[], :dress_fit)
    _FN_FREE[]    = dlsym(_LIB_HANDLE[], :dress_free_graph)
    _FN_GET[]     = dlsym(_LIB_HANDLE[], :dress_get)
    _FN_DELTA[]   = dlsym(_LIB_HANDLE[], :dress_delta_fit)
    _FN_DELTA_STRIDED[] = dlsym(_LIB_HANDLE[], :dress_delta_fit_strided)
    _FN_NABLA[]   = dlsym(_LIB_HANDLE[], :dress_nabla_fit)
end

# ── result type ──────────────────────────────────────────────────────

"""
    DRESSResult

Holds the output of `fit`.

Fields:
- `sources::Vector{Int32}`      – edge source vertices
- `targets::Vector{Int32}`      – edge target vertices
- `edge_weight::Vector{Float64}` – variant-specific edge weights
- `edge_dress::Vector{Float64}` – per-edge dress similarity
- `vertex_dress::Vector{Float64}` – per-vertex aggregated similarity
- `iterations::Int`             – iterations performed
- `delta::Float64`              – final max per-edge change
"""
struct DRESSResult
    sources     :: Vector{Int32}
    targets     :: Vector{Int32}
    edge_weight :: Vector{Float64}
    edge_dress  :: Vector{Float64}
    vertex_dress  :: Vector{Float64}
    iterations  :: Int
    delta       :: Float64
    vertex_weights :: Union{Vector{Float64}, Nothing}
end

struct HistogramEntry
    value :: Float64
    count :: Int64
end

Base.:(==)(a::HistogramEntry, b::HistogramEntry) =
    a.value == b.value && a.count == b.count

function Base.show(io::IO, entry::HistogramEntry)
    print(io, "HistogramEntry(value=$(entry.value), count=$(entry.count))")
end

function _copy_histogram(h_ptr::Ptr{HistogramEntry}, hsize::Integer)
    if h_ptr == C_NULL || hsize <= 0
        return HistogramEntry[]
    end
    copy(unsafe_wrap(Array, h_ptr, Int(hsize)))
end

function Base.show(io::IO, r::DRESSResult)
    print(io, "DRESSResult(E=$(length(r.sources)), " *
              "iterations=$(r.iterations), δ=$(r.delta))")
end

# ── core wrapper ─────────────────────────────────────────────────────

"""
    fit(N, sources, targets;
              weights=nothing, vertex_weights=nothing,
              variant=UNDIRECTED,
              max_iterations=100, epsilon=1e-6,
              precompute_intercepts=false) → DRESSResult

Run the DRESS iterative fitting algorithm.

# Arguments
- `N::Int`                – number of vertices (vertex ids must be in 0:N-1)
- `sources::Vector{Int}`  – edge source vertices (0-based)
- `targets::Vector{Int}`  – edge target vertices (0-based)

# Keyword arguments
- `weights`                – optional edge weights (Float64 vector, same length as sources)
- `vertex_weights`           – optional vertex weights (Float64 vector, length N)
- `variant`                – one of `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD`
- `max_iterations::Int`    – maximum fitting iterations (default 100)
- `epsilon::Float64`       – convergence threshold (default 1e-6)
- `precompute_intercepts`  – pre-compute neighbour intercepts (faster but more memory)
"""
function fit(N::Integer,
                   sources::AbstractVector{<:Integer},
                   targets::AbstractVector{<:Integer};
                   weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   vertex_weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   variant::Integer   = UNDIRECTED,
                   max_iterations::Integer = 100,
                   epsilon::Real      = 1e-6,
                   precompute_intercepts::Bool = false)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))
    if weights !== nothing
        length(weights) == E || throw(ArgumentError("weights must have the same length as sources"))
    end
    if vertex_weights !== nothing
        length(vertex_weights) == N || throw(ArgumentError("vertex_weights must have length N"))
    end

    _ensure_lib()

    # The C library takes ownership of U, V, W, NW. We must pass malloc'd copies.
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

    # dress_init_graph(N, E, U, V, W, NW, variant, precompute_intercepts) → ptr
    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), Ptr{Cdouble}(W_c), Ptr{Cdouble}(NW_c),
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("dress_init_graph returned NULL")

    # fit(g, max_iterations, epsilon, &iterations, &delta)
    iters_ref = Ref{Cint}(0)
    delta_ref = Ref{Cdouble}(0.0)
    ccall(_FN_FIT[], Cvoid,
          (Ptr{Cvoid}, Cint, Cdouble, Ptr{Cint}, Ptr{Cdouble}),
          g, Cint(max_iterations), Cdouble(epsilon),
          iters_ref, delta_ref)

    # Read struct fields via known offsets (LP64: Cint=4, Ptr=8).
    #
    # Field order (from dress.h):
    #   dress_variant_t variant;   // Cint   offset 0
    #   int N;                     // Cint   offset 4
    #   int E;                     // Cint   offset 8
    #   <pad 4 bytes>
    #   int *U;                    // Ptr    offset 16
    #   int *V;                    // Ptr    offset 24
    #   int *adj_offset;           // Ptr    offset 32
    #   int *adj_target;           // Ptr    offset 40
    #   int *adj_edge_idx;         // Ptr    offset 48
    #   int  max_degree;           // i32    offset 56 (+pad4)
    #   double *W;                 // Ptr    offset 64  (raw input weights)
    #   double *edge_weight;       // Ptr    offset 72
    #   double *edge_dress;        // Ptr    offset 80
    #   double *edge_dress_next;   // Ptr    offset 88
    #   double *vertex_dress;        // Ptr    offset 96
    #   double *NW;                // Ptr    offset 104

    edge_weight_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 72))
    edge_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 80))
    node_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 96))
    nw_ptr          = unsafe_load(Ptr{Ptr{Cdouble}}(g + 104))

    # Copy results into Julia-owned arrays before freeing the C struct
    ew = copy(unsafe_wrap(Array, edge_weight_ptr, E))
    ed = copy(unsafe_wrap(Array, edge_dress_ptr,  E))
    nd = copy(unsafe_wrap(Array, node_dress_ptr,  Cint(N)))
    src_out = copy(unsafe_wrap(Array, Ptr{Cint}(U_c), E))
    tgt_out = copy(unsafe_wrap(Array, Ptr{Cint}(V_c), E))

    nw_out = nothing
    if nw_ptr != C_NULL
        nw_out = copy(unsafe_wrap(Array, nw_ptr, Cint(N)))
    end

    # Free
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DRESSResult(src_out, tgt_out, ew, ed, nd,
                       Int(iters_ref[]), Float64(delta_ref[]), nw_out)
end

# ── persistent DressGraph object ─────────────────────────────────────

"""
    DressGraph

A persistent DRESS graph that supports repeated `fit!` and `get` calls.

```julia
g = DressGraph(4, [0,1,2,0], [1,2,3,3])
fit!(g)
d = get(g, 0, 2)           # virtual edge query
close!(g)                   # explicit cleanup (also runs at GC)
```
"""
mutable struct DressGraph
    ptr      :: Ptr{Cvoid}
    n        :: Int
    e        :: Int
    sources  :: Vector{Int32}
    targets  :: Vector{Int32}
end

"""
    DressGraph(N, sources, targets; weights=nothing, vertex_weights=nothing,
               variant=UNDIRECTED, precompute_intercepts=false)

Create a persistent DRESS graph object.
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

Fit the DRESS model on a persistent graph.
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

"""
    Base.get(g::DressGraph, u, v; max_iterations=100, epsilon=1e-6, edge_weight=1.0) → Float64

Query the DRESS value for an edge (existing or virtual).
"""
function Base.get(g::DressGraph, u::Integer, v::Integer;
                  max_iterations::Integer=100, epsilon::Real=1e-6,
                  edge_weight::Real=1.0)
    g.ptr == C_NULL && error("DressGraph already closed")
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

"""
    delta_fit!(g::DressGraph, k; max_iterations=100, epsilon=1e-6,
              n_samples=0, seed=0,
              keep_multisets=false, compute_histogram=true) → DeltaDRESSResult

Δ^k-DRESS on a persistent graph.
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
              n_samples=0, seed=0,
              keep_multisets=false, compute_histogram=true) → NablaDRESSResult

∇^k-DRESS on a persistent graph.
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
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint,
                   Ptr{Cint}, Cint,
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
    print(io, "DressGraph(N=$(g.n), E=$(g.e), $state)")
end

# ── delta result type ────────────────────────────────────────────────

"""
    DeltaDRESSResult

Holds the output of `delta_fit`.

Fields:
- `histogram::Vector{HistogramEntry}` – exact sparse histogram entries `(value, count)`
- `multisets::Union{Matrix{Float64}, Nothing}` – C(N,k) × E matrix of per-subgraph
  edge values (NaN = removed edge); `nothing` when `keep_multisets=false`
- `num_subgraphs::Int`                – C(N,k)
"""
struct DeltaDRESSResult
    histogram :: Vector{HistogramEntry}
    multisets :: Union{Matrix{Float64}, Nothing}
    num_subgraphs :: Int
end

function Base.show(io::IO, r::DeltaDRESSResult)
    total = sum((entry.count for entry in r.histogram); init=Int64(0))
    print(io, "DeltaDRESSResult(histogram_entries=$(length(r.histogram)), total_values=$total)")
end

# ── delta wrapper ────────────────────────────────────────────────────

"""
    delta_fit(N, sources, targets;
                    weights=nothing, vertex_weights=nothing,
                    k=0, variant=UNDIRECTED,
                    max_iterations=100, epsilon=1e-6,
                    precompute=false,
                    keep_multisets=false) → DeltaDRESSResult

Run Δ^k-DRESS: enumerate all C(N,k) vertex-deletion subsets, fit DRESS on
each subgraph, and return the pooled histogram.

# Arguments
- `N::Int`                – number of vertices (0-based ids)
- `sources::Vector{Int}`  – edge source vertices (0-based)
- `targets::Vector{Int}`  – edge target vertices (0-based)

# Keyword arguments
- `weights`              – optional edge weights (Float64 vector, same length as sources)
- `vertex_weights`         – optional vertex weights (Float64 vector, length N)
- `k::Int`               – deletion depth (default 0 = original graph)
- `variant`              – one of `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD`
- `max_iterations::Int`  – max DRESS iterations per subgraph (default 100)
- `epsilon::Float64`     – convergence tolerance and bin width (default 1e-6)
- `precompute::Bool`     – precompute intercepts in subgraphs (default false)
- `keep_multisets::Bool` – if true, return per-subgraph edge values in a
                           C(N,k) × E matrix (NaN = removed edge; default false)
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

    # The C library takes ownership of U, V via free().
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
        nw_arr = unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N)
        nw_arr .= Cdouble.(vertex_weights)
        Ptr{Cdouble}(nw_ptr)
    else
        Ptr{Cdouble}(C_NULL)
    end

    # dress_init_graph(N, E, U, V, W, NW, variant, precompute_intercepts) → ptr
    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c, NW_c,
              Cint(variant), Cint(precompute))

    g == C_NULL && error("dress_init_graph returned NULL")

    # dress_delta_fit_strided(g, k, iterations, epsilon, &hist_size, keep_multisets, &multisets, &num_subgraphs, offset, stride) → *dress_hist_pair_t
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

    # Extract multisets if requested
    ms_mat = nothing
    nsub = Int(nsub_ref[])
    if keep_multisets
        ms_p = ms_ref[]
        if ms_p != Ptr{Cdouble}(C_NULL) && nsub > 0
            # C stores row-major: ms_ptr[s * E + e]
            # Julia Matrix is column-major.  Read flat, then reshape + transpose.
            flat = copy(unsafe_wrap(Array, ms_p, nsub * E))
            ms_mat = permutedims(reshape(flat, E, nsub))
            Libc.free(ms_p)
        else
            ms_mat = Matrix{Float64}(undef, 0, 0)
        end
    end

    # Free the C histogram and graph
    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DeltaDRESSResult(histogram, ms_mat, nsub)
end
# ── nabla result type ────────────────────────────────────────────────────────

"""
    NablaDRESSResult

Holds the output of `nabla_fit`.

Fields:
- `histogram::Vector{HistogramEntry}` – exact sparse histogram entries `(value, count)`
- `multisets::Union{Matrix{Float64}, Nothing}` – P(N,k) × E matrix of per-tuple
  edge values; `nothing` when `keep_multisets=false`
- `num_tuples::Int`                   – P(N,k)
"""
struct NablaDRESSResult
    histogram :: Vector{HistogramEntry}
    multisets :: Union{Matrix{Float64}, Nothing}
    num_tuples :: Int
end

function Base.show(io::IO, r::NablaDRESSResult)
    total = sum((entry.count for entry in r.histogram); init=Int64(0))
    print(io, "NablaDRESSResult(histogram_entries=$(length(r.histogram)), total_values=$total)")
end

# ── nabla wrapper ────────────────────────────────────────────────────────────

"""
    nabla_fit(N, sources, targets; k=0, ...) → NablaDRESSResult

Compute the ∇^k-DRESS histogram by enumerating all P(N,k) ordered k-tuples,
marking each with generic injective vertex weights, and pooling the converged
edge values.
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
        nw_arr = unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N)
        nw_arr .= Cdouble.(vertex_weights)
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
# ── CUDA submodule ───────────────────────────────────────────────────

include("CUDA.jl")

# ── OMP submodule ────────────────────────────────────────────────────

include("OMP.jl")

# ── MPI submodule ────────────────────────────────────────────────────

include("MPI.jl")
const MPI = MPI_DRESS
export MPI

end # module DRESS
