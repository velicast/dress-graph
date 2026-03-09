"""
    DRESS

Julia package for the DRESS algorithm.

# Quick start

```julia
using DRESS

result = dress_fit(4, [0,1,2,0], [1,2,3,3])

result.edge_dress          # per-edge similarity
result.node_dress          # per-node aggregated similarity
result.iterations          # iterations until convergence
result.delta               # final max change
```
"""
module DRESS

using Libdl

export dress_fit, delta_dress_fit, DRESSResult, DeltaDRESSResult, DressGraph, UNDIRECTED, DIRECTED, FORWARD, BACKWARD
export fit!, close!

# ── variant constants ────────────────────────────────────────────────
const UNDIRECTED = Cint(0)
const DIRECTED   = Cint(1)
const FORWARD    = Cint(2)
const BACKWARD   = Cint(3)

# ── locate / build shared library ────────────────────────────────────

const _PKG_DIR  = dirname(@__DIR__)
const _LIB_DIR  = normpath(joinpath(_PKG_DIR, "..", "libdress"))
const _SO_NAME  = "libdress" * (Sys.iswindows() ? ".dll" :
                                Sys.isapple()   ? ".dylib" : ".so")
const _SO_PATH  = joinpath(_PKG_DIR, _SO_NAME)

"""
    dress_build()

Compile the shared library from `libdress/src/dress.c` into `julia/libdress.so`.
Called automatically on first use if the `.so` is missing.
"""
function dress_build()
    src = joinpath(_LIB_DIR, "src", "dress.c")
    src_delta = joinpath(_LIB_DIR, "src", "delta_dress.c")
    inc = joinpath(_LIB_DIR, "include")
    isfile(src) || error("Cannot find dress.c at $src")
    isfile(src_delta) || error("Cannot find delta_dress.c at $src_delta")
    cc = get(ENV, "CC", "gcc")
    cmd = `$cc -shared -fPIC -O3 -fopenmp -I$inc -o $_SO_PATH $src $src_delta -lm`
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

function _try_dlopen(name::String, local_path::String)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    isfile(local_path) || dress_build()
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_HANDLE[] != C_NULL && return
    _LIB_HANDLE[] = _try_dlopen(_SO_NAME, _SO_PATH)
    _FN_INIT[]    = dlsym(_LIB_HANDLE[], :init_dress_graph)
    _FN_FIT[]     = dlsym(_LIB_HANDLE[], :dress_fit)
    _FN_FREE[]    = dlsym(_LIB_HANDLE[], :free_dress_graph)
    _FN_GET[]     = dlsym(_LIB_HANDLE[], :dress_get)
    _FN_DELTA[]   = dlsym(_LIB_HANDLE[], :delta_dress_fit)
    _FN_DELTA_STRIDED[] = dlsym(_LIB_HANDLE[], :delta_dress_fit_strided)
end

# ── result type ──────────────────────────────────────────────────────

"""
    DRESSResult

Holds the output of `dress_fit`.

Fields:
- `sources::Vector{Int32}`      – edge source vertices
- `targets::Vector{Int32}`      – edge target vertices
- `edge_weight::Vector{Float64}` – variant-specific edge weights
- `edge_dress::Vector{Float64}` – per-edge dress similarity
- `node_dress::Vector{Float64}` – per-node aggregated similarity
- `iterations::Int`             – iterations performed
- `delta::Float64`              – final max per-edge change
"""
struct DRESSResult
    sources     :: Vector{Int32}
    targets     :: Vector{Int32}
    edge_weight :: Vector{Float64}
    edge_dress  :: Vector{Float64}
    node_dress  :: Vector{Float64}
    iterations  :: Int
    delta       :: Float64
end

function Base.show(io::IO, r::DRESSResult)
    print(io, "DRESSResult(E=$(length(r.sources)), " *
              "iterations=$(r.iterations), δ=$(r.delta))")
end

# ── core wrapper ─────────────────────────────────────────────────────

"""
    dress_fit(N, sources, targets;
              weights=nothing, variant=UNDIRECTED,
              max_iterations=100, epsilon=1e-6,
              precompute_intercepts=false) → DRESSResult

Run the DRESS iterative fitting algorithm.

# Arguments
- `N::Int`                – number of vertices (vertex ids must be in 0:N-1)
- `sources::Vector{Int}`  – edge source vertices (0-based)
- `targets::Vector{Int}`  – edge target vertices (0-based)

# Keyword arguments
- `weights`                – optional edge weights (Float64 vector, same length as sources)
- `variant`                – one of `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD`
- `max_iterations::Int`    – maximum fitting iterations (default 100)
- `epsilon::Float64`       – convergence threshold (default 1e-6)
- `precompute_intercepts`  – pre-compute neighbour intercepts (faster but more memory)
"""
function dress_fit(N::Integer,
                   sources::AbstractVector{<:Integer},
                   targets::AbstractVector{<:Integer};
                   weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   variant::Integer   = UNDIRECTED,
                   max_iterations::Integer = 100,
                   epsilon::Real      = 1e-6,
                   precompute_intercepts::Bool = false)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))
    if weights !== nothing
        length(weights) == E || throw(ArgumentError("weights must have the same length as sources"))
    end

    _ensure_lib()

    # The C library takes ownership of U, V, W. We must pass malloc'd copies.
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

    # init_dress_graph(N, E, U, V, W, variant, precompute_intercepts) → ptr
    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), Ptr{Cdouble}(W_c),
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("init_dress_graph returned NULL")

    # dress_fit(g, max_iterations, epsilon, &iterations, &delta)
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
    #   double *W;                 // Ptr    offset 56  (raw input weights)
    #   double *edge_weight;       // Ptr    offset 64
    #   double *edge_dress;        // Ptr    offset 72
    #   double *edge_dress_next;   // Ptr    offset 80
    #   double *node_dress;        // Ptr    offset 88

    edge_weight_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 64))
    edge_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 72))
    node_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 88))

    # Copy results into Julia-owned arrays before freeing the C struct
    ew = copy(unsafe_wrap(Array, edge_weight_ptr, E))
    ed = copy(unsafe_wrap(Array, edge_dress_ptr,  E))
    nd = copy(unsafe_wrap(Array, node_dress_ptr,  Cint(N)))
    src_out = copy(unsafe_wrap(Array, Ptr{Cint}(U_c), E))
    tgt_out = copy(unsafe_wrap(Array, Ptr{Cint}(V_c), E))

    # Free
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DRESSResult(src_out, tgt_out, ew, ed, nd,
                       Int(iters_ref[]), Float64(delta_ref[]))
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
    DressGraph(N, sources, targets; weights=nothing,
               variant=UNDIRECTED, precompute_intercepts=false)

Create a persistent DRESS graph object.
"""
function DressGraph(N::Integer,
                    sources::AbstractVector{<:Integer},
                    targets::AbstractVector{<:Integer};
                    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                    variant::Integer   = UNDIRECTED,
                    precompute_intercepts::Bool = false)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources/targets length mismatch"))
    if weights !== nothing
        length(weights) == E || throw(ArgumentError("weights length mismatch"))
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

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c,
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("init_dress_graph returned NULL")

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
    ew = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 64)), g.e))
    ed = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 72)), g.e))
    nd = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g.ptr + 88)), g.n))
    DRESSResult(copy(g.sources), copy(g.targets), ew, ed, nd, 0, 0.0)
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

Holds the output of `delta_dress_fit`.

Fields:
- `histogram::Vector{Int64}`          – bin-count vector
- `hist_size::Int`                    – number of bins (floor(dmax/ε) + 1; dmax = 2 unweighted)
- `multisets::Union{Matrix{Float64}, Nothing}` – C(N,k) × E matrix of per-subgraph
  edge values (NaN = removed edge); `nothing` when `keep_multisets=false`
- `num_subgraphs::Int`                – C(N,k)
"""
struct DeltaDRESSResult
    histogram :: Vector{Int64}
    hist_size :: Int
    multisets :: Union{Matrix{Float64}, Nothing}
    num_subgraphs :: Int
end

function Base.show(io::IO, r::DeltaDRESSResult)
    total = sum(r.histogram)
    print(io, "DeltaDRESSResult(hist_size=$(r.hist_size), total_values=$total)")
end

# ── delta wrapper ────────────────────────────────────────────────────

"""
    delta_dress_fit(N, sources, targets;
                    k=0, variant=UNDIRECTED,
                    max_iterations=100, epsilon=1e-6,
                    precompute=false,
                    keep_multisets=false,
                    offset=0, stride=1) → DeltaDRESSResult

Run Δ^k-DRESS: enumerate all C(N,k) node-deletion subsets, fit DRESS on
each subgraph, and return the pooled histogram.

# Arguments
- `N::Int`                – number of vertices (0-based ids)
- `sources::Vector{Int}`  – edge source vertices (0-based)
- `targets::Vector{Int}`  – edge target vertices (0-based)

# Keyword arguments
- `k::Int`               – deletion depth (default 0 = original graph)
- `variant`              – one of `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD`
- `max_iterations::Int`  – max DRESS iterations per subgraph (default 100)
- `epsilon::Float64`     – convergence tolerance and bin width (default 1e-6)
- `precompute::Bool`     – precompute intercepts in subgraphs (default false)
- `keep_multisets::Bool` – if true, return per-subgraph edge values in a
                           C(N,k) × E matrix (NaN = removed edge; default false)
- `offset::Int`          – process only subgraphs where index % stride == offset (default 0)
- `stride::Int`          – total number of strides (default 1 = process all)
"""
function delta_dress_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         k::Integer         = 0,
                         variant::Integer   = UNDIRECTED,
                         max_iterations::Integer = 100,
                         epsilon::Real      = 1e-6,
                         precompute::Bool   = false,
                         keep_multisets::Bool = false,
                         offset::Integer    = 0,
                         stride::Integer    = 1)

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

    # init_dress_graph(N, E, U, V, W, variant, precompute_intercepts) → ptr
    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c,
              Cint(variant), Cint(precompute))

    g == C_NULL && error("init_dress_graph returned NULL")

    # delta_dress_fit_strided(g, k, iterations, epsilon, &hist_size, keep_multisets, &multisets, &num_subgraphs, offset, stride) → *int64
    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)
    h_ptr = ccall(_FN_DELTA_STRIDED[], Ptr{Int64},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  hsize_ref,
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref,
                  Cint(offset), Cint(stride))

    hsize = Int(hsize_ref[])
    histogram = if h_ptr != C_NULL && hsize > 0
        copy(unsafe_wrap(Array, h_ptr, hsize))
    else
        Int64[]
    end

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

    return DeltaDRESSResult(histogram, hsize, ms_mat, nsub)
end

# ── CUDA submodule ───────────────────────────────────────────────────

include("CUDA.jl")

# ── MPI submodule ────────────────────────────────────────────────────

include("MPI.jl")
const MPI = MPI_DRESS
export MPI

end # module DRESS
