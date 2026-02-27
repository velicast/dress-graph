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

export dress_fit, delta_dress_fit, nabla_dress_fit, UNDIRECTED, DIRECTED, FORWARD, BACKWARD

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
    src_nabla = joinpath(_LIB_DIR, "src", "nabla_dress.c")
    inc = joinpath(_LIB_DIR, "include")
    isfile(src) || error("Cannot find dress.c at $src")
    isfile(src_delta) || error("Cannot find delta_dress.c at $src_delta")
    isfile(src_nabla) || error("Cannot find nabla_dress.c at $src_nabla")
    cc = get(ENV, "CC", "gcc")
    cmd = `$cc -shared -fPIC -O3 -fopenmp -I$inc -o $_SO_PATH $src $src_delta $src_nabla -lm`
    @info "Building DRESS shared library…" cmd
    run(cmd)
    @info "Built $_SO_PATH"
end

# ── library handle and function pointers ─────────────────────────────

const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA   = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA   = Ref{Ptr{Cvoid}}(C_NULL)

function _ensure_lib()
    _LIB_HANDLE[] != C_NULL && return
    isfile(_SO_PATH) || dress_build()
    _LIB_HANDLE[] = dlopen(_SO_PATH)
    _FN_INIT[]    = dlsym(_LIB_HANDLE[], :init_dress_graph)
    _FN_FIT[]     = dlsym(_LIB_HANDLE[], :fit)
    _FN_FREE[]    = dlsym(_LIB_HANDLE[], :free_dress_graph)
    _FN_DELTA[]   = dlsym(_LIB_HANDLE[], :delta_fit)
    _FN_NABLA[]   = dlsym(_LIB_HANDLE[], :nabla_fit)
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
              precompute_intercepts=true) → DRESSResult

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
                   precompute_intercepts::Bool = true)

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
    #   double *edge_weight;       // Ptr    offset 56
    #   double *edge_dress;        // Ptr    offset 64
    #   double *edge_dress_next;   // Ptr    offset 72
    #   double *node_dress;        // Ptr    offset 80

    edge_weight_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 56))
    edge_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 64))
    node_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 80))

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

# ── delta result type ────────────────────────────────────────────────

"""
    DeltaDRESSResult

Holds the output of `delta_dress_fit`.

Fields:
- `histogram::Vector{Int64}`    – bin-count vector
- `hist_size::Int`              – number of bins (floor(2/ε) + 1)
"""
struct DeltaDRESSResult
    histogram :: Vector{Int64}
    hist_size :: Int
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
                    precompute=false) → DeltaDRESSResult

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
"""
function delta_dress_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         k::Integer         = 0,
                         variant::Integer   = UNDIRECTED,
                         max_iterations::Integer = 100,
                         epsilon::Real      = 1e-6,
                         precompute::Bool   = false)

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

    # delta_fit(g, k, iterations, epsilon, &hist_size, keep_multisets, &multisets, &num_subgraphs) → *int64
    hsize_ref = Ref{Cint}(0)
    h_ptr = ccall(_FN_DELTA[], Ptr{Int64},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  hsize_ref, Cint(0), Ptr{Ptr{Cdouble}}(C_NULL), Ptr{Int64}(C_NULL))

    hsize = Int(hsize_ref[])
    histogram = if h_ptr != C_NULL && hsize > 0
        copy(unsafe_wrap(Array, h_ptr, hsize))
    else
        Int64[]
    end

    # Free the C histogram and graph
    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DeltaDRESSResult(histogram, hsize)
end

# ── nabla result type ────────────────────────────────────────────────

"""
    NablaDRESSResult

Holds the output of `nabla_dress_fit`.

Fields:
- `histogram::Vector{Int64}`    – bin-count vector
- `hist_size::Int`              – number of bins (floor(2/ε) + 1)
"""
struct NablaDRESSResult
    histogram :: Vector{Int64}
    hist_size :: Int
end

function Base.show(io::IO, r::NablaDRESSResult)
    total = sum(r.histogram)
    print(io, "NablaDRESSResult(hist_size=$(r.hist_size), total_values=$total)")
end

# ── nabla wrapper ────────────────────────────────────────────────────

"""
    nabla_dress_fit(N, sources, targets;
                    k=0, nabla_weight=2.0, variant=UNDIRECTED,
                    max_iterations=100, epsilon=1e-6,
                    precompute=false) → NablaDRESSResult

Run ∇^k-DRESS: enumerate all C(N,k) vertex subsets, individualize
(mark) each subset by multiplying incident edge weights by
`nabla_weight`, fit DRESS, and return the pooled histogram.

# Arguments
- `N::Int`                – number of vertices (0-based ids)
- `sources::Vector{Int}`  – edge source vertices (0-based)
- `targets::Vector{Int}`  – edge target vertices (0-based)

# Keyword arguments
- `k::Int`               – individualization depth (default 0)
- `nabla_weight::Float64` – multiplicative weight factor (default 2.0)
- `variant`              – one of `UNDIRECTED`, `DIRECTED`, `FORWARD`, `BACKWARD`
- `max_iterations::Int`  – max DRESS iterations per round (default 100)
- `epsilon::Float64`     – convergence tolerance and bin width (default 1e-6)
- `precompute::Bool`     – precompute intercepts (default false)
"""
function nabla_dress_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         k::Integer         = 0,
                         nabla_weight::Real  = 2.0,
                         variant::Integer   = UNDIRECTED,
                         max_iterations::Integer = 100,
                         epsilon::Real      = 1e-6,
                         precompute::Bool   = false)

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

    # nabla_fit(g, k, iterations, epsilon, nabla_weight, &hist_size,
    #           keep_multisets, &multisets, &num_subsets) → *int64
    hsize_ref = Ref{Cint}(0)
    h_ptr = ccall(_FN_NABLA[], Ptr{Int64},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cdouble, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon), Cdouble(nabla_weight),
                  hsize_ref, Cint(0), Ptr{Ptr{Cdouble}}(C_NULL), Ptr{Int64}(C_NULL))

    hsize = Int(hsize_ref[])
    histogram = if h_ptr != C_NULL && hsize > 0
        copy(unsafe_wrap(Array, h_ptr, hsize))
    else
        Int64[]
    end

    # Free the C histogram and graph
    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end
    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return NablaDRESSResult(histogram, hsize)
end

end # module DRESS
