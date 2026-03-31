"""
    DRESS.OMP

OpenMP-parallel DRESS — same API as the CPU functions, different import.

# Usage (import-based switching)

```julia
# CPU
using DRESS
result = fit(4, [0,1,2,0], [1,2,3,3])

# OpenMP — same call, different import
using DRESS.OMP
result = fit(4, [0,1,2,0], [1,2,3,3])
```

`fit` parallelises edges within each iteration.
`delta_fit` parallelises the outer subgraph loop.
"""
module OMP

using Libdl

using ..DRESS: DRESSResult, DeltaDRESSResult, NablaDRESSResult, HistogramEntry,
               UNDIRECTED, DIRECTED, FORWARD, BACKWARD,
               _copy_histogram, _ensure_lib as _cpu_ensure_lib,
               _SO_NAME, _SO_PATH, _LIB_HANDLE as _CPU_LIB

export fit, delta_fit, nabla_fit, DRESSResult, DeltaDRESSResult, NablaDRESSResult,
       UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── function pointers ────────────────────────────────────────────────

const _FN_INIT    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA_STRIDED = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA = Ref{Ptr{Cvoid}}(C_NULL)

function _ensure_lib()
    _FN_FIT[] != C_NULL && return
    _cpu_ensure_lib()  # ensure libdress.so is loaded
    lib = _CPU_LIB[]
    _FN_INIT[]    = dlsym(lib, :dress_init_graph)
    _FN_FIT[]     = dlsym(lib, :dress_fit_omp)
    _FN_FREE[]    = dlsym(lib, :dress_free_graph)
    _FN_DELTA_STRIDED[] = dlsym(lib, :dress_delta_fit_omp_strided)
    _FN_NABLA[] = dlsym(lib, :dress_nabla_fit_omp)
end

# ── fit (OMP) ─────────────────────────────────────────────────

"""
    fit(N, sources, targets; kwargs...) → DRESSResult

OpenMP-parallel DRESS fitting (edge-parallel).  Same signature as `DRESS.fit()`.
"""
function fit(N::Integer,
                   sources::AbstractVector{<:Integer},
                   targets::AbstractVector{<:Integer};
                   weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   node_weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   variant::Integer   = UNDIRECTED,
                   max_iterations::Integer = 100,
                   epsilon::Real      = 1e-6,
                   precompute_intercepts::Bool = false)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))
    weights !== nothing && length(weights) != E && throw(ArgumentError("weights length mismatch"))
    node_weights !== nothing && length(node_weights) != N && throw(ArgumentError("node_weights length mismatch"))

    _ensure_lib()

    U_c = Libc.malloc(E * sizeof(Cint))
    V_c = Libc.malloc(E * sizeof(Cint))
    unsafe_wrap(Array, Ptr{Cint}(U_c), E) .= Cint.(sources)
    unsafe_wrap(Array, Ptr{Cint}(V_c), E) .= Cint.(targets)

    W_c = C_NULL
    if weights !== nothing
        W_c = Libc.malloc(E * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(W_c), E) .= Cdouble.(weights)
    end

    NW_c = C_NULL
    if node_weights !== nothing
        NW_c = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(NW_c), N) .= Cdouble.(node_weights)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E), Ptr{Cint}(U_c), Ptr{Cint}(V_c),
              Ptr{Cdouble}(W_c), Ptr{Cdouble}(NW_c),
              Cint(variant), Cint(precompute_intercepts))
    g == C_NULL && error("dress_init_graph returned NULL")

    iters_ref = Ref{Cint}(0)
    delta_ref = Ref{Cdouble}(0.0)
    ccall(_FN_FIT[], Cvoid,
          (Ptr{Cvoid}, Cint, Cdouble, Ptr{Cint}, Ptr{Cdouble}),
          g, Cint(max_iterations), Cdouble(epsilon), iters_ref, delta_ref)

    ew = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g + 72)), E))
    ed = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g + 80)), E))
    nd = copy(unsafe_wrap(Array, unsafe_load(Ptr{Ptr{Cdouble}}(g + 96)), Cint(N)))
    nw_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 104))
    nw_out = nw_ptr != C_NULL ? copy(unsafe_wrap(Array, nw_ptr, Cint(N))) : nothing

    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DRESSResult(Cint.(sources), Cint.(targets), ew, ed, nd,
                       Int(iters_ref[]), Float64(delta_ref[]), nw_out)
end

# ── delta_fit (OMP) ────────────────────────────────────────────

"""
    delta_fit(N, sources, targets; kwargs...) → DeltaDRESSResult

OpenMP-parallel Δ^k-DRESS (subgraph-parallel).  Same signature as `DRESS.delta_fit()`.
"""
function delta_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         node_weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
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
    unsafe_wrap(Array, Ptr{Cint}(U_c), E) .= Cint.(sources)
    unsafe_wrap(Array, Ptr{Cint}(V_c), E) .= Cint.(targets)

    W_c = C_NULL
    if weights !== nothing
        W_c = Libc.malloc(E * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(W_c), E) .= Cdouble.(weights)
    end

    NW_c = C_NULL
    if node_weights !== nothing
        NW_c = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(NW_c), N) .= Cdouble.(node_weights)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E), Ptr{Cint}(U_c), Ptr{Cint}(V_c),
              Ptr{Cdouble}(W_c), Ptr{Cdouble}(NW_c),
              Cint(variant), Cint(precompute))
    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref = Ref{Ptr{Cdouble}}(C_NULL)
    nsub_ref = Ref{Int64}(0)

    h_ptr = ccall(_FN_DELTA_STRIDED[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint,
                   Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  ms_ref, nsub_ref,
                  Cint(0), Cint(1))

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    multisets = nothing
    ns = Int(nsub_ref[])
    if keep_multisets && ms_ref[] != C_NULL && ns > 0
        total = ns * E
        multisets = copy(unsafe_wrap(Array, ms_ref[], total))
        multisets = reshape(multisets, E, ns)'  # row-major → Julia column-major
        Libc.free(ms_ref[])
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end

    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DeltaDRESSResult(histogram, multisets, ns)
end

# ── nabla_fit (OMP) ────────────────────────────────────────────

"""
    nabla_fit(N, sources, targets; kwargs...) → NablaDRESSResult

OpenMP-parallel ∇^k-DRESS (tuple-parallel).  Same signature as `DRESS.nabla_fit()`.
"""
function nabla_fit(N::Integer,
                         sources::AbstractVector{<:Integer},
                         targets::AbstractVector{<:Integer};
                         weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
                         node_weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
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
    unsafe_wrap(Array, Ptr{Cint}(U_c), E) .= Cint.(sources)
    unsafe_wrap(Array, Ptr{Cint}(V_c), E) .= Cint.(targets)

    W_c = C_NULL
    if weights !== nothing
        W_c = Libc.malloc(E * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(W_c), E) .= Cdouble.(weights)
    end

    NW_c = C_NULL
    if node_weights !== nothing
        NW_c = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(NW_c), N) .= Cdouble.(node_weights)
    end

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E), Ptr{Cint}(U_c), Ptr{Cint}(V_c),
              Ptr{Cdouble}(W_c), Ptr{Cdouble}(NW_c),
              Cint(variant), Cint(precompute))
    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref = Ref{Ptr{Cdouble}}(C_NULL)
    ntup_ref = Ref{Int64}(0)

    h_ptr = ccall(_FN_NABLA[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint,
                   Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  ms_ref, ntup_ref)

    hsize = Int(hsize_ref[])
    histogram = _copy_histogram(h_ptr, hsize)

    multisets = nothing
    ntup = Int(ntup_ref[])
    if keep_multisets && ms_ref[] != C_NULL && ntup > 0
        total = ntup * E
        multisets = copy(unsafe_wrap(Array, ms_ref[], total))
        multisets = reshape(multisets, E, ntup)'
        Libc.free(ms_ref[])
    end

    if h_ptr != C_NULL
        Libc.free(h_ptr)
    end

    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return NablaDRESSResult(histogram, multisets, ntup)
end

end # module OMP
