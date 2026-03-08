"""
    DRESS.CUDA

GPU-accelerated DRESS — same API as the CPU functions, different import.

# Usage (import-based switching)

```julia
# CPU
using DRESS
result = dress_fit(4, [0,1,2,0], [1,2,3,3])

# CUDA — same call, different import
using DRESS.CUDA
result = dress_fit(4, [0,1,2,0], [1,2,3,3])
```
"""
module CUDA

using Libdl

# Re-export types and constants from the parent module.
using ..DRESS: DRESSResult, DeltaDRESSResult, UNDIRECTED, DIRECTED, FORWARD, BACKWARD
export dress_fit, delta_dress_fit, DRESSResult, DeltaDRESSResult,
       UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── locate CUDA shared library ───────────────────────────────────────

const _PKG_DIR  = dirname(@__DIR__)
const _LIB_DIR  = normpath(joinpath(_PKG_DIR, "..", "libdress"))
const _CUDA_SO  = "libdress_cuda" * (Sys.iswindows() ? ".dll" :
                                      Sys.isapple()   ? ".dylib" : ".so")
const _CUDA_PATH = joinpath(_PKG_DIR, _CUDA_SO)

# Also need the CPU library for init_dress_graph / free_dress_graph.
const _CPU_SO   = "libdress" * (Sys.iswindows() ? ".dll" :
                                 Sys.isapple()   ? ".dylib" : ".so")
const _CPU_PATH = joinpath(_PKG_DIR, _CPU_SO)

# ── library handles and function pointers ────────────────────────────

const _LIB_CPU    = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_CUDA   = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE    = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_DELTA   = Ref{Ptr{Cvoid}}(C_NULL)

function _ensure_lib()
    _LIB_CUDA[] != C_NULL && return

    # Load CPU library (init/free live there)
    if _LIB_CPU[] == C_NULL
        isfile(_CPU_PATH) || DRESS.dress_build()
        _LIB_CPU[] = dlopen(_CPU_PATH)
    end
    _FN_INIT[] = dlsym(_LIB_CPU[], :init_dress_graph)
    _FN_FREE[] = dlsym(_LIB_CPU[], :free_dress_graph)

    # Load CUDA library
    isfile(_CUDA_PATH) || error("CUDA library not found at $_CUDA_PATH. Build it first.")
    _LIB_CUDA[] = dlopen(_CUDA_PATH)
    _FN_FIT[]   = dlsym(_LIB_CUDA[], :dress_fit_cuda)
    _FN_DELTA[] = dlsym(_LIB_CUDA[], :delta_dress_fit_cuda)
end

# ── dress_fit (CUDA) ─────────────────────────────────────────────────

"""
    dress_fit(N, sources, targets; kwargs...) → DRESSResult

GPU-accelerated DRESS fitting.  Same signature as `DRESS.dress_fit()`.
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

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), Ptr{Cdouble}(W_c),
              Cint(variant), Cint(precompute_intercepts))

    g == C_NULL && error("init_dress_graph returned NULL")

    iters_ref = Ref{Cint}(0)
    delta_ref = Ref{Cdouble}(0.0)
    ccall(_FN_FIT[], Cvoid,
          (Ptr{Cvoid}, Cint, Cdouble, Ptr{Cint}, Ptr{Cdouble}),
          g, Cint(max_iterations), Cdouble(epsilon),
          iters_ref, delta_ref)

    # Read struct fields (LP64 offsets — same as CPU)
    edge_weight_ptr = unsafe_load(Ptr{Ptr{Cdouble}}(g + 64))
    edge_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 72))
    node_dress_ptr  = unsafe_load(Ptr{Ptr{Cdouble}}(g + 88))

    ew = copy(unsafe_wrap(Array, edge_weight_ptr, E))
    ed = copy(unsafe_wrap(Array, edge_dress_ptr,  E))
    nd = copy(unsafe_wrap(Array, node_dress_ptr,  Cint(N)))
    src_out = copy(unsafe_wrap(Array, Ptr{Cint}(U_c), E))
    tgt_out = copy(unsafe_wrap(Array, Ptr{Cint}(V_c), E))

    ccall(_FN_FREE[], Cvoid, (Ptr{Cvoid},), g)

    return DRESSResult(src_out, tgt_out, ew, ed, nd,
                       Int(iters_ref[]), Float64(delta_ref[]))
end

# ── delta_dress_fit (CUDA) ───────────────────────────────────────────

"""
    delta_dress_fit(N, sources, targets; kwargs...) → DeltaDRESSResult

GPU-accelerated Δ^k-DRESS.  Same signature as `DRESS.delta_dress_fit()`.
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
                         keep_multisets::Bool = false)

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

    g = ccall(_FN_INIT[], Ptr{Cvoid},
              (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Cint, Cint),
              Cint(N), Cint(E),
              Ptr{Cint}(U_c), Ptr{Cint}(V_c), W_c,
              Cint(variant), Cint(precompute))

    g == C_NULL && error("init_dress_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)
    h_ptr = ccall(_FN_DELTA[], Ptr{Int64},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cint}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Int64}),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  hsize_ref,
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref)

    hsize = Int(hsize_ref[])
    histogram = if h_ptr != C_NULL && hsize > 0
        copy(unsafe_wrap(Array, h_ptr, hsize))
    else
        Int64[]
    end

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

    return DeltaDRESSResult(histogram, hsize, ms_mat, nsub)
end

end # module CUDA
