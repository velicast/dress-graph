"""
    DRESS.MPI.CUDA

MPI-distributed Δ^k-DRESS with GPU acceleration.

```julia
using MPI; MPI.Init()
using DRESS.MPI.CUDA
result = delta_dress_fit(4, [0,1,2,0,1,2], [1,2,0,3,3,3]; k=2)
MPI.Finalize()
```
"""
module CUDA

using Libdl

using ...DRESS: DeltaDRESSResult, UNDIRECTED, DIRECTED, FORWARD, BACKWARD
export delta_dress_fit, DeltaDRESSResult, UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── locate shared libraries ──────────────────────────────────────────

const _PKG_DIR    = dirname(dirname(@__DIR__))   # julia/
const _SO_NAME    = "libdress" * (Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so")
const _SO_PATH    = joinpath(_PKG_DIR, _SO_NAME)
const _CUDA_SO    = "libdress_cuda" * (Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so")
const _CUDA_PATH  = joinpath(_PKG_DIR, _CUDA_SO)

const _LIB_CPU         = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_CUDA        = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT         = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE         = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_CUDA_FCOMM = Ref{Ptr{Cvoid}}(C_NULL)

function _try_dlopen(name::String, local_path::String; build::Bool=false)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    build && !isfile(local_path) && DRESS.dress_build()
    isfile(local_path) || error("Library $name not found. Install it or set LD_LIBRARY_PATH.")
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_CPU[] != C_NULL && return

    _LIB_CPU[] = _try_dlopen(_SO_NAME, _SO_PATH; build=true)
    _FN_INIT[] = dlsym(_LIB_CPU[], :init_dress_graph)
    _FN_FREE[] = dlsym(_LIB_CPU[], :free_dress_graph)

    _LIB_CUDA[] = _try_dlopen(_CUDA_SO, _CUDA_PATH)

    sym = dlsym_e(_LIB_CUDA[], :delta_dress_fit_mpi_cuda_fcomm)
    sym == C_NULL && error(
        "delta_dress_fit_mpi_cuda_fcomm not found in libdress_cuda. " *
        "Ensure libdress_cuda.so was built with MPI support."
    )
    _FN_MPI_CUDA_FCOMM[] = sym
end

# ── delta_dress_fit (MPI + CUDA) ─────────────────────────────────────

"""
    delta_dress_fit(N, sources, targets; k=0, comm=nothing, kwargs...) → DeltaDRESSResult

MPI-distributed Δ^k-DRESS (CUDA backend).  Each rank runs GPU-accelerated
DRESS on its stride of subgraphs.

Same keyword arguments as `DRESS.MPI.delta_dress_fit`.
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
                         comm               = nothing)

    if comm === nothing
        MPI_jl = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        comm = getfield(MPI_jl, :COMM_WORLD)
    end
    comm_f = ccall((:MPI_Comm_c2f, "libmpi"), Cint, (Ptr{Cvoid},), comm.val)

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

    h_ptr = ccall(_FN_MPI_CUDA_FCOMM[], Ptr{Int64},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  hsize_ref,
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref,
                  comm_f)

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
