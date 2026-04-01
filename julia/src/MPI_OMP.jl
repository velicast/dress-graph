"""
    DRESS.MPI.OMP

MPI-distributed Δ^k-DRESS with OpenMP parallelism within each rank.

```julia
using MPI; MPI.Init()
using DRESS.MPI.OMP
result = delta_fit(4, [0,1,2,0,1,2], [1,2,0,3,3,3]; k=2)
MPI.Finalize()
```

MPI distributes subgraphs across ranks; within each rank, OpenMP threads
further parallelise the subgraph slice.
"""
module OMP

using Libdl

using ...DRESS: DRESSResult, DeltaDRESSResult, NablaDRESSResult, HistogramEntry, UNDIRECTED, DIRECTED, FORWARD, BACKWARD, _copy_histogram
import ...DRESS: dress_build as _cpu_build
export delta_fit, nabla_fit, DeltaDRESSResult, NablaDRESSResult, UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── locate shared library ────────────────────────────────────────────

const _PKG_DIR    = dirname(dirname(@__DIR__))   # julia/
const _SO_NAME    = "libdress" * (Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so")
const _SO_PATH    = joinpath(_PKG_DIR, _SO_NAME)

const _LIB_CPU         = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT         = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE         = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_OMP_FCOMM = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA_MPI_OMP_FCOMM = Ref{Ptr{Cvoid}}(C_NULL)

function _try_dlopen(name::String, local_path::String)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    !isfile(local_path) && _cpu_build()
    isfile(local_path) || error("Library $name not found.")
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_CPU[] != C_NULL && return
    _LIB_CPU[] = _try_dlopen(_SO_NAME, _SO_PATH)
    _FN_INIT[] = dlsym(_LIB_CPU[], :dress_init_graph)
    _FN_FREE[] = dlsym(_LIB_CPU[], :dress_free_graph)

    sym = dlsym_e(_LIB_CPU[], :dress_delta_fit_mpi_omp_fcomm)
    sym == C_NULL && error(
        "dress_delta_fit_mpi_omp_fcomm not found in libdress. " *
        "Ensure libdress.so was built with -DDRESS_MPI=ON and OpenMP."
    )
    _FN_MPI_OMP_FCOMM[] = sym

    sym_nabla = dlsym_e(_LIB_CPU[], :dress_nabla_fit_mpi_omp_fcomm)
    sym_nabla == C_NULL && error(
        "dress_nabla_fit_mpi_omp_fcomm not found in libdress. " *
        "Ensure libdress.so was built with -DDRESS_MPI=ON and OpenMP."
    )
    _FN_NABLA_MPI_OMP_FCOMM[] = sym_nabla
end

# ── delta_fit (MPI + OMP) ──────────────────────────────────────

"""
    delta_fit(N, sources, targets; k=0, comm=nothing, kwargs...) → DeltaDRESSResult

MPI+OMP Δ^k-DRESS.  MPI distributes subgraphs across ranks; within each rank
OpenMP threads parallelise the subgraph slice.

Same keyword arguments as `DRESS.MPI.delta_fit`.
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
                         compute_histogram::Bool = true,
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
              Cint(variant), Cint(precompute))

    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_MPI_OMP_FCOMM[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  nsub_ref,
                  comm_f)

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

# ── nabla_fit (MPI + OMP) ──────────────────────────────────────

"""
    nabla_fit(N, sources, targets; k=0, comm=nothing, kwargs...) → NablaDRESSResult

MPI+OMP ∇^k-DRESS.  MPI distributes tuples across ranks; within each rank
OpenMP threads parallelise the tuple slice.

Same keyword arguments as `DRESS.MPI.nabla_fit`.
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
                         compute_histogram::Bool = true,
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
              Cint(variant), Cint(precompute))

    g == C_NULL && error("dress_init_graph returned NULL")

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    ntup_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_NABLA_MPI_OMP_FCOMM[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint),
                  g, Cint(k), Cint(max_iterations), Cdouble(epsilon),
                  Cint(n_samples), Cuint(seed),
                  compute_histogram ? hsize_ref : Ptr{Cint}(C_NULL),
                  Cint(keep_multisets),
                  keep_multisets ? ms_ref : Ptr{Ptr{Cdouble}}(C_NULL),
                  ntup_ref,
                  comm_f)

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

end # module OMP
