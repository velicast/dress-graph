"""
    DRESS.MPI

MPI-distributed Δ^k-DRESS — all MPI logic (stride partitioning +
Allreduce) runs in C.

# Usage

```julia
using MPI
MPI.Init()

using DRESS.MPI
result = DRESS.MPI.delta_fit(4, [0,1,2,0,1,2], [1,2,0,3,3,3]; k=2)

MPI.Finalize()
```

# CUDA + MPI

```julia
using MPI; MPI.Init()
using DRESS.MPI.CUDA
result = DRESS.MPI.CUDA.delta_fit(4, [0,1,2,0,1,2], [1,2,0,3,3,3]; k=2)
```
"""
module MPI_DRESS     # internal name; re-exported as DRESS.MPI

using Libdl

# Re-export types and constants from the parent module.
using ..DRESS: DRESSResult, DeltaDRESSResult, NablaDRESSResult, HistogramEntry, UNDIRECTED, DIRECTED, FORWARD, BACKWARD, _copy_histogram
import ..DRESS: dress_build as _cpu_build
export delta_fit, nabla_fit, DeltaDRESSResult, NablaDRESSResult, UNDIRECTED, DIRECTED, FORWARD, BACKWARD

# ── locate shared library ────────────────────────────────────────────

const _PKG_DIR  = dirname(@__DIR__)
const _LIB_DIR  = normpath(joinpath(_PKG_DIR, "..", "libdress"))
const _SO_NAME  = "libdress" * (Sys.iswindows() ? ".dll" :
                                Sys.isapple()   ? ".dylib" : ".so")
const _SO_PATH  = joinpath(_PKG_DIR, _SO_NAME)

# ── library handles and function pointers ────────────────────────────

const _LIB_HANDLE  = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_INIT     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_FREE     = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI      = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_MPI_FCOMM = Ref{Ptr{Cvoid}}(C_NULL)
const _FN_NABLA_MPI_FCOMM = Ref{Ptr{Cvoid}}(C_NULL)

function _try_dlopen(name::String, local_path::String; build::Bool=false)
    h = dlopen(name; throw_error=false)
    h !== nothing && return h
    build && !isfile(local_path) && _cpu_build()
    isfile(local_path) || error("Library $name not found. Install it or set LD_LIBRARY_PATH.")
    return dlopen(local_path)
end

function _ensure_lib()
    _LIB_HANDLE[] != C_NULL && return

    _LIB_HANDLE[] = _try_dlopen(_SO_NAME, _SO_PATH; build=true)

    _FN_INIT[] = dlsym(_LIB_HANDLE[], :dress_init_graph)
    _FN_FREE[] = dlsym(_LIB_HANDLE[], :dress_free_graph)

    # MPI function — must exist (library built with -DDRESS_MPI=ON)
    sym = dlsym_e(_LIB_HANDLE[], :dress_delta_fit_mpi_fcomm)
    sym == C_NULL && error(
        "dress_delta_fit_mpi_fcomm not found in $(_SO_PATH). " *
        "Rebuild with: cmake -DDRESS_MPI=ON -S libdress -B build && cmake --build build"
    )
    _FN_MPI_FCOMM[] = sym

    sym_nabla = dlsym_e(_LIB_HANDLE[], :dress_nabla_fit_mpi_fcomm)
    sym_nabla == C_NULL && error(
        "dress_nabla_fit_mpi_fcomm not found in $(_SO_PATH). " *
        "Rebuild with: cmake -DDRESS_MPI=ON -S libdress -B build && cmake --build build"
    )
    _FN_NABLA_MPI_FCOMM[] = sym_nabla
end

# ── delta_fit (MPI, CPU backend) ───────────────────────────────

"""
    delta_fit(N, sources, targets; k=0, comm=nothing, kwargs...) → DeltaDRESSResult

MPI-distributed Δ^k-DRESS (CPU backend).

All MPI logic (stride partitioning + Allreduce) runs in C.

# Keyword arguments
- `comm`  – MPI communicator (default: `MPI.COMM_WORLD`).
              Pass any `MPI.Comm` from the MPI.jl package.
- All other keyword arguments are identical to `DRESS.delta_fit`.
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
                         compute_histogram::Bool = true,
                         comm               = nothing)

    # Get Fortran communicator handle via MPI.jl
    if comm === nothing
        MPI_jl = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        comm = getfield(MPI_jl, :COMM_WORLD)
    end
    # MPI.API.MPI_Comm_c2f(comm) → Cint  (OpenMPI: MPI_Comm is a pointer)
    comm_f = ccall((:MPI_Comm_c2f, "libmpi"), Cint, (Ptr{Cvoid},), comm.val)

    E = length(sources)
    length(targets) == E || throw(ArgumentError("sources and targets must have equal length"))

    _ensure_lib()

    # Malloc copies (dress_init_graph takes ownership)
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

    NW_c = if node_weights !== nothing
        nw_ptr = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N) .= Cdouble.(node_weights)
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

    # dress_delta_fit_mpi_fcomm(g, k, iters, eps, &hsize, keep_ms, &ms, &nsub, comm_f)
    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_MPI_FCOMM[], Ptr{HistogramEntry},
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

# ── nabla_fit (MPI, CPU backend) ───────────────────────────────

"""
    nabla_fit(N, sources, targets; k=0, comm=nothing, kwargs...) → NablaDRESSResult

MPI-distributed ∇^k-DRESS (CPU backend).

All MPI logic (stride partitioning + Allreduce) runs in C.

# Keyword arguments
- `comm`  – MPI communicator (default: `MPI.COMM_WORLD`).
              Pass any `MPI.Comm` from the MPI.jl package.
- All other keyword arguments are identical to `DRESS.nabla_fit`.
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
                         compute_histogram::Bool = true,
                         comm               = nothing)

    # Get Fortran communicator handle via MPI.jl
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

    NW_c = if node_weights !== nothing
        nw_ptr = Libc.malloc(N * sizeof(Cdouble))
        unsafe_wrap(Array, Ptr{Cdouble}(nw_ptr), N) .= Cdouble.(node_weights)
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

    h_ptr = ccall(_FN_NABLA_MPI_FCOMM[], Ptr{HistogramEntry},
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

# ── persistent DressGraph (MPI) ──────────────────────────────────────

"""
    DressGraph

A persistent MPI DRESS graph.  `fit!` runs on the CPU;
`delta_fit!` uses MPI distribution across ranks.

```julia
using MPI; MPI.Init()
using DRESS.MPI
g = DressGraph(4, [0,1,2,0,1,2], [1,2,0,3,3,3])
fit!(g)
dr = delta_fit!(g, 2)
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

export DressGraph, delta_fit!, nabla_fit!, close!

"""
    DressGraph(N, sources, targets; weights=nothing,
               variant=UNDIRECTED, precompute_intercepts=false)

Create a persistent MPI DRESS graph object.
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

    g == C_NULL && error("dress_init_graph returned NULL")

    obj = DressGraph(g, Int(N), E, Int32.(sources), Int32.(targets))
    finalizer(obj) do o
        close!(o)
    end
    obj
end

# dress_get symbol for persistent graph queries
const _FN_GET = Ref{Ptr{Cvoid}}(C_NULL)

function _ensure_get()
    _FN_GET[] != C_NULL && return
    _ensure_lib()
    _FN_GET[] = dlsym(_LIB_HANDLE[], :dress_get)
end

"""
    delta_fit!(g::DressGraph, k; max_iterations=100, epsilon=1e-6,
               keep_multisets=false, comm=nothing) → DeltaDRESSResult

MPI-distributed Δ^k-DRESS on a persistent graph.
"""
function delta_fit!(g::DressGraph, k::Integer;
                    max_iterations::Integer=100,
                    epsilon::Real=1e-6,
                    n_samples::Integer=0,
                    seed::Integer=0,
                    keep_multisets::Bool=false,
                    compute_histogram::Bool=true,
                    comm=nothing)
    g.ptr == C_NULL && error("DressGraph already closed")
    _ensure_lib()

    if comm === nothing
        MPI_jl = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        comm = getfield(MPI_jl, :COMM_WORLD)
    end
    comm_f = ccall((:MPI_Comm_c2f, "libmpi"), Cint, (Ptr{Cvoid},), comm.val)

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    nsub_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_MPI_FCOMM[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint),
                  g.ptr, Cint(k), Cint(max_iterations), Cdouble(epsilon),
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
    E = g.e
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

    return DeltaDRESSResult(histogram, ms_mat, nsub)
end

"""
    nabla_fit!(g::DressGraph, k; max_iterations=100, epsilon=1e-6,
               keep_multisets=false, comm=nothing) → NablaDRESSResult

MPI-distributed ∇^k-DRESS on a persistent graph.
"""
function nabla_fit!(g::DressGraph, k::Integer;
                    max_iterations::Integer=100,
                    epsilon::Real=1e-6,
                    n_samples::Integer=0,
                    seed::Integer=0,
                    keep_multisets::Bool=false,
                    compute_histogram::Bool=true,
                    comm=nothing)
    g.ptr == C_NULL && error("DressGraph already closed")
    _ensure_lib()

    if comm === nothing
        MPI_jl = Base.require(Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI"))
        comm = getfield(MPI_jl, :COMM_WORLD)
    end
    comm_f = ccall((:MPI_Comm_c2f, "libmpi"), Cint, (Ptr{Cvoid},), comm.val)

    hsize_ref = Ref{Cint}(0)
    ms_ref    = Ref{Ptr{Cdouble}}(Ptr{Cdouble}(C_NULL))
    ntup_ref  = Ref{Int64}(0)

    h_ptr = ccall(_FN_NABLA_MPI_FCOMM[], Ptr{HistogramEntry},
                  (Ptr{Cvoid}, Cint, Cint, Cdouble, Cint, Cuint, Ptr{Cint}, Cint,
                   Ptr{Ptr{Cdouble}}, Ptr{Int64}, Cint),
                  g.ptr, Cint(k), Cint(max_iterations), Cdouble(epsilon),
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
    Base.get(g::DressGraph, u, v; max_iterations=100, epsilon=1e-6, edge_weight=1.0)

Query the DRESS value for an edge (existing or virtual).
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
    DRESSResult(copy(g.sources), copy(g.targets), ew, ed, nd, 0, 0.0)
end

export result

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
    print(io, "DressGraph{MPI}(N=$(g.n), E=$(g.e), $state)")
end

# ── CUDA + MPI submodule ─────────────────────────────────────────────

include("MPI_CUDA.jl")

# ── OMP + MPI submodule ──────────────────────────────────────────────

include("MPI_OMP.jl")

end # module MPI_DRESS
