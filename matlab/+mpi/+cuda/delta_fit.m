function result = delta_fit(n_vertices, sources, targets, varargin)
% MPI.CUDA.DELTA_DRESS_FIT  MPI+CUDA distributed Delta-k-DRESS exact histogram.
%
%   result = mpi.cuda.delta_fit(n_vertices, sources, targets)
%   result = mpi.cuda.delta_fit(n_vertices, sources, targets, Name, Value, ...)
%
%   Same API as delta_fit(), but subgraph enumeration is distributed
%   across MPI ranks and each subgraph fitting runs on the GPU via CUDA.
%   Switch from CPU to MPI+CUDA by adding the mpi.cuda. prefix:
%
%     % CPU
%     r = delta_fit(4, sources, targets, 'K', 1);
%
%     % MPI+CUDA (same call, different namespace)
%     r = mpi.cuda.delta_fit(4, sources, targets, 'K', 1);
%
%   Required inputs:
%     n_vertices — Number of vertices (vertex ids must be in 0..n_vertices-1).
%     sources    — int32 or double [E x 1] edge source endpoints (0-based).
%     targets    — int32 or double [E x 1] edge target endpoints (0-based).
%
%   Optional name-value pairs:
%     'Weights'          — double [E x 1] edge weights (default [] = unweighted).
%     'K'                — Vertices to remove per subset (default 0 = original).
%     'Variant'          — Graph variant (default 0 = UNDIRECTED).
%     'MaxIterations'    — Maximum fitting iterations (default 100).
%     'Epsilon'          — Convergence threshold / bin width (default 1e-6).
%     'Precompute'       — Logical; precompute intercepts (default false).
%     'KeepMultisets'    — Logical; return per-subgraph edge values (default false).
%     'CommF'            — Fortran MPI communicator handle (default: from pbdMPI).
%
%   Output:
%     result — struct with fields:
%       .histogram      — struct with sparse exact histogram entries:
%                         .value [num_entries x 1] double
%                         .count [num_entries x 1] int64
%       .multisets      — double [C(N,k) x E]     per-subgraph edge values
%                         (only when KeepMultisets is true; NaN = removed edge)
%       .num_subgraphs  — int32 scalar            C(N,k)
%                         (only when KeepMultisets is true)
%
%   See also: delta_fit, mpi.cuda.DRESS, delta_dress_mpi_cuda_mex

    p = inputParser;
    addParameter(p, 'Weights',       [],    @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(p, 'K',             0,     @(x) isscalar(x) && isnumeric(x) && x >= 0);
    addParameter(p, 'Variant',       0,     @(x) isscalar(x) && isnumeric(x));
    addParameter(p, 'MaxIterations', 100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
    addParameter(p, 'Epsilon',       1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
    addParameter(p, 'Precompute',    false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    addParameter(p, 'KeepMultisets', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    addParameter(p, 'CommF',         [],    @(x) isempty(x) || (isscalar(x) && isnumeric(x)));
    parse(p, varargin{:});

    weights        = double(p.Results.Weights(:));
    k              = int32(p.Results.K);
    variant        = int32(p.Results.Variant);
    max_iterations = int32(p.Results.MaxIterations);
    epsilon        = double(p.Results.Epsilon);
    precompute     = int32(logical(p.Results.Precompute));
    keep_multisets = int32(logical(p.Results.KeepMultisets));

    assert(isscalar(n_vertices) && n_vertices >= 1, ...
           'mpi_cuda:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'mpi_cuda:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'mpi_cuda:invalidInput', 'Variant must be 0, 1, 2, or 3.');
    if ~isempty(weights)
        assert(numel(weights) == numel(sources), ...
               'mpi_cuda:invalidInput', 'weights must have the same length as sources.');
    end

    % Resolve communicator handle
    comm_f = p.Results.CommF;
    if isempty(comm_f)
        comm_f = int32(pbdMPI.spmd.comm.c2f());
    else
        comm_f = int32(comm_f);
    end

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    result = delta_dress_mpi_cuda_mex(n_vertices, sources, targets, weights, k, ...
                                      variant, max_iterations, epsilon, precompute, ...
                                      keep_multisets, comm_f);
end
