function result = dress_fit(n_vertices, sources, targets, varargin)
% DRESS_FIT  Compute DRESS edge similarity on a graph.
%
%   result = DRESS_FIT(n_vertices, sources, targets)
%   result = DRESS_FIT(n_vertices, sources, targets, Name, Value, ...)
%
%   Build a DRESS graph from an edge list and run iterative fitting.
%   Vertex ids are 0-based (C convention).
%
%   Required inputs:
%     n_vertices — Number of vertices (vertex ids must be in 0..n_vertices-1).
%     sources    — int32 or double [E x 1] edge source endpoints (0-based).
%     targets    — int32 or double [E x 1] edge target endpoints (0-based).
%
%   Optional name-value pairs:
%     'Weights'               — double [E x 1] per-edge weights (default: unweighted).
%     'Variant'               — Graph variant (default 0):
%                                 0 = UNDIRECTED, 1 = DIRECTED,
%                                 2 = FORWARD,    3 = BACKWARD.
%     'MaxIterations'         — Maximum fitting iterations (default 100).
%     'Epsilon'               — Convergence threshold (default 1e-6).
%     'PrecomputeIntercepts'  — Logical; precompute common-neighbor index
%                                for faster iteration (default false).
%
%   Output:
%     result — struct with fields:
%       .sources      — int32 [E x 1]  edge source endpoints (0-based)
%       .targets      — int32 [E x 1]  edge target endpoints (0-based)
%       .edge_dress   — double [E x 1] DRESS similarity per edge
%       .edge_weight  — double [E x 1] variant-specific edge weight
%       .node_dress   — double [N x 1] per-node norm
%       .iterations   — int32 scalar   iterations performed
%       .delta        — double scalar  final max per-edge change
%
%   Examples:
%     % Triangle + pendant: 0-1, 1-2, 2-0, 2-3
%     r = dress_fit(4, int32([0;1;2;2]), int32([1;2;0;3]));
%     disp(r.edge_dress);
%
%     % Weighted, directed
%     r = dress_fit(4, [0;1;2;2], [1;2;0;3], ...
%                   'Weights', [1;2;1;0.5], 'Variant', 1);
%
%     % With precomputed intercepts (faster for large/dense graphs)
%     r = dress_fit(4, [0;1;2;2], [1;2;0;3], 'PrecomputeIntercepts', true);
%
%   See also: dress_mex, dress_to_table

    % ---- Parse optional arguments ----
    p = inputParser;
    addParameter(p, 'Weights',              [],    @(x) isempty(x) || (isnumeric(x) && isreal(x)));
    addParameter(p, 'Variant',              0,     @(x) isscalar(x) && isnumeric(x));
    addParameter(p, 'MaxIterations',        100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
    addParameter(p, 'Epsilon',              1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
    addParameter(p, 'PrecomputeIntercepts', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    parse(p, varargin{:});

    weights        = p.Results.Weights;
    variant        = int32(p.Results.Variant);
    max_iterations = int32(p.Results.MaxIterations);
    epsilon        = double(p.Results.Epsilon);
    precompute     = int32(logical(p.Results.PrecomputeIntercepts));

    % ---- Validate ----
    assert(isscalar(n_vertices) && n_vertices >= 1, ...
           'dress:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'dress:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'dress:invalidInput', 'Variant must be 0, 1, 2, or 3.');

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    if isempty(weights)
        weights = [];
    else
        weights = double(weights(:));
        assert(numel(weights) == numel(sources), ...
               'dress:invalidInput', 'Weights must have the same length as sources.');
    end

    % ---- Call MEX ----
    result = dress_mex(n_vertices, sources, targets, weights, ...
                       variant, max_iterations, epsilon, precompute);
end
