function result = delta_dress_fit(n_vertices, sources, targets, varargin)
% DELTA_DRESS_FIT  Compute the Delta-k-DRESS histogram.
%
%   result = DELTA_DRESS_FIT(n_vertices, sources, targets)
%   result = DELTA_DRESS_FIT(n_vertices, sources, targets, Name, Value, ...)
%
%   Exhaustively remove all k-vertex subsets from the graph and measure
%   the change in edge similarity values, collecting the results in a
%   histogram with bin width equal to epsilon.
%
%   Required inputs:
%     n_vertices — Number of vertices (vertex ids must be in 0..n_vertices-1).
%     sources    — int32 or double [E x 1] edge source endpoints (0-based).
%     targets    — int32 or double [E x 1] edge target endpoints (0-based).
%
%   Optional name-value pairs:
%     'K'                — Vertices to remove per subset (default 0 = original).
%     'Variant'          — Graph variant (default 0 = UNDIRECTED).
%     'MaxIterations'    — Maximum fitting iterations (default 100).
%     'Epsilon'          — Convergence threshold / bin width (default 1e-6).
%     'Precompute'       — Logical; precompute intercepts (default false).
%
%   Output:
%     result — struct with fields:
%       .histogram  — double [hist_size x 1]  bin counts
%       .hist_size  — int32 scalar            number of bins
%
%   Examples:
%     % Triangle K3, delta-1
%     r = delta_dress_fit(3, int32([0;1;2]), int32([1;2;0]), 'K', 1);
%     disp(r.hist_size);
%
%   See also: dress_fit, delta_dress_mex

    p = inputParser;
    addParameter(p, 'K',             0,     @(x) isscalar(x) && isnumeric(x) && x >= 0);
    addParameter(p, 'Variant',       0,     @(x) isscalar(x) && isnumeric(x));
    addParameter(p, 'MaxIterations', 100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
    addParameter(p, 'Epsilon',       1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
    addParameter(p, 'Precompute',    false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    parse(p, varargin{:});

    k              = int32(p.Results.K);
    variant        = int32(p.Results.Variant);
    max_iterations = int32(p.Results.MaxIterations);
    epsilon        = double(p.Results.Epsilon);
    precompute     = int32(logical(p.Results.Precompute));

    assert(isscalar(n_vertices) && n_vertices >= 1, ...
           'delta_dress:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'delta_dress:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'delta_dress:invalidInput', 'Variant must be 0, 1, 2, or 3.');

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    result = delta_dress_mex(n_vertices, sources, targets, k, ...
                             variant, max_iterations, epsilon, precompute);
end
