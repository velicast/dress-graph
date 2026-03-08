function result = delta_dress_fit(n_vertices, sources, targets, varargin)
% CUDA.DELTA_DRESS_FIT  GPU-accelerated Delta-k-DRESS histogram.
%
%   result = cuda.delta_dress_fit(n_vertices, sources, targets)
%   result = cuda.delta_dress_fit(n_vertices, sources, targets, Name, Value, ...)
%
%   Same API as delta_dress_fit(), but each subgraph fitting runs on the
%   GPU via CUDA.  Switch from CPU to GPU by adding the cuda. prefix:
%
%     % CPU
%     r = delta_dress_fit(4, sources, targets, 'K', 1);
%
%     % CUDA (same call, different namespace)
%     r = cuda.delta_dress_fit(4, sources, targets, 'K', 1);
%
%   See also: delta_dress_fit, delta_dress_cuda_mex

    p = inputParser;
    addParameter(p, 'Weights',       [],    @(x) isempty(x) || (isnumeric(x) && isvector(x)));
    addParameter(p, 'K',             0,     @(x) isscalar(x) && isnumeric(x) && x >= 0);
    addParameter(p, 'Variant',       0,     @(x) isscalar(x) && isnumeric(x));
    addParameter(p, 'MaxIterations', 100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
    addParameter(p, 'Epsilon',       1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
    addParameter(p, 'Precompute',    false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    addParameter(p, 'KeepMultisets', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    parse(p, varargin{:});

    weights        = double(p.Results.Weights(:));
    k              = int32(p.Results.K);
    variant        = int32(p.Results.Variant);
    max_iterations = int32(p.Results.MaxIterations);
    epsilon        = double(p.Results.Epsilon);
    precompute     = int32(logical(p.Results.Precompute));
    keep_multisets = int32(logical(p.Results.KeepMultisets));

    assert(isscalar(n_vertices) && n_vertices >= 1, ...
           'cuda:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'cuda:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'cuda:invalidInput', 'Variant must be 0, 1, 2, or 3.');
    if ~isempty(weights)
        assert(numel(weights) == numel(sources), ...
               'cuda:invalidInput', 'weights must have the same length as sources.');
    end

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    result = delta_dress_cuda_mex(n_vertices, sources, targets, weights, k, ...
                                  variant, max_iterations, epsilon, precompute, ...
                                  keep_multisets);
end
