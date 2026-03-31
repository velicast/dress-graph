function result = fit(n_vertices, sources, targets, varargin)
% CUDA.DRESS_FIT  GPU-accelerated DRESS edge similarity.
%
%   result = cuda.fit(n_vertices, sources, targets)
%   result = cuda.fit(n_vertices, sources, targets, Name, Value, ...)
%
%   Same API as fit(), but the iterative fitting loop runs on the
%   GPU via CUDA.  Switch from CPU to GPU by adding the cuda. prefix:
%
%     % CPU
%     r = fit(4, sources, targets);
%
%     % CUDA (same call, different namespace)
%     r = cuda.fit(4, sources, targets);
%
%   See also: fit, dress_cuda_mex

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

    assert(isscalar(n_vertices) && n_vertices >= 1, ...
           'cuda:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'cuda:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'cuda:invalidInput', 'Variant must be 0, 1, 2, or 3.');

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    if isempty(weights)
        weights = [];
    else
        weights = double(weights(:));
        assert(numel(weights) == numel(sources), ...
               'cuda:invalidInput', 'Weights must have the same length as sources.');
    end

    result = dress_cuda_mex(n_vertices, sources, targets, weights, ...
                            variant, max_iterations, epsilon, precompute);
end
