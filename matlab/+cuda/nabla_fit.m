function result = nabla_fit(n_vertices, sources, targets, varargin)
% CUDA.NABLA_DRESS_FIT  GPU-accelerated Nabla-k-DRESS exact histogram.
%
%   result = cuda.nabla_fit(n_vertices, sources, targets)
%   result = cuda.nabla_fit(n_vertices, sources, targets, Name, Value, ...)
%
%   Same API as nabla_fit(), but each tuple fitting runs on the
%   GPU via CUDA. The returned histogram is sparse exact entries in
%   result.histogram.value and result.histogram.count. Switch from CPU to
%   GPU by adding the cuda. prefix:
%
%     % CPU
%     r = nabla_fit(4, sources, targets, 'K', 1);
%
%     % CUDA (same call, different namespace)
%     r = cuda.nabla_fit(4, sources, targets, 'K', 1);
%
%   See also: nabla_fit, nabla_dress_cuda_mex

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

    result = nabla_dress_cuda_mex(n_vertices, sources, targets, weights, k, ...
                                  variant, max_iterations, epsilon, precompute, ...
                                  keep_multisets);
end
