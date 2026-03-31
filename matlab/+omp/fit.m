function result = fit(n_vertices, sources, targets, varargin)
% OMP.DRESS_FIT  OpenMP-parallel DRESS edge similarity.
%
%   result = omp.fit(n_vertices, sources, targets)
%   result = omp.fit(n_vertices, sources, targets, Name, Value, ...)
%
%   Same API as fit(), but the iterative fitting loop runs on the
%   OpenMP.  Switch from CPU to GPU by adding the omp. prefix:
%
%     % CPU
%     r = fit(4, sources, targets);
%
%     % OpenMP (same call, different namespace)
%     r = omp.fit(4, sources, targets);
%
%   See also: fit, dress_omp_mex

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
           'omp:invalidInput', 'n_vertices must be a positive scalar.');
    assert(numel(sources) == numel(targets), ...
           'omp:invalidInput', 'sources and targets must have the same length.');
    assert(variant >= 0 && variant <= 3, ...
           'omp:invalidInput', 'Variant must be 0, 1, 2, or 3.');

    n_vertices = int32(n_vertices);
    sources    = int32(sources(:));
    targets    = int32(targets(:));

    if isempty(weights)
        weights = [];
    else
        weights = double(weights(:));
        assert(numel(weights) == numel(sources), ...
               'omp:invalidInput', 'Weights must have the same length as sources.');
    end

    result = dress_omp_mex(n_vertices, sources, targets, weights, ...
                            variant, max_iterations, epsilon, precompute);
end
