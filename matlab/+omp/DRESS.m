classdef DRESS < handle
% OMP.DRESS  Persistent OpenMP-accelerated DRESS graph object.
%
%   g = omp.DRESS(n_vertices, sources, targets)
%   g = omp.DRESS(n_vertices, sources, targets, 'Weights', w, ...)
%
%   Methods:
%     g.fit()                            — run OpenMP DRESS fitting
%     g.fit('MaxIterations', 100, 'Epsilon', 1e-6)
%     res = g.delta_fit('K', 2)          — OpenMP Δ^k-DRESS
%     res = g.nabla_fit('K', 2)          — OpenMP ∇^k-DRESS
%     val = g.get(u, v)                  — query existing/virtual edge (CPU)
%     res = g.result()                   — extract current results
%     g.close()                          — explicitly free C graph
%
%   The C graph is freed automatically when the object is deleted.
%
%   Example:
%     g = omp.DRESS(4, int32([0;1;2;2]), int32([1;2;0;3]));
%     g.fit();
%     d = g.get(0, 3);
%     disp(d);
%     g.close();

    properties (Access = private)
        ptr = uint64(0)   % opaque C pointer
    end

    methods
        function obj = DRESS(n_vertices, sources, targets, varargin)
            p = inputParser;
            addParameter(p, 'Weights',              [],    @(x) isempty(x) || (isnumeric(x) && isreal(x)));
            addParameter(p, 'Variant',              0,     @(x) isscalar(x) && isnumeric(x));
            addParameter(p, 'PrecomputeIntercepts', false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
            parse(p, varargin{:});

            weights    = p.Results.Weights;
            variant    = int32(p.Results.Variant);
            precompute = int32(logical(p.Results.PrecomputeIntercepts));

            n_vertices = int32(n_vertices);
            sources    = int32(sources(:));
            targets    = int32(targets(:));
            if ~isempty(weights)
                weights = double(weights(:));
            end

            obj.ptr = dress_init_mex(n_vertices, sources, targets, ...
                                     weights, variant, precompute);
        end

        function result = fit(obj, varargin)
            p = inputParser;
            addParameter(p, 'MaxIterations', 100,  @(x) isscalar(x) && isnumeric(x) && x >= 1);
            addParameter(p, 'Epsilon',       1e-6, @(x) isscalar(x) && isnumeric(x) && x > 0);
            parse(p, varargin{:});

            result = dress_fit_omp_obj_mex(obj.ptr, ...
                                           int32(p.Results.MaxIterations), ...
                                           double(p.Results.Epsilon));
        end

        function result = delta_fit(obj, varargin)
        % DELTA_FIT  OpenMP Δ^k-DRESS on this persistent graph.
            p = inputParser;
            addParameter(p, 'K',              0,     @(x) isscalar(x) && isnumeric(x) && x >= 0);
            addParameter(p, 'MaxIterations',  100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
            addParameter(p, 'Epsilon',        1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
            addParameter(p, 'KeepMultisets',  false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
            parse(p, varargin{:});

            result = delta_dress_omp_obj_mex(obj.ptr, ...
                                             int32(p.Results.K), ...
                                             int32(p.Results.MaxIterations), ...
                                             double(p.Results.Epsilon), ...
                                             int32(logical(p.Results.KeepMultisets)));
        end

        function result = nabla_fit(obj, varargin)
        % NABLA_FIT  OpenMP ∇^k-DRESS on this persistent graph.
            p = inputParser;
            addParameter(p, 'K',              0,     @(x) isscalar(x) && isnumeric(x) && x >= 0);
            addParameter(p, 'MaxIterations',  100,   @(x) isscalar(x) && isnumeric(x) && x >= 1);
            addParameter(p, 'Epsilon',        1e-6,  @(x) isscalar(x) && isnumeric(x) && x > 0);
            addParameter(p, 'KeepMultisets',  false, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
            parse(p, varargin{:});

            result = nabla_dress_omp_obj_mex(obj.ptr, ...
                                             int32(p.Results.K), ...
                                             int32(p.Results.MaxIterations), ...
                                             double(p.Results.Epsilon), ...
                                             int32(logical(p.Results.KeepMultisets)));
        end

        function val = get(obj, u, v, varargin)
            p = inputParser;
            addParameter(p, 'MaxIterations', 100,  @(x) isscalar(x) && isnumeric(x) && x >= 1);
            addParameter(p, 'Epsilon',       1e-6, @(x) isscalar(x) && isnumeric(x) && x > 0);
            addParameter(p, 'EdgeWeight',    1.0,  @(x) isscalar(x) && isnumeric(x));
            parse(p, varargin{:});

            val = dress_get_mex(obj.ptr, int32(u), int32(v), ...
                                int32(p.Results.MaxIterations), ...
                                double(p.Results.Epsilon), ...
                                double(p.Results.EdgeWeight));
        end

        function res = result(obj)
            res = dress_result_mex(obj.ptr);
        end

        function close(obj)
            if obj.ptr ~= uint64(0)
                dress_free_mex(obj.ptr);
                obj.ptr = uint64(0);
            end
        end

        function delete(obj)
            obj.close();
        end
    end
end
