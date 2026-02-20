function T = dress_to_table(result, varargin)
% DRESS_TO_TABLE  Convert a DRESS result struct to a MATLAB table.
%
%   T = DRESS_TO_TABLE(result)
%   T = DRESS_TO_TABLE(result, 'OneBased', true)
%
%   Input:
%     result — struct returned by dress_fit.
%
%   Optional name-value pair:
%     'OneBased' — Logical; if true (default), shift vertex ids to 1-based
%                  for natural MATLAB indexing.
%
%   Output:
%     T — table with columns: src, dst, dress, weight
%
%   See also: dress_fit

    p = inputParser;
    addParameter(p, 'OneBased', true, @(x) isscalar(x) && (islogical(x) || isnumeric(x)));
    parse(p, varargin{:});

    offset = int32(logical(p.Results.OneBased));

    T = table(double(result.sources) + offset, ...
              double(result.targets) + offset, ...
              result.edge_dress, ...
              result.edge_weight, ...
              'VariableNames', {'src', 'dst', 'dress', 'weight'});
end
