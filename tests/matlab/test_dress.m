% TEST_DRESS  Test suite for the MATLAB/Octave DRESS wrapper.
%
%   Run from the repo root:
%     octave --no-gui tests/matlab/test_dress.m
%
%   Or from tests/matlab/:
%     octave --no-gui test_dress.m

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab'));

passed = 0;
failed = 0;

%% --- helpers ---

function assert_eq(a, b, msg)
    if ~isequal(a, b)
        error('FAIL: %s (got %s, expected %s)', msg, mat2str(a), mat2str(b));
    end
end

function assert_near(a, b, tol, msg)
    if any(abs(a - b) > tol)
        error('FAIL: %s (got %s, expected %s, tol %g)', msg, mat2str(a), mat2str(b), tol);
    end
end

function assert_true(cond, msg)
    if ~cond
        error('FAIL: %s', msg);
    end
end

%% === Construction ===

% -- unweighted triangle --
try
    r = dress_fit(3, int32([0;1;0]), int32([1;2;2]));
    assert_eq(length(r.edge_dress), 3, 'triangle: 3 edges');
    assert_eq(length(r.node_dress), 3, 'triangle: 3 nodes');
    assert_true(r.iterations > 0, 'triangle: iterations > 0');
    passed = passed + 1;
    fprintf('PASS: unweighted triangle\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: unweighted triangle: %s\n', e.message);
end

% -- weighted triangle --
try
    r = dress_fit(3, int32([0;1;0]), int32([1;2;2]), 'Weights', [1.0;2.0;3.0]);
    assert_eq(length(r.edge_dress), 3, 'weighted: 3 edges');
    assert_near(r.edge_weight(1), 2.0, 1e-12, 'weighted: w0 doubled');
    assert_near(r.edge_weight(2), 4.0, 1e-12, 'weighted: w1 doubled');
    assert_near(r.edge_weight(3), 6.0, 1e-12, 'weighted: w2 doubled');
    passed = passed + 1;
    fprintf('PASS: weighted triangle\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: weighted triangle: %s\n', e.message);
end

% -- all four variants --
try
    for v = 0:3
        r = dress_fit(3, int32([0;1;0]), int32([1;2;2]), 'Variant', v);
        assert_eq(length(r.edge_dress), 3, sprintf('variant %d: 3 edges', v));
    end
    passed = passed + 1;
    fprintf('PASS: all variants\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: all variants: %s\n', e.message);
end

% -- precompute intercepts --
try
    r1 = dress_fit(3, int32([0;1;0]), int32([1;2;2]), 'PrecomputeIntercepts', true);
    r2 = dress_fit(3, int32([0;1;0]), int32([1;2;2]), 'PrecomputeIntercepts', false);
    assert_near(r1.edge_dress, r2.edge_dress, 1e-10, 'intercepts: same result');
    passed = passed + 1;
    fprintf('PASS: precompute intercepts\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: precompute intercepts: %s\n', e.message);
end

%% === Fitting ===

% -- triangle convergence --
try
    r = dress_fit(3, int32([0;1;0]), int32([1;2;2]));
    assert_true(r.delta < 1e-6, 'triangle: converged');
    assert_near(r.edge_dress, [2;2;2], 1e-10, 'triangle: all edges = 2');
    passed = passed + 1;
    fprintf('PASS: triangle convergence\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: triangle convergence: %s\n', e.message);
end

% -- path graph 0-1-2-3 --
try
    r = dress_fit(4, int32([0;1;2]), int32([1;2;3]));
    assert_true(all(r.edge_dress > 0), 'path: all > 0');
    assert_true(all(r.edge_dress <= 2), 'path: all <= 2');
    assert_near(r.edge_dress(1), r.edge_dress(3), 1e-6, 'path: endpoints symmetric');
    assert_true(r.edge_dress(2) < r.edge_dress(1), 'path: interior < endpoint');
    passed = passed + 1;
    fprintf('PASS: path graph\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: path graph: %s\n', e.message);
end

% -- complete K4 --
try
    r = dress_fit(4, int32([0;0;0;1;1;2]), int32([1;2;3;2;3;3]));
    assert_near(r.edge_dress, 2*ones(6,1), 1e-10, 'K4: all edges = 2');
    passed = passed + 1;
    fprintf('PASS: complete K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: complete K4: %s\n', e.message);
end

%% === Boundedness ===

try
    r = dress_fit(3, int32([0;1;0]), int32([1;2;2]));
    assert_true(all(r.edge_dress >= 0), 'bounded: >= 0');
    assert_true(all(r.edge_dress <= 2), 'bounded: <= 2');
    passed = passed + 1;
    fprintf('PASS: boundedness\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: boundedness: %s\n', e.message);
end

%% === Node dress ===

try
    r = dress_fit(3, int32([0;1;0]), int32([1;2;2]));
    assert_true(all(r.node_dress > 0), 'node_dress: all > 0');
    assert_near(r.node_dress(1), r.node_dress(2), 1e-10, 'node_dress: symmetric');
    assert_near(r.node_dress(2), r.node_dress(3), 1e-10, 'node_dress: symmetric');
    passed = passed + 1;
    fprintf('PASS: node dress\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: node dress: %s\n', e.message);
end

%% === Star graph ===

try
    r = dress_fit(5, int32([0;0;0;0]), int32([1;2;3;4]));
    assert_true(all(r.edge_dress > 0), 'star: all > 0');
    assert_near(r.edge_dress(1), r.edge_dress(2), 1e-10, 'star: symmetric edges');
    assert_near(r.edge_dress(3), r.edge_dress(4), 1e-10, 'star: symmetric edges');
    passed = passed + 1;
    fprintf('PASS: star graph\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: star graph: %s\n', e.message);
end

%% === Determinism ===

try
    r1 = dress_fit(4, int32([0;1;2;0]), int32([1;2;3;3]));
    r2 = dress_fit(4, int32([0;1;2;0]), int32([1;2;3;3]));
    assert_near(r1.edge_dress, r2.edge_dress, 0, 'determinism: identical');
    passed = passed + 1;
    fprintf('PASS: determinism\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: determinism: %s\n', e.message);
end

%% --- summary ---

fprintf('\n%d passed, %d failed, %d total\n', passed, failed, passed + failed);
if failed > 0
    exit(1);
end
