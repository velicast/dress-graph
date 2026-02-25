% TEST_DELTA_DRESS  Test suite for the MATLAB/Octave delta-k-DRESS wrapper.
%
%   Run from the repo root:
%     octave --no-gui tests/matlab/test_delta_dress.m
%
%   Or from tests/matlab/:
%     octave --no-gui test_delta_dress.m

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab'));

passed = 0;
failed = 0;

%% --- helpers ---

function assert_eq(a, b, msg)
    if ~isequal(a, b)
        error('FAIL: %s (got %s, expected %s)', msg, mat2str(a), mat2str(b));
    end
end

function assert_true(cond, msg)
    if ~cond
        error('FAIL: %s', msg);
    end
end

K3_SRC = int32([0;1;0]);
K3_TGT = int32([1;2;2]);
K4_SRC = int32([0;0;0;1;1;2]);
K4_TGT = int32([1;2;3;2;3;3]);
P4_SRC = int32([0;1;2]);
P4_TGT = int32([1;2;3]);
EPS = 1e-3;

%% === Histogram size ===

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'Epsilon', 1e-3);
    assert_eq(r.hist_size, int32(2001), 'hist_size with eps=1e-3');
    assert_eq(length(r.histogram), double(r.hist_size), 'histogram length');
    passed = passed + 1;
    fprintf('PASS: histogram size eps=1e-3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: histogram size eps=1e-3: %s\n', e.message);
end

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'Epsilon', 1e-6);
    assert_eq(r.hist_size, int32(2000001), 'hist_size with eps=1e-6');
    passed = passed + 1;
    fprintf('PASS: histogram size eps=1e-6\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: histogram size eps=1e-6: %s\n', e.message);
end

%% === Delta-0 on K3 ===

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'K', 0, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 3, 'K3 delta0 total = 3');
    assert_true(r.histogram(end) > 0, 'K3 delta0 top bin > 0');
    nonzero = sum(r.histogram > 0);
    assert_eq(nonzero, 1, 'K3 delta0 single non-zero bin');
    passed = passed + 1;
    fprintf('PASS: delta-0 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-0 on K3: %s\n', e.message);
end

%% === Delta-1 on K3 ===

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'K', 1, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 3, 'K3 delta1 total = 3');
    passed = passed + 1;
    fprintf('PASS: delta-1 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on K3: %s\n', e.message);
end

%% === Delta-2 on K3 ===

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'K', 2, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 0, 'K3 delta2 total = 0');
    passed = passed + 1;
    fprintf('PASS: delta-2 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-2 on K3: %s\n', e.message);
end

%% === Delta-0 on K4 ===

try
    r = delta_dress_fit(4, K4_SRC, K4_TGT, 'K', 0, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 6, 'K4 delta0 total = 6');
    assert_eq(r.histogram(end), 6, 'K4 delta0 top bin = 6');
    passed = passed + 1;
    fprintf('PASS: delta-0 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-0 on K4: %s\n', e.message);
end

%% === Delta-1 on K4 ===

try
    r = delta_dress_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 12, 'K4 delta1 total = 12');
    assert_eq(r.histogram(end), 12, 'K4 delta1 top bin = 12');
    passed = passed + 1;
    fprintf('PASS: delta-1 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on K4: %s\n', e.message);
end

%% === Delta-2 on K4 ===

try
    r = delta_dress_fit(4, K4_SRC, K4_TGT, 'K', 2, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 6, 'K4 delta2 total = 6');
    passed = passed + 1;
    fprintf('PASS: delta-2 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-2 on K4: %s\n', e.message);
end

%% === k >= N (empty) ===

try
    r = delta_dress_fit(3, K3_SRC, K3_TGT, 'K', 3, 'Epsilon', EPS);
    assert_eq(sum(r.histogram), 0, 'k=N total = 0');
    r2 = delta_dress_fit(3, K3_SRC, K3_TGT, 'K', 10, 'Epsilon', EPS);
    assert_eq(sum(r2.histogram), 0, 'k>N total = 0');
    passed = passed + 1;
    fprintf('PASS: k >= N (empty)\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: k >= N (empty): %s\n', e.message);
end

%% === Precompute flag ===

try
    r1 = delta_dress_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS, ...
                          'Precompute', false);
    r2 = delta_dress_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS, ...
                          'Precompute', true);
    assert_eq(r1.hist_size, r2.hist_size, 'precompute: same hist_size');
    assert_true(isequal(r1.histogram, r2.histogram), 'precompute: same histogram');
    passed = passed + 1;
    fprintf('PASS: precompute flag\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: precompute flag: %s\n', e.message);
end

%% === Path P4 ===

try
    r = delta_dress_fit(4, P4_SRC, P4_TGT, 'K', 0, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 3, 'P4 delta0 total = 3');
    nonzero = sum(r.histogram > 0);
    assert_true(nonzero >= 2, 'P4 at least 2 distinct bins');
    passed = passed + 1;
    fprintf('PASS: path P4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: path P4: %s\n', e.message);
end

%% === Delta-1 on P4 ===

try
    r = delta_dress_fit(4, P4_SRC, P4_TGT, 'K', 1, 'Epsilon', EPS);
    total = sum(r.histogram);
    assert_eq(total, 6, 'P4 delta1 total = 6');
    passed = passed + 1;
    fprintf('PASS: delta-1 on P4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on P4: %s\n', e.message);
end

%% === Summary ===

fprintf('\n%d passed, %d failed.\n', passed, failed);
if failed > 0
    exit(1);
end
