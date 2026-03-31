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

function total = hist_total(r)
    total = sum(double(r.histogram.count));
end

function n = hist_num_entries(r)
    n = numel(r.histogram.count);
end

function count = hist_count_value(r, value, tol)
    if nargin < 3
        tol = 1e-9;
    end
    idx = abs(r.histogram.value - value) < tol;
    count = sum(double(r.histogram.count(idx)));
end

function same = hist_equal(r1, r2)
    same = isequal(r1.histogram.value, r2.histogram.value) && ...
           isequal(r1.histogram.count, r2.histogram.count);
end

K3_SRC = int32([0;1;0]);
K3_TGT = int32([1;2;2]);
K4_SRC = int32([0;0;0;1;1;2]);
K4_TGT = int32([1;2;3;2;3;3]);
P4_SRC = int32([0;1;2]);
P4_TGT = int32([1;2;3]);
EPS = 1e-3;

%% === Histogram entries ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'Epsilon', 1e-3);
    assert_true(isstruct(r.histogram), 'histogram is a struct');
    assert_true(isfield(r.histogram, 'value') && isfield(r.histogram, 'count'), ...
                'histogram has value/count fields');
    assert_true(~isfield(r, 'hist_size'), 'hist_size is not exposed');
    assert_eq(hist_num_entries(r), 1, 'single entry with eps=1e-3');
    passed = passed + 1;
    fprintf('PASS: histogram entries eps=1e-3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: histogram entries eps=1e-3: %s\n', e.message);
end

try
    r1 = dress_delta_fit(3, K3_SRC, K3_TGT, 'Epsilon', 1e-3);
    r2 = dress_delta_fit(3, K3_SRC, K3_TGT, 'Epsilon', 1e-6);
    assert_eq(hist_num_entries(r2), 1, 'single entry with eps=1e-6');
    assert_true(hist_equal(r1, r2), 'same exact histogram across eps values');
    passed = passed + 1;
    fprintf('PASS: histogram entries eps=1e-6\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: histogram entries eps=1e-6: %s\n', e.message);
end

%% === Weighted histogram entries ===

try
    rw = dress_delta_fit(3, K3_SRC, K3_TGT, 'Weights', [1.0; 10.0; 1.0], ...
                         'Epsilon', 1e-3);
    assert_true(hist_num_entries(rw) > 1, 'weighted exact histogram has multiple entries');
    assert_eq(hist_total(rw), 3, 'weighted K3 delta0 total = 3');
    passed = passed + 1;
    fprintf('PASS: weighted histogram entries\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: weighted histogram entries: %s\n', e.message);
end

%% === Return type ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'Epsilon', EPS);
    assert_true(isstruct(r), 'result is a struct');
    assert_true(isfield(r, 'histogram'), 'has histogram field');
    assert_true(isstruct(r.histogram), 'histogram is a struct');
    assert_true(isequal(sort(fieldnames(r.histogram)), {'count'; 'value'}), ...
                'histogram has value/count fields');
    assert_true(~isfield(r, 'hist_size'), 'does not expose hist_size');
    passed = passed + 1;
    fprintf('PASS: return type\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: return type: %s\n', e.message);
end

%% === Delta-0 on K3 ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 0, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 3, 'K3 delta0 total = 3');
    assert_eq(hist_num_entries(r), 1, 'K3 delta0 single histogram entry');
    assert_eq(hist_count_value(r, 2.0), 3, 'K3 delta0 value 2.0 count = 3');
    passed = passed + 1;
    fprintf('PASS: delta-0 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-0 on K3: %s\n', e.message);
end

%% === Delta-1 on K3 ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 1, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 3, 'K3 delta1 total = 3');
    passed = passed + 1;
    fprintf('PASS: delta-1 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on K3: %s\n', e.message);
end

%% === Delta-2 on K3 ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 2, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 0, 'K3 delta2 total = 0');
    passed = passed + 1;
    fprintf('PASS: delta-2 on K3\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-2 on K3: %s\n', e.message);
end

%% === Delta-0 on K4 ===

try
    r = dress_delta_fit(4, K4_SRC, K4_TGT, 'K', 0, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 6, 'K4 delta0 total = 6');
    assert_eq(hist_num_entries(r), 1, 'K4 delta0 single histogram entry');
    assert_eq(hist_count_value(r, 2.0), 6, 'K4 delta0 value 2.0 count = 6');
    passed = passed + 1;
    fprintf('PASS: delta-0 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-0 on K4: %s\n', e.message);
end

%% === Delta-1 on K4 ===

try
    r = dress_delta_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 12, 'K4 delta1 total = 12');
    assert_eq(hist_num_entries(r), 1, 'K4 delta1 single histogram entry');
    assert_eq(hist_count_value(r, 2.0), 12, 'K4 delta1 value 2.0 count = 12');
    passed = passed + 1;
    fprintf('PASS: delta-1 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on K4: %s\n', e.message);
end

%% === Delta-2 on K4 ===

try
    r = dress_delta_fit(4, K4_SRC, K4_TGT, 'K', 2, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 6, 'K4 delta2 total = 6');
    passed = passed + 1;
    fprintf('PASS: delta-2 on K4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-2 on K4: %s\n', e.message);
end

%% === k >= N (empty) ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 3, 'Epsilon', EPS);
    assert_eq(hist_total(r), 0, 'k=N total = 0');
    r2 = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 10, 'Epsilon', EPS);
    assert_eq(hist_total(r2), 0, 'k>N total = 0');
    passed = passed + 1;
    fprintf('PASS: k >= N (empty)\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: k >= N (empty): %s\n', e.message);
end

%% === Precompute flag ===

try
    r1 = dress_delta_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS, ...
                          'Precompute', false);
    r2 = dress_delta_fit(4, K4_SRC, K4_TGT, 'K', 1, 'Epsilon', EPS, ...
                          'Precompute', true);
    assert_true(hist_equal(r1, r2), 'precompute: same histogram');
    passed = passed + 1;
    fprintf('PASS: precompute flag\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: precompute flag: %s\n', e.message);
end

%% === Path P4 ===

try
    r = dress_delta_fit(4, P4_SRC, P4_TGT, 'K', 0, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 3, 'P4 delta0 total = 3');
    assert_true(hist_num_entries(r) >= 2, 'P4 at least 2 distinct values');
    passed = passed + 1;
    fprintf('PASS: path P4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: path P4: %s\n', e.message);
end

%% === Delta-1 on P4 ===

try
    r = dress_delta_fit(4, P4_SRC, P4_TGT, 'K', 1, 'Epsilon', EPS);
    total = hist_total(r);
    assert_eq(total, 6, 'P4 delta1 total = 6');
    passed = passed + 1;
    fprintf('PASS: delta-1 on P4\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: delta-1 on P4: %s\n', e.message);
end

%% === Length mismatch ===

try
    ok = false;
    try
        dress_delta_fit(3, int32([0; 1]), int32([1; 2; 2]), 'Epsilon', EPS);
    catch
        ok = true;
    end
    assert_true(ok, 'mismatched lengths throws error');
    passed = passed + 1;
    fprintf('PASS: length mismatch\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: length mismatch: %s\n', e.message);
end

%% === Multisets ===

try
    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 0, 'Epsilon', EPS);
    assert_true(~isfield(r, 'multisets'), 'multisets omitted by default');

    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 0, 'Epsilon', EPS, ...
                        'KeepMultisets', true);
    assert_eq(r.num_subgraphs, int32(1), 'num_subgraphs = 1');
    assert_true(isequal(size(r.multisets), [1, 3]), 'multisets size = 1x3');
    assert_true(all(abs(r.multisets(:) - 2.0) < EPS), 'multisets values all ~= 2.0');

    r = dress_delta_fit(3, K3_SRC, K3_TGT, 'K', 1, 'Epsilon', EPS, ...
                        'KeepMultisets', true);
    assert_eq(r.num_subgraphs, int32(3), 'delta1 num_subgraphs = 3');
    assert_true(isequal(size(r.multisets), [3, 3]), 'delta1 multisets size = 3x3');
    for row = 1:size(r.multisets, 1)
        row_values = r.multisets(row, :);
        assert_eq(sum(isnan(row_values)), 2, sprintf('row %d has 2 NaN', row));
        kept = row_values(~isnan(row_values));
        assert_eq(numel(kept), 1, sprintf('row %d has one kept edge', row));
        assert_true(abs(kept(1) - 2.0) < EPS, sprintf('row %d kept edge ~= 2.0', row));
    end

    passed = passed + 1;
    fprintf('PASS: multisets\n');
catch e
    failed = failed + 1;
    fprintf('FAIL: multisets: %s\n', e.message);
end

%% === Summary ===

fprintf('\n%d passed, %d failed.\n', passed, failed);
if failed > 0
    exit(1);
end
