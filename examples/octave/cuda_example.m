% cuda_example.m — Prism vs K₃,₃ with DRESS (CUDA, Octave)
%
% Run:
%   octave cuda_example.m

% Prism (C₃ □ K₂): 6 vertices, 18 directed edges (0-based)
prism_s = int32([0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3]);
prism_t = int32([1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5]);

% K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s = int32([0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5]);
k33_t = int32([3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2]);

rp = cuda.dress_fit(6, prism_s, prism_t);
rk = cuda.dress_fit(6, k33_s, k33_t);

fp = sort(rp.edge_dress);
fk = sort(rk.edge_dress);

fprintf('Prism: [');
fprintf('%.6f ', fp);
fprintf(']\n');
fprintf('K3,3:  [');
fprintf('%.6f ', fk);
fprintf(']\n');

if isequal(fp, fk)
    fprintf('Distinguished: false\n');
else
    fprintf('Distinguished: true\n');
end
