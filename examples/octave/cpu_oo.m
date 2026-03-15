% cpu_oo.m — Prism vs K₃,₃ with DRESS (CPU, OO API, Octave)
%
% Demonstrates the persistent DRESS object: construct once, then
% fit, query virtual edges, and extract results without rebuilding.
%
% Run:
%   octave cpu_oo.m

% Prism (C₃ □ K₂): 6 vertices, 18 directed edges (0-based)
prism_s = int32([0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3]);
prism_t = int32([1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5]);

% K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s = int32([0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5]);
k33_t = int32([3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2]);

% Construct persistent graph objects
prism = DRESS(6, prism_s, prism_t);
k33   = DRESS(6, k33_s, k33_t);

% Fit
prism.fit();
k33.fit();

% Extract result snapshots
rp = prism.result();
rk = k33.result();

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

% Virtual edge queries
vp = prism.get(0, 4);   % 0-4 not in prism
vk = k33.get(0, 1);     % 0-1 not in K₃,₃
fprintf('\nVirtual edge prism(0,4) = %.6f\n', vp);
fprintf('Virtual edge k33(0,1)   = %.6f\n', vk);

% Cleanup
prism.close();
k33.close();
