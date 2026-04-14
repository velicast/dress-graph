% dress-graph MATLAB package
% Version 0.8.2 (2026-04-10)
%
% DRESS: A Continuous Framework for Structural Graph Refinement.
%   fit       - Run DRESS on an edge list and return edge and vertex values.
%   delta_fit - Compute exact sparse Delta-k-DRESS histogram entries.
%   DRESS           - Persistent graph object wrapper.
%   dress_to_table  - Convert result structs to MATLAB tables.
%   dress_build     - Build the MATLAB MEX wrappers.
%
% CUDA namespace:
%   cuda.fit       - GPU-accelerated DRESS fitting.
%   cuda.delta_fit - GPU-accelerated Delta-k-DRESS.
%   cuda.DRESS           - Persistent GPU-backed graph object.
%
% MPI namespace:
%   mpi.delta_fit      - MPI-distributed Delta-k-DRESS.
%   mpi.DRESS                - Persistent MPI graph object.
%   mpi.cuda.delta_fit - MPI+CUDA Delta-k-DRESS.
%   mpi.cuda.DRESS           - Persistent MPI+CUDA graph object.
%
% Author: Eduar Castrillo Velilla
% Repository: https://github.com/velicast/dress-graph
% Documentation: https://velicast.github.io/dress-graph/