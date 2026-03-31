#!/usr/bin/env python3
"""mpi_cuda.py — Rook vs Shrikhande with Δ¹-DRESS (MPI + CUDA)

Both are SRG(16,6,2,2) — indistinguishable by Δ⁰-DRESS.
Δ¹-DRESS (k=1) separates them.
Keeps multisets and compares them to guarantee distinguishability.

Run:
    mpirun -np 4 python mpi_cuda.py
"""
from mpi4py import MPI
import numpy as np
from dress.mpi.cuda import delta_fit

# Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges
rook_s = [0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15]
rook_t = [1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14]

# Shrikhande — 16 vertices, 96 directed edges
shri_s = [0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15]
shri_t = [4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14]

dr = delta_fit(16, rook_s, rook_t, k=1, keep_multisets=True)
ds = delta_fit(16, shri_s, shri_t, k=1, keep_multisets=True)

if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Rook:       {len(dr.histogram)} bins, {dr.num_subgraphs} subgraphs")
    print(f"Shrikhande: {len(ds.histogram)} bins, {ds.num_subgraphs} subgraphs")
    print(f"Histograms differ:  {dr.histogram != ds.histogram}")

    def canonicalize(ms):
        s = np.sort(ms, axis=1)
        return s[np.lexsort(s.T[::-1])]

    cr = canonicalize(dr.multisets)
    cs = canonicalize(ds.multisets)
    ms_same = cr.shape == cs.shape and np.array_equal(cr, cs)
    print(f"Multisets differ:   {not ms_same}")
