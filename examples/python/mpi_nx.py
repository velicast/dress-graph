#!/usr/bin/env python3
"""mpi_nx.py — Rook vs Shrikhande with Δ¹-DRESS via NetworkX (MPI, CPU)

Both are SRG(16,6,2,2) — indistinguishable by Δ⁰-DRESS.
Δ¹-DRESS (k=1) separates them.
Keeps multisets and compares them to guarantee distinguishability.

Run:
    mpirun -np 4 python mpi_nx.py
"""
from mpi4py import MPI
import numpy as np
import networkx as nx
from dress.mpi.networkx import delta_fit

# Rook L₂(4) = K₄ □ K₄
rook = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))

# Shrikhande graph
shrikhande = nx.Graph()
shrikhande.add_nodes_from(range(16))
shrikhande_edges = [
    (0,1),(0,3),(0,4),(0,5),(0,12),(0,15),
    (1,2),(1,5),(1,6),(1,12),(1,13),
    (2,3),(2,6),(2,7),(2,13),(2,14),
    (3,4),(3,7),(3,14),(3,15),
    (4,5),(4,7),(4,8),(4,9),
    (5,6),(5,9),(5,10),
    (6,7),(6,10),(6,11),
    (7,8),(7,11),
    (8,9),(8,11),(8,12),(8,13),
    (9,10),(9,13),(9,14),
    (10,11),(10,14),(10,15),
    (11,12),(11,15),
    (12,13),(12,15),
    (13,14),
    (14,15),
]
shrikhande.add_edges_from(shrikhande_edges)

dr = delta_fit(rook, k=1, keep_multisets=True)
ds = delta_fit(shrikhande, k=1, keep_multisets=True)

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
