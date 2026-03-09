#!/usr/bin/env python3
"""cpu_nx.py — Prism vs K₃,₃ with DRESS via NetworkX (CPU)

Run:
    python cpu_nx.py
"""
import networkx as nx
from dress.networkx import dress_graph

# Prism graph (C₃ □ K₂)
prism = nx.circular_ladder_graph(3)

# K₃,₃ (complete bipartite)
k33 = nx.complete_bipartite_graph(3, 3)

rp = dress_graph(prism)
rk = dress_graph(k33)

fp = sorted(rp.edge_dress)
fk = sorted(rk.edge_dress)

print("Prism:", [f"{v:.6f}" for v in fp])
print("K3,3: ", [f"{v:.6f}" for v in fk])
print("Distinguished:", fp != fk)
