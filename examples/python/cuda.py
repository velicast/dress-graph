#!/usr/bin/env python3
"""cuda.py — Prism vs K₃,₃ with DRESS (CUDA)

Run:
    python cuda.py
"""
from dress.cuda import dress_fit

# Prism (C₃ □ K₂): 6 vertices, 18 directed edges (both directions)
prism_s = [0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3]
prism_t = [1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5]

# K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s = [0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5]
k33_t = [3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2]

rp = dress_fit(6, prism_s, prism_t)
rk = dress_fit(6, k33_s, k33_t)

fp = sorted(rp.edge_dress)
fk = sorted(rk.edge_dress)

print("Prism:", [f"{v:.6f}" for v in fp])
print("K3,3: ", [f"{v:.6f}" for v in fk])
print("Distinguished:", fp != fk)
