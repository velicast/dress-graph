#!/usr/bin/env python3
"""cpu_oo.py — Prism vs K₃,₃ with DRESS (CPU, OO API)

Demonstrates the persistent DRESS object: construct once, then
fit, query virtual edges, and extract results without rebuilding.

Run:
    python cpu_oo.py
"""
from dress import DRESS

# Prism (C₃ □ K₂): 6 vertices, 18 directed edges (both directions)
prism_s = [0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3]
prism_t = [1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5]

# K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s = [0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5]
k33_t = [3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2]

# Construct persistent graph objects
prism = DRESS(6, prism_s, prism_t)
k33   = DRESS(6, k33_s, k33_t)

# Fit
prism.fit()
k33.fit()

# Sorted edge DRESS fingerprints
fp = sorted(prism.dress_values)
fk = sorted(k33.dress_values)

print("Prism:", [f"{v:.6f}" for v in fp])
print("K3,3: ", [f"{v:.6f}" for v in fk])
print("Distinguished:", fp != fk)

# Virtual edge queries — edges that don't exist in the graph
vp = prism.get(0, 4)  # 0-4 not in prism
vk = k33.get(0, 1)    # 0-1 not in K₃,₃
print(f"\nVirtual edge prism(0,4) = {vp:.6f}")
print(f"Virtual edge k33(0,1)   = {vk:.6f}")
