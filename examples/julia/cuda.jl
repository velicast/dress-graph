# cuda.jl — Prism vs K₃,₃ with DRESS (CUDA)
#
# Run:
#   julia cuda.jl
using DRESS.CUDA

# Prism (C₃ □ K₂): 6 vertices, 18 directed edges (0-based)
prism_s = [0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3]
prism_t = [1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5]

# K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s = [0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5]
k33_t = [3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2]

rp = fit(6, prism_s, prism_t)
rk = fit(6, k33_s, k33_t)

fp = sort(rp.edge_dress)
fk = sort(rk.edge_dress)

println("Prism: ", round.(fp; digits=6))
println("K3,3:  ", round.(fk; digits=6))
println("Distinguished: ", fp != fk)
