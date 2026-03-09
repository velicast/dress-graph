# cuda.R — Prism vs K₃,₃ with DRESS (CUDA)
#
# Run:
#   Rscript cuda.R
library(dress.graph)

# Prism (C₃ □ K₂): 6 vertices, 18 directed edges (0-based)
prism_s <- c(0L,1L,1L,2L,2L,0L,0L,3L,1L,4L,2L,5L,3L,4L,4L,5L,5L,3L)
prism_t <- c(1L,0L,2L,1L,0L,2L,3L,0L,4L,1L,5L,2L,4L,3L,5L,4L,3L,5L)

# K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
k33_s <- c(0L,3L,0L,4L,0L,5L,1L,3L,1L,4L,1L,5L,2L,3L,2L,4L,2L,5L)
k33_t <- c(3L,0L,4L,0L,5L,0L,3L,1L,4L,1L,5L,1L,3L,2L,4L,2L,5L,2L)

rp <- cuda$dress_fit(6L, prism_s, prism_t)
rk <- cuda$dress_fit(6L, k33_s, k33_t)

fp <- sort(rp$edge_dress)
fk <- sort(rk$edge_dress)

cat("Prism:", round(fp, 6), "\n")
cat("K3,3: ", round(fk, 6), "\n")
cat("Distinguished:", !identical(fp, fk), "\n")
