#!/usr/bin/env Rscript
# =======================================================================
#  Tests for the dress R package
#
#  Run from the repo root after installing the package:
#      R CMD INSTALL r/
#      Rscript tests/r/test_dress.R
# =======================================================================

library(dress.graph)

passed <- 0L
failed <- 0L

assert <- function(desc, expr) {
  ok <- tryCatch(expr, error = function(e) FALSE)
  if (isTRUE(ok)) {
    passed <<- passed + 1L
    cat(sprintf("  PASS  %s\n", desc))
  } else {
    failed <<- failed + 1L
    cat(sprintf("  FAIL  %s\n", desc))
  }
}

assert_equal <- function(desc, a, b, tol = 1e-6) {
  assert(desc, isTRUE(all.equal(a, b, tolerance = tol)))
}

# =======================================================================
cat("== Version ==\n")
# =======================================================================

assert("dress_version returns a string", is.character(dress_version()))
assert("dress_version is non-empty",     nchar(dress_version()) > 0L)

# =======================================================================
cat("\n== Variant constants ==\n")
# =======================================================================

assert("DRESS_UNDIRECTED == 0", DRESS_UNDIRECTED == 0L)
assert("DRESS_DIRECTED   == 1", DRESS_DIRECTED   == 1L)
assert("DRESS_FORWARD    == 2", DRESS_FORWARD    == 2L)
assert("DRESS_BACKWARD   == 3", DRESS_BACKWARD   == 3L)

# =======================================================================
cat("\n== Triangle (unweighted, undirected) ==\n")
# =======================================================================

res <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L))

assert("returns a list",                     is.list(res))
assert("has 7 elements",                     length(res) == 7L)
assert("iterations > 0",                     res$iterations > 0L)
assert("delta >= 0",                         res$delta >= 0.0)
assert("edge_dress length == 3",             length(res$edge_dress) == 3L)
assert("edge_weight length == 3",            length(res$edge_weight) == 3L)
assert("node_dress length == 3",             length(res$node_dress) == 3L)
assert("sources length == 3",               length(res$sources) == 3L)
assert("targets length == 3",               length(res$targets) == 3L)

# All edges in a triangle should have equal dress
assert("triangle edges equal dress",
       max(res$edge_dress) - min(res$edge_dress) < 1e-6)

# All nodes in a triangle should have equal node_dress
assert("triangle nodes equal dress",
       max(res$node_dress) - min(res$node_dress) < 1e-6)

# Edge dress values should be positive
assert("edge_dress > 0", all(res$edge_dress > 0))

# =======================================================================
cat("\n== Path 0-1-2-3 (unweighted, undirected) ==\n")
# =======================================================================

res_path <- dress_fit(4L, c(0L, 1L, 2L), c(1L, 2L, 3L))

assert("path converges",                   res_path$iterations > 0L)
assert("path 3 edges",                     length(res_path$edge_dress) == 3L)
assert("path 4 nodes",                     length(res_path$node_dress) == 4L)
assert("path edge_dress >= 0",             all(res_path$edge_dress >= 0))
assert("path node_dress >= 0",             all(res_path$node_dress >= 0))

# Leaf edges should be symmetric: dress(0-1) == dress(2-3)
assert_equal("leaf edges symmetric",
             res_path$edge_dress[1], res_path$edge_dress[3])

# Interior and leaf edges should differ (not all equal)
assert("interior != leaf dress",
       abs(res_path$edge_dress[2] - res_path$edge_dress[1]) > 1e-6)

# =======================================================================
cat("\n== Triangle + pendant (4 vertices) ==\n")
# =======================================================================

# 0-1, 1-2, 0-2, 2-3
res_tp <- dress_fit(4L, c(0L, 1L, 0L, 2L), c(1L, 2L, 2L, 3L))

assert("tri+pendant 4 edges",             length(res_tp$edge_dress) == 4L)
assert("tri+pendant 4 nodes",             length(res_tp$node_dress) == 4L)
assert("tri+pendant converges",           res_tp$delta < 1e-6)

# Triangle edges should have higher dress than the pendant edge
tri_min <- min(res_tp$edge_dress[1:3])
pendant <- res_tp$edge_dress[4]
assert("triangle edge > pendant edge",    tri_min > pendant + 1e-9)

# =======================================================================
cat("\n== Weighted graph ==\n")
# =======================================================================

res_w <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L),
                   weights = c(1.0, 2.0, 3.0))

assert("weighted returns list",           is.list(res_w))
assert("weighted converges",              res_w$iterations > 0L)
assert("weighted edge_dress len 3",       length(res_w$edge_dress) == 3L)
assert("weighted edge_dress > 0",         all(res_w$edge_dress > 0))

# =======================================================================
cat("\n== Directed variants ==\n")
# =======================================================================

for (v in c(DRESS_UNDIRECTED, DRESS_DIRECTED, DRESS_FORWARD, DRESS_BACKWARD)) {
  label <- c("undirected", "directed", "forward", "backward")[v + 1L]
  r <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L), variant = v)
  assert(sprintf("variant %s converges", label), r$iterations > 0L)
  assert(sprintf("variant %s has 3 edges", label), length(r$edge_dress) == 3L)
}

# =======================================================================
cat("\n== Precompute intercepts ==\n")
# =======================================================================

res_pre <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L),
                     precompute_intercepts = TRUE)

assert("precompute converges",            res_pre$iterations > 0L)
assert_equal("precompute same as default",
             res_pre$edge_dress, res$edge_dress)

# =======================================================================
cat("\n== Convergence parameters ==\n")
# =======================================================================

# Very few iterations should give a larger delta
res_1 <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L), max_iterations = 1L)
assert("1 iteration executes",            res_1$iterations == 1L)

# Tight epsilon with many iterations should converge
res_tight <- dress_fit(3L, c(0L, 1L, 0L), c(1L, 2L, 2L),
                       max_iterations = 1000L, epsilon = 1e-12)
assert("tight epsilon fully converges",   res_tight$delta < 1e-12)

# =======================================================================
cat("\n== Star graph (1 hub + 5 leaves) ==\n")
# =======================================================================

res_star <- dress_fit(6L,
                      c(0L, 0L, 0L, 0L, 0L),
                      c(1L, 2L, 3L, 4L, 5L))

assert("star 5 edges",                    length(res_star$edge_dress) == 5L)
assert("star 6 nodes",                    length(res_star$node_dress) == 6L)

# All edges in a star should have equal dress (by symmetry)
assert("star edges equal dress",
       max(res_star$edge_dress) - min(res_star$edge_dress) < 1e-6)

# Leaf nodes should have equal node_dress
leaf_norms <- res_star$node_dress[2:6]
assert("star leaf nodes equal",
       max(leaf_norms) - min(leaf_norms) < 1e-6)

# =======================================================================
cat("\n== Disconnected pair of edges ==\n")
# =======================================================================

# Two disjoint edges: 0-1 and 2-3
res_disc <- dress_fit(4L, c(0L, 2L), c(1L, 3L))

assert("disconnected 2 edges",            length(res_disc$edge_dress) == 2L)
assert("disconnected 4 nodes",            length(res_disc$node_dress) == 4L)

# Both isolated edges should be symmetric
assert_equal("disconnected edges equal",
             res_disc$edge_dress[1], res_disc$edge_dress[2])

# =======================================================================
cat("\n== Complete graph K4 ==\n")
# =======================================================================

# K4: all 6 edges
k4_s <- c(0L, 0L, 0L, 1L, 1L, 2L)
k4_t <- c(1L, 2L, 3L, 2L, 3L, 3L)
res_k4 <- dress_fit(4L, k4_s, k4_t)

assert("K4 has 6 edges",                  length(res_k4$edge_dress) == 6L)
assert("K4 has 4 nodes",                  length(res_k4$node_dress) == 4L)

# All edges in K4 should be equal (by symmetry)
assert("K4 edges equal dress",
       max(res_k4$edge_dress) - min(res_k4$edge_dress) < 1e-6)

# All nodes in K4 should be equal
assert("K4 nodes equal dress",
       max(res_k4$node_dress) - min(res_k4$node_dress) < 1e-6)

# =======================================================================
cat("\n== Summary ==\n")
# =======================================================================

total <- passed + failed
cat(sprintf("\n%d / %d tests passed", passed, total))
if (failed > 0L) {
  cat(sprintf("  (%d FAILED)\n", failed))
  quit(status = 1L)
} else {
  cat("  — all OK\n")
}
