#!/usr/bin/env Rscript
# =======================================================================
#  Tests for the delta-k-DRESS R wrapper
#
#  Run from the repo root after installing the package:
#      R CMD INSTALL r/
#      Rscript tests/r/test_delta_dress.R
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

# ── graph helpers ───────────────────────────────────────────────────

K3_SRC <- c(0L, 1L, 0L)
K3_TGT <- c(1L, 2L, 2L)
K4_SRC <- c(0L, 0L, 0L, 1L, 1L, 2L)
K4_TGT <- c(1L, 2L, 3L, 2L, 3L, 3L)
P4_SRC <- c(0L, 1L, 2L)
P4_TGT <- c(1L, 2L, 3L)

EPS <- 1e-3

hist_total <- function(r) sum(r$histogram)

# =======================================================================
cat("== Histogram size ==\n")
# =======================================================================

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = 1e-3)
assert_equal("hist_size with eps=1e-3", r$hist_size, 2001L)

r2 <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = 1e-6)
assert_equal("hist_size with eps=1e-6", r2$hist_size, 2000001L)

assert("histogram length == hist_size", length(r$histogram) == r$hist_size)

# =======================================================================
cat("== Weighted histogram size ==\n")
# =======================================================================

rw <- delta_dress_fit(3L, K3_SRC, K3_TGT,
                      weights = c(1.0, 10.0, 1.0),
                      k = 0L, epsilon = 1e-3)
assert("weighted hist_size > 2001", rw$hist_size > 2001L)
assert("weighted histogram length == hist_size",
       length(rw$histogram) == rw$hist_size)
assert_equal("weighted K3 delta0 total = 3", hist_total(rw), 3)

# =======================================================================
cat("\n== Return type ==\n")
# =======================================================================

assert("returns a list", is.list(r))
assert("has histogram field",  !is.null(r$histogram))
assert("has hist_size field",  !is.null(r$hist_size))

# =======================================================================
cat("\n== Delta-0 on K3 ==\n")
# =======================================================================

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = EPS)
assert_equal("total = 3 edges",  hist_total(r), 3)
assert("top bin > 0",            r$histogram[r$hist_size] > 0)

nonzero <- sum(r$histogram > 0)
assert_equal("single non-zero bin", nonzero, 1L)

# =======================================================================
cat("\n== Delta-1 on K3 ==\n")
# =======================================================================

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 1L, epsilon = EPS)
assert_equal("total = 3 (C(3,1)*1)", hist_total(r), 3)

# =======================================================================
cat("\n== Delta-2 on K3 ==\n")
# =======================================================================

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 2L, epsilon = EPS)
assert_equal("total = 0", hist_total(r), 0)

# =======================================================================
cat("\n== Delta-0 on K4 ==\n")
# =======================================================================

r <- delta_dress_fit(4L, K4_SRC, K4_TGT, k = 0L, epsilon = EPS)
assert_equal("total = 6", hist_total(r), 6)
assert_equal("top bin = 6", r$histogram[r$hist_size], 6)

# =======================================================================
cat("\n== Delta-1 on K4 ==\n")
# =======================================================================

r <- delta_dress_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS)
assert_equal("total = 12 (C(4,1)*3)", hist_total(r), 12)
assert_equal("top bin = 12",          r$histogram[r$hist_size], 12)

# =======================================================================
cat("\n== Delta-2 on K4 ==\n")
# =======================================================================

r <- delta_dress_fit(4L, K4_SRC, K4_TGT, k = 2L, epsilon = EPS)
assert_equal("total = 6 (C(4,2)*1)", hist_total(r), 6)

# =======================================================================
cat("\n== k >= N (empty) ==\n")
# =======================================================================

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 3L, epsilon = EPS)
assert_equal("k=N total = 0",  hist_total(r), 0)

r <- delta_dress_fit(3L, K3_SRC, K3_TGT, k = 10L, epsilon = EPS)
assert_equal("k>N total = 0",  hist_total(r), 0)

# =======================================================================
cat("\n== Precompute flag ==\n")
# =======================================================================

r1 <- delta_dress_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS,
                       precompute = FALSE)
r2 <- delta_dress_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS,
                       precompute = TRUE)
assert_equal("precompute: same hist_size",  r1$hist_size, r2$hist_size)
assert("precompute: same histogram",
       isTRUE(all.equal(r1$histogram, r2$histogram)))

# =======================================================================
cat("\n== Path P4 ==\n")
# =======================================================================

r <- delta_dress_fit(4L, P4_SRC, P4_TGT, k = 0L, epsilon = EPS)
assert_equal("P4 total = 3", hist_total(r), 3)

nonzero <- sum(r$histogram > 0)
assert("P4 ≥ 2 distinct bins", nonzero >= 2L)

# =======================================================================
cat("\n== Delta-1 on P4 ==\n")
# =======================================================================

r <- delta_dress_fit(4L, P4_SRC, P4_TGT, k = 1L, epsilon = EPS)
assert_equal("P4 delta1 total = 6", hist_total(r), 6)

# =======================================================================
cat("\n== Length mismatch ==\n")
# =======================================================================

ok <- tryCatch({
  delta_dress_fit(3L, c(0L, 1L), c(1L, 2L, 2L))
  FALSE
}, error = function(e) TRUE)
assert("mismatched lengths throws error", ok)

# =======================================================================
#  Summary
# =======================================================================

cat(sprintf("\n%d passed, %d failed.\n", passed, failed))
if (failed > 0L) quit(status = 1L)
