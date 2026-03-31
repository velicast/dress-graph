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

hist_total <- function(r) sum(r$histogram$count)

hist_count_value <- function(r, value, tol = 1e-9) {
  idx <- abs(r$histogram$value - value) < tol
  if (!any(idx)) return(0)
  sum(r$histogram$count[idx])
}

# =======================================================================
cat("== Histogram entries ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = 1e-3)
assert_equal("single entry with eps=1e-3", nrow(r$histogram), 1L)

r2 <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = 1e-6)
assert_equal("single entry with eps=1e-6", nrow(r2$histogram), 1L)

assert("same exact histogram across eps values",
  isTRUE(all.equal(r$histogram, r2$histogram)))

# =======================================================================
cat("== Weighted histogram entries ==\n")
# =======================================================================

rw <- dress_delta_fit(3L, K3_SRC, K3_TGT,
                      weights = c(1.0, 10.0, 1.0),
                      k = 0L, epsilon = 1e-3)
assert("weighted histogram has multiple entries", nrow(rw$histogram) > 1L)
assert_equal("weighted K3 delta0 total = 3", hist_total(rw), 3)

# =======================================================================
cat("\n== Return type ==\n")
# =======================================================================

assert("returns a list", is.list(r))
assert("has histogram field", !is.null(r$histogram))
assert("histogram is a data frame", is.data.frame(r$histogram))
assert("histogram has value/count columns",
  identical(names(r$histogram), c("value", "count")))
assert("does not expose hist_size", is.null(r$hist_size))

# =======================================================================
cat("\n== Delta-0 on K3 ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = EPS)
assert_equal("total = 3 edges",  hist_total(r), 3)
assert_equal("single histogram entry", nrow(r$histogram), 1L)
assert_equal("value 2.0 count = 3", hist_count_value(r, 2.0), 3)

# =======================================================================
cat("\n== Delta-1 on K3 ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 1L, epsilon = EPS)
assert_equal("total = 3 (C(3,1)*1)", hist_total(r), 3)

# =======================================================================
cat("\n== Delta-2 on K3 ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 2L, epsilon = EPS)
assert_equal("total = 0", hist_total(r), 0)

# =======================================================================
cat("\n== Delta-0 on K4 ==\n")
# =======================================================================

r <- dress_delta_fit(4L, K4_SRC, K4_TGT, k = 0L, epsilon = EPS)
assert_equal("total = 6", hist_total(r), 6)
assert_equal("single histogram entry", nrow(r$histogram), 1L)
assert_equal("value 2.0 count = 6", hist_count_value(r, 2.0), 6)

# =======================================================================
cat("\n== Delta-1 on K4 ==\n")
# =======================================================================

r <- dress_delta_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS)
assert_equal("total = 12 (C(4,1)*3)", hist_total(r), 12)
assert_equal("single histogram entry", nrow(r$histogram), 1L)
assert_equal("value 2.0 count = 12", hist_count_value(r, 2.0), 12)

# =======================================================================
cat("\n== Delta-2 on K4 ==\n")
# =======================================================================

r <- dress_delta_fit(4L, K4_SRC, K4_TGT, k = 2L, epsilon = EPS)
assert_equal("total = 6 (C(4,2)*1)", hist_total(r), 6)

# =======================================================================
cat("\n== k >= N (empty) ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 3L, epsilon = EPS)
assert_equal("k=N total = 0",  hist_total(r), 0)

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 10L, epsilon = EPS)
assert_equal("k>N total = 0",  hist_total(r), 0)

# =======================================================================
cat("\n== Precompute flag ==\n")
# =======================================================================

r1 <- dress_delta_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS,
                       precompute = FALSE)
r2 <- dress_delta_fit(4L, K4_SRC, K4_TGT, k = 1L, epsilon = EPS,
                       precompute = TRUE)
assert("precompute: same histogram",
       isTRUE(all.equal(r1$histogram, r2$histogram)))

# =======================================================================
cat("\n== Path P4 ==\n")
# =======================================================================

r <- dress_delta_fit(4L, P4_SRC, P4_TGT, k = 0L, epsilon = EPS)
assert_equal("P4 total = 3", hist_total(r), 3)

assert("P4 ≥ 2 distinct values", nrow(r$histogram) >= 2L)

# =======================================================================
cat("\n== Delta-1 on P4 ==\n")
# =======================================================================

r <- dress_delta_fit(4L, P4_SRC, P4_TGT, k = 1L, epsilon = EPS)
assert_equal("P4 delta1 total = 6", hist_total(r), 6)

# =======================================================================
cat("\n== Length mismatch ==\n")
# =======================================================================

ok <- tryCatch({
  dress_delta_fit(3L, c(0L, 1L), c(1L, 2L, 2L))
  FALSE
}, error = function(e) TRUE)
assert("mismatched lengths throws error", ok)

# =======================================================================
cat("\n== Multisets ==\n")
# =======================================================================

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = EPS)
assert("multisets omitted by default", is.null(r$multisets))

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 0L, epsilon = EPS,
                     keep_multisets = TRUE)
assert_equal("num_subgraphs = 1", r$num_subgraphs, 1)
assert("multisets dim = 1 x 3", identical(dim(r$multisets), c(1L, 3L)))
assert("multisets values all ~= 2.0", all(abs(r$multisets - 2.0) < EPS))

r <- dress_delta_fit(3L, K3_SRC, K3_TGT, k = 1L, epsilon = EPS,
                     keep_multisets = TRUE)
assert_equal("delta1 num_subgraphs = 3", r$num_subgraphs, 3)
assert("delta1 multisets dim = 3 x 3", identical(dim(r$multisets), c(3L, 3L)))
for (row in seq_len(nrow(r$multisets))) {
  row_values <- r$multisets[row, ]
  assert(sprintf("row %d has 2 NaN", row), sum(is.nan(row_values)) == 2L)
  kept <- row_values[!is.nan(row_values)]
  assert(sprintf("row %d has one kept edge", row), length(kept) == 1L)
  assert(sprintf("row %d kept edge ~= 2.0", row), abs(kept[1] - 2.0) < EPS)
}

# =======================================================================
#  Summary
# =======================================================================

cat(sprintf("\n%d passed, %d failed.\n", passed, failed))
if (failed > 0L) quit(status = 1L)
