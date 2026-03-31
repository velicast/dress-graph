# mpi_omp_oo.R — Rook vs Shrikhande with Δ¹-DRESS (MPI + OpenMP, OO API)
#
# Demonstrates the persistent DRESS object with GPU fitting and
# MPI-distributed delta_fit.
#
# Run:
#   mpirun -np 4 Rscript mpi_omp_oo.R
library(dress.graph)
library(pbdMPI)

# Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges (0-based)
rook_s <- c(0L,1L,0L,4L,0L,2L,0L,8L,0L,3L,0L,12L,1L,5L,1L,2L,1L,9L,1L,3L,1L,13L,2L,6L,2L,10L,2L,3L,2L,14L,3L,7L,3L,11L,3L,15L,4L,5L,4L,6L,4L,8L,4L,7L,4L,12L,5L,6L,5L,9L,5L,7L,5L,13L,6L,10L,6L,7L,6L,14L,7L,11L,7L,15L,8L,9L,8L,10L,8L,11L,8L,12L,9L,10L,9L,11L,9L,13L,10L,11L,10L,14L,11L,15L,12L,13L,12L,14L,12L,15L,13L,14L,13L,15L,14L,15L)
rook_t <- c(1L,0L,4L,0L,2L,0L,8L,0L,3L,0L,12L,0L,5L,1L,2L,1L,9L,1L,3L,1L,13L,1L,6L,2L,10L,2L,3L,2L,14L,2L,7L,3L,11L,3L,15L,3L,5L,4L,6L,4L,8L,4L,7L,4L,12L,4L,6L,5L,9L,5L,7L,5L,13L,5L,10L,6L,7L,6L,14L,6L,11L,7L,15L,7L,9L,8L,10L,8L,11L,8L,12L,8L,10L,9L,11L,9L,13L,9L,11L,10L,14L,10L,15L,11L,13L,12L,14L,12L,15L,12L,14L,13L,15L,13L,15L,14L)

# Shrikhande — 16 vertices, 96 directed edges
shri_s <- c(0L,4L,0L,12L,0L,1L,0L,3L,0L,5L,0L,15L,1L,5L,1L,13L,1L,2L,1L,6L,1L,12L,2L,6L,2L,14L,2L,3L,2L,7L,2L,13L,3L,7L,3L,15L,3L,4L,3L,14L,4L,8L,4L,5L,4L,7L,4L,9L,5L,9L,5L,6L,5L,10L,6L,10L,6L,7L,6L,11L,7L,11L,7L,8L,8L,12L,8L,9L,8L,11L,8L,13L,9L,13L,9L,10L,9L,14L,10L,14L,10L,11L,10L,15L,11L,15L,11L,12L,12L,13L,12L,15L,13L,14L,14L,15L)
shri_t <- c(4L,0L,12L,0L,1L,0L,3L,0L,5L,0L,15L,0L,5L,1L,13L,1L,2L,1L,6L,1L,12L,1L,6L,2L,14L,2L,3L,2L,7L,2L,13L,2L,7L,3L,15L,3L,4L,3L,14L,3L,8L,4L,5L,4L,7L,4L,9L,4L,9L,5L,6L,5L,10L,5L,10L,6L,7L,6L,11L,6L,11L,7L,8L,7L,12L,8L,9L,8L,11L,8L,13L,8L,13L,9L,10L,9L,14L,9L,14L,10L,11L,10L,15L,10L,15L,11L,12L,11L,13L,12L,15L,12L,14L,13L,15L,14L)

# Construct persistent graph objects (MPI+OpenMP)
rook <- mpi$omp$DRESS(16L, rook_s, rook_t)
shri <- mpi$omp$DRESS(16L, shri_s, shri_t)


# MPI+OpenMP distributed Δ¹-DRESS
dr <- rook$delta_fit(k = 1L, keep_multisets = TRUE)
ds <- shri$delta_fit(k = 1L, keep_multisets = TRUE)

if (comm.rank() == 0L) {
  cat(sprintf("Rook:       %d exact values, %d subgraphs\n", nrow(dr$histogram), dr$num_subgraphs))
  cat(sprintf("Shrikhande: %d exact values, %d subgraphs\n", nrow(ds$histogram), ds$num_subgraphs))
  cat("Histograms differ: ", !identical(dr$histogram, ds$histogram), "\n")

  canonicalize <- function(ms) {
    s <- t(apply(ms, 1, sort, na.last = TRUE))
    s[do.call(order, as.data.frame(s)), ]
  }

  cr <- canonicalize(dr$multisets)
  cs <- canonicalize(ds$multisets)
  ms_same <- identical(dim(cr), dim(cs)) &&
             all(cr == cs | (is.nan(cr) & is.nan(cs)), na.rm = TRUE)
  cat("Multisets differ:  ", !ms_same, "\n")
}

# Cleanup
rook$close()
shri$close()

finalize()
