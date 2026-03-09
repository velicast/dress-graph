# ========================================================================
#  dress MPI — R interface to MPI-distributed DRESS
# ========================================================================
#
# Usage:
#   library(dress.graph)
#   Rmpi::mpi.bcast.cmd(cmd = {library(dress.graph)})
#   result <- mpi$delta_dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L), k = 1L)
#
# Requires the dress.graph package built with DRESS_MPI support and an
# MPI-capable R environment (pbdMPI, Rmpi, or manual comm handle).

#' MPI-distributed DRESS functions.
#'
#' An environment containing MPI-distributed versions of
#' \code{delta_dress_fit}.  Switch from CPU to MPI by prefixing calls
#' with \code{mpi$}.
#'
#' @details
#' The \code{mpi} environment provides:
#' \describe{
#'   \item{\code{mpi$delta_dress_fit(...)}}{MPI-distributed
#'         \code{\link{delta_dress_fit}} (CPU backend).
#'         Same arguments plus \code{comm_f}.}
#'   \item{\code{mpi$cuda$delta_dress_fit(...)}}{MPI-distributed
#'         \code{\link{delta_dress_fit}} (CUDA backend).
#'         Each rank runs GPU-accelerated DRESS.}
#' }
#'
#' MPI support requires rebuilding the package with \code{DRESS_MPI}
#' (auto-detected when \code{mpicc} is available).
#'
#' @examples
#' \dontrun{
#'   # CPU
#'   r1 <- delta_dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L), k = 1L)
#'
#'   # MPI -- same signature, distributed
#'   r2 <- mpi$delta_dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L), k = 1L)
#'
#'   # MPI + CUDA
#'   r3 <- mpi$cuda$delta_dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L), k = 1L)
#' }
#' @export
mpi <- new.env(parent = emptyenv())

#' MPI-distributed Delta-k-DRESS histogram
#'
#' Compute the Delta-k-DRESS distribution using MPI.
#' All MPI logic (stride partitioning + Allreduce) runs in C.
#'
#' @param n_vertices Integer. Number of vertices (vertex ids in 0..N-1).
#' @param sources Integer vector — edge sources (0-based).
#' @param targets Integer vector — edge targets (0-based).
#' @param weights Numeric vector or NULL.
#' @param k Integer. Deletion depth.
#' @param variant Graph variant (default 0 = undirected).
#' @param max_iterations Max DRESS iterations per subgraph.
#' @param epsilon Convergence tolerance and bin width.
#' @param precompute Logical.
#' @param keep_multisets Logical.
#' @param comm_f Integer. Fortran MPI communicator handle.
#'   If NULL, attempts to get MPI_COMM_WORLD from pbdMPI.
#'
#' @return A list with histogram, hist_size, and optionally multisets.
#' @export
mpi$delta_dress_fit <- function(n_vertices,
                                sources,
                                targets,
                                weights          = NULL,
                                k                = 0L,
                                variant          = 0L,
                                max_iterations   = 100L,
                                epsilon          = 1e-6,
                                precompute       = FALSE,
                                keep_multisets   = FALSE,
                                comm_f           = NULL) {

  n_vertices     <- as.integer(n_vertices)
  sources        <- as.integer(sources)
  targets        <- as.integer(targets)
  if (!is.null(weights)) weights <- as.double(weights)
  k              <- as.integer(k)
  variant        <- as.integer(variant)
  max_iterations <- as.integer(max_iterations)
  epsilon        <- as.double(epsilon)
  precompute     <- as.integer(precompute)
  keep_multisets <- as.integer(keep_multisets)

  stopifnot(length(sources) == length(targets))

  # Resolve communicator handle
  if (is.null(comm_f)) {
    if (requireNamespace("pbdMPI", quietly = TRUE)) {
      comm_f <- pbdMPI::spmd.comm.c2f(.pbd_env$SPMD.CT$comm)
    } else {
      stop("Provide 'comm_f' (Fortran MPI comm handle) or install pbdMPI.")
    }
  }
  comm_f <- as.integer(comm_f)

  .Call(C_delta_dress_fit_mpi,
        n_vertices,
        sources,
        targets,
        weights,
        k,
        variant,
        max_iterations,
        epsilon,
        precompute,
        keep_multisets,
        comm_f)
}

# ---- MPI + CUDA environment ---------------------------------------------

mpi$cuda <- new.env(parent = emptyenv())

#' MPI+CUDA distributed Delta-k-DRESS histogram
#'
#' Same as \code{mpi$delta_dress_fit} but each rank runs GPU-accelerated DRESS.
#'
#' @inheritParams mpi$delta_dress_fit
#' @export
mpi$cuda$delta_dress_fit <- function(n_vertices,
                                     sources,
                                     targets,
                                     weights          = NULL,
                                     k                = 0L,
                                     variant          = 0L,
                                     max_iterations   = 100L,
                                     epsilon          = 1e-6,
                                     precompute       = FALSE,
                                     keep_multisets   = FALSE,
                                     comm_f           = NULL) {

  n_vertices     <- as.integer(n_vertices)
  sources        <- as.integer(sources)
  targets        <- as.integer(targets)
  if (!is.null(weights)) weights <- as.double(weights)
  k              <- as.integer(k)
  variant        <- as.integer(variant)
  max_iterations <- as.integer(max_iterations)
  epsilon        <- as.double(epsilon)
  precompute     <- as.integer(precompute)
  keep_multisets <- as.integer(keep_multisets)

  stopifnot(length(sources) == length(targets))

  if (is.null(comm_f)) {
    if (requireNamespace("pbdMPI", quietly = TRUE)) {
      comm_f <- pbdMPI::spmd.comm.c2f(.pbd_env$SPMD.CT$comm)
    } else {
      stop("Provide 'comm_f' (Fortran MPI comm handle) or install pbdMPI.")
    }
  }
  comm_f <- as.integer(comm_f)

  .Call(C_delta_dress_fit_mpi_cuda,
        n_vertices,
        sources,
        targets,
        weights,
        k,
        variant,
        max_iterations,
        epsilon,
        precompute,
        keep_multisets,
        comm_f)
}
