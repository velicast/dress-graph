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

#' Create a persistent MPI DRESS graph object
#'
#' Same as \code{DRESS()} but \code{$fit()} uses the CPU and
#' \code{$delta_fit()} uses MPI distribution.
#'
#' @inheritParams DRESS
#' @return An environment (class \code{"DRESS"}) with methods:
#' \describe{
#'   \item{\code{$fit(max_iterations, epsilon)}}{Fit (CPU).}
#'   \item{\code{$delta_fit(k, ...)}}{MPI-distributed Δ^k-DRESS.}
#'   \item{\code{$get(u, v, ...)}}{Query edge value.}
#'   \item{\code{$result()}}{Extract current results.}
#'   \item{\code{$close()}}{Free C graph.}
#' }
#' @export
mpi$DRESS <- function(n_vertices,
                      sources,
                      targets,
                      weights               = NULL,
                      variant               = DRESS_UNDIRECTED,
                      precompute_intercepts = FALSE) {

  n_vertices <- as.integer(n_vertices)
  sources    <- as.integer(sources)
  targets    <- as.integer(targets)
  variant    <- as.integer(variant)
  precompute <- as.integer(precompute_intercepts)

  stopifnot(length(sources) == length(targets))
  stopifnot(n_vertices >= 1L)
  stopifnot(variant >= 0L && variant <= 3L)

  if (!is.null(weights)) {
    weights <- as.double(weights)
    stopifnot(length(weights) == length(sources))
  }

  ptr <- .Call(C_dress_init,
               n_vertices, sources, targets, weights,
               variant, precompute)

  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr

  self$fit <- function(max_iterations = 100L, epsilon = 1e-6) {
    .Call(C_dress_fit_obj,
          self$.ptr,
          as.integer(max_iterations),
          as.double(epsilon))
  }

  self$delta_fit <- function(k                = 0L,
                             max_iterations   = 100L,
                             epsilon          = 1e-6,
                             keep_multisets   = FALSE,
                             comm_f           = NULL) {
    if (is.null(comm_f)) {
      if (requireNamespace("pbdMPI", quietly = TRUE)) {
        comm_f <- pbdMPI::spmd.comm.c2f(.pbd_env$SPMD.CT$comm)
      } else {
        stop("Provide 'comm_f' (Fortran MPI comm handle) or install pbdMPI.")
      }
    }
    mpi$delta_dress_fit(n_vertices, sources, targets,
                        weights          = weights,
                        k                = as.integer(k),
                        variant          = variant,
                        max_iterations   = as.integer(max_iterations),
                        epsilon          = as.double(epsilon),
                        precompute       = as.logical(precompute),
                        keep_multisets   = keep_multisets,
                        comm_f           = as.integer(comm_f))
  }

  self$get <- function(u, v, max_iterations = 100L, epsilon = 1e-6,
                       edge_weight = 1.0) {
    .Call(C_dress_get_obj,
          self$.ptr,
          as.integer(u),
          as.integer(v),
          as.integer(max_iterations),
          as.double(epsilon),
          as.double(edge_weight))
  }

  self$result <- function() {
    .Call(C_dress_result, self$.ptr)
  }

  self$close <- function() {
    .Call(C_dress_close, self$.ptr)
    invisible(NULL)
  }

  class(self) <- "DRESS"
  self
}

#' Create a persistent MPI+CUDA DRESS graph object
#'
#' Same as \code{cuda$DRESS()} but \code{$delta_fit()} uses MPI+CUDA.
#'
#' @inheritParams DRESS
#' @return An environment (class \code{"DRESS"}) with \code{$fit} (CUDA),
#'   \code{$delta_fit} (MPI+CUDA), \code{$get}, \code{$result}, \code{$close}.
#' @export
mpi$cuda$DRESS <- function(n_vertices,
                           sources,
                           targets,
                           weights               = NULL,
                           variant               = DRESS_UNDIRECTED,
                           precompute_intercepts = FALSE) {

  n_vertices <- as.integer(n_vertices)
  sources    <- as.integer(sources)
  targets    <- as.integer(targets)
  variant    <- as.integer(variant)
  precompute <- as.integer(precompute_intercepts)

  stopifnot(length(sources) == length(targets))
  stopifnot(n_vertices >= 1L)
  stopifnot(variant >= 0L && variant <= 3L)

  if (!is.null(weights)) {
    weights <- as.double(weights)
    stopifnot(length(weights) == length(sources))
  }

  ptr <- .Call(C_dress_init,
               n_vertices, sources, targets, weights,
               variant, precompute)

  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr

  self$fit <- function(max_iterations = 100L, epsilon = 1e-6) {
    .Call("C_dress_fit_cuda_obj",
          self$.ptr,
          as.integer(max_iterations),
          as.double(epsilon),
          PACKAGE = "dress.graph")
  }

  self$delta_fit <- function(k                = 0L,
                             max_iterations   = 100L,
                             epsilon          = 1e-6,
                             keep_multisets   = FALSE,
                             comm_f           = NULL) {
    if (is.null(comm_f)) {
      if (requireNamespace("pbdMPI", quietly = TRUE)) {
        comm_f <- pbdMPI::spmd.comm.c2f(.pbd_env$SPMD.CT$comm)
      } else {
        stop("Provide 'comm_f' (Fortran MPI comm handle) or install pbdMPI.")
      }
    }
    mpi$cuda$delta_dress_fit(n_vertices, sources, targets,
                             weights          = weights,
                             k                = as.integer(k),
                             variant          = variant,
                             max_iterations   = as.integer(max_iterations),
                             epsilon          = as.double(epsilon),
                             precompute       = as.logical(precompute),
                             keep_multisets   = keep_multisets,
                             comm_f           = as.integer(comm_f))
  }

  self$get <- function(u, v, max_iterations = 100L, epsilon = 1e-6,
                       edge_weight = 1.0) {
    .Call(C_dress_get_obj,
          self$.ptr,
          as.integer(u),
          as.integer(v),
          as.integer(max_iterations),
          as.double(epsilon),
          as.double(edge_weight))
  }

  self$result <- function() {
    .Call(C_dress_result, self$.ptr)
  }

  self$close <- function() {
    .Call(C_dress_close, self$.ptr)
    invisible(NULL)
  }

  class(self) <- "DRESS"
  self
}
