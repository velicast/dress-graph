# ========================================================================
#  cuda — GPU-accelerated DRESS, same API as the CPU functions.
#
#  Usage (import-based switching):
#
#    library(dress.graph)
#
#    # CPU
#    result <- dress_fit(4L, sources, targets)
#
#    # CUDA — same function name, accessed via cuda$ environment
#    result <- cuda$dress_fit(4L, sources, targets)
# ========================================================================

#' CUDA-accelerated DRESS functions.
#'
#' An environment containing GPU-accelerated versions of \code{dress_fit}
#' and \code{delta_dress_fit} with identical signatures.  Switch from
#' CPU to GPU by prefixing calls with \code{cuda$}.
#'
#' @examples
#' \dontrun{
#'   # CPU
#'   r1 <- dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L))
#'
#'   # CUDA — same signature
#'   r2 <- cuda$dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L))
#' }
#' @export
cuda <- local({

  env <- new.env(parent = emptyenv())

  .check_cuda <- function() {
    ok <- tryCatch(
      { getNativeSymbolInfo("C_dress_fit_cuda", "dress.graph"); TRUE },
      error = function(e) FALSE
    )
    if (!ok)
      stop("CUDA support not available. Rebuild dress.graph with DRESS_CUDA=1.",
           call. = FALSE)
  }

  env$dress_fit <- function(n_vertices,
                            sources,
                            targets,
                            weights              = NULL,
                            variant              = DRESS_UNDIRECTED,
                            max_iterations       = 100L,
                            epsilon              = 1e-6,
                            precompute_intercepts = FALSE) {

    .check_cuda()

    n_vertices     <- as.integer(n_vertices)
    sources        <- as.integer(sources)
    targets        <- as.integer(targets)
    variant        <- as.integer(variant)
    max_iterations <- as.integer(max_iterations)
    epsilon        <- as.double(epsilon)
    precompute     <- as.integer(precompute_intercepts)

    stopifnot(length(sources) == length(targets))
    stopifnot(n_vertices >= 1L)
    stopifnot(variant >= 0L && variant <= 3L)
    stopifnot(max_iterations >= 1L)
    stopifnot(epsilon > 0)

    if (!is.null(weights)) {
      weights <- as.double(weights)
      stopifnot(length(weights) == length(sources))
    }

    .Call("C_dress_fit_cuda",
          n_vertices, sources, targets, weights,
          variant, max_iterations, epsilon, precompute,
          PACKAGE = "dress.graph")
  }

  env$delta_dress_fit <- function(n_vertices,
                                  sources,
                                  targets,
                                  weights          = NULL,
                                  k                = 0L,
                                  variant          = DRESS_UNDIRECTED,
                                  max_iterations   = 100L,
                                  epsilon          = 1e-6,
                                  precompute       = FALSE,
                                  keep_multisets   = FALSE) {

    .check_cuda()

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
    if (!is.null(weights)) stopifnot(length(weights) == length(sources))
    stopifnot(n_vertices >= 1L)
    stopifnot(k >= 0L)
    stopifnot(variant >= 0L && variant <= 3L)
    stopifnot(max_iterations >= 1L)
    stopifnot(epsilon > 0)

    .Call("C_delta_dress_fit_cuda",
          n_vertices, sources, targets, weights,
          k, variant, max_iterations, epsilon,
          precompute, keep_multisets,
          PACKAGE = "dress.graph")
  }

  env
})
