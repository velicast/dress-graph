# ========================================================================
#  dress — R interface to the DRESS on Graphs library
# ========================================================================

# ---- Variant constants ------------------------------------------------

#' Graph variant constants.
#'
#' Determine how adjacency lists are constructed from the input edge list.
#' @export
DRESS_UNDIRECTED <- 0L

#' @rdname DRESS_UNDIRECTED
#' @export
DRESS_DIRECTED   <- 1L

#' @rdname DRESS_UNDIRECTED
#' @export
DRESS_FORWARD    <- 2L

#' @rdname DRESS_UNDIRECTED
#' @export
DRESS_BACKWARD   <- 3L

# ---- Main entry point -------------------------------------------------

#' Compute DRESS edge similarity
#'
#' Build a DRESS graph from an edge list and run iterative fitting.
#' Returns per-edge similarity values along with per-node norms.
#'
#' @param n_vertices Integer. Number of vertices (vertex ids must be in
#'   \code{0 .. n_vertices - 1}).
#' @param sources Integer vector of length E — edge source endpoints (0-based).
#' @param targets Integer vector of length E — edge target endpoints (0-based).
#' @param weights Optional numeric vector of length E — per-edge weights.
#'   \code{NULL} (default) gives every edge weight 1.
#' @param variant Graph variant (default \code{DRESS_UNDIRECTED}).
#'   One of \code{DRESS_UNDIRECTED} (0), \code{DRESS_DIRECTED} (1),
#'   \code{DRESS_FORWARD} (2), \code{DRESS_BACKWARD} (3).
#' @param max_iterations Maximum number of fitting iterations (default 100).
#' @param epsilon Convergence threshold — stop when the max per-edge
#'   change falls below this value (default 1e-6).
#' @param precompute_intercepts Logical. Pre-compute common-neighbor index
#'   for faster iteration at the cost of more memory (default \code{FALSE}).
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{sources}}{Integer vector [E] — edge source endpoints (0-based).}
#'   \item{\code{targets}}{Integer vector [E] — edge target endpoints (0-based).}
#'   \item{\code{edge_dress}}{Numeric vector [E] — DRESS similarity per edge.}
#'   \item{\code{edge_weight}}{Numeric vector [E] — variant-specific weight.}
#'   \item{\code{node_dress}}{Numeric vector [N] — per-node norm.}
#'   \item{\code{iterations}}{Integer — number of iterations performed.}
#'   \item{\code{delta}}{Numeric — final max per-edge change.}
#' }
#'
#' @examples
#' # Triangle + pendant: 0-1, 1-2, 2-0, 2-3
#' res <- dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L))
#' res$edge_dress
#'
#' # Weighted, directed variant
#' res2 <- dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L),
#'                   weights = c(1.0, 2.0, 1.0, 0.5),
#'                   variant = DRESS_DIRECTED)
#'
#' @useDynLib dress.graph, .registration = TRUE
#' @export
dress_fit <- function(n_vertices,
                      sources,
                      targets,
                      weights              = NULL,
                      variant              = DRESS_UNDIRECTED,
                      max_iterations       = 100L,
                      epsilon              = 1e-6,
                      precompute_intercepts = FALSE) {

  # ---- input validation ----
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

  .Call(C_dress_fit,
        n_vertices,
        sources,
        targets,
        weights,
        variant,
        max_iterations,
        epsilon,
        precompute)
}

# ---- Library version -------------------------------------------------

#' DRESS library version string
#' @return Character scalar.
#' @useDynLib dress.graph, .registration = TRUE
#' @export
dress_version <- function() {
  .Call(C_dress_version)
}
