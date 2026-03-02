\name{delta_dress_fit}
\alias{delta_dress_fit}
\title{Compute the Delta-k-DRESS Histogram}
\description{
Compute the \eqn{\Delta^k}{Delta^k}-DRESS histogram by exhaustively removing
all k-vertex subsets and measuring the change in edge similarity values.
The result is a fixed-size histogram of edge DRESS values pooled across all
\eqn{\binom{N}{k}}{C(N,k)} subgraphs.
}
\usage{
delta_dress_fit(n_vertices, sources, targets, weights = NULL, k = 0L,
                variant = DRESS_UNDIRECTED, max_iterations = 100L,
                epsilon = 1e-6, precompute = FALSE,
                keep_multisets = FALSE)
}
\arguments{
  \item{n_vertices}{Integer. Number of vertices (vertex ids must be in
    \code{0 .. n_vertices - 1}).}
  \item{sources}{Integer vector of length E -- edge source endpoints (0-based).}
  \item{targets}{Integer vector of length E -- edge target endpoints (0-based).}
  \item{weights}{Numeric vector of length E -- edge weights, or \code{NULL}
    for unweighted (all ones). Default \code{NULL}.}
  \item{k}{Integer. Number of vertices to remove per subset
    (0 = original graph, default 0).}
  \item{variant}{Graph variant (default \code{DRESS_UNDIRECTED}).
    One of \code{DRESS_UNDIRECTED} (0), \code{DRESS_DIRECTED} (1),
    \code{DRESS_FORWARD} (2), \code{DRESS_BACKWARD} (3).}
  \item{max_iterations}{Maximum number of fitting iterations per subgraph
    (default 100).}
  \item{epsilon}{Convergence threshold and histogram bin width (default 1e-6).}
  \item{precompute}{Logical. Pre-compute common-neighbor intercept index
    for faster iteration at the cost of more memory (default \code{FALSE}).}
  \item{keep_multisets}{Logical. If \code{TRUE}, return per-subgraph edge
    DRESS values in a C(N,k) x E matrix (\code{NaN} for removed edges).
    Default \code{FALSE}.}
}
\value{
A list with components:
  \item{histogram}{Numeric vector -- bin counts of edge DRESS values
    pooled across all subgraphs.}
  \item{hist_size}{Integer -- number of bins (\code{floor(dmax/epsilon) + 1}; dmax = 2 unweighted).}
  \item{multisets}{Matrix (C(N,k) x E) of per-subgraph edge DRESS values
    (only present when \code{keep_multisets = TRUE}; \code{NaN} = removed edge).}
  \item{num_subgraphs}{Integer -- C(N,k) (only present when
    \code{keep_multisets = TRUE}).}
}
\examples{
# Triangle K3, delta-1: remove 1 vertex at a time
res <- delta_dress_fit(3L, c(0L, 1L, 2L), c(1L, 2L, 0L), k = 1L)
res$hist_size

# K4, delta-0 (original graph)
res0 <- delta_dress_fit(4L, c(0L,0L,0L,1L,1L,2L), c(1L,2L,3L,2L,3L,3L))
sum(res0$histogram)  # 6 edge values
}
\seealso{
\code{\link{dress_fit}} for the standard DRESS computation.
}
