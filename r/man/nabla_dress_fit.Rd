\name{nabla_dress_fit}
\alias{nabla_dress_fit}
\title{Compute the Nabla-k-DRESS Histogram}
\description{
Compute the \eqn{\nabla^k}{Nabla^k}-DRESS histogram by exhaustively
individualizing all k-vertex subsets --- multiplying incident edge weights
by \code{nabla_weight} --- and collecting the resulting edge similarity
values into a fixed-size histogram pooled across all
\eqn{\binom{N}{k}}{C(N,k)} configurations.
}
\usage{
nabla_dress_fit(n_vertices, sources, targets, weights = NULL,
                k = 0L, nabla_weight = 2.0,
                variant = DRESS_UNDIRECTED, max_iterations = 100L,
                epsilon = 1e-6, precompute = FALSE)
}
\arguments{
  \item{n_vertices}{Integer. Number of vertices (vertex ids must be in
    \code{0 .. n_vertices - 1}).}
  \item{sources}{Integer vector of length E -- edge source endpoints (0-based).}
  \item{targets}{Integer vector of length E -- edge target endpoints (0-based).}
  \item{weights}{Optional numeric vector of length E -- per-edge weights.
    \code{NULL} (default) gives every edge weight 1.}
  \item{k}{Integer. Number of vertices to individualize per subset
    (0 = original graph, default 0).}
  \item{nabla_weight}{Numeric. Multiplicative factor applied to edges
    incident to the individualized vertices (default 2.0).  Any value
    other than 1.0 breaks symmetry among the selected vertices.}
  \item{variant}{Graph variant (default \code{DRESS_UNDIRECTED}).
    One of \code{DRESS_UNDIRECTED} (0), \code{DRESS_DIRECTED} (1),
    \code{DRESS_FORWARD} (2), \code{DRESS_BACKWARD} (3).}
  \item{max_iterations}{Maximum number of fitting iterations per
    individualization round (default 100).}
  \item{epsilon}{Convergence threshold and histogram bin width (default 1e-6).}
  \item{precompute}{Logical. Pre-compute common-neighbor intercept index
    for faster iteration at the cost of more memory (default \code{FALSE}).}
}
\value{
A list with components:
  \item{histogram}{Numeric vector -- bin counts of edge DRESS values
    pooled across all individualized configurations.}
  \item{hist_size}{Integer -- number of bins (\code{floor(2/epsilon) + 1}).}
}
\examples{
# Triangle K3, nabla-1: individualize 1 vertex at a time
res <- nabla_dress_fit(3L, c(0L, 1L, 2L), c(1L, 2L, 0L), k = 1L)
res$hist_size

# K4, nabla-0 (original graph, nabla_weight has no effect)
res0 <- nabla_dress_fit(4L, c(0L,0L,0L,1L,1L,2L), c(1L,2L,3L,2L,3L,3L))
sum(res0$histogram)  # 6 edge values
}
\seealso{
\code{\link{dress_fit}} for the standard DRESS computation,
\code{\link{delta_dress_fit}} for the vertex-deletion variant.
}
