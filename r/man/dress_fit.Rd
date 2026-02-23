\name{dress_fit}
\alias{dress_fit}
\alias{DRESS_UNDIRECTED}
\alias{DRESS_DIRECTED}
\alias{DRESS_FORWARD}
\alias{DRESS_BACKWARD}
\alias{dress_version}
\title{Compute DRESS Edge Similarity on Graphs}
\description{
Build a DRESS graph from an edge list and run iterative fitting to
compute per-edge structural similarity values.
}
\usage{
dress_fit(n_vertices, sources, targets, weights = NULL,
          variant = DRESS_UNDIRECTED, max_iterations = 100L,
          epsilon = 1e-6, precompute_intercepts = FALSE)

dress_version()

DRESS_UNDIRECTED
DRESS_DIRECTED
DRESS_FORWARD
DRESS_BACKWARD
}
\arguments{
  \item{n_vertices}{Integer. Number of vertices (vertex ids must be in
    \code{0 .. n_vertices - 1}).}
  \item{sources}{Integer vector of length E -- edge source endpoints (0-based).}
  \item{targets}{Integer vector of length E -- edge target endpoints (0-based).}
  \item{weights}{Optional numeric vector of length E -- per-edge weights.
    \code{NULL} (default) gives every edge weight 1.}
  \item{variant}{Graph variant (default \code{DRESS_UNDIRECTED}).
    One of \code{DRESS_UNDIRECTED} (0), \code{DRESS_DIRECTED} (1),
    \code{DRESS_FORWARD} (2), \code{DRESS_BACKWARD} (3).}
  \item{max_iterations}{Maximum number of fitting iterations (default 100).}
  \item{epsilon}{Convergence threshold -- stop when the max per-edge
    change falls below this value (default 1e-6).}
  \item{precompute_intercepts}{Logical. Pre-compute common-neighbor index
    for faster iteration at the cost of more memory (default \code{FALSE}).}
}
\value{
A list with components:
  \item{sources}{Integer vector [E] -- edge source endpoints (0-based).}
  \item{targets}{Integer vector [E] -- edge target endpoints (0-based).}
  \item{edge_dress}{Numeric vector [E] -- DRESS similarity per edge.}
  \item{edge_weight}{Numeric vector [E] -- variant-specific weight.}
  \item{node_dress}{Numeric vector [N] -- per-node norm.}
  \item{iterations}{Integer -- number of iterations performed.}
  \item{delta}{Numeric -- final max per-edge change.}
}
\examples{
# Triangle + pendant: 0-1, 1-2, 2-0, 2-3
res <- dress_fit(4L, c(0L,1L,2L,2L), c(1L,2L,0L,3L))
res$edge_dress
}
\references{
E. Castrillo, E. Leon, J. Gomez. Dynamic Structural Similarity on Graphs.
arXiv:1805.01419, 2018.
}
\keyword{graphs}
