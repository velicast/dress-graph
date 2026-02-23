# The Role of Weights

Edge weights in DRESS are integral to the
equation and have a clean structural interpretation.

## How weights enter the equation

Every term in the DRESS update has the form \(\bar{w}(u,x) \cdot d_{ux}\).
Weights always appear as a multiplicative factor on the similarity value.
This coupling means weights and similarity are inseparable: the structural
profile of a node is the vector of \(\bar{w} \cdot d\) products over its
incident edges.

## Interpretation

### Conductance

In the diffusion analogy, weights control how much structural information
flows along each edge.  A high-weight edge is a strong conductor; a
low-weight edge is a weak channel.  At steady state, the similarity
landscape reflects not just topology but how strongly connected each path is.

### Attention

When computing the update for edge \((u, v)\), the weight \(\bar{w}(u, x)\)
scales how much neighbour \(x\) contributes to \(u\)'s structural profile.
Heavy edges amplify a neighbour's vote; light edges dampen it.  Weights act
as a built-in attention mechanism.

### Relative, not absolute

Because weights appear identically in the numerator and denominator (both are
degree-1 in \(\bar{w} \cdot d\)), uniformly scaling all weights does not
change the result.  Only the **relative** weights matter.  DRESS captures the
pattern of connection strengths, not their absolute magnitude.

## Unweighted as a special case

When all weights are 1, the equation reduces to pure structural counting.
Weights generalise this: they let the graph express "this connection matters
more than that one," and DRESS respects it throughout its fixed-point
computation without introducing any additional parameters.
