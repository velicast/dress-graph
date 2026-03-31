// Package dress provides Go bindings for the DRESS C library.
//
// DRESS is a deterministic, parameter-free framework that iteratively refines
// the structural similarity of edges in a graph to produce a canonical
// fingerprint: a real-valued edge vector, obtained by converging a non-linear
// dynamical system to its unique fixed point. The fingerprint is
// isomorphism-invariant by construction, numerically stable (no overflow, no
// error amplification, no undefined behavior), fast and embarrassingly parallel to compute: DRESS total runtime
// is O(I * m * d_max) for I iterations to convergence, and convergence is
// guaranteed by Birkhoff contraction.
//
// # Quick start
//
//	result, err := dress.Fit(4,
//	    []int32{0, 1, 2, 0},
//	    []int32{1, 2, 3, 3},
//	    nil, // no weights
//	    dress.Undirected, 100, 1e-6, true)
//	fmt.Printf("iterations: %d\n", result.Iterations)
//
// # Build requirements
//
// Requires CGo and a C compiler.  The dress.c source is compiled
// automatically via the #cgo directives.
package dress

/*
#cgo CFLAGS:  -O3 -Ivendor/include -I../libdress/include
#cgo LDFLAGS: -lm -fopenmp
#include <stdlib.h>
#include "dress/dress.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Variant determines how adjacency lists are constructed from the edge list.
type Variant int32

const (
	Undirected Variant = 0
	Directed   Variant = 1
	Forward    Variant = 2
	Backward   Variant = 3
)

// Result holds the output of a DRESS fitting operation.
type Result struct {
	Sources     []int32
	Targets     []int32
	EdgeWeight  []float64
	EdgeDress   []float64
	NodeDress   []float64
	NodeWeights []float64
	Iterations  int
	Delta       float64
}

func (r *Result) String() string {
	return fmt.Sprintf("FitResult(E=%d, iterations=%d, delta=%.6e)",
		len(r.Sources), r.Iterations, r.Delta)
}

// Fit runs the DRESS iterative fitting algorithm.
//
// Parameters:
//   - n: number of vertices (vertex ids in 0..n-1)
//   - sources, targets: edge list (0-based, same length)
//   - weights: optional edge weights (nil for unweighted)
//   - nodeWeights: optional node weights (nil for unweighted, length n)
//   - variant: one of Undirected, Directed, Forward, Backward
//   - maxIterations: maximum fitting iterations
//   - epsilon: convergence threshold
//   - precomputeIntercepts: precompute neighbourhood intercepts (faster, more memory)
func Fit(n int, sources, targets []int32, weights []float64, nodeWeights []float64,
	variant Variant, maxIterations int, epsilon float64,
	precomputeIntercepts bool) (*Result, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}
	if weights != nil && len(weights) != e {
		return nil, fmt.Errorf("dress: weights length (%d) != edge count (%d)", len(weights), e)
	}
	if nodeWeights != nil && len(nodeWeights) != n {
		return nil, fmt.Errorf("dress: node weights length (%d) != node count (%d)", len(nodeWeights), n)
	}

	// The C library takes ownership of U, V, W via free().
	// We must pass C-malloc'd memory.
	uPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))
	vPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))

	uSlice := unsafe.Slice(uPtr, e)
	vSlice := unsafe.Slice(vPtr, e)
	for i := 0; i < e; i++ {
		uSlice[i] = C.int(sources[i])
		vSlice[i] = C.int(targets[i])
	}

	var wPtr *C.double
	if weights != nil {
		wPtr = (*C.double)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.double(0)))))
		wSlice := unsafe.Slice(wPtr, e)
		for i := 0; i < e; i++ {
			wSlice[i] = C.double(weights[i])
		}
	}

	var nwPtr *C.double
	if nodeWeights != nil {
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(nodeWeights[i])
		}
	}

	precomp := C.int(0)
	if precomputeIntercepts {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: dress_init_graph returned NULL")
	}

	var iterations C.int
	var delta C.double
	C.dress_fit(g, C.int(maxIterations), C.double(epsilon), &iterations, &delta)

	// Read directly from C struct fields
	ewPtr := g.edge_weight
	edPtr := g.edge_dress
	ndPtr := g.node_dress
	cnwPtr := g.NW

	var ewSlice, edSlice []C.double
	var ndSlice, nwSlice []C.double

	if ewPtr != nil {
		ewSlice = unsafe.Slice(ewPtr, e)
	}
	if edPtr != nil {
		edSlice = unsafe.Slice(edPtr, e)
	}
	if ndPtr != nil {
		ndSlice = unsafe.Slice(ndPtr, n)
	}
	if cnwPtr != nil {
		nwSlice = unsafe.Slice(cnwPtr, n)
	}

	result := &Result{
		Sources:     make([]int32, e),
		Targets:     make([]int32, e),
		EdgeWeight:  make([]float64, e),
		EdgeDress:   make([]float64, e),
		NodeDress:   make([]float64, n),
		NodeWeights: nil,
		Iterations:  int(iterations),
		Delta:       float64(delta),
	}
	copy(result.Sources, sources)
	copy(result.Targets, targets)

	if len(ewSlice) > 0 {
		for i := 0; i < e; i++ {
			result.EdgeWeight[i] = float64(ewSlice[i])
		}
	} else {
		for i := 0; i < e; i++ {
			result.EdgeWeight[i] = 1.0
		}
	}

	if len(edSlice) > 0 {
		for i := 0; i < e; i++ {
			result.EdgeDress[i] = float64(edSlice[i])
		}
	}

	if len(ndSlice) > 0 {
		for i := 0; i < n; i++ {
			result.NodeDress[i] = float64(ndSlice[i])
		}
	}

	if len(nwSlice) > 0 {
		result.NodeWeights = make([]float64, n)
		for i := 0; i < n; i++ {
			result.NodeWeights[i] = float64(nwSlice[i])
		}
	}

	C.dress_free_graph(g)
	return result, nil
}

// DeltaResult holds the output of a Δ^k-DRESS fitting operation.
type HistogramEntry struct {
	Value float64
	Count int64
}

// DeltaResult holds the output of a Δ^k-DRESS fitting operation.
type DeltaResult struct {
	Histogram    []HistogramEntry
	Multisets    []float64 // row-major C(N,k) × E; NaN = removed edge (nil when not requested)
	NumSubgraphs int64
}

func (r *DeltaResult) String() string {
	var total int64
	for _, entry := range r.Histogram {
		total += entry.Count
	}
	return fmt.Sprintf("DeltaResult(histogram_entries=%d, total_values=%d)",
		len(r.Histogram), total)
}

// NablaResult holds the output of a ∇^k-DRESS fitting operation.
type NablaResult struct {
	Histogram []HistogramEntry
	Multisets []float64 // row-major P(N,k) × E; NaN = removed edge (nil when not requested)
	NumTuples int64
}

func (r *NablaResult) String() string {
	var total int64
	for _, entry := range r.Histogram {
		total += entry.Count
	}
	return fmt.Sprintf("NablaResult(histogram_entries=%d, total_values=%d)",
		len(r.Histogram), total)
}

// DeltaFit runs Δ^k-DRESS: enumerates all C(N,k) node-deletion subsets,
// runs DRESS on each subgraph, and returns the pooled histogram.
//
// Parameters:
//   - n: number of vertices
//   - sources, targets: edge list (0-based, same length)
//   - weights: per-edge weights (nil for unweighted)
//   - k: deletion depth (0 = original graph)
//   - variant: graph variant
//   - maxIterations: maximum DRESS iterations per subgraph
//   - epsilon: convergence tolerance
//   - precompute: precompute intercepts in each subgraph
//   - keepMultisets: if true, return per-subgraph edge values
func DeltaFit(n int, sources, targets []int32, weights []float64, nodeWeights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	nSamples int, seed uint32,
	precompute bool, keepMultisets bool, computeHistogram bool) (*DeltaResult, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}

	// Allocate C arrays for dress_init_graph (takes ownership)
	uPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))
	vPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))

	uSlice := unsafe.Slice(uPtr, e)
	vSlice := unsafe.Slice(vPtr, e)
	for i := 0; i < e; i++ {
		uSlice[i] = C.int(sources[i])
		vSlice[i] = C.int(targets[i])
	}

	var wPtr *C.double
	if len(weights) > 0 {
		wPtr = (*C.double)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.double(0)))))
		wSlice := unsafe.Slice(wPtr, e)
		for i := 0; i < e; i++ {
			wSlice[i] = C.double(weights[i])
		}
	}

	var nwPtr *C.double
	if len(nodeWeights) > 0 {
		if len(nodeWeights) != n {
			return nil, fmt.Errorf("dress: node weights length (%d) != node count (%d)", len(nodeWeights), n)
		}
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(nodeWeights[i])
		}
	}

	precomp := C.int(0)
	if precompute {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: dress_init_graph returned NULL")
	}

	var histSize C.int
	var msPtr *C.double
	var numSub C.int64_t

	keepMS := C.int(0)
	if keepMultisets {
		keepMS = C.int(1)
	}

	hPtr := C.dress_delta_fit_strided(g, C.int(k), C.int(maxIterations),
		C.double(epsilon), C.int(nSamples), C.uint(seed),
		func() *C.int {
			if computeHistogram {
				return &histSize
			}
			return (*C.int)(nil)
		}(),
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.int(0), C.int(1))

	result := &DeltaResult{NumSubgraphs: int64(numSub)}

	if hPtr != nil && histSize > 0 {
		hSlice := unsafe.Slice((*C.struct___dress_hist_pair_t)(unsafe.Pointer(hPtr)), int(histSize))
		result.Histogram = make([]HistogramEntry, int(histSize))
		for i := 0; i < int(histSize); i++ {
			result.Histogram[i] = HistogramEntry{
				Value: float64(hSlice[i].value),
				Count: int64(hSlice[i].count),
			}
		}
		C.free(unsafe.Pointer(hPtr))
	}

	if keepMultisets && msPtr != nil && numSub > 0 {
		len := int(int64(numSub) * int64(e))
		msSlice := unsafe.Slice(msPtr, len)
		result.Multisets = make([]float64, len)
		for i := 0; i < len; i++ {
			result.Multisets[i] = float64(msSlice[i])
		}
		C.free(unsafe.Pointer(msPtr))
	}

	C.dress_free_graph(g)
	return result, nil
}

// NablaFit runs ∇^k-DRESS: enumerates all P(N,k) ordered node-tuples,
// runs DRESS on each subgraph, and returns the pooled histogram.
//
// Parameters:
//   - n: number of vertices
//   - sources, targets: edge list (0-based, same length)
//   - weights: per-edge weights (nil for unweighted)
//   - k: tuple depth
//   - variant: graph variant
//   - maxIterations: maximum DRESS iterations per subgraph
//   - epsilon: convergence tolerance
//   - precompute: precompute intercepts in each subgraph
//   - keepMultisets: if true, return per-tuple edge values
func NablaFit(n int, sources, targets []int32, weights []float64, nodeWeights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	nSamples int, seed uint32,
	precompute bool, keepMultisets bool, computeHistogram bool) (*NablaResult, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}

	uPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))
	vPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))

	uSlice := unsafe.Slice(uPtr, e)
	vSlice := unsafe.Slice(vPtr, e)
	for i := 0; i < e; i++ {
		uSlice[i] = C.int(sources[i])
		vSlice[i] = C.int(targets[i])
	}

	var wPtr *C.double
	if len(weights) > 0 {
		wPtr = (*C.double)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.double(0)))))
		wSlice := unsafe.Slice(wPtr, e)
		for i := 0; i < e; i++ {
			wSlice[i] = C.double(weights[i])
		}
	}

	var nwPtr *C.double
	if len(nodeWeights) > 0 {
		if len(nodeWeights) != n {
			return nil, fmt.Errorf("dress: node weights length (%d) != node count (%d)", len(nodeWeights), n)
		}
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(nodeWeights[i])
		}
	}

	precomp := C.int(0)
	if precompute {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: dress_init_graph returned NULL")
	}

	var histSize C.int
	var msPtr *C.double
	var numTuples C.int64_t

	keepMS := C.int(0)
	if keepMultisets {
		keepMS = C.int(1)
	}

	hPtr := C.dress_nabla_fit(g, C.int(k), C.int(maxIterations),
		C.double(epsilon), C.int(nSamples), C.uint(seed),
		func() *C.int {
			if computeHistogram {
				return &histSize
			}
			return (*C.int)(nil)
		}(),
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numTuples)

	result := &NablaResult{NumTuples: int64(numTuples)}

	if hPtr != nil && histSize > 0 {
		hSlice := unsafe.Slice((*C.struct___dress_hist_pair_t)(unsafe.Pointer(hPtr)), int(histSize))
		result.Histogram = make([]HistogramEntry, int(histSize))
		for i := 0; i < int(histSize); i++ {
			result.Histogram[i] = HistogramEntry{
				Value: float64(hSlice[i].value),
				Count: int64(hSlice[i].count),
			}
		}
		C.free(unsafe.Pointer(hPtr))
	}

	if keepMultisets && msPtr != nil && numTuples > 0 {
		ln := int(int64(numTuples) * int64(e))
		msSlice := unsafe.Slice(msPtr, ln)
		result.Multisets = make([]float64, ln)
		for i := 0; i < ln; i++ {
			result.Multisets[i] = float64(msSlice[i])
		}
		C.free(unsafe.Pointer(msPtr))
	}

	C.dress_free_graph(g)
	return result, nil
}

// ── Persistent graph object ─────────────────────────────────────────

// DRESS holds a persistent, fitted DRESS graph that supports
// repeated .Get() queries without rebuilding.
type DRESS struct {
	g unsafe.Pointer // *C.struct_dress_graph (owned)
	n int
	e int
}

// NewDRESS constructs a DRESS graph from an edge list.
// The returned graph is NOT fitted yet — call .Fit() before .Get().
// When done, call .Close() to release memory.
func NewDRESS(n int, sources, targets []int32, weights []float64, nodeWeights []float64,
	variant Variant, precomputeIntercepts bool) (*DRESS, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}
	if weights != nil && len(weights) != e {
		return nil, fmt.Errorf("dress: weights length (%d) != edge count (%d)", len(weights), e)
	}
	if nodeWeights != nil && len(nodeWeights) != n {
		return nil, fmt.Errorf("dress: node weights length (%d) != node count (%d)", len(nodeWeights), n)
	}

	uPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))
	vPtr := (*C.int)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.int(0)))))
	uSlice := unsafe.Slice(uPtr, e)
	vSlice := unsafe.Slice(vPtr, e)
	for i := 0; i < e; i++ {
		uSlice[i] = C.int(sources[i])
		vSlice[i] = C.int(targets[i])
	}

	var wPtr *C.double
	if weights != nil {
		wPtr = (*C.double)(C.malloc(C.size_t(e) * C.size_t(unsafe.Sizeof(C.double(0)))))
		wSlice := unsafe.Slice(wPtr, e)
		for i := 0; i < e; i++ {
			wSlice[i] = C.double(weights[i])
		}
	}

	var nwPtr *C.double
	if nodeWeights != nil {
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(nodeWeights[i])
		}
	}

	precomp := C.int(0)
	if precomputeIntercepts {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: dress_init_graph returned NULL")
	}

	return &DRESS{g: unsafe.Pointer(g), n: n, e: e}, nil
}

// Fit runs the DRESS iterative fitting algorithm on this graph.
func (dg *DRESS) Fit(maxIterations int, epsilon float64) (iterations int, delta float64, err error) {
	if dg.g == nil {
		return 0, 0, fmt.Errorf("dress: graph has been closed")
	}
	var iters C.int
	var d C.double
	C.dress_fit((C.p_dress_graph_t)(dg.g), C.int(maxIterations), C.double(epsilon), &iters, &d)
	return int(iters), float64(d), nil
}

// DeltaFit runs Δ^k-DRESS on this persistent graph.
func (dg *DRESS) DeltaFit(k int, maxIterations int, epsilon float64,
	nSamples int, seed uint32,
	keepMultisets bool, computeHistogram bool) (*DeltaResult, error) {
	if dg.g == nil {
		return nil, fmt.Errorf("dress: graph has been closed")
	}

	var histSize C.int
	var msPtr *C.double
	var numSub C.int64_t

	keepMS := C.int(0)
	if keepMultisets {
		keepMS = C.int(1)
	}

	hPtr := C.dress_delta_fit_strided((C.p_dress_graph_t)(dg.g), C.int(k), C.int(maxIterations),
		C.double(epsilon), C.int(nSamples), C.uint(seed),
		func() *C.int {
			if computeHistogram {
				return &histSize
			}
			return (*C.int)(nil)
		}(),
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.int(0), C.int(1))

	result := &DeltaResult{NumSubgraphs: int64(numSub)}

	if hPtr != nil && histSize > 0 {
		hSlice := unsafe.Slice((*C.struct___dress_hist_pair_t)(unsafe.Pointer(hPtr)), int(histSize))
		result.Histogram = make([]HistogramEntry, int(histSize))
		for i := 0; i < int(histSize); i++ {
			result.Histogram[i] = HistogramEntry{
				Value: float64(hSlice[i].value),
				Count: int64(hSlice[i].count),
			}
		}
		C.free(unsafe.Pointer(hPtr))
	}

	if keepMultisets && msPtr != nil && numSub > 0 {
		ln := int(int64(numSub) * int64(dg.e))
		msSlice := unsafe.Slice(msPtr, ln)
		result.Multisets = make([]float64, ln)
		for i := 0; i < ln; i++ {
			result.Multisets[i] = float64(msSlice[i])
		}
		C.free(unsafe.Pointer(msPtr))
	}

	return result, nil
}

// NablaFit runs ∇^k-DRESS on this persistent graph.
func (dg *DRESS) NablaFit(k int, maxIterations int, epsilon float64,
	nSamples int, seed uint32,
	keepMultisets bool, computeHistogram bool) (*NablaResult, error) {
	if dg.g == nil {
		return nil, fmt.Errorf("dress: graph has been closed")
	}

	var histSize C.int
	var msPtr *C.double
	var numTuples C.int64_t

	keepMS := C.int(0)
	if keepMultisets {
		keepMS = C.int(1)
	}

	hPtr := C.dress_nabla_fit((C.p_dress_graph_t)(dg.g), C.int(k), C.int(maxIterations),
		C.double(epsilon), C.int(nSamples), C.uint(seed),
		func() *C.int {
			if computeHistogram {
				return &histSize
			}
			return (*C.int)(nil)
		}(),
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numTuples)

	result := &NablaResult{NumTuples: int64(numTuples)}

	if hPtr != nil && histSize > 0 {
		hSlice := unsafe.Slice((*C.struct___dress_hist_pair_t)(unsafe.Pointer(hPtr)), int(histSize))
		result.Histogram = make([]HistogramEntry, int(histSize))
		for i := 0; i < int(histSize); i++ {
			result.Histogram[i] = HistogramEntry{
				Value: float64(hSlice[i].value),
				Count: int64(hSlice[i].count),
			}
		}
		C.free(unsafe.Pointer(hPtr))
	}

	if keepMultisets && msPtr != nil && numTuples > 0 {
		ln := int(int64(numTuples) * int64(dg.e))
		msSlice := unsafe.Slice(msPtr, ln)
		result.Multisets = make([]float64, ln)
		for i := 0; i < ln; i++ {
			result.Multisets[i] = float64(msSlice[i])
		}
		C.free(unsafe.Pointer(msPtr))
	}

	return result, nil
}

// Get queries the DRESS value for any vertex pair (u, v) on a fitted graph.
//
// If edge (u,v) exists, returns its converged value.
// If the edge does not exist (virtual edge), estimates it via local
// fixed-point iteration. edgeWeight is the hypothetical weight for a
// virtual edge (use 1.0 for unweighted graphs).
func (dg *DRESS) Get(u, v int, maxIterations int, epsilon float64, edgeWeight float64) (float64, error) {
	if dg.g == nil {
		return 0, fmt.Errorf("dress: graph has been closed")
	}
	val := float64(C.dress_get(
		(C.p_dress_graph_t)(dg.g),
		C.int(u), C.int(v),
		C.int(maxIterations), C.double(epsilon), C.double(edgeWeight)))
	return val, nil
}

// Result extracts a snapshot of the fitted graph into a Result struct.
func (dg *DRESS) Result() (*Result, error) {
	if dg.g == nil {
		return nil, fmt.Errorf("dress: graph has been closed")
	}
	// Read directly from C struct fields
	g := (C.p_dress_graph_t)(dg.g)

	uwPtr := g.U
	uvPtr := g.V
	ewPtr := g.edge_weight
	edPtr := g.edge_dress
	ndPtr := g.node_dress

	uSlice := unsafe.Slice(uwPtr, dg.e)
	vSlice := unsafe.Slice(uvPtr, dg.e)
	var ewSlice, edSlice []C.double
	var ndSlice []C.double

	if ewPtr != nil {
		ewSlice = unsafe.Slice(ewPtr, dg.e)
	}
	if edPtr != nil {
		edSlice = unsafe.Slice(edPtr, dg.e)
	}
	if ndPtr != nil {
		ndSlice = unsafe.Slice(ndPtr, dg.n)
	}

	result := &Result{
		Sources:    make([]int32, dg.e),
		Targets:    make([]int32, dg.e),
		EdgeWeight: make([]float64, dg.e),
		EdgeDress:  make([]float64, dg.e),
		NodeDress:  make([]float64, dg.n),
		Iterations: 0,
		Delta:      0,
	}

	for i := 0; i < dg.e; i++ {
		result.Sources[i] = int32(uSlice[i])
		result.Targets[i] = int32(vSlice[i])
	}

	if len(ewSlice) > 0 {
		for i := 0; i < dg.e; i++ {
			result.EdgeWeight[i] = float64(ewSlice[i])
		}
	} else {
		for i := 0; i < dg.e; i++ {
			result.EdgeWeight[i] = 1.0
		}
	}

	if len(edSlice) > 0 {
		for i := 0; i < dg.e; i++ {
			result.EdgeDress[i] = float64(edSlice[i])
		}
	}

	if len(ndSlice) > 0 {
		for i := 0; i < dg.n; i++ {
			result.NodeDress[i] = float64(ndSlice[i])
		}
	}

	return result, nil
}

// Close frees the underlying C graph. Safe to call multiple times.
func (dg *DRESS) Close() {
	if dg.g != nil {
		C.dress_free_graph((C.p_dress_graph_t)(dg.g))
		dg.g = nil
	}
}
