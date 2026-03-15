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
//	result, err := dress.DressFit(4,
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
#include "dress/delta_dress.h"
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
	Sources    []int32
	Targets    []int32
	EdgeWeight []float64
	EdgeDress  []float64
	NodeDress  []float64
	Iterations int
	Delta      float64
}

func (r *Result) String() string {
	return fmt.Sprintf("DressResult(E=%d, iterations=%d, delta=%.6e)",
		len(r.Sources), r.Iterations, r.Delta)
}

// DressFit runs the DRESS iterative fitting algorithm.
//
// Parameters:
//   - n: number of vertices (vertex ids in 0..n-1)
//   - sources, targets: edge list (0-based, same length)
//   - weights: optional edge weights (nil for unweighted)
//   - variant: one of Undirected, Directed, Forward, Backward
//   - maxIterations: maximum fitting iterations
//   - epsilon: convergence threshold
//   - precomputeIntercepts: precompute neighbourhood intercepts (faster, more memory)
func DressFit(n int, sources, targets []int32, weights []float64,
	variant Variant, maxIterations int, epsilon float64,
	precomputeIntercepts bool) (*Result, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}
	if weights != nil && len(weights) != e {
		return nil, fmt.Errorf("dress: weights length (%d) != edge count (%d)", len(weights), e)
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

	precomp := C.int(0)
	if precomputeIntercepts {
		precomp = C.int(1)
	}

	g := C.init_dress_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: init_dress_graph returned NULL")
	}

	var iterations C.int
	var delta C.double
	C.dress_fit(g, C.int(maxIterations), C.double(epsilon), &iterations, &delta)

	// Read C struct by offset (LP64 layout — see dress.h):
	//   offset 56: *W             (double*)  raw input weights
	//   offset 64: *edge_weight   (double*)
	//   offset 72: *edge_dress    (double*)
	//   offset 88: *node_dress    (double*)
	base := uintptr(unsafe.Pointer(g))

	ewPtr := *(*(*C.double))(unsafe.Pointer(base + 64))
	edPtr := *(*(*C.double))(unsafe.Pointer(base + 72))
	ndPtr := *(*(*C.double))(unsafe.Pointer(base + 88))

	ewSlice := unsafe.Slice(ewPtr, e)
	edSlice := unsafe.Slice(edPtr, e)
	ndSlice := unsafe.Slice(ndPtr, n)

	result := &Result{
		Sources:    make([]int32, e),
		Targets:    make([]int32, e),
		EdgeWeight: make([]float64, e),
		EdgeDress:  make([]float64, e),
		NodeDress:  make([]float64, n),
		Iterations: int(iterations),
		Delta:      float64(delta),
	}
	copy(result.Sources, sources)
	copy(result.Targets, targets)
	for i := 0; i < e; i++ {
		result.EdgeWeight[i] = float64(ewSlice[i])
		result.EdgeDress[i] = float64(edSlice[i])
	}
	for i := 0; i < n; i++ {
		result.NodeDress[i] = float64(ndSlice[i])
	}

	C.free_dress_graph(g)
	return result, nil
}

// DeltaResult holds the output of a Δ^k-DRESS fitting operation.
type DeltaResult struct {
	Histogram    []int64
	HistSize     int
	Multisets    []float64 // row-major C(N,k) × E; NaN = removed edge (nil when not requested)
	NumSubgraphs int64
}

func (r *DeltaResult) String() string {
	var total int64
	for _, v := range r.Histogram {
		total += v
	}
	return fmt.Sprintf("DeltaDressResult(hist_size=%d, total_values=%d)",
		r.HistSize, total)
}

// DeltaDressFit runs Δ^k-DRESS: enumerates all C(N,k) node-deletion subsets,
// runs DRESS on each subgraph, and returns the pooled histogram.
//
// Parameters:
//   - n: number of vertices
//   - sources, targets: edge list (0-based, same length)
//   - weights: per-edge weights (nil for unweighted)
//   - k: deletion depth (0 = original graph)
//   - variant: graph variant
//   - maxIterations: maximum DRESS iterations per subgraph
//   - epsilon: convergence tolerance and bin width
//   - precompute: precompute intercepts in each subgraph
//   - keepMultisets: if true, return per-subgraph edge values
//   - offset: process only subgraphs where index % stride == offset (0)
//   - stride: total number of strides (1 = process all)
func DeltaDressFit(n int, sources, targets []int32, weights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	precompute bool, keepMultisets bool, offset int, stride int) (*DeltaResult, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}

	// Allocate C arrays for init_dress_graph (takes ownership)
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

	precomp := C.int(0)
	if precompute {
		precomp = C.int(1)
	}

	g := C.init_dress_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: init_dress_graph returned NULL")
	}

	var histSize C.int
	var msPtr *C.double
	var numSub C.int64_t

	keepMS := C.int(0)
	if keepMultisets {
		keepMS = C.int(1)
	}

	hPtr := C.delta_dress_fit_strided(g, C.int(k), C.int(maxIterations),
		C.double(epsilon), &histSize,
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.int(offset), C.int(stride))

	result := &DeltaResult{
		HistSize:     int(histSize),
		Histogram:    make([]int64, int(histSize)),
		NumSubgraphs: int64(numSub),
	}

	if hPtr != nil && histSize > 0 {
		hSlice := unsafe.Slice((*C.int64_t)(unsafe.Pointer(hPtr)), int(histSize))
		for i := 0; i < int(histSize); i++ {
			result.Histogram[i] = int64(hSlice[i])
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

	C.free_dress_graph(g)
	return result, nil
}

// Fit is a deprecated alias for DressFit. Use DressFit instead.
//
// Deprecated: will be removed in v1.0.
func Fit(n int, sources, targets []int32, weights []float64,
	variant Variant, maxIterations int, epsilon float64,
	precomputeIntercepts bool) (*Result, error) {
	return DressFit(n, sources, targets, weights, variant, maxIterations, epsilon, precomputeIntercepts)
}

// DeltaFit is a deprecated alias for DeltaDressFit. Use DeltaDressFit instead.
//
// Deprecated: will be removed in v1.0.
func DeltaFit(n int, sources, targets []int32, weights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	precompute bool, keepMultisets bool) (*DeltaResult, error) {
	return DeltaDressFit(n, sources, targets, weights, k, variant, maxIterations, epsilon, precompute, keepMultisets, 0, 1)
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
func NewDRESS(n int, sources, targets []int32, weights []float64,
	variant Variant, precomputeIntercepts bool) (*DRESS, error) {

	e := len(sources)
	if len(targets) != e {
		return nil, fmt.Errorf("dress: sources and targets must have equal length (%d vs %d)", e, len(targets))
	}
	if weights != nil && len(weights) != e {
		return nil, fmt.Errorf("dress: weights length (%d) != edge count (%d)", len(weights), e)
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

	precomp := C.int(0)
	if precomputeIntercepts {
		precomp = C.int(1)
	}

	g := C.init_dress_graph(
		C.int(n), C.int(e),
		uPtr, vPtr, wPtr,
		C.dress_variant_t(variant), precomp,
	)
	if g == nil {
		return nil, fmt.Errorf("dress: init_dress_graph returned NULL")
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
	base := uintptr(dg.g)

	// Struct field offsets (LP64): U=16, V=24, edge_weight=64, edge_dress=72, node_dress=88
	uwPtr := *(*(*C.int))(unsafe.Pointer(base + 16))
	uvPtr := *(*(*C.int))(unsafe.Pointer(base + 24))
	ewPtr := *(*(*C.double))(unsafe.Pointer(base + 64))
	edPtr := *(*(*C.double))(unsafe.Pointer(base + 72))
	ndPtr := *(*(*C.double))(unsafe.Pointer(base + 88))

	uSlice := unsafe.Slice(uwPtr, dg.e)
	vSlice := unsafe.Slice(uvPtr, dg.e)
	ewSlice := unsafe.Slice(ewPtr, dg.e)
	edSlice := unsafe.Slice(edPtr, dg.e)
	ndSlice := unsafe.Slice(ndPtr, dg.n)

	result := &Result{
		Sources:    make([]int32, dg.e),
		Targets:    make([]int32, dg.e),
		EdgeWeight: make([]float64, dg.e),
		EdgeDress:  make([]float64, dg.e),
		NodeDress:  make([]float64, dg.n),
	}
	for i := 0; i < dg.e; i++ {
		result.Sources[i] = int32(uSlice[i])
		result.Targets[i] = int32(vSlice[i])
		result.EdgeWeight[i] = float64(ewSlice[i])
		result.EdgeDress[i] = float64(edSlice[i])
	}
	for i := 0; i < dg.n; i++ {
		result.NodeDress[i] = float64(ndSlice[i])
	}
	return result, nil
}

// Close frees the underlying C graph. Safe to call multiple times.
func (dg *DRESS) Close() {
	if dg.g != nil {
		C.free_dress_graph((C.p_dress_graph_t)(dg.g))
		dg.g = nil
	}
}
