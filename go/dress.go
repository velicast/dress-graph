// Package dress provides Go bindings for the DRESS C library — iterative
// edge-similarity computation on graphs.
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
#cgo CFLAGS:  -O3 -I../libdress/include
#cgo LDFLAGS: -lm -fopenmp
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

// Fit runs the DRESS iterative fitting algorithm.
//
// Parameters:
//   - n: number of vertices (vertex ids in 0..n-1)
//   - sources, targets: edge list (0-based, same length)
//   - weights: optional edge weights (nil for unweighted)
//   - variant: one of Undirected, Directed, Forward, Backward
//   - maxIterations: maximum fitting iterations
//   - epsilon: convergence threshold
//   - precomputeIntercepts: precompute neighbourhood intercepts (faster, more memory)
func Fit(n int, sources, targets []int32, weights []float64,
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
	C.fit(g, C.int(maxIterations), C.double(epsilon), &iterations, &delta)

	// Read C struct by offset (LP64 layout — see dress.h):
	//   offset 56: *edge_weight   (double*)
	//   offset 64: *edge_dress    (double*)
	//   offset 80: *node_dress    (double*)
	base := uintptr(unsafe.Pointer(g))

	ewPtr := *(*(*C.double))(unsafe.Pointer(base + 56))
	edPtr := *(*(*C.double))(unsafe.Pointer(base + 64))
	ndPtr := *(*(*C.double))(unsafe.Pointer(base + 80))

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
