// Package dress provides MPI-distributed Go bindings for the DRESS C library.
//
// All MPI logic (stride partitioning + Allreduce) runs in C.
// Switch from CPU to MPI by changing only the import path:
//
//	// CPU
//	import "github.com/velicast/dress-graph/go"
//	result, _ := dress.DeltaDressFit(4, U, V, nil, 2, dress.Undirected, 100, 1e-6, false, false, 0, 1)
//
//	// MPI — same call, just change the import and add MPI_Init
//	import dress "github.com/velicast/dress-graph/go/mpi"
//	result, _ := dress.DeltaDressFit(4, U, V, nil, 2, dress.Undirected, 100, 1e-6, false, false)
//
// # Build requirements
//
// Requires CGo, a C compiler, and MPI (mpicc / libmpi).
package dress

/*
#cgo CFLAGS:  -O3 -Ivendor/include -I../../libdress/include
#cgo linux CFLAGS: -I/usr/lib/x86_64-linux-gnu/openmpi/include
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo LDFLAGS: -lm -lmpi -fopenmp
#include <stdlib.h>
#include "dress/dress.h"
#include "dress/delta_dress.h"
#include "dress/mpi/dress_mpi.h"
#include <mpi.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Init initialises MPI. Must be called before any MPI function.
func Init() { C.MPI_Init(nil, nil) }

// Finalize shuts down MPI. Call once when done.
func Finalize() { C.MPI_Finalize() }

// Variant determines how adjacency lists are constructed from the edge list.
type Variant int32

const (
	Undirected Variant = 0
	Directed   Variant = 1
	Forward    Variant = 2
	Backward   Variant = 3
)

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

// DeltaDressFit runs MPI-distributed Δ^k-DRESS (CPU backend).
//
// Same signature as the CPU dress.DeltaDressFit — switch from CPU
// to MPI by changing only the import path.  All MPI logic (stride
// partitioning + Allreduce) runs in C using MPI_COMM_WORLD.
// Call MPI_Init before and MPI_Finalize after.
//
// Parameters:
//   - n: number of vertices
//   - sources, targets: edge list (0-based, same length)
//   - weights: per-edge weights (nil for unweighted)
//   - k: deletion depth (0 = original graph)
//   - variant: graph variant
//   - maxIterations: max DRESS iterations per subgraph
//   - epsilon: convergence tolerance and bin width
//   - precompute: precompute intercepts in each subgraph
//   - keepMultisets: if true, return per-subgraph edge values
func DeltaDressFit(n int, sources, targets []int32, weights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	precompute bool, keepMultisets bool) (*DeltaResult, error) {

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

	hPtr := C.delta_dress_fit_mpi(g, C.int(k), C.int(maxIterations),
		C.double(epsilon), &histSize,
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.MPI_COMM_WORLD)

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
		ln := int(int64(numSub) * int64(e))
		msSlice := unsafe.Slice(msPtr, ln)
		result.Multisets = make([]float64, ln)
		for i := 0; i < ln; i++ {
			result.Multisets[i] = float64(msSlice[i])
		}
		C.free(unsafe.Pointer(msPtr))
	}

	C.free_dress_graph(g)
	return result, nil
}

// ── Persistent graph object ─────────────────────────────────────────

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

// DRESS holds a persistent DRESS graph that supports repeated
// .Fit(), .DeltaFit(), and .Get() queries without rebuilding.
// Fitting uses the CPU; DeltaFit uses MPI-distributed computation.
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

// Fit runs the CPU DRESS iterative fitting algorithm on this graph.
func (dg *DRESS) Fit(maxIterations int, epsilon float64) (iterations int, delta float64, err error) {
	if dg.g == nil {
		return 0, 0, fmt.Errorf("dress: graph has been closed")
	}
	var iters C.int
	var d C.double
	C.dress_fit((C.p_dress_graph_t)(dg.g), C.int(maxIterations), C.double(epsilon), &iters, &d)
	return int(iters), float64(d), nil
}

// DeltaFit runs MPI-distributed Δ^k-DRESS on this persistent graph.
func (dg *DRESS) DeltaFit(k int, maxIterations int, epsilon float64,
	keepMultisets bool) (*DeltaResult, error) {
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

	hPtr := C.delta_dress_fit_mpi((C.p_dress_graph_t)(dg.g), C.int(k), C.int(maxIterations),
		C.double(epsilon), &histSize,
		keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.MPI_COMM_WORLD)

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

// Get queries the DRESS value for any vertex pair (u, v) on a fitted graph.
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

	uwPtr := *(*(*C.int))(unsafe.Pointer(base + 16))
	uvPtr := *(*(*C.int))(unsafe.Pointer(base + 24))
	ewPtr := *(*(*C.double))(unsafe.Pointer(base + 72))
	edPtr := *(*(*C.double))(unsafe.Pointer(base + 80))
	ndPtr := *(*(*C.double))(unsafe.Pointer(base + 96))

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
