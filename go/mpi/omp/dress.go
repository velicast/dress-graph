// Package dress provides MPI+OMP distributed Go bindings for the DRESS C library.
//
// All MPI logic (stride partitioning + merge of exact sparse histograms) runs in C.
// Within each rank, OpenMP threads parallelise the subgraph slice.
//
// Switch from CPU to MPI+OMP by changing only the import path:
//
//	// CPU
//	import "github.com/velicast/dress-graph/go"
//	result, _ := dress.DeltaFit(4, U, V, nil, nil, 2, dress.Undirected, 100, 1e-6, false, false, 0, 1)
//
//	// MPI + OMP — same call, just change the import
//	import dress "github.com/velicast/dress-graph/go/mpi/omp"
//	dress.Init()
//	result, _ := dress.DeltaFit(4, U, V, nil, nil, 2, dress.Undirected, 100, 1e-6, false, false, 0, 1)
//	dress.Finalize()
//
// # Build requirements
//
// Requires CGo, a C compiler with OpenMP, and MPI (mpicc / libmpi).
package dress

/*
#cgo CFLAGS:  -O3 -fopenmp -Ivendor/include -I../../../libdress/include
#cgo linux CFLAGS: -I/usr/lib/x86_64-linux-gnu/openmpi/include
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo LDFLAGS: -lm -lmpi -fopenmp
#include <stdlib.h>
#include "dress/dress.h"
#include "dress/omp/dress_omp.h"
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

// Result holds the output of a DRESS fitting operation.
type Result struct {
	Sources     []int32
	Targets     []int32
	EdgeWeight  []float64
	EdgeDress   []float64
	VertexDress   []float64
	VertexWeights []float64
	Iterations  int
	Delta       float64
}

func (r *Result) String() string {
	return fmt.Sprintf("FitResult(E=%d, iterations=%d, delta=%.6e)",
		len(r.Sources), r.Iterations, r.Delta)
}

// HistogramEntry is one exact sparse histogram entry produced by Δ^k-DRESS.
type HistogramEntry struct {
	Value float64
	Count int64
}

// DeltaResult holds the output of a Δ^k-DRESS fitting operation.
type DeltaResult struct {
	Histogram    []HistogramEntry
	Multisets    []float64
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

// DeltaFit runs MPI+OMP Δ^k-DRESS.
//
// Same signature as the CPU dress.DeltaFit — switch from CPU
// to MPI+OMP by changing only the import path. MPI distributes
// subgraphs across ranks; within each rank, OpenMP threads
// parallelise the subgraph slice.
func DeltaFit(n int, sources, targets []int32, weights []float64, vertexWeights []float64,
	k int, variant Variant, maxIterations int, epsilon float64,
	nSamples int, seed uint32,
	precompute bool, keepMultisets bool, computeHistogram bool) (*DeltaResult, error) {

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
	if len(vertexWeights) > 0 {
		if len(vertexWeights) != n {
			return nil, fmt.Errorf("dress: vertex weights length (%d) != node count (%d)", len(vertexWeights), n)
		}
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(vertexWeights[i])
		}
	}

	precomp := C.int(0)
	if precompute {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(C.int(n), C.int(e), uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp)
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

	hPtr := C.dress_delta_fit_mpi_omp(g, C.int(k), C.int(maxIterations),
		C.double(epsilon), C.int(nSamples), C.uint(seed),
		&histSize, keepMS,
		func() **C.double {
			if keepMultisets {
				return &msPtr
			}
			return (**C.double)(nil)
		}(),
		&numSub, C.MPI_COMM_WORLD)

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
		ln := int(int64(numSub) * int64(e))
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

// NablaFit runs MPI+OMP ∇^k-DRESS.
//
// Same signature as the CPU dress.NablaFit — switch from CPU
// to MPI+OMP by changing only the import path. MPI distributes
// tuples across ranks; within each rank, OpenMP threads
// parallelise the tuple slice.
func NablaFit(n int, sources, targets []int32, weights []float64, vertexWeights []float64,
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
	if len(vertexWeights) > 0 {
		if len(vertexWeights) != n {
			return nil, fmt.Errorf("dress: vertex weights length (%d) != node count (%d)", len(vertexWeights), n)
		}
		nwPtr = (*C.double)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.double(0)))))
		nwSlice := unsafe.Slice(nwPtr, n)
		for i := 0; i < n; i++ {
			nwSlice[i] = C.double(vertexWeights[i])
		}
	}

	precomp := C.int(0)
	if precompute {
		precomp = C.int(1)
	}

	g := C.dress_init_graph(C.int(n), C.int(e), uPtr, vPtr, wPtr, nwPtr,
		C.dress_variant_t(variant), precomp)
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

	hPtr := C.dress_nabla_fit_mpi_omp(g, C.int(k), C.int(maxIterations),
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
		&numTuples, C.MPI_COMM_WORLD)

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
