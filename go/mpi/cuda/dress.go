// Package dress provides MPI+CUDA distributed Go bindings for the DRESS C library.
//
// All MPI logic (stride partitioning + Allreduce) runs in C.
// Each rank runs GPU-accelerated DRESS on its stride of subgraphs.
//
//	// MPI + CUDA
//	import dress "github.com/velicast/dress-graph/go/mpi/cuda"
//	result, _ := dress.DeltaDressFit(4, U, V, nil, 2, dress.Undirected, 100, 1e-6, false, false)
//
// # Build requirements
//
// Requires CGo, a C compiler, MPI, and the CUDA toolkit (nvcc / libcudart).
package dress

/*
#cgo CFLAGS:  -O3 -I../../../libdress/include -I../../../libdress/src -DDRESS_CUDA
#cgo linux CFLAGS: -I/usr/lib/x86_64-linux-gnu/openmpi/include
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo LDFLAGS: -L../../../libdress/src/cuda -l:libdress_cuda.a -lcudart_static -lm -lmpi -fopenmp -ldl -lrt -lpthread
#include <stdlib.h>
#include "dress/dress.h"
#include "dress/delta_dress.h"
#include "dress/cuda/dress_cuda.h"
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
	Multisets    []float64
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

// DeltaDressFit runs MPI-distributed Δ^k-DRESS (CUDA backend).
//
// Same signature as the CPU dress.DeltaDressFit — switch from CPU
// to MPI+CUDA by changing only the import path.  All MPI logic
// runs in C; each rank runs GPU-accelerated DRESS on its stride.
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

	hPtr := C.delta_dress_fit_mpi_cuda(g, C.int(k), C.int(maxIterations),
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
