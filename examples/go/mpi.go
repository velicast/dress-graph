// mpi.go — Rook vs Shrikhande with Δ¹-DRESS (MPI, CPU backend)
// Keeps multisets and compares them to guarantee distinguishability.
//
// Run:
//
//	mpirun -np 4 go run mpi.go
package main

import (
	"fmt"
	"math"
	"sort"

	dress "github.com/velicast/dress-graph/go/mpi"
)

func main() {
	dress.Init()
	defer dress.Finalize()

	// Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges
	rookS := []int32{0, 1, 0, 4, 0, 2, 0, 8, 0, 3, 0, 12, 1, 5, 1, 2, 1, 9, 1, 3, 1, 13, 2, 6, 2, 10, 2, 3, 2, 14, 3, 7, 3, 11, 3, 15, 4, 5, 4, 6, 4, 8, 4, 7, 4, 12, 5, 6, 5, 9, 5, 7, 5, 13, 6, 10, 6, 7, 6, 14, 7, 11, 7, 15, 8, 9, 8, 10, 8, 11, 8, 12, 9, 10, 9, 11, 9, 13, 10, 11, 10, 14, 11, 15, 12, 13, 12, 14, 12, 15, 13, 14, 13, 15, 14, 15}
	rookT := []int32{1, 0, 4, 0, 2, 0, 8, 0, 3, 0, 12, 0, 5, 1, 2, 1, 9, 1, 3, 1, 13, 1, 6, 2, 10, 2, 3, 2, 14, 2, 7, 3, 11, 3, 15, 3, 5, 4, 6, 4, 8, 4, 7, 4, 12, 4, 6, 5, 9, 5, 7, 5, 13, 5, 10, 6, 7, 6, 14, 6, 11, 7, 15, 7, 9, 8, 10, 8, 11, 8, 12, 8, 10, 9, 11, 9, 13, 9, 11, 10, 14, 10, 15, 11, 13, 12, 14, 12, 15, 12, 14, 13, 15, 13, 15, 14}

	// Shrikhande — 16 vertices, 96 directed edges
	shriS := []int32{0, 4, 0, 12, 0, 1, 0, 3, 0, 5, 0, 15, 1, 5, 1, 13, 1, 2, 1, 6, 1, 12, 2, 6, 2, 14, 2, 3, 2, 7, 2, 13, 3, 7, 3, 15, 3, 4, 3, 14, 4, 8, 4, 5, 4, 7, 4, 9, 5, 9, 5, 6, 5, 10, 6, 10, 6, 7, 6, 11, 7, 11, 7, 8, 8, 12, 8, 9, 8, 11, 8, 13, 9, 13, 9, 10, 9, 14, 10, 14, 10, 11, 10, 15, 11, 15, 11, 12, 12, 13, 12, 15, 13, 14, 14, 15}
	shriT := []int32{4, 0, 12, 0, 1, 0, 3, 0, 5, 0, 15, 0, 5, 1, 13, 1, 2, 1, 6, 1, 12, 1, 6, 2, 14, 2, 3, 2, 7, 2, 13, 2, 7, 3, 15, 3, 4, 3, 14, 3, 8, 4, 5, 4, 7, 4, 9, 4, 9, 5, 6, 5, 10, 5, 10, 6, 7, 6, 11, 6, 11, 7, 8, 7, 12, 8, 9, 8, 11, 8, 13, 8, 13, 9, 10, 9, 14, 9, 14, 10, 11, 10, 15, 10, 15, 11, 12, 11, 13, 12, 15, 12, 14, 13, 15, 14}

	dr, _ := dress.DeltaFit(16, rookS, rookT, nil, nil, 1, dress.Undirected, 100, 1e-6, 0, 0, false, true, true)
	ds, _ := dress.DeltaFit(16, shriS, shriT, nil, nil, 1, dress.Undirected, 100, 1e-6, 0, 0, false, true, true)

	// Only rank 0 prints (all ranks have the reduced result after MPI_Allreduce)
	fmt.Printf("Rook:       %d bins, %d subgraphs\n", len(dr.Histogram), dr.NumSubgraphs)
	fmt.Printf("Shrikhande: %d bins, %d subgraphs\n", len(ds.Histogram), ds.NumSubgraphs)

	same := len(dr.Histogram) == len(ds.Histogram)
	if same {
		for i := range dr.Histogram {
			if dr.Histogram[i] != ds.Histogram[i] {
				same = false
				break
			}
		}
	}
	fmt.Printf("Histograms differ:  %v\n", !same)

	// Canonicalize multisets: sort each row, then sort rows
	E := 96
	canonicalize := func(ms []float64, ns, e int) [][]float64 {
		rows := make([][]float64, ns)
		for i := 0; i < ns; i++ {
			row := make([]float64, e)
			copy(row, ms[i*e:(i+1)*e])
			sort.Float64s(row) // NaN sorts last in Go
			rows[i] = row
		}
		sort.Slice(rows, func(i, j int) bool {
			for c := 0; c < e; c++ {
				if rows[i][c] < rows[j][c] {
					return true
				}
				if rows[i][c] > rows[j][c] {
					return false
				}
			}
			return false
		})
		return rows
	}

	cr := canonicalize(dr.Multisets, int(dr.NumSubgraphs), E)
	cs := canonicalize(ds.Multisets, int(ds.NumSubgraphs), E)

	msSame := len(cr) == len(cs)
	if msSame {
		for i := range cr {
			for j := range cr[i] {
				a, b := cr[i][j], cs[i][j]
				if math.IsNaN(a) && math.IsNaN(b) {
					continue
				}
				if a != b {
					msSame = false
					break
				}
			}
			if !msSame {
				break
			}
		}
	}
	fmt.Printf("Multisets differ:   %v\n", !msSame)
}
