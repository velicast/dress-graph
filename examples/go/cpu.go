// cpu.go — Prism vs K₃,₃ with DRESS (CPU)
//
// Run:
//
//	go run cpu.go
package main

import (
	"fmt"
	"sort"

	dress "github.com/velicast/dress-graph/go"
)

func main() {
	// Prism (C₃ □ K₂): 6 vertices, 18 directed edges
	prismS := []int32{0, 1, 1, 2, 2, 0, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 5, 3}
	prismT := []int32{1, 0, 2, 1, 0, 2, 3, 0, 4, 1, 5, 2, 4, 3, 5, 4, 3, 5}

	// K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
	k33S := []int32{0, 3, 0, 4, 0, 5, 1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5}
	k33T := []int32{3, 0, 4, 0, 5, 0, 3, 1, 4, 1, 5, 1, 3, 2, 4, 2, 5, 2}

	rp, _ := dress.Fit(6, prismS, prismT, nil, nil, dress.Undirected, 100, 1e-6, false)
	rk, _ := dress.Fit(6, k33S, k33T, nil, nil, dress.Undirected, 100, 1e-6, false)

	fp := make([]float64, len(rp.EdgeDress))
	fk := make([]float64, len(rk.EdgeDress))
	copy(fp, rp.EdgeDress)
	copy(fk, rk.EdgeDress)
	sort.Float64s(fp)
	sort.Float64s(fk)

	fmt.Printf("Prism: %v\n", fp)
	fmt.Printf("K3,3:  %v\n", fk)

	same := len(fp) == len(fk)
	if same {
		for i := range fp {
			if fp[i] != fk[i] {
				same = false
				break
			}
		}
	}
	fmt.Printf("Distinguished: %v\n", !same)
}
