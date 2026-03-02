package dress_test

import (
	"math"
	"testing"

	dress "github.com/velicast/dress-graph/go"
)

func histTotal(h []int64) int64 {
	var total int64
	for _, v := range h {
		total += v
	}
	return total
}

func TestDeltaHistSize(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if r.HistSize != 2001 {
		t.Fatalf("expected hist_size=2001, got %d", r.HistSize)
	}
}

func TestDeltaWeightedHistSize(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2},
		[]float64{1.0, 10.0, 1.0},
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if r.HistSize <= 2001 {
		t.Fatalf("weighted hist_size should be > 2001, got %d", r.HistSize)
	}
	if len(r.Histogram) != r.HistSize {
		t.Fatalf("histogram length %d != hist_size %d", len(r.Histogram), r.HistSize)
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("weighted K3 delta0: expected 3 edge values, got %d", total)
	}
}

func TestDeltaDelta0K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("delta0 K3: expected 3 edge values, got %d", total)
	}
	// Top bin should hold all 3 edges (dress = 2.0)
	if r.Histogram[r.HistSize-1] != 3 {
		t.Errorf("delta0 K3: expected top bin=3, got %d", r.Histogram[r.HistSize-1])
	}
}

func TestDeltaDelta1K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		1, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	// C(3,1)=3 subgraphs * 1 edge = 3
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("delta1 K3: expected 3, got %d", total)
	}
}

func TestDeltaDelta2K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		2, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 0 {
		t.Fatalf("delta2 K3: expected 0, got %d", total)
	}
}

func TestDeltaDelta0K4(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 6 {
		t.Fatalf("delta0 K4: expected 6, got %d", total)
	}
	if r.Histogram[r.HistSize-1] != 6 {
		t.Errorf("delta0 K4: expected top bin=6, got %d", r.Histogram[r.HistSize-1])
	}
}

func TestDeltaDelta1K4(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil,
		1, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	// C(4,1)=4 subgraphs * 3 edges = 12
	total := histTotal(r.Histogram)
	if total != 12 {
		t.Fatalf("delta1 K4: expected 12, got %d", total)
	}
	if r.Histogram[r.HistSize-1] != 12 {
		t.Errorf("delta1 K4: expected top bin=12, got %d", r.Histogram[r.HistSize-1])
	}
}

func TestDeltaDelta2K4(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil,
		2, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	// C(4,2)=6 subgraphs * 1 edge = 6
	total := histTotal(r.Histogram)
	if total != 6 {
		t.Fatalf("delta2 K4: expected 6, got %d", total)
	}
}

func TestDeltaKGeN(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		3, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 0 {
		t.Fatalf("k==N: expected 0, got %d", total)
	}

	r2, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		10, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total2 := histTotal(r2.Histogram)
	if total2 != 0 {
		t.Fatalf("k>N: expected 0, got %d", total2)
	}
}

func TestDeltaPrecompute(t *testing.T) {
	r1, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil,
		1, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil,
		1, dress.Undirected, 100, 1e-3, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if r1.HistSize != r2.HistSize {
		t.Fatalf("precompute: hist_size mismatch %d vs %d", r1.HistSize, r2.HistSize)
	}
	for i := 0; i < r1.HistSize; i++ {
		if r1.Histogram[i] != r2.Histogram[i] {
			t.Fatalf("precompute: histogram mismatch at bin %d", i)
		}
	}
}

func TestDeltaDelta0Path(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 1, 2}, []int32{1, 2, 3}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("delta0 P4: expected 3, got %d", total)
	}
	// P4 edges not all equal → at least 2 distinct bins
	nonzero := 0
	for _, v := range r.Histogram {
		if v > 0 {
			nonzero++
		}
	}
	if nonzero < 2 {
		t.Errorf("delta0 P4: expected ≥2 distinct bins, got %d", nonzero)
	}
}

func TestDeltaLengthMismatch(t *testing.T) {
	_, err := dress.DeltaFit(3,
		[]int32{0, 1}, []int32{1, 2, 2}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

// ── multisets ─────────────────────────────────────────────────────────

func TestMultisetsDisabled(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		0, dress.Undirected, 100, 1e-3, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if r.Multisets != nil {
		t.Fatalf("expected nil multisets, got len=%d", len(r.Multisets))
	}
}

func TestMultisetsDelta0K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		0, dress.Undirected, 100, 1e-3, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if r.NumSubgraphs != 1 {
		t.Fatalf("expected 1 subgraph, got %d", r.NumSubgraphs)
	}
	// 1 subgraph * 3 edges = 3 values
	if len(r.Multisets) != 3 {
		t.Fatalf("expected 3 values, got %d", len(r.Multisets))
	}
	for i, v := range r.Multisets {
		if v < 2.0-1e-3 || v > 2.0+1e-3 {
			t.Errorf("multisets[%d] = %f, expected ~2.0", i, v)
		}
	}
}

func TestMultisetsDelta1K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil,
		1, dress.Undirected, 100, 1e-3, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if r.NumSubgraphs != 3 {
		t.Fatalf("expected 3 subgraphs, got %d", r.NumSubgraphs)
	}
	// 3 subgraphs * 3 edges = 9 flat values
	if len(r.Multisets) != 9 {
		t.Fatalf("expected 9 values, got %d", len(r.Multisets))
	}
	// Each row: 2 NaN + 1 value ≈ 2.0
	E := 3
	for s := 0; s < 3; s++ {
		nans := 0
		for e := 0; e < E; e++ {
			v := r.Multisets[s*E+e]
			if math.IsNaN(v) {
				nans++
			} else if v < 2.0-1e-3 || v > 2.0+1e-3 {
				t.Errorf("row %d col %d: %f, expected ~2.0", s, e, v)
			}
		}
		if nans != 2 {
			t.Errorf("row %d: expected 2 NaN, got %d", s, nans)
		}
	}
}
