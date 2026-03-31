package dress_test

import (
	"math"
	"testing"

	dress "github.com/velicast/dress-graph/go"
)

func histTotal(h []dress.HistogramEntry) int64 {
	var total int64
	for _, entry := range h {
		total += entry.Count
	}
	return total
}

func histCountValue(h []dress.HistogramEntry, value float64) int64 {
	for _, entry := range h {
		if math.Abs(entry.Value-value) < 1e-9 {
			return entry.Count
		}
	}
	return 0
}

func TestDeltaHistSize(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Histogram) != 1 {
		t.Fatalf("expected 1 histogram entry, got %d", len(r.Histogram))
	}
}

func TestDeltaWeightedHistSize(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2},
		[]float64{1.0, 10.0, 1.0}, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Histogram) <= 1 {
		t.Fatalf("weighted exact histogram should have multiple entries, got %d", len(r.Histogram))
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("weighted K3 delta0: expected 3 edge values, got %d", total)
	}
}

func TestDeltaDelta0K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("delta0 K3: expected 3 edge values, got %d", total)
	}
	if len(r.Histogram) != 1 {
		t.Fatalf("delta0 K3: expected 1 histogram entry, got %d", len(r.Histogram))
	}
	if histCountValue(r.Histogram, 2.0) != 3 {
		t.Errorf("delta0 K3: expected value 2.0 count=3, got %d", histCountValue(r.Histogram, 2.0))
	}
}

func TestDeltaDelta1K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		1, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
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
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		2, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
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
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 6 {
		t.Fatalf("delta0 K4: expected 6, got %d", total)
	}
	if len(r.Histogram) != 1 {
		t.Fatalf("delta0 K4: expected 1 histogram entry, got %d", len(r.Histogram))
	}
	if histCountValue(r.Histogram, 2.0) != 6 {
		t.Errorf("delta0 K4: expected value 2.0 count=6, got %d", histCountValue(r.Histogram, 2.0))
	}
}

func TestDeltaDelta1K4(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil, nil,
		1, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	// C(4,1)=4 subgraphs * 3 edges = 12
	total := histTotal(r.Histogram)
	if total != 12 {
		t.Fatalf("delta1 K4: expected 12, got %d", total)
	}
	if len(r.Histogram) != 1 {
		t.Fatalf("delta1 K4: expected 1 histogram entry, got %d", len(r.Histogram))
	}
	if histCountValue(r.Histogram, 2.0) != 12 {
		t.Errorf("delta1 K4: expected value 2.0 count=12, got %d", histCountValue(r.Histogram, 2.0))
	}
}

func TestDeltaDelta2K4(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil, nil,
		2, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
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
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		3, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 0 {
		t.Fatalf("k==N: expected 0, got %d", total)
	}

	r2, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		10, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
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
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil, nil,
		1, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := dress.DeltaFit(4,
		[]int32{0, 0, 0, 1, 1, 2}, []int32{1, 2, 3, 2, 3, 3}, nil, nil,
		1, dress.Undirected, 100, 1e-3, 0, 0, true, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r1.Histogram) != len(r2.Histogram) {
		t.Fatalf("precompute: histogram length mismatch %d vs %d", len(r1.Histogram), len(r2.Histogram))
	}
	for i := range r1.Histogram {
		if r1.Histogram[i] != r2.Histogram[i] {
			t.Fatalf("precompute: histogram mismatch at entry %d", i)
		}
	}
}

func TestDeltaDelta0Path(t *testing.T) {
	r, err := dress.DeltaFit(4,
		[]int32{0, 1, 2}, []int32{1, 2, 3}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	total := histTotal(r.Histogram)
	if total != 3 {
		t.Fatalf("delta0 P4: expected 3, got %d", total)
	}
	if len(r.Histogram) < 2 {
		t.Errorf("delta0 P4: expected at least 2 distinct values, got %d", len(r.Histogram))
	}
}

func TestDeltaLengthMismatch(t *testing.T) {
	_, err := dress.DeltaFit(3,
		[]int32{0, 1}, []int32{1, 2, 2}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

// ── multisets ─────────────────────────────────────────────────────────

func TestMultisetsDisabled(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if r.Multisets != nil {
		t.Fatalf("expected nil multisets, got len=%d", len(r.Multisets))
	}
}

func TestMultisetsDelta0K3(t *testing.T) {
	r, err := dress.DeltaFit(3,
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		0, dress.Undirected, 100, 1e-3, 0, 0, false, true, true)
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
		[]int32{0, 1, 0}, []int32{1, 2, 2}, nil, nil,
		1, dress.Undirected, 100, 1e-3, 0, 0, false, true, true)
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
