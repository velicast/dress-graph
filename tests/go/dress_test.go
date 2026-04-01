package dress_test

import (
	"math"
	"testing"

	dress "github.com/velicast/dress-graph/go"
)

func TestTriangle(t *testing.T) {
	// Triangle: 0-1, 1-2, 0-2
	r, err := dress.Fit(3,
		[]int32{0, 1, 0},
		[]int32{1, 2, 2},
		nil, nil, dress.Undirected, 100, 1e-8, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.EdgeDress) != 3 {
		t.Fatalf("expected 3 edge_dress values, got %d", len(r.EdgeDress))
	}
	if r.Iterations < 1 {
		t.Fatal("expected at least 1 iteration")
	}
	// All edges in a triangle should have equal dress
	d0 := r.EdgeDress[0]
	for i, d := range r.EdgeDress {
		if math.Abs(d-d0) > 1e-6 {
			t.Errorf("edge %d dress %.6f != edge 0 dress %.6f", i, d, d0)
		}
	}
}

func TestPath(t *testing.T) {
	// Path: 0-1-2-3 (no triangles — dress values are positive but lower
	// than in a triangle, because the self-loop constant contributes to
	// the numerator even without common neighbours).
	r, err := dress.Fit(4,
		[]int32{0, 1, 2},
		[]int32{1, 2, 3},
		nil, nil, dress.Undirected, 100, 1e-6, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, d := range r.EdgeDress {
		if d <= 0 {
			t.Errorf("edge %d dress %.6f should be positive (self-loop term)", i, d)
		}
		if d >= 2.0 {
			t.Errorf("edge %d dress %.6f should be well below 2", i, d)
		}
	}
	// Endpoint edges (0-1 and 2-3) should be symmetric
	if math.Abs(r.EdgeDress[0]-r.EdgeDress[2]) > 1e-10 {
		t.Errorf("symmetric path edges should have equal dress: %.10f vs %.10f",
			r.EdgeDress[0], r.EdgeDress[2])
	}
}

func TestVariants(t *testing.T) {
	for _, v := range []dress.Variant{dress.Undirected, dress.Directed, dress.Forward, dress.Backward} {
		r, err := dress.Fit(3,
			[]int32{0, 1, 0},
			[]int32{1, 2, 2},
			nil, nil, v, 100, 1e-6, true)
		if err != nil {
			t.Errorf("variant %d: %v", v, err)
		}
		if len(r.EdgeDress) != 3 {
			t.Errorf("variant %d: expected 3 edges", v)
		}
	}
}

func TestWeighted(t *testing.T) {
	r, err := dress.Fit(3,
		[]int32{0, 1, 0},
		[]int32{1, 2, 2},
		[]float64{1.0, 2.0, 3.0}, nil,
		dress.Undirected, 100, 1e-6, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.EdgeWeight) != 3 {
		t.Fatalf("expected 3 edge weights, got %d", len(r.EdgeWeight))
	}
}

func TestLengthMismatch(t *testing.T) {
	_, err := dress.Fit(3,
		[]int32{0, 1},
		[]int32{1, 2, 3},
		nil, nil, dress.Undirected, 100, 1e-6, true)
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

func TestVertexWeights(t *testing.T) {
	// K3
	n := 3
	src := []int32{0, 1, 0}
	tgt := []int32{1, 2, 2}

	// 1. Default (implicit All-1 vertex weights)
	r1, err := dress.Fit(n, src, tgt, nil, nil, dress.Undirected, 100, 1e-8, true)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Explicit All-1 vertex weights
	nw := []float64{1.0, 1.0, 1.0}
	r2, err := dress.Fit(n, src, tgt, nil, nw, dress.Undirected, 100, 1e-8, true)
	if err != nil {
		t.Fatal(err)
	}

	for i, d1 := range r1.EdgeDress {
		d2 := r2.EdgeDress[i]
		if math.Abs(d1-d2) > 1e-12 {
			t.Errorf("Explicit vertex_weights=1.0 differs from default: %.12f != %.12f", d2, d1)
		}
	}
}

func TestDRESS(t *testing.T) {
	// Triangle: 0-1, 1-2, 0-2
	g, err := dress.NewDRESS(3,
		[]int32{0, 1, 0},
		[]int32{1, 2, 2},
		nil, nil, dress.Undirected, true)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	iters, delta, err := g.Fit(100, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	if iters < 1 {
		t.Fatal("expected at least 1 iteration")
	}
	if delta > 1e-6 {
		t.Errorf("expected convergence, got delta=%.6e", delta)
	}

	// All edges in a triangle should have equal dress via Result()
	r, err := g.Result()
	if err != nil {
		t.Fatal(err)
	}
	d0 := r.EdgeDress[0]
	for i, d := range r.EdgeDress {
		if math.Abs(d-d0) > 1e-6 {
			t.Errorf("edge %d dress %.6f != edge 0 dress %.6f", i, d, d0)
		}
	}

	// Query an existing edge — should return same as Result()
	d01, err := g.Get(0, 1, 100, 1e-8, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(d01-d0) > 1e-4 {
		t.Errorf("Get(0,1)=%.6f differs from Result edge dress %.6f", d01, d0)
	}

	// Query a virtual edge (not in graph) — 0-based, e.g. vertex 0 to itself
	// doesn't cause a crash
	_, err = g.Get(0, 0, 100, 1e-6, 1.0)
	if err != nil {
		t.Fatal(err)
	}
}

func TestVertexWeighted(t *testing.T) {
	// Triangle: 0-1, 1-2, 0-2
	nw := []float64{0.1, 0.2, 0.3}
	r, err := dress.Fit(3,
		[]int32{0, 1, 0},
		[]int32{1, 2, 2},
		nil, nw,
		dress.Undirected, 100, 1e-6, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.VertexWeights) != 3 {
		t.Fatalf("expected 3 vertex weights, got %d", len(r.VertexWeights))
	}
	// Floating point comparison with small epsilon if needed, but here exact copy is expected
	if r.VertexWeights[0] != 0.1 || r.VertexWeights[1] != 0.2 || r.VertexWeights[2] != 0.3 {
		t.Fatalf("vertex weights mismatch: %v", r.VertexWeights)
	}
}
