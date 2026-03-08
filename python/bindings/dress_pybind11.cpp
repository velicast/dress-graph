// pybind11 wrapper for the DRESS C++ class (lib/DRESS.hpp)
//
// Build (example):
//   c++ -O3 -shared -std=c++11 -fPIC \
//       $(python3 -m pybind11 --includes) \
//       -I.. -I../lib \
//       dress_bind.cpp ../dress.o \
//       -o dress$(python3-config --extension-suffix) -lm -fopenmp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dress/dress.hpp"

namespace py = pybind11;

// ---- helpers: expose C arrays as zero-copy NumPy views ----

template <typename T>
static py::array_t<T> wrap_array(const T* ptr, py::ssize_t n, py::object owner) {
    // Return a NumPy view (no copy) that keeps `owner` alive.
    return py::array_t<T>(
        {n},            // shape
        {sizeof(T)},    // strides
        ptr,            // data pointer
        std::move(owner)// prevent GC while view is alive
    );
}

// ---- module definition ----

PYBIND11_MODULE(_core, m) {

    m.doc() = "Python bindings for the dress graph library";

    // ---- enums ----

    py::enum_<dress_variant_t>(m, "Variant",
            "Graph variant — determines how adjacency lists are built")
        .value("UNDIRECTED", DRESS_VARIANT_UNDIRECTED, "N[u] = N(u) (symmetric)")
        .value("DIRECTED",   DRESS_VARIANT_DIRECTED,   "N[u] = in(u) ∪ out(u)")
        .value("FORWARD",    DRESS_VARIANT_FORWARD,    "N[u] = out(u)")
        .value("BACKWARD",   DRESS_VARIANT_BACKWARD,   "N[u] = in(u)")
        .export_values();

    // ---- FitResult ----

    py::class_<DRESS::FitResult>(m, "FitResult",
            "Result returned by DRESS.fit()")
        .def_readonly("iterations", &DRESS::FitResult::iterations,
                      "Number of iterations performed")
        .def_readonly("delta",      &DRESS::FitResult::delta,
                      "Final max per-edge change")
        .def("__repr__", [](const DRESS::FitResult& r) {
            return "FitResult(iterations=" + std::to_string(r.iterations)
                 + ", delta=" + std::to_string(r.delta) + ")";
        });

    // ---- DeltaFitResult ----

    py::class_<DRESS::DeltaFitResult>(m, "DeltaFitResult",
            "Result returned by DRESS.delta_fit()")
        .def_readonly("hist_size",  &DRESS::DeltaFitResult::hist_size,
                      "Number of histogram bins (floor(dmax/epsilon) + 1; dmax=2 unweighted)")
        .def_readonly("num_subgraphs", &DRESS::DeltaFitResult::num_subgraphs,
                      "Number of subgraphs (rows in multisets), 0 if not requested")
        .def_property_readonly("histogram", [](const DRESS::DeltaFitResult& r) {
            return py::array_t<int64_t>(
                {static_cast<py::ssize_t>(r.histogram.size())},
                {sizeof(int64_t)},
                r.histogram.data()
            );
        }, "Histogram as NumPy int64 array (copy)")
        .def_property_readonly("multisets", [](const DRESS::DeltaFitResult& r)
                -> py::object {
            if (r.multisets.empty())
                return py::none();
            // Return as 2D NumPy array (num_subgraphs x E)
            py::ssize_t nrows = r.num_subgraphs;
            py::ssize_t ncols = nrows > 0
                ? static_cast<py::ssize_t>(r.multisets.size()) / nrows
                : 0;
            py::array_t<double> arr(
                {nrows, ncols},
                {static_cast<py::ssize_t>(ncols * sizeof(double)),
                 static_cast<py::ssize_t>(sizeof(double))},
                r.multisets.data()
            );
            return std::move(arr);
        }, "Per-subgraph DRESS values as 2D NumPy array (C(N,k) x E), or None")
        .def("__repr__", [](const DRESS::DeltaFitResult& r) {
            int64_t total = 0;
            for (auto v : r.histogram) total += v;
            return "DeltaFitResult(hist_size=" + std::to_string(r.hist_size)
                 + ", total_values=" + std::to_string(total) + ")";
        });

    // ---- DRESS graph class ----

    py::class_<DRESS>(m, "DRESS", R"doc(
dress graph.

Construct from edge lists (NumPy arrays or Python lists), run iterative
fitting, and read back the per-edge dress similarity values.

Parameters
----------
n_vertices : int
    Number of vertices (vertices are 0-indexed).
sources, targets : array-like of int
    Edge endpoint arrays (same length).
weights : array-like of float, optional
    Per-edge weights. Omit or pass an empty list for unweighted.
variant : Variant
    UNDIRECTED (default), DIRECTED, FORWARD, or BACKWARD.
precompute_intercepts : bool
    Pre-compute common-neighbor index for faster iteration (uses more RAM).
)doc")

        // --- constructors ---

        // Weighted
        .def(py::init<int,
                       const std::vector<int>&,
                       const std::vector<int>&,
                       const std::vector<double>&,
                       dress_variant_t,
                       bool>(),
             py::arg("n_vertices"),
             py::arg("sources"),
             py::arg("targets"),
             py::arg("weights"),
             py::arg("variant")               = DRESS_VARIANT_UNDIRECTED,
             py::arg("precompute_intercepts")  = false)

        // Unweighted
        .def(py::init<int,
                       const std::vector<int>&,
                       const std::vector<int>&,
                       dress_variant_t,
                       bool>(),
             py::arg("n_vertices"),
             py::arg("sources"),
             py::arg("targets"),
             py::arg("variant")               = DRESS_VARIANT_UNDIRECTED,
             py::arg("precompute_intercepts")  = false)

        // --- fitting ---

        .def("fit", &DRESS::fit,
             py::arg("max_iterations"),
             py::arg("epsilon"),
             R"doc(
Run iterative dress fitting.

Parameters
----------
max_iterations : int
    Upper bound on iterations.
epsilon : float
    Convergence threshold (max per-edge change).

Returns
-------
FitResult
    Named result with `iterations` and `delta` fields.
)doc")

        .def("delta_fit", &DRESS::deltaFit,
             py::arg("k"),
             py::arg("max_iterations"),
             py::arg("epsilon"),
             py::arg("keep_multisets") = false,
             R"doc(
Run Δ^k-DRESS: enumerate all C(N,k) node-deletion subsets, fit DRESS
on each subgraph, and accumulate edge values into a histogram.

Parameters
----------
k : int
    Deletion depth (0 = original graph, 1 = single-node deletion, etc.)
max_iterations : int
    Maximum DRESS iterations per subgraph.
epsilon : float
    Convergence tolerance and histogram bin width.
keep_multisets : bool
    If True, also return per-subgraph edge DRESS values in a
    2D array of shape (C(N,k), E).  NaN marks removed edges.

Returns
-------
DeltaFitResult
    Result with `histogram`, `hist_size`, and optionally `multisets`.
)doc")

        .def("get", &DRESS::get,
             py::arg("u"),
             py::arg("v"),
             py::arg("max_iterations") = 100,
             py::arg("epsilon")        = 1e-6,
             py::arg("edge_weight")    = 1.0,
             R"doc(
Query the DRESS value for any vertex pair (u, v).

If edge (u,v) exists in the graph, returns its converged dress value.
If the edge does not exist (virtual edge), estimates the value using
a local fixed-point iteration against the frozen steady state.
Useful for link prediction.

The graph must have been fitted (call `fit()` first).

Parameters
----------
u, v : int
    Vertex ids (0-based).
max_iterations : int
    Max local iterations for virtual edges (default 100).
epsilon : float
    Convergence threshold for local iteration (default 1e-6).
edge_weight : float
    Hypothetical weight of the virtual edge (ignored when the edge
    already exists).  Pass 1.0 for unweighted graphs (default 1.0).

Returns
-------
float
    DRESS similarity value for the (u, v) pair.
)doc")

        // --- scalar accessors ---

        .def_property_readonly("n_vertices", &DRESS::numVertices,
                               "Number of vertices")
        .def_property_readonly("n_edges",    &DRESS::numEdges,
                               "Number of edges")
        .def_property_readonly("variant",    &DRESS::variant,
                               "Graph variant")

        // --- per-element accessors ---

        .def("edge_source", &DRESS::edgeSource, py::arg("e"),
             "Source vertex of edge e")
        .def("edge_target", &DRESS::edgeTarget, py::arg("e"),
             "Target vertex of edge e")
        .def("edge_weight", &DRESS::edgeWeight, py::arg("e"),
             "Weight of edge e")
        .def("edge_dress",  &DRESS::edgeDress,  py::arg("e"),
             "dress value of edge e")
        .def("node_dress",  &DRESS::nodeDress,  py::arg("u"),
             "dress norm of vertex u")

        // --- bulk NumPy views (zero-copy) ---

        .def_property_readonly("sources", [](py::object self) {
            auto& g = self.cast<DRESS&>();
            return wrap_array(g.edgeSources(), g.numEdges(), self);
        }, "Edge source array (NumPy view, int32)")

        .def_property_readonly("targets", [](py::object self) {
            auto& g = self.cast<DRESS&>();
            return wrap_array(g.edgeTargets(), g.numEdges(), self);
        }, "Edge target array (NumPy view, int32)")

        .def_property_readonly("weights", [](py::object self) {
            auto& g = self.cast<DRESS&>();
            return wrap_array(g.edgeWeights(), g.numEdges(), self);
        }, "Edge weight array (NumPy view, float64)")

        .def_property_readonly("dress_values", [](py::object self) {
            auto& g = self.cast<DRESS&>();
            return wrap_array(g.edgeDressValues(), g.numEdges(), self);
        }, "Per-edge dress similarity array (NumPy view, float64)")

        .def_property_readonly("node_dress_values", [](py::object self) {
            auto& g = self.cast<DRESS&>();
            return wrap_array(g.nodeDressValues(), g.numVertices(), self);
        }, "Per-node dress norm array (NumPy view, float64)")

        // --- repr ---

        .def("__repr__", [](const DRESS& g) {
            return "DRESS(n_vertices=" + std::to_string(g.numVertices())
                 + ", n_edges=" + std::to_string(g.numEdges()) + ")";
        });
}
