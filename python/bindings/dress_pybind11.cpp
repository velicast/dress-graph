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
