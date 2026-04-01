//! # dress-graph
//!
//! Safe Rust bindings for the **DRESS** C library — A Continuous Framework for Structural Graph Refinement.  See the [DRESS repository](https://github.com/velicat/dress-graph) for more information.
//!
//! ```no_run
//! use dress_graph::{DRESS, Variant};
//!
//! let sources = vec![0, 1, 2, 0];
//! let targets = vec![1, 2, 3, 3];
//!
//! let mut g = DRESS::new(4, sources, targets,
//!                        None, None, Variant::Undirected, false).unwrap();
//! let (iters, delta) = g.fit(100, 1e-6);
//! let result = g.result();
//!
//! println!("iterations: {}", iters);
//! for (i, d) in result.edge_dress.iter().enumerate() {
//!     println!("  edge {}: dress = {:.6}", i, d);
//! }
//! ```

use std::ffi::c_void;
use std::fmt;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "omp")]
pub mod omp;

#[cfg(feature = "mpi")]
pub mod mpi;

// ── FFI declarations ────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;
#[allow(non_camel_case_types)]
type c_uint = u32;
#[allow(non_camel_case_types)]
type c_double = f64;

#[allow(dead_code)]
mod ffi {
    use super::*;
    extern "C" {
        pub(crate) fn dress_init_graph(
            n: c_int, e: c_int,
            u: *mut c_int, v: *mut c_int,
            w: *mut c_double, nw: *mut c_double,
            variant: c_int, precompute_intercepts: c_int,
        ) -> *mut c_void;

        pub(crate) fn dress_fit(
            g: *mut c_void, max_iterations: c_int, epsilon: c_double,
            iterations: *mut c_int, delta: *mut c_double,
        );

        pub(crate) fn dress_free_graph(g: *mut c_void);

        pub(crate) fn dress_get(
            g: *mut c_void, u: c_int, v: c_int,
            max_iterations: c_int, epsilon: c_double, edge_weight: c_double,
        ) -> c_double;

        pub(crate) fn dress_delta_fit_strided(
            g: *mut c_void, k: c_int, iterations: c_int, epsilon: c_double,
            n_samples: c_int, seed: c_uint,
            hist_size: *mut c_int, keep_multisets: c_int,
            multisets: *mut *mut c_double, num_subgraphs: *mut i64,
            offset: c_int, stride: c_int,
        ) -> *mut HistogramEntry;

        pub(crate) fn dress_nabla_fit(
            g: *mut c_void, k: c_int, iterations: c_int, epsilon: c_double,
            n_samples: c_int, seed: c_uint,
            hist_size: *mut c_int, keep_multisets: c_int,
            multisets: *mut *mut c_double, num_tuples: *mut i64,
        ) -> *mut HistogramEntry;
    }
}

// ── Public types ────────────────────────────────────────────────────

/// Graph variant — determines how neighbourhoods are constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Variant {
    Undirected = 0,
    Directed   = 1,
    Forward    = 2,
    Backward   = 3,
}

/// Result of the DRESS fitting procedure.
#[derive(Debug, Clone)]
pub struct DressResult {
    pub sources:     Vec<i32>,
    pub targets:     Vec<i32>,
    pub edge_weight: Vec<f64>,
    pub edge_dress:  Vec<f64>,
    pub vertex_dress:  Vec<f64>,
    pub vertex_weights: Option<Vec<f64>>,
    pub iterations:  i32,
    pub delta:       f64,
}

/// Exact sparse histogram entry produced by Δ^k-DRESS.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct HistogramEntry {
    pub value: f64,
    pub count: i64,
}

impl fmt::Display for DressResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DressResult(E={}, iterations={}, delta={:.6e})",
            self.sources.len(),
            self.iterations,
            self.delta,
        )
    }
}

// ── Persistent graph object ─────────────────────────────────────────

/// A persistent DRESS graph that supports repeated `fit` and `get` calls.
///
/// The underlying C graph is freed automatically when dropped.
///
/// ```no_run
/// use dress_graph::{DRESS, Variant};
///
/// let mut g = DRESS::new(4, vec![0,1,2,0], vec![1,2,3,3],
///                        None, None, Variant::Undirected, false)?;
/// g.fit(100, 1e-6);
/// let d = g.get(0, 2, 100, 1e-6, 1.0);
/// ```
pub struct DRESS {
    g: *mut c_void,
    n: i32,
    e: usize,
    sources: Vec<i32>,
    targets: Vec<i32>,
}

impl DRESS {
    /// Construct a persistent DRESS graph.
    pub fn new(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        vertex_weights: Option<Vec<f64>>,
        variant: Variant,
        precompute_intercepts: bool,
    ) -> Result<DRESS, DressError> {
        let e = sources.len();
        if targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }
        unsafe {
            let u_ptr = libc_malloc_copy_i32(&sources);
            let v_ptr = libc_malloc_copy_i32(&targets);
            let w_ptr = weights.as_ref().map_or(std::ptr::null_mut(), |w| libc_malloc_copy_f64(w));
            let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| libc_malloc_copy_f64(nw));
            let g = ffi::dress_init_graph(n, e as c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                                     variant as c_int, precompute_intercepts as c_int);
            if g.is_null() {
                return Err(DressError::InitFailed);
            }
            Ok(DRESS { g, n, e, sources, targets })
        }
    }

    /// Fit the DRESS model.  Returns `(iterations, delta)`.
    pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut iterations: c_int = 0;
        let mut delta: c_double = 0.0;
        unsafe {
            ffi::dress_fit(self.g, max_iterations, epsilon, &mut iterations, &mut delta);
        }
        (iterations, delta)
    }

    /// Query the DRESS value for an edge (existing or virtual).
    pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
        assert!(!self.g.is_null(), "DRESS already closed");
        unsafe { ffi::dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
    }

    /// Extract a snapshot of the current results without freeing.
    pub fn result(&self) -> DressResult {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;
        let n = self.n as usize;
        unsafe {
            let base = self.g as *const u8;
            let ew_ptr = *(base.add(72) as *const *const f64);
            let ed_ptr = *(base.add(80) as *const *const f64);
            let nd_ptr = *(base.add(96) as *const *const f64);
            let nw_ptr = *(base.add(104) as *const *const f64);

            let vertex_weights = if !nw_ptr.is_null() {
                Some(std::slice::from_raw_parts(nw_ptr, n).to_vec())
            } else {
                None
            };

            DressResult {
                sources:     self.sources.clone(),
                targets:     self.targets.clone(),
                edge_weight: std::slice::from_raw_parts(ew_ptr, e).to_vec(),
                edge_dress:  std::slice::from_raw_parts(ed_ptr, e).to_vec(),
                vertex_dress:  std::slice::from_raw_parts(nd_ptr, n).to_vec(),
                vertex_weights,
                iterations:  0,
                delta:       0.0,
            }
        }
    }

    /// Run Δ^k-DRESS on the persistent graph: enumerate all C(N,k)
    /// vertex-deletion subsets, fit DRESS on each subgraph, and return the
    /// pooled histogram.
    pub fn delta_fit(
        &self,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        keep_multisets: bool,
        compute_histogram: bool,
    ) -> Result<DeltaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;

        unsafe {
            let mut hsize: c_int = 0;
            let mut ms_ptr: *mut c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;
            let h = ffi::dress_delta_fit_strided(
                self.g,
                k,
                max_iterations,
                epsilon,
                n_samples,
                seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                0,
                1,
            );

            let histogram = histogram_from_raw(h, hsize);

            extern "C" { fn free(ptr: *mut std::ffi::c_void); }

            let multisets = if keep_multisets && !ms_ptr.is_null() && num_sub > 0 {
                let len = (num_sub as usize) * e;
                let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
                free(ms_ptr as *mut std::ffi::c_void);
                Some(ms)
            } else {
                if keep_multisets && !ms_ptr.is_null() {
                    free(ms_ptr as *mut std::ffi::c_void);
                }
                None
            };

            if !h.is_null() {
                free(h as *mut std::ffi::c_void);
            }

            Ok(DeltaDressResult {
                histogram,
                multisets,
                num_subgraphs: num_sub,
            })
        }
    }

    /// Run ∇^k-DRESS on the persistent graph.
    pub fn nabla_fit(
        &self,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        keep_multisets: bool,
        compute_histogram: bool,
    ) -> Result<NablaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;

        unsafe {
            let mut hsize: c_int = 0;
            let mut ms_ptr: *mut c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;
            let h = ffi::dress_nabla_fit(
                self.g,
                k,
                max_iterations,
                epsilon,
                n_samples,
                seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_tup,
            );

            let histogram = histogram_from_raw(h, hsize);

            extern "C" { fn free(ptr: *mut std::ffi::c_void); }

            let multisets = if keep_multisets && !ms_ptr.is_null() && num_tup > 0 {
                let len = (num_tup as usize) * e;
                let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
                free(ms_ptr as *mut std::ffi::c_void);
                Some(ms)
            } else {
                if keep_multisets && !ms_ptr.is_null() {
                    free(ms_ptr as *mut std::ffi::c_void);
                }
                None
            };

            if !h.is_null() {
                free(h as *mut std::ffi::c_void);
            }

            Ok(NablaDressResult {
                histogram,
                multisets,
                num_tuples: num_tup,
            })
        }
    }

    /// Explicitly free the underlying C graph.
    pub fn close(&mut self) {
        if !self.g.is_null() {
            unsafe { ffi::dress_free_graph(self.g); }
            self.g = std::ptr::null_mut();
        }
    }
}

impl Drop for DRESS {
    fn drop(&mut self) {
        self.close();
    }
}

/// Result of the Δ^k-DRESS fitting procedure.
#[derive(Debug, Clone)]
pub struct DeltaDressResult {
    /// Sorted exact histogram entries as `(value, count)` pairs.
    pub histogram: Vec<HistogramEntry>,
    /// Per-subgraph edge values, row-major C(N,k) × E.
    /// `NaN` marks edges removed in a given subgraph.
    /// `None` when `keep_multisets` is `false`.
    pub multisets: Option<Vec<f64>>,
    /// Number of subgraphs: C(N,k).
    pub num_subgraphs: i64,
}

/// Result of the ∇^k-DRESS fitting procedure.
#[derive(Debug, Clone)]
pub struct NablaDressResult {
    /// Sorted exact histogram entries as `(value, count)` pairs.
    pub histogram: Vec<HistogramEntry>,
    /// Per-tuple edge values, row-major.
    /// `None` when `keep_multisets` is `false`.
    pub multisets: Option<Vec<f64>>,
    /// Number of tuples.
    pub num_tuples: i64,
}

impl fmt::Display for DeltaDressResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: i64 = self.histogram.iter().map(|entry| entry.count).sum();
        write!(
            f,
            "DeltaDressResult(histogram_entries={}, total_values={})",
            self.histogram.len(), total,
        )
    }
}

impl fmt::Display for NablaDressResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: i64 = self.histogram.iter().map(|entry| entry.count).sum();
        write!(
            f,
            "NablaDressResult(histogram_entries={}, total_values={})",
            self.histogram.len(), total,
        )
    }
}

/// Errors that can occur when building or fitting a DRESS graph.
#[derive(Debug)]
pub enum DressError {
    /// Mismatched array lengths.
    LengthMismatch(String),
    /// The C library returned a null pointer.
    InitFailed,
}

impl fmt::Display for DressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch(msg) => write!(f, "length mismatch: {msg}"),
            Self::InitFailed => write!(f, "dress_init_graph returned NULL"),
        }
    }
}

impl std::error::Error for DressError {}

// ── One-shot free functions ─────────────────────────────────────────

/// One-shot DRESS fit: build graph, fit, return results, free graph.
pub fn fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    max_iterations: i32, epsilon: f64,
) -> Result<DressResult, DressError> {
    let mut g = DRESS::new(n, sources, targets, weights, vertex_weights, variant, precompute)?;
    let (iterations, delta) = g.fit(max_iterations, epsilon);
    let mut r = g.result();
    r.iterations = iterations;
    r.delta = delta;
    Ok(r)
}

/// One-shot Δ^k-DRESS: build graph, run delta fit, return results, free graph.
pub fn delta_fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    k: i32, max_iterations: i32, epsilon: f64,
    n_samples: i32, seed: u32,
    keep_multisets: bool, compute_histogram: bool,
) -> Result<DeltaDressResult, DressError> {
    let g = DRESS::new(n, sources, targets, weights, vertex_weights, variant, precompute)?;
    g.delta_fit(k, max_iterations, epsilon, n_samples, seed,
                keep_multisets, compute_histogram)
}

/// One-shot ∇^k-DRESS: build graph, run nabla fit, return results, free graph.
pub fn nabla_fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    k: i32, max_iterations: i32, epsilon: f64,
    n_samples: i32, seed: u32,
    keep_multisets: bool, compute_histogram: bool,
) -> Result<NablaDressResult, DressError> {
    let g = DRESS::new(n, sources, targets, weights, vertex_weights, variant, precompute)?;
    g.nabla_fit(k, max_iterations, epsilon, n_samples, seed,
                keep_multisets, compute_histogram)
}

// ── Internal helpers ────────────────────────────────────────────────

/// Allocate a C-compatible (malloc'd) copy of an i32 slice.
pub(crate) unsafe fn libc_malloc_copy_i32(data: &[i32]) -> *mut c_int {
    let bytes = data.len() * std::mem::size_of::<c_int>();
    let ptr = libc::malloc(bytes) as *mut c_int;
    assert!(!ptr.is_null(), "malloc failed");
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    ptr
}

/// Allocate a C-compatible (malloc'd) copy of an f64 slice.
pub(crate) unsafe fn libc_malloc_copy_f64(data: &[f64]) -> *mut c_double {
    let bytes = data.len() * std::mem::size_of::<c_double>();
    let ptr = libc::malloc(bytes) as *mut c_double;
    assert!(!ptr.is_null(), "malloc failed");
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    ptr
}

pub(crate) unsafe fn histogram_from_raw(
    data: *mut HistogramEntry,
    hist_size: c_int,
) -> Vec<HistogramEntry> {
    if !data.is_null() && hist_size > 0 {
        std::slice::from_raw_parts(data, hist_size as usize).to_vec()
    } else {
        vec![]
    }
}

// We use libc::malloc — pull in the libc crate minimally via extern.
pub(crate) mod libc {
    extern "C" {
        pub fn malloc(size: usize) -> *mut u8;
    }
}
