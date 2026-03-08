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
//! let result = DRESS::builder(4, sources, targets)
//!     .variant(Variant::Undirected)
//!     .max_iterations(100)
//!     .epsilon(1e-6)
//!     .build_and_fit()
//!     .unwrap();
//!
//! println!("iterations: {}", result.iterations);
//! for (i, d) in result.edge_dress.iter().enumerate() {
//!     println!("  edge {}: dress = {:.6}", i, d);
//! }
//! ```

use std::ffi::c_void;
use std::fmt;

#[cfg(feature = "cuda")]
pub mod cuda;

// ── FFI declarations ────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;
#[allow(non_camel_case_types)]
type c_double = f64;

extern "C" {
    fn init_dress_graph(
        n: c_int,
        e: c_int,
        u: *mut c_int,
        v: *mut c_int,
        w: *mut c_double,
        variant: c_int,
        precompute_intercepts: c_int,
    ) -> *mut c_void;

    fn dress_fit(
        g: *mut c_void,
        max_iterations: c_int,
        epsilon: c_double,
        iterations: *mut c_int,
        delta: *mut c_double,
    );

    fn free_dress_graph(g: *mut c_void);

    fn dress_get(
        g: *mut c_void,
        u: c_int,
        v: c_int,
        max_iterations: c_int,
        epsilon: c_double,
        edge_weight: c_double,
    ) -> c_double;

    fn delta_dress_fit(
        g: *mut c_void,
        k: c_int,
        iterations: c_int,
        epsilon: c_double,
        hist_size: *mut c_int,
        keep_multisets: c_int,
        multisets: *mut *mut c_double,
        num_subgraphs: *mut i64,
    ) -> *mut i64;
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
    pub node_dress:  Vec<f64>,
    pub iterations:  i32,
    pub delta:       f64,
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
/// let mut g = DRESS::builder(4, vec![0,1,2,0], vec![1,2,3,3])
///     .variant(Variant::Undirected)
///     .build()
///     .unwrap();
///
/// g.fit(100, 1e-6);
/// let d = g.get(0, 2, 100, 1e-6, 1.0);  // virtual edge query
/// println!("dress(0,2) = {d:.6}");
/// ```
pub struct DRESS {
    g: *mut c_void,
    n: i32,
    e: usize,
    sources: Vec<i32>,
    targets: Vec<i32>,
}

impl DRESS {
    /// Create a builder.
    ///
    /// * `n` – number of vertices (vertex ids in `0..n`)
    /// * `sources` / `targets` – edge list (0-based)
    pub fn builder(n: i32, sources: Vec<i32>, targets: Vec<i32>) -> DRESSBuilder {
        DRESSBuilder {
            n,
            sources,
            targets,
            weights: None,
            variant: Variant::Undirected,
            max_iterations: 100,
            epsilon: 1e-6,
            precompute_intercepts: false,
        }
    }

    /// Fit the DRESS model.  Returns `(iterations, delta)`.
    pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut iterations: c_int = 0;
        let mut delta: c_double = 0.0;
        unsafe {
            dress_fit(self.g, max_iterations, epsilon, &mut iterations, &mut delta);
        }
        (iterations, delta)
    }

    /// Query the DRESS value for an edge (existing or virtual).
    pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
        assert!(!self.g.is_null(), "DRESS already closed");
        unsafe { dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
    }

    /// Extract a snapshot of the current results without freeing.
    pub fn result(&self) -> DressResult {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;
        let n = self.n as usize;
        unsafe {
            let base = self.g as *const u8;
            let ew_ptr = *(base.add(64) as *const *const f64);
            let ed_ptr = *(base.add(72) as *const *const f64);
            let nd_ptr = *(base.add(88) as *const *const f64);
            DressResult {
                sources:     self.sources.clone(),
                targets:     self.targets.clone(),
                edge_weight: std::slice::from_raw_parts(ew_ptr, e).to_vec(),
                edge_dress:  std::slice::from_raw_parts(ed_ptr, e).to_vec(),
                node_dress:  std::slice::from_raw_parts(nd_ptr, n).to_vec(),
                iterations:  0,
                delta:       0.0,
            }
        }
    }

    /// Run Δ^k-DRESS on a graph: enumerate all C(N,k) node-deletion
    /// subsets, fit DRESS on each subgraph, and return the pooled histogram.
    ///
    /// * `n` – number of vertices
    /// * `sources` / `targets` – edge list (0-based)
    /// * `k` – deletion depth (0 = original graph)
    /// * `max_iterations` – max DRESS iterations per subgraph
    /// * `epsilon` – convergence tolerance and bin width
    /// * `variant` – graph variant
    /// * `precompute` – precompute intercepts in subgraphs
    /// * `keep_multisets` – if true, return per-subgraph edge values
    pub fn delta_fit(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        variant: Variant,
        precompute: bool,
        keep_multisets: bool,
    ) -> Result<DeltaDressResult, DressError> {
        let e = sources.len();
        if targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }

        unsafe {
            let u_ptr = libc_malloc_copy_i32(&sources);
            let v_ptr = libc_malloc_copy_i32(&targets);
            let w_ptr = match &weights {
                Some(w) => libc_malloc_copy_f64(w),
                None => std::ptr::null_mut(),
            };

            let g = init_dress_graph(
                n,
                e as c_int,
                u_ptr,
                v_ptr,
                w_ptr,
                variant as c_int,
                precompute as c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: c_int = 0;
            let mut ms_ptr: *mut c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;
            let h = delta_dress_fit(
                g,
                k,
                max_iterations,
                epsilon,
                &mut hsize,
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
            );

            let histogram = if !h.is_null() && hsize > 0 {
                std::slice::from_raw_parts(h, hsize as usize).to_vec()
            } else {
                vec![]
            };

            extern "C" { fn free(ptr: *mut std::ffi::c_void); }

            let multisets = if keep_multisets && !ms_ptr.is_null() && num_sub > 0 {
                let len = (num_sub as usize) * (e as usize);
                let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
                free(ms_ptr as *mut std::ffi::c_void);
                Some(ms)
            } else {
                if keep_multisets && !ms_ptr.is_null() {
                    free(ms_ptr as *mut std::ffi::c_void);
                }
                None
            };

            // Free the C-allocated histogram
            if !h.is_null() {
                free(h as *mut std::ffi::c_void);
            }

            free_dress_graph(g);

            Ok(DeltaDressResult {
                histogram,
                hist_size: hsize,
                multisets,
                num_subgraphs: num_sub,
            })
        }
    }

    /// Explicitly free the underlying C graph.
    pub fn close(&mut self) {
        if !self.g.is_null() {
            unsafe { free_dress_graph(self.g); }
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
    /// Histogram bin counts (length = `hist_size`).
    pub histogram: Vec<i64>,
    /// Number of bins: floor(dmax/epsilon) + 1 (dmax = 2 for unweighted graphs).
    pub hist_size: i32,
    /// Per-subgraph edge values, row-major C(N,k) × E.
    /// `NaN` marks edges removed in a given subgraph.
    /// `None` when `keep_multisets` is `false`.
    pub multisets: Option<Vec<f64>>,
    /// Number of subgraphs: C(N,k).
    pub num_subgraphs: i64,
}

impl fmt::Display for DeltaDressResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total: i64 = self.histogram.iter().sum();
        write!(
            f,
            "DeltaDressResult(hist_size={}, total_values={})",
            self.hist_size, total,
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
            Self::InitFailed => write!(f, "init_dress_graph returned NULL"),
        }
    }
}

impl std::error::Error for DressError {}

// ── Builder ─────────────────────────────────────────────────────────

/// Ergonomic builder for constructing and fitting a DRESS graph.
///
/// ```no_run
/// use dress_graph::{DRESS, Variant};
///
/// let r = DRESS::builder(4, vec![0,1,2,0], vec![1,2,3,3])
///     .variant(Variant::Undirected)
///     .build_and_fit()
///     .unwrap();
/// ```
pub struct DRESSBuilder {
    n:       i32,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    variant: Variant,
    max_iterations: i32,
    epsilon: f64,
    precompute_intercepts: bool,
}

impl DRESSBuilder {
    pub fn weights(mut self, w: Vec<f64>) -> Self {
        self.weights = Some(w);
        self
    }

    pub fn variant(mut self, v: Variant) -> Self {
        self.variant = v;
        self
    }

    pub fn max_iterations(mut self, i: i32) -> Self {
        self.max_iterations = i;
        self
    }

    pub fn epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
        self
    }

    pub fn precompute_intercepts(mut self, p: bool) -> Self {
        self.precompute_intercepts = p;
        self
    }

    /// Build the internal C graph without fitting.  Returns a persistent
    /// `DRESS` that supports repeated `fit` and `get` calls.
    pub fn build(self) -> Result<DRESS, DressError> {
        let e = self.sources.len();
        if self.targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }
        if let Some(ref w) = self.weights {
            if w.len() != e {
                return Err(DressError::LengthMismatch(
                    "weights must have the same length as sources".into(),
                ));
            }
        }

        unsafe {
            let u_ptr = libc_malloc_copy_i32(&self.sources);
            let v_ptr = libc_malloc_copy_i32(&self.targets);
            let w_ptr = match &self.weights {
                Some(w) => libc_malloc_copy_f64(w),
                None => std::ptr::null_mut(),
            };

            let g = init_dress_graph(
                self.n,
                e as c_int,
                u_ptr,
                v_ptr,
                w_ptr,
                self.variant as c_int,
                self.precompute_intercepts as c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            Ok(DRESS {
                g,
                n: self.n,
                e,
                sources: self.sources,
                targets: self.targets,
            })
        }
    }

    /// Build the internal C graph, run the fitting algorithm, and return
    /// an owned `DressResult`.  The C graph is freed before returning.
    pub fn build_and_fit(self) -> Result<DressResult, DressError> {
        let e = self.sources.len();
        if self.targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }
        if let Some(ref w) = self.weights {
            if w.len() != e {
                return Err(DressError::LengthMismatch(
                    "weights must have the same length as sources".into(),
                ));
            }
        }

        // The C library takes ownership of U, V, W via free().
        // We allocate with libc::malloc so free() is safe.
        let n_c = self.n;
        let e_c = e as c_int;

        unsafe {
            let u_ptr = libc_malloc_copy_i32(&self.sources);
            let v_ptr = libc_malloc_copy_i32(&self.targets);
            let w_ptr = match &self.weights {
                Some(w) => libc_malloc_copy_f64(w),
                None => std::ptr::null_mut(),
            };

            let g = init_dress_graph(
                n_c,
                e_c,
                u_ptr,
                v_ptr,
                w_ptr,
                self.variant as c_int,
                self.precompute_intercepts as c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut iterations: c_int = 0;
            let mut delta: c_double = 0.0;
            dress_fit(
                g,
                self.max_iterations,
                self.epsilon,
                &mut iterations,
                &mut delta,
            );

            // Read results from the C struct before freeing.
            // Struct layout (LP64):
            //   offset  0: variant   (i32)
            //   offset  4: N         (i32)
            //   offset  8: E         (i32)
            //   offset 12: <pad 4>
            //   offset 16: *U        (ptr)
            //   offset 24: *V        (ptr)
            //   offset 32: *adj_offset
            //   offset 40: *adj_target
            //   offset 48: *adj_edge_idx
            //   offset 56: *W             (raw input weights)
            //   offset 64: *edge_weight
            //   offset 72: *edge_dress
            //   offset 80: *edge_dress_next
            //   offset 88: *node_dress
            let base = g as *const u8;

            let ew_ptr = *(base.add(64) as *const *const f64);
            let ed_ptr = *(base.add(72) as *const *const f64);
            let nd_ptr = *(base.add(88) as *const *const f64);

            let edge_weight = std::slice::from_raw_parts(ew_ptr, e).to_vec();
            let edge_dress  = std::slice::from_raw_parts(ed_ptr, e).to_vec();
            let node_dress  = std::slice::from_raw_parts(nd_ptr, n_c as usize).to_vec();

            // We need our own copies of sources/targets because free_dress_graph
            // frees U and V.
            let sources_out = self.sources.clone();
            let targets_out = self.targets.clone();

            free_dress_graph(g);

            Ok(DressResult {
                sources: sources_out,
                targets: targets_out,
                edge_weight,
                edge_dress,
                node_dress,
                iterations,
                delta,
            })
        }
    }
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

// We use libc::malloc — pull in the libc crate minimally via extern.
pub(crate) mod libc {
    extern "C" {
        pub fn malloc(size: usize) -> *mut u8;
    }
}
