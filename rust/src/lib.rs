//! # dress-graph
//!
//! Safe Rust bindings for the **DRESS** C library — Diffusive Recursive Structural Similarity on Graphs.  See the [DRESS repository](https://github.com/velicat/dress-graph) for more information.
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

    fn fit(
        g: *mut c_void,
        max_iterations: c_int,
        epsilon: c_double,
        iterations: *mut c_int,
        delta: *mut c_double,
    );

    fn free_dress_graph(g: *mut c_void);
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
            fit(
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
            //   offset 56: *edge_weight
            //   offset 64: *edge_dress
            //   offset 72: *edge_dress_next
            //   offset 80: *node_dress
            let base = g as *const u8;

            let ew_ptr = *(base.add(56) as *const *const f64);
            let ed_ptr = *(base.add(64) as *const *const f64);
            let nd_ptr = *(base.add(80) as *const *const f64);

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

// ── Main entry point ────────────────────────────────────────────────

/// Namespace for static constructors.
pub struct DRESS;

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
            precompute_intercepts: true,
        }
    }
}

// ── Internal helpers ────────────────────────────────────────────────

/// Allocate a C-compatible (malloc'd) copy of an i32 slice.
unsafe fn libc_malloc_copy_i32(data: &[i32]) -> *mut c_int {
    let bytes = data.len() * std::mem::size_of::<c_int>();
    let ptr = libc::malloc(bytes) as *mut c_int;
    assert!(!ptr.is_null(), "malloc failed");
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    ptr
}

/// Allocate a C-compatible (malloc'd) copy of an f64 slice.
unsafe fn libc_malloc_copy_f64(data: &[f64]) -> *mut c_double {
    let bytes = data.len() * std::mem::size_of::<c_double>();
    let ptr = libc::malloc(bytes) as *mut c_double;
    assert!(!ptr.is_null(), "malloc failed");
    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    ptr
}

// We use libc::malloc — pull in the libc crate minimally via extern.
mod libc {
    extern "C" {
        pub fn malloc(size: usize) -> *mut u8;
    }
}
