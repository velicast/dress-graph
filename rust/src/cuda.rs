//! GPU-accelerated DRESS — import-based switching.
//!
//! Same API as the CPU [`crate::DRESS`], but fitting runs on the GPU via CUDA.
//!
//! ```no_run
//! // CPU
//! use dress_graph::DRESS;
//! let r = DRESS::builder(4, sources.clone(), targets.clone()).build_and_fit().unwrap();
//!
//! // CUDA — same call, different import
//! use dress_graph::cuda::DRESS;
//! let r = DRESS::builder(4, sources, targets).build_and_fit().unwrap();
//! ```

use std::ffi::c_void;

// Re-export result types so callers don't need a separate import.
pub use crate::{DeltaDressResult, DressError, DressResult, Variant};

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

    fn dress_fit_cuda(
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

    fn delta_dress_fit_cuda_strided(
        g: *mut c_void,
        k: c_int,
        iterations: c_int,
        epsilon: c_double,
        hist_size: *mut c_int,
        keep_multisets: c_int,
        multisets: *mut *mut c_double,
        num_subgraphs: *mut i64,
        offset: c_int,
        stride: c_int,
    ) -> *mut i64;
}

// ── Builder ─────────────────────────────────────────────────────────

/// Ergonomic builder for constructing and fitting a DRESS graph on the GPU.
///
/// Identical API to [`crate::DRESSBuilder`].
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
    /// `DRESS` whose `fit()` runs on the GPU.
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
            let u_ptr = crate::libc_malloc_copy_i32(&self.sources);
            let v_ptr = crate::libc_malloc_copy_i32(&self.targets);
            let w_ptr = match &self.weights {
                Some(w) => crate::libc_malloc_copy_f64(w),
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

    /// Build the internal C graph, run CUDA fitting, and return an owned
    /// [`DressResult`].  The C graph is freed before returning.
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

        let n_c = self.n;
        let e_c = e as c_int;

        unsafe {
            let u_ptr = crate::libc_malloc_copy_i32(&self.sources);
            let v_ptr = crate::libc_malloc_copy_i32(&self.targets);
            let w_ptr = match &self.weights {
                Some(w) => crate::libc_malloc_copy_f64(w),
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
            dress_fit_cuda(
                g,
                self.max_iterations,
                self.epsilon,
                &mut iterations,
                &mut delta,
            );

            let base = g as *const u8;

            let ew_ptr = *(base.add(72) as *const *const f64);
            let ed_ptr = *(base.add(80) as *const *const f64);
            let nd_ptr = *(base.add(96) as *const *const f64);

            let edge_weight = std::slice::from_raw_parts(ew_ptr, e).to_vec();
            let edge_dress  = std::slice::from_raw_parts(ed_ptr, e).to_vec();
            let node_dress  = std::slice::from_raw_parts(nd_ptr, n_c as usize).to_vec();

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

// ── Persistent CUDA graph object ────────────────────────────────────

/// A persistent DRESS graph whose `fit()` runs on the GPU via CUDA.
///
/// `get()` runs on the CPU (same as the CPU variant).
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

    /// Fit the DRESS model on the GPU.  Returns `(iterations, delta)`.
    pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut iterations: c_int = 0;
        let mut delta: c_double = 0.0;
        unsafe {
            dress_fit_cuda(self.g, max_iterations, epsilon, &mut iterations, &mut delta);
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
            let ew_ptr = *(base.add(72) as *const *const f64);
            let ed_ptr = *(base.add(80) as *const *const f64);
            let nd_ptr = *(base.add(96) as *const *const f64);
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

    /// GPU-accelerated Δ^k-DRESS (same signature as [`crate::DRESS::delta_fit`]).
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
        offset: i32,
        stride: i32,
    ) -> Result<DeltaDressResult, DressError> {
        let e = sources.len();
        if targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }

        unsafe {
            let u_ptr = crate::libc_malloc_copy_i32(&sources);
            let v_ptr = crate::libc_malloc_copy_i32(&targets);
            let w_ptr = match &weights {
                Some(w) => crate::libc_malloc_copy_f64(w),
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
            let h = delta_dress_fit_cuda_strided(
                g,
                k,
                max_iterations,
                epsilon,
                &mut hsize,
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                offset,
                stride,
            );

            let histogram = if !h.is_null() && hsize > 0 {
                std::slice::from_raw_parts(h, hsize as usize).to_vec()
            } else {
                vec![]
            };

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
