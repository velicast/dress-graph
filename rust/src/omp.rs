//! OpenMP-parallel DRESS — import-based switching.
//!
//! Same API as the CPU [`crate::DRESS`], but `fit()` parallelises edges
//! with OpenMP and `delta_fit()` parallelises subgraphs with OpenMP.
//!
//! ```no_run
//! // CPU
//! use dress_graph::DRESS;
//! let r = DRESS::builder(4, sources.clone(), targets.clone()).build_and_fit().unwrap();
//!
//! // OpenMP — same call, different import
//! use dress_graph::omp::DRESS;
//! let r = DRESS::builder(4, sources, targets).build_and_fit().unwrap();
//! ```

use std::ffi::c_void;

// Re-export result types so callers don't need a separate import.
pub use crate::{DeltaDressResult, DressError, DressResult, HistogramEntry, NablaDressResult, Variant};

// ── FFI declarations ────────────────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;
#[allow(non_camel_case_types)]
type c_uint = u32;
#[allow(non_camel_case_types)]
type c_double = f64;

extern "C" {
    fn dress_init_graph(
        n: c_int, e: c_int,
        u: *mut c_int, v: *mut c_int,
        w: *mut c_double, nw: *mut c_double,
        variant: c_int, precompute_intercepts: c_int,
    ) -> *mut c_void;

    fn dress_fit_omp(
        g: *mut c_void, max_iterations: c_int, epsilon: c_double,
        iterations: *mut c_int, delta: *mut c_double,
    );

    fn dress_free_graph(g: *mut c_void);

    fn dress_get(
        g: *mut c_void, u: c_int, v: c_int,
        max_iterations: c_int, epsilon: c_double, edge_weight: c_double,
    ) -> c_double;

    fn dress_delta_fit_omp_strided(
        g: *mut c_void, k: c_int, iterations: c_int, epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int, keep_multisets: c_int,
        multisets: *mut *mut c_double, num_subgraphs: *mut i64,
        offset: c_int, stride: c_int,
    ) -> *mut HistogramEntry;

    fn dress_nabla_fit_omp(
        g: *mut c_void, k: c_int, iterations: c_int, epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int, keep_multisets: c_int,
        multisets: *mut *mut c_double, num_tuples: *mut i64,
    ) -> *mut HistogramEntry;
}

// ── One-shot free functions ──────────────────────────────────────────

/// One-shot OpenMP-parallel DRESS fit.
pub fn fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, node_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    max_iterations: i32, epsilon: f64,
) -> Result<DressResult, DressError> {
    let mut g = DRESS::new(n, sources, targets, weights, node_weights, variant, precompute)?;
    g.fit(max_iterations, epsilon);
    Ok(g.result())
}

/// One-shot OpenMP-parallel Δ^k-DRESS.
pub fn delta_fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, node_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    k: i32, max_iterations: i32, epsilon: f64,
    n_samples: i32, seed: u32,
    keep_multisets: bool, compute_histogram: bool,
) -> Result<DeltaDressResult, DressError> {
    let g = DRESS::new(n, sources, targets, weights, node_weights, variant, precompute)?;
    g.delta_fit(k, max_iterations, epsilon, n_samples, seed,
                keep_multisets, compute_histogram)
}

/// One-shot OpenMP-parallel ∇^k-DRESS.
pub fn nabla_fit(
    n: i32, sources: Vec<i32>, targets: Vec<i32>,
    weights: Option<Vec<f64>>, node_weights: Option<Vec<f64>>,
    variant: Variant, precompute: bool,
    k: i32, max_iterations: i32, epsilon: f64,
    n_samples: i32, seed: u32,
    keep_multisets: bool, compute_histogram: bool,
) -> Result<NablaDressResult, DressError> {
    let g = DRESS::new(n, sources, targets, weights, node_weights, variant, precompute)?;
    g.nabla_fit(k, max_iterations, epsilon, n_samples, seed,
                keep_multisets, compute_histogram)
}

// ── Persistent OMP graph object ─────────────────────────────────────

/// A persistent DRESS graph whose `fit()` uses OpenMP edge-parallelism
/// and `delta_fit()` uses OpenMP subgraph-parallelism.
pub struct DRESS {
    g: *mut c_void,
    n: i32,
    e: usize,
    sources: Vec<i32>,
    targets: Vec<i32>,
}

impl DRESS {
    /// Construct a persistent DRESS graph (OpenMP backend).
    pub fn new(
        n: i32, sources: Vec<i32>, targets: Vec<i32>,
        weights: Option<Vec<f64>>, node_weights: Option<Vec<f64>>,
        variant: Variant, precompute_intercepts: bool,
    ) -> Result<DRESS, DressError> {
        let e = sources.len();
        if targets.len() != e {
            return Err(DressError::LengthMismatch("sources and targets must have equal length".into()));
        }
        unsafe {
            let u_ptr = crate::libc_malloc_copy_i32(&sources);
            let v_ptr = crate::libc_malloc_copy_i32(&targets);
            let w_ptr = weights.as_ref().map_or(std::ptr::null_mut(), |w| crate::libc_malloc_copy_f64(w));
            let nw_ptr = node_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));
            let g = dress_init_graph(n, e as c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                                     variant as c_int, precompute_intercepts as c_int);
            if g.is_null() { return Err(DressError::InitFailed); }
            Ok(DRESS { g, n, e, sources, targets })
        }
    }

    /// Fit with OpenMP edge-parallelism.  Returns `(iterations, delta)`.
    pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut iterations: c_int = 0;
        let mut delta: c_double = 0.0;
        unsafe { dress_fit_omp(self.g, max_iterations, epsilon, &mut iterations, &mut delta); }
        (iterations, delta)
    }

    /// Query the DRESS value for an edge (existing or virtual).
    pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
        assert!(!self.g.is_null(), "DRESS already closed");
        unsafe { dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
    }

    /// Extract a snapshot of the current results.
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

            DressResult {
                sources: self.sources.clone(),
                targets: self.targets.clone(),
                edge_weight: std::slice::from_raw_parts(ew_ptr, e).to_vec(),
                edge_dress: std::slice::from_raw_parts(ed_ptr, e).to_vec(),
                node_dress: std::slice::from_raw_parts(nd_ptr, n).to_vec(),
                node_weights: if !nw_ptr.is_null() { Some(std::slice::from_raw_parts(nw_ptr, n).to_vec()) } else { None },
                iterations: 0,
                delta: 0.0,
            }
        }
    }

    /// OpenMP-parallel Δ^k-DRESS on the persistent graph.
    pub fn delta_fit(
        &self,
        k: i32, max_iterations: i32, epsilon: f64,
        n_samples: i32, seed: u32,
        keep_multisets: bool, compute_histogram: bool,
    ) -> Result<DeltaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;

        unsafe {
            let mut hsize: c_int = 0;
            let mut ms_ptr: *mut c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;
            let h = dress_delta_fit_omp_strided(
                self.g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub, 0, 1,
            );

            let histogram = crate::histogram_from_raw(h, hsize);

            extern "C" { fn free(ptr: *mut std::ffi::c_void); }

            let multisets = if keep_multisets && !ms_ptr.is_null() && num_sub > 0 {
                let len = (num_sub as usize) * e;
                let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
                free(ms_ptr as *mut std::ffi::c_void);
                Some(ms)
            } else {
                if keep_multisets && !ms_ptr.is_null() { free(ms_ptr as *mut std::ffi::c_void); }
                None
            };

            if !h.is_null() { free(h as *mut std::ffi::c_void); }

            Ok(DeltaDressResult { histogram, multisets, num_subgraphs: num_sub })
        }
    }

    /// OpenMP-parallel ∇^k-DRESS on the persistent graph.
    pub fn nabla_fit(
        &self,
        k: i32, max_iterations: i32, epsilon: f64,
        n_samples: i32, seed: u32,
        keep_multisets: bool, compute_histogram: bool,
    ) -> Result<NablaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;

        unsafe {
            let mut hsize: c_int = 0;
            let mut ms_ptr: *mut c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;
            let h = dress_nabla_fit_omp(
                self.g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_tup,
            );

            let histogram = crate::histogram_from_raw(h, hsize);

            extern "C" { fn free(ptr: *mut std::ffi::c_void); }

            let multisets = if keep_multisets && !ms_ptr.is_null() && num_tup > 0 {
                let len = (num_tup as usize) * e;
                let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
                free(ms_ptr as *mut std::ffi::c_void);
                Some(ms)
            } else {
                if keep_multisets && !ms_ptr.is_null() { free(ms_ptr as *mut std::ffi::c_void); }
                None
            };

            if !h.is_null() { free(h as *mut std::ffi::c_void); }

            Ok(NablaDressResult { histogram, multisets, num_tuples: num_tup })
        }
    }

    pub fn close(&mut self) {
        if !self.g.is_null() {
            unsafe { dress_free_graph(self.g); }
            self.g = std::ptr::null_mut();
        }
    }

}

impl Drop for DRESS {
    fn drop(&mut self) { self.close(); }
}
