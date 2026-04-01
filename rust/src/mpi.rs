//! MPI-distributed Δ^k-DRESS.
//!
//! All MPI logic (stride partitioning + merge of exact sparse histograms) runs in C.
//! This module is a thin FFI wrapper around `dress_delta_fit_mpi` /
//! `dress_delta_fit_mpi_cuda` from `libdress`.
//!
//! ## CPU + MPI
//!
//! ```no_run
//! use dress_graph::mpi;
//!
//! let universe = mpi::mpi_crate::initialize().unwrap();
//! let world = universe.world();
//!
//! let result = mpi::delta_fit(
//!     4, vec![0,1,2,0,1,2], vec![1,2,0,3,3,3], None,
//!     2, 100, 1e-6,
//!     dress_graph::Variant::Undirected, false, false, &world,
//! ).unwrap();
//!
//! println!("histogram entries = {}", result.histogram.len());
//! ```
//!
//! ## CUDA + MPI
//!
//! ```no_run
//! use dress_graph::mpi::cuda;
//!
//! let universe = dress_graph::mpi::mpi_crate::initialize().unwrap();
//! let world = universe.world();
//!
//! let result = cuda::delta_fit(
//!     4, vec![0,1,2,0,1,2], vec![1,2,0,3,3,3], None,
//!     2, 100, 1e-6,
//!     dress_graph::Variant::Undirected, false, false, &world,
//! ).unwrap();
//! ```

pub use ::mpi as mpi_crate;

use std::ffi::c_void;
use mpi::traits::*;
use mpi::ffi::MPI_Comm;

use crate::{DeltaDressResult, DressError, HistogramEntry, NablaDressResult, Variant};

// ── FFI declarations (C backend) ────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;
#[allow(non_camel_case_types)]
type c_uint = u32;
#[allow(non_camel_case_types)]
type c_double = f64;

#[allow(dead_code)]
extern "C" {
    fn dress_init_graph(
        n: c_int,
        e: c_int,
        u: *mut c_int,
        v: *mut c_int,
        w: *mut c_double,
        nw: *mut c_double,
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

    fn dress_get(
        g: *mut c_void,
        u: c_int,
        v: c_int,
        max_iterations: c_int,
        epsilon: c_double,
        edge_weight: c_double,
    ) -> c_double;

    fn dress_free_graph(g: *mut c_void);

    fn dress_delta_fit_mpi(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_subgraphs: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;

    fn dress_nabla_fit_mpi(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_tuples: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;
}

#[cfg(feature = "cuda")]
extern "C" {
    fn dress_delta_fit_mpi_cuda(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_subgraphs: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;

    fn dress_nabla_fit_mpi_cuda(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_tuples: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;
}

#[cfg(feature = "omp")]
#[allow(dead_code)]
extern "C" {
    fn dress_fit_omp(
        g: *mut c_void, max_iterations: c_int, epsilon: c_double,
        iterations: *mut c_int, delta: *mut c_double,
    );

    fn dress_delta_fit_mpi_omp(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_subgraphs: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;

    fn dress_nabla_fit_mpi_omp(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double,
        n_samples: c_int, seed: c_uint,
        hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_tuples: *mut i64, comm: MPI_Comm,
    ) -> *mut HistogramEntry;
}

// ── CPU + MPI ───────────────────────────────────────────────────────

/// MPI-distributed Δ^k-DRESS (CPU backend).
///
/// All MPI logic (stride partitioning + Allreduce) runs in C.
pub fn delta_fit<C: Communicator>(
    n: i32,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    vertex_weights: Option<Vec<f64>>,
    k: i32,
    max_iterations: i32,
    epsilon: f64,
    n_samples: i32,
    seed: u32,
    variant: Variant,
    precompute: bool,
    keep_multisets: bool,
    compute_histogram: bool,
    comm: &C,
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
        let nw_ptr = match &vertex_weights {
            Some(nw) => crate::libc_malloc_copy_f64(nw),
            None => std::ptr::null_mut(),
        };

        let g = dress_init_graph(
            n, e as c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
            variant as c_int, precompute as c_int,
        );
        if g.is_null() {
            return Err(DressError::InitFailed);
        }

        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_sub: i64 = 0;

        let h = dress_delta_fit_mpi(
            g, k, max_iterations, epsilon,
            n_samples, seed,
            if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
            if keep_multisets { 1 } else { 0 },
            if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
            &mut num_sub,
            comm.as_raw(),
        );

        let result = unpack_result(h, hsize, ms_ptr, num_sub, e, keep_multisets);
        dress_free_graph(g);
        result
    }
}

/// MPI-distributed ∇^k-DRESS (CPU backend).
pub fn nabla_fit<C: Communicator>(
    n: i32,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    vertex_weights: Option<Vec<f64>>,
    k: i32,
    max_iterations: i32,
    epsilon: f64,
    n_samples: i32,
    seed: u32,
    variant: Variant,
    precompute: bool,
    keep_multisets: bool,
    compute_histogram: bool,
    comm: &C,
) -> Result<NablaDressResult, DressError> {
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
        let nw_ptr = match &vertex_weights {
            Some(nw) => crate::libc_malloc_copy_f64(nw),
            None => std::ptr::null_mut(),
        };

        let g = dress_init_graph(
            n, e as c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
            variant as c_int, precompute as c_int,
        );
        if g.is_null() {
            return Err(DressError::InitFailed);
        }

        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_tup: i64 = 0;

        let h = dress_nabla_fit_mpi(
            g, k, max_iterations, epsilon,
            n_samples, seed,
            if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
            if keep_multisets { 1 } else { 0 },
            if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
            &mut num_tup,
            comm.as_raw(),
        );

        let result = unpack_nabla_result(h, hsize, ms_ptr, num_tup, e, keep_multisets);
        dress_free_graph(g);
        result
    }
}

/// CUDA + MPI submodule.
#[cfg(feature = "cuda")]
pub mod cuda {
    use mpi::traits::*;
    use crate::{DeltaDressResult, DressError, DressResult, NablaDressResult, Variant};

    /// MPI-distributed Δ^k-DRESS (CUDA backend).
    ///
    /// All MPI logic (stride partitioning + Allreduce) runs in C.
    pub fn delta_fit<C: Communicator>(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        vertex_weights: Option<Vec<f64>>,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        variant: Variant,
        precompute: bool,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &C,
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
            let nw_ptr = match &vertex_weights {
                Some(nw) => crate::libc_malloc_copy_f64(nw),
                None => std::ptr::null_mut(),
            };

            let g = super::dress_init_graph(
                n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                variant as super::c_int, precompute as super::c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            let h = super::dress_delta_fit_mpi_cuda(
                g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                comm.as_raw(),
            );

            let result = super::unpack_result(h, hsize, ms_ptr, num_sub, e, keep_multisets);
            super::dress_free_graph(g);
            result
        }
    }

    /// MPI-distributed ∇^k-DRESS (CUDA backend).
    pub fn nabla_fit<C: Communicator>(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        vertex_weights: Option<Vec<f64>>,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        variant: Variant,
        precompute: bool,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &C,
    ) -> Result<NablaDressResult, DressError> {
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
            let nw_ptr = match &vertex_weights {
                Some(nw) => crate::libc_malloc_copy_f64(nw),
                None => std::ptr::null_mut(),
            };

            let g = super::dress_init_graph(
                n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                variant as super::c_int, precompute as super::c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;

            let h = super::dress_nabla_fit_mpi_cuda(
                g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_tup,
                comm.as_raw(),
            );

            let result = super::unpack_nabla_result(h, hsize, ms_ptr, num_tup, e, keep_multisets);
            super::dress_free_graph(g);
            result
        }
    }

    // ── Persistent MPI+CUDA graph object ────────────────────────────

    /// A persistent MPI+CUDA DRESS graph.
    ///
    /// `fit()` runs on the GPU via CUDA; `delta_fit()` uses MPI+CUDA.
    /// `get()` runs on the CPU.
    pub struct DRESS {
        pub(super) g: *mut std::ffi::c_void,
        pub(super) n: i32,
        pub(super) e: usize,
        pub(super) sources: Vec<i32>,
        pub(super) targets: Vec<i32>,
    }

    impl DRESS {
        /// Construct a persistent MPI+CUDA DRESS graph.
        pub fn new(
            n: i32, sources: Vec<i32>, targets: Vec<i32>,
            weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
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
                let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));
                let g = super::dress_init_graph(n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                                               variant as super::c_int, precompute_intercepts as super::c_int);
                if g.is_null() { return Err(DressError::InitFailed); }
                Ok(DRESS { g, n, e, sources, targets })
            }
        }

        /// MPI+CUDA distributed Δ^k-DRESS on the persistent graph.
        pub fn delta_fit<CC: Communicator>(
            &self,
            k: i32,
            max_iterations: i32,
            epsilon: f64,
            n_samples: i32,
            seed: u32,
            keep_multisets: bool,
            compute_histogram: bool,
            comm: &CC,
        ) -> Result<DeltaDressResult, DressError> {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            unsafe {
                let h = super::dress_delta_fit_mpi_cuda(
                    self.g, k, max_iterations, epsilon,
                    n_samples, seed,
                    if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                    if keep_multisets { 1 } else { 0 },
                    if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                    &mut num_sub,
                    comm.as_raw(),
                );
                super::unpack_result(h, hsize, ms_ptr, num_sub, self.e, keep_multisets)
            }
        }

        /// MPI+CUDA distributed ∇^k-DRESS on the persistent graph.
        pub fn nabla_fit<CC: Communicator>(
            &self,
            k: i32,
            max_iterations: i32,
            epsilon: f64,
            n_samples: i32,
            seed: u32,
            keep_multisets: bool,
            compute_histogram: bool,
            comm: &CC,
        ) -> Result<NablaDressResult, DressError> {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;

            unsafe {
                let h = super::dress_nabla_fit_mpi_cuda(
                    self.g, k, max_iterations, epsilon,
                    n_samples, seed,
                    if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                    if keep_multisets { 1 } else { 0 },
                    if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                    &mut num_tup,
                    comm.as_raw(),
                );
                super::unpack_nabla_result(h, hsize, ms_ptr, num_tup, self.e, keep_multisets)
            }
        }

        /// Query the DRESS value for an edge (existing or virtual).  Runs on the CPU.
        pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
            assert!(!self.g.is_null(), "DRESS already closed");
            unsafe { super::dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
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
                let nw_c_ptr = *(base.add(104) as *const *const f64);

                let vertex_weights = if !nw_c_ptr.is_null() {
                    Some(std::slice::from_raw_parts(nw_c_ptr, n).to_vec())
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

        /// Explicitly free the underlying C graph.
        pub fn close(&mut self) {
            if !self.g.is_null() {
                unsafe { super::dress_free_graph(self.g); }
                self.g = std::ptr::null_mut();
            }
        }
    }

    impl Drop for DRESS {
        fn drop(&mut self) {
            self.close();
        }
    }
}

// ── Persistent MPI graph object ─────────────────────────────────────

/// A persistent MPI DRESS graph.
pub struct DRESS {
    g: *mut c_void,
    n: i32,
    e: usize,
    sources: Vec<i32>,
    targets: Vec<i32>,
}

impl DRESS {
    /// Construct a persistent MPI DRESS graph.
    pub fn new(
        n: i32, sources: Vec<i32>, targets: Vec<i32>,
        weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
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
            let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));
            let g = dress_init_graph(n, e as c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                                     variant as c_int, precompute_intercepts as c_int);
            if g.is_null() { return Err(DressError::InitFailed); }
            Ok(DRESS { g, n, e, sources, targets })
        }
    }

    /// MPI-distributed Δ^k-DRESS on the persistent graph.
    pub fn delta_fit<CC: Communicator>(
        &self,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &CC,
    ) -> Result<DeltaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_sub: i64 = 0;

        unsafe {
            let h = dress_delta_fit_mpi(
                self.g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                comm.as_raw(),
            );
            unpack_result(h, hsize, ms_ptr, num_sub, self.e, keep_multisets)
        }
    }

    /// MPI-distributed ∇^k-DRESS on the persistent graph.
    pub fn nabla_fit<CC: Communicator>(
        &self,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &CC,
    ) -> Result<NablaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_tup: i64 = 0;

        unsafe {
            let h = dress_nabla_fit_mpi(
                self.g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_tup,
                comm.as_raw(),
            );
            unpack_nabla_result(h, hsize, ms_ptr, num_tup, self.e, keep_multisets)
        }
    }

    /// Query the DRESS value for an edge (existing or virtual).
    pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
        assert!(!self.g.is_null(), "DRESS already closed");
        unsafe { dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
    }

    /// Extract a snapshot of the current results without freeing.
    pub fn result(&self) -> crate::DressResult {
        assert!(!self.g.is_null(), "DRESS already closed");
        let e = self.e;
        let n = self.n as usize;
        unsafe {
            let base = self.g as *const u8;
            let ew_ptr = *(base.add(72) as *const *const f64);
            let ed_ptr = *(base.add(80) as *const *const f64);
            let nd_ptr = *(base.add(96) as *const *const f64);
            let nw_c_ptr = *(base.add(104) as *const *const f64);

            let vertex_weights = if !nw_c_ptr.is_null() {
                Some(std::slice::from_raw_parts(nw_c_ptr, n).to_vec())
            } else {
                None
            };

            crate::DressResult {
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

    /// Explicitly free the underlying C graph.
    pub fn close(&mut self) {
        if !self.g.is_null() {
            unsafe { dress_free_graph(self.g); }
            self.g = std::ptr::null_mut();
        }
    }
}

impl Drop for DRESS {
    fn drop(&mut self) {
        self.close();
    }
}

// ── OMP + MPI submodule ─────────────────────────────────────────────

/// MPI+OMP distributed Δ^k-DRESS.
///
/// MPI distributes subgraphs across ranks; within each rank,
/// OpenMP threads further parallelise the subgraph slice.
///
/// ```no_run
/// use dress_graph::mpi::omp;
///
/// let universe = dress_graph::mpi::mpi_crate::initialize().unwrap();
/// let world = universe.world();
///
/// let result = omp::delta_fit(
///     4, vec![0,1,2,0,1,2], vec![1,2,0,3,3,3], None, None,
///     2, 100, 1e-6,
///     dress_graph::Variant::Undirected, false, false, &world,
/// ).unwrap();
/// ```
#[cfg(feature = "omp")]
pub mod omp {
    use mpi::traits::*;
    use crate::{DeltaDressResult, DressError, DressResult, NablaDressResult, Variant};

    /// MPI+OMP distributed Δ^k-DRESS.
    pub fn delta_fit<C: Communicator>(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        vertex_weights: Option<Vec<f64>>,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        variant: Variant,
        precompute: bool,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &C,
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
            let w_ptr = weights.as_ref().map_or(std::ptr::null_mut(), |w| crate::libc_malloc_copy_f64(w));
            let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));

            let g = super::dress_init_graph(
                n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                variant as super::c_int, precompute as super::c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            let h = super::dress_delta_fit_mpi_omp(
                g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                comm.as_raw(),
            );

            let result = super::unpack_result(h, hsize, ms_ptr, num_sub, e, keep_multisets);
            super::dress_free_graph(g);
            result
        }
    }

    /// MPI+OMP distributed ∇^k-DRESS.
    pub fn nabla_fit<C: Communicator>(
        n: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        weights: Option<Vec<f64>>,
        vertex_weights: Option<Vec<f64>>,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        n_samples: i32,
        seed: u32,
        variant: Variant,
        precompute: bool,
        keep_multisets: bool,
        compute_histogram: bool,
        comm: &C,
    ) -> Result<NablaDressResult, DressError> {
        let e = sources.len();
        if targets.len() != e {
            return Err(DressError::LengthMismatch(
                "sources and targets must have equal length".into(),
            ));
        }

        unsafe {
            let u_ptr = crate::libc_malloc_copy_i32(&sources);
            let v_ptr = crate::libc_malloc_copy_i32(&targets);
            let w_ptr = weights.as_ref().map_or(std::ptr::null_mut(), |w| crate::libc_malloc_copy_f64(w));
            let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));

            let g = super::dress_init_graph(
                n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                variant as super::c_int, precompute as super::c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;

            let h = super::dress_nabla_fit_mpi_omp(
                g, k, max_iterations, epsilon,
                n_samples, seed,
                if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_tup,
                comm.as_raw(),
            );

            let result = super::unpack_nabla_result(h, hsize, ms_ptr, num_tup, e, keep_multisets);
            super::dress_free_graph(g);
            result
        }
    }

    // ── Persistent MPI+OMP graph object ────────────────────────────

    /// A persistent MPI+OMP DRESS graph.
    ///
    /// `fit()` uses OpenMP edge-parallelism; `delta_fit()` uses MPI+OMP.
    pub struct DRESS {
        pub(super) g: *mut std::ffi::c_void,
        pub(super) n: i32,
        pub(super) e: usize,
        pub(super) sources: Vec<i32>,
        pub(super) targets: Vec<i32>,
    }

    impl DRESS {
        /// Construct a persistent MPI+OMP DRESS graph.
        pub fn new(
            n: i32, sources: Vec<i32>, targets: Vec<i32>,
            weights: Option<Vec<f64>>, vertex_weights: Option<Vec<f64>>,
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
                let nw_ptr = vertex_weights.as_ref().map_or(std::ptr::null_mut(), |nw| crate::libc_malloc_copy_f64(nw));
                let g = super::dress_init_graph(n, e as super::c_int, u_ptr, v_ptr, w_ptr, nw_ptr,
                                               variant as super::c_int, precompute_intercepts as super::c_int);
                if g.is_null() { return Err(DressError::InitFailed); }
                Ok(DRESS { g, n, e, sources, targets })
            }
        }

        /// MPI+OMP distributed Δ^k-DRESS.
        pub fn delta_fit<CC: mpi::traits::Communicator>(
            &self, k: i32, max_iterations: i32, epsilon: f64,
            n_samples: i32, seed: u32,
            keep_multisets: bool,
            compute_histogram: bool,
            comm: &CC,
        ) -> Result<DeltaDressResult, DressError> {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            unsafe {
                let h = super::dress_delta_fit_mpi_omp(
                    self.g, k, max_iterations, epsilon,
                    n_samples, seed,
                    if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                    if keep_multisets { 1 } else { 0 },
                    if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                    &mut num_sub, comm.as_raw(),
                );
                super::unpack_result(h, hsize, ms_ptr, num_sub, self.e, keep_multisets)
            }
        }

        /// MPI+OMP distributed ∇^k-DRESS.
        pub fn nabla_fit<CC: mpi::traits::Communicator>(
            &self, k: i32, max_iterations: i32, epsilon: f64,
            n_samples: i32, seed: u32,
            keep_multisets: bool,
            compute_histogram: bool,
            comm: &CC,
        ) -> Result<NablaDressResult, DressError> {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_tup: i64 = 0;

            unsafe {
                let h = super::dress_nabla_fit_mpi_omp(
                    self.g, k, max_iterations, epsilon,
                    n_samples, seed,
                    if compute_histogram { &mut hsize } else { std::ptr::null_mut() },
                    if keep_multisets { 1 } else { 0 },
                    if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                    &mut num_tup, comm.as_raw(),
                );
                super::unpack_nabla_result(h, hsize, ms_ptr, num_tup, self.e, keep_multisets)
            }
        }

        pub fn get(&self, u: i32, v: i32, max_iterations: i32, epsilon: f64, edge_weight: f64) -> f64 {
            assert!(!self.g.is_null(), "DRESS already closed");
            unsafe { super::dress_get(self.g, u, v, max_iterations, epsilon, edge_weight) }
        }

        pub fn result(&self) -> DressResult {
            assert!(!self.g.is_null(), "DRESS already closed");
            let e = self.e;
            let n = self.n as usize;
            unsafe {
                let base = self.g as *const u8;
                let ew_ptr = *(base.add(72) as *const *const f64);
                let ed_ptr = *(base.add(80) as *const *const f64);
                let nd_ptr = *(base.add(96) as *const *const f64);
                let nw_c_ptr = *(base.add(104) as *const *const f64);

                DressResult {
                    sources: self.sources.clone(),
                    targets: self.targets.clone(),
                    edge_weight: std::slice::from_raw_parts(ew_ptr, e).to_vec(),
                    edge_dress: std::slice::from_raw_parts(ed_ptr, e).to_vec(),
                    vertex_dress: std::slice::from_raw_parts(nd_ptr, n).to_vec(),
                    vertex_weights: if !nw_c_ptr.is_null() { Some(std::slice::from_raw_parts(nw_c_ptr, n).to_vec()) } else { None },
                    iterations: 0,
                    delta: 0.0,
                }
            }
        }

        pub fn close(&mut self) {
            if !self.g.is_null() {
                unsafe { super::dress_free_graph(self.g); }
                self.g = std::ptr::null_mut();
            }
        }
    }

    impl Drop for DRESS {
        fn drop(&mut self) { self.close(); }
    }
}

// ── Internal helper ─────────────────────────────────────────────────

unsafe fn unpack_result(
    h: *mut HistogramEntry,
    hsize: c_int,
    ms_ptr: *mut c_double,
    num_sub: i64,
    e: usize,
    keep_multisets: bool,
) -> Result<DeltaDressResult, DressError> {
    extern "C" { fn free(ptr: *mut c_void); }

    let histogram = crate::histogram_from_raw(h, hsize);

    let multisets = if keep_multisets && !ms_ptr.is_null() && num_sub > 0 {
        let len = (num_sub as usize) * e;
        let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
        free(ms_ptr as *mut c_void);
        Some(ms)
    } else {
        if keep_multisets && !ms_ptr.is_null() {
            free(ms_ptr as *mut c_void);
        }
        None
    };

    if !h.is_null() {
        free(h as *mut c_void);
    }

    Ok(DeltaDressResult {
        histogram,
        multisets,
        num_subgraphs: num_sub,
    })
}

unsafe fn unpack_nabla_result(
    h: *mut HistogramEntry,
    hsize: c_int,
    ms_ptr: *mut c_double,
    num_tup: i64,
    e: usize,
    keep_multisets: bool,
) -> Result<NablaDressResult, DressError> {
    extern "C" { fn free(ptr: *mut c_void); }

    let histogram = crate::histogram_from_raw(h, hsize);

    let multisets = if keep_multisets && !ms_ptr.is_null() && num_tup > 0 {
        let len = (num_tup as usize) * e;
        let ms = std::slice::from_raw_parts(ms_ptr, len).to_vec();
        free(ms_ptr as *mut c_void);
        Some(ms)
    } else {
        if keep_multisets && !ms_ptr.is_null() {
            free(ms_ptr as *mut c_void);
        }
        None
    };

    if !h.is_null() {
        free(h as *mut c_void);
    }

    Ok(NablaDressResult {
        histogram,
        multisets,
        num_tuples: num_tup,
    })
}
