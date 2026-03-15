//! MPI-distributed Δ^k-DRESS.
//!
//! All MPI logic (stride partitioning + Allreduce) runs in C.
//! This module is a thin FFI wrapper around `delta_dress_fit_mpi` /
//! `delta_dress_fit_mpi_cuda` from `libdress`.
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
//! println!("hist_size = {}", result.hist_size);
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

use crate::{DeltaDressResult, DressError, Variant};

// ── FFI declarations (C backend) ────────────────────────────────────

#[allow(non_camel_case_types)]
type c_int = i32;
#[allow(non_camel_case_types)]
type c_double = f64;

extern "C" {
    fn init_dress_graph(
        n: c_int, e: c_int,
        u: *mut c_int, v: *mut c_int, w: *mut c_double,
        variant: c_int, precompute_intercepts: c_int,
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

    fn free_dress_graph(g: *mut c_void);

    fn delta_dress_fit_mpi(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double, hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_subgraphs: *mut i64, comm: MPI_Comm,
    ) -> *mut i64;
}

#[cfg(feature = "cuda")]
extern "C" {
    fn delta_dress_fit_mpi_cuda(
        g: *mut c_void, k: c_int, iterations: c_int,
        epsilon: c_double, hist_size: *mut c_int,
        keep_multisets: c_int, multisets: *mut *mut c_double,
        num_subgraphs: *mut i64, comm: MPI_Comm,
    ) -> *mut i64;
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
    k: i32,
    max_iterations: i32,
    epsilon: f64,
    variant: Variant,
    precompute: bool,
    keep_multisets: bool,
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

        let g = init_dress_graph(
            n, e as c_int, u_ptr, v_ptr, w_ptr,
            variant as c_int, precompute as c_int,
        );
        if g.is_null() {
            return Err(DressError::InitFailed);
        }

        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_sub: i64 = 0;

        let h = delta_dress_fit_mpi(
            g, k, max_iterations, epsilon, &mut hsize,
            if keep_multisets { 1 } else { 0 },
            if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
            &mut num_sub,
            comm.as_raw(),
        );

        let result = unpack_result(h, hsize, ms_ptr, num_sub, e, keep_multisets);
        free_dress_graph(g);
        result
    }
}

/// CUDA + MPI submodule.
#[cfg(feature = "cuda")]
pub mod cuda {
    use mpi::traits::*;
    use mpi::ffi::MPI_Comm;
    use crate::{DeltaDressResult, DressError, DressResult, Variant};

    /// MPI-distributed Δ^k-DRESS (CUDA backend).
    ///
    /// All MPI logic (stride partitioning + Allreduce) runs in C.
    pub fn delta_fit<C: Communicator>(
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

            let g = super::init_dress_graph(
                n, e as super::c_int, u_ptr, v_ptr, w_ptr,
                variant as super::c_int, precompute as super::c_int,
            );
            if g.is_null() {
                return Err(DressError::InitFailed);
            }

            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            let h = super::delta_dress_fit_mpi_cuda(
                g, k, max_iterations, epsilon, &mut hsize,
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                comm.as_raw(),
            );

            let result = super::unpack_result(h, hsize, ms_ptr, num_sub, e, keep_multisets);
            super::free_dress_graph(g);
            result
        }
    }

    // ── Persistent MPI+CUDA graph object ────────────────────────────

    #[allow(non_camel_case_types)]
    type c_int = i32;
    #[allow(non_camel_case_types)]
    type c_double = f64;

    extern "C" {
        fn dress_fit_cuda(
            g: *mut std::ffi::c_void,
            max_iterations: c_int,
            epsilon: c_double,
            iterations: *mut c_int,
            delta: *mut c_double,
        );
    }

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
        /// Create a builder (same API as [`crate::DRESS::builder`]).
        pub fn builder(n: i32, sources: Vec<i32>, targets: Vec<i32>) -> super::DRESSBuilder {
            super::DRESSBuilder {
                n,
                sources,
                targets,
                weights: None,
                variant: Variant::Undirected,
                precompute_intercepts: false,
            }
        }

        /// Fit on the GPU via CUDA.  Returns `(iterations, delta)`.
        pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut iterations: c_int = 0;
            let mut delta: c_double = 0.0;
            unsafe {
                dress_fit_cuda(self.g, max_iterations, epsilon, &mut iterations, &mut delta);
            }
            (iterations, delta)
        }

        /// MPI+CUDA distributed Δ^k-DRESS on the persistent graph.
        pub fn delta_fit<CC: Communicator>(
            &self,
            k: i32,
            max_iterations: i32,
            epsilon: f64,
            keep_multisets: bool,
            comm: &CC,
        ) -> Result<DeltaDressResult, DressError> {
            assert!(!self.g.is_null(), "DRESS already closed");
            let mut hsize: super::c_int = 0;
            let mut ms_ptr: *mut super::c_double = std::ptr::null_mut();
            let mut num_sub: i64 = 0;

            unsafe {
                let h = super::delta_dress_fit_mpi_cuda(
                    self.g, k, max_iterations, epsilon, &mut hsize,
                    if keep_multisets { 1 } else { 0 },
                    if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                    &mut num_sub,
                    comm.as_raw(),
                );
                super::unpack_result(h, hsize, ms_ptr, num_sub, self.e, keep_multisets)
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

        /// Explicitly free the underlying C graph.
        pub fn close(&mut self) {
            if !self.g.is_null() {
                unsafe { super::free_dress_graph(self.g); }
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

/// Builder for constructing a persistent MPI DRESS graph.
pub struct DRESSBuilder {
    n:       i32,
    sources: Vec<i32>,
    targets: Vec<i32>,
    weights: Option<Vec<f64>>,
    variant: Variant,
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

    pub fn precompute_intercepts(mut self, p: bool) -> Self {
        self.precompute_intercepts = p;
        self
    }

    /// Build the internal C graph.  Returns a persistent [`DRESS`]
    /// that supports `fit()`, `delta_fit()`, and `get()`.
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

    #[cfg(feature = "cuda")]
    /// Build for MPI+CUDA (returns [`cuda::DRESS`]).
    pub fn build_cuda(self) -> Result<cuda::DRESS, DressError> {
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

            Ok(cuda::DRESS {
                g,
                n: self.n,
                e,
                sources: self.sources,
                targets: self.targets,
            })
        }
    }
}

/// A persistent MPI DRESS graph.
///
/// `fit()` runs on the CPU; `delta_fit()` uses MPI distribution.
/// `get()` runs on the CPU.
pub struct DRESS {
    g: *mut c_void,
    n: i32,
    e: usize,
    sources: Vec<i32>,
    targets: Vec<i32>,
}

impl DRESS {
    /// Create a builder.
    pub fn builder(n: i32, sources: Vec<i32>, targets: Vec<i32>) -> DRESSBuilder {
        DRESSBuilder {
            n,
            sources,
            targets,
            weights: None,
            variant: Variant::Undirected,
            precompute_intercepts: false,
        }
    }

    /// Fit on the CPU.  Returns `(iterations, delta)`.
    pub fn fit(&mut self, max_iterations: i32, epsilon: f64) -> (i32, f64) {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut iterations: c_int = 0;
        let mut delta: c_double = 0.0;
        unsafe {
            dress_fit(self.g, max_iterations, epsilon, &mut iterations, &mut delta);
        }
        (iterations, delta)
    }

    /// MPI-distributed Δ^k-DRESS on the persistent graph.
    pub fn delta_fit<CC: Communicator>(
        &self,
        k: i32,
        max_iterations: i32,
        epsilon: f64,
        keep_multisets: bool,
        comm: &CC,
    ) -> Result<DeltaDressResult, DressError> {
        assert!(!self.g.is_null(), "DRESS already closed");
        let mut hsize: c_int = 0;
        let mut ms_ptr: *mut c_double = std::ptr::null_mut();
        let mut num_sub: i64 = 0;

        unsafe {
            let h = delta_dress_fit_mpi(
                self.g, k, max_iterations, epsilon, &mut hsize,
                if keep_multisets { 1 } else { 0 },
                if keep_multisets { &mut ms_ptr } else { std::ptr::null_mut() },
                &mut num_sub,
                comm.as_raw(),
            );
            unpack_result(h, hsize, ms_ptr, num_sub, self.e, keep_multisets)
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
            let ew_ptr = *(base.add(64) as *const *const f64);
            let ed_ptr = *(base.add(72) as *const *const f64);
            let nd_ptr = *(base.add(88) as *const *const f64);
            crate::DressResult {
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

// ── Internal helper ─────────────────────────────────────────────────

unsafe fn unpack_result(
    h: *mut i64,
    hsize: c_int,
    ms_ptr: *mut c_double,
    num_sub: i64,
    e: usize,
    keep_multisets: bool,
) -> Result<DeltaDressResult, DressError> {
    extern "C" { fn free(ptr: *mut c_void); }

    let histogram = if !h.is_null() && hsize > 0 {
        std::slice::from_raw_parts(h, hsize as usize).to_vec()
    } else {
        vec![]
    };

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
        hist_size: hsize,
        multisets,
        num_subgraphs: num_sub,
    })
}
