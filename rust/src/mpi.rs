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
    use crate::{DeltaDressResult, DressError, Variant};

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
