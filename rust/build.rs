// build.rs — compile dress.c and delta_dress.c as a static library linked into the crate.
fn main() {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor");

    let mut build = cc::Build::new();
    build
        .file(vendor.join("dress.c"))
        .file(vendor.join("delta_dress.c"))
        .file(vendor.join("delta_dress_impl.c"))
        .include(vendor.join("include"))
        .include(&vendor)
        .opt_level(3)
        .flag_if_supported("-fopenmp");

    // MPI feature: compile dress_mpi.c (needs system MPI headers)
    if std::env::var("CARGO_FEATURE_MPI").is_ok() {
        build.file(vendor.join("mpi/dress_mpi.c"));

        // Find MPI include path via mpicc --showme:compile or pkg-config
        if let Ok(output) = std::process::Command::new("mpicc")
            .args(["--showme:compile"])
            .output()
        {
            let flags = String::from_utf8_lossy(&output.stdout);
            for flag in flags.split_whitespace() {
                if let Some(path) = flag.strip_prefix("-I") {
                    build.include(path);
                }
            }
        }

        // If CUDA is also enabled, define DRESS_CUDA so the MPI source
        // compiles the delta_dress_fit_mpi_cuda path.
        if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
            build.define("DRESS_CUDA", None);
        }
    }

    build.compile("dress");

    // Link OpenMP runtime if the compiler supports it.
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");

    // CUDA feature: statically link pre-compiled CUDA kernel + delta wrapper
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Compile delta_dress_cuda.c (plain C) into the static lib
        let mut cuda_build = cc::Build::new();
        cuda_build
            .file(vendor.join("delta_dress_cuda.c"))
            .include(vendor.join("include"))
            .include(&vendor)
            .opt_level(3)
            .define("DRESS_CUDA", None);
        cuda_build.compile("dress_cuda_c");

        // Link the pre-compiled CUDA kernel object (nvcc output, archived as libdress_cuda.a)
        let src_o = vendor.join("dress_cuda.o");
        if src_o.exists() {
            let out_dir = std::env::var("OUT_DIR").unwrap();
            let dst = std::path::Path::new(&out_dir).join("libdress_cuda_kernel.a");
            std::process::Command::new("ar")
                .args(["rcs", dst.to_str().unwrap(), src_o.to_str().unwrap()])
                .status()
                .expect("failed to archive dress_cuda.o");
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=dress_cuda_kernel");
        }

        // Link static CUDA runtime + its dependencies
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
        }
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

    println!("cargo:rerun-if-changed=vendor/dress.c");
    println!("cargo:rerun-if-changed=vendor/delta_dress.c");
    println!("cargo:rerun-if-changed=vendor/delta_dress_impl.h");
    println!("cargo:rerun-if-changed=vendor/delta_dress_impl.c");
    println!("cargo:rerun-if-changed=vendor/delta_dress_cuda.c");
    println!("cargo:rerun-if-changed=vendor/dress_cuda.o");
    println!("cargo:rerun-if-changed=vendor/mpi/dress_mpi.c");
    println!("cargo:rerun-if-changed=vendor/include/dress/dress.h");
    println!("cargo:rerun-if-changed=vendor/include/dress/delta_dress.h");
    println!("cargo:rerun-if-changed=vendor/include/dress/cuda/dress_cuda.h");
}
