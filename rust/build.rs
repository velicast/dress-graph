// build.rs — compile dress.c, delta_dress.c and nabla_dress.c as a static library linked into the crate.
fn main() {
    cc::Build::new()
        .file("vendor/dress.c")
        .file("vendor/delta_dress.c")
        .file("vendor/nabla_dress.c")
        .include("vendor/include")
        .opt_level(3)
        .flag_if_supported("-fopenmp")
        .compile("dress");

    // Link OpenMP runtime if the compiler supports it.
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rerun-if-changed=vendor/dress.c");
    println!("cargo:rerun-if-changed=vendor/delta_dress.c");
    println!("cargo:rerun-if-changed=vendor/nabla_dress.c");
    println!("cargo:rerun-if-changed=vendor/include/dress/dress.h");
    println!("cargo:rerun-if-changed=vendor/include/dress/delta_dress.h");
    println!("cargo:rerun-if-changed=vendor/include/dress/nabla_dress.h");
}
