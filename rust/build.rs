// build.rs — compile dress.c as a static library linked into the crate.
fn main() {
    cc::Build::new()
        .file("vendor/dress.c")
        .include("vendor/include")
        .opt_level(3)
        .flag_if_supported("-fopenmp")
        .compile("dress");

    // Link OpenMP runtime if the compiler supports it.
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rerun-if-changed=vendor/dress.c");
    println!("cargo:rerun-if-changed=vendor/include/dress/dress.h");
}
