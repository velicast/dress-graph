"""Build the ``dress`` Python package.

Always attempts to build the native C extension (pybind11).
Falls back to a pure-Python install if compilation fails (e.g. no compiler).

Force pure-Python only:
    DRESS_PURE_PYTHON=1 pip install .
"""

import os
import platform
import shutil
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── Locate C/C++ sources (vendored first, then repo layout) ─────────

def _find_sources():
    """Return (include_dirs, c_source_files) or None if sources not found."""
    # 1. Vendored inside the package (sdist / cibuildwheel)
    vendored = os.path.join(HERE, "src", "dress", "_vendored")
    if os.path.isfile(os.path.join(vendored, "src", "dress.c")):
        inc = os.path.join(vendored, "include")
        src_dir = os.path.join(vendored, "src")
        c_srcs = [os.path.join(src_dir, f) for f in
                  ("dress.c", "delta_dress.c", "delta_dress_impl.c")
                  if os.path.isfile(os.path.join(src_dir, f))]
        # Make relative to setup.py directory for setuptools
        c_srcs = [os.path.relpath(p, HERE) for p in c_srcs]
        inc = os.path.relpath(inc, HERE)
        return [inc], c_srcs

    # 2. In-tree development (sibling directories in git repo)
    libdress_src = os.path.join(ROOT, "libdress", "src", "dress.c")
    if os.path.isfile(libdress_src):
        incs = [
            os.path.relpath(os.path.join(ROOT, "libdress", "include"), HERE),
            os.path.relpath(os.path.join(ROOT, "libdress++", "include"), HERE),
        ]
        src_dir = os.path.join(ROOT, "libdress", "src")
        c_srcs = [os.path.relpath(os.path.join(src_dir, f), HERE) for f in
                  ("dress.c", "delta_dress.c", "delta_dress_impl.c")
                  if os.path.isfile(os.path.join(src_dir, f))]
        return incs, c_srcs

    return None


# ── Platform-aware compiler/linker flags ─────────────────────────────

def _get_flags():
    """Return (compile_args, link_args) appropriate for the platform."""
    if platform.system() == "Windows":
        return (
            ["/std:c++14", "/O2", "/openmp:llvm", "/EHsc"],
            [],
        )
    elif platform.system() == "Darwin":
        # Apple Clang: use Homebrew libomp if available
        omp_compile = ["-Xpreprocessor", "-fopenmp"]
        omp_link = ["-lomp"]
        try:
            import subprocess
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            omp_compile += [f"-I{prefix}/include"]
            omp_link += [f"-L{prefix}/lib"]
        except Exception:
            # OpenMP not available — build without it (still works, just slower)
            omp_compile = []
            omp_link = []
        return (
            ["-std=c++11", "-O3", "-fPIC"] + omp_compile,
            ["-lm"] + omp_link,
        )
    else:
        # Linux / other POSIX
        return (
            ["-std=c++11", "-O3", "-fPIC", "-fopenmp"],
            ["-fopenmp", "-lm"],
        )


# ── Build logic ──────────────────────────────────────────────────────

class build_ext(_build_ext):
    """Gracefully handle build failures so pip falls back to pure-Python."""

    def build_extensions(self):
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())

        # Ensure C files don't receive C++ standard flags
        orig_compile = self.compiler._compile
        def _compile_filtered(obj, src, ext, cc_args, extra_postargs, pp_opts):
            postargs = list(extra_postargs)
            if src.endswith('.c'):
                postargs = [a for a in postargs
                            if not a.startswith(('-std=c++', '/std:c++', '/EHsc'))]
            return orig_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler._compile = _compile_filtered

        try:
            super().build_extensions()
        except Exception as exc:
            if os.environ.get("DRESS_REQUIRE_NATIVE", "0") == "1":
                raise RuntimeError(f"C extension build failed: {exc}") from exc
            print(f"\n*** C extension build failed: {exc}")
            print("*** Installing pure-Python fallback.\n")

    def run(self):
        try:
            super().run()
        except Exception as exc:
            if os.environ.get("DRESS_REQUIRE_NATIVE", "0") == "1":
                raise RuntimeError(f"C extension build failed: {exc}") from exc
            print(f"\n*** C extension build failed: {exc}")
            print("*** Installing pure-Python fallback.\n")


ext_modules = []
cmdclass = {}

_force_pure = os.environ.get("DRESS_PURE_PYTHON", "0") == "1"

if not _force_pure:
    found = _find_sources()
    if found is not None:
        include_dirs, c_source_files = found
        compile_args, link_args = _get_flags()

        ext_modules = [Extension(
            name="dress._core",
            sources=[
                os.path.join("bindings", "dress_pybind11.cpp"),
                *c_source_files,
            ],
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )]
        cmdclass = {"build_ext": build_ext}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)