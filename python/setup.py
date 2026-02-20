"""Build the `dress` Python package.

Pure-Python install (no compiler needed):
    pip install .

With native C extension (faster, requires C/C++ compiler + OpenMP):
    DRESS_BUILD_NATIVE=1 pip install .
"""

import os
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# ---------- decide whether to build the C extension ----------
_build_native = os.environ.get("DRESS_BUILD_NATIVE", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRESS_C_SRC = os.path.join(ROOT, "libdress", "src", "dress.c")

ext_modules = []
cmdclass = {}

if _build_native and os.path.isfile(DRESS_C_SRC):
    LIBDRESS_INC = os.path.join(ROOT, "libdress", "include")
    LIBDRESSPP_INC = os.path.join(ROOT, "libdress++", "include")
    DRESS_C_LOCAL = os.path.join(HERE, "_dress.c")
    shutil.copy2(DRESS_C_SRC, DRESS_C_LOCAL)

    class build_ext(_build_ext):
        """Defer pybind11 include lookup until build time."""
        def build_extensions(self):
            import pybind11
            for ext in self.extensions:
                ext.include_dirs.append(pybind11.get_include())
            super().build_extensions()

    ext_modules = [Extension(
        name="dress._core",
        sources=[
            os.path.join("bindings", "dress_pybind11.cpp"),
            "_dress.c",
        ],
        include_dirs=[LIBDRESS_INC, LIBDRESSPP_INC],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-fPIC", "-fopenmp"],
        extra_link_args=["-fopenmp", "-lm"],
    )]
    cmdclass = {"build_ext": build_ext}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)