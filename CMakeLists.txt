cmake_minimum_required(VERSION 3.14)
project(dress-graph C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DRESS_BUILD_PYTHON "Build Python pybind11 bindings" OFF)
option(DRESS_BUILD_IGRAPH "Build igraph wrapper (requires libigraph)" OFF)

# ── core C library ──────────────────────────────────────────────────
add_subdirectory(libdress)

# ── C++ header-only wrapper ─────────────────────────────────────────
add_subdirectory(libdress++)

# ── igraph wrapper (optional) ───────────────────────────────────────
if(DRESS_BUILD_IGRAPH)
    add_subdirectory(libdress-igraph)
endif()

# ── Python bindings (optional) ──────────────────────────────────────
if(DRESS_BUILD_PYTHON)
    add_subdirectory(python)
endif()
