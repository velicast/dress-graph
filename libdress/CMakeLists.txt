cmake_minimum_required(VERSION 3.14)
project(dress C)

set(CMAKE_C_STANDARD 11)

# ── libdress (static + shared) ──────────────────────────────────────

add_library(dress_static STATIC src/dress.c src/delta_dress.c src/delta_dress_impl.c)
target_include_directories(dress_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(dress_static PROPERTIES OUTPUT_NAME dress)

add_library(dress_shared SHARED src/dress.c src/delta_dress.c src/delta_dress_impl.c)
target_include_directories(dress_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(dress_shared PROPERTIES OUTPUT_NAME dress)
target_link_libraries(dress_shared PRIVATE m)

# OpenMP (optional)
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(dress_static PUBLIC OpenMP::OpenMP_C)
    target_link_libraries(dress_shared PUBLIC OpenMP::OpenMP_C)
endif()

# CUDA (optional — detection only, for downstream targets.
# libdress itself does NOT link CUDA; GPU code lives in libdress_cuda.)
find_package(CUDAToolkit QUIET)

# MPI (optional — only enabled when -DDRESS_MPI=ON is passed.
# The MPI headers in include/dress/mpi/ are usable without linking MPI
# to libdress itself, but we expose MPI::MPI_C so downstream targets
# that #include dress/mpi/dress_mpi.h get the MPI include path.)
option(DRESS_MPI "Enable MPI support" OFF)
if(DRESS_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    target_sources(dress_static PRIVATE src/mpi/dress_mpi.c)
    target_sources(dress_shared PRIVATE src/mpi/dress_mpi.c)
    target_link_libraries(dress_static PUBLIC MPI::MPI_C)
    target_link_libraries(dress_shared PUBLIC MPI::MPI_C)
endif()

target_compile_options(dress_static PRIVATE -O3)
target_compile_options(dress_shared PRIVATE -O3)

# ── alias targets ───────────────────────────────────────────────────
add_library(dress::static ALIAS dress_static)
add_library(dress::shared ALIAS dress_shared)

# Convenience target name
add_library(dress ALIAS dress_static)

# ── install ─────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS dress_static dress_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/dress DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
