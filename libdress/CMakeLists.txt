cmake_minimum_required(VERSION 3.14)
project(dress C)

set(CMAKE_C_STANDARD 11)

# ── libdress (static + shared) ──────────────────────────────────────

add_library(dress_static STATIC src/dress.c src/delta_dress.c)
target_include_directories(dress_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(dress_static PROPERTIES OUTPUT_NAME dress)

add_library(dress_shared SHARED src/dress.c src/delta_dress.c)
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

# CUDA (optional)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(dress_static PRIVATE DRESS_CUDA)
    target_compile_definitions(dress_shared PRIVATE DRESS_CUDA)
    target_link_libraries(dress_static PRIVATE CUDA::cudart)
    target_link_libraries(dress_shared PRIVATE CUDA::cudart)
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
