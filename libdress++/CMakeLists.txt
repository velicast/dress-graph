cmake_minimum_required(VERSION 3.14)
project(dress++ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libdress++ is header-only — it wraps the C library via dress.hpp
add_library(dresspp INTERFACE)
target_include_directories(dresspp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link to the C library
if(TARGET dress_static)
    target_link_libraries(dresspp INTERFACE dress_static)
else()
    message(WARNING "dress_static target not found; add libdress first via add_subdirectory()")
endif()

# Alias
add_library(dress::hpp ALIAS dresspp)

# ── install ─────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/dress DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
