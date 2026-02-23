cmake_minimum_required(VERSION 3.14)
project(dress-igraph C)

set(CMAKE_C_STANDARD 11)

# ── find igraph ─────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(IGRAPH REQUIRED igraph)

# ── libdress-igraph (static + shared) ───────────────────────────────

add_library(dress_igraph_static STATIC src/dress_igraph.c)
target_include_directories(dress_igraph_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(dress_igraph_static SYSTEM PUBLIC ${IGRAPH_INCLUDE_DIRS})
target_link_libraries(dress_igraph_static PUBLIC dress::static ${IGRAPH_LIBRARIES})
set_target_properties(dress_igraph_static PROPERTIES OUTPUT_NAME dress_igraph)

add_library(dress_igraph_shared SHARED src/dress_igraph.c)
target_include_directories(dress_igraph_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_include_directories(dress_igraph_shared SYSTEM PUBLIC ${IGRAPH_INCLUDE_DIRS})
target_link_libraries(dress_igraph_shared PUBLIC dress::shared ${IGRAPH_LIBRARIES})
set_target_properties(dress_igraph_shared PROPERTIES OUTPUT_NAME dress_igraph)

target_compile_options(dress_igraph_static PRIVATE -O3)
target_compile_options(dress_igraph_shared PRIVATE -O3)

# ── alias targets ───────────────────────────────────────────────────
add_library(dress_igraph::static ALIAS dress_igraph_static)
add_library(dress_igraph::shared ALIAS dress_igraph_shared)
add_library(dress_igraph ALIAS dress_igraph_static)

# ── install ─────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS dress_igraph_static dress_igraph_shared
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/dress_igraph.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
