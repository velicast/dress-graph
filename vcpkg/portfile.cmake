vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO velicast/dress-graph
    REF "v0.6.0"
    SHA512 6aa2c65fe3387621359c2a615f128aa610969f64e47766ad43bd172e1ddce469344bb886230ce25758c12688d7e4d19769cce2ae1a53849578973a73d54c959f
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DDRESS_BUILD_PYTHON=OFF
        -DDRESS_BUILD_IGRAPH=OFF
)

vcpkg_cmake_install()

# Remove duplicate filesgit 
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
