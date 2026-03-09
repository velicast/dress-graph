vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO velicast/dress-graph
    REF "v0.5.0"
    SHA512 6e02879190b7506cdd389678de5b1c0b20578033d1384c29d85fa3f304d2de2c21e443b5d26a2ca058f0155d780683114ee9eb68235a55db2250eb6d343fb271
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
