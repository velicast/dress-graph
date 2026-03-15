vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO velicast/dress-graph
    REF "v0.5.3"
    SHA512 e8da994407ba6189194b49390730fbc30a93dbcf5a737c21f740b59caf5eecc571003825be1b953d4c8f781d9d025f0baa07cb1f0b89f4a480dea030070c5590
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
