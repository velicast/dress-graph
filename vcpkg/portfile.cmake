vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO velicast/dress-graph
    REF "v0.5.2"
    SHA512 41d4d663ede8c4678439a989121809a6d999198d1b096c0aa4d1fe260f637e712df7e095952c011e0a73779e7625fab8bc9696ce050ee0e1bf27999e4826ed3b
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
