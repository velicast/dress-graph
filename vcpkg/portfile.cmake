vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO velicast/dress-graph
    REF "v0.3.1"
    SHA512 fc7ea1c7fd96436653297a8246b16ee844caa458b58f7daf48ae1be694c728dfcf40190719a711dd1a8f59241e1d91c9c502db10759855b764ac0640ef43212d
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DDRESS_BUILD_PYTHON=OFF
        -DDRESS_BUILD_IGRAPH=OFF
)

vcpkg_cmake_install()

# Remove duplicate files
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
