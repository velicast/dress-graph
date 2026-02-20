cmake_minimum_required(VERSION 3.14)
project(dress_python CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(dress bindings/dress_pybind11.cpp)

# Link to both the C library and the C++ header-only wrapper
target_link_libraries(dress PRIVATE dress_static dresspp)
target_link_libraries(dress PRIVATE m)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(dress PRIVATE OpenMP::OpenMP_C)
endif()
