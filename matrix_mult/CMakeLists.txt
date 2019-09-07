cmake_minimum_required(VERSION 3.14)
project(matrix_mult)

add_compile_options("-Wall" "-Wextra")
set(CLANG_FLAGS "-Rpass-analysis=loop-vectorize -Rpass=loop-vectorize")
set(GCC_FLAGS "-march=native -fopt-info-vec-all")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 ")
string(APPEND CMAKE_CXX_FLAGS_RELEASE ${CLANG_FLAGS})

set(CMAKE_CXX_FLAGS_DEBUG " -g -D _DEBUG ")
string(APPEND CMAKE_CXX_FLAGS_DEBUG ${CLANG_FLAGS})



add_compile_definitions("BOOST_ALL_DYN_LINK")
find_package(Boost COMPONENTS log system unit_test_framework REQUIRED)
include_directories()

set(SOURCES
        du5main.cpp
        du5matrix.hpp
        du5matrix.cpp
        task5.hpp
        testbed.hpp
        )

add_executable(matrix_mult
        ${SOURCES}
        )

target_link_libraries(matrix_mult)

add_executable(my_try
        try.cpp
        )
