cmake_minimum_required(VERSION 3.14)
project(matrix_mult)

add_compile_options("-Wall" "-Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE " -O2")
set(CMAKE_CXX_FLAGS_DEBUG " -g ")

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
