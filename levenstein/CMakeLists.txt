cmake_minimum_required(VERSION 3.13)
project(levenstein)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -D_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(USE_SSE TRUE)
set(USE_AVX512 FALSE)

if (USE_SSE AND USE_AVX512)
    message(FATAL_ERROR "Must specify only one of USE_SSE or USE_AVX512 bool variables")
endif ()

if (USE_SSE)
    add_compile_options("-msse4.2")
    add_compile_definitions("USE_SSE")
elseif(USE_AVX512)
    add_compile_options("-mavx512f")
    add_compile_definitions("USE_AVX512")
endif()

set (SOURCES
        du4levenstein.hpp
        du4levenstein.cpp
        exp4main.cpp
        task4.hpp
        testbed.hpp
        )

add_executable(levenstein ${SOURCES})

include(my_main.cmake)

