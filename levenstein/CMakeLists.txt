cmake_minimum_required(VERSION 3.13)
project(levenstein)

add_compile_options("-Wall" "-msse4.2")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set (SOURCES du4levenstein.cpp
        exp4main.cpp
        du4levenstein.hpp
        task4.hpp
        testbed.hpp
        )

add_executable(levenstein ${SOURCES})

include(my_main.cmake)

