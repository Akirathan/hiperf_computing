cmake_minimum_required(VERSION 3.13)
project(my_main)

#set(CMAKE_CXX_COMPILER clang++-7)
#add_compile_options("-Wall" "-msse4.2")
#set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(my_main
        my_main.cpp
        du4levenstein.hpp
        du4levenstein.cpp
        dummy_levenstein.hpp)
