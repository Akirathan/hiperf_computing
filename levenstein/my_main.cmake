cmake_minimum_required(VERSION 3.13)
project(my_main)

add_executable(my_main
        my_main.cpp
        du4levenstein.hpp
        du4levenstein.cpp
        levenstein_tester_sse.hpp
        levenstein_tester_avx512.hpp
        dummy_levenstein.hpp)
