cmake_minimum_required(VERSION 3.23)
project(Proyecto_final_progra)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Proyecto_final_progra
        grafo.h
        main.cpp
        mnist_vector.h)
