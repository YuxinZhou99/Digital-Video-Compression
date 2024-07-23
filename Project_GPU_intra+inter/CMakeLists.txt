cmake_minimum_required(VERSION 3.26)
project(Project CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Project main.cu
        preprocessing.cpp
        predict_motion.cpp
        decoder.cpp
        write_to_txt.cpp)

set_target_properties(Project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
