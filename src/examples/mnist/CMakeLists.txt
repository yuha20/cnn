cmake_minimum_required(VERSION 3.6)

project(mnist_training CXX)

set (CMAKE_CXX_STANDARD 11)
add_compile_options( -msse3 -mavx -march=native -fopenmp)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
link_libraries(m)
set(SOURCES
    parsing/bmp_image.h
    parsing/bmp_image.cpp
    parsing/parsed_images.h
    parsing/parsed_images.cpp
    parsing/parsed_labels.h
    parsing/parsed_labels.cpp
    parsing/mnist_dataset.h
    parsing/mnist_dataset.cpp)

#add_executable(mnist_training ${SOURCES} mnist_simple.cpp)
#add_executable(mnist_training ${SOURCES} mnist_fc_network.cpp)
add_executable(mnist_training ${SOURCES} mnist_deeplearning.cpp)
add_executable(convolve convolve.cpp)
add_executable(convolve1 convolve1.cpp)
add_executable(convolve_base convolve_base.cpp)
target_include_directories(mnist_training PRIVATE ${YANNPP_SOURCE_DIR})
target_include_directories(mnist_training PRIVATE .)

target_link_libraries(mnist_training yannpp)
