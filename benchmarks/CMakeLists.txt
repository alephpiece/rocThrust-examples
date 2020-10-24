cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

# Setup HIP, find necessary packages
include(SetupHIP)
setup_hip()

# Add an interface library for flags
add_library(benchmark_flags INTERFACE)
target_include_directories(benchmark_flags INTERFACE
                            ${PROJECT_SOURCE_DIR}
                            ${ROCM_PATH}/include)

# Add an executable for running benchmarks
hip_add_executable(run_benchmarks dummy.cpp)
target_compile_features(run_benchmarks PRIVATE cxx_std_11)

# Add library gpu_utils
add_subdirectory(utils)

# Link benchmarks
find_package(benchmark REQUIRED)

add_subdirectory(copy)
add_subdirectory(saxpy)
add_subdirectory(norm)
add_subdirectory(scan)
add_subdirectory(sort)
add_subdirectory(sum)
target_link_libraries(run_benchmarks PRIVATE
                      benchmark::benchmark_main
                      gpu_utils
                      bm_copy bm_saxpy bm_norm bm_scan bm_sort bm_sum)
