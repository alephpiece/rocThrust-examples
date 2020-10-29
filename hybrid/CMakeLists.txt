cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

# Setup HIP, find necessary packages.
include(SetupHIP)
setup_hip()

find_package(MPI 3.0 REQUIRED COMPONENTS C CXX)
find_package(fmt REQUIRED)

#========================================
# Options
#========================================
set(FLOAT "double" CACHE STRING "Floating-point type")

#========================================
# Utilities
#   gpu_utils
#   mpi_utils
#========================================
add_subdirectory(utils)

#========================================
# Interface libraries:
#   deps_flags
#   hip_flags
#   mpi_flags
#   mpi_hip_flags
#========================================
add_library(deps_flags INTERFACE)
target_include_directories(deps_flags INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(deps_flags INTERFACE fmt::fmt)
target_compile_definitions(deps_flags INTERFACE FLOAT=${FLOAT})

add_library(hip_flags INTERFACE)
target_include_directories(hip_flags INTERFACE ${ROCM_PATH}/include)
target_link_libraries(hip_flags INTERFACE deps_flags gpu_utils)

add_library(mpi_flags INTERFACE)
target_link_libraries(mpi_flags INTERFACE deps_flags mpi_utils)

add_library(mpi_hip_flags INTERFACE)
target_link_libraries(mpi_hip_flags INTERFACE hip_flags mpi_utils)

#========================================
# Executables:
#   saxpy
#   sum
#   rank_color
#========================================
add_subdirectory(saxpy)
add_subdirectory(sum)
add_subdirectory(rank_color)
