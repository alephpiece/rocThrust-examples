cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

# Setup HIP, find necessary packages.
include(SetupHIP)
setup_hip()

find_package(MPI 3.0 REQUIRED COMPONENTS C CXX)
find_package(fmt REQUIRED)

# Add utilities.
add_subdirectory(utils)

# Add an interface library for MPI.
add_library(mpi_flags INTERFACE)
target_include_directories(mpi_flags INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_definitions(mpi_flags INTERFACE USE_MPI_IN_PLACE)
target_link_libraries(mpi_flags INTERFACE MPI::MPI_C MPI::MPI_CXX fmt::fmt)

# Add an interface library for HIP and linked with gpu_utils.
add_library(hip_flags INTERFACE)
target_include_directories(hip_flags INTERFACE ${PROJECT_SOURCE_DIR} ${ROCM_PATH}/include)
target_link_libraries(hip_flags INTERFACE gpu_utils)

# Add an interface library for hybrid MPI-HIP.
add_library(mpi_hip_flags INTERFACE)
target_link_libraries(mpi_hip_flags INTERFACE mpi_flags hip_flags)

# Add executables for testing.
add_subdirectory(rank_color)
add_subdirectory(sum)
add_subdirectory(saxpy)
