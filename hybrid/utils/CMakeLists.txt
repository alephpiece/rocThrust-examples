#========================================
# Libraries:
#   gpu_utils
#   mpi_utils
#========================================
set(cpp_sources gpu_utils.hip.cpp)
set_source_files_properties(${cpp_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(gpu_utils gpu_utils.hip.cpp)
target_link_libraries(gpu_utils PRIVATE hip_flags)


add_library(mpi_utils mpi_utils.cpp)
target_compile_definitions(mpi_utils PUBLIC USE_MPI_IN_PLACE)
target_link_libraries(mpi_utils PUBLIC deps_flags MPI::MPI_C MPI::MPI_CXX)
