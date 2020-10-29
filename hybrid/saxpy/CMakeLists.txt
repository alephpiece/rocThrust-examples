
get_filename_component(case_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Library
set(lib_name "hip_${case_name}")
set(hip_sources ${case_name}.hip.cpp)
set_source_files_properties(${hip_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(${lib_name} ${hip_sources})
set_property(TARGET ${lib_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${lib_name} PRIVATE mpi_hip_flags)

# Executable
set(cpp_sources main.cpp)

hip_add_executable(${case_name} ${cpp_sources})
target_compile_features(${case_name} PRIVATE cxx_std_17)
target_link_libraries(${case_name} PRIVATE ${lib_name} mpi_flags)

