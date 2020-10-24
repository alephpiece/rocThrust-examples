# Library gpu_utils
set(cpp_sources gpu_utils.hip.cpp)
set_source_files_properties(${cpp_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(gpu_utils gpu_utils.hip.cpp)
target_link_libraries(gpu_utils PRIVATE hip_flags)
