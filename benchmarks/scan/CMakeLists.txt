get_filename_component(case_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(lib_name "bm_${case_name}")

set(cpp_sources benchmarks.hip.cpp)
set_source_files_properties(${cpp_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(${lib_name} ${cpp_sources})
target_link_libraries(${lib_name} PUBLIC benchmark_flags)
