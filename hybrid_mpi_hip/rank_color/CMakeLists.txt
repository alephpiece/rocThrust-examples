get_filename_component(case_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Executable
set(cpp_sources main.cpp)

hip_add_executable(${case_name} ${cpp_sources})
target_compile_features(${case_name} PRIVATE cxx_std_17)
target_link_libraries(${case_name} PRIVATE mpi_hip_flags)
