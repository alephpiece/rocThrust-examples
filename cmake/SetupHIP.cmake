# ==============================================================================
#
#   Figure out ROCm and HIP paths and find necessary packages.
#
#   setup_hip()
#
#   Variables:
#       ROCM_PATH
#       HIP_PATH
#   Packages:
#       HIP
#       rocprim
#       rocthrust
#   CMake module path:  (in case the environment variables are missing)
#       ${HIP_PATH}/cmake
#
# ==============================================================================
macro(setup_hip)

    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm path")
        elseif(DEFINED ENV{HIP_PATH})
            set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "ROCm path")
        else()
            set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm path")
        endif()
    endif()

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "HIP path")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "HIP path")
        endif()
    endif()

    list(APPEND CMAKE_MODULE_PATH "${HIP_PATH}/cmake")

    # Find HIP, rocprim, rocthrust
    find_package(HIP REQUIRED)
    find_package(rocprim REQUIRED CONFIG)
    find_package(rocthrust REQUIRED CONFIG)

endmacro()
