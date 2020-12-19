This repository contains examples taken from [Nvidia Thrust Guide](https://docs.nvidia.com/cuda/thrust/).

Examples:

- Basic rocThrust APIs
- Hybrid MPI/HIP with robust coloring
- Hybrid MPI/HIP/std::thread with host/device overlapping


# Requirements

- C++17 supported
- ROCm with HIPCC, HCC, and rocThrust < 3.5
- CMake >= 3.15
- MPI >= 3.0
- fmt >= 6.2.1
- google benchmark >= 1.5.2


# Source tree


```
.
|-- cmake
|   `-- SetupHIP.cmake
|-- benchmarks              # examples for rocThrust, up to 16G VRAM
|   |-- CMakeLists.txt
|   |-- run_benchmarks.sh   # script to run all benchmarks
|   |-- dummy.cpp
|   |-- copy
|   |-- norm
|   |-- saxpy
|   |-- scan
|   |-- sort
|   |-- sum
|   `-- utils
`-- hybrid                  # examples for hybrid programming
    |-- CMakeLists.txt
    |-- build.sh            # build all cases
    |-- run.sh              # run some example
    |-- rank_color          # show GPU assigned to each rank
    |-- saxpy               # hybrid MPI/HIP/std::thread, host/device overlapping
    |-- sum                 # MPI_Reduce + thrust::reduce
    `-- utils               # utilities for MPI, HIP, logging, and timing
```

