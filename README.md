# Examples for rocThrust

This repository contains examples taken from [Nvidia Thrust Guide](https://docs.nvidia.com/cuda/thrust/).


## Requirements

- ROCm with HIPCC, HCC, and rocThrust
- MPI
- fmt
- google benchmark


## Source tree


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
`-- hybrid_mpi_hip          # examples for hybrid MPI/HIP programming
    |-- CMakeLists.txt
    |-- build.sh            # build all cases
    |-- run.sh              # run some example
    |-- rank_color
    |-- saxpy
    |-- sum
    `-- utils
```

