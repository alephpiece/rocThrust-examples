#!/bin/bash

#=======================================================================
#
#   With SLURM:
#
#       srun -n <nproc> --cpu-bind=<proc_map> ./run.sh <args...>
#
#   With MPI:
#
#       mpirun -n <nproc> --bind-to <proc_map> ./run.sh <args...>
#   or
#       mpirun -n <nproc> --map-by <proc_map> ./run.sh <args...>
#
#   With rocprof:
#
#       mpirun -n 1 --bind-to none rocprof <args...> ./run.sh <args...>
#
#=======================================================================

# Block HIP kernels and asynchronous copies to disable overlapping
#export HIP_LAUNCH_BLOCKING=1
#export HIP_API_BLOCKING=1

BUILD_DIR=build
CASE_NAME=saxpy

RUN_COMMAND="./$BUILD_DIR/$CASE_NAME/$CASE_NAME"

$RUN_COMMAND "$@"
