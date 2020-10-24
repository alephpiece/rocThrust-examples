#!/bin/bash

set -e

#export HIPCC_VERBOSE=7
#export HIP_DB=0xf

BUILD_DIR=build
RUN_COMMAND="./$BUILD_DIR/run_benchmarks --benchmark_min_time=1"

# Build benchmarks
cmake -S . -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release
cmake --build $BUILD_DIR -j8

# Run benchmarks
$RUN_COMMAND "$@"

