#!/bin/bash

set -e

#export HIPCC_VERBOSE=7

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx

cmake --build build -j8

