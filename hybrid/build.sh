#!/bin/bash

set -e

#export HIPCC_VERBOSE=7

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLOAT=float

cmake --build build -j8

