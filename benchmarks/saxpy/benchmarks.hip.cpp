#include <benchmark/benchmark.h>

#include "utils/gpu_utils.h"    /* getNumGPUs, hipSetDevice */
#include "saxpy.hip.h"


///----------------------------------------------------------------------------
/// Templated benchmark for saxpy_fast
///----------------------------------------------------------------------------
template <typename T>
void bm_saxpy_fast(benchmark::State &state) {

    // Switch to another GPU
    auto n_gpus = gpuutils::getNumGPUs();
    if (n_gpus > 1)
        hipSetDevice(n_gpus - 1);

    // Number of items (million)
    size_t N = state.range(0);

    // Define scalar A and allocate memory for vector X and Y
    T A = 2.0;
    thrust::device_vector<T> X(N << 20);
    thrust::device_vector<T> Y(N << 20);

    // Fill the vectors
    thrust::sequence(X.begin(), X.end());
    thrust::fill(Y.begin(), Y.end(), 1.);

    for (auto _ : state) {
        run_saxpy_fast(A, X, Y);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


///----------------------------------------------------------------------------
/// Templated benchmark for saxpy_slow
///----------------------------------------------------------------------------
template <typename T>
void bm_saxpy_slow(benchmark::State &state) {

    // Switch to another GPU
    auto n_gpus = gpuutils::getNumGPUs();
    if (n_gpus > 1)
        hipSetDevice(n_gpus - 1);

    // Number of items (million)
    size_t N = state.range(0);

    // Define scalar A and allocate memory for vector X and Y
    T A = 2.0;
    thrust::device_vector<T> X(N << 20);
    thrust::device_vector<T> Y(N << 20);

    // Fill the vectors
    thrust::sequence(X.begin(), X.end());
    thrust::fill(Y.begin(), Y.end(), 1.);

    for (auto _ : state) {
        run_saxpy_slow(A, X, Y);
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(bm_saxpy_fast, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(32, 2000);

BENCHMARK_TEMPLATE(bm_saxpy_fast, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(32, 1000);

BENCHMARK_TEMPLATE(bm_saxpy_slow, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 1024);

BENCHMARK_TEMPLATE(bm_saxpy_slow, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 512);

