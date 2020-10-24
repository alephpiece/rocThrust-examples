#include <benchmark/benchmark.h>

#include "copy.hip.h"


///----------------------------------------------------------------------------
/// Templated benchmark for copy_h2d
///----------------------------------------------------------------------------
template <typename T>
void bm_copy_h2d(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate memory for host and device vectors
    thrust::host_vector<T>   host_X(N << 20, 0.5);
    thrust::device_vector<T> dev_X(N << 20);

    for (auto _ : state) {
        run_copy_h2d(host_X, dev_X);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


///----------------------------------------------------------------------------
/// Templated benchmark for copy_d2h
///----------------------------------------------------------------------------
template <typename T>
void bm_copy_d2h(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate memory for host and device vectors
    thrust::device_vector<T> dev_X(N << 20, 0.5);
    thrust::host_vector<T>   host_X(N << 20);

    for (auto _ : state) {
        run_copy_d2h(dev_X, host_X);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(bm_copy_h2d, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(2, 4000);

BENCHMARK_TEMPLATE(bm_copy_h2d, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(2, 2000);

BENCHMARK_TEMPLATE(bm_copy_d2h, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(2, 4000);

BENCHMARK_TEMPLATE(bm_copy_d2h, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(2, 2000);
