#include <benchmark/benchmark.h>

#include "scan.hip.h"


///----------------------------------------------------------------------------
/// thrust::inclusive_scan
///----------------------------------------------------------------------------
template <typename T>
void bm_inclusive_scan(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate a device vector
    thrust::device_vector<T> X(N<< 20);

    // Fill the vector
    thrust::sequence(X.begin(), X.end());

    for (auto _ : state) {
        run_inclusive_scan(X);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


///----------------------------------------------------------------------------
/// thrust::exclusive_scan
///----------------------------------------------------------------------------
template <typename T>
void bm_exclusive_scan(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate a device vector
    thrust::device_vector<T> X(N<< 20);

    // Fill the vector.
    thrust::sequence(X.begin(), X.end());

    for (auto _ : state) {
        run_exclusive_scan(X);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(bm_inclusive_scan, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 1024);

BENCHMARK_TEMPLATE(bm_inclusive_scan, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 512);

BENCHMARK_TEMPLATE(bm_exclusive_scan, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 1024);

BENCHMARK_TEMPLATE(bm_exclusive_scan, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 512);


