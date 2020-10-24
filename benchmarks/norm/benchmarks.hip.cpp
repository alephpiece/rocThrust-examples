#include <benchmark/benchmark.h>

#include "norm.hip.h"


///----------------------------------------------------------------------------
/// thrust::transform_reduce for norm
///----------------------------------------------------------------------------
template <typename T>
void bm_reduce_norm(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate memory for the device vector
    thrust::device_vector<T> X(N << 20);

    // Fill the vector
    thrust::sequence(X.begin(), X.end());

    for (auto _ : state) {
        benchmark::DoNotOptimize(run_norm(X));
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(bm_reduce_norm, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(1, 1024 * 3);

BENCHMARK_TEMPLATE(bm_reduce_norm, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(1, 1024);

