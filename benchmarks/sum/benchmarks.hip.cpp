#include <benchmark/benchmark.h>

#include "sum.hip.h"


///----------------------------------------------------------------------------
/// thrust::reduce for sum
///----------------------------------------------------------------------------
template <typename T>
void reduce_sum(benchmark::State &state) {

    // Number of values (million)
    size_t N = state.range(0);

    // Sum of the vector elements
    T sum = 0.;

    // Allocate a device vector.
    thrust::device_vector<T> X(N << 20, 1);

    for (auto _ : state) {
        sum = run_sum(X);
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(reduce_sum, float)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(32, 4000);

BENCHMARK_TEMPLATE(reduce_sum, double)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(8)
    ->Range(32, 2000);

