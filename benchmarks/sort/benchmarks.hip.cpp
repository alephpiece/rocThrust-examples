#include <benchmark/benchmark.h>

#include "sort.hip.h"


///----------------------------------------------------------------------------
/// thrust::sort for random numbers
///----------------------------------------------------------------------------
template <typename T>
void bm_sort(benchmark::State &state) {

    // Number of items (million)
    size_t N = state.range(0);

    // Allocate a device vector
    thrust::device_vector<T> X(N << 20);

    // Generate a sequence
    thrust::sequence(X.begin(), X.end());

    for (auto _ : state) {
        run_sort(X);
        hipDeviceSynchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations())
                            * int64_t(sizeof(T) * N << 20));
}


/// Benchmark registration
BENCHMARK_TEMPLATE(bm_sort, int)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(32, 1024);


