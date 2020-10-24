#ifndef BENCHMARK_SORT_H_
#define BENCHMARK_SORT_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>


/// \brief Sort vector elements on device
template <typename T>
void run_sort(thrust::device_vector<T> &X) {
    thrust::sort(X.begin(), X.end(), thrust::greater<T>());
}


#endif  // BENCHMARK_SORT_H_
