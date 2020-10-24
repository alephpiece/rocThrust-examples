#ifndef BENCHMARK_SCAN_H_
#define BENCHMARK_SCAN_H_

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>


/// \brief Inclusively scan a vector on device
template <typename T>
void run_inclusive_scan(thrust::device_vector<T> &X) {
    thrust::inclusive_scan(X.begin(), X.end(), X.begin());
}


/// \brief Exclusively scan a vector on device
template <typename T>
void run_exclusive_scan(thrust::device_vector<T> &X) {
    thrust::exclusive_scan(X.begin(), X.end(), X.begin());
}

#endif  // BENCHMARK_SCAN_H_
