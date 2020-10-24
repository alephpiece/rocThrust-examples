#ifndef BENCHMARK_SUM_H_
#define BENCHMARK_SUM_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>


/// \brief Sum up vector elements on device
template <typename T>
T run_sum(thrust::device_vector<T> &X) {
    return thrust::reduce(X.begin(), X.end(), (T)0, thrust::plus<T>());
}


#endif  // BENCHMARK_SUM_H_
