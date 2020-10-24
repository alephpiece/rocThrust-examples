#ifndef BENCHMARK_SAXPY_H_
#define BENCHMARK_SAXPY_H_

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>


/// \brief SAXPY using kernel fusion
template <typename T>
void run_saxpy_fast(T A, thrust::device_vector<T>& X,
                         thrust::device_vector<T>& Y) {

    // Y = A * X + Y
    thrust::transform(
        X.begin(), X.end(),             // InputIterator1 begin, InputIterator1 end
        Y.begin(),                      // InputIterator2 begin
        Y.begin(),                      // OutputIterator result
        [=](const T &x, const T &y) {   // BinaryFunction op
            return A * x + y;
        }
    );
}


/// \brief SAXPY using multiple thrust::transform
template <typename T>
void run_saxpy_slow(T A, thrust::device_vector<T>& X,
                         thrust::device_vector<T>& Y) {

    thrust::device_vector<T> temp(X.size());

    // temp = A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp = A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(),
        thrust::multiplies<T>());

    // Y = A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(),
        thrust::plus<T>());
}

#endif  // BENCHMARK_SAXPY_H_
