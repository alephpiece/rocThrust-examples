#ifndef BENCHMARK_NORM_H_
#define BENCHMARK_NORM_H_

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <cmath>


/// \brief A functor for computing the square of a number f(x) -> x*x
template <typename T>
struct square {

    __host__ __device__
    T operator()(const T& x) const {
        return x * x;
    }
};


/// \brief Compute sqrt(x*x)
template <typename T>
T run_norm(thrust::device_vector<T> &X) {

    return std::sqrt(
            thrust::transform_reduce(
                X.begin(), X.end(), // InputIterator  begin, InputIterator end
                square<T>(),        // UnaryFunction  unary_op
                T(0),               // OutputType     init
                thrust::plus<T>()   // BinaryFunction binary_op
            )
           );
}


#endif  // BENCHMARK_NORM_H_
