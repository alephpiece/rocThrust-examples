#include <thrust/device_vector.h>   /* device_vector */
#include <thrust/functional.h>      /* plus */
#include <thrust/reduce.h>          /* reduce */

#include "utils/gpu_utils.h"        /* getMyGPU */
#include "sum.h"


template <typename T>
T launch_sum(T *data, size_t N) {

    // Switch to the working device
    hipSetDevice(gpuutils::getMyGPU());

    // Allocate memory on device
    T *raw_ptr;
    hipMalloc((void **) &raw_ptr, N * sizeof(T));

    // Memory copy from host to device
    hipMemcpyHtoD(raw_ptr, data, N * sizeof(T));

    // Wrap the raw pointer with a device_ptr
    thrust::device_ptr<T> dev_ptr(raw_ptr);

    // Reduction
    T result =
    thrust::reduce(
        dev_ptr,            // InputIterator  begin
        dev_ptr + N,        // InputIterator  end
        T(0),               // T              init
        thrust::plus<T>()   // BinaryFunction op
    );

    // Clean up
    hipFree(raw_ptr);

    return result;
}


/// Explicit instantiation
template float launch_sum(float *data, size_t N);
template double launch_sum(double *data, size_t N);
