#include <thrust/device_vector.h>   /* device_vector */
#include <thrust/functional.h>      /* plus */
#include <thrust/transform.h>       /* transform */

#include "utils/gpu_utils.h"        /* getMyGPU */
#include "saxpy.h"


/// \class A function object for saxpy (can be replaced with a lambda)
template <typename T>
struct saxpy_functor {

    ///< Scalar A
    const T a;

    /// \brief Constructor
    /// \param _a Scalar A
    saxpy_functor(T _a) : a(_a) {}

    /// \brief Parenthesis operator indicating a binary function
    /// \param x Element in vector X
    /// \param y Element in vector Y
    /// \return  AXPY
    __host__ __device__
    T operator()(const T& x, const T& y) const {
        return a * x + y;
    }
};


/// \brief Allocate memory and do saxpy on device
template <typename T>
void launch_saxpy(T A, T *X, T *Y, size_t N) {

    // Switch to the working device
    hipSetDevice(gpuutils::getMyGPU());

    // Allocate memory on device
    T *raw_ptr_x, *raw_ptr_y;
    hipMalloc((void **) &raw_ptr_x, N * sizeof(T));
    hipMalloc((void **) &raw_ptr_y, N * sizeof(T));


    // Memory copy from host to device
    hipMemcpyHtoD(raw_ptr_x, X, N * sizeof(T));
    hipMemcpyHtoD(raw_ptr_y, Y, N * sizeof(T));

    // Wrap the raw pointers with a device_ptrs
    thrust::device_ptr<T> dev_ptr_x(raw_ptr_x);
    thrust::device_ptr<T> dev_ptr_y(raw_ptr_y);

    // Transformation, Y = A * X + Y
    thrust::transform(
        dev_ptr_x, dev_ptr_x + N,   // InputIterator1 begin, InputIterator1 end
        dev_ptr_y,                  // InputIterator2 begin
        dev_ptr_y,                  // OutputIterator result
        saxpy_functor<T>(A)         // BinaryFunction op
    );

    // Memory copy from device back to host
    hipMemcpyDtoH(Y, raw_ptr_y, N * sizeof(T));

    // Clean up
    hipFree(raw_ptr_x);
    hipFree(raw_ptr_y);
}


/// Explicit instantiation
template void launch_saxpy(float A, float *X, float *Y, size_t N);
template void launch_saxpy(double A, double *X, double *Y, size_t N);

