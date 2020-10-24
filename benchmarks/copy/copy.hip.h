#ifndef BENCHMARK_COPY_H_
#define BENCHMARK_COPY_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


/// \brief Memory copy from Host to Device
template <typename T>
void run_copy_h2d(thrust::host_vector<T>& host_X,
                  thrust::device_vector<T>& dev_X) {

    thrust::copy(host_X.begin(), host_X.end(), dev_X.begin());
}


/// \brief Memory copy from Device to Host
template <typename T>
void run_copy_d2h(thrust::device_vector<T>& dev_X,
                  thrust::host_vector<T>& host_X) {

    thrust::copy(dev_X.begin(), dev_X.end(), host_X.begin());
}

#endif  // BENCHMARK_COPY_H_
