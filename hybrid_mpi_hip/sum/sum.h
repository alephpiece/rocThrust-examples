#ifndef HYBRID_MPI_HIP_SUM_H_
#define HYBRID_MPI_HIP_SUM_H_


/// \brief Sum up vector elements on device
/// \param data    C-style array
/// \param N       Array size
/// \return        Sum of array elements
template <typename T>
T launch_sum(T *data, size_t N);

/// \brief Extern template declaration
extern template float launch_sum(float *data, size_t N);
extern template double launch_sum(double *data, size_t N);


#endif  // HYBRID_MPI_HIP_SUM_H_
