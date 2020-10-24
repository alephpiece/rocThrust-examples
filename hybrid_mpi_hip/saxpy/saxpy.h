#ifndef HYBRID_MPI_HIP_SAXPY_H_
#define HYBRID_MPI_HIP_SAXPY_H_


/// \brief SAXPY using kernel fusion
/// \param A    Scalar
/// \param X    C-style array
/// \param Y    C-style array
/// \param N    Array size
template <typename T>
void launch_saxpy(T A, T *X, T *Y, size_t N);


/// \brief Extern template declaration
extern template void launch_saxpy(float A, float *X, float *Y, size_t N);
extern template void launch_saxpy(double A, double *X, double *Y, size_t N);


#endif  // HYBRID_MPI_HIP_SAXPY_H_
