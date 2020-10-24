#ifndef HYBRID_MPI_HIP_GPU_UTILS_H_
#define HYBRID_MPI_HIP_GPU_UTILS_H_


/// \namespace gpuutils
/// \brief     Helper functions for retrieving device information.
namespace gpuutils {

/// \brief Get the ID of the GPU assigned to current rank.
int getMyGPU();

/// \brief Set the ID of the GPU assigned to current rank.
/// \param id Valide GPU ID
void setMyGPU(int id);

/// \brief  Get the number of available GPUs on this node.
/// \return Number of GPUs
int getNumGPUs();

}   // namespace


#endif  // HYBRID_MPI_HIP_GPU_UTILS_H_
