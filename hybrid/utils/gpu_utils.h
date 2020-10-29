#ifndef HYBRID_GPU_UTILS_H_
#define HYBRID_GPU_UTILS_H_


/// \namespace gpuutils
/// \brief     Helper functions for retrieving device information.
namespace gpuutils {


/// \brief Print device properties
void printDeviceProperties();

/// \brief Warm up the GPU
void warmUp();

/// \brief  Get the ID of the GPU assigned to current rank.
/// \return GPU ID (also the node-local rank)
int getMyGPU();

/// \brief Set the ID of the GPU assigned to current rank.
/// \param id Valide GPU ID
void setMyGPU(int id);

/// \brief  Get the number of available GPUs on this node.
/// \return Number of GPUs
int getNumGPUs();

/// \brief  Get the number of compute nodes
/// \return Number of CUs
int getNumCUs();

/// \brief Allocate memory on host
/// \param ptr  Pointer to the buffer pointer
/// \param size Size of the buffer
void hostMalloc(void **ptr, size_t size);

/// \brief Free memory allocated on host
/// \param ptr  Pointer to the buffer
void hostFree(void *ptr);

/// \brief Allocate memory on device
/// \param ptr  Pointer to the buffer pointer
/// \param size Size of the buffer
void deviceMalloc(void **ptr, size_t size);

/// \brief Free memory allocated on device
/// \param ptr  Pointer to the buffer
void deviceFree(void *ptr);

}   // namespace


#endif  // HYBRID_GPU_UTILS_H_
