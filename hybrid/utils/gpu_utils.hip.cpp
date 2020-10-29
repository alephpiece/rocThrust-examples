#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <stdexcept>

#include "gpu_utils.h"
#include "log_utils.h"
#include "mpi_utils.h"


namespace gpuutils {

///< My GPU
static int _gpu_id = -1;


/// \brief Macro to check hip error
#define hipCheckErr(ret) { hipAssert((ret), __FILE__, __LINE__); }
void hipAssert(hipError_t err, const char *file, int line) {
   if (err != hipSuccess) {
      logutils::print("GPU ERROR: {} {} {}\n", hipGetErrorString(err), file, line);
      exit(err);
   }
}


void printDeviceProperties() {

    auto gpu_id = getMyGPU();

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, gpu_id);

    logutils::print(
        "Device properties:\n"
        "\tname               = {}\n"
        "\tGCN architecture   = {}\n"
        "\tCUs                = {}\n"
        "\tcompute capability = {}.{}\n"
        "\twarp size          = {}\n"
        "\tglobal mem         = {:.2f} MiB\n"
        "\tshared mem/block   = {:.2f} MiB\n"
        "\tregisters/block    = {}\n"
        "\tconcurrent kernels = {}\n"
        "\tis multi-gpu board = {}\n"
        , prop.name
        , prop.gcnArch
        , prop.multiProcessorCount
        , prop.major, prop.minor
        , prop.warpSize
        , prop.totalGlobalMem / double(1 << 20)
        , prop.sharedMemPerBlock / double(1 << 20)
        , prop.regsPerBlock
        , prop.concurrentKernels
        , prop.isMultiGpuBoard
        );
}


void warmUp() {
    thrust::device_vector<float> X(1);
    thrust::device_vector<float> Y(1);
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<float>());
}


int getMyGPU() {

    // Compute once
    if (_gpu_id < 0) {

        // Number of GPUs
        auto n_gpus = getNumGPUs();

        // MPI communicator for node-local ranks
        MPI_Comm comm_node;

        // Creates new communicators based on split types and keys
        MPI_Comm_split_type(
            MPI_COMM_WORLD,         // MPI_Comm communicator
            MPI_COMM_TYPE_SHARED,   // int      split_type
            0,                      // int      key
            MPI_INFO_NULL,          // MPI_Info info
            &comm_node              // MPI_Comm *new_communicator
        );

        MPI_Comm_rank(comm_node, &_gpu_id);

        if (_gpu_id < 0 || _gpu_id >= n_gpus) {
            throw std::out_of_range(
                fmt::format("MPI rank {} was assigned an invalid GPU: {}, total is {}\n",
                    mpiutils::getCommRank(), _gpu_id, n_gpus)
            );
            mpiutils::finalize();
            exit(EXIT_FAILURE);
        }
    }
    return _gpu_id;
}


void setMyGPU(int id) {
    _gpu_id = id;
}

int getNumGPUs() {
    int n_gpus;
    hipGetDeviceCount(&n_gpus);
    return n_gpus;
}

int getNumCUs() {
    hipDeviceProp_t dev_prop;
    hipGetDeviceProperties(&dev_prop, getMyGPU());
    return dev_prop.multiProcessorCount;
}

void hostMalloc(void **ptr, size_t size) {
    hipCheckErr( hipHostMalloc(ptr, size, hipHostMallocDefault) );
}

void hostFree(void *ptr) {
    hipCheckErr( hipHostFree(ptr) );
}

void deviceMalloc(void **ptr, size_t size) {
    hipCheckErr( hipMalloc(ptr, size) );
}

void deviceFree(void *ptr) {
    hipCheckErr( hipFree(ptr) );
}

}   // namespace
