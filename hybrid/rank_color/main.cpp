#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/log_utils.h"    /* namespace logutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */


int main() {

    // Initialize MPI
    mpiutils::initialize();

    // Get the number of available GPUs
    int n_gpus = gpuutils::getNumGPUs();

    // Split the communicator to determine node-local ranks
    int gpu_id = gpuutils::getMyGPU();

    // Get my rank
    int rank = mpiutils::getCommRank();

    logutils::print("Assigned GPU {}, there are {} in total\n", gpu_id, n_gpus);

    // Finalize MPI
    mpiutils::finalize();

    return 0;
}
