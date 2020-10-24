#include <fmt/core.h>           /* print */

#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */


int main() {

    // Initialize MPI
    mpiutils::initialize();

    // Get the number of available GPUs
    int n_gpus = gpuutils::getNumGPUs();

    // Split the communicator to determine node-local ranks
    int gpu_id = mpiutils::getMyGPU(n_gpus);

    // Get my rank
    int rank = mpiutils::getCommRank();

    fmt::print("Rank {} is assigned GPU {}, total is {}.\n", rank, gpu_id, n_gpus);

    // Finalize MPI
    mpiutils::finalize();

    return 0;
}
