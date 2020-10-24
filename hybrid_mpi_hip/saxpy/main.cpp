#include <algorithm>            /* fill_n */
#include <fmt/core.h>           /* print */

#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */
#include "saxpy.h"              /* launch_saxpy */


// Floating-point precision
using FLOAT = double;

int main(void) {

    // Initialize MPI
    mpiutils::initialize();

    // Get the number of available GPUs
    int n_gpus = gpuutils::getNumGPUs();

    // Split the communicator to determine node-local ranks
    auto gpu_id = mpiutils::getMyGPU(n_gpus);
    gpuutils::setMyGPU(gpu_id);

    // Get my rank
    auto rank = mpiutils::getCommRank();

    fmt::print("Rank {} is assigned GPU {}\n", rank, gpu_id);

    // Number of items
    constexpr size_t N = 1000 << 20;

    // Memory usage in MiB
    constexpr size_t M = sizeof(FLOAT) * N >> 20;

    // Allocate memory on host
    auto A = FLOAT(gpu_id);             // A, scalar
    auto X = new FLOAT[N];              // X, vector
    auto Y = new FLOAT[N];              // Y, vector
    std::fill_n(X, N, 1 / FLOAT(3));
    std::fill_n(Y, N, 1 / FLOAT(3));

    fmt::print("Rank {}: memory = {}MiB, X[0] = {}\n", rank, M, X[0]);

    // Compute saxpy on multiple devices
    launch_saxpy(A, X, Y, N);

    fmt::print("Rank {}: after saxpy, X[0] = {}, Y[0] = {}\n", rank, X[0], Y[0]);

    // Clean up
    delete[] X;
    delete[] Y;

    mpiutils::finalize();

    return 0;
}
