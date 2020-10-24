#include <algorithm>            /* fill_n */
#include <fmt/core.h>           /* print */

#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */
#include "sum.h"                /* launch_sum */


// Floating-point precision
using FLOAT = double;

int main() {

    // Initialize MPI
    mpiutils::initialize();


    // Get the number of available GPUs
    auto n_gpus = gpuutils::getNumGPUs();

    // Split the communicator to determine node-local ranks
    auto gpu_id = mpiutils::getMyGPU(n_gpus);
    gpuutils::setMyGPU(gpu_id);

    // Get my rank
    auto rank = mpiutils::getCommRank();

    fmt::print("Rank {} is assigned GPU {}\n", rank, gpu_id);

    // Number of items
    constexpr size_t N = 500 << 20;
    // Memory usage in MiB
    constexpr size_t M = sizeof(FLOAT) * N >> 20;

    // Allocate memory on host
    auto X = new FLOAT[N];
    std::fill_n(X, N, gpu_id / FLOAT(3));

    fmt::print("Rank {}: memory = {}MiB, X[0] = {}\n", rank, M, X[0]);

    // Compute the sum on multiple devices
    auto sum = launch_sum(X, N);

    fmt::print("Rank {}: sum = {}, expected = {}\n", rank, sum, X[0] * N);

    // Reduction across processes
    auto total_sum = FLOAT(0.);
    MPI_Reduce(&sum,                            // void         *sendbuf [IN]
               &total_sum,                      // void         *recvbuf [OUT]
               1,                               // int          count
               mpiutils::getDatatype<FLOAT>(),  // MPI_Datatype datatype
               MPI_SUM,                         // MPI_Op       op
               0,                               // int          root
               mpiutils::getComm());            // MPI_Comm     comm

    if (mpiutils::isRoot()) fmt::print("Total sum = {}\n", total_sum);

    // Clean up
    delete[] X;

    // Finalize MPI
    mpiutils::finalize();

    return 0;
}
