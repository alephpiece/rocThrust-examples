#include <algorithm>            /* fill_n */

#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/log_utils.h"    /* namespace logutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */
#include "sum.h"                /* launch_sum */


int main() {

    // Initialize MPI
    mpiutils::initialize();

    // Get the number of available GPUs
    auto n_gpus = gpuutils::getNumGPUs();

    // Split the communicator to determine node-local ranks
    auto gpu_id = gpuutils::getMyGPU();

    logutils::print("Assigned GPU {}, there are {} in total\n", gpu_id, n_gpus);

    // Number of items
    constexpr size_t N = 500 << 20;
    // Memory usage in MiB
    constexpr size_t M = sizeof(FLOAT) * N >> 20;

    // Allocate memory on host
    auto X = new FLOAT[N];
    std::fill_n(X, N, gpu_id / FLOAT(3));

    logutils::print("Memory usage = {} MiB, X[0] = {}\n", M, X[0]);

    // Compute the sum on multiple devices
    auto sum = launch_sum(X, N);

    logutils::print("Sum = {}, expected = {}\n", sum, X[0] * N);

    // Reduction across processes
    auto total_sum = FLOAT(0.);
    MPI_Reduce(&sum,                            // void         *sendbuf [IN]
               &total_sum,                      // void         *recvbuf [OUT]
               1,                               // int          count
               mpiutils::getDatatype<FLOAT>(),  // MPI_Datatype datatype
               MPI_SUM,                         // MPI_Op       op
               0,                               // int          root
               mpiutils::getComm());            // MPI_Comm     comm

    if (mpiutils::isRoot())
        logutils::print("Total sum = {}\n", total_sum);

    // Clean up
    delete[] X;

    // Finalize MPI
    mpiutils::finalize();

    return 0;
}
