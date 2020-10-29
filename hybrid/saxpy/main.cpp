#include <unistd.h>             /* getopt */

#include <algorithm>            /* fill_n */
#include <string>               /* stoi, stod */
#include <thread>

#include "utils/gpu_utils.h"    /* namespace gpuutils */
#include "utils/log_utils.h"    /* namespace logutils */
#include "utils/mpi_utils.h"    /* namespace mpiutils */
#include "utils/tinytimer.h"    /* TinyTimer */
#include "saxpy.h"              /* launch_saxpy */


static size_t N_ITEMS     = 1000;   ///< Number of items (million)
static int    N_THREADS   = 0;      ///< Number of threads (def: auto)
static size_t N_STREAMS   = 2;      ///< Number of streams
static float  GPU_RATIO   = -1.;    ///< Workload ratio (def: auto)
static bool   DEPTH_FIRST = false;  ///< Depth-first or depth-first


/// \brief Parse command line options
void parseOptions(int argc, char *argv[]);


int main(int argc, char *argv[]) {

    // Initialize MPI
    mpiutils::initialize();

    // Read options
    parseOptions(argc, argv);

    // Print device info
    gpuutils::printDeviceProperties();

    // A tiny timer using milliseconds as period type
    TinyTimer timer;

    // Total time
    timer.start();

    //------------------------------------------------------------------
    // Setup
    //------------------------------------------------------------------
    // Check and set the number of host threads
    if (!N_THREADS) {
        N_THREADS = std::thread::hardware_concurrency();
    }

    // Distribute data over MPI ranks
    N_ITEMS = mpiutils::blockDecompositionCount(N_ITEMS);

    size_t N = N_ITEMS << 20;   // Number of items
    size_t N_gpu, N_cpu;        // Vector sizes
    FLOAT A = 1.;               // A, scalar
    FLOAT *X_gpu = nullptr;     // GPU workload
    FLOAT *Y_gpu = nullptr;     // GPU workload
    FLOAT *X_cpu = nullptr;     // CPU workload
    FLOAT *Y_cpu = nullptr;     // CPU workload

    // Determine workloads for host and device
    if (GPU_RATIO < 0.) {
        logutils::print("Set GPU workload to CUs/(CUs + host threads)\n");

        float units = N_THREADS + gpuutils::getNumCUs();
        GPU_RATIO   = gpuutils::getNumCUs() / units;
    }
    else if (GPU_RATIO > 1.) {
        GPU_RATIO = 1;
    }

    N_gpu = N * GPU_RATIO;
    N_cpu = N - N_gpu;

    // Print job info
    logutils::print(
        "Job info:\n"
        "\trank                = {} ({} in total)\n"
        "\tgpu                 = {} ({} in total)\n"
        "\t# host threads      = {}\n"
        "\t# streams           = {}\n"
        "\tdepth-first calls   = {}\n"
        "\tvector size         = {} M\n"
        "\tmemory usage        = 2 * {} MiB\n"
        "\tdevice memory usage = 2 * {} MiB ({:.2f}\%)\n"
         , mpiutils::getCommRank(), mpiutils::getCommSize()
         , gpuutils::getMyGPU(),    gpuutils::getNumGPUs()
         , N_THREADS
         , N_STREAMS
         , DEPTH_FIRST
         , N_ITEMS
         , sizeof(FLOAT) * N_ITEMS
         , sizeof(FLOAT) * N_gpu >> 20, GPU_RATIO * 100.);


    //------------------------------------------------------------------
    // Warmup
    //------------------------------------------------------------------
    logutils::print("GPU warming up...\n");
    timer.start();

    gpuutils::warmUp();

    timer.stop("Warm up");


    //------------------------------------------------------------------
    // Initialization
    //------------------------------------------------------------------
    timer.start();

    // Allocate and initialize host memory
    auto init = 1 / FLOAT(3);
    if (N_gpu) {
        gpuutils::hostMalloc((void **)&X_gpu, N_gpu * sizeof(FLOAT));
        gpuutils::hostMalloc((void **)&Y_gpu, N_gpu * sizeof(FLOAT));
        std::fill_n(X_gpu, N_gpu, init);
        std::fill_n(Y_gpu, N_gpu, init);
    }

    if (N_cpu) {
        X_cpu = new FLOAT[N_cpu];
        Y_cpu = new FLOAT[N_cpu];
        std::fill_n(X_cpu, N_cpu, init);
        std::fill_n(Y_cpu, N_cpu, init);
    }
    timer.stop("Allocate host vectors");

    if (N_gpu) {
        logutils::print("Before saxpy, Y_gpu[0] = {}\n", Y_gpu[0]);
    }

    if (N_cpu) {
        logutils::print("Before saxpy, Y_cpu[0] = {}\n", Y_cpu[0]);
    }


    //------------------------------------------------------------------
    // Computation
    //------------------------------------------------------------------
    // Create an SAXPY launcher
    SAXPYLauncher launcher(A, X_gpu, Y_gpu, N_gpu, N_STREAMS);

    if (N_gpu) {
        launcher.initialize();
    }

    // Total time for computation (no host memory management)
    timer.start();

    // Computation on device (potentially overlapped with host)
    if (N_gpu) {
        logutils::print("Starting memcpyHtoD, kernel, and memcpyDtoH...\n");
        timer.start();

        launcher.run(DEPTH_FIRST);

        timer.stop("(c)Async memcpy & kernel launch");
    }

    // Computation on host
    if (N_cpu) {
        logutils::print("Starting computation on host...\n");
        timer.start();

        std::vector<std::thread> threads;
        for (size_t tid = 0; tid < N_THREADS; tid++) {
            threads.push_back(
                std::thread(
                [&]() {
                    size_t i = tid;
                    while (i < N_cpu) {
                        for (size_t rounds = 0; rounds < 500; ++rounds)
                            Y_cpu[i] = A * X_cpu[i] + Y_cpu[i];
                        i += N_THREADS;
                    }
                })
            );
        }

        for (auto &thread : threads)
            thread.join();

        timer.stop("(c)Host multi-threading");
    }

    // Wait for devices
    if (N_gpu) {
        logutils::print("Synchronizing streams...\n");
        timer.start();

        launcher.synchronize();

        timer.stop("(c)Synchronize streams");
    }

    timer.stop("(c)Total time for computation");

    if (N_gpu) {
        logutils::print("After saxpy, Y_gpu[0] = {}\n", Y_gpu[0]);
    }

    if (N_cpu) {
        logutils::print("After saxpy, Y_cpu[0] = {}\n", Y_cpu[0]);
    }


    //------------------------------------------------------------------
    // Cleanup
    //------------------------------------------------------------------
    logutils::print("Clean up host memory...\n");
    timer.start();

    // Clean up host memory
    gpuutils::hostFree(X_gpu);
    gpuutils::hostFree(Y_gpu);
    delete [] X_cpu;
    delete [] Y_cpu;

    timer.stop("Deallocate host memory");

    // Clean up device memory
    if (N_gpu) {
        logutils::print("Clean up device memory...\n");
        timer.start();

        launcher.~SAXPYLauncher();

        timer.stop("Deallocate device memory");
    }


    timer.stop("Total time");

    // Print timing report
    timer.report();

    mpiutils::finalize();

    return 0;
}


void parseOptions(int argc, char *argv[]) {

    int opt;

    while ((opt = getopt(argc, argv, "hn:s:t:r:d")) != -1) {
        switch (opt) {
        case 'n':
            N_ITEMS = std::stoi(optarg);
            break;
        case 's':
            N_STREAMS = std::stoi(optarg);
            break;
        case 't':
            N_THREADS = std::stoi(optarg);
            break;
        case 'r':
            GPU_RATIO = std::stod(optarg);
            break;
        case 'd':
            DEPTH_FIRST = true;
            break;
        default:    // help
            if (mpiutils::isRoot())
                fmt::print("Usage: {} [-n N] [-s S] [-t T] [-r R] [-b]\n"
                           "\n"
                           "Options:\n"
                           "\t-n N, number of items (million)\n"
                           "\t-s S, number of HIP streams\n"
                           "\t-t T, number of host threads\n"
                           "\t-r R, ratio of GPU/overall workload\n"
                           "\t      '-r 0' means pure host threads\n"
                           "\t      '-r 1' means pure GPU\n"
                           "\n"
                            , argv[0]);
            mpiutils::finalize();
            exit(0);
        }
    }
}
