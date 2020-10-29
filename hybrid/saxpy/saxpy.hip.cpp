#include <thrust/device_vector.h>       /* device_vector */
#include <thrust/execution_policy.h>    /* par */
#include <thrust/transform.h>           /* transform */
#include <string>
#include <vector>

#include "utils/gpu_utils.h"            /* namespace gpuutils */
#include "utils/log_utils.h"            /* namespace logutils */
#include "saxpy.h"


/// \class Array of launching plans
struct GPUPlans {

    std::vector<hipStream_t> streams;   ///< Streams
    std::vector<size_t>      sizes;     ///< Task counts
    std::vector<size_t>      offsets;   ///< Task offsets

    GPUPlans() = default;

    /// \brief Initialize arrays of streams, counts, and offsets
    /// \param nstreams Number of streams
    /// \param N        Number of tasks
    GPUPlans(const size_t nstreams, const size_t N) {

        streams.resize(nstreams);
        sizes.resize(nstreams);
        offsets.resize(nstreams);

        // Create streams
        for (auto &stream : streams)
            hipStreamCreate(&stream);

        // Count tasks
        for (size_t i = 0; i < nstreams; ++i)
            sizes[i] = N / nstreams;

        for (size_t i = 0; i < N % nstreams; ++i)
            sizes[i]++;

        // Compute offsets
        size_t offset = 0;
        for (size_t i = 0; i < nstreams; ++i) {
            offsets[i] = offset;
            offset += sizes[i];
        }

    }

    /// \brief Destroy all streams
    ~GPUPlans() {
        for (auto &stream : streams)
            hipStreamDestroy(stream);
    }

    std::string toString() const {
        return logutils::format(
                    "Stream plans:\n\tsizes = {}\n\toffsets = {}\n",
                     logutils::join(sizes, ", "),
                     logutils::join(offsets, ", "));
    }

    void printString() const {
        fmt::print("{}", toString());
    }
};


/// \class Functor for thrust::transform
template <typename T>
struct saxpy_functor {

    ///< Scalar A
    const T a;

    /// \brief Constructor
    /// \param _a Scalar A
    saxpy_functor(T _a) : a(_a) {}

    /// \brief Parenthesis operator indicating a binary function
    /// \param x Element in vector X
    /// \param y Element in vector Y
    __host__ __device__
    T operator()(const T& x, const T& y) const {
        T temp = 0.;
        for (int rounds = 0; rounds < 500; ++rounds)
            temp += a * x + y;
        return temp;
    }
};


SAXPYLauncher::SAXPYLauncher(const FLOAT _A, FLOAT *_X, FLOAT *_Y, size_t _N, size_t _nstreams)
    : A(_A),
      X(_X), Y(_Y),
      dev_X(nullptr), dev_Y(nullptr),
      N(_N), nstreams(_nstreams),
      plans(nullptr) {
}


SAXPYLauncher::~SAXPYLauncher() {

    hipFree(dev_X);
    dev_X = nullptr;

    hipFree(dev_Y);
    dev_X = nullptr;

    delete plans;
    plans = nullptr;
}


void SAXPYLauncher::initialize() {

    // Switch to the working device
    hipSetDevice(gpuutils::getMyGPU());

    createStreams();
    mallocDevice();
}


void SAXPYLauncher::createStreams() {

    logutils::print("Initializing streams, sizes, and offsets...\n");

    plans = new GPUPlans(nstreams, N);
    plans->printString();
}


void SAXPYLauncher::mallocDevice() {

    logutils::print("Allocating device memory...\n");

    hipMalloc((void **)&dev_X, N * sizeof(FLOAT));
    hipMalloc((void **)&dev_Y, N * sizeof(FLOAT));
}


void SAXPYLauncher::run(bool depth_first) {

    if (depth_first) {

        for (size_t i = 0; i < nstreams; ++i) {
            logutils::print("Starting memcpy & thrust for stream {}...\n", i);
            memcpyHtoD(i);
            execute_thrust(i);
            memcpyDtoH(i);
        }
    }
    else {

        logutils::print("Starting memcpy from host to device...\n");
        for (size_t i = 0; i < nstreams; ++i)
            memcpyHtoD(i);

        logutils::print("Starting SAXPY using thrust::transform...\n");
        for (size_t i = 0; i < nstreams; ++i)
            execute_thrust(i);

        logutils::print("Starting memcpy from device to host...\n");
        for (size_t i = 0; i < nstreams; ++i)
            memcpyDtoH(i);
    }
}


void SAXPYLauncher::memcpyHtoD(size_t i) {

    logutils::print("\tStream {}, addr(X) % 4K = {}, addr(Y) % 4K = {}\n", i,
                     size_t(X + plans->offsets[i]) % 4096*8,
                     size_t(Y + plans->offsets[i]) % 4096*8);

    hipMemcpyHtoDAsync(dev_X + plans->offsets[i], X + plans->offsets[i],
                       plans->sizes[i] * sizeof(FLOAT), plans->streams[i]);
    hipMemcpyHtoDAsync(dev_Y + plans->offsets[i], Y + plans->offsets[i],
                       plans->sizes[i] * sizeof(FLOAT), plans->streams[i]);
}


void SAXPYLauncher::execute_thrust(size_t i) {

    // Wrap the raw pointers with a device_ptrs
    thrust::device_ptr<FLOAT> dev_ptr_x(dev_X + plans->offsets[i]);
    thrust::device_ptr<FLOAT> dev_ptr_y(dev_Y + plans->offsets[i]);

    // Transformation, Y = A * X + Y
    thrust::transform(
        thrust::hip::par.on(plans->streams[i]), // ExecutionPolicy policy
        dev_ptr_x,                              // InputIterator1 begin
        dev_ptr_x + plans->sizes[i],            // InputIterator1 end
        dev_ptr_y,                              // InputIterator2 begin
        dev_ptr_y,                              // OutputIterator result
        saxpy_functor<FLOAT>(A)                 // BinaryFunction op
    );
}


void SAXPYLauncher::memcpyDtoH(size_t i) {

    hipMemcpyDtoHAsync(Y + plans->offsets[i], dev_Y + plans->offsets[i],
                       plans->sizes[i] * sizeof(FLOAT), plans->streams[i]);
}


void SAXPYLauncher::synchronize() {

    for (auto &stream : plans->streams)
        hipStreamSynchronize(stream);
}

