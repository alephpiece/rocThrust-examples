#ifndef HYBRID_SAXPY_H_
#define HYBRID_SAXPY_H_

#ifndef FLOAT
#define FLOAT double
#endif


// Forward declaration
struct GPUPlans;

struct SAXPYLauncher {

    FLOAT A;
    FLOAT *X;
    FLOAT *Y;
    FLOAT *dev_X;
    FLOAT *dev_Y;
    size_t N;
    size_t nstreams;
    GPUPlans *plans;

    /// \brief Initialize an SAXPYLauncher
    /// \param A        Scalar
    /// \param X        C-style array on host
    /// \param Y        C-style array on host
    /// \param N        Array size
    /// \param nstreams Number of streams
    SAXPYLauncher(const FLOAT _A, FLOAT *_X, FLOAT *_Y, size_t _N, size_t _nstreams);

    /// \brief Deallocation
    ~SAXPYLauncher();

    /// \brief Initialize streams and device memory
    void initialize();

    /// \brief Create streams
    void createStreams();

    /// \brief Allocate device memory
    void mallocDevice();

    /// \brief Memory copy and kernel launch
    /// \param depth_first Depth-first function calls
    void run(bool depth_first);

    /// \brief Copy memory from host to device
    /// \param i Stream id
    void memcpyHtoD(size_t i);

    /// \brief Compute SAXPY using thrust::transform in multiple streams
    /// \param i Stream id
    void execute_thrust(size_t i);

    /// \brief Copy memory from device to host
    /// \param i Stream id
    void memcpyDtoH(size_t i);

    /// \brief Synchronization
    void synchronize();
};


#endif  // HYBRID_SAXPY_H_
