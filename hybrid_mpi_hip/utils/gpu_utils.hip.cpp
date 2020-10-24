#include <hip/hip_runtime.h>    /* hipGetDeviceCount */
#include <stdexcept>


namespace gpuutils {

static int _gpu_id = -1;

int getMyGPU() {
    if (_gpu_id < 0) {
        throw std::invalid_argument("GPU id not set");
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

}   // namespace
