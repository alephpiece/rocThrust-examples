#ifndef HYBRID_MPI_HIP_MPI_UTILS_H_
#define HYBRID_MPI_HIP_MPI_UTILS_H_

#include <mpi.h>
#include <fmt/core.h>   /* format */

#include <stdexcept>
#include <type_traits>
#include <vector>


/// \namespace  mpiutils
/// \brief      Helper functions for MPI routines.
namespace mpiutils {

/// \brief Initialize MPI
void initialize() {
    MPI_Init(nullptr, nullptr);
}


/// \brief Finalize MPI
void finalize() {
    MPI_Finalize();
}


/// \brief  Get MPI communicator
/// \return Communicator
MPI_Comm getComm() {
    return MPI_COMM_WORLD;
}


/// \brief  Get MPI communicator size
/// \return Communicator size
int getCommSize() {
    int size;
    MPI_Comm_size(getComm(), &size);
    return size;
}


/// \brief  Get MPI rank
/// \return Rank
int getCommRank() {
    int rank;
    MPI_Comm_rank(getComm(), &rank);
    return rank;
}


/// \brief Check whether I'm root
/// \return True if I'm root, false otherwise
bool isRoot() {
    return getCommRank() == 0;
}


/// \brief Barrier
void barrier() {
    MPI_Barrier(getComm());
}


/// \brief  Get the MPI datatype corresponding to a C++ type
/// \return MPI datatype
template <typename T> MPI_Datatype getDatatype() {

    if (std::is_same<T, int>::value) {
        return MPI_INT32_T;
    } else if (std::is_same<T, long>::value) {
        return MPI_INT64_T;
    } else if (std::is_same<T, float>::value) {
        return MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
        return MPI_DOUBLE;
    } else {
        throw std::invalid_argument("Unsupported type");
        return MPI_BYTE;
    }
}


///-----------------------------------------------------------------------------
/// Block distribution
///-----------------------------------------------------------------------------
///< Number of items assigned to each rank
static std::vector<size_t> _loads;


/// \brief Generate a block distribution
/// \param N  Total number of items
/// \return   Vector of loads
std::vector<size_t> blockDecomposition(const size_t N) {

    // Compute once
    if (_loads.empty()) {

        auto np = getCommSize();
        _loads.reserve(np);

        for (int r = 0; r < np; ++r)
            _loads.push_back(N / np);

        for (int r = 0; r < N % np; ++r)
            _loads[r]++;
    }
    return _loads;
}


/// \brief Get the number of items belonging to me
/// \param N  Total number of items
/// \return   Number of local items
size_t blockDecompositionCount(const size_t N) {
    auto loads = blockDecomposition(N);
    return loads[getCommRank()];
}


/// \brief Get the index of the starting item
/// \param N  Total number of items
/// \return   Global offset of local items
size_t blockDecompositionOffset(const size_t N) {
    auto offset = size_t(0);
    auto rank   = getCommRank();
    auto loads  = blockDecomposition(N);

    for (int r = 0; r < rank; ++r)
        offset += loads[r];
    return offset;
}


///-----------------------------------------------------------------------------
/// GPU identification
///-----------------------------------------------------------------------------
///< GPU ID assgined to me
static int _gpu_id = -1;
///< MPI communicator for node-local ranks
static MPI_Comm _comm_node;


/// \brief Get the GPU assigned to me
/// \param n_gpus Number of available GPUs per node
/// \return       GPU ID (also the node-local rank)
int getMyGPU(int n_gpus) {

    // Compute once
    if (_gpu_id < 0) {

        // Creates new communicators based on split types and keys
        MPI_Comm_split_type(
            MPI_COMM_WORLD,         // MPI_Comm communicator
            MPI_COMM_TYPE_SHARED,   // int      split_type
            0,                      // int      key
            MPI_INFO_NULL,          // MPI_Info info
            &_comm_node             // MPI_Comm *new_communicator
        );

        MPI_Comm_rank(_comm_node, &_gpu_id);

        if (_gpu_id < 0 || _gpu_id >= n_gpus) {
            throw std::out_of_range(
                fmt::format("MPI rank {} was assigned an invalid GPU: {}, total is {}\n",
                    getCommRank(), _gpu_id, n_gpus)
            );
            mpiutils::finalize();
        }
    }
    return _gpu_id;

}

}   // namespace

#endif  // HYBRID_MPI_HIP_MPI_UTILS_H_
