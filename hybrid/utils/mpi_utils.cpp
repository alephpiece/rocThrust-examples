#include "utils/mpi_utils.h"


namespace mpiutils {


void initialize() {
    MPI_Init(nullptr, nullptr);
}


void finalize() {
    MPI_Finalize();
}


MPI_Comm getComm() {
    return MPI_COMM_WORLD;
}


int getCommSize() {
    int size;
    MPI_Comm_size(getComm(), &size);
    return size;
}


int getCommRank() {
    int rank;
    MPI_Comm_rank(getComm(), &rank);
    return rank;
}


bool isRoot() {
    return getCommRank() == 0;
}


void barrier() {
    MPI_Barrier(getComm());
}


///---------------------------------------
/// Block distribution
///---------------------------------------
///< Number of items assigned to each rank
static std::vector<size_t> _loads;


std::vector<size_t> blockDecomposition(const size_t N) {

    auto np = getCommSize();

    std::vector<size_t> loads;
    loads.reserve(np);

    for (int r = 0; r < np; ++r)
        loads.push_back(N / np);

    for (int r = 0; r < N % np; ++r)
        loads[r]++;

    return loads;
}


size_t blockDecompositionCount(const size_t N) {
    auto loads = blockDecomposition(N);
    return loads[getCommRank()];
}


size_t blockDecompositionOffset(const size_t N) {
    auto offset = size_t(0);
    auto rank   = getCommRank();
    auto loads  = blockDecomposition(N);

    for (int r = 0; r < rank; ++r)
        offset += loads[r];
    return offset;
}


}   // namespace
