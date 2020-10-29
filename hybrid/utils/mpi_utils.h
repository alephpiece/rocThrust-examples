#ifndef HYBRID_MPI_UTILS_H_
#define HYBRID_MPI_UTILS_H_

#include <mpi.h>

#include <stdexcept>
#include <type_traits>
#include <vector>


/// \namespace  mpiutils
/// \brief      Helper functions for MPI routines.
namespace mpiutils {


/// \brief Initialize MPI
void initialize();


/// \brief Finalize MPI
void finalize();


/// \brief  Get MPI communicator
/// \return Communicator
MPI_Comm getComm();


/// \brief  Get MPI communicator size
/// \return Communicator size
int getCommSize();


/// \brief  Get MPI rank
/// \return Rank
int getCommRank();


/// \brief Check whether I'm root
/// \return True if I'm root, false otherwise
bool isRoot();


/// \brief Barrier
void barrier();


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


/// \brief Generate a block distribution
/// \param N  Total number of items
/// \return   Vector of loads
std::vector<size_t> blockDecomposition(const size_t N);


/// \brief Get the number of items belonging to me
/// \param N  Total number of items
/// \return   Number of local items
size_t blockDecompositionCount(const size_t N);


/// \brief Get the index of the starting item
/// \param N  Total number of items
/// \return   Global offset of local items
size_t blockDecompositionOffset(const size_t N);


}   // namespace

#endif  // HYBRID_MPI_UTILS_H_
