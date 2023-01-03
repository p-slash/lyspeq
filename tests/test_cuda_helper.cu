#include "mathtools/cuda_helper.cu"
#include "io/logger.hpp"

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

const int
NA = 500, NB = 5;

const double
IN_ARR[] = {1.1, 2.2, 3.3, 4.4, 5.5};


int main(int argc, char *argv[])
{
    int r=0;
    #if defined(ENABLE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process::this_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &process::total_pes);
    #else
    process::this_pe   = 0;
    process::total_pes = 1;
    #endif

    LOG::LOGGER.STD("Allocating a single MyCuDouble.\n");
    MyCuDouble covariance_matrix;
    assert(covariance_matrix.get() == nullptr);
    covariance_matrix.realloc(NA*NA);
    LOG::LOGGER.STD("Reset a single MyCuDouble.\n");
    covariance_matrix.reset();
    assert(covariance_matrix.get() == nullptr);

    LOG::LOGGER.STD("Allocating array of two MyCuDouble.\n");
    MyCuDouble temp_matrix[2];
    assert(temp_matrix[0].get() == nullptr);
    assert(temp_matrix[1].get() == nullptr);
    temp_matrix[0].realloc(NA*NA);
    temp_matrix[1].realloc(NA*NA);
    LOG::LOGGER.STD("Reset array of two MyCuDouble.\n");
    temp_matrix[0].reset();
    temp_matrix[1].reset();

    LOG::LOGGER.STD("Asnyc copy a single MyCuDouble.\n");
    MyCuDouble vec(NB, IN_ARR);

}
