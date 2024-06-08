#ifndef MPI_MANAGER_H
#define MPI_MANAGER_H

#if defined(ENABLE_MPI)
#include "mpi.h"

namespace mympi {
    inline int this_pe = 0, total_pes = 1;
    inline const extern bool MPI_ENABLED = true;
    inline void init(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &this_pe);
        MPI_Comm_size(MPI_COMM_WORLD, &total_pes);
    }
    inline void finalize() { MPI_Finalize(); }
    inline void abort() { MPI_Abort(MPI_COMM_WORLD, 1); }
    inline void barrier() { MPI_Barrier(MPI_COMM_WORLD); }
}

#else
namespace mympi {
    inline int this_pe = 0, total_pes = 1;
    inline const extern bool MPI_ENABLED = false;
    inline void init(int argc, char *argv[]) {};
    inline void finalize() {};
    inline void abort() {};
    inline void barrier() {};
}
#endif

#endif
