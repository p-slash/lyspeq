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

    // These function can only be used under #if defined(ENABLE_MPI) block
    template<class T>
    inline MPI_Datatype getMpiDataType();
    template<> inline MPI_Datatype getMpiDataType<double>() { return MPI_DOUBLE; }
    template<> inline MPI_Datatype getMpiDataType<int>() { return MPI_INT; }
    template<> inline MPI_Datatype getMpiDataType<bool>() { return MPI_CXX_BOOL; }
    // template<> inline MPI_Datatype getMpiDataType<unsigned long>() { return MPI_UNSIGNED_LONG; }

    template<class T>
    inline void allreduceInplace(T *x, int N, MPI_Op op=MPI_SUM) {
        MPI_Allreduce(MPI_IN_PLACE, x, N, getMpiDataType<T>(), op, MPI_COMM_WORLD);
    }

    template<class T>
    inline void reduceInplace(T *x, int N, MPI_Op op=MPI_SUM) {
        MPI_Datatype mpi_dtype = getMpiDataType<T>();
        if (this_pe != 0)
            MPI_Reduce(x, nullptr, N, mpi_dtype, op, 0, MPI_COMM_WORLD);
        else
            MPI_Reduce(MPI_IN_PLACE, x, N, mpi_dtype, op, 0, MPI_COMM_WORLD);
    }

    template<class T>
    inline void reduceToOther(T *source, T *target, int N, MPI_Op op=MPI_SUM) {
        MPI_Reduce(source, target, N, getMpiDataType<T>(), op, 0, MPI_COMM_WORLD);
    }

    template<class T>
    inline void gather(T x, std::vector<T> &result) {
        result.resize(total_pes);
        MPI_Gather(
            &x, 1, getMpiDataType<T>(), result.data(), 1, getMpiDataType<T>(),
            0, MPI_COMM_WORLD);
    }

    template<class T>
    inline void bcast(T *x, int N=1) {
        MPI_Bcast(x, N, getMpiDataType<T>(), 0, MPI_COMM_WORLD);
    }
}

#else
namespace mympi {
    inline int this_pe = 0, total_pes = 1;
    inline const extern bool MPI_ENABLED = false;
    inline void init(int argc, char *argv[]) {};
    inline void finalize() {};
    inline void abort() {};
    inline void barrier() {};
    inline void reduceInplace(...) {};
    inline void bcast(...) {};

    template<class T>
    inline void gather(T x, std::vector<T> &result) {
        result.resize(total_pes);
        result[0] = x;
    }

    template<class T>
    inline void reduceToOther(T *source, T *target, int N) {
        std::copy_n(source, N, target);
    }
}
#endif

#endif
