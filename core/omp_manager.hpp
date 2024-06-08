#ifndef OMP_MANAGER_H
#define OMP_MANAGER_H

#if defined(ENABLE_OMP)
#include "omp.h"

namespace myomp {
    inline const extern bool OMP_ENABLED = true;
    inline int getThreadNum() { return omp_get_thread_num(); }
    inline int getNumThreads() { return omp_get_num_threads(); }
    inline int getMaxNumThreads() { return omp_get_max_threads(); }
    inline void setNumThreads(int n) { omp_set_num_threads(n); }
}

#else
namespace myomp {
    inline const extern bool OMP_ENABLED = false;
    inline int getThreadNum() { return 0; }
    inline int getNumThreads() { return 1; }
    inline int getMaxNumThreads() { return 1; }
    inline void setNumThreads(int n) {};
}
#endif

#endif
