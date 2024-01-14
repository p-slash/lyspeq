#ifndef OMP_MANAGER_H
#define OMP_MANAGER_H

#if defined(ENABLE_OMP)
#include "omp.h"

namespace myomp {
    inline const extern bool OMP_ENABLED = true;
    inline int getThreadNum() {
        return omp_get_thread_num();
    }

    inline int getMaxNumThreads() {
        return omp_get_max_threads();
    }
}

#else
namespace myomp {
    inline const extern bool OMP_ENABLED = false;
    inline int getThreadNum() const {
        return 0;
    }

    inline int getMaxNumThreads() const {
        return 1;
    }
}
#endif

#endif
