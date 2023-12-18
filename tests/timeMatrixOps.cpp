#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#if defined(ENABLE_OMP)
#include "omp.h"
#endif

#include "mathtools/matrix_helper.hpp"



int TOTAL_KZ_BINS = 35 * 13, N_LOOPS = 1000;

std::mt19937_64 rng_engine;
std::normal_distribution<double> distribution;

namespace mytime {
    static double time_spent_on_func[] = {0, 0};

    void printfBootstrapTimeSpentDetails(const char funcname[]="addBoot")
    {
        printf(
            "Total time spent in %s (no omp) is %.2f mins.\n"
            "Total time spent in %s (w/ omp) is %.2f mins.\n",
            funcname, time_spent_on_func[0], funcname, time_spent_on_func[1]);
        fflush(stdout);
    }

    void resetTime() {
        time_spent_on_func[0] = 0;
        time_spent_on_func[1] = 0;
    }

    class Timer
    {
        using steady_c  = std::chrono::steady_clock;
        using seconds_t = std::chrono::duration<double>;

        std::chrono::time_point<steady_c> m0;
    public:
        Timer() : m0(steady_c::now()) {};
        ~Timer() {};

        double getTime() const
        {
            return std::chrono::duration_cast<seconds_t>(steady_c::now() - m0).count();
        } 
    };

    static Timer timer;
}


std::unique_ptr<double[]> getRandomSymmMatrix(int ndim) {
    auto A = std::make_unique<double[]>(ndim * ndim);
    for (int i = 0; i < ndim; ++i)
        for (int j = 0; j < ndim; ++j)
            A[j + ndim * i] = distribution(rng_engine);

    mxhelp::copyUpperToLower(A.get(), ndim);
    return A;
}


double timeDsymm(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = getRandomSymmMatrix(ndim);
    auto C = std::make_unique<double[]>(ndim * ndim);

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        cblas_dsymm(
            CblasRowMajor, CblasLeft, CblasUpper,
            ndim, ndim, 1., A.get(), ndim,
            B.get(), ndim, 0, C.get(), ndim);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 3);
}


double timeDgemm(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = getRandomSymmMatrix(ndim);
    auto C = std::make_unique<double[]>(ndim * ndim);

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            ndim, ndim, ndim, 1., A.get(), ndim,
            B.get(), ndim, 0, C.get(), ndim);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 3);
}


double timeDsymv(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = std::make_unique<double[]>(ndim);
    auto C = std::make_unique<double[]>(ndim);

    for (int i = 0; i < ndim; ++i)
        B[i] = distribution(rng_engine);

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        cblas_dsymv(
            CblasRowMajor, CblasUpper, ndim, 1.,
            A.get(), ndim, B.get(), 1, 0, C.get(), 1);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 2);
}


double timeDdot(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = getRandomSymmMatrix(ndim);
    double C = 0;
    int ndim2 = ndim * ndim;

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        C = cblas_ddot(ndim2, A.get(), 1, B.get(), 1);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 2);
}


int main(int argc, char *argv[]) {
    int number_lapse = 400;
    if (argc == 2)
        number_lapse = atoi(argv[1]);
    else if (argc == 3) {
        number_lapse = atoi(argv[1]);
        N_LOOPS = atoi(argv[2]);
    }
    else if (argc > 3)
    {
        printf("Extra arguments! Usage lapse (default: %d) loops (%d)\n",
               number_lapse, N_LOOPS);
        return 1;
    }

    rng_engine.seed(0);
    std::vector<int> ndims = {
        100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
        750, 800, 850
    };

    std::vector<double> times_dsymm, times_dgemm, times_dsmyv, times_ddot;

    for (const int ndim : ndims) {
        times_dsymm.push_back(timeDsymm(ndim));
        times_dgemm.push_back(timeDgemm(ndim));
        times_dsmyv.push_back(timeDsymv(ndim));
        times_ddot.push_back(timeDdot(ndim));
    }

    double mean_dsymm = 0., mean_dgemm = 0., mean_dsymv = 0., mean_ddot = 0.;

    // for (int i = 0; i < ndims.size(); ++i)
    int i1 = 3, i2 = 9, nsample = i2 - i1;
    for (int i = i1; i < i2; ++i)
    {
        mean_dsymm += times_dsymm[i];
        mean_dgemm += times_dgemm[i];
        mean_dsymv += times_dsmyv[i];
        mean_ddot += times_ddot[i];
    }

    mean_dsymm /= nsample;
    mean_dgemm /= nsample;
    mean_dsymv /= nsample;
    mean_ddot /= nsample;

    printf("Ndim,dsymm,dgemm,dsymv,ddot\n");
    for (int i = 0; i < ndims.size(); ++i)
        printf("%d,%.3e,%.3e,%.3e,%.3e\n",
               ndims[i], times_dsymm[i], times_dgemm[i],
               times_dsmyv[i], times_ddot[i]);
    printf("--------------\n");
    printf("Ave,%.3e,%.3e,%.3e,%.3e\n",
           mean_dsymm, mean_dgemm, mean_dsymv, mean_ddot);
    printf("Rat,%.3f,%.3f,%.3f,%.3f\n",
           mean_dsymm / mean_ddot, mean_dgemm / mean_ddot,
           mean_dsymv / mean_ddot, mean_ddot / mean_ddot);
}
