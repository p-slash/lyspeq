#include <chrono>
#include <map>
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


int N_Q_MATRICES = 250, N_LOOPS = 1000;

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


std::unique_ptr<double[]> getRandomVector(int size) {
    auto A = std::make_unique<double[]>(size);
    for (int i = 0; i < size; ++i)
        A[i] = distribution(rng_engine);
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


double timeDgemv(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = std::make_unique<double[]>(ndim);
    auto C = std::make_unique<double[]>(ndim);

    for (int i = 0; i < ndim; ++i)
        B[i] = distribution(rng_engine);

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, ndim, ndim, 1.,
            A.get(), ndim, B.get(), 1, 0, C.get(), 1);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 2);
}


// double timeDmydsymv(int ndim) {
//     auto A = getRandomSymmMatrix(ndim);
//     auto B = std::make_unique<double[]>(ndim);

//     for (int i = 0; i < ndim; ++i)
//         B[i] = distribution(rng_engine);

//     double t1 = mytime::timer.getTime(), t2 = 0;

//     for (int i = 0; i < N_LOOPS; ++i)
//         double sum = mxhelp::my_cblas_dsymvdot(B.get(), A.get(), ndim);

//     t2 = mytime::timer.getTime();
//     return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 2);
// }


double timeDmydgemv(int ndim) {
    auto A = getRandomSymmMatrix(ndim);
    auto B = std::make_unique<double[]>(ndim);
    auto C = std::make_unique<double[]>(ndim);

    for (int i = 0; i < ndim; ++i)
        B[i] = distribution(rng_engine);

    double t1 = mytime::timer.getTime(), t2 = 0, sum = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        sum = mxhelp::my_cblas_dgemvdot(B.get(), A.get(), C.get(), ndim);

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


double timeDgemvdot(int ndim) {
    int ndim2 = ndim * ndim;
    auto A = getRandomVector(ndim2 * N_Q_MATRICES);
    auto B = getRandomVector(ndim2);
    auto C = std::make_unique<double[]>(N_Q_MATRICES);

    double t1 = mytime::timer.getTime(), t2 = 0;

    for (int i = 0; i < N_LOOPS; ++i)
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, N_Q_MATRICES, ndim2, 1.0,
            A.get(), ndim2, B.get(), 1, 0, C.get(), 1);

    t2 = mytime::timer.getTime();
    return (t2 - t1) / N_LOOPS / std::pow(ndim / 100., 2) / N_Q_MATRICES;
}


void timeOsampLeft(int ndim, int oversampling=3) {
    const int ndiags = 11;
    int noff = ndiags / 2, nelem_per_row = 2 * noff * oversampling + 1;
    int ncols = ndim * oversampling + nelem_per_row - 1;
    auto A = getRandomSymmMatrix(ncols);
    auto R = std::make_unique<double[]>(ndim * nelem_per_row);
    auto B = std::make_unique<double[]>(ndim * ncols);
    for (int i = 0; i < ndim * nelem_per_row; ++i)
        R[i] = i * 1. / ndim;
 
    double t1 = mytime::timer.getTime(), t2 = 0;
    for (int loop = 0; loop < N_LOOPS; ++loop) {

        for (int i = 0; i < ndim; ++i) {
            double *bsub = B.get() + i * ncols;
            const double *Asub = A.get() + i * ncols * oversampling;

            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, ncols, nelem_per_row, 1., R.get() + i * nelem_per_row, nelem_per_row,
                Asub, ncols, 0, bsub, ncols);
        }

    }
    t2 = mytime::timer.getTime();
    double tgemm = (t2 - t1) / N_LOOPS;
    printf("Osamp gemm: %.3e", tgemm);


    t1 = mytime::timer.getTime();
    for (int loop = 0; loop < N_LOOPS; ++loop) {

        for (int i = 0; i < ndim; ++i) {
            double *bsub = B.get() + i * ncols;
            const double *Asub = A.get() + i * ncols * oversampling;

            cblas_dgemv(
                CblasRowMajor, CblasTrans,
                nelem_per_row, ncols, 1., Asub, ncols, 
                R.get() + i * nelem_per_row, 1, 0, bsub, 1);
        }

    }
    t2 = mytime::timer.getTime();
    double tgemv = (t2 - t1) / N_LOOPS;
    printf("\tOsamp gemv: %.3e\tRatio %.3f\n", tgemv, tgemm / tgemv);
}


typedef double (*timeFunc)(int ndim);
typedef std::pair<timeFunc, std::vector<double>> func_vector_pair;


int main(int argc, char *argv[]) {
    if (argc == 2)
        N_LOOPS = atoi(argv[1]);
    else if (argc > 2) {
        printf("Extra arguments! Usage loops (default: %d)\n",
               N_LOOPS);
        return 1;
    }

    rng_engine.seed(0);
    std::vector<int> ndims = {
        100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
        // 750, 800, 850
    };

    std::map<std::string, func_vector_pair> times{
        {"dsymm", std::make_pair(&timeDsymm, std::vector<double>())},
        {"dgemm", std::make_pair(&timeDgemm, std::vector<double>())},
        {"dsymv", std::make_pair(&timeDsymv, std::vector<double>())},
        {"dgemv", std::make_pair(&timeDgemv, std::vector<double>())},
        // {"mygot", std::make_pair(&timeDmydgemv, std::vector<double>())},
        // {"mysyt", std::make_pair(&timeDmydsymv, std::vector<double>())},
        {"ddot", std::make_pair(&timeDdot, std::vector<double>())},
        {"dgemvdot", std::make_pair(&timeDgemvdot, std::vector<double>())}
    };

    std::map<std::string, double> mean_times;

    for (const int ndim : ndims) {
        for (auto &[key, pair] : times)
            pair.second.push_back(pair.first(ndim));
    }

    
    for (const auto &[key, pair] : times)
        mean_times[key] = 0;

    int i1 = 3, i2 = 9, nsample = i2 - i1;
    for (int i = i1; i < i2; ++i)
        for (auto &[key, mean] : mean_times)
            mean += times[key].second[i] / nsample;

    printf("Ndim");
    for (const auto &[key, pair] : times)
        printf(",%s", key.c_str());
    printf("\n");


    for (int i = 0; i < ndims.size(); ++i) {
        printf("%d", ndims[i]);
        for (const auto &[key, pair] : times)
            printf(",%.3e", pair.second[i]);
        printf("\n");
    }
    printf("--------------\nAve");
    for (const auto &[key, mean] : mean_times)
        printf(",%.3e", mean);
    printf("\nRat");
    for (const auto &[key, mean] : mean_times)
        printf(",%.3f", mean / mean_times["ddot"]);
    printf("\n");

    for (const int ndim : ndims)
        timeOsampLeft(ndim);
}
