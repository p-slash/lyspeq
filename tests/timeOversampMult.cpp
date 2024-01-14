#include <chrono>
#include <memory>
#include <random>
#include <string>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include "core/omp_manager.hpp"

int TOTAL_KZ_BINS = 35 * 13, N_LOOPS = 450000;


#define NrowsOversamp 7
#define NcolsOversamp 18
#define Nelemprow 5
#define Oversampling 2
const double
oversamp_values[] = {
    0.25, 0.29, 0.31, 0.29, 0.25, 0.25, 0.29,
    0.31, 0.29, 0.25, 0.25, 0.29, 0.31, 0.29,
    0.25, 0.25, 0.29, 0.31, 0.29, 0.25, 0.25,
    0.29, 0.31, 0.29, 0.25, 0.25, 0.29, 0.31,
    0.29, 0.25, 0.25, 0.29, 0.31, 0.29, 0.25},
oversample_multiplier_A[] = {
    0.42,0.08,0.15,0.73,0.72,0.25,0.5,0.05,0.87,0.11,0.33,0.74,0.1,0.63,
    0.67,0.67,0.0,0.84,0.38,0.76,0.74,0.21,0.4,0.59,0.72,0.11,0.48,0.33,0.91,0.3,0.26,
    0.62,0.1,0.57,0.81,0.51,0.05,0.67,0.24,0.89,0.32,0.87,0.31,0.27,0.32,0.19,0.72,0.2,
    0.33,0.34,0.05,0.91,0.8,0.68,0.15,0.1,0.15,0.14,0.36,0.56,0.21,0.72,0.17,0.76,0.06,
    0.94,0.03,0.09,0.05,0.98,0.93,0.5,0.26,0.62,0.24,0.83,0.75,0.02,0.49,0.76,0.45,0.67,
    0.96,0.72,0.23,0.65,0.88,0.09,0.03,0.38,0.88,0.17,0.13,0.21,0.5,0.6,0.09,0.31,0.13,
    0.75,0.08,0.21,0.11,0.01,0.33,0.48,0.21,0.22,0.85,0.59,0.05,0.18,0.27,0.62,0.58,0.5,
    0.18,0.25,0.07,0.91,0.54,0.49,0.53,0.93,0.98,0.76,0.33,0.4,0.44,0.4,0.44,0.92,0.8,0.26,
    0.49,0.44,0.7,0.18,0.42,0.5,0.12,0.61,0.71,0.95,0.48,0.75,0.24,0.76,0.18,0.07,0.5,0.37,
    0.96,0.3,0.43,0.63,0.95,0.04,0.87,0.32,0.03,0.03,0.92,0.46,0.54,0.33,0.6,0.28,0.63,0.18,
    0.28,0.88,0.19,0.69,0.87,0.74,0.92,0.16,0.23,0.26,0.72,0.56,0.56,0.57,0.52,0.94,0.83,0.51,
    0.16,0.96,0.43,0.57,0.97,0.7,0.01,0.04,0.0,0.37,0.09,0.89,0.7,0.4,0.04,0.13,0.67,0.82,0.42,
    0.31,0.02,0.47,0.68,0.59,0.78,0.76,0.29,0.65,0.89,0.6,0.41,0.91,0.11,0.62,0.34,0.88,0.5,
    0.75,0.98,0.45,0.97,0.54,0.14,0.99,0.49,0.48,0.42,0.77,0.11,0.36,0.29,0.07,0.85,0.06,0.96,
    0.41,0.42,0.35,0.9,0.66,0.09,0.65,0.89,0.48,0.94,0.31,0.09,0.36,0.43,0.17,0.9,0.91,0.52,
    0.35,0.52,0.42,0.16,0.95,0.12,0.21,0.53,0.29,0.61,0.85,0.67,0.42,0.11,0.96,0.53,0.78,
    0.37,0.29,0.12,0.79,0.73,0.5,0.24,0.04,0.21,0.85,0.03,0.73,0.64,0.41,0.42,0.34,0.13,
    0.36,0.29,0.69,0.53,0.13,0.64,0.07,0.66,0.99,0.83,0.51,0.07,0.36,0.64,0.26,0.76,0.96,
    0.73,0.43,0.63,0.22,0.71,0.86,0.32,0.93,1.0,0.27,0.37,0.83};


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
        using minutes_t = std::chrono::duration<double, std::ratio<60>>;

        std::chrono::time_point<steady_c> m0;
    public:
        Timer() : m0(steady_c::now()) {};
        ~Timer() {};

        double getTime() const
        {
            return std::chrono::duration_cast<minutes_t>(steady_c::now() - m0).count();
        } 
    };

    static Timer timer;
}


void multiplyLeft1(const double* A, const double *values, double *B)
{
    std::fill_n(B, NrowsOversamp * NcolsOversamp, 0);

    for (int i = 0; i < NrowsOversamp; ++i)
    {
        double *bsub = B + i * NcolsOversamp;
        const double
            *Asub = A + i * NcolsOversamp * Oversampling,
            *rrow = values + i * Nelemprow;

        cblas_dgemv(
            CblasRowMajor, CblasTrans,
            Nelemprow, NcolsOversamp, 1., Asub, NcolsOversamp, 
            rrow, 1, 0, bsub, 1);
    }
}

void multiplyLeft2(const double* A, const double *values, double *B)
{
    std::fill_n(B, NrowsOversamp * NcolsOversamp, 0);

    #pragma omp parallel for simd collapse(3)
    for (int j = 0; j < Nelemprow; ++j)
        for (int i = 0; i < NrowsOversamp; ++i)
            for (int k = 0; k < NcolsOversamp; ++k)
                B[k + i * NcolsOversamp] += 
                    A[k + (j + i * Oversampling) * NcolsOversamp] * values[j + i * Nelemprow];
}

void timeMultiplyLeft() {
    double t1, t2, difft;
    std::vector<double> mtrxB1(NrowsOversamp * NcolsOversamp, 0);

    // --------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        multiplyLeft1(oversample_multiplier_A, oversamp_values, mtrxB1.data());

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    printf("multiplyLeft1: %.2f s\n", difft * 60.);

    // --------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        multiplyLeft2(oversample_multiplier_A, oversamp_values, mtrxB1.data());

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    printf("multiplyLeft2: %.2f s\n", difft * 60.);
 }


int main(int argc, char *argv[]) {
    #if !defined(ENABLE_OMP)
    #error "testOMPTiming needs --enable-openmp to compile."
    #endif

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

    printf("OMP num threads %d.\n", omp_get_max_threads());
    timeMultiplyLeft();
}