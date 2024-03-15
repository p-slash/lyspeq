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
#include "mathtools/discrete_interpolation.hpp"

int TOTAL_KZ_BINS = 35 * 13, N_LOOPS = 450000;
const double SPEED_OF_LIGHT = 299792.458, LYA_REST = 1215.67;


inline
bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8)
{
    double mag = std::max(fabs(a),fabs(b));
    return fabs(a-b) < (abserr + relerr * mag);
}


bool allClose(const double *a, const double *b, int size)
{
    bool result = true;
    for (int i = 0; i < size; ++i)
        result &= isClose(a[i], b[i]);
    return result;
}


class PoissonRNG {
public:
    PoissonRNG(unsigned long int seed) {
        rng_engine.seed(seed);
    }

    unsigned int generate() {
        return p_dist(rng_engine);
    }

private:
    std::mt19937_64 rng_engine;
    std::poisson_distribution<unsigned int> p_dist;
};


namespace mytime {
    static double time_spent_on_func[] = {0, 0};

    void printfBootstrapTimeSpentDetails(const char funcname[]="addBoot")
    {
        printf(
            "Total time spent in %s (no omp) is %.2f s.\n"
            "Total time spent in %s (w/ omp) is %.2f s.\n",
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
        using minutes_t = std::chrono::duration<double, std::ratio<1>>;

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


class DummyChunk {
protected:
    const int fisher_index_start = 35 * 2, N_Q_MATRICES = 35 * 2;
    const int DATA_SIZE = 350;

    std::unique_ptr<double[]> fisher_matrix, d_vector;
    std::unique_ptr<double[]> vmatrix, zmatrix, lambda, qmatrix;
    std::unique_ptr<PoissonRNG> pgenerator;
    std::unique_ptr<DiscreteInterpolation1D> interp1d;
public:
    DummyChunk() {
        pgenerator = std::make_unique<PoissonRNG>(123);
        fisher_matrix = std::make_unique<double[]>(N_Q_MATRICES * N_Q_MATRICES),
        d_vector = std::make_unique<double[]>(N_Q_MATRICES);

        lambda = std::make_unique<double[]>(DATA_SIZE);
        vmatrix = std::make_unique<double[]>(DATA_SIZE * DATA_SIZE);
        zmatrix = std::make_unique<double[]>(DATA_SIZE * DATA_SIZE);
        qmatrix = std::make_unique<double[]>(DATA_SIZE * DATA_SIZE);

        for (int i = 0; i < N_Q_MATRICES; ++i)
            d_vector[i] = pgenerator->generate();
        for (int i = 0; i < N_Q_MATRICES * N_Q_MATRICES; ++i)
            fisher_matrix[i] = pgenerator->generate();
        for (int i = 0; i < DATA_SIZE; ++i)
            lambda[i] = (3800. + 0.8 * i) / LYA_REST;

        double vmax = SPEED_OF_LIGHT * log(lambda[DATA_SIZE - 1] / lambda[0]),
               dv = 10.;
        int narr = vmax / dv + 1;
        auto yarr = std::make_unique<double[]>(narr);
        for (int i = 0; i < narr; ++i)
            yarr[i] = 10. * exp(-i * dv / 100.);

        interp1d = std::make_unique<DiscreteInterpolation1D>(
            0, dv, narr, yarr.get());
    }

    void setVZMatrices_noomp() {
        for (int i = 0; i < DATA_SIZE; ++i)
        {
            for (int j = i; j < DATA_SIZE; ++j)
            {
                double li = lambda[i], lj = lambda[j];

                vmatrix[j + i * DATA_SIZE] = SPEED_OF_LIGHT * log(lj / li);
                zmatrix[j + i * DATA_SIZE] = sqrt(li * lj) - 1.;
            }
        }
    }

    void setVZMatrices_omp() {
        #pragma omp parallel for simd collapse(2)
        for (int i = 0; i < DATA_SIZE; ++i)
        {
            for (int j = i; j < DATA_SIZE; ++j)
            {
                double li = lambda[i], lj = lambda[j];

                vmatrix[j + i * DATA_SIZE] = SPEED_OF_LIGHT * log(lj / li);
                zmatrix[j + i * DATA_SIZE] = sqrt(li * lj) - 1.;
            }
        }
    }

    void setQiMatrix_noomp() {
        for (int i = 0; i < DATA_SIZE; ++i) {
            for (int j = i; j < DATA_SIZE; ++j) {
                int idx = j + i * DATA_SIZE;
                qmatrix[idx] = zmatrix[idx] * interp1d->evaluate(vmatrix[idx]);
            }
        }
    }

    void setQiMatrix_omp() {
        #pragma omp parallel for simd collapse(2)
        for (int i = 0; i < DATA_SIZE; ++i) {
            for (int j = i; j < DATA_SIZE; ++j) {
                int idx = j + i * DATA_SIZE;
                qmatrix[idx] = zmatrix[idx] * interp1d->evaluate(vmatrix[idx]);
            }
        }
    }

    void addboot_noomp(int p, double *temppower, double* tempfisher) {
        double *outfisher =
            tempfisher + (TOTAL_KZ_BINS + 1) * fisher_index_start;

        for (int i = 0; i < N_Q_MATRICES; ++i) {
            for (int j = i; j < N_Q_MATRICES; ++j) {
                outfisher[j + i * TOTAL_KZ_BINS] +=
                    p * fisher_matrix[j + i * N_Q_MATRICES];
            } 
        }

        cblas_daxpy(
            N_Q_MATRICES,
            p, d_vector.get(), 1,
            temppower + fisher_index_start, 1);
    }

    void addboot_omp(int p, double *temppower, double* tempfisher) {
        double *outfisher =
            tempfisher + (TOTAL_KZ_BINS + 1) * fisher_index_start;

        #pragma omp parallel for simd collapse(2)
        for (int i = 0; i < N_Q_MATRICES; ++i) {
            for (int j = i; j < N_Q_MATRICES; ++j) {
                outfisher[j + i * TOTAL_KZ_BINS] +=
                    p * fisher_matrix[j + i * N_Q_MATRICES];
            } 
        }

        cblas_daxpy(
            N_Q_MATRICES,
            p, d_vector.get(), 1,
            temppower + fisher_index_start, 1);
    }
};


void time_addboot() {
    DummyChunk dumdum;
    auto temppower = std::make_unique<double[]>(TOTAL_KZ_BINS);
    auto tempfisher = std::make_unique<double[]>(TOTAL_KZ_BINS * TOTAL_KZ_BINS);
    int p = 1;

    double t1 = mytime::timer.getTime(), t2;

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.addboot_noomp(p, temppower.get(), tempfisher.get());

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[0] += t2 - t1;

    std::fill_n(temppower.get(), TOTAL_KZ_BINS, 0);
    std::fill_n(tempfisher.get(), TOTAL_KZ_BINS * TOTAL_KZ_BINS, 0);

    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.addboot_omp(p, temppower.get(), tempfisher.get());

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[1] += t2 - t1;
}

void time_vzsetting() {
    DummyChunk dumdum;

    double t1 = mytime::timer.getTime(), t2;

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.setVZMatrices_noomp();

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[0] += t2 - t1;

    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.setVZMatrices_omp();

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[1] += t2 - t1;
}

void time_setqi() {
    DummyChunk dumdum;

    double t1 = mytime::timer.getTime(), t2;

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.setQiMatrix_noomp();

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[0] += t2 - t1;

    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.setQiMatrix_omp();

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_func[1] += t2 - t1;
}


void transpose_copy_base(const double *A, double *B, int N) {
    for (int i = 0; i < N; ++i)
        cblas_dcopy(N, A + i * N, 1, B + i, N);
}


void transpose_copy_block_lk(const double *A, double *B, int N) {
    #define BLOCK_SIZE 32

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // transpose the block beginning at [i,j]
            int kmax = std::min(i + BLOCK_SIZE, N),
                lmax = std::min(j + BLOCK_SIZE, N);

            #pragma omp simd collapse(2)
            for (int l = j; l < lmax; ++l)
                for (int k = i; k < kmax; ++k)
                    B[k + l * N] = A[l + k * N];
        }
    }

    #undef BLOCK_SIZE
}

void transpose_copy_block_kl(const double *A, double *B, int N) {
    #define BLOCK_SIZE 32

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // transpose the block beginning at [i,j]
            int kmax = std::min(i + BLOCK_SIZE, N),
                lmax = std::min(j + BLOCK_SIZE, N);

            #pragma omp simd collapse(2)
            for (int k = i; k < kmax; ++k)
                for (int l = j; l < lmax; ++l)
                    B[k + l * N] = A[l + k * N];
        }
    }

    #undef BLOCK_SIZE
}

void transpose_copy_block_mem(const double *A, double *B, int N) {
    #define BLOCK_SIZE 32

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            // transpose the block beginning at [i,j]
            int kmax = std::min(i + BLOCK_SIZE, N) - i,
                lmax = std::min(j + BLOCK_SIZE, N) - j;
            double block[BLOCK_SIZE * BLOCK_SIZE];

            #pragma omp simd collapse(2)
            for (int k = 0; k < kmax; ++k)
                for (int l = 0; l < lmax; ++l)
                    block[l + k * BLOCK_SIZE] = A[l + j + (k + i) * N];

            #pragma omp simd collapse(2)
            for (int l = 0; l < lmax; ++l)
                for (int k = 0; k < kmax; ++k)
                    B[k + i + (l + j) * N] = block[l + k * BLOCK_SIZE];
        }
    }

    #undef BLOCK_SIZE
}

void copyUpperToLower_base(double *A, int N) {
    for (int i = 1; i < N; ++i)
        cblas_dcopy(i, A+i, N, A+N*i, 1);
}


void copyUpperToLower(double *A, int N)
{
    #define BLOCK_SIZE 32

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = i; j < N; j += BLOCK_SIZE) {
            int kmax = std::min(i + BLOCK_SIZE, N),
                lmax = std::min(j + BLOCK_SIZE, N);

            // #pragma omp simd collapse(2)
            for (int k = i; k < kmax; ++k)
                #pragma omp simd
                for (int l = std::max(k, j); l < lmax; ++l)
                    A[k + l * N] = A[l + k * N];
        }
    }

    #undef BLOCK_SIZE
}


void time_transpose_copy() {
    #define NDIM 1000
    auto A = std::make_unique<double[]>(NDIM * NDIM),
         B = std::make_unique<double[]>(NDIM * NDIM),
         T = std::make_unique<double[]>(NDIM * NDIM);

    for (int i = 0; i < NDIM * NDIM; ++i)
        A[i] = i;

    // ------
    double t1 = mytime::timer.getTime(), t2, difft;

    for (int i = 0; i < N_LOOPS; ++i)
        transpose_copy_base(A.get(), T.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    printf("transpose_copy_base: %.2f s\n", difft);

    // ------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        transpose_copy_block_lk(A.get(), B.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    if (not allClose(B.get(), T.get(), NDIM * NDIM))
        printf("Error. ");
    printf("transpose_copy_block_lk: %.2f s\n", difft);

    // ------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        transpose_copy_block_kl(A.get(), B.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    if (not allClose(B.get(), T.get(), NDIM * NDIM))
        printf("Error. ");
    printf("transpose_copy_block_kl: %.2f s\n", difft);

    // ------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        transpose_copy_block_mem(A.get(), B.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    if (not allClose(B.get(), T.get(), NDIM * NDIM))
        printf("Error. ");
    printf("transpose_copy_block_mem: %.2f s\n", difft);

    for (int i = 0; i < NDIM * NDIM; ++i)
        A[i] = i;
    for (int i = 0; i < NDIM * NDIM; ++i)
        B[i] = i;
    for (int i = 0; i < NDIM * NDIM; ++i)
        T[i] = i;

    // ------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        copyUpperToLower_base(T.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    printf("copyUpperToLower_base: %.2f s\n", difft);

    // ------
    t1 = mytime::timer.getTime();

    for (int i = 0; i < N_LOOPS; ++i)
        copyUpperToLower(B.get(), NDIM);

    t2 = mytime::timer.getTime();
    difft = t2 - t1;
    if (not allClose(B.get(), T.get(), NDIM * NDIM))
        printf("Error. ");
    printf("copyUpperToLower: %.2f s\n", difft);
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
    printf(
        "Timing transpose_copy...\nDoing %d loops each.\n", N_LOOPS);
    time_transpose_copy();
    fflush(stdout);

    printf(
        "Timing addBoot...\nDoing %d lapses with %d loops each.\n",
        number_lapse, N_LOOPS);
    fflush(stdout);

    for (int i = 0; i < number_lapse; ++i)
        time_addboot();

    mytime::printfBootstrapTimeSpentDetails();
    printf(
        "NOTE: addBoot does not use OpenMP. Enabling or disabling OpenMP "
        "will not change performance.\n");

    if (argc == 1)
        number_lapse = 1;

    mytime::resetTime();
    printf(
        "Timing setVZMatrices...\nDoing %d lapses with %d loops each.\n",
        number_lapse, N_LOOPS);
    fflush(stdout);

    for (int i = 0; i < number_lapse; ++i)
        time_vzsetting();

    mytime::printfBootstrapTimeSpentDetails("setVZMatrices");
    printf("NOTE: enabling or disabling OpenMP can improve performance.\n");

    mytime::resetTime();
    printf(
        "Timing setQiMatrix...\nDoing %d lapses with %d loops each.\n",
        number_lapse, N_LOOPS);
    fflush(stdout);

    for (int i = 0; i < number_lapse; ++i)
        time_setqi();

    mytime::printfBootstrapTimeSpentDetails("setQiMatrix");
    printf("NOTE: enabling or disabling OpenMP can improve performance.\n");

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 5; i++)
        for (int j = i; j < 5; j++)
            printf("%d %d %d\n", i, j, omp_get_thread_num());

    return 0;
}

