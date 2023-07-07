#include <chrono>
#include <memory>
#include <random>
#include <string>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif


const int TOTAL_KZ_BINS = 35 * 13, N_LOOPS = 450000;


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


namespace mytime
{
    static double time_spent_on_addboot[] = {0, 0}, time_spent_on_pgenerate = 0;

    void printfBootstrapTimeSpentDetails()
    {
        printf(
            "Total time spent in addboot1 is %.2f mins.\n"
            "Total time spent in addboot2 is %.2f mins.\n"
            "Total time spent in pgenerate is %.2f mins.\n",
            time_spent_on_addboot[0], time_spent_on_addboot[1],
            time_spent_on_pgenerate);
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


class DummyChunk {
protected:
    const int fisher_index_start = 35 * 2, N_Q_MATRICES = 35 * 2;
    std::unique_ptr<double[]> fisher_matrix, d_vector;
    std::unique_ptr<PoissonRNG> pgenerator;
public:
    DummyChunk() {
        pgenerator = std::make_unique<PoissonRNG>(123);
        fisher_matrix = std::make_unique<double[]>(N_Q_MATRICES * N_Q_MATRICES),
        d_vector = std::make_unique<double[]>(N_Q_MATRICES);
        int nruns = (N_Q_MATRICES + 1) * N_Q_MATRICES;

        for (int i = 0; i < N_Q_MATRICES; ++i)
            d_vector[i] = pgenerator->generate();
        for (int i = 0; i < N_Q_MATRICES * N_Q_MATRICES; ++i)
            fisher_matrix[i] = pgenerator->generate();
    }

    void addboot(int p, double *temppower, double* tempfisher) {
        for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
        {
            int idx_fji_0 =
                (TOTAL_KZ_BINS + 1) * (i_kz + fisher_index_start);
            int ncopy = N_Q_MATRICES - i_kz;

            cblas_daxpy(
                ncopy,
                p, fisher_matrix.get() + i_kz * (N_Q_MATRICES + 1), 1,
                tempfisher + idx_fji_0, 1);
        }

        cblas_daxpy(
            N_Q_MATRICES,
            p, d_vector.get(), 1,
            temppower + fisher_index_start, 1);
    }

    void addboot2(int p, double *temppower, double* tempfisher) {
        for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
        {
            int idx_fji_0 = TOTAL_KZ_BINS * (i_kz + fisher_index_start);

            cblas_daxpy(
                N_Q_MATRICES,
                p, fisher_matrix.get() + i_kz * N_Q_MATRICES, 1,
                tempfisher + idx_fji_0, 1);
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
        dumdum.addboot(p, temppower.get(), tempfisher.get());

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_addboot[0] += t2 - t1;

    for (int i = 0; i < N_LOOPS; ++i)
        dumdum.addboot2(p, temppower.get(), tempfisher.get());

    t1 = mytime::timer.getTime();
    mytime::time_spent_on_addboot[1] += t1 - t2;
}


void time_poisson() {
    PoissonRNG pgenerator(234);
    auto dumrandoms = std::make_unique<double[]>(N_LOOPS);

    double t1 = mytime::timer.getTime(), t2;

    for (int i = 0; i < N_LOOPS; ++i)
        dumrandoms[i] = pgenerator.generate();

    t2 = mytime::timer.getTime();
    mytime::time_spent_on_pgenerate += t2 - t1;
}


int main(int argc, char *argv[]) {
    int number_lapse = 10;
    if (argc == 2)
        number_lapse = atoi(argv[1]);

    printf("Doing %d lapses with %d loops each.\n", number_lapse, N_LOOPS);

    for (int i = 0; i < number_lapse; ++i)
    {
        time_poisson();
        time_addboot();
    }

    mytime::printfBootstrapTimeSpentDetails();

    return 0;
}

