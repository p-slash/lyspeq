#ifndef BOOTSTRAPPER_H
#define BOOTSTRAPPER_H

#include <memory>
#include <random>
#include <string>
#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/progress.hpp"
#include "mathtools/matrix_helper.hpp"


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
    static double time_spent_on_oneboot_loop = 0, time_spent_on_oneboot_mpi = 0,
                  time_spent_on_oneboot_solve = 0;

    void printfBootstrapTimeSpentDetails()
    {
        LOG::LOGGER.STD(
            "Total time spent in loop is %.2f mins.\n"
            "Total time spent in MPI is %.2f mins.\n"
            "Total time spent in solve is %.2f mins.\n",
            time_spent_on_oneboot_loop, time_spent_on_oneboot_mpi,
            time_spent_on_oneboot_solve);
    }
}

class PoissonBootstrapper {
public:
    PoissonBootstrapper(int num_boots) : nboots(num_boots) {
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);
        temppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
        tempfisher = std::make_unique<double[]>(bins::FISHER_SIZE);

        if (process::this_pe == 0)
            allpowers = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
    }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto one_qso = local_queue.begin(); one_qso != local_queue.end(); ++one_qso)
            (*one_qso)->collapseBootstrap();

        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);
        Progress prog_tracker(nboots);

        int ii = 0;
        for (int jj = 0; jj < nboots; ++jj) {
            bool valid = _one_boot(ii, local_queue);
            ++prog_tracker;
            if (valid) ++ii;
        }

        if (process::this_pe != 0)
            return;

        double success_ratio = (100. * ii) / nboots;
        LOG::LOGGER.STD(
            "Number of valid bootstraps %d of %d. Ratio %.2f%%\n",
            ii, nboots, success_ratio);
        if (success_ratio < 90.)
            LOG::LOGGER.ERR("WARNING: Low success ratio!\n");

        mytime::printfBootstrapTimeSpentDetails();

        _calcuate_covariance();

        std::string buffer(process::FNAME_BASE);
        buffer += std::string("_bootstrap_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.get(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    unsigned int nboots;
    std::unique_ptr<double[]> temppower, tempfisher, allpowers;
    std::unique_ptr<PoissonRNG> pgenerator;

    bool _one_boot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2;

        std::fill_n(temppower.get(), bins::TOTAL_KZ_BINS, 0);
        std::fill_n(tempfisher.get(), bins::FISHER_SIZE, 0);

        for (const auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            one_qso->addBoot(p, temppower.get(), tempfisher.get());
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.get(),
            allpowers.get() + jj * bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
        );
        if (process::this_pe != 0) {
            MPI_Reduce(
                tempfisher.get(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                tempfisher.get(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        bool valid = true;
        if (process::this_pe == 0) {
            // mxhelp::copyUpperToLower(tempfisher.get(), bins::TOTAL_KZ_BINS);

            t1 = mytime::timer.getTime();
            mytime::time_spent_on_oneboot_mpi += t1 - t2;

            try {
                mxhelp::LAPACKE_safeSolveCho(
                    tempfisher.get(), bins::TOTAL_KZ_BINS,
                    allpowers.get() + jj * bins::TOTAL_KZ_BINS);
            } catch (std::exception& e) {
                valid = false;
            }

            t2 = mytime::timer.getTime();
            mytime::time_spent_on_oneboot_solve += t2 - t1;
        }

        #if defined(ENABLE_MPI)
        MPI_Bcast(&valid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        #endif
        return valid;
    }

    void _calcuate_covariance() {
        LOG::LOGGER.STD("Calculating bootstrap covariance.\n");
        std::fill_n(temppower.get(), bins::TOTAL_KZ_BINS, 0);
        std::fill_n(tempfisher.get(), bins::FISHER_SIZE, 0);

        // Calculate mean power, store into temppower
        for (int jj = 0; jj < nboots; ++jj) {
            mxhelp::vector_add(
                temppower.get(),
                allpowers.get() + jj * bins::TOTAL_KZ_BINS,
                bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / nboots, temppower.get(), 1);

        for (int jj = 0; jj < nboots; ++jj) {
            double *x = allpowers.get() + jj * bins::TOTAL_KZ_BINS;
            mxhelp::vector_sub(x, temppower.get(), bins::TOTAL_KZ_BINS);
            cblas_dsyr(
                CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 1, x, 1,
                tempfisher.get(), bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::FISHER_SIZE, 1. / (nboots - 1), tempfisher.get(), 1);
        mxhelp::copyUpperToLower(tempfisher.get(), bins::TOTAL_KZ_BINS);
    }
};

#endif
