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
    PoissonBootstrapper(int num_boots, double *ifisher)
            : nboots(num_boots), invfisher(nullptr)
    {
        process::updateMemory(-getMinMemUsage());
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);

        if (specifics::FAST_BOOTSTRAP) {
            temppower.resize(nboots * bins::TOTAL_KZ_BINS);
            invfisher = ifisher;
        } else
            temppower.resize(bins::TOTAL_KZ_BINS);

        tempfisher.resize(bins::FISHER_SIZE);
        if (process::this_pe == 0)
            allpowers.resize(nboots * bins::TOTAL_KZ_BINS);
    }
    ~PoissonBootstrapper() { process::updateMemory(getMinMemUsage()); }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto one_qso = local_queue.begin(); one_qso != local_queue.end(); ++one_qso)
            (*one_qso)->collapseBootstrap();

        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);

        int ii = 0;
        if (specifics::FAST_BOOTSTRAP) {
            _fastBootstrap(local_queue);
            ii = nboots;
        }
        else {
            Progress prog_tracker(nboots);
            for (int jj = 0; jj < nboots; ++jj) {
                bool valid = _one_boot(ii, local_queue);
                ++prog_tracker;
                if (valid) ++ii;
            }
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
            buffer.c_str(), tempfisher.data(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    unsigned int nboots;
    std::vector<double> temppower, tempfisher, allpowers;
    std::unique_ptr<PoissonRNG> pgenerator;
    double *invfisher;

    double getMinMemUsage() {
        double memfull = process::getMemoryMB(nboots * bins::TOTAL_KZ_BINS);
        double needed_mem = (
            process::getMemoryMB(bins::FISHER_SIZE)
            + memfull / process::total_pes
        );

        if (specifics::FAST_BOOTSTRAP)
            needed_mem += memfull;
        else
            needed_mem += process::getMemoryMB(bins::TOTAL_KZ_BINS);
        return needed_mem;
    }

    void _fastBootstrap(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2 = 0;
        std::fill(temppower.begin(), temppower.end(), 0);

        Progress prog_tracker(nboots);
        for (int jj = 0; jj < nboots; ++jj)
        {
            for (const auto &one_qso : local_queue) {
                int p = pgenerator->generate();
                if (p == 0)
                    continue;

                one_qso->addBootPowerOnly(p, temppower.data() + jj * bins::TOTAL_KZ_BINS);
            }
            ++prog_tracker;
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        if (process::this_pe != 0) {
            MPI_Reduce(
                temppower.data(),
                nullptr, temppower.size(),
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                temppower.data(), temppower.size(),
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        if (process::this_pe == 0)
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nboots, bins::FISHER_SIZE, bins::FISHER_SIZE, 1., temppower.data(),
                bins::FISHER_SIZE, invfisher, bins::FISHER_SIZE,
                0, allpowers.data(), bins::FISHER_SIZE);

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;
    }

    bool _one_boot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2;

        std::fill(temppower.begin(), temppower.end(), 0);
        std::fill(tempfisher.begin(), tempfisher.end(), 0);

        for (const auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            one_qso->addBoot(p, temppower.data(), tempfisher.data());
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.data(),
            allpowers.data() + jj * bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
        );
        if (process::this_pe != 0) {
            MPI_Reduce(
                tempfisher.data(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                tempfisher.data(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        bool valid = true;
        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        if (process::this_pe == 0) {
            try {
                mxhelp::LAPACKE_safeSolveCho(
                    tempfisher.data(), bins::TOTAL_KZ_BINS,
                    allpowers.data() + jj * bins::TOTAL_KZ_BINS);
            } catch (std::exception& e) {
                valid = false;
            }
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Bcast(&valid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        #endif

        return valid;
    }

    void _calcuate_covariance() {
        LOG::LOGGER.STD("Calculating bootstrap covariance.\n");
        std::fill(temppower.begin(), temppower.begin() + bins::TOTAL_KZ_BINS, 0);
        std::fill(tempfisher.begin(), tempfisher.end(), 0);

        // Calculate mean power, store into temppower
        for (int jj = 0; jj < nboots; ++jj) {
            mxhelp::vector_add(
                temppower.data(),
                allpowers.data() + jj * bins::TOTAL_KZ_BINS,
                bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / nboots, temppower.data(), 1);

        for (int jj = 0; jj < nboots; ++jj) {
            double *x = allpowers.data() + jj * bins::TOTAL_KZ_BINS;
            mxhelp::vector_sub(x, temppower.data(), bins::TOTAL_KZ_BINS);
            cblas_dsyr(
                CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 1, x, 1,
                tempfisher.data(), bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::FISHER_SIZE, 1. / (nboots - 1), tempfisher.data(), 1);
        mxhelp::copyUpperToLower(tempfisher.data(), bins::TOTAL_KZ_BINS);
    }
};

#endif
