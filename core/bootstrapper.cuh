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
#include "mathtools/cuda_helper.cuh"
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
        tempdata = std::make_unique<double[]>(
            bins::FISHER_SIZE + bins::TOTAL_KZ_BINS);

        dev_tmp_data.realloc(bins::FISHER_SIZE + bins::TOTAL_KZ_BINS);

        if (process::this_pe == 0)
            allpowers = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
    }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        _prerun(local_queue);

        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);
        Progress prog_tracker(nboots);

        for (int jj = 0; jj < nboots; ++jj) {
            _one_boot(jj, local_queue);
            ++prog_tracker;
        }

        if (process::this_pe != 0)
            return;

        mytime::printfBootstrapTimeSpentDetails();

        _calcuate_covariance();

        std::string buffer(process::FNAME_BASE);
        buffer += std::string("_bootstrap_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempdata.get() + bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    unsigned int nboots;
    MyCuPtr<double> dev_tmp_data;
    std::unique_ptr<double[]> tempdata, allpowers;
    std::unique_ptr<PoissonRNG> pgenerator;

    void _prerun(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double subt_alpha = -1;
        // Get thetas to prepare.
        for (auto &one_qso : local_queue) {
            for (auto &one_chunk : one_qso->chunks) {
                one_chunk->releaseFile();

                int ndim = one_chunk->N_Q_MATRICES;
                double *pk = one_chunk->dev_dbt_vector.get(),
                       *nk = pk + ndim,
                       *tk = nk + ndim;

                cublas_helper.daxpy(&subt_alpha, nk, pk, ndim);
                cublas_helper.daxpy(&subt_alpha, tk, pk, ndim);
            }
        }
    }

    void _one_boot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        dev_tmp_data.memset();

        double t1 = mytime::timer.getTime(), t2;
        double *dev_tmp_power = dev_tmp_data.get(),
               *dev_tmp_fisher = dev_tmp_power + bins::TOTAL_KZ_BINS;

        for (auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            for (auto &one_chunk : one_qso->chunks)
                one_chunk->addBoot(p, dev_tmp_power, dev_tmp_fisher);
        }

        dev_tmp_data.syncDownload(
            tempdata.get(), bins::FISHER_SIZE + bins::TOTAL_KZ_BINS);

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        if (process::this_pe != 0) {
            MPI_Reduce(
                tempdata.get(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                tempdata.get(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        if (process::this_pe != 0)
            return;

        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        double *temppower = allpowers.get() + jj * bins::TOTAL_KZ_BINS,
               *tempfisher = tempdata.get() + bins::TOTAL_KZ_BINS;

        std::copy_n(tempdata.get(), bins::TOTAL_KZ_BINS, temppower);
        mxhelp::LAPACKE_solve_safe(tempfisher, bins::TOTAL_KZ_BINS, temppower);

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;
    }

    void _calcuate_covariance() {
        LOG::LOGGER.STD("Calculating bootstrap covariance.\n");
        std::fill_n(tempdata.get(), bins::FISHER_SIZE + bins::TOTAL_KZ_BINS, 0);

        double *temppower = tempdata.get(), 
               *tempfisher = temppower + bins::TOTAL_KZ_BINS;

        // Calculate mean power, store into temppower
        for (int jj = 0; jj < nboots; ++jj) {
            mxhelp::vector_add(
                temppower,
                allpowers.get() + jj * bins::TOTAL_KZ_BINS,
                bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / nboots, temppower, 1);

        for (int jj = 0; jj < nboots; ++jj) {
            double *x = allpowers.get() + jj * bins::TOTAL_KZ_BINS;
            mxhelp::vector_sub(x, temppower, bins::TOTAL_KZ_BINS);
            cblas_dsyr(
                CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 1, x, 1,
                tempfisher, bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::FISHER_SIZE, 1. / (nboots - 1), tempfisher, 1);
        mxhelp::copyUpperToLower(tempfisher, bins::TOTAL_KZ_BINS);
    }
};

#endif
