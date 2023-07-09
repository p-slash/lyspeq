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

    int generate() {
        return p_dist(rng_engine);
    }

private:
    std::mt19937_64 rng_engine;
    std::poisson_distribution<int> p_dist;
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
    PoissonBootstrapper(int num_boots) {
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);

        nboots_per_batch = std::min(num_boots, _calculate_nboots_per_batch());
        #if defined(ENABLE_MPI)
        MPI_Allreduce(
            MPI_IN_PLACE,
            &nboots_per_batch, 1,
            MPI_INT, MPI_MIN, MPI_COMM_WORLD
        );
        #endif

        if (nboots_per_batch == 0)
            throw std::runtime_error("Not enough memory to bootstrap.\n");

        nbatches = num_boots / nboots_per_batch;
        if (num_boots % nboots_per_batch != 0)
            ++nbatches;
        nboots = nboots_per_batch * nbatches;

        temppower = std::make_unique<double[]>(
            nboots_per_batch * bins::TOTAL_KZ_BINS);
        tempfisher = std::make_unique<double[]>(
            nboots_per_batch * bins::FISHER_SIZE);

        if (process::this_pe == 0)
            allpowers = std::make_unique<double[]>(
                nboots * bins::TOTAL_KZ_BINS);
    }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        _prerun(local_queue);

        LOG::LOGGER.STD(
            "Generating %u bootstrap realizations.\n"
            "Realizations per batch is %u.\n", nboots, nboots_per_batch);
        Progress prog_tracker(nboots);

        for (int ibatch = 0; ibatch < nbatches; ++ibatch)
            _batch_boot(ibatch, prog_tracker, local_queue);

        if (process::this_pe != 0)
            return;

        mytime::printfBootstrapTimeSpentDetails();

        _calcuate_covariance();

        std::string buffer(process::FNAME_BASE);
        buffer += std::string("_bootstrap_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.get(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    int nboots, nboots_per_batch, nbatches;
    std::unique_ptr<double[]> temppower, tempfisher, allpowers;
    std::unique_ptr<PoissonRNG> pgenerator;

    int _calculate_nboots_per_batch() {
        double size_one_batch =
            (double) sizeof(double)
            * (bins::FISHER_SIZE + bins::TOTAL_KZ_BINS)
            / 1048576.;

        double size_all_powers =
            (double) sizeof(double) * nboots * bins::TOTAL_KZ_BINS / 1048576.;

        process::updateMemory(-size_all_powers / process::total_pes);
        return (0.9 * process::MEMORY_ALLOC) / size_one_batch;
    }

    void _prerun(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto &one_qso : local_queue) {
            for (auto &one_chunk : one_qso->chunks) {
                one_chunk->releaseFile();

                int ndim = one_chunk->N_Q_MATRICES;
                double *pk = one_chunk->dbt_estimate_before_fisher_vector[0].get();
                double *nk = one_chunk->dbt_estimate_before_fisher_vector[1].get();
                double *tk = one_chunk->dbt_estimate_before_fisher_vector[2].get();

                mxhelp::vector_sub(pk, nk, ndim);
                mxhelp::vector_sub(pk, tk, ndim);
            }
        }
    }

    void _one_boot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double *tpk = temppower.get() + jj * bins::TOTAL_KZ_BINS,
               *tfm = tempfisher.get() + jj * bins::FISHER_SIZE;

        for (auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            for (auto &one_chunk : one_qso->chunks)
                one_chunk->addBoot(p, tpk, tfm);
        }
    }

    void _batch_boot(
            int ibatch, Progress &prog_tracker,
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2;

        std::fill_n(temppower.get(), nboots_per_batch * bins::TOTAL_KZ_BINS, 0);
        std::fill_n(tempfisher.get(), nboots_per_batch * bins::FISHER_SIZE, 0);

        double *apk =
            allpowers.get() + ibatch * nboots_per_batch * bins::TOTAL_KZ_BINS;

        for (int jj = 0; jj < nboots_per_batch; ++jj) {
            _one_boot(jj, local_queue);
            ++prog_tracker;
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.get(),
            apk, nboots_per_batch * bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
        );

        if (process::this_pe != 0) {
            MPI_Reduce(
                tempfisher.get(),
                nullptr, nboots_per_batch * bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                tempfisher.get(), nboots_per_batch * bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        if (process::this_pe != 0)
            return;

        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        for (int jj = 0; jj < nboots_per_batch; ++jj) {
            mxhelp::LAPACKE_solve_safe(
                tempfisher.get() + jj * bins::FISHER_SIZE,
                bins::TOTAL_KZ_BINS,
                apk + jj * bins::TOTAL_KZ_BINS);
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;
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
