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


class PoissonBootstrapper
{
public:
    PoissonBootstrapper() {
        nboots = 2 * bins::FISHER_SIZE;
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);
        temppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
        tempfisher = std::make_unique<double[]>(bins::FISHER_SIZE);

        if (process::this_pe == 0)
            allpowers = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
    }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);
        Progress prog_tracker(nboots);

        for (int jj = 0; jj < nboots; ++jj)
        {
            _one_boot(jj, local_queue);
            ++prog_tracker;
        }

        _calcuate_covariance();

        std::string buffer(process::FNAME_BASE);
        buffer += std::string("_bootstrap_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.get(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    unsigned int nboots, nperiter;
    std::unique_ptr<double[]> temppower, tempfisher, allpowers;
    std::unique_ptr<PoissonRNG> pgenerator;

    void _prerun(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto &one_qso : local_queue) {
            for (auto &one_chunk : one_qso->chunks) {
                one_chunk->releaseFile();
                if (one_chunk->ZBIN == -1)  continue;

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
        std::fill_n(temppower.get(), bins::TOTAL_KZ_BINS, 0);
        std::fill_n(tempfisher.get(), bins::FISHER_SIZE, 0);

        for (auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            for (auto &one_chunk : one_qso->chunks) {
                if (one_chunk->ZBIN == -1)  continue;

                one_chunk->addBoot(p, temppower.get(), tempfisher.get());
            }
        }

        #if defined(ENABLE_MPI)
        if (process::this_pe != 0) {
            MPI_Reduce(
                temppower.get(),
                nullptr, bins::TOTAL_KZ_BINS,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(
                tempfisher.get(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                temppower.get(), bins::TOTAL_KZ_BINS,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(
                MPI_IN_PLACE,
                tempfisher.get(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        if (process::this_pe == 0)
        {
            mxhelp::copyUpperToLower(tempfisher.get(), bins::TOTAL_KZ_BINS);
            mxhelp::LAPACKE_InvertMatrixLU_safe(
                tempfisher.get(), bins::TOTAL_KZ_BINS);

            cblas_dsymv(
                CblasRowMajor, CblasUpper, bins::TOTAL_KZ_BINS, 1, 
                tempfisher.get(), bins::TOTAL_KZ_BINS,
                temppower.get(), 1,
                0, allpowers.get() + jj * bins::TOTAL_KZ_BINS, 1);
        }
    }

    void _calcuate_covariance() {
        std::fill_n(tempfisher.get(), bins::FISHER_SIZE, 0);
        auto meanpower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
        for (int jj = 0; jj < nboots; ++jj) {
            mxhelp::vector_add(
                meanpower.get(),
                allpowers.get() + jj * bins::TOTAL_KZ_BINS,
                bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / nboots, meanpower.get(), 1);

        for (int jj = 0; jj < nboots; ++jj) {
            double *x = allpowers.get() + jj * bins::TOTAL_KZ_BINS;
            mxhelp::vector_sub(x, meanpower.get(), bins::TOTAL_KZ_BINS);
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
