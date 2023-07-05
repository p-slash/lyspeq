#ifndef BOOTSTRAPPER_H
#define BOOTSTRAPPER_H

#include <memory>
#include <random>
#include <string>
#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include <curand.h>

#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/progress.hpp"
#include "mathtools/cuda_helper.cuh"
#include "mathtools/matrix_helper.hpp"


__global__
void _convert_uint_double(int size, unsigned int *in, double *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        out[i] = in[i];
}

class PoissonRNG {
public:
    PoissonRNG(unsigned long int seed) {
        curand_stat = curandCreateGenerator(&rng_engine, CURAND_RNG_PSEUDO_DEFAULT);
        check_cuda_error("curandCreateGenerator");
        curand_stat = curandSetPseudoRandomGeneratorSeed(rng_engine, seed);
        check_cuda_error("curandSetPseudoRandomGeneratorSeed");
    }

    ~PoissonRNG() {
        curand_stat = curandDestroyGenerator(rng_engine);
        check_cuda_error("curandDestroyGenerator");
    }

    void generate(unsigned int *output, int n) {
        curand_stat = curandGeneratePoisson(rng_engine, output, n, 1);
        check_cuda_error("curandGeneratePoisson");
    }

private:
    curandStatus_t curand_stat;
    curandGenerator_t rng_engine;

    void check_cuda_error(std::string err_msg) {
        if (curand_stat != CURAND_STATUS_SUCCESS)
            throw std::runtime_error(err_msg);
    }
};


class PoissonBootstrapper {
public:
    PoissonBootstrapper(int num_boots) : nboots(num_boots) {
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);
        temppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
        tempfisher = std::make_unique<double[]>(bins::FISHER_SIZE);

        dev_tmp_power.realloc(bins::TOTAL_KZ_BINS);
        dev_tmp_fisher.realloc(bins::FISHER_SIZE);

        if (process::this_pe == 0)
            allpowers = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
    }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        nqsos = local_queue.size();
        dev_uint_coefficients.realloc(nqsos);
        cpu_uint_coefficients = std::make_unique<unsigned int[]>(nqsos);

        _prerun(local_queue);

        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);
        Progress prog_tracker(nboots);

        for (int jj = 0; jj < nboots; ++jj) {
            _one_boot(jj, local_queue);
            ++prog_tracker;
        }

        if (process::this_pe != 0)
            return;

        _calcuate_covariance();

        std::string buffer(process::FNAME_BASE);
        buffer += std::string("_bootstrap_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.get(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

private:
    unsigned int nboots, nqsos;
    MyCuPtr<double> dev_tmp_power, dev_tmp_fisher;
    MyCuPtr<unsigned int> dev_uint_coefficients;

    std::unique_ptr<double[]> temppower, tempfisher, allpowers;
    std::unique_ptr<unsigned int[]> cpu_uint_coefficients;

    std::unique_ptr<PoissonRNG> pgenerator;

    void _prerun(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto &one_qso : local_queue) {
            for (auto &one_chunk : one_qso->chunks) {
                one_chunk->releaseFile();

                int ndim = one_chunk->N_Q_MATRICES;
                double *pk = one_chunk->dev_dbt_vector.get(),
                       *nk = pk + ndim,
                       *tk = nk + ndim;

                cublas_helper.daxpy(-1, nk, pk, ndim);
                cublas_helper.daxpy(-1, tk, pk, ndim);
            }
        }
    }

    void _one_boot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        dev_tmp_power.memset();
        dev_tmp_fisher.memset();
        pgenerator->generate(dev_uint_coefficients.get(), nqsos);

        dev_uint_coefficients.syncDownload(cpu_uint_coefficients.get(), nqsos);

        unsigned int *p = cpu_uint_coefficients.get();
        for (auto &one_qso : local_queue) {
            if (*p == 0) {
                ++p;
                continue;
            }

            for (auto &one_chunk : one_qso->chunks)
                one_chunk->addBoot(*p, dev_tmp_power.get(), dev_tmp_fisher.get());
            ++p;
        }

        dev_tmp_power.syncDownload(temppower.get(), bins::TOTAL_KZ_BINS);

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.get(),
            allpowers.get() + jj * bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
        );
        if (process::this_pe != 0) {
            MPI_Reduce(
                dev_tmp_fisher.get(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                dev_tmp_fisher.get(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        if (process::this_pe != 0)
            return;

        dev_tmp_fisher.syncDownload(tempfisher.get(), bins::FISHER_SIZE);
        mxhelp::LAPACKE_solve_safe(
            tempfisher.get(), bins::TOTAL_KZ_BINS,
            allpowers.get() + jj * bins::TOTAL_KZ_BINS);
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
