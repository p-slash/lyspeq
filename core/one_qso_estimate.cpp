#include <algorithm>

#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include "mathtools/smoother.hpp"

#include <stdexcept>

const int
MAX_PIXELS_IN_FOREST = 700;

std::vector<int> OneQSOEstimate::decideIndices(int size) {
    int nchunks = 1;
    if (specifics::NUMBER_OF_CHUNKS > 1) {
        nchunks += (specifics::NUMBER_OF_CHUNKS * size) / MAX_PIXELS_IN_FOREST;
        nchunks = (nchunks > specifics::NUMBER_OF_CHUNKS) ? specifics::NUMBER_OF_CHUNKS : nchunks;
    }

    std::vector<int> indices;
    indices.reserve(nchunks + 1);
    for (int i = 0; i < nchunks; ++i)
        indices.push_back((int)((size * i) / nchunks));
    indices.push_back(size);

    return indices;
}


OneQSOEstimate::OneQSOEstimate(const std::string &f_qso)
{
    fname_qso = f_qso;
    qio::QSOFile qFile(fname_qso, specifics::INPUT_QSO_FILE);

    qFile.readParameters();
    qFile.readData();

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile.readAllocResolutionMatrix();

        if (process::smoother->isSmoothingOnRmat())
            process::smoother->smooth1D(
                qFile.Rmat->matrix(), qFile.Rmat->getNCols(),
                qFile.Rmat->getNElemPerRow());

        if (specifics::RESOMAT_DECONVOLUTION_M > 0)
            qFile.Rmat->deconvolve(specifics::RESOMAT_DECONVOLUTION_M);

        if (specifics::OVERSAMPLING_FACTOR > 0)
            qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    }

    qFile.closeFile();

    // Boundary cut
    qFile.cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

    if (qFile.realSize() < MIN_PIXELS_IN_CHUNK) {
        LOG::LOGGER.ERR(
            "OneQSOEstimate::OneQSOEstimate::Short file in %s.\n",
            fname_qso.c_str());

        return;
    }

    // decide nchunk with lambda points array[nchunks+1]
    std::vector<int> indices = OneQSOEstimate::decideIndices(qFile.size());
    int nchunks = indices.size() - 1;

    // create chunk objects
    chunks.reserve(nchunks);

    for (int nc = 0; nc < nchunks; ++nc)
    {
        try
        {
            auto _chunk = std::make_unique<Chunk>(qFile, indices[nc], indices[nc+1]);
            chunks.push_back(std::move(_chunk));
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR(
                "OneQSOEstimate::OneQSOEstimate::%s Skipping chunk %d/%d of %s.\n",
                e.what(), nc, nchunks, fname_qso.c_str());
        }
    }
}

double OneQSOEstimate::getComputeTimeEst(std::string fname_qso, int &zbin)
{
    zbin=-1;

    try
    {
        qio::QSOFile qtemp(fname_qso, specifics::INPUT_QSO_FILE);

        qtemp.readParameters();
        qtemp.readData();
        qtemp.cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        if (qtemp.realSize() < MIN_PIXELS_IN_CHUNK)
            return 0;

        double z1, z2, zm;
        qtemp.readMinMaxMedRedshift(z1, z2, zm);
        zbin = bins::findRedshiftBin(zm);

        // decide chunks
        std::vector<int> indices = OneQSOEstimate::decideIndices(qtemp.size());
        int nchunks = indices.size() - 1;

        // add compute time from chunks
        double res = 0;
        for (int nc = 0; nc < nchunks; ++nc)
            res += Chunk::getComputeTimeEst(qtemp, indices[nc], indices[nc+1]);
        return res;
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR(
            "OneQSOEstimate::getComputeTimeEst::%s. Skipping %s.\n",
            e.what(), fname_qso.c_str());
        return 0;
    }
}

void OneQSOEstimate::oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum
) {
    for (auto &chunk : chunks) {
        try {
            chunk->oneQSOiteration(ps_estimate, dbt_sum_vector, fisher_sum);
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR(
                "OneQSOEstimate::oneQSOiteration::%s Skipping %s.\n",
                e.what(), fname_qso.c_str());
            chunk.reset();
        }
    }

    chunks.erase(std::remove_if(
        chunks.begin(), chunks.end(), [](const auto &x) { return !x; }),
        chunks.end()
    );
}


void OneQSOEstimate::collapseBootstrap() {
    int fisher_index_end = 0;

    istart = bins::TOTAL_KZ_BINS;
    for (const auto &chunk : chunks) {
        istart = std::min(istart, chunk->fisher_index_start);
        fisher_index_end = std::max(
            fisher_index_end, chunk->fisher_index_start + chunk->N_Q_MATRICES);
    }

    ndim = fisher_index_end - istart;

    theta_vector = std::make_unique<double[]>(ndim);
    fisher_matrix = std::make_unique<double[]>(ndim * ndim);

    for (const auto &chunk : chunks) {
        int offset = chunk->fisher_index_start - istart;
        double *pk = chunk->dbt_estimate_before_fisher_vector[0].get();
        double *nk = chunk->dbt_estimate_before_fisher_vector[1].get();
        double *tk = chunk->dbt_estimate_before_fisher_vector[2].get();

        cblas_daxpy(chunk->N_Q_MATRICES, -1, nk, 1, pk, 1);
        cblas_daxpy(chunk->N_Q_MATRICES, -1, tk, 1, pk, 1);
        cblas_daxpy(chunk->N_Q_MATRICES, 1, pk, 1, theta_vector.get() + offset, 1);

        for (int i = 0; i < chunk->N_Q_MATRICES; ++i) {
            for (int j = i; j < chunk->N_Q_MATRICES; ++j) {
                fisher_matrix[j + i * ndim + (ndim + 1) * offset] +=
                    chunk->fisher_matrix[j + i * chunk->N_Q_MATRICES];
            } 
        }
    }

    chunks.clear();
}




