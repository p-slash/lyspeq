#include <algorithm>

#include "core/one_qso_estimate.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include "mathtools/smoother.hpp"

#include <stdexcept>

const int MIN_PIXELS_IN_FILE = 100;


std::vector<int> OneQSOEstimate::decideIndices(int size, double *wave) {
    int nchunks = 1;
    auto velarr = std::make_unique<double[]>(size);
    double wave0 = wave[0];
    std::transform(
        wave, wave + size, velarr.get(),
        [&wave0](const double &w) { return SPEED_OF_LIGHT * log(w / wave0); });
    double vmax = velarr[size - 1];

    if (specifics::NUMBER_OF_CHUNKS > 1) {
        nchunks += vmax / specifics::MAX_FOREST_LENGTH_V;
        // nchunks = std::min(nchunks, specifics::NUMBER_OF_CHUNKS);
        // nchunks can be greater than specifics::NUMBER_OF_CHUNKS to
        // achieve consistent velocity lengths. However, this should not
        // happen if specifics::MAX_FOREST_LENGTH_V > vmax for all spectra.

        vmax = std::min(specifics::MAX_FOREST_LENGTH_V, vmax / nchunks);
    }

    std::vector<int> indices;
    indices.reserve(nchunks + 1);
    indices.push_back(0);
    double *v1 = velarr.get(), *v2 = v1 + size, *vl = v1;
    for (int i = 0; i < nchunks; ++i) {
        int jj = std::upper_bound(vl, v2, vmax) - v1;
        indices.push_back(jj);
        if (jj == size)  break;
        vl = v1 + jj;
        double vl0 = *vl;
        std::for_each(vl, v2, [&vl0](double &v) { v -= vl0; });
    }

    return indices;
}

std::unique_ptr<qio::QSOFile> OneQSOEstimate::_readQsoFile(const std::string &f_qso) {
    auto qFile = std::make_unique<qio::QSOFile>(f_qso, specifics::INPUT_QSO_FILE);

    qFile->readParameters();

    if (qFile->snr < specifics::MIN_SNR_CUT)
        throw std::runtime_error("OneQSOEstimate::_readQsoFile::Low SNR");

    qFile->readData();

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(
        qFile->wave(), qFile->delta(), qFile->ivar(), qFile->size());

    int num_outliers = qFile->maskOutliers();
    if (num_outliers > 0)
        LOG::LOGGER.ERR(
            "WARNING::OneQSOEstimate::_readQsoFile::"
            "Found %d outlier pixels in %s.\n",
            num_outliers, f_qso.c_str());

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile->readAllocResolutionMatrix();

        if (process::smoother->isSmoothingOnRmat())
            process::smoother->smooth1D(
                qFile->Rmat->matrix(), qFile->Rmat->getNCols(),
                qFile->Rmat->getNElemPerRow());

        if (specifics::RESOMAT_DECONVOLUTION_M > 0)
            qFile->Rmat->deconvolve(specifics::RESOMAT_DECONVOLUTION_M);

        if (specifics::OVERSAMPLING_FACTOR > 0)
            qFile->Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile->dlambda);
    }

    qFile->closeFile();

    // Boundary cut
    qFile->cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

    if (qFile->realSize() < MIN_PIXELS_IN_FILE)
        throw std::runtime_error("OneQSOEstimate::_readQsoFile::Short file");

    return qFile;
}


OneQSOEstimate::OneQSOEstimate(const std::string &f_qso)
{
    try {
        auto qFile = _readQsoFile(f_qso);

        // decide nchunk with lambda points array[nchunks+1]
        std::vector<int> indices = OneQSOEstimate::decideIndices(
            qFile->size(), qFile->wave());
        int nchunks = indices.size() - 1;

        // create chunk objects
        chunks.reserve(nchunks);

        for (int nc = 0; nc < nchunks; ++nc) {
            try {
                auto _chunk = std::make_unique<Chunk>(*qFile, indices[nc], indices[nc+1]);
                chunks.push_back(std::move(_chunk));
            } catch (std::exception& e) {
                LOG::LOGGER.ERR(
                    "OneQSOEstimate::OneQSOEstimate::%s Skipping chunk %d/%d of %s.\n",
                    e.what(), nc + 1, nchunks, f_qso.c_str());
            }
        }
    } catch (std::exception &e) {
        LOG::LOGGER.ERR("%s in %s.\n", e.what(), f_qso.c_str());
        return;
    }
}

double OneQSOEstimate::getComputeTimeEst(std::string fname_qso, int &zbin, long &targetid)
{
    zbin=-1;
    targetid=-1;

    try
    {
        qio::QSOFile qtemp(fname_qso, specifics::INPUT_QSO_FILE);

        qtemp.readParameters();
        targetid = qtemp.id;
        if (qtemp.snr < specifics::MIN_SNR_CUT)
            return 0;

        qtemp.readData();
        qtemp.cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        if (qtemp.realSize() < MIN_PIXELS_IN_FILE)
            return 0;

        double z1, z2, zm;
        qtemp.readMinMaxMedRedshift(z1, z2, zm);
        zbin = bins::findRedshiftBin(zm);

        // decide chunks
        std::vector<int> indices = OneQSOEstimate::decideIndices(
            qtemp.size(), qtemp.wave());
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
                e.what(), chunk->qFile->fname.c_str());
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

    for (const auto &chunk : chunks) {
        int offset = chunk->fisher_index_start - istart;
        double *pk = chunk->dbt_estimate_before_fisher_vector[0].get();
        double *nk = chunk->dbt_estimate_before_fisher_vector[1].get();
        double *tk = chunk->dbt_estimate_before_fisher_vector[2].get();

        cblas_daxpy(chunk->N_Q_MATRICES, -1, nk, 1, pk, 1);
        cblas_daxpy(chunk->N_Q_MATRICES, -1, tk, 1, pk, 1);
        cblas_daxpy(chunk->N_Q_MATRICES, 1, pk, 1, theta_vector.get() + offset, 1);
    }

    if (!specifics::FAST_BOOTSTRAP) {
        fisher_matrix = std::make_unique<double[]>(ndim * ndim);

        for (const auto &chunk : chunks) {
            int offset = chunk->fisher_index_start - istart;

            for (int i = 0; i < chunk->N_Q_MATRICES; ++i)
                for (int j = i; j < chunk->N_Q_MATRICES; ++j)
                    fisher_matrix[j + i * ndim + (ndim + 1) * offset] +=
                        chunk->fisher_matrix[j + i * chunk->N_Q_MATRICES];
        }
    }

    chunks.clear();
}




