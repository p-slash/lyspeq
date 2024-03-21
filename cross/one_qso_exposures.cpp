#include <algorithm>

#include "cross/one_qso_exposures.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include "mathtools/smoother.hpp"

#include <stdexcept>


OneQsoExposures::OneQsoExposures(const std::string &f_qso)
{
    qio::QSOFile qFile(f_qso, specifics::INPUT_QSO_FILE);

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
            "OneQsoExposures::OneQsoExposures::Short file in %s.\n",
            f_qso.c_str());

        return;
    }

    targetid = qFile.id;
    exposures.reserve(20);
    std::vector<int> indices = OneQSOEstimate::decideIndices(qFile.size());
    int nchunks = indices.size() - 1;

    for (int nc = 0; nc < nchunks; ++nc)
    {
        try
        {
            auto _chunk = std::make_unique<Exposure>(
                qFile, indices[nc], indices[nc + 1]);
            exposures.push_back(std::move(_chunk));
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR(
                "OneQsoExposures::appendChunkedQsoFile::%s "
                "Skipping chunk %d/%d of %s.\n",
                e.what(), nc, nchunks, qFile.fname.c_str());
        }
    }
}


void OneQsoExposures::oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum
) {
    for (auto &exp : exposures) {
        try {
            exp->oneQSOiteration(ps_estimate, dbt_sum_vector, fisher_sum);
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR(
                "OneQsoExposures::oneQSOiteration::%s Skipping %s.\n",
                e.what(), exp->qFile->fname.c_str());
            exp.reset();
        }
    }

    exposures.erase(std::remove_if(
        exposures.begin(), exposures.end(), [](const auto &x) { return !x; }),
        exposures.end()
    );
}


void OneQsoExposures::collapseBootstrap() {
    int fisher_index_end = 0;

    istart = bins::TOTAL_KZ_BINS;
    for (const auto &exp : exposures) {
        istart = std::min(istart, exp->fisher_index_start);
        fisher_index_end = std::max(
            fisher_index_end, exp->fisher_index_start + exp->N_Q_MATRICES);
    }

    ndim = fisher_index_end - istart;

    theta_vector = std::make_unique<double[]>(ndim);
    fisher_matrix = std::make_unique<double[]>(ndim * ndim);

    for (const auto &exp : exposures) {
        int offset = exp->fisher_index_start - istart;
        double *pk = exp->dbt_estimate_before_fisher_vector[0].get();
        double *nk = exp->dbt_estimate_before_fisher_vector[1].get();
        double *tk = exp->dbt_estimate_before_fisher_vector[2].get();

        cblas_daxpy(exp->N_Q_MATRICES, -1, nk, 1, pk, 1);
        cblas_daxpy(exp->N_Q_MATRICES, -1, tk, 1, pk, 1);
        cblas_daxpy(exp->N_Q_MATRICES, 1, pk, 1, theta_vector.get() + offset, 1);

        for (int i = 0; i < exp->N_Q_MATRICES; ++i) {
            for (int j = i; j < exp->N_Q_MATRICES; ++j) {
                fisher_matrix[j + i * ndim + (ndim + 1) * offset] +=
                    exp->fisher_matrix[j + i * exp->N_Q_MATRICES];
            } 
        }
    }

    exposures.clear();
}




