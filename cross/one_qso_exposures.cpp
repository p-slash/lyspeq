#include <algorithm>
#include <stdexcept>
#include <functional>

#include "cross/one_qso_exposures.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include "mathtools/smoother.hpp"


int N_Q_MATRICES, KBIN_UPP, fisher_index_start;
double *_vmatrix, *_zmatrix;
std::vector<std::pair<int, double*>> stored_ikz_qi, stored_weighted_qi;

int N1, N2;
const qio::QSOFile *q1, *q2;


void _setNQandFisherIndex() {
    double a1 = q1->wave()[0], a2 = q1->wave()[N1 - 1],
           b1 = q2->wave()[0], b2 = q2->wave()[N2 - 1];

    double zmin = sqrt(a1 * b1) - 1., zmax = sqrt(a2 * b2) - 1.;
    int ZBIN_LOW = bins::findRedshiftBin(zmin),
        ZBIN_UPP = bins::findRedshiftBin(zmax);

    N_Q_MATRICES = ZBIN_UPP - ZBIN_LOW + 1;
    fisher_index_start = bins::getFisherMatrixIndex(0, ZBIN_LOW);

    if (bins::Z_BINNING_METHOD == bins::TriangleBinningMethod) {
        // If we need to distribute low end to a lefter bin
        if ((zmin < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0))
        {
            ++N_Q_MATRICES;
            fisher_index_start -= bins::NUMBER_OF_K_BANDS;
        }
        // If we need to distribute high end to righter bin
        if ((bins::ZBIN_CENTERS[ZBIN_UPP] < zmax) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
            ++N_Q_MATRICES;
    }

    N_Q_MATRICES *= bins::NUMBER_OF_K_BANDS;
    DEBUG_LOG(
        "OneQsoExposures::_setNQandFisherIndex: nq: %d, fi: %d\n",
        N_Q_MATRICES, fisher_index_start);
}


typedef std::vector<std::unique_ptr<Exposure>>::const_iterator vecExpIt;
void _setInternalVariablesForTwoExposures(vecExpIt exp1, vecExpIt exp2) {
    q1 = (*exp1)->qFile.get();
    N1 = q1->size();
    q2 = (*exp2)->qFile.get();
    N2 = q2->size();
    KBIN_UPP = std::min((*exp1)->KBIN_UPP, (*exp2)->KBIN_UPP);

    _setNQandFisherIndex();
}


double _calculateOverlapRatio(const qio::QSOFile *qa, const qio::QSOFile *qb) {
    double a1 = qa->wave()[0], a2 = qa->wave()[qa->size() - 1],
           b1 = qb->wave()[0], b2 = qb->wave()[qb->size() - 1];

    double w1 = std::max(a1, b1), w2 = std::min(a2, b2);
    return (w2 - w1) / sqrt((a2 - a1) * (b2 - b1));
}


void _setVMatrix(double *m) {
    DEBUG_LOG("OneQsoExposures::_setVMatrix\n");

    int N1 = q1->size(), N2 = q2->size();
    double *l1 = q1->wave(), *l2 = q2->wave();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < N2; ++j)
            m[j + i * N2] = SPEED_OF_LIGHT * fabs(log(l2[j] / l1[i]));
}


void _setZMatrix(double *m) {
    DEBUG_LOG("OneQsoExposures::_setZMatrix\n");

    double *l1 = q1->wave(), *l2 = q2->wave();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < N2; ++j)
            m[j + i * N2] = sqrt(l2[j] * l1[i]) - 1.;
}


void _setStoredIkzQiVector() {
    DEBUG_LOG("OneQsoExposures::_setStoredIkzQiVector\n");
    stored_ikz_qi.clear();
    stored_weighted_qi.clear();
    stored_ikz_qi.reserve(N_Q_MATRICES);
    stored_weighted_qi.reserve(N_Q_MATRICES);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
        int kn, zm;
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        if (kn < KBIN_UPP) {
            stored_ikz_qi.push_back(std::make_pair(i_kz, nullptr));
            stored_weighted_qi.push_back(std::make_pair(i_kz, nullptr));
        }
    }

    int delta_size = 2 * stored_ikz_qi.size() - glmemory::stored_ikz_qi.size();
    if (delta_size > 0) {
        LOG::LOGGER.ERR("Need more memory to store glmemory::stored_ikz_qi.\n");
        glmemory::updateMemUsed(
            process::getMemoryMB(glmemory::max_size_2 * delta_size));

        for (int i = 0; i < delta_size; ++i)
            glmemory::stored_ikz_qi.push_back(
                std::make_unique<double[]>(glmemory::max_size_2));
    }

    int ndim = stored_ikz_qi.size();
    for (int i = 0; i < ndim; ++i) {
        stored_ikz_qi[i].second = glmemory::stored_ikz_qi[i].get();
        stored_weighted_qi[i].second = glmemory::stored_ikz_qi[i + ndim].get();
    }
}


void _doubleRmatSandwich(double *m) {
    mxhelp::DiaMatrix *r1 = q1->Rmat->getDiaMatrixPointer(),
                      *r2 = q2->Rmat->getDiaMatrixPointer();
    double *buf = glmemory::getSandwichBuffer(N1 * N2);
    r1->multiplyLeft(m, buf, N2);
    r2->multiplyRightT(buf, m, N1);
}


void _setQiMatrix(double *m, int fi_kz) {
    ++mytime::number_of_times_called_setq;
    double t = mytime::timer.getTime(), t_interp;
    int kn, zm;

    bins::getFisherMatrixBinNoFromIndex(fi_kz, kn, zm);
    bins::setRedshiftBinningFunction(zm);

    std::function<double(double)> eval_deriv_kn;

    if (specifics::USE_RESOLUTION_MATRIX) {
        double kc = bins::KBAND_CENTERS[kn],
               dk = bins::KBAND_EDGES[kn + 1] - bins::KBAND_EDGES[kn];

        eval_deriv_kn = [kc, dk](double v) {
            double x = dk * v / 2;
            return (dk / MY_PI) * cos(kc * v) * sin(x) / (x + DOUBLE_EPSILON);
        };
    } else {
        eval_deriv_kn = [idkn = glmemory::interp_derivative_matrix[kn]](double v) {
            return idkn->evaluate(v);
        };
    }

    std::fill_n(m, N1 * N2, 0);

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < N1; ++i) {
        int idx = i * N2, l1, u1;

        bins::redshiftBinningFunction(
            _zmatrix + idx, N2, zm,
            m + idx, l1, u1);

        #pragma omp simd
        for (int j = l1; j < u1; ++j)
            m[j + idx] *= eval_deriv_kn(_vmatrix[j + idx]);
    }

    t_interp = mytime::timer.getTime() - t;

    if (specifics::USE_RESOLUTION_MATRIX)
        _doubleRmatSandwich(m);

    t = mytime::timer.getTime() - t; 

    mytime::time_spent_set_qs += t;
    mytime::time_spent_on_q_interp += t_interp;
    mytime::time_spent_on_q_copy += t - t_interp;
}


void _setFiducialSignalMatrix(double *sm) {
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::timer.getTime();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            int idx = j + i * N2;
            sm[idx] = glmemory::interp2d_signal_matrix->evaluate(
                _zmatrix[idx], _vmatrix[idx]);
        }
    }

    if (specifics::USE_RESOLUTION_MATRIX)
        _doubleRmatSandwich(sm);

    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}


OneQsoExposures::OneQsoExposures(const std::string &f_qso) : OneQSOEstimate() {
    try {
        std::unique_ptr<qio::QSOFile> qFile = _readQsoFile(f_qso);

        targetid = qFile->id;
        exposures.reserve(30);
        std::vector<int> indices = OneQSOEstimate::decideIndices(qFile->size());
        int nchunks = indices.size() - 1;

        for (int nc = 0; nc < nchunks; ++nc) {
            try {
                auto _chunk = std::make_unique<Exposure>(
                    *qFile, indices[nc], indices[nc + 1]);
                exposures.push_back(std::move(_chunk));
            } catch (std::exception& e) {
                LOG::LOGGER.ERR(
                    "OneQsoExposures::appendChunkedQsoFile::%s "
                    "Skipping chunk %d/%d of %s.\n",
                    e.what(), nc, nchunks, qFile->fname.c_str());
            }
        }
    } catch (std::exception &e) {
        LOG::LOGGER.ERR("%s in %s.\n", e.what(), f_qso.c_str());
        return;
    }

    istart = bins::TOTAL_KZ_BINS;
    ndim = 0;
}


void OneQsoExposures::addExposures(OneQsoExposures *other) {
    exposures.reserve(exposures.size() + other->exposures.size());
    std::move(std::begin(other->exposures), std::end(other->exposures),
              std::back_inserter(exposures));
    other->exposures.clear();
}


void OneQsoExposures::setAllocPowerSpMemory() {
    DEBUG_LOG("OneQsoExposures::setAllocPowerSpMemory::TARGETID %ld\n", targetid);
    // decide max_ndim, istart, etc
    int fisher_index_end = 0;
    istart = bins::TOTAL_KZ_BINS;
    for (vecExpIt exp1 = exposures.cbegin(); exp1 != exposures.cend() - 1; ++exp1) {
        for (vecExpIt exp2 = exp1 + 1; exp2 != exposures.cend(); ++exp2) {
            _setInternalVariablesForTwoExposures(exp1, exp2);

            istart = std::min(istart, fisher_index_start);
            fisher_index_end = std::max(
                fisher_index_end, fisher_index_start + N_Q_MATRICES);
        }
    }

    ndim = fisher_index_end - istart;

    fisher_matrix = std::make_unique<double[]>(ndim * ndim);
    theta_vector = std::make_unique<double[]>(ndim);
    for (int i = 0; i < 3; ++i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(ndim));
}

bool skipCombo(vecExpIt exp1, vecExpIt exp2, double overlap_cut=0.5) {
    bool skip = (
        (*exp1)->getExpId() == (*exp2)->getExpId()
        || (*exp1)->getNight() == (*exp2)->getNight()
        || (_calculateOverlapRatio((*exp1)->qFile.get(), (*exp2)->qFile.get())
            < overlap_cut)
    );
    
    return skip;
}

int OneQsoExposures::countExposureCombos() {
    int numcombos = 0;

    for (vecExpIt exp1 = exposures.cbegin(); exp1 != exposures.cend() - 1; ++exp1) {
        for (vecExpIt exp2 = exp1 + 1; exp2 != exposures.cend(); ++exp2) {
            if (skipCombo(exp1, exp2))
                continue;

            ++numcombos;
        }
    }

    return numcombos;
}

void OneQsoExposures::xQmlEstimate() {
    _vmatrix = glmemory::temp_matrices[0].get();
    _zmatrix = glmemory::temp_matrices[1].get();
    double *dk0 = dbt_estimate_before_fisher_vector[0].get(),
           *tk0 = dbt_estimate_before_fisher_vector[2].get();

    // For each combo calculate derivatives
    for (vecExpIt exp1 = exposures.cbegin(); exp1 != exposures.cend() - 1; ++exp1) {
        for (vecExpIt exp2 = exp1 + 1; exp2 != exposures.cend(); ++exp2) {
            if (skipCombo(exp1, exp2))
                continue;

            _setInternalVariablesForTwoExposures(exp1, exp2);

            // Contruct VZ matrices
            _setVMatrix(_vmatrix);
            _setZMatrix(_zmatrix);
            _setStoredIkzQiVector();

            // Construct derivative
            DEBUG_LOG("OneQsoExposures::xQmlEstimate::_setQiMatrix\n");
            for (auto iqt = stored_ikz_qi.begin(); iqt != stored_ikz_qi.end(); ++iqt)
                _setQiMatrix(iqt->second, iqt->first + fisher_index_start);

            // Construct fiducial
            _setFiducialSignalMatrix(glmemory::stored_sfid.get());

            // Calculate
            int diff_idx = fisher_index_start - istart;

            double t = mytime::timer.getTime();
            for (int i = 0; i != stored_ikz_qi.size(); ++i) {
                auto &[i_kz, Q_ikz_matrix] = stored_ikz_qi[i];
                dk0[i_kz + diff_idx] = mxhelp::my_cblas_dgemvdot(
                    (*exp1)->getWeightedData(), N1,
                    (*exp2)->getWeightedData(), N2,
                    Q_ikz_matrix, glmemory::temp_vector.get());

                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N1, N2, N1, 1., (*exp1)->inverse_covariance_matrix, N1,
                    Q_ikz_matrix, N2, 0,
                    glmemory::temp_matrices[0].get(), N2);

                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N1, N2, N2, 1., glmemory::temp_matrices[0].get(), N2,
                    (*exp2)->inverse_covariance_matrix, N2, 0.,
                    stored_weighted_qi[i].second, N2);
            }

            mytime::time_spent_set_modqs += mytime::timer.getTime() - t;

            int size2 = N1 * N2;
            #pragma omp parallel for
            for (const auto &[i_kz, Q_ikz_matrix] : stored_weighted_qi)
                tk0[i_kz + diff_idx] = cblas_ddot(
                    size2, Q_ikz_matrix, 1, glmemory::stored_sfid.get(), 1);

            if (specifics::USE_PRECOMPUTED_FISHER)
                continue;

            t = mytime::timer.getTime();

            #pragma omp parallel for
            for (auto iqt = stored_weighted_qi.cbegin(); iqt != stored_weighted_qi.cend(); ++iqt) {
                int idx_fji_0 = ndim * (iqt->first + diff_idx) + diff_idx;

                for (auto jqt = stored_ikz_qi.cbegin(); jqt != stored_ikz_qi.cend(); ++jqt) {
                    #ifdef FISHER_OPTIMIZATION
                    int diff_ji = abs(jqt->first - iqt->first);
                    if ((diff_ji > 5) && (abs(diff_ji - bins::NUMBER_OF_K_BANDS) > 2))
                        continue;
                    #endif

                    fisher_matrix[jqt->first + idx_fji_0] = 
                        cblas_ddot(size2, iqt->second, 1, jqt->second, 1);
                }
            }

            mytime::time_spent_set_fisher += mytime::timer.getTime() - t;;
        }
    }

    for (auto &expo : exposures)
        expo->deallocMatrices();

    std::copy_n(dk0, ndim, theta_vector.get());
    cblas_daxpy(ndim, 1, tk0, 1, theta_vector.get(), 1);
}


int OneQsoExposures::oneQSOiteration(
        std::vector<std::unique_ptr<double[]>> &dt_sum_vector,
        double *fisher_sum
) {
    DEBUG_LOG("\n\nOneQsoExposures::oneQSOiteration::TARGETID %ld\n", targetid);

    // Get inverse cov and weighted delta for all exposures
    for (auto &expo : exposures) {
        try {
            expo->initMatrices();
            expo->setCovarianceMatrix();
            expo->invertCovarianceMatrix();
            expo->weightDataVector();
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR(
                "OneQsoExposures::oneQSOiteration::%s Skipping %s.\n",
                e.what(), expo->qFile->fname.c_str());
            expo.reset();
        }
    }

    exposures.erase(std::remove_if(
        exposures.begin(), exposures.end(), [](const auto &x) { return !x; }),
        exposures.end()
    );

    if (exposures.size() < 2) {
        LOG::LOGGER.ERR(
            "OneQsoExposures::oneQSOiteration::Not enough valid exposures"
            " for TARGETID %ld.\n", targetid);
        return 0;
    }

    int numcombos = countExposureCombos();

    if (numcombos == 0) {
        LOG::LOGGER.ERR(
            "OneQsoExposures::oneQSOiteration::Not enough valid exposures combos"
            " for TARGETID %ld.\n", targetid);
        return 0;
    }

    setAllocPowerSpMemory();
    xQmlEstimate();

    double *outfisher = fisher_sum + (bins::TOTAL_KZ_BINS + 1) * istart;

    for (int i = 0; i < ndim; ++i)
        for (int j = 0; j < ndim; ++j)
            outfisher[j + i * bins::TOTAL_KZ_BINS] += fisher_matrix[j + i * ndim];

    for (int i = 0; i < 3; ++i)
        cblas_daxpy(
            ndim, 1, dbt_estimate_before_fisher_vector[i].get(), 1,
            dt_sum_vector[i].get() + istart, 1);

    return numcombos;
}


std::unique_ptr<OneQSOEstimate> OneQsoExposures::move2OneQSOEstimate() {
    auto qso = std::make_unique<OneQSOEstimate>(true);
    qso->istart = istart;
    qso->ndim = ndim;
    qso->theta_vector = std::move(theta_vector);
    qso->fisher_matrix = std::move(fisher_matrix);
    return qso;
}



