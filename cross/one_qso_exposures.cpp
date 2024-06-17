#include <algorithm>
#include <stdexcept>
#include <functional>
#ifdef DEBUG
#include <cassert>
#endif

#include "cross/one_qso_exposures.hpp"
#include "core/global_numbers.hpp"

#include "mathtools/stats.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

// typedef std::vector<std::unique_ptr<Exposure>>::const_iterator vecExpIt;

namespace specifics {
    bool X_NIGHT = true, X_FIBER = false, X_PETAL = false;
    double X_WAVE_OVERLAP_RATIO = 0.6;
}


int N_Q_MATRICES, KBIN_UPP, fisher_index_start;
double *_vmatrix, *_zmatrix;
std::vector<std::tuple<int, double*, double*>> stored_ikz_qi_qwi;

int N1, N2, CROSS_SIZE_2;
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


void _setInternalVariablesForTwoExposures(
        const Exposure* expo1, const Exposure* expo2
) {
    q1 = expo1->qFile.get();
    N1 = q1->size();
    q2 = expo2->qFile.get();
    N2 = q2->size();
    KBIN_UPP = std::min(expo1->KBIN_UPP, expo2->KBIN_UPP);
    CROSS_SIZE_2 = N1 * N2;
    _setNQandFisherIndex();
}


double _calculateOverlapRatio(const qio::QSOFile *qa, const qio::QSOFile *qb) {
    double a1 = qa->wave()[0], a2 = qa->wave()[qa->size() - 1],
           b1 = qb->wave()[0], b2 = qb->wave()[qb->size() - 1];

    double w1 = std::max(a1, b1), w2 = std::min(a2, b2);
    return (w2 - w1) / (std::max(a2, b2) - std::min(a1, b1));
}


void _setVZMatrix(double *vm, double *zm) {
    DEBUG_LOG("OneQsoExposures::_setVZMatrix\n");

    double *l1 = q1->wave(), *l2 = q2->wave();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < N2; ++j) {
            vm[j + i * N2] = SPEED_OF_LIGHT * fabs(log(l2[j] / l1[i]));
            zm[j + i * N2] = sqrt(l2[j] * l1[i]) - 1.;
        }
}


void _setStoredIkzQiVector() {
    DEBUG_LOG("OneQsoExposures::_setStoredIkzQiVector\n");
    stored_ikz_qi_qwi.clear();
    stored_ikz_qi_qwi.reserve(N_Q_MATRICES);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
        int kn, zm;
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        if (kn < KBIN_UPP)
            stored_ikz_qi_qwi.push_back(std::make_tuple(i_kz, nullptr, nullptr));
    }

    int delta_size = 2 * stored_ikz_qi_qwi.size() - glmemory::stored_ikz_qi.size();
    if (delta_size > 0) {
        LOG::LOGGER.ERR("Need more memory to store glmemory::stored_ikz_qi.\n");
        glmemory::updateMemUsed(
            process::getMemoryMB(glmemory::max_size_2 * delta_size));

        for (int i = 0; i < delta_size; ++i)
            glmemory::stored_ikz_qi.push_back(
                std::make_unique<double[]>(glmemory::max_size_2));
    }

    for (size_t i = 0; i < stored_ikz_qi_qwi.size(); ++i) {
        std::get<1>(stored_ikz_qi_qwi[i]) = glmemory::stored_ikz_qi[2 * i].get();
        std::get<2>(stored_ikz_qi_qwi[i]) = glmemory::stored_ikz_qi[2 * i + 1].get();
    }
}


void _doubleRmatSandwich(double *m) {
    mxhelp::DiaMatrix *r1 = q1->Rmat->getDiaMatrixPointer(),
                      *r2 = q2->Rmat->getDiaMatrixPointer();
    double *buf = glmemory::getSandwichBuffer(CROSS_SIZE_2);
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

    std::fill_n(m, CROSS_SIZE_2, 0);

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
        z_qso = qFile->z_qso;
        ra = qFile->ra;
        dec = qFile->dec;

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
                    "Skipping chunk %d/%d of TARGETID %ld in %s.\n",
                    e.what(), nc, nchunks, targetid, qFile->fname.c_str());
            }
        }
    } catch (std::exception &e) {
        LOG::LOGGER.ERR(
            "%s in TARGETID %ld in %s.\n", e.what(), targetid, f_qso.c_str());
        return;
    }

    istart = bins::TOTAL_KZ_BINS;
    ndim = 0;
    if (!exposures.empty())
        unique_expid_set.insert(exposures[0]->getExpId());
}


void OneQsoExposures::addExposures(OneQsoExposures *other) {
    exposures.reserve(exposures.size() + other->exposures.size());
    std::move(std::begin(other->exposures), std::end(other->exposures),
              std::back_inserter(exposures));
    other->exposures.clear();
    unique_expid_set.merge(std::move(other->unique_expid_set));
}


void OneQsoExposures::setAllocPowerSpMemory() {
    DEBUG_LOG("OneQsoExposures::setAllocPowerSpMemory::TARGETID %ld\n", targetid);
    // decide max_ndim, istart, etc
    int fisher_index_end = 0;
    istart = bins::TOTAL_KZ_BINS;
    for (const auto &[expo1, expo2] : exposure_combos) {
        _setInternalVariablesForTwoExposures(expo1, expo2);

        istart = std::min(istart, fisher_index_start);
        fisher_index_end = std::max(
            fisher_index_end, fisher_index_start + N_Q_MATRICES);
    }

    ndim = fisher_index_end - istart;

    fisher_matrix = std::make_unique<double[]>(ndim * ndim);
    theta_vector = std::make_unique<double[]>(ndim);
    for (int i = 0; i < 3; ++i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(ndim));
}

bool skipCombo(const Exposure *expo1, const Exposure *expo2) {
    bool skip = expo1->getExpId() == expo2->getExpId();
    skip |= specifics::X_NIGHT && (expo1->getNight() == expo2->getNight());
    skip |= specifics::X_FIBER && (expo1->getFiber() == expo2->getFiber());
    skip |= specifics::X_PETAL && (expo1->getPetal() == expo2->getPetal());
    skip |= (
        _calculateOverlapRatio(expo1->qFile.get(), expo2->qFile.get())
        < specifics::X_WAVE_OVERLAP_RATIO);
    
    return skip;
}

int OneQsoExposures::countExposureCombos() {
    exposure_combos.clear();
    for (auto expo1 = exposures.cbegin(); expo1 != exposures.cend() - 1; ++expo1) {
        for (auto expo2 = expo1 + 1; expo2 != exposures.cend(); ++expo2) {
            const Exposure *e1 = (*expo1).get(), *e2 = (*expo2).get();
            if (skipCombo(e1, e2))
                continue;

            exposure_combos.push_back(std::make_pair(e1, e2));
        }
    }

    return exposure_combos.size();
}

void OneQsoExposures::xQmlEstimate() {
    _vmatrix = glmemory::temp_matrices[0].get();
    _zmatrix = glmemory::temp_matrices[1].get();
    double *dk0 = dbt_estimate_before_fisher_vector[0].get(),
           *tk0 = dbt_estimate_before_fisher_vector[2].get();

    // For each combo calculate derivatives
    for (const auto &[expo1, expo2] : exposure_combos) {
        _setInternalVariablesForTwoExposures(expo1, expo2);
        #ifdef DEBUG
        assert(!skipCombo(expo1, expo2));
        #endif

        // Contruct VZ matrices
        _setVZMatrix(_vmatrix, _zmatrix);
        _setStoredIkzQiVector();

        // Construct derivative
        DEBUG_LOG("OneQsoExposures::xQmlEstimate::_setQiMatrix\n");
        for (auto &[i_kz, qi, qwi] : stored_ikz_qi_qwi)
            _setQiMatrix(qi, i_kz + fisher_index_start);

        // Calculate
        int diff_idx = fisher_index_start - istart;

        double t = mytime::timer.getTime();
        for (auto &[i_kz, qi, qwi] : stored_ikz_qi_qwi) {
            dk0[i_kz + diff_idx] += mxhelp::my_cblas_dgemvdot(
                expo1->getWeightedData(), N1,
                expo2->getWeightedData(), N2,
                qi, glmemory::temp_vector.get());

            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N1, N2, N1, 1., expo1->getInverseCov(), N1,
                qi, N2, 0,
                glmemory::temp_matrices[0].get(), N2);

            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N1, N2, N2, 1., glmemory::temp_matrices[0].get(), N2,
                expo2->getInverseCov(), N2, 0.,
                qwi, N2);
        }

        mytime::time_spent_set_modqs += mytime::timer.getTime() - t;

        if (!specifics::TURN_OFF_SFID) {
            _setFiducialSignalMatrix(glmemory::stored_sfid.get());

            #pragma omp parallel for
            for (const auto &[i_kz, qi, qwi] : stored_ikz_qi_qwi)
                tk0[i_kz + diff_idx] += cblas_ddot(
                    CROSS_SIZE_2, qwi, 1, glmemory::stored_sfid.get(), 1);
        }

        if (specifics::USE_PRECOMPUTED_FISHER)
            continue;

        t = mytime::timer.getTime();

        diff_idx *= ndim + 1;
        // I think this is still symmetric
        #pragma omp parallel for schedule(static, 1)
        for (auto iqt = stored_ikz_qi_qwi.cbegin(); iqt != stored_ikz_qi_qwi.cend(); ++iqt) {
            int i_kz = 0, j_kz = 0;
            double *qwi = nullptr, *qjT = nullptr;
            std::tie(i_kz, std::ignore, qwi) = *iqt;

            for (auto jqt = iqt; jqt != stored_ikz_qi_qwi.cend(); ++jqt) {
                std::tie(j_kz, qjT, std::ignore) = *jqt;

                #ifdef FISHER_OPTIMIZATION
                int diff_ji = abs(j_kz - i_kz);
                if ((diff_ji > 5) && (abs(diff_ji - bins::NUMBER_OF_K_BANDS) > 2))
                    continue;
                #endif

                fisher_matrix[j_kz + ndim * i_kz + diff_idx] += 
                    cblas_ddot(CROSS_SIZE_2, qwi, 1, qjT, 1);
            }
        }

        mytime::time_spent_set_fisher += mytime::timer.getTime() - t;;
    }

    for (auto &expo : exposures)
        expo->deallocMatrices();

    cblas_dscal(ndim, 2.0, dk0, 1);
    cblas_dscal(ndim, 2.0, tk0, 1);
    for (int i = 0; i < ndim; ++i)
        theta_vector[i] = dk0[i] - tk0[i];
}


bool OneQsoExposures::isAnOutlier() {
    double *v = dbt_estimate_before_fisher_vector[1].get(),
           mean_dev = 0, median_dev = 0, s = 0;
    int j = 0;

    for (int i = 0; i < ndim; ++i) {
        // Covariance of theta is 4xFisher
        s = fisher_matrix[i * (ndim + 1)];
        if (s == 0)
            continue;

        s = 2 * sqrt(s);

        v[j] = theta_vector[i] / s;
        v[j] *= v[j];
        mean_dev += v[j];
        ++j;
    }

    mean_dev /= j;
    median_dev = stats::medianOfUnsortedVector(v, j);

    std::fill_n(v, ndim, 0);
    bool is_an_outlier = (median_dev > 3.5);

    if (!is_an_outlier)
        return false;

    LOG::LOGGER.ERR(
        "OneQsoExposures::isAnOutlier::Outlier P1D estimate in "
        "TARGETID %ld: Mean dev: %.1f, Median dev: %.1f.\n",
        targetid, mean_dev, median_dev);

    return true;
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

    setAllocPowerSpMemory();
    xQmlEstimate();

    if (isAnOutlier())
        return 0;

    double *outfisher = fisher_sum + (bins::TOTAL_KZ_BINS + 1) * istart;

    for (int i = 0; i < ndim; ++i)
        for (int j = i; j < ndim; ++j)
            outfisher[j + i * bins::TOTAL_KZ_BINS] += fisher_matrix[j + i * ndim];

    for (int i = 0; i < 3; ++i)
        cblas_daxpy(
            ndim, 1, dbt_estimate_before_fisher_vector[i].get(), 1,
            dt_sum_vector[i].get() + istart, 1);

    return numcombos;
}
