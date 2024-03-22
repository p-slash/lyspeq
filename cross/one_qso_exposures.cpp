#include <algorithm>
#include <stdexcept>
#include <functional>

#include "cross/one_qso_exposures.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include "mathtools/smoother.hpp"


double *_vmatrix, *_zmatrix;
int N1, N2;
const qio::QSOFile *q1, *q2;

double _calculateOverlapRatio() {
    double a1 = q1->wave()[0], a2 = q1->wave()[q1->size() - 1],
           b1 = q2->wave()[0], b2 = q2->wave()[q2->size() - 1];

    double w1 = std::max(a1, b1), w2 = std::min(a2, b2);
    return (w2 - w1) / sqrt((a2 - a1) * (b2 - b1));
}

void _setVMatrix(double *m) {
    DEBUG_LOG("Setting v matrix\n");

    int N1 = q1->size(), N2 = q2->size();
    double *l1 = q1->wave(), *l2 = q2->wave();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < N2; ++j)
            m[j + i * N2] = SPEED_OF_LIGHT * fabs(log(l2[j] / l1[i]));
}


void _setZMatrix(double *m) {
    DEBUG_LOG("Setting z matrix\n");

    double *l1 = q1->wave(), *l2 = q2->wave();

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < N2; ++j)
            m[j + i * N2] = sqrt(l2[j] * l1[i]) - 1.;
}


void _doubleRmatSandwich(double *m) {
    mxhelp::DiaMatrix *r1 = q1->Rmat->getDiaMatrixPointer(),
                      *r2 = q1->Rmat->getDiaMatrixPointer();
    double *buf = glmemory::getSandwichBuffer(N1 * N2);
    r1->multiplyLeft(m, buf, N2);
    r2->multiplyRightT(buf, m, N1);
}


void _setQiMatrix(double *m, int fi_kz)
{
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


void _setFiducialSignalMatrix(double *sm)
{
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


OneQsoExposures::OneQsoExposures(const std::string &f_qso)
{
    try {
        auto qFile = _readQsoFile(f_qso);

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
}


void OneQsoExposures::addExposures(OneQsoExposures *other) {
    exposures.reserve(exposures.size() + other->exposures.size());
    std::move(std::begin(other->exposures),
              std::end(other->exposures),
              std::back_inserter(exposures));
    other->exposures.clear();
}


void OneQsoExposures::oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum
) {
    // Get inverse cov and weighted delta for all exposures
    for (auto &exp : exposures) {
        try {
            exp->initMatrices();
            exp->setCovarianceMatrix();
            exp->invertCovarianceMatrix();
            exp->weightDataVector();
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

    if (exposures.size() < 2) {
        LOG::LOGGER.ERR(
            "OneQsoExposures::oneQSOiteration::Not enough valid exposures"
            " for TARGETID %ld.\n", targetid);
        return;
    }

    _vmatrix = glmemory::temp_matrices[0].get(),
    _zmatrix = glmemory::temp_matrices[1].get();
    // For each combo calculate derivatives
    for (auto exp1 = exposures.cbegin(); exp1 != exposures.cend() - 1; ++exp1) {
        for (auto exp2 = exp1 + 1; exp2 != exposures.cend(); ++exp2) {
            q1 = (*exp1)->qFile.get();
            N1 = q1->size();
            q2 = (*exp2)->qFile.get();
            N2 = q2->size();

            if ((*exp1)->getExpId() == (*exp2)->getExpId())
                continue;

            double r_overlap = _calculateOverlapRatio();

            if (r_overlap < 0.5)
                continue;

            // Contruct VZ matrices
            _setVMatrix(_vmatrix);
            _setZMatrix(_zmatrix);

            // Construct derivative
            DEBUG_LOG("Setting qi matrices\n");

            // if ((*exp1)->stored_ikz_qi.size() <= (*exp2)->stored_ikz_qi.size())
            //     auto &stored_ikz_qi = (*exp1)->stored_ikz_qi;
            // else
            //     auto &stored_ikz_qi = (*exp2)->stored_ikz_qi;

            for (auto iqt = (*exp1)->stored_ikz_qi.begin(); iqt != (*exp1)->stored_ikz_qi.end(); ++iqt)
                _setQiMatrix(iqt->second, iqt->first + (*exp1)->fisher_index_start);

            // Construct fiducial
            _setFiducialSignalMatrix(glmemory::stored_sfid.get());

            // Calculate

        }
    }
}




