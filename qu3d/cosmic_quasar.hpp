#ifndef COSMIC_QUASAR_H
#define COSMIC_QUASAR_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "mathtools/matrix_helper.hpp"
#include "core/omp_manager.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"
#include "mathtools/real_field_3d.hpp"

#include "qu3d/cosmology_3d.hpp"
#include "qu3d/contmarg_file.hpp"

// The shift of RA to have continous regions in the sky (for eBOSS)
constexpr double ra_shift = 1.0;

namespace specifics {
    extern double MIN_RA, MAX_RA, MIN_DEC, MAX_DEC;
    static int DOWNSAMPLE_FACTOR;
}

template <class T>
struct CompareCosmicQuasarPtr {
    bool operator()(const T *lhs, const T *rhs) const {
        return lhs->qFile->id < rhs->qFile->id;
    }
};

class CosmicQuasar {
private:
    double _quasar_dist;
public:
    std::unique_ptr<qio::QSOFile> qFile;
    int N, fidx;
    /* z1: 1 + z */
    /* Cov . in = out, out should be compared to truth for inversion. */
    double *z1, *isig, angles[3], *in, *out, *truth, *in_isig, *sc_eta;
    double cos_dec, sin_dec;
    std::unique_ptr<float[]> r;
    std::unique_ptr<double[]> y, Cy, residual, search, y_isig,
                              sod_cinv_eta, _z1_mem;

    std::set<size_t> grid_indices;
    std::set<const CosmicQuasar*, CompareCosmicQuasarPtr<CosmicQuasar>> neighbors;
    size_t min_x_idx;

    CosmicQuasar(const qio::PiccaFile *pf, int hdunum) {
        qFile = std::make_unique<qio::QSOFile>(pf, hdunum);
        N = 0;

        qFile->readParameters();

        if (qFile->snr < specifics::MIN_SNR_CUT) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::Low SNR in TARGETID "
                    << qFile->id;
            throw std::runtime_error(err_msg.str());
        }

        angles[0] = qFile->ra + ra_shift;
        if (angles[0] >= 2 * MY_PI)
            angles[0] -= 2 * MY_PI;

        angles[1] = qFile->dec;
        angles[2] = 1;

        if ((angles[0] > specifics::MAX_RA) || (angles[0] < specifics::MIN_RA)) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::Outside RA range in TARGETID "
                    << qFile->id;
            throw std::runtime_error(err_msg.str());
        }

        if ((angles[1] > specifics::MAX_DEC) || (angles[1] < specifics::MIN_DEC)) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::Outside DEC range in TARGETID "
                    << qFile->id;
            throw std::runtime_error(err_msg.str());
        }

        qFile->readData();
        int num_outliers = qFile->maskOutliers();
        if (num_outliers > 0)
            LOG::LOGGER.STD(
                "WARNING::CosmicQuasar::CosmicQuasar::"
                "Found %d outlier pixels in %d.\n",
                num_outliers, qFile->id);

        qFile->cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        if (qFile->realSize() < 20) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::No pixels in TARGETID "
                    << qFile->id;
            throw std::runtime_error(err_msg.str());
        }

        // Convert wave to 1 + z
        std::for_each(
            qFile->wave(), qFile->wave() + qFile->size(), [](double &ld) {
                ld = ld / LYA_REST;
            }
        );

        if (specifics::DOWNSAMPLE_FACTOR > 1)
            qFile->downsample(specifics::DOWNSAMPLE_FACTOR);

        N = qFile->size();
        isig = qFile->ivar();

        r = std::make_unique<float[]>(3 * N);
        y = std::make_unique<double[]>(N);
        y_isig = std::make_unique<double[]>(N);
        Cy = std::make_unique<double[]>(N);
        residual = std::make_unique<double[]>(N);
        search = std::make_unique<double[]>(N);
        sod_cinv_eta = std::make_unique<double[]>(N);
        _z1_mem = std::make_unique<double[]>(N);
        z1 = _z1_mem.get();

        // Convert to inverse sigma and weight deltas
        for (int i = 0; i < N; ++i) {
            isig[i] = sqrt(isig[i]);
            qFile->delta()[i] *= isig[i];
            z1[i] = qFile->wave()[i];
        }

        /* Will be reset in _shiftByMedianDec in optimal_qu3d.cpp */
        cos_dec = cos(angles[1]);
        sin_dec = sin(angles[1]);

        in = y.get();
        in_isig = y_isig.get();
        truth = qFile->delta();
        out = Cy.get();
        sc_eta = sod_cinv_eta.get();
    }
    CosmicQuasar(CosmicQuasar &&rhs) = delete;
    CosmicQuasar(const CosmicQuasar &rhs) = delete;

    void replaceTruthWithGaussMocks(RealField3D &m, MyRNG &rng_) {
        for (int i = 0; i < N; ++i) {
            if (isig[i] != 0) {
                truth[i] = z1[i] * m.interpolate(r.get() + 3 * i)
                           + rng_.normal() / isig[i];
            }
            else
                truth[i] = 0;

            // transform isig to ivar for project and write
            isig[i] *= isig[i];
        }
    }
    void project(double varlss, int order) {
        /* Assumes isig is ivar. */
        assert(order < 2 && order >= 0);

        double sum_weights = 0.0, mean_delta = 0.0;
        auto weights = std::make_unique<double[]>(N);
        auto log_lambda = std::make_unique<double[]>(N);

        for (int i = 0; i < N; ++i) {
            weights[i] = isig[i] / (1.0 + isig[i] * varlss * z1[i] * z1[i]);
            sum_weights += weights[i];
            mean_delta += truth[i] * weights[i];
        }

        mean_delta /= sum_weights;
        if (order == 0) {
            for (int i = 0; i < N; ++i)
                truth[i] -= mean_delta;
            return;
        }

        double mean_log_lambda = 0.0;
        for (int i = 0; i < N; ++i) {
            log_lambda[i] = log10(qFile->wave()[i]);
            mean_log_lambda += weights[i] * log_lambda[i];
        }
        mean_log_lambda /= sum_weights;
        for (int i = 0; i < N; ++i)
            log_lambda[i] -= mean_log_lambda;

        double sum_weights_ll = 0.0, mean_delta_ll = 0.0;
        for (int i = 0; i < N; ++i) {
            sum_weights_ll += weights[i] * log_lambda[i] * log_lambda[i];
            mean_delta_ll += weights[i] * log_lambda[i] * truth[i];
        }
        mean_delta_ll /= sum_weights_ll;

        for (int i = 0; i < N; ++i)
            truth[i] -= mean_delta + mean_delta_ll * log_lambda[i];
    }

    void write(fitsfile *fits_file) {
        /* Assumes isig is ivar. */
        int status = 0;
        constexpr int ncolumns = 3;
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wwrite-strings"
        char *column_names[] = {"LAMBDA", "DELTA", "IVAR"};
        char *column_types[] = {"1D", "1D", "1D"};
        char *column_units[] = { "\0", "\0", "\0"};
        #pragma GCC diagnostic pop

        auto a_lambda = std::make_unique<double[]>(N);
        std::transform(
            qFile->wave(), qFile->wave() + N, a_lambda.get(),
            [](double ld) { return ld * LYA_REST; }
        );

        fits_create_tbl(
            fits_file, BINARY_TBL, N, ncolumns, column_names, column_types,
            column_units, std::to_string(qFile->id).c_str(), &status);
        ioh::checkFitsStatus(status);
        // Header keys
        fits_write_key(
            fits_file, TLONG, "TARGETID", &qFile->id, nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "Z", &qFile->z_qso, nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "RA", &qFile->ra, nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "DEC", &qFile->dec, nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "MEANRESO", &qFile->R_kms, nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "MEANSNR", &qFile->snr, nullptr, &status);
        ioh::checkFitsStatus(status);
        // Data
        fits_write_col(
            fits_file, TDOUBLE, 1, 1, 1, N, a_lambda.get(), &status);
        fits_write_col(
            fits_file, TDOUBLE, 2, 1, 1, N, truth, &status);
        fits_write_col(
            fits_file, TDOUBLE, 3, 1, 1, N, isig, &status);
        ioh::checkFitsStatus(status);
    }

    void setComovingDistances(const fidcosmo::FlatLCDM *cosmo, double radial) {
        _quasar_dist = cosmo->getComovingDist(qFile->z_qso + 1.0);
        /* Equirectangular projection */
        for (int i = 0; i < N; ++i) {
            r[0 + 3 * i] = angles[0] * radial;
            r[1 + 3 * i] = angles[1] * radial;
            r[2 + 3 * i] = cosmo->getComovingDist(z1[i]);
        }
    }

    void getSpectroWindowParams(
            const fidcosmo::FlatLCDM *cosmo, double &sigma, double &delta_r
    ) {
        /* Spectrograph window function params. Assumes r is set. */
        double mean_z1 = std::accumulate(z1, z1 + N, 0.0) / N,
               Mpc2kms = cosmo->getHubble(mean_z1) / mean_z1;
        sigma = qFile->R_kms / Mpc2kms;
        delta_r = (r[3 * N - 1] - r[2]) / (N - 1);
    }

    void getSumRadialDistance(
            const fidcosmo::FlatLCDM *cosmo,
            double &sum_chi_weights, double &sum_weights
    ) {
        for (int i = 0; i < N; ++i) {
            double ivar = isig[i] * isig[i];
            sum_chi_weights += cosmo->getComovingDist(z1[i]) * ivar;
            sum_weights += ivar;
        }
    }

    void transformZ1toG(const fidcosmo::ArinyoP3DModel *p3d_model) {
        for (int i = 0; i < N; ++i)
            z1[i] = p3d_model->getRedshiftEvolution(z1[i]);
    }

    void setInIsigNoMarg() {
        for (int i = 0; i < N; ++i)
            in_isig[i] = in[i] * isig[i] * z1[i];
    }

    void setInIsigWithMarg() {
        #ifdef DEBUG_IO
        try {
            assert(fidx == myomp::getThreadNum());
        #endif
        // -- core function
            double *rrmat = GL_RMAT[myomp::getThreadNum()].get();
            ioh::continuumMargFileHandler->read(N, qFile->id, rrmat);
            cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0,
                        rrmat, N, in, 1, 0, in_isig, 1);

            for (int i = 0; i < N; ++i)
                in_isig[i] *= isig[i] * z1[i];
        // --
        #ifdef DEBUG_IO
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR(
                "CosmicQuasar::setInIsigWithMarg::%d-%d::%s\n",
                fidx, myomp::getThreadNum(), e.what());
        }
        #endif
    }

    double* multInputWithMarg(const double *input) {
        /* Output is in_isig */
        #ifdef DEBUG_IO
        try {
            assert(fidx == myomp::getThreadNum());
        #endif
        // -- core function
            double *rrmat = GL_RMAT[myomp::getThreadNum()].get();
            ioh::continuumMargFileHandler->read(N, qFile->id, rrmat);
            cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0,
                        rrmat, N, input, 1, 0, in_isig, 1);
        // --
        #ifdef DEBUG_IO
        }
        catch (std::exception& e) {
            LOG::LOGGER.ERR(
                "CosmicQuasar::multInputWithMarg::%d-%d::%s\n",
                fidx, myomp::getThreadNum(), e.what());
        }
        #endif

        return rrmat;
    }

    void multInvCov(
            const fidcosmo::ArinyoP3DModel *p3d_model,
            const double *input, double *output, bool pp
    ) {
        double varlss = p3d_model->getVarLss();
        auto appDiagonalEst = [this, &varlss](const double *x_, double *y_) {
            for (int i = 0; i < N; ++i) {
                double isigG = isig[i] * z1[i];
                y_[i] = x_[i] / (1.0 + isigG * isigG * varlss);
            }
        };

        if (!pp) {
            appDiagonalEst(input, output);
        }
        else {
            double *ccov = GL_CCOV[myomp::getThreadNum()].get();
            for (int i = 0; i < N; ++i) {
                double isigG = isig[i] * z1[i];

                ccov[i * (N + 1)] = 1.0 + p3d_model->getVarLss() * isigG * isigG;

                for (int j = i + 1; j < N; ++j) {
                    float rz = r[3 * j + 2] - r[3 * i + 2];
                    double isigG_ij = isigG * isig[j] * z1[j];
                    ccov[j + i * N] = p3d_model->evalCorrFunc1dT(rz) * isigG_ij;
                }
            }

            std::copy_n(input, N, output);
            lapack_int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', N, 1,
                                            ccov, N, output, 1);
            if (info != 0) {
                LOG::LOGGER.STD("Error in CosmicQuasar::multInvCov::LAPACKE_dposv");
                appDiagonalEst(input, output);
            }
        }
    }

    void interpMesh2Out(const RealField3D &mesh) {
        for (int i = 0; i < N; ++i)
            out[i] = mesh.interpolate(r.get() + 3 * i);
    }

    void interpMesh2TruthIsig(const RealField3D &mesh) {
        for (int i = 0; i < N; ++i)
            truth[i] = isig[i] * z1[i] * mesh.interpolate(r.get() + 3 * i);
    }

    /* overwrite qFile->delta */
    void fillRngNoise(MyRNG &rng) {
        rng.fillVectorNormal(truth, N);
    }

    void blockRandom(MyRNG &rng, const fidcosmo::ArinyoP3DModel *p3d_model) {
        double *ccov = GL_CCOV[myomp::getThreadNum()].get();

        for (int i = 0; i < N; ++i) {
            double isigG = isig[i] * z1[i];

            ccov[i * (N + 1)] = 1.0 + p3d_model->getVarLss() * isigG * isigG;

            for (int j = i + 1; j < N; ++j) {
                float rz = r[3 * j + 2] - r[3 * i + 2];
                double isigG_ij = isigG * isig[j] * z1[j];
                ccov[j + i * N] = p3d_model->evalCorrFunc1dT(rz) * isigG_ij;
            }
        }

        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', N, ccov, N);
        /*if (info != 0) {
            LOG::LOGGER.STD("Error in CosmicQuasar::blockRandom::LAPACKE_dpotrf");
        }*/
        rng.fillVectorNormal(truth, N);
        cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    N, ccov, N, truth, 1);
    }

    void multIsigInVector() {
        for (int i = 0; i < N; ++i)
            in[i] *= isig[i];
    }

    void findGridPoints(const RealField3D &mesh) {
        min_x_idx = round(r[0] / mesh.dx[0]);
        for (int i = 0; i < N; ++i) {
            min_x_idx = std::min(
                min_x_idx, size_t(round(r[3 * i] / mesh.dx[0])));
            grid_indices.insert(mesh.getNgpIndex(r.get() + 3 * i));
        }
    }

    std::set<size_t> findNeighborPixels(
            const RealField3D &mesh, double radius
    ) {
        std::vector<size_t> neighboring_pixels;
        for (const size_t &i : grid_indices) {
            auto other = mesh.findNeighboringPixels(i, radius);
            neighboring_pixels.reserve(neighboring_pixels.size() + other.size());
            std::move(other.begin(), other.end(),
                      std::back_inserter(neighboring_pixels));
            other.clear();
        }

        std::set<size_t> unique_neighboring_pixels(
            neighboring_pixels.begin(), neighboring_pixels.end());
        return unique_neighboring_pixels;
    }

    void trimNeighbors(
            double radius2, double radial,
            float ratio=0.1, double sep_arcsec=20.0,
            float dist_Mpc=60.0, bool remove_low_overlap=false,
            bool remove_identicals=false
    ) {
        /* Removes neighbors with low overlap and self. Also removes neighbors
        that are suspected be identical. Correct way to tackle identical
        objects would be to coadd them, but this is hard and left for later.
        */
        /* Removes self from neighbors. Self will be treated specially. */
        neighbors.erase(this);

        if (remove_identicals) {
            auto isSameQuasar = [this, &sep_arcsec, &dist_Mpc](
                    const CosmicQuasar* const &q
            ) {
                double sep = acos(
                    sin_dec * q->sin_dec
                    + cos_dec * q->cos_dec * cos(q->angles[0] - angles[0])
                ) * 3600.0 * 180.0 / MY_PI;

                double delta_dis = fabs(_quasar_dist - q->_quasar_dist);
                return (sep < sep_arcsec) && (delta_dis < dist_Mpc);
            };

            std::erase_if(neighbors, isSameQuasar);
        }

        auto lowOverlap =
            [this, &radius2, &radial, &ratio, &remove_low_overlap](
                const CosmicQuasar* const &q
        ) {
            int M = q->N, ninc_i = 0, ninc_j = 0;
            std::set<int> jdxs;

            #ifdef USE_SPHERICAL_DIST
                double cos_sep =
                    sin_dec * q->sin_dec
                    + cos_dec * q->cos_dec * cos(q->angles[0] - angles[0]);
            #else
                double ddec = angles[0] - q->angles[0],
                       dra = angles[1] - q->angles[1];
                double rperp2 = radial * radial * (ddec * ddec + dra * dra);
                if (rperp2 > radius2)
                    return true;
                float rz_max = sqrt(radius2 - rperp2);
            #endif

            for (int i = 0; i < N; ++i) {
                bool _in_i = false;
                for (int j = 0; j < M; ++j) {
                    #ifdef USE_SPHERICAL_DIST
                        double r2 = (
                            q->r[3 * j + 2] * q->r[3 * j + 2]
                            + r[3 * i + 2] * r[3 * i + 2]
                            - 2.0 * r[3 * i + 2] * q->r[3 * j + 2] * cos_sep);
                        if (r2 <= radius2) {
                            jdxs.insert(j);
                            _in_i = true;
                        }
                    #else
                        float rz = fabsf(q->r[3 * j + 2] - r[3 * i + 2]);
                        if (rz < rz_max) {
                            jdxs.insert(j);
                            _in_i = true;
                        }
                    #endif
                }

                if (_in_i)
                    ++ninc_i;
            }

            ninc_j = jdxs.size();
            bool low_overlap = remove_low_overlap;
            low_overlap &= ninc_i < (N * ratio) || ninc_j < (M * ratio);
            low_overlap |= ninc_i == 0;

            return low_overlap;
        };

        std::erase_if(neighbors, lowOverlap);
    }

    void constructMarginalization(int order) {
        /* assumes order >= 0 */
        int nvecs = order + 1;

        double *ccov = GL_CCOV[myomp::getThreadNum()].get(),
               *rrmat = GL_RMAT[myomp::getThreadNum()].get();
        auto Emat = std::make_unique<double[]>(nvecs * nvecs);
        std::vector<std::unique_ptr<double[]>> uvecs(nvecs);
        for (int a = 0; a < nvecs; ++a)
            uvecs[a] = std::make_unique<double[]>(N);

        std::copy_n(isig, N, uvecs[0].get());  // Zeroth order
        for (int a = 1; a < nvecs; ++a)
            for (int i = 0; i < N; ++i)
                uvecs[a][i] = isig[i] * pow(log(z1[i]), a);

        for (int a = 0; a < nvecs; ++a)
            mxhelp::normalize_vector(N, uvecs[a].get());

        for (int a = 0; a < nvecs; ++a) {
            for (int b = 0; b < nvecs; ++b) {
                Emat[b + a * nvecs] = cblas_ddot(
                    N, uvecs[a].get(), 1, uvecs[b].get(), 1);

                // if (a == b)
                // assert(fabs(Emat[a * (1 + nvecs)] - 1.0) < DOUBLE_EPSILON);
            }
        }

        mxhelp::LAPACKE_InvertMatrixLU(Emat.get(), nvecs);

        // Get (N + Sigma)^(-1)
        std::fill_n(ccov, N * N, 0);
        for (int i = 0; i < N; ++i)
            ccov[i * (N + 1)] = 1.0;
        for (int a = 0; a < nvecs; ++a)
            for (int b = 0; b < nvecs; ++b)
                cblas_dger(CblasRowMajor, N, N, -1.0 * Emat[b + a * nvecs],
                           uvecs[a].get(), 1, uvecs[b].get(), 1, ccov, N);

        mxhelp::LAPACKE_sym_eigens(ccov, N, uvecs[0].get(), rrmat);

        // D^1/2
        for (int i = 0; i < N; ++i) {
            if (uvecs[0][i] < N * DOUBLE_EPSILON)
                uvecs[0][i] = 0;
            else
                uvecs[0][i] = sqrt(uvecs[0][i]);
        }

        // R^-1/2 such that S D^1/2 S^T
        mxhelp::transpose_copy(rrmat, ccov, N, N);
        std::fill_n(rrmat, N * N, 0);
        for (int a = 0; a < N; ++a)
            if (uvecs[0][a] != 0)
                cblas_dsyr(CblasRowMajor, CblasUpper, N,
                           uvecs[0][a], ccov + a * N, 1, rrmat, N);
        mxhelp::copyUpperToLower(rrmat, N);

        ioh::continuumMargFileHandler->write(rrmat, N, qFile->id, fidx);
    }

    #ifdef USE_SPHERICAL_DIST
    void setCrossCov(
            const CosmicQuasar *q, const fidcosmo::ArinyoP3DModel *p3d_model,
            double radial, double *ccov
    ) const {
        int M = q->N;
        double cos_sep =
            sin_dec * q->sin_dec
            + cos_dec * q->cos_dec * cos(q->angles[0] - angles[0]);
        float cos_half = sqrt((1.0 + cos_sep) / 2.0),
              sin_half = sqrt((1.0 - cos_sep) / 2.0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float rperp = (q->r[3 * j + 2] + r[3 * i + 2]) * sin_half,
                      rz = fabsf(q->r[3 * j + 2] - r[3 * i + 2]) * cos_half;
                ccov[j + i * M] = p3d_model->evalCorrFunc2dS(rperp, rz);
            }
        }
    }
    #else
    void setCrossCov(
            const CosmicQuasar *q, const fidcosmo::ArinyoP3DModel *p3d_model,
            double radial, double *ccov
    ) const {
        int M = q->N;
        double ddec = angles[0] - q->angles[0],
               dra = angles[1] - q->angles[1];
        float rperp = radial * sqrt(ddec * ddec + dra * dra);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float rz = fabsf(q->r[3 * j + 2] - r[3 * i + 2]);
                ccov[j + i * M] = p3d_model->evalCorrFunc2dS(rperp, rz);
            }
        }
    }
    #endif

    void multCovNeighbors(
            const fidcosmo::ArinyoP3DModel *p3d_model, double radial
    ) {
        /* We cannot use symmetry (update neighbor's out with C^T) here
           since it will cause race condition for the neighboring quasar.

        Impossible (self is always included)
        if (neighbors.empty())
            return;

        Single precision ccov (and DiscreteInterpolation2D) does not yield
        significant speed improvement. They result in numerical instability.
        */
        double *ccov = GL_CCOV[myomp::getThreadNum()].get();

        /* Multiply self */
        for (int i = 0; i < N; ++i) {
            ccov[i * (N + 1)] = p3d_model->getVar1dS();

            for (int j = i + 1; j < N; ++j) {
                float rz = r[3 * j + 2] - r[3 * i + 2];
                ccov[j + i * N] = p3d_model->evalCorrFunc1dS(rz);
            }
        }

        cblas_dsymv(CblasRowMajor, CblasUpper, N, 1.0,
                    ccov, N, in_isig, 1, 1.0, out, 1);

        /* Multiply others */
        for (const CosmicQuasar* q : neighbors) {
            setCrossCov(q, p3d_model, radial, ccov);
            int M = q->N;

            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, M, 1.0,
                        ccov, M, q->in_isig, 1, 1.0, out, 1);
        }
    }

    void multCovNeighborsOnly(
            const fidcosmo::ArinyoP3DModel *p3d_model, double radial
    ) {
        /* See comments in multCovNeighbors. Check for neighbors outside this
           function. */
        double *ccov = GL_CCOV[myomp::getThreadNum()].get();

        for (const CosmicQuasar* q : neighbors) {
            setCrossCov(q, p3d_model, radial, ccov);
            int M = q->N;

            cblas_dgemv(CblasRowMajor, CblasNoTrans, N, M, 1.0,
                        ccov, M, q->in_isig, 1, 1.0, out, 1);
        }
    }

    static void allocCcov(size_t size) {
        GL_CCOV.resize(myomp::getMaxNumThreads());
        for (auto it = GL_CCOV.begin(); it != GL_CCOV.end(); ++it)
            *it = std::make_unique<double[]>(size);
    }

    static void allocRrmat(size_t size) {
        GL_RMAT.resize(myomp::getMaxNumThreads());
        for (auto it = GL_RMAT.begin(); it != GL_RMAT.end(); ++it)
            *it = std::make_unique<double[]>(size);
    }

private:
    inline static std::vector<std::unique_ptr<double[]>> GL_CCOV, GL_RMAT;
};

#endif
