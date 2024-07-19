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

// Line of sight coarsing for mesh
#ifndef M_LOS
#define M_LOS 1
#endif

#if M_LOS > 1
#define COARSE_INTERP
#endif

namespace specifics {
    static int DOWNSAMPLE_FACTOR;
}

template <class T>
struct CompareCosmicQuasarPtr {
    bool operator()(const T *lhs, const T *rhs) const {
        return lhs->qFile->id < rhs->qFile->id;
    }
};

class CosmicQuasar {
public:
    std::unique_ptr<qio::QSOFile> qFile;
    int N, coarse_N, fidx;
    long fpos;
    /* z1: 1 + z */
    /* Cov . in = out, out should be compared to truth for inversion. */
    double *z1, *isig, angles[3], *in, *out, *truth, *in_isig;
    std::unique_ptr<double[]> r, y, Cy, residual, search, coarse_r, coarse_in, y_isig;

    std::set<size_t> grid_indices;
    std::set<const CosmicQuasar*, CompareCosmicQuasarPtr<CosmicQuasar>> neighbors;
    size_t min_x_idx;

    CosmicQuasar(const qio::PiccaFile *pf, int hdunum) {
        qFile = std::make_unique<qio::QSOFile>(pf, hdunum);
        N = 0;  coarse_N = 0;

        qFile->readParameters();

        if (qFile->snr < specifics::MIN_SNR_CUT) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::Low SNR in TARGETID "
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
        z1 = qFile->wave();
        isig = qFile->ivar();

        r = std::make_unique<double[]>(3 * N);
        y = std::make_unique<double[]>(N);
        y_isig = std::make_unique<double[]>(N);
        Cy = std::make_unique<double[]>(N);
        residual = std::make_unique<double[]>(N);
        search = std::make_unique<double[]>(N);

        #ifdef COARSE_INTERP
            coarse_N = N / M_LOS;
            coarse_N += N % M_LOS != 0;
            coarse_r = std::make_unique<double[]>(3 * coarse_N);
            coarse_in = std::make_unique<double[]>(coarse_N);
        #endif
        // Convert to inverse sigma and weight deltas
        for (int i = 0; i < N; ++i) {
            isig[i] = sqrt(isig[i]);
            qFile->delta()[i] *= isig[i];
        }

        angles[0] = qFile->ra + ra_shift;
        if (angles[0] >= 2 * MY_PI)
            angles[0] -= 2 * MY_PI;

        angles[1] = qFile->dec;
        angles[2] = 1;

        in = y.get();
        in_isig = y_isig.get();
        truth = qFile->delta();
        out = Cy.get();
    }
    CosmicQuasar(CosmicQuasar &&rhs) = delete;
    CosmicQuasar(const CosmicQuasar &rhs) = delete;

    void setComovingDistances(const fidcosmo::FlatLCDM *cosmo) {
        /* Equirectangular projection */
        for (int i = 0; i < N; ++i) {
            double chi = cosmo->getComovingDist(z1[i]);
            r[0 + 3 * i] = angles[0] * chi;
            r[1 + 3 * i] = angles[1] * chi;
            r[2 + 3 * i] = angles[2] * chi;
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
        double *rrmat = GL_RMAT[myomp::getThreadNum()].get();
        ioh::continuumMargFileHandler->read(fidx, fpos, N, rrmat);
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, N, N, 1.0,
            rrmat, N, in, 1, 0, in_isig, 1);

        for (int i = 0; i < N; ++i)
            in_isig[i] *= isig[i] * z1[i];
    }

    void multTruthWithMarg() {
        double *rrmat = GL_RMAT[myomp::getThreadNum()].get();
        ioh::continuumMargFileHandler->read(fidx, fpos, N, rrmat);
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans, N, N, 1.0,
            rrmat, N, truth, 1, 0, in_isig, 1);

        std::swap(truth, in_isig);
    }

    void interpMesh2Out(const RealField3D &mesh) {
        for (int i = 0; i < N; ++i)
            out[i] = mesh.interpolate(r.get() + 3 * i);
    }

    void interpMesh2TruthIsig(const RealField3D &mesh) {
        for (int i = 0; i < N; ++i)
            truth[i] = isig[i] * z1[i] * mesh.interpolate(r.get() + 3 * i);
    }

    #ifdef COARSE_INTERP
        void setCoarseComovingDistances() {
            for (int i = 0; i < N; ++i) {
                int j = i / M_LOS;
                coarse_r[0 + 3 * j] += r[0 + 3 * i];
                coarse_r[1 + 3 * j] += r[1 + 3 * i];
                coarse_r[2 + 3 * j] += r[2 + 3 * i];
            }

            int rem = N % M_LOS;
            if (rem == 0) {
                cblas_dscal(3 * coarse_N, 1.0 / M_LOS, coarse_r.get(), 1);
            }
            else{
                cblas_dscal(3 * coarse_N - 3, 1.0 / M_LOS, coarse_r.get(), 1);
                cblas_dscal(3, 1.0 / rem, coarse_r.get() + 3 * coarse_N - 3, 1);
            }
        }

        void coarseGrainIn() {
            std::fill_n(coarse_in.get(), coarse_N, 0);
            for (int i = 0; i < N; ++i)
                coarse_in[i / M_LOS] += in[i] * z1[i];
        }

        void coarseGrainInIsig() {
            std::fill_n(coarse_in.get(), coarse_N, 0);
            for (int i = 0; i < N; ++i)
                coarse_in[i / M_LOS] += in_isig[i];
        }

        void interpMesh2Coarse(const RealField3D &mesh) {
            for (int i = 0; i < coarse_N; ++i)
                coarse_in[i] = mesh.interpolate(coarse_r.get() + 3 * i);
        }

        void interpNgpCoarse2Out() {
            for (int i = 0; i < N; ++i)
                out[i] = coarse_in[i / M_LOS];
        }

        void interpNgpCoarse2TruthIsig() {
            for (int i = 0; i < N; ++i)
                truth[i] = isig[i] * z1[i] * coarse_in[i / M_LOS];
        }

        void interpLinCoarseIsig() {
            for (int i = 0; i < N; ++i) {
                int I = std::min(coarse_N - 2, std::max(0, (i - M_LOS / 2) / M_LOS));
                double m =
                    (coarse_in[I + 1] - coarse_in[I])
                    / (coarse_r[3 * I + 5] - coarse_r[3 * I + 2]);
                double y = coarse_in[I] + m * (r[3 * i + 2] - coarse_r[3 * I + 2]);
                out[i] += isig[i] * y;
            }
        }
    #endif

    /* overwrite qFile->delta */
    void fillRngNoise(MyRNG &rng) {
        rng.fillVectorNormal(truth, N);
        for (int i = 0; i < N; ++i)
            if (isig[i] == 0)
                truth[i] = 0;
    }

    void fillRngOnes(MyRNG &rng) {
        rng.fillVectorOnes(truth, N);
        for (int i = 0; i < N; ++i)
            if (isig[i] == 0)
                truth[i] = 0;
    }

    void multIsigInVector() {
        for (int i = 0; i < N; ++i)
            in[i] *= isig[i];
    }

    void divIsigInVector() {
        for (int i = 0; i < N; ++i)
            in[i] /= isig[i] + DOUBLE_EPSILON;
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

    void constructMarginalization(int order) {
        /* assumes order >= 0 */
        int nvecs = order + 1;

        double *ccov = GL_CCOV[myomp::getThreadNum()].get();
        double *rrmat = GL_RMAT[myomp::getThreadNum()].get();
        auto Emat = std::make_unique<double[]>(nvecs * nvecs);
        std::vector<std::unique_ptr<double[]>> uvecs(nvecs);
        for (int a = 0; a < nvecs; ++a)
            uvecs[a] = std::make_unique<double[]>(N);

        std::fill_n(uvecs[0].get(), N, 1.0 / sqrt(N * 1.0));  // Zeroth order

        for (int a = 1; a < nvecs; ++a) {
            double *out = uvecs[a].get();
            std::transform(
                z1, z1 + N, out,
                [a](const double &l) { return pow(log(l), a); }
            );
            double norm = 1.0 / cblas_dnrm2(N, out, 1);
            cblas_dscal(N, norm, out, 1);
        }

        for (int a = 0; a < nvecs; ++a)
            for (int i = 0; i < N; ++i)
                uvecs[a][i] *= isig[i];

        for (int a = 0; a < nvecs; ++a)
            for (int b = 0; b < nvecs; ++b)
                Emat[b + a * nvecs] = cblas_ddot(
                    N, uvecs[a].get(), 1, uvecs[b].get(), 1);

        mxhelp::LAPACKE_InvertMatrixLU(Emat.get(), nvecs);

        // Get (N + Sigma)^(-1)
        std::fill_n(ccov, N * N, 0);
        for (int i = 0; i < N; ++i)
            ccov[i * (N + 1)] = 1.0;
        for (int a = 0; a < nvecs; ++a)
            for (int b = 0; b < nvecs; ++b)
                cblas_dger(CblasRowMajor, N, N, -1.0 * Emat[b + a * nvecs],
                           uvecs[a].get(), 1, uvecs[b].get(), 1, ccov, N);

        // for (int i = 0; i < N; ++i)
        //     for (int j = 0; j < N; ++j)
        //         ccov[j + i * N] *= isig[j] * isig[i];

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

        fpos = ioh::continuumMargFileHandler->write(rrmat, N, fidx);
    }

    void setCrossCov(
            const CosmicQuasar *q, const fidcosmo::ArinyoP3DModel *p3d_model,
            double *ccov
    ) const {
        int M = q->N;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                double dx = r[3 * i] - q->r[3 * j],
                       dy = r[3 * i + 1] - q->r[3 * j + 1];

                double rz = fabs(r[3 * i + 2] - q->r[3 * j + 2]),
                       rperp = sqrt(dx * dx + dy * dy);
                ccov[j + i * M] = p3d_model->evalCorrFunc2dS(rperp, rz);
            }
        }
    }

    void multCovNeighbors(const fidcosmo::ArinyoP3DModel *p3d_model) {
        /* We cannot use symmetry (update neighbor's out with C^T) here
           since it will cause race condition for the neighboring quasar. */
        if (neighbors.empty())
            return;

        double *ccov = GL_CCOV[myomp::getThreadNum()].get();

        for (const auto &q : neighbors) {
            setCrossCov(q, p3d_model, ccov);
            int M = q->N;

            cblas_dgemv(
                CblasRowMajor, CblasNoTrans, N, M, 1.0,
                ccov, M, q->in_isig, 1, 1, out, 1);

            // The following creates race conditions
            // cblas_dgemv(
            //     CblasRowMajor, CblasTrans, M, N, 1.0,
            //     ccov, N, q->in, 1, 1, q->out, 1);
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
