#ifndef COSMIC_QUASAR_H
#define COSMIC_QUASAR_H

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "io/qso_file.hpp"
#include "mathtools/real_field_3d.hpp"
#include "qu3d/cosmology_3d.hpp"


// The median of DEC distribution in radians
constexpr double med_dec = 0.14502735752295168;

namespace specifics {
    static int DOWNSAMPLE_FACTOR;
}

class MyRNG {
public:
    MyRNG() :uidist(0, 1) {};

    void seed(long in) { rng_engine.seed(in); }

    void fillVectorNormal(double *v, unsigned int size) {
        for (unsigned int i = 0; i < size; ++i)
            v[i] = n_dist(rng_engine);
    }

    void fillVectorOnes(double *v, size_t size) {
        for (size_t i = 0; i < size; ++i)
            v[i] = (uidist(rng_engine) == 0) ? -1 : +1;
    }

private:
    std::mt19937_64 rng_engine;
    std::normal_distribution<double> n_dist;
    std::uniform_int_distribution<int> uidist;
};


class CosmicQuasar {
public:
    std::unique_ptr<qio::QSOFile> qFile;
    int N;
    /* z1: 1 + z */
    /* Cov . in = out, out should be compared to truth for inversion. */
    double *z1, *isig, angles[3], *in, *out, *truth;
    std::unique_ptr<double[]> r, y, Cy, residual, search;
    MyRNG rng;

    std::set<size_t> grid_indices;
    std::set<const CosmicQuasar*> neighbors;

    CosmicQuasar(const qio::PiccaFile *pf, int hdunum) {
        qFile = std::make_unique<qio::QSOFile>(pf, hdunum);
        // qFile->fname = fpath.str();
        qFile->readParameters();

        if (qFile->snr < specifics::MIN_SNR_CUT) {
            std::ostringstream err_msg;
            err_msg << "CosmicQuasar::CosmicQuasar::Low SNR in TARGETID "
                    << qFile->id;
            throw std::runtime_error(err_msg.str());
        }

        qFile->readData();
        qFile->maskOutliers();
        qFile->cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        // Convert wave to 1 + z
        std::for_each(
            qFile->wave(), qFile->wave() + qFile->size(), [](double &ld) {
                ld = ld / LYA_REST;
            }
        );

        qFile->convertNoiseToIvar();

        if (specifics::DOWNSAMPLE_FACTOR > 1)
            qFile->downsample(specifics::DOWNSAMPLE_FACTOR);

        N = qFile->size();
        z1 = qFile->wave();
        isig = qFile->noise();

        r = std::make_unique<double[]>(N);
        y = std::make_unique<double[]>(N);
        Cy = std::make_unique<double[]>(N);
        residual = std::make_unique<double[]>(N);
        search = std::make_unique<double[]>(N);

        // Convert to inverse sigma and weight deltas
        for (int i = 0; i < N; ++i) {
            isig[i] = sqrt(isig[i]);
            qFile->delta()[i] *= isig[i];
        }

        angles[0] = qFile->ra;
        angles[1] = qFile->dec - med_dec;
        angles[2] = 1;

        rng.seed(qFile->id);
        in = y.get();
        truth = qFile->delta();
        out = Cy.get();
    }

    void setRadialComovingDistance(const fidcosmo::FlatLCDM *cosmo) {
        for (int i = 0; i < N; ++i)
            r[i] = cosmo->getComovingDist(z1[i]);
    }

    /* Equirectangular projection */
    void getCartesianCoords(int i, double coord[3]) const {
        for (int axis = 0; axis < 3; ++axis)
            coord[axis] = r[i] * angles[axis];
    }

    /* overwrite qFile->delta */
    void fillRngNoise() { rng.fillVectorNormal(truth, N); }
    void fillRngOnes() { rng.fillVectorOnes(truth, N); }

    void multIsigInVector() {
        for (int i = 0; i < N; ++i)
            in[i] *= isig[i];
    }

    void divIsigInVector() {
        for (int i = 0; i < N; ++i)
            in[i] /= isig[i] + DOUBLE_EPSILON;
    }

    void findGridPoints(const RealField3D &mesh) {
        double coord[3];
        for (int i = 0; i < N; ++i) {
            getCartesianCoords(i, coord);
            grid_indices.insert(mesh.getNgpIndex(coord));
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
};

#endif
