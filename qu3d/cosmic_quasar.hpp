#ifndef COSMIC_QUASAR_H
#define COSMIC_QUASAR_H

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "io/qso_file.hpp"
#include "qu3d/cosmology_3d.hpp"


// The median of DEC distribution in radians
constexpr double med_dec = 0.14502735752295168;

namespace specifics {
    int DOWNSAMPLE_FACTOR = 3;
}

class NormalRNG {
public:
    NormalRNG() {};

    void seed(long in) { rng_engine.seed(in); }

    double generate() { return n_dist(rng_engine); }

    void fillVector(double *v, unsigned int size) {
        for (unsigned int i = 0; i < size; ++i)
            v[i] = generate();
    }

private:
    std::mt19937_64 rng_engine;
    std::normal_distribution<double> n_dist;
};


class CosmicQuasar {
public:
    std::unique_ptr<qio::QSOFile> qFile;
    int N;
    /* z1: 1 + z */
    /* Cov . in = out, out should be compared to truth for inversion. */
    double *z1, *ivar, angles[3], *in, *out, *truth;
    std::unique_ptr<double[]> r, y, Cy, residual, search;
    NormalRNG rng;

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

        N = qFile->size();
        // Convert wave to 1 + z
        std::for_each(
            qFile->wave(), qFile->wave() + N, [](double &ld) {
                ld = ld / LYA_REST;
            }
        );
        z1 = qFile->wave();

        qFile->convertNoiseToIvar();
        ivar = qFile->noise();

        r = std::make_unique<double[]>(N);
        y = std::make_unique<double[]>(N);
        Cy = std::make_unique<double[]>(N);
        residual = std::make_unique<double[]>(N);
        search = std::make_unique<double[]>(N);

        // Weight deltas
        for (int i = 0; i < N; ++i)
            qFile->delta()[i] *= ivar[i];

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
    void getCartesianCoords(int i, double coord[3]) {
        for (int axis = 0; axis < 3; ++axis)
            coord[axis] = r[i] * angles[axis];
    }

    void fillRngNoise() {
        /* overwrite qFile->delta */
        rng.fillVector(truth, N);
        for (int i = 0; i < N; ++i)
            truth[i] *= sqrt(ivar[i]);
    }
};

#endif
