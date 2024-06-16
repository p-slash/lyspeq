#ifndef COSMIC_QUASAR_H
#define COSMIC_QUASAR_H

#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "io/qso_file.hpp"
#include "qu3d/cosmology_3d.hpp"

class CosmicQuasar {
public:
    std::unique_ptr<qio::QSOFile> qFile;

    CosmicQuasar(qio::PiccaFile *pf, int hdunum) {
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
        z1 = qFile->wave();

        // Convert to ivar again
        std::for_each(
            qFile->noise(), qFile->noise() + qFile->size(), [](double &n) {
                n *= n;
                n = 1 / n;
                if (n < DOUBLE_EPSILON)
                    n = 0;
            }
        );
        ivar = qFile->noise();
    }

    /* return 1 + z */
    double* getZ1() const { return z1; }
    double* getIvar() const { return ivar; }

    void setRadialComovingDistance(const fidcosmo::FlatLCDM *cosmo) {
        r = std::make_unique<double[]>(qFile->size());
        for (int i = 0; i < qFile->size(); ++i)
            r[i] = cosmo->getComovingDist(z1[i]);
    }

private:
    double *z1, *ivar;
    std::unique_ptr<double[]> r, CinvDelta;
};

#endif
