#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"

#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define MIN_PIXELS_IN_SPEC 20

OneQSOEstimate::OneQSOEstimate(std::string fname_qso)
{
    qio::QSOFile *qFile = new qio::QSOFile(fname_qso, specifics::INPUT_QSO_FILE);

    qFile->readParameters();
    qFile->readData();

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile->readAllocResolutionMatrix();

        if (specifics::RESOMAT_DECONVOLUTION_M>0)
            qFile->Rmat->deconvolve(specifics::RESOMAT_DECONVOLUTION_M);

        if (specifics::OVERSAMPLING_FACTOR > 0)
            qFile->Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile->dlambda);
    }

    qFile->closeFile();

    // Boundary cut
    int newsize = qFile->cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

    if (newsize < MIN_PIXELS_IN_SPEC)   return;

    // decide nchunk with lambda points array[nchunks+1]
======
    // create chunk objects
    chunks.reserve(nchunks);
    for (int nc = 0; nc < nchunks; ++nc)
        chunks.emplace_back(qFile, lamchunk[nc], lamchunk[nc+1]);
}

OneQSOEstimate::~OneQSOEstimate() {}

double OneQSOEstimate::getComputeTimeEst(std::string fname_qso, int &zbin)
{
    zbin=-1;

    try
    {
        qio::QSOFile qtemp(fname_qso, specifics::INPUT_QSO_FILE);

        qtemp.readParameters();
        qtemp.readData();
        int newsize = qtemp.cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        if (newsize < MIN_PIXELS_IN_SPEC)
            return 0;

        double z1, z2, zm;
        qtemp.readMinMaxMedRedshift(z1, z2, zm);
        zbin = bins::findRedshiftBin(zm);

        // decide chunks
        ======
        // add compute time from chunks
        res = 0;
        for (int nc = 0; nc < nchunks; ++nc)
            res += Chunk::getComputeTimeEst(&qFile, lamchunk[nc], lamchunk[nc+1]);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("%s. Skipping %s.\n", e.what(), fname_qso.c_str());
        return 0;
    }
}

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate, 
    double *dbt_sum_vector[3], double *fisher_sum)
{
    for (auto it = chunks.begin(); it != chunks.end(); ++it)
        it->oneQSOiteration(ps_estimate, dbt_sum_vector, fisher_sum);
}








