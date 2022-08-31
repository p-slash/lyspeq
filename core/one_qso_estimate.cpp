#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"

#include "io/logger.hpp"
#include "io/qso_file.hpp"

#include <stdexcept>

const int
MIN_PIXELS_IN_SPEC = 20,
MAX_PIXELS_IN_FOREST = 1000;

int _decideNChunks(int size, std::vector<int> &indices)
{
    int nchunks = 1;
    if (specifics::NUMBER_OF_CHUNKS>1)
        nchunks += (specifics::NUMBER_OF_CHUNKS*size)/MAX_PIXELS_IN_FOREST;

    indices.reserve(nchunks+1);
    for (int i = 0; i < nchunks; ++i)
        indices.push_back((size*i)/nchunks);
    indices.push_back(size);

    return nchunks;
}

OneQSOEstimate::OneQSOEstimate(const std::string &f_qso)
{
    fname_qso = f_qso;
    qio::QSOFile qFile(fname_qso, specifics::INPUT_QSO_FILE);

    qFile.readParameters();
    qFile.readData();

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile.readAllocResolutionMatrix();

        if (specifics::RESOMAT_DECONVOLUTION_M>0)
            qFile.Rmat->deconvolve(specifics::RESOMAT_DECONVOLUTION_M);

        if (specifics::OVERSAMPLING_FACTOR > 0)
            qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    }

    qFile.closeFile();

    // Boundary cut
    int newsize = qFile.cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

    if (newsize < MIN_PIXELS_IN_SPEC)   return;

    // decide nchunk with lambda points array[nchunks+1]
    int nchunks = _decideNChunks(newsize, indices);

    // create chunk objects
    chunks.reserve(nchunks);
    for (int nc = 0; nc < nchunks; ++nc)
        chunks.emplace_back(qFile, indices[nc], indices[nc+1]);
}

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
        std::vector<int> indices;
        int nchunks = _decideNChunks(newsize, indices);

        // add compute time from chunks
        double res = 0;
        for (int nc = 0; nc < nchunks; ++nc)
            res += Chunk::getComputeTimeEst(qtemp, indices[nc], indices[nc+1]);
        return res;
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("%s. Skipping %s.\n", e.what(), fname_qso.c_str());
        return 0;
    }
}

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate, 
    std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
    double *fisher_sum)
{
    for (auto it = chunks.begin(); it != chunks.end(); ++it)
    {
        try
        {
            it->oneQSOiteration(ps_estimate, dbt_sum_vector, fisher_sum);
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("%s. Skipping %s.\n", e.what(), fname_qso.c_str());
        }
    }
}








