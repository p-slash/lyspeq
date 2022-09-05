#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>

#include <gsl/gsl_errno.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/global_numbers.hpp"
#include "core/sq_table.hpp"
#include "core/one_qso_estimate.hpp"
#include "core/fiducial_cosmology.hpp"

#include "mathtools/matrix_helper.hpp"

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"

#include "tests/test_utils.hpp"

class TestOneQSOEstimate: public OneQSOEstimate
{
public:
    TestOneQSOEstimate(const std::string &fname_qso) : OneQSOEstimate(fname_qso)
    {
        chunks[0]->_allocateMatrices();
        process::sq_private_table->readSQforR(chunks[0]->RES_INDEX, 
            chunks[0]->interp2d_signal_matrix, chunks[0]->interp_derivative_matrix);
    };

    ~TestOneQSOEstimate() { chunks[0]->_freeMatrices(); };

    int test_setFiducialSignalMatrix();
    int test_setQiMatrix();
};

int TestOneQSOEstimate::test_setFiducialSignalMatrix()
{
    chunks[0]->_setFiducialSignalMatrix(chunks[0]->covariance_matrix);

    const std::string
    fname_sfid_matrix = std::string(SRCDIR) + "/tests/truth/signal_matrix.txt";
    const int ndim = 488;

    std::vector<double> A, vec;
    int nrows, ncols;

    A = mxhelp::fscanfMatrix(fname_sfid_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    assert(chunks[0]->qFile->size() == ndim);

    if (not allClose(A.data(), chunks[0]->covariance_matrix, ndim))
    {
        fprintf(stderr, "ERROR Chunk::_setFiducialSignalMatrix.\n");
        // printMatrices(A.data(), chunks[0]->covariance_matrix, ndim, ndim);
        return 1;
    }

    return 0;
}

int TestOneQSOEstimate::test_setQiMatrix()
{
    chunks[0]->_setQiMatrix(chunks[0]->temp_matrix[0], 0);

    const std::string
    fname_q0_matrix = std::string(SRCDIR) + "/tests/truth/q0_matrix.txt";
    const int ndim = 488;

    std::vector<double> A, vec;
    int nrows, ncols;

    A = mxhelp::fscanfMatrix(fname_q0_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    assert(chunks[0]->qFile->size() == ndim);

    if (not allClose(A.data(), chunks[0]->temp_matrix[0], ndim))
    {
        fprintf(stderr, "ERROR Chunk::_setQiMatrix.\n");
        // printMatrices(A.data(), chunks[0]->temp_matrix[0], ndim, ndim);
        return 1;
    }

    return 0;
}

int test_SQLookupTable(const ConfigFile &config)
{
    int r = 0;
    // Allocate truth
    const std::string truth_dir = std::string(SRCDIR) + "/tests/truth/";
    const config_map truth_sq_map ({{"OutputDir", truth_dir}});

    ConfigFile truth_config(truth_sq_map);
    truth_config.addDefaults(config);
    auto truth_sq_table = std::make_unique<SQLookupTable>(truth_config);
    truth_sq_table->readTables();

    auto truth_sig = truth_sq_table->getSignalMatrixInterp(0);
    auto calc_sig  = process::sq_private_table->getSignalMatrixInterp(0);

    if (*calc_sig != *truth_sig)
    {
        fprintf(stderr, "ERROR SQLookupTable::getSignalMatrixInterp.\n");
        r += 1;
    }

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        auto truth_q = truth_sq_table->getDerivativeMatrixInterp(kn, 0);
        auto calc_q  = process::sq_private_table->getDerivativeMatrixInterp(kn, 0);

        if (*calc_q != *truth_q)
        {
            fprintf(stderr, "ERROR SQLookupTable::getDerivativeMatrixInterp.\n");
            r += 1;
        }
    }

    return r;
}

int main(int argc, char *argv[])
{
    int r=0;
    #if defined(ENABLE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process::this_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &process::total_pes);
    #else
    process::this_pe   = 0;
    process::total_pes = 1;
    #endif

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        #if defined(ENABLE_MPI)
        MPI_Finalize();
        #endif
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();

    try
    {
        // Read variables from config file and set up bins.
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), process::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return 1;
    }

    try
    {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
        fidcosmo::readFiducialCosmo(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        return 1;
    }

    try
    {
        // Allocate and read look up tables
        process::sq_private_table = std::make_unique<SQLookupTable>(config);
        process::sq_private_table->computeTables(true);
        process::sq_private_table->readTables();

        r+=test_SQLookupTable(config);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while SQ Table contructed: %s\n", e.what());
        
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    std::vector<std::string> filepaths;
    std::string INPUT_DIR = config.get("FileInputDir");
    ioh::readList(config.get("FileNameList").c_str(), filepaths);
    // Add parent directory to file path
    for (std::vector<std::string>::iterator fq = filepaths.begin(); fq != filepaths.end(); ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, INPUT_DIR);
    }

    TestOneQSOEstimate toqso(filepaths[0]);
    r+=toqso.test_setFiducialSignalMatrix();
    r+=toqso.test_setQiMatrix();
    return r;
}

