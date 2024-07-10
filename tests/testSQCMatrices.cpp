#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>

#include <gsl/gsl_errno.h>

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
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
        glmemory::allocMemory();
        chunks[0]->_initMatrices();
        chunks[0]->_setVZMatrices();
    };

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
    chunks[0]->_setQiMatrix(chunks[0]->stored_ikz_qi[0].second, 0);
    chunks[0]->_applyRedshiftInterp();

    const std::string
    fname_q0_matrix = std::string(SRCDIR) + "/tests/truth/q0_matrix.txt";
    const int ndim = 488;

    std::vector<double> A, vec;
    int nrows, ncols;

    A = mxhelp::fscanfMatrix(fname_q0_matrix.c_str(), nrows, ncols);
    assert(nrows == ndim);
    assert(ncols == ndim);
    assert(chunks[0]->qFile->size() == ndim);

    if (not allClose(A.data(), chunks[0]->stored_ikz_qi[0].second, ndim))
    {
        fprintf(stderr, "ERROR Chunk::_setQiMatrix.\n");
        // printMatrices(A.data(), chunks[0]->stored_ikz_qi[0].second, 50, ndim);
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
    mympi::init(argc, argv);

    if (argc<2)
    {
        fprintf(stderr, "Missing config file!\n");
        mympi::finalize();
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();

    try
    {
        // Read variables from config file and set up bins.
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), mympi::this_pe);
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
        mympi::abort();
        return 1;
    }

    std::vector<std::string> filepaths;
    std::string INPUT_DIR = config.get("FileInputDir") + "/";
    ioh::readList(config.get("FileNameList").c_str(), filepaths);
    // Add parent directory to file path
    for (auto &fq : filepaths)
        fq.insert(0, INPUT_DIR);

    TestOneQSOEstimate toqso(filepaths[0]);
    r+=toqso.test_setFiducialSignalMatrix();
    r+=toqso.test_setQiMatrix();

    if (r == 0)
        LOG::LOGGER.STD("SQ matrices work!\n");

    mympi::finalize();

    return r;
}

