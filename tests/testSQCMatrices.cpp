#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/global_numbers.hpp"
#include "core/matrix_helper.hpp"
#include "core/one_qso_estimate.hpp"
#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"

class TestOneQSOEstimate: public OneQSOEstimate
{
public:
    TestOneQSOEstimate(std::string fname_qso) : OneQSOEstimate(fname_qso) {};
    ~TestOneQSOEstimate() {};

    void saveMatrices(std::string out_dir)
    {
        _allocateMatrices();
        process::sq_private_table->readSQforR(RES_INDEX, interp2d_signal_matrix, interp_derivative_matrix);

        // Save fiducial signal matrix
        _setFiducialSignalMatrix(covariance_matrix);
        std::string fsave(out_dir);
        fsave+="/signal_matrix.txt";
        mxhelp::fprintfMatrix(fsave.c_str(), covariance_matrix, DATA_SIZE, DATA_SIZE);

        // Save Q0 matrix
        _setQiMatrix(temp_matrix[0], 0);
        fsave=out_dir+"/q0_matrix.txt";
        mxhelp::fprintfMatrix(fsave.c_str(), temp_matrix[0], DATA_SIZE, DATA_SIZE);

        _freeMatrices();
        delete interp2d_signal_matrix;
        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
            delete interp_derivative_matrix[kn];
    }
};

int main(int argc, char *argv[])
{
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
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];
    int r = 0;

    gsl_set_error_handler_off();

    char FNAME_LIST[300], FNAME_RLIST[300], INPUT_DIR[300], FILEBASE_S[300], FILEBASE_Q[300],
         OUTPUT_DIR[300], OUTPUT_FILEBASE[300], buf[700];

    int NUMBER_OF_ITERATIONS;

    try
    {
        // Read variables from config file and set up bins.
        ioh::readConfigFile( FNAME_CONFIG, FNAME_LIST, FNAME_RLIST, INPUT_DIR, OUTPUT_DIR,
            OUTPUT_FILEBASE, FILEBASE_S, FILEBASE_Q, &NUMBER_OF_ITERATIONS, NULL, NULL, NULL);
    }
    catch (std::exception& e)
    {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        return 1;
    }

    try
    {
        LOG::LOGGER.open(OUTPUT_DIR, process::this_pe);
    }
    catch (std::exception& e)
    {   
        fprintf(stderr, "Error while logging contructed: %s\n", e.what());
        bins::cleanUpBins();

        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    try
    {
        // Allocate and read look up tables
        process::sq_private_table = new SQLookupTable(INPUT_DIR, FILEBASE_S, FILEBASE_Q, FNAME_RLIST);
        // process::sq_private_table->readTables();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while SQ Table contructed: %s\n", e.what());
        bins::cleanUpBins();
        
        #if defined(ENABLE_MPI)
        MPI_Abort(MPI_COMM_WORLD, 1);
        #endif

        return 1;
    }

    std::vector<std::string> filepaths;
    int NUMBER_OF_QSOS = ioh::readList(FNAME_LIST, filepaths);
    // Add parent directory to file path
    for (std::vector<std::string>::iterator fq = filepaths.begin(); fq != filepaths.end(); ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, INPUT_DIR);
    }

    TestOneQSOEstimate toqso(filepaths[0]);
    toqso.saveMatrices(std::string(OUTPUT_DIR));
    return 0;
}

