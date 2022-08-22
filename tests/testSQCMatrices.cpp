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
#include "core/sq_table.hpp"
#include "core/one_qso_estimate.hpp"

#include "mathtools/matrix_helper.hpp"

#include "io/logger.hpp"
#include "io/io_helper_functions.hpp"
#include "io/config_file.hpp"

class TestOneQSOEstimate: public OneQSOEstimate
{
public:
    TestOneQSOEstimate(const std::string &fname_qso) : OneQSOEstimate(fname_qso) {};
    ~TestOneQSOEstimate() {};

    void saveMatrices(std::string out_dir)
    {
        chunks[0]._allocateMatrices();
        process::sq_private_table->readSQforR(chunks[0].RES_INDEX, 
            chunks[0].interp2d_signal_matrix, chunks[0].interp_derivative_matrix);

        // Save fiducial signal matrix
        chunks[0]._setFiducialSignalMatrix(chunks[0].covariance_matrix);
        std::string fsave(out_dir);
        fsave+="/signal_matrix.txt";
        mxhelp::fprintfMatrix(fsave.c_str(), chunks[0].covariance_matrix, 
            chunks[0].qFile->size, chunks[0].qFile->size);

        // Save Q0 matrix
        chunks[0]._setQiMatrix(chunks[0].temp_matrix[0], 0);
        fsave=out_dir+"/q0_matrix.txt";
        mxhelp::fprintfMatrix(fsave.c_str(), chunks[0].temp_matrix[0], 
            chunks[0].qFile->size, chunks[0].qFile->size);

        chunks[0]._freeMatrices();
        delete chunks[0].interp2d_signal_matrix;
        chunks[0].interp2d_signal_matrix = NULL;
        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        {
            delete chunks[0].interp_derivative_matrix[kn];
            chunks[0].interp_derivative_matrix[kn] = NULL;
        }
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
        // conv::readConversion(config);
        // fidcosmo::readFiducialCosmo(config);

        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        bins::cleanUpBins();
        return 1;
    }

    try
    {
        // Allocate and read look up tables
        process::sq_private_table = new SQLookupTable(config);
        if (process::SAVE_ALL_SQ_FILES)
            process::sq_private_table->readTables();
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
    std::string INPUT_DIR = config.get("FileInputDir");
    ioh::readList(config.get("FileNameList").c_str(), filepaths);
    // Add parent directory to file path
    for (std::vector<std::string>::iterator fq = filepaths.begin(); fq != filepaths.end(); ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, INPUT_DIR);
    }

    TestOneQSOEstimate toqso(filepaths[0]);
    toqso.saveMatrices(config.get("OutputDir"));

    delete process::sq_private_table;
    return 0;
}

