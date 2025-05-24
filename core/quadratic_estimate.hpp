#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include <vector>
#include <string>
#include <memory>

#include "mathtools/stats.hpp"
#include "io/config_file.hpp"
#include "io/myfitsio.hpp"

// This umbrella class manages the quadratic estimator by 
//      storing the total Fisher matrix and its inverse,
//      computing the power spectrum estimate,
//      and fitting a smooth function to the power spectrum estimate using Python3 script.

// This object performs a load balancing operation based on N^3 estimation
// and skips chunks that do not belong to any redshift bin
//

// It takes + a file path which should start with number of quasars followed by a list of quasar files,
//          + the directory these qso placed as const char *dir
// Call iterate with max number of iterations and output filename base to const char *fname_base
//      fname_base should have output folder and output filebase in one string

class OneDQuadraticPowerEstimate
{
protected:
    ConfigFile &config;
    int NUMBER_OF_QSOS, NUMBER_OF_QSOS_OUT, NUMBER_OF_ITERATIONS;
    std::vector<int> Z_BIN_COUNTS;

    // 3 TOTAL_KZ_BINS sized vectors
    std::vector<std::unique_ptr<double[]>>
                dbt_estimate_sum_before_fisher_vector,
                dbt_estimate_fisher_weighted_vector;
    std::unique_ptr<double[]>   previous_power_estimate_vector,
                                current_power_estimate_vector,
                                powerspectra_fits;

    std::unique_ptr<double[]> temp_vector;
    std::vector<double> precomputed_fisher;
    void _readPrecomputedFisher(const std::string &fname);

    // 3 TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrices
    std::unique_ptr<double[]>
        fisher_matrix_sum, inverse_fisher_matrix_sum,
        solver_invfisher_matrix;

    bool isFisherInverted;

    // reads all qso files, load balances
    // Returns local_fpaths
    std::vector<std::string> _readQSOFiles();
    // Performs a load balancing operation based on N^3 estimation
    // Returns local_fpaths
    std::vector<std::string>
    _loadBalancing(std::vector<std::string> &filepaths, 
        std::vector< std::pair<double, int> > &cpu_fname_vector);
    void _savePEResult();

    // The next 2 functions call Python scripts.
    // Intermadiate files are saved in TMP_FOLDER (read as TemporaryFolder in config file)
    // make install will copy this script to $HOME/bin and make it executable.
    // Add $HOME/bin to your $PATH

    // Fitting procedure calls Python3 script lorentzian_fit.py.
    // void _fitPowerSpectra(double *fitted_power);

    // Weighted smoothing using 2D spline calls Python3 script smbivspline.py
    void _smoothPowerSpectra();
    void _readScriptOutput(const char *fname, void *itsfits=NULL);

public:
    /* This function reads following keys from config file:
    NumberOfIterations: int
        Number of iterations. Default 1.
    PrecomputedFisher: string
        File to precomputed Fisher matrix. If present, Fisher matrix is not
            calculated for spectra. Off by default.
    FileNameList: string
        File to spectra to list. Filenames are wrt FileInputDir.
    FileInputDir: string
        Directory where files reside.
    */
    OneDQuadraticPowerEstimate(ConfigFile &con);
    
    double powerSpectrumFiducial(int kn, int zm);

    void initializeIteration();
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimates();

    // Passing fit values for the power spectrum for numerical stability
    void iterate();

    // Deviation between actual estimates (not fit values)
    bool hasConverged();
    
    void printfSpectra();
    void writeFisherMatrix(const char *fname);

    // Does not write the last bin since it is ignored when LAST_K_EDGE defined
    // You can find that value in logs--printfSpectra prints all
    void writeSpectrumEstimates(const char *fname);
    void writeDetailedSpectrumEstimates(fitsfile *fits_file, const std::string ext);
    void iterationOutput(
        int it, double t1, double tot,
        std::vector<double> &times_all_pes);
};

#endif

