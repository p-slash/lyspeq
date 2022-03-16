#ifndef QUADRATIC_ESTIMATE_H
#define QUADRATIC_ESTIMATE_H

#include <vector>
#include <string>

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
    int NUMBER_OF_QSOS, NUMBER_OF_QSOS_OUT, *Z_BIN_COUNTS;

    std::vector<std::string> local_fpaths;

    // 3 TOTAL_KZ_BINS sized vectors
    double  *dbt_estimate_sum_before_fisher_vector[3],
            *dbt_estimate_fisher_weighted_vector[3], 
            *previous_power_estimate_vector, *current_power_estimate_vector,
            *powerspectra_fits;

    // 2 TOTAL_KZ_BINS x TOTAL_KZ_BINS sized matrices
    double *fisher_matrix_sum, *inverse_fisher_matrix_sum;

    bool isFisherInverted;

    void _readQSOFiles(const char *fname_list, const char *dir);
    void _savePEResult();

    // The next 2 functions call Python scripts.
    // Intermadiate files are saved in TMP_FOLDER (read as TemporaryFolder in config file)
    // make install will copy this script to $HOME/bin and make it executable.
    // Add $HOME/bin to your $PATH

    // Fitting procedure calls Python3 script lorentzian_fit.py.
    void _fitPowerSpectra(double *fitted_power);

    // Weighted smoothing using 2D spline calls Python3 script smbivspline.py
    void _smoothPowerSpectra(double *smoothed_power);
    void _readScriptOutput(double *script_power, const char *fname, void *itsfits=NULL);

    // Performs a load balancing operation based on N^3 estimation
    void _loadBalancing(std::vector<std::string> &filepaths, 
        std::vector< std::pair<double, int> > &cpu_fname_vector);

public:
    static double *precomputed_fisher;
    static void readPrecomputedFisher(const char *fname);

    OneDQuadraticPowerEstimate(const char *fname_list, const char *dir);

    ~OneDQuadraticPowerEstimate();
    
    double powerSpectrumFiducial(int kn, int zm);

    void initializeIteration();
    void invertTotalFisherMatrix();
    void computePowerSpectrumEstimates();

    // Passing fit values for the power spectrum for numerical stability
    void iterate(int number_of_iterations, const char *fname_base);

    // Deviation between actual estimates (not fit values)
    bool hasConverged();
    
    void printfSpectra();
    void writeFisherMatrix(const char *fname);

    // Does not write the last bin since it is ignored when LAST_K_EDGE defined
    // You can find that value in logs--printfSpectra prints all
    void writeSpectrumEstimates(const char *fname);
    void writeDetailedSpectrumEstimates(const char *fname);
    void iterationOutput(const char *fnamebase, int it, double t1, double tot);
};

class Smoother
{
    #define HWSIZE 25
    #define KS 2*HWSIZE+1

    static int sigmapix;
    static double gaussian_kernel[KS];
    static bool isKernelSet, useMeanNoise, isSmoothingOn;

public:
    static void setParameters(int noisefactor);
    static void setGaussianKernel();
    static void smoothNoise(const double *n2, double *out, int size);

    #undef HWSIZE
    #undef KS
};


#endif

