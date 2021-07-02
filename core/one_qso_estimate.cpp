#include "core/one_qso_estimate.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/global_numbers.hpp"

#include "io/io_helper_functions.hpp"
#include "io/qso_file.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define DATA_SIZE_2 DATA_SIZE*DATA_SIZE

void OneQSOEstimate::_readFromFile(std::string fname_qso)
{
    qso_sp_fname = fname_qso;
    qio::QSOFile qFile(qso_sp_fname, specifics::INPUT_QSO_FILE);

    double dummy_qso_z, dummy_s2n, dum_Rkms;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, SPECT_RES_FWHM, dummy_s2n, DV_KMS);
    dum_Rkms = SPEED_OF_LIGHT/SPECT_RES_FWHM/ONE_SIGMA_2_FWHM;

    lambda_array    = new double[DATA_SIZE];
    velocity_array  = new double[DATA_SIZE];
    flux_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(lambda_array, flux_array, noise_array);

    // Covert from wavelength to velocity units around median wavelength
    conv::convertLambdaToVelocity(MEDIAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);
    LOWER_REDSHIFT = lambda_array[0] / LYA_REST - 1;
    UPPER_REDSHIFT = lambda_array[DATA_SIZE-1] / LYA_REST - 1;

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        RES_INDEX = 0;
        if (specifics::INPUT_QSO_FILE == qio::Picca)
            qFile.readAllocResolutionMatrix(reso_matrix);
        else
        {
            reso_matrix = new mxhelp::Resolution(DATA_SIZE, 11);
            reso_matrix->constructGaussian(velocity_array, dum_Rkms, DV_KMS);
        }
    }
    else
    {
        reso_matrix = NULL;
        // Find the resolution index for the look up table
        RES_INDEX = process::sq_private_table->findSpecResIndex(SPECT_RES_FWHM, DV_KMS);

        if (RES_INDEX == -1)      throw std::runtime_error("SPECRES not found in tables!");
    }

    interp2d_signal_matrix   = NULL;
    interp_derivative_matrix = new DiscreteInterpolation1D*[bins::NUMBER_OF_K_BANDS];
}

bool OneQSOEstimate::_findRedshiftBin()
{
    ZBIN     = bins::findRedshiftBin(MEDIAN_REDSHIFT);
    ZBIN_LOW = bins::findRedshiftBin(LOWER_REDSHIFT);
    ZBIN_UPP = bins::findRedshiftBin(UPPER_REDSHIFT);

    // Chunk is completely out
    if ((ZBIN_LOW > (bins::NUMBER_OF_Z_BINS-1)) || (ZBIN_UPP < 0))
    {
        // LOG::LOGGER.IO("This QSO is completely out!\n");
        LOG::LOGGER.ERR("This QSO is completely out:\n" "File: %s\n" "Redshift range: %.2f--%.2f\n",
            qso_sp_fname.c_str(), LOWER_REDSHIFT, UPPER_REDSHIFT);

        return false;
    }

    if (ZBIN_LOW < 0)
    {
        // LOG::LOGGER.IO("This QSO is out on the low end!\n");
        LOG::LOGGER.ERR("This QSO is out on the low end:\n" "File: %s\n" "Redshift range: %.2f--%.2f\n",
            qso_sp_fname.c_str(), LOWER_REDSHIFT, UPPER_REDSHIFT);
        
        ZBIN_LOW = 0;
    }
    
    if (ZBIN_UPP > (bins::NUMBER_OF_Z_BINS-1))
    {
        // LOG::LOGGER.IO("This QSO is out on the high end!\n");
        LOG::LOGGER.ERR("This QSO is out on the high end:\n" "File: %s\n" "Redshift range: %.2f--%.2f\n",
            qso_sp_fname.c_str(), LOWER_REDSHIFT, UPPER_REDSHIFT);

        ZBIN_UPP = bins::NUMBER_OF_Z_BINS - 1;
    }

    // Assign to a redshift bin according to median redshift of this chunk
    // This is just for bookkeeping purposes
    if ((ZBIN >= 0) && (ZBIN < bins::NUMBER_OF_Z_BINS))
        BIN_REDSHIFT = bins::ZBIN_CENTERS[ZBIN];

    return true;
}

// For a top hat redshift bin, only access 1 redshift bin for each pixel
// For triangular z bins, access 2 redshift bins for each pixel
void OneQSOEstimate::_setNQandFisherIndex()
{   
    #if defined(TOPHAT_Z_BINNING_FN)
    N_Q_MATRICES        = ZBIN_UPP - ZBIN_LOW + 1;
    fisher_index_start  = bins::getFisherMatrixIndex(0, ZBIN_LOW);
    
    #elif defined(TRIANGLE_Z_BINNING_FN)
    // Assuming low and end points stay within their respective bins
    N_Q_MATRICES        = ZBIN_UPP - ZBIN_LOW + 1;
    fisher_index_start  = bins::getFisherMatrixIndex(0, ZBIN_LOW);
    
    // If we need to distribute low end to a lefter bin
    if ((LOWER_REDSHIFT < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0))
    {
        ++N_Q_MATRICES;
        fisher_index_start -= bins::NUMBER_OF_K_BANDS;
    }
    // If we need to distribute high end to righter bin
    if ((bins::ZBIN_CENTERS[ZBIN_UPP] < UPPER_REDSHIFT) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
    {
        ++N_Q_MATRICES;
    }
    #else
    #error "DEFINE A Z BINNING FUNCTION!"
    #endif

    N_Q_MATRICES *= bins::NUMBER_OF_K_BANDS;
}

// Find number of Qj matrices to preload.
void OneQSOEstimate::_setStoredMatrices()
{
    double size_m1 = (double)sizeof(double) * DATA_SIZE_2 / 1048576.; // in MB
    double remain_mem = process::MEMORY_ALLOC;

    // Resolution matrix needs another temp storage.
    if (specifics::USE_RESOLUTION_MATRIX)
        remain_mem -= reso_matrix->getMaxMemUsage();

    isSfidStored = false;

    // Need at least 3 matrices as temp
    if (remain_mem > (3+N_Q_MATRICES)*size_m1)
    {
        remain_mem -= (3+N_Q_MATRICES)*size_m1;
        nqj_eff     = N_Q_MATRICES;

        if (remain_mem > size_m1)
            isSfidStored = !specifics::TURN_OFF_SFID;
    }
    else
        nqj_eff = remain_mem / size_m1 - 3;

    if (nqj_eff < 0)  nqj_eff = 0;
    else              stored_qj = new double*[nqj_eff];

    if (nqj_eff != N_Q_MATRICES)
        LOG::LOGGER.IO("===============\n""Not all matrices are stored: %s\n"
            "#stored: %d vs #required:%d.\n""ND: %d, M1: %.1f MB. "
            "Avail mem after R & SQ subtracted: %.1lf MB\n""===============\n", 
            qso_sp_fname.c_str(), nqj_eff, N_Q_MATRICES, DATA_SIZE, size_m1, 
            remain_mem);

    // LOG::LOGGER.IO("Number of stored Q matrices: %d\n", nqj_eff);
    // if (isSfidStored)   LOG::LOGGER.IO("Fiducial signal matrix is stored.\n");

    isQjSet   = false;
    isSfidSet = false;
}

OneQSOEstimate::OneQSOEstimate(std::string fname_qso)
{
    isCovInverted = false;
    _readFromFile(fname_qso);

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(lambda_array, flux_array, noise_array, DATA_SIZE);
    
    // Keep noise as error squared (variance)
    std::for_each(noise_array, noise_array+DATA_SIZE, [](double &n) { n*=n; });

    nqj_eff = 0;

    if(!_findRedshiftBin()) return;
    
    // Set up number of matrices, index for Fisher matrix
    _setNQandFisherIndex();

    _setStoredMatrices();
}

OneQSOEstimate::~OneQSOEstimate()
{
    delete [] flux_array;
    delete [] lambda_array;
    delete [] velocity_array;
    delete [] noise_array;

    if (nqj_eff > 0)
        delete [] stored_qj;

    if (reso_matrix != NULL)
        delete reso_matrix;
}

double OneQSOEstimate::getComputeTimeEst(std::string fname_qso, int &zbin)
{
    #ifdef FISHER_OPTIMIZATION
    #define N_M_COMBO 3.
    #else
    #define N_M_COMBO (N_Q_MATRICES + 1.)
    #endif

    int DATA_SIZE, dumb_int;
    double dumb_double;

    qio::QSOFile qFile(fname_qso, specifics::INPUT_QSO_FILE);
    qFile.readParameters(DATA_SIZE, dumb_double, dumb_int, dumb_double, dumb_double);

    double *wave = new double[DATA_SIZE], *dumbarr = new double[DATA_SIZE];
    qFile.readData(wave, dumbarr, dumbarr);

    double z1 = wave[0]/LYA_REST-1, z2=wave[DATA_SIZE-1]/LYA_REST-1,
        zm = wave[DATA_SIZE]/LYA_REST-1;
    delete [] wave;
    delete [] dumbarr;

    zbin         = bins::findRedshiftBin(zm);
    int ZBIN_LOW = bins::findRedshiftBin(z1);
    int ZBIN_UPP = bins::findRedshiftBin(z2);
    int N_Q_MATRICES = ZBIN_UPP - ZBIN_LOW + 1;

    #if defined(TRIANGLE_Z_BINNING_FN)
    // If we need to distribute low end to a lefter bin
    if ((z1 < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0))
        ++N_Q_MATRICES;
    // If we need to distribute high end to righter bin
    if ((bins::ZBIN_CENTERS[ZBIN_UPP] < z2) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
        ++N_Q_MATRICES;
    #endif
    N_Q_MATRICES *= bins::NUMBER_OF_K_BANDS;

    if ((ZBIN_LOW > (bins::NUMBER_OF_Z_BINS-1)) || (ZBIN_UPP < 0))
        return 0;
    else
        return std::pow(DATA_SIZE/100., 3) * N_Q_MATRICES * N_M_COMBO;

    #undef N_M_COMBO
}

void OneQSOEstimate::_getVandZ(double &v_ij, double &z_ij, int i, int j)
{
    v_ij = velocity_array[j] - velocity_array[i];
    z_ij = sqrt(lambda_array[j] * lambda_array[i]) / LYA_REST - 1.;
}

void OneQSOEstimate::_setFiducialSignalMatrix(double *&sm, bool copy)
{
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::timer.getTime();
    double v_ij, z_ij;

    if (isSfidSet)
    {
        if (copy)   std::copy(stored_sfid, stored_sfid + DATA_SIZE_2, sm);
        else        sm = stored_sfid;
    }
    else
    {
        double *ptr = sm;
        for (int row = 0; row < DATA_SIZE; ++row)
        {
            ptr += row;
            for (int col = row; col < DATA_SIZE; ++col, ++ptr)
            {
                _getVandZ(v_ij, z_ij, row, col);

                *ptr = interp2d_signal_matrix->evaluate(z_ij, v_ij);
            }
        }

        mxhelp::copyUpperToLower(sm, DATA_SIZE);
        // Multiply reso mat
        if (reso_matrix != NULL)
            reso_matrix->sandwich(DATA_SIZE, sm);
    }
    
    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}

void OneQSOEstimate::_setQiMatrix(double *&qi, int i_kz, bool copy)
{
    ++mytime::number_of_times_called_setq;
    double t = mytime::timer.getTime(), t_interp;
    int kn, zm;
    double v_ij, z_ij;

    if (isQjSet && (i_kz >= (N_Q_MATRICES - nqj_eff)))
    {
        double *ptr = stored_qj[N_Q_MATRICES-i_kz-1];
        t_interp = 0;

        if (copy)   std::copy(ptr, ptr + DATA_SIZE_2, qi);
        else        qi = ptr;
    }
    else
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);
        double *ptr = qi;

        for (int row = 0; row < DATA_SIZE; ++row)
        {
            ptr += row;
            for (int col = row; col < DATA_SIZE; ++col, ++ptr)
            {
                _getVandZ(v_ij, z_ij, row, col);

                *ptr  = interp_derivative_matrix[kn]->evaluate(v_ij);
                *ptr *= bins::redshiftBinningFunction(z_ij, zm);
                // Every pixel pair should scale to the bin redshift
                #ifdef REDSHIFT_GROWTH_POWER
                *ptr *= fidcosmo::fiducialPowerGrowthFactor(z_ij, bins::KBAND_CENTERS[kn], 
                    bins::ZBIN_CENTERS[zm], &fidpd13::FIDUCIAL_PD13_PARAMS);
                #endif
            }
        }

        t_interp = mytime::timer.getTime() - t;

        mxhelp::copyUpperToLower(qi, DATA_SIZE);
        // Multiply reso mat
        if (reso_matrix != NULL)
            reso_matrix->sandwich(DATA_SIZE, qi);
    }

    t = mytime::timer.getTime() - t; 

    mytime::time_spent_set_qs += t;
    mytime::time_spent_on_q_interp += t_interp;
    mytime::time_spent_on_q_copy += t - t_interp;
}

void OneQSOEstimate::setCovarianceMatrix(const double *ps_estimate)
{
    // Set fiducial signal matrix
    if (!specifics::TURN_OFF_SFID)
        _setFiducialSignalMatrix(covariance_matrix);
    else
    {
        for (int i = 0; i < DATA_SIZE*DATA_SIZE; ++i)
            *(covariance_matrix+i) = 0;
    }

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        // SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        _setQiMatrix(temp_matrix[0], i_kz);

        cblas_daxpy(DATA_SIZE*DATA_SIZE, ps_estimate[i_kz + fisher_index_start], temp_matrix[0], 1, 
            covariance_matrix, 1);
    }

    // add noise matrix diagonally
    cblas_daxpy(DATA_SIZE, 1., noise_array, 1, covariance_matrix, DATA_SIZE+1);

    if (specifics::CONTINUUM_MARGINALIZATION_AMP > 0)
    {
        std::for_each(covariance_matrix, covariance_matrix + DATA_SIZE_2, 
            [&](double &c) { c += specifics::CONTINUUM_MARGINALIZATION_AMP; });
    }

    if (specifics::CONTINUUM_MARGINALIZATION_DERV > 0)
    {
        double *temp_t_vector = new double[DATA_SIZE];
        // double MEDIAN_LAMBDA = LYA_REST * (1 + MEDIAN_REDSHIFT);

        std::transform(lambda_array, lambda_array+DATA_SIZE, temp_t_vector, 
            [](const double &l) { return log(l/LYA_REST); });

        cblas_dger(CblasRowMajor, DATA_SIZE, DATA_SIZE, specifics::CONTINUUM_MARGINALIZATION_DERV, 
            temp_t_vector, 1, temp_t_vector, 1, covariance_matrix, DATA_SIZE);

        delete [] temp_t_vector;
    }
    
    isCovInverted = false;
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void OneQSOEstimate::invertCovarianceMatrix()
{
    double t = mytime::timer.getTime();

    mxhelp::LAPACKE_InvertMatrixLU(covariance_matrix, DATA_SIZE);
    
    inverse_covariance_matrix = covariance_matrix;

    isCovInverted = true;

    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_c_inv += t;
}

void OneQSOEstimate::_getWeightedMatrix(double *m)
{
    double t = mytime::timer.getTime();

    //C-1 . Q
    cblas_dsymm( CblasRowMajor, CblasLeft, CblasUpper,
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix, DATA_SIZE,
                 m, DATA_SIZE,
                 0, temp_matrix[1], DATA_SIZE);

    //C-1 . Q . C-1
    cblas_dsymm( CblasRowMajor, CblasRight, CblasUpper,
                 DATA_SIZE, DATA_SIZE, 1., inverse_covariance_matrix, DATA_SIZE,
                 temp_matrix[1], DATA_SIZE,
                 0, m, DATA_SIZE);

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_modqs += t;
}

void OneQSOEstimate::_getFisherMatrix(const double *Qw_ikz_matrix, int i_kz)
{
    double temp;
    double *Q_jkz_matrix = temp_matrix[1];

    double t = mytime::timer.getTime();
    
    // Now compute Fisher Matrix
    for (int j_kz = i_kz; j_kz < N_Q_MATRICES; ++j_kz)
    {
        #ifdef FISHER_OPTIMIZATION
        int diff_ji = j_kz - i_kz;

        if ((diff_ji != 0) && (diff_ji != 1) && (diff_ji != bins::NUMBER_OF_K_BANDS))
            continue;
        #endif
        
        Q_jkz_matrix = temp_matrix[1];
        _setQiMatrix(Q_jkz_matrix, j_kz, false);

        temp = 0.5 * mxhelp::trace_dsymm(Qw_ikz_matrix, Q_jkz_matrix, DATA_SIZE);

        int ind_ij = (i_kz + fisher_index_start) + bins::TOTAL_KZ_BINS * (j_kz + fisher_index_start),
            ind_ji = (j_kz + fisher_index_start) + bins::TOTAL_KZ_BINS * (i_kz + fisher_index_start);

        *(fisher_matrix + ind_ij) = temp;
        *(fisher_matrix + ind_ji) = temp;
    }

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_fisher += t;
}

void OneQSOEstimate::computePSbeforeFvector()
{
    double *weighted_data_vector = new double[DATA_SIZE];

    cblas_dsymv(CblasRowMajor, CblasUpper,
                DATA_SIZE, 1., inverse_covariance_matrix, DATA_SIZE,
                flux_array, 1,
                0, weighted_data_vector, 1);

    // #pragma omp parallel for
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        double *Q_ikz_matrix = temp_matrix[0], *Sfid_matrix = temp_matrix[1], temp_tk = 0;

        // Set derivative matrix ikz
        _setQiMatrix(Q_ikz_matrix, i_kz);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        double temp_dk = mxhelp::my_cblas_dsymvdot(weighted_data_vector, Q_ikz_matrix, DATA_SIZE);

        // Get weighted derivative matrix ikz: C-1 Qi C-1
        _getWeightedMatrix(Q_ikz_matrix);

        // Get Noise contribution: Tr(C-1 Qi C-1 N)
        double temp_bk = mxhelp::trace_ddiagmv(Q_ikz_matrix, noise_array, DATA_SIZE);

        // Set Fiducial Signal Matrix
        if (!specifics::TURN_OFF_SFID)
        {
            _setFiducialSignalMatrix(Sfid_matrix, false);

            // Tr(C-1 Qi C-1 Sfid)
            temp_tk = mxhelp::trace_dsymm(Q_ikz_matrix, Sfid_matrix, DATA_SIZE);
        }
        
        dbt_estimate_before_fisher_vector[0][i_kz + fisher_index_start] = temp_dk;
        dbt_estimate_before_fisher_vector[1][i_kz + fisher_index_start] = temp_bk;
        dbt_estimate_before_fisher_vector[2][i_kz + fisher_index_start] = temp_tk;
        
        // Do not compute fisher matrix if it is precomputed
        if (OneDQuadraticPowerEstimate::precomputed_fisher == NULL)
            _getFisherMatrix(Q_ikz_matrix, i_kz);
    }

    delete [] weighted_data_vector;
}

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate, double *dbt_sum_vector[3], 
    double *fisher_sum)
{
    _allocateMatrices();

    // This function allocates new signal & deriv matrices if process::SAVE_ALL_SQ_FILES=false 
    // i.e., no caching of SQ files
    // If all tables are cached, then this function simply points to those in process:sq_private_table
    process::sq_private_table->readSQforR(RES_INDEX, interp2d_signal_matrix, interp_derivative_matrix);

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    for (int j_kz = 0; j_kz < nqj_eff; ++j_kz)
        _setQiMatrix(stored_qj[j_kz], N_Q_MATRICES - j_kz - 1);
    
    if (nqj_eff > 0)    isQjSet = true;

    // Preload fiducial signal matrix if memory allows
    if (isSfidStored)
    {
        _setFiducialSignalMatrix(stored_sfid);
        isSfidSet = true;
    }

    setCovarianceMatrix(ps_estimate);

    try
    {
        invertCovarianceMatrix();

        computePSbeforeFvector();

        mxhelp::vector_add(fisher_sum, fisher_matrix, bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS);
        
        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            mxhelp::vector_add(dbt_sum_vector[dbt_i], dbt_estimate_before_fisher_vector[dbt_i], 
                bins::TOTAL_KZ_BINS);

        // Write results to file with their qso filename as base
        if (process::SAVE_EACH_SPEC_RESULT)
            _saveIndividualResult();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("%d/%d - ERROR %s: Covariance matrix is not invertable. %s\n",
                process::this_pe, process::total_pes, e.what(), qso_sp_fname.c_str());

        LOG::LOGGER.ERR("Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
                DATA_SIZE, MEDIAN_REDSHIFT, DV_KMS, SPECT_RES_FWHM);
    }
    
    _freeMatrices();

    // Do not delete if these are pointers to process::sq_private_table
    if (!process::SAVE_ALL_SQ_FILES)
    {
        if (interp2d_signal_matrix!=NULL)
            delete interp2d_signal_matrix;
        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
            delete interp_derivative_matrix[kn];
    }
}

void OneQSOEstimate::_allocateMatrices()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector[dbt_i] = new double[bins::TOTAL_KZ_BINS]();

    fisher_matrix = new double[bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS]();

    covariance_matrix = new double[DATA_SIZE_2];

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = new double[DATA_SIZE_2];
    
    for (int i = 0; i < nqj_eff; ++i)
        stored_qj[i] = new double[DATA_SIZE_2];
    
    if (isSfidStored)
        stored_sfid = new double[DATA_SIZE_2];
    
    isQjSet   = false;
    isSfidSet = false;
}

void OneQSOEstimate::_freeMatrices()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        delete [] dbt_estimate_before_fisher_vector[dbt_i];

    delete [] fisher_matrix;

    delete [] covariance_matrix;

    for (int i = 0; i < 2; ++i)
        delete [] temp_matrix[i];
    
    for (int i = 0; i < nqj_eff; ++i)
        delete [] stored_qj[i];

    if (isSfidStored)
        delete [] stored_sfid;
    
    isQjSet   = false;
    isSfidSet = false;
}

void OneQSOEstimate::_saveIndividualResult()
{
    mxhelp::vector_sub(dbt_estimate_before_fisher_vector[0], dbt_estimate_before_fisher_vector[1], 
        bins::TOTAL_KZ_BINS);
    mxhelp::vector_sub(dbt_estimate_before_fisher_vector[0], dbt_estimate_before_fisher_vector[2], 
        bins::TOTAL_KZ_BINS);

    std::string resfname = qso_sp_fname;
    resfname.replace(resfname.end()-4, resfname.end(), "_Fp.bin");

    FILE *toWrite = ioh::open_file(resfname.c_str(), "wb");
    
    int r = fwrite(&bins::TOTAL_KZ_BINS, sizeof(int), 1, toWrite);
    r+=fwrite(fisher_matrix, sizeof(double), bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS, toWrite);
    r+=fwrite(dbt_estimate_before_fisher_vector[0], sizeof(double), bins::TOTAL_KZ_BINS, toWrite);
    fclose(toWrite);

    if (r != bins::TOTAL_KZ_BINS*(bins::TOTAL_KZ_BINS+1)+1)
        LOG::LOGGER.ERR("ERROR %d saving individual results: %s\n", r, resfname.c_str());
}

void OneQSOEstimate::fprintfMatrices(const char *fname_base)
{
    char buf[1024];

    if (isSfidStored)
    {
        sprintf(buf, "%s-signal.txt", fname_base);
        mxhelp::fprintfMatrix(buf, stored_sfid, DATA_SIZE, DATA_SIZE);
    }

    if (reso_matrix != NULL)
    {
        sprintf(buf, "%s-resolution.txt", fname_base);
        reso_matrix->fprintfMatrix(buf);
    }
}

#undef DATA_SIZE_2







