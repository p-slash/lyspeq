#include "core/one_qso_estimate.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"

#include "io/io_helper_functions.hpp"
#include "io/qso_file.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

void OneQSOEstimate::_readFromFile(std::string fname_qso)
{
    qso_sp_fname = fname_qso;

    // Construct and read data arrays
    QSOFile qFile(qso_sp_fname.c_str());

    double dummy_qso_z, dummy_s2n;

    qFile.readParameters(DATA_SIZE, dummy_qso_z, SPECT_RES_FWHM, dummy_s2n, DV_KMS);

    // LOG::LOGGER.IO("Reading from %s.\n" "Data size is %d\n" "Pixel Width is %.1f\n" "Spectral Resolution is %d.\n",
    //     qso_sp_fname.c_str(), DATA_SIZE, DV_KMS, SPECT_RES_FWHM);
    
    lambda_array    = new double[DATA_SIZE];
    velocity_array  = new double[DATA_SIZE];
    flux_array      = new double[DATA_SIZE];
    noise_array     = new double[DATA_SIZE];

    qFile.readData(lambda_array, flux_array, noise_array);

    // Find the resolution index for the look up table
    r_index = process::sq_private_table->findSpecResIndex(SPECT_RES_FWHM);
    
    if (r_index == -1)      throw std::runtime_error("SPECRES not found in tables!");

    interp2d_signal_matrix   = process::sq_private_table->getSignalMatrixInterp(r_index);
    interp_derivative_matrix = new Interpolation*[bins::NUMBER_OF_K_BANDS];

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        interp_derivative_matrix[kn] = process::sq_private_table->getDerivativeMatrixInterp(kn, r_index);
}

bool OneQSOEstimate::_findRedshiftBin()
{
    ZBIN     = bins::findRedshiftBin(MEDIAN_REDSHIFT);
    ZBIN_LOW = bins::findRedshiftBin(LOWER_REDSHIFT);
    ZBIN_UPP = bins::findRedshiftBin(UPPER_REDSHIFT);

    // LOG::LOGGER.IO("Redshift bins: %.2f--%d, %.2f--%d, %.2f--%d\n", LOWER_REDSHIFT, ZBIN_LOW, MEDIAN_REDSHIFT, ZBIN,
    //     UPPER_REDSHIFT, ZBIN_UPP);

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

void OneQSOEstimate::_setStoredMatrices()
{
    // Number of Qj matrices to preload.
    double size_m1 = (double)sizeof(double) * DATA_SIZE * DATA_SIZE / 1048576.; // in MB
    
    // Need at least 3 matrices as temp
    nqj_eff      = process::MEMORY_ALLOC / size_m1 - 3;
    isSfidStored = false;
    
    if (nqj_eff <= 0)
        nqj_eff = 0;
    else 
    {
        if (nqj_eff > N_Q_MATRICES)
        {
            nqj_eff      = N_Q_MATRICES;
            isSfidStored = !specifics::TURN_OFF_SFID;
        }

        stored_qj = new double*[nqj_eff];
    }

    // LOG::LOGGER.IO("Number of stored Q matrices: %d\n", nqj_eff);
    // if (isSfidStored)   LOG::LOGGER.IO("Fiducial signal matrix is stored.\n");

    isQjSet   = false;
    isSfidSet = false;
}

OneQSOEstimate::OneQSOEstimate(std::string fname_qso)
{
    isCovInverted = false;
    _readFromFile(fname_qso);

    // Covert from wavelength to velocity units around median wavelength
    conv::convertLambdaToVelocity(MEDIAN_REDSHIFT, velocity_array, lambda_array, DATA_SIZE);
    LOWER_REDSHIFT = lambda_array[0] / LYA_REST - 1;
    UPPER_REDSHIFT = lambda_array[DATA_SIZE-1] / LYA_REST - 1;

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(lambda_array, flux_array, noise_array, DATA_SIZE);
    
    // Keep noise as error squared (variance)
    std::for_each(noise_array, noise_array+DATA_SIZE, [](double &n) { n*=n; });

    // LOG::LOGGER.IO("Length of v is %.1f\n" "Median redshift: %.2f\n" "Redshift range: %.2f--%.2f\n", 
    //     velocity_array[DATA_SIZE-1] - velocity_array[0], MEDIAN_REDSHIFT, LOWER_REDSHIFT, UPPER_REDSHIFT);

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
}

double OneQSOEstimate::getComputeTimeEst()
{
    #ifdef FISHER_OPTIMIZATION
    #define N_M_COMBO 3.
    #else
    #define N_M_COMBO (N_Q_MATRICES + 1.)
    #endif

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
    double v_ij, z_ij, temp;

    if (isSfidSet)
    {
        if (copy)
            std::copy(stored_sfid, stored_sfid + (DATA_SIZE*DATA_SIZE), sm);
        else
            sm = stored_sfid;
    }
    else
    {
        for (int row = 0; row < DATA_SIZE; ++row)
        {
            for (int col = row; col < DATA_SIZE; ++col)
            {
                _getVandZ(v_ij, z_ij, row, col);

                temp = interp2d_signal_matrix->evaluate(z_ij, v_ij);
                *(sm+col+DATA_SIZE*row) = temp;
            }
        }

        mxhelp::copyUpperToLower(sm, DATA_SIZE);
    }
    
    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}

void OneQSOEstimate::_setQiMatrix(double *&qi, int i_kz, bool copy)
{
    ++mytime::number_of_times_called_setq;
    double t = mytime::timer.getTime(), t_interp;
    int kn, zm;
    double v_ij, z_ij, temp;

    if (isQjSet && (i_kz >= (N_Q_MATRICES - nqj_eff)))
    {
        t_interp = 0;
        if (copy)
            std::copy(stored_qj[N_Q_MATRICES-i_kz-1], stored_qj[N_Q_MATRICES-i_kz-1] + (DATA_SIZE*DATA_SIZE), qi);
        else
            qi = &stored_qj[N_Q_MATRICES-i_kz-1][0];
    }
    else
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        for (int row = 0; row < DATA_SIZE; ++row)
        {
            for (int col = row; col < DATA_SIZE; ++col)
            {
                _getVandZ(v_ij, z_ij, row, col);

                temp  = bins::redshiftBinningFunction(z_ij, zm);
                temp *= interp_derivative_matrix[kn]->evaluate(v_ij);

                // Every pixel pair should scale to the bin redshift
                #ifdef REDSHIFT_GROWTH_POWER
                temp *= fidcosmo::fiducialPowerGrowthFactor(z_ij, bins::KBAND_CENTERS[kn], bins::ZBIN_CENTERS[zm], 
                    &fidpd13::FIDUCIAL_PD13_PARAMS);
                #endif

                *(qi+col+DATA_SIZE*row) = temp;
            }
        }

        t_interp = mytime::timer.getTime() - t;

        mxhelp::copyUpperToLower(qi, DATA_SIZE);
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
        std::for_each(covariance_matrix, covariance_matrix+DATA_SIZE*DATA_SIZE, 
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

        _getFisherMatrix(Q_ikz_matrix, i_kz);
    }

    delete [] weighted_data_vector;
}

void OneQSOEstimate::oneQSOiteration(const double *ps_estimate, double *dbt_sum_vector[3], double *fisher_sum)
{
    _allocateMatrices();

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
            mxhelp::vector_add(dbt_sum_vector[dbt_i], dbt_estimate_before_fisher_vector[dbt_i], bins::TOTAL_KZ_BINS);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("%d/%d - ERROR %s: Covariance matrix is not invertable. %s\n",
                process::this_pe, process::total_pes, e.what(), qso_sp_fname.c_str());

        LOG::LOGGER.ERR("Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
                DATA_SIZE, MEDIAN_REDSHIFT, DV_KMS, SPECT_RES_FWHM);
    }
    
    _freeMatrices();
}

void OneQSOEstimate::_allocateMatrices()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector[dbt_i] = new double[bins::TOTAL_KZ_BINS]();

    fisher_matrix = new double[bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS]();

    covariance_matrix = new double[DATA_SIZE * DATA_SIZE];

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = new double[DATA_SIZE * DATA_SIZE];
    
    for (int i = 0; i < nqj_eff; ++i)
        stored_qj[i] = new double[DATA_SIZE * DATA_SIZE];
    
    if (isSfidStored)
        stored_sfid = new double[DATA_SIZE * DATA_SIZE];
    
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


/*
// Move constructor
OneQSOEstimate::OneQSOEstimate(OneQSOEstimate &&rhs)
    : qso_sp_fname(rhs.qso_sp_fname), SPECT_RES_FWHM(rhs.SPECT_RES_FWHM), DATA_SIZE(rhs.DATA_SIZE), 
      N_Q_MATRICES(rhs.N_Q_MATRICES), fisher_index_start(rhs.fisher_index_start), r_index(rhs.r_index),
      MEDIAN_REDSHIFT(rhs.MEDIAN_REDSHIFT), BIN_REDSHIFT(rhs.BIN_REDSHIFT), DV_KMS(rhs.DV_KMS), 
      lambda_array(NULL), velocity_array(NULL), flux_array(NULL), noise_array(NULL),
      covariance_matrix(NULL), inverse_covariance_matrix(NULL),
      stored_qj(NULL), stored_sfid(NULL),
      nqj_eff(rhs.nqj_eff), isQjSet(rhs.isQjSet), isSfidSet(rhs.isSfidSet), isSfidStored(rhs.isSfidStored),
      isCovInverted(rhs.isCovInverted), 
      ZBIN(rhs.ZBIN), ps_before_fisher_estimate_vector(NULL), fisher_matrix(NULL)
{
    // Move
    lambda_array   = rhs.lambda_array;
    velocity_array = rhs.velocity_array;
    flux_array     = rhs.flux_array;
    noise_array    = rhs.noise_array;

    covariance_matrix         = rhs.covariance_matrix;
    inverse_covariance_matrix = rhs.inverse_covariance_matrix;

    temp_matrix[0] = rhs.temp_matrix[0];
    temp_matrix[1] = rhs.temp_matrix[1];

    stored_qj   = rhs.stored_qj;
    stored_sfid = rhs.stored_sfid;

    ps_before_fisher_estimate_vector = rhs.ps_before_fisher_estimate_vector;
    fisher_matrix                    = rhs.fisher_matrix;

    // Set rhs to NULL to prevent double deleting
    rhs.lambda_array   = NULL;
    rhs.velocity_array = NULL;
    rhs.flux_array     = NULL;
    rhs.noise_array    = NULL;

    rhs.covariance_matrix         = NULL;
    rhs.inverse_covariance_matrix = NULL;

    rhs.temp_matrix[0] = NULL;
    rhs.temp_matrix[1] = NULL;

    rhs.stored_qj   = NULL;
    rhs.stored_sfid = NULL;

    rhs.ps_before_fisher_estimate_vector = NULL;
    rhs.fisher_matrix                    = NULL;
}

OneQSOEstimate& OneQSOEstimate::operator=(OneQSOEstimate&& rhs)
: qso_sp_fname(rhs.qso_sp_fname), SPECT_RES_FWHM(rhs.SPECT_RES_FWHM), DATA_SIZE(rhs.DATA_SIZE), 
      N_Q_MATRICES(rhs.N_Q_MATRICES), fisher_index_start(rhs.fisher_index_start), r_index(rhs.r_index),
      MEDIAN_REDSHIFT(rhs.MEDIAN_REDSHIFT), BIN_REDSHIFT(rhs.BIN_REDSHIFT), DV_KMS(rhs.DV_KMS), 
      lambda_array(NULL), velocity_array(NULL), flux_array(NULL), noise_array(NULL),
      covariance_matrix(NULL), inverse_covariance_matrix(NULL),
      stored_qj(NULL), stored_sfid(NULL),
      nqj_eff(rhs.nqj_eff), isQjSet(rhs.isQjSet), isSfidSet(rhs.isSfidSet), isSfidStored(rhs.isSfidStored),
      isCovInverted(rhs.isCovInverted), 
      ZBIN(rhs.ZBIN), ps_before_fisher_estimate_vector(NULL), fisher_matrix(NULL)
{
    if (this != &rhs)
    {
        // Free the existing resource.
        delete [] flux_array;
        delete [] lambda_array;
        delete [] velocity_array;
        delete [] noise_array;

        if (nqj_eff > 0)
            delete [] stored_qj;

        // Copy the data pointer and its length from the
        // source object.
        qso_sp_fname = rhs.qso_sp_fname;
        
        lambda_array   = rhs.lambda_array;
        velocity_array = rhs.velocity_array;
        flux_array     = rhs.flux_array;
        noise_array    = rhs.noise_array;

        covariance_matrix         = rhs.covariance_matrix;
        inverse_covariance_matrix = rhs.inverse_covariance_matrix;

        temp_matrix[0] = rhs.temp_matrix[0];
        temp_matrix[1] = rhs.temp_matrix[1];

        stored_qj   = rhs.stored_qj;
        stored_sfid = rhs.stored_sfid;

        ps_before_fisher_estimate_vector = rhs.ps_before_fisher_estimate_vector;
        fisher_matrix                    = rhs.fisher_matrix;

        // Release the data pointer from the source object so that
        // the destructor does not free the memory multiple times.
        // Set rhs to NULL to prevent double deleting
        rhs.lambda_array   = NULL;
        rhs.velocity_array = NULL;
        rhs.flux_array     = NULL;
        rhs.noise_array    = NULL;

        rhs.covariance_matrix         = NULL;
        rhs.inverse_covariance_matrix = NULL;

        rhs.temp_matrix[0] = NULL;
        rhs.temp_matrix[1] = NULL;

        rhs.stored_qj   = NULL;
        rhs.stored_sfid = NULL;

        rhs.ps_before_fisher_estimate_vector = NULL;
        rhs.fisher_matrix                    = NULL;
    }
    return *this;
}

*/










