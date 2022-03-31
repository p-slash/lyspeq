#include "core/chunk_estimate.hpp"
#include "core/quadratic_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"

#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define DATA_SIZE_2 qFile->size*qFile->size

double _L2MAX, _L2MIN;
inline
void _setL2Limits(int zm)
{
    #if defined(TOPHAT_Z_BINNING_FN)
    #define ZSTART (bins::ZBIN_CENTERS[zm]-bins::Z_BIN_WIDTH/2)
    #define ZEND (bins::ZBIN_CENTERS[zm]+bins::Z_BIN_WIDTH/2)
    #elif defined(TRIANGLE_Z_BINNING_FN)
    #define ZSTART (bins::ZBIN_CENTERS[zm]-bins::Z_BIN_WIDTH)
    #define ZEND (bins::ZBIN_CENTERS[zm]+bins::Z_BIN_WIDTH)
    #endif
    _L2MAX  = (1 + ZEND) * LYA_REST;
    _L2MAX *=_L2MAX;
    _L2MIN  = (1 + ZSTART) * LYA_REST;
    _L2MIN *=_L2MIN;
    #undef ZSTART
    #undef ZEND
}

// inline
// void _getZBinLimits(double *li, int remsize, double *&lptr1, double *&lptr2)
// {
//     lptr1 = std::lower_bound(li, li+remsize, _L2MIN/(*li));
//     lptr2 = std::upper_bound(li, li+remsize, _L2MAX/(*li));
// }

inline
void _getVandZ(double li, double lj, double &v_ij, double &z_ij)
{
    v_ij = SPEED_OF_LIGHT * log(lj / li);
    z_ij = sqrt(li * lj) / LYA_REST - 1.;
}

void Chunk::_copyQSOFile(const qio::QSOFile &qmaster, int i1, int i2)
{
    qFile = new qio::QSOFile(qmaster, i1, i2);

    // If using resolution matrix, read resolution matrix from picca file
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        RES_INDEX = 0;

        _matrix_n = qFile->Rmat->getNCols();
    }
    else
    {
        // Find the resolution index for the look up table
        RES_INDEX = process::sq_private_table->findSpecResIndex(qFile->R_fwhm, qFile->dv_kms);

        if (RES_INDEX == -1) throw std::out_of_range("SPECRES not found in tables!");
        _matrix_n = qFile->size;
    }

    interp2d_signal_matrix   = NULL;
    interp_derivative_matrix = new DiscreteInterpolation1D*[bins::NUMBER_OF_K_BANDS];
}

void Chunk::_findRedshiftBin()
{
    ZBIN     = bins::findRedshiftBin(MEDIAN_REDSHIFT);
    ZBIN_LOW = bins::findRedshiftBin(LOWER_REDSHIFT);
    ZBIN_UPP = bins::findRedshiftBin(UPPER_REDSHIFT);

    // Assign to a redshift bin according to median redshift of this chunk
    // This is just for bookkeeping purposes
    if ((ZBIN >= 0) && (ZBIN < bins::NUMBER_OF_Z_BINS))
        BIN_REDSHIFT = bins::ZBIN_CENTERS[ZBIN];
}

// For a top hat redshift bin, only access 1 redshift bin for each pixel
// For triangular z bins, access 2 redshift bins for each pixel
void Chunk::_setNQandFisherIndex()
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
void Chunk::_setStoredMatrices()
{
    double size_m1 = (double)sizeof(double) * DATA_SIZE_2 / 1048576.; // in MB
    double remain_mem = process::MEMORY_ALLOC;

    // Resolution matrix needs another temp storage.
    if (specifics::USE_RESOLUTION_MATRIX)
        remain_mem -= qFile->Rmat->getBufMemUsage();

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
        LOG::LOGGER.ERR("===============\n""Not all matrices are stored: %s\n"
            "#stored: %d vs #required:%d.\n""ND: %d, M1: %.1f MB. "
            "Avail mem after R & SQ subtracted: %.1lf MB\n""===============\n", 
            qFile->fname.c_str(), nqj_eff, N_Q_MATRICES, qFile->size, size_m1, 
            remain_mem);

    isQjSet   = false;
    isSfidSet = false;
}

Chunk::Chunk(const qio::QSOFile &qmaster, int i1, int i2)
{
    isCovInverted = false;
    _copyQSOFile(qmaster, i1, i2);

    qFile->readMinMaxMedRedshift(LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT);

    _findRedshiftBin();

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(qFile->wave, qFile->delta, qFile->noise, qFile->size);

    // Keep noise as error squared (variance)
    std::for_each(qFile->noise, qFile->noise+qFile->size, [](double &n) { n*=n; });

    nqj_eff = 0;
    
    // Set up number of matrices, index for Fisher matrix
    _setNQandFisherIndex();

    _setStoredMatrices();
    process::updateMemory(-getMinMemUsage());
}

Chunk::Chunk(Chunk &&rhs)
{
    RES_INDEX = rhs.RES_INDEX;
    N_Q_MATRICES = rhs.N_Q_MATRICES;
    fisher_index_start = rhs.fisher_index_start;

    // LOWER_REDSHIFT = rhs.LOWER_REDSHIFT; 
    // UPPER_REDSHIFT = rhs.UPPER_REDSHIFT;
    MEDIAN_REDSHIFT = rhs.MEDIAN_REDSHIFT;
    // BIN_REDSHIFT = rhs.BIN_REDSHIFT;

    stored_qj = std::move(rhs.stored_qj);
    rhs.stored_qj = NULL;
    qFile = std::move(rhs.qFile);
    rhs.qFile = NULL;

    nqj_eff = rhs.nqj_eff;
    isSfidStored = rhs.isSfidStored;

    interp_derivative_matrix = std::move(rhs.interp_derivative_matrix);
    rhs.interp_derivative_matrix = NULL;
}

Chunk::~Chunk()
{
    process::updateMemory(getMinMemUsage());
    delete qFile;

    if (nqj_eff > 0)
        delete [] stored_qj;
    delete [] interp_derivative_matrix;
}

double Chunk::getMinMemUsage()
{
    double minmem = (double)sizeof(double) * qFile->size * 3 / 1048576.; // in MB

    if (specifics::USE_RESOLUTION_MATRIX)
        minmem += qFile->Rmat->getMinMemUsage();

    return minmem;
}

double Chunk::getComputeTimeEst(const qio::QSOFile &qmaster, int i1, int i2)
{
    try
    {
        qio::QSOFile qtemp(qmaster, i1, i2);

        double z1, z2, zm; 
        qtemp.readMinMaxMedRedshift(z1, z2, zm);

        int ZBIN_LOW = bins::findRedshiftBin(z1), ZBIN_UPP = bins::findRedshiftBin(z2);

        if ((ZBIN_LOW > (bins::NUMBER_OF_Z_BINS-1)) || (ZBIN_UPP < 0))
            return 0;

        int N_Q_MATRICES = ZBIN_UPP - ZBIN_LOW + 1;
        double res = std::pow(qtemp.size/100., 3);

        #if defined(TRIANGLE_Z_BINNING_FN)
        // If we need to distribute low end to a lefter bin
        if ((z1 < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0))
            ++N_Q_MATRICES;
        // If we need to distribute high end to righter bin
        if ((bins::ZBIN_CENTERS[ZBIN_UPP] < z2) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
            ++N_Q_MATRICES;
        #endif
        N_Q_MATRICES *= bins::NUMBER_OF_K_BANDS;

        #ifdef FISHER_OPTIMIZATION
        #define N_M_COMBO 3.
        #else
        #define N_M_COMBO (N_Q_MATRICES + 1.)
        #endif

        return res * N_Q_MATRICES * N_M_COMBO;

        #undef N_M_COMBO
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("%s. Skipping %s.\n", e.what(), qmaster.fname.c_str());
        return 0;
    }
}

void Chunk::_setFiducialSignalMatrix(double *&sm, bool copy)
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
        double *inter_mat = (_matrix_n == qFile->size) ? sm : qFile->Rmat->temp_highres_mat;
        double *ptr = inter_mat, *li=highres_lambda;

        for (int row = 0; row < _matrix_n; ++row, ++li)
        {
            ptr += row;

            for (double *lj=li; lj != (highres_lambda+_matrix_n); ++lj, ++ptr)
            {
                _getVandZ(*li, *lj, v_ij, z_ij);

                *ptr = interp2d_signal_matrix->evaluate(z_ij, v_ij);
            }
        }

        mxhelp::copyUpperToLower(inter_mat, _matrix_n);

        if (specifics::USE_RESOLUTION_MATRIX)
            qFile->Rmat->sandwich(sm);
    }
    
    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}

void Chunk::_setQiMatrix(double *&qi, int i_kz, bool copy)
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
        bins::setRedshiftBinningFunction(zm);
        _setL2Limits(zm);

        double *inter_mat = (_matrix_n == qFile->size) ? qi : qFile->Rmat->temp_highres_mat;
        double *ptr = inter_mat, *li=highres_lambda, *lj, *lstart, *lend, 
            *highres_l_end = highres_lambda+_matrix_n;
        DiscreteInterpolation1D *interp_deriv_kn=interp_derivative_matrix[kn];

        for (int row = 0; row < _matrix_n; ++row, ++li)
        {
            ptr += row;
            lstart = std::lower_bound(li, highres_l_end, _L2MIN/(*li));
            lend   = std::upper_bound(li, highres_l_end, _L2MAX/(*li));

            // printf("i: %d \t %ld - %ld \n", row, lstart-highres_lambda, lend-highres_lambda);

            for (lj = li; lj != lstart; ++lj, ++ptr)
                *ptr = 0;

            for (; lj != lend; ++lj, ++ptr)
            {
                _getVandZ(*li, *lj, v_ij, z_ij);

                *ptr  = interp_deriv_kn->evaluate(v_ij);
                *ptr *= bins::redshiftBinningFunction(z_ij, zm);
                // Every pixel pair should scale to the bin redshift
                #ifdef REDSHIFT_GROWTH_POWER
                *ptr *= fidcosmo::fiducialPowerGrowthFactor(z_ij, 
                    bins::KBAND_CENTERS[kn], bins::ZBIN_CENTERS[zm], 
                    &fidpd13::FIDUCIAL_PD13_PARAMS);
                #endif
            }

            for (; lj != highres_l_end; ++lj, ++ptr)
                *ptr = 0;
        }

        t_interp = mytime::timer.getTime() - t;

        mxhelp::copyUpperToLower(inter_mat, _matrix_n);

        if (specifics::USE_RESOLUTION_MATRIX)
            qFile->Rmat->sandwich(qi);
    }

    t = mytime::timer.getTime() - t; 

    mytime::time_spent_set_qs += t;
    mytime::time_spent_on_q_interp += t_interp;
    mytime::time_spent_on_q_copy += t - t_interp;
}

void Chunk::setCovarianceMatrix(const double *ps_estimate)
{
    // Set fiducial signal matrix
    if (!specifics::TURN_OFF_SFID)
        _setFiducialSignalMatrix(covariance_matrix);
    else
        std::fill_n(covariance_matrix, DATA_SIZE_2, 0);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        _setQiMatrix(temp_matrix[0], i_kz);

        cblas_daxpy(DATA_SIZE_2, ps_estimate[i_kz + fisher_index_start], 
            temp_matrix[0], 1, covariance_matrix, 1);
    }

    // add noise matrix diagonally
    // but smooth before adding
    double *smooth_noise = new double[qFile->size];
    Smoother::smoothNoise(qFile->noise, smooth_noise, qFile->size);
    cblas_daxpy(qFile->size, 1., smooth_noise, 1, covariance_matrix, qFile->size+1);
    delete [] smooth_noise;
    
    isCovInverted = false;
}

void Chunk::_addMarginalizations()
{
    double *temp_v = temp_matrix[0], *temp_y = temp_matrix[1];
    double norm;

    // Zeroth order
    std::fill_n(temp_v, qFile->size, 1);
    cblas_dsymv(CblasRowMajor, CblasUpper, qFile->size, 1., inverse_covariance_matrix, 
        qFile->size, temp_v, 1, 0, temp_y, 1);
    norm = cblas_ddot(qFile->size, temp_v, 1, temp_y, 1);

    cblas_dger(CblasRowMajor, qFile->size, qFile->size, -1./norm, temp_y, 1, 
        temp_y, 1, inverse_covariance_matrix, qFile->size);

    // Higher orders
    for (int cmo = 1; cmo <= specifics::CONT_MARG_ORDER; ++cmo)
    {
        std::transform(qFile->wave, qFile->wave+qFile->size, temp_v, [cmo](const double &l) 
            { return pow(log(l/LYA_REST), cmo); });

        cblas_dsymv(CblasRowMajor, CblasUpper, qFile->size, 1., inverse_covariance_matrix, 
            qFile->size, temp_v, 1, 0, temp_y, 1);
        norm = cblas_ddot(qFile->size, temp_v, 1, temp_y, 1);

        cblas_dger(CblasRowMajor, qFile->size, qFile->size, -1./norm, temp_y, 1, 
            temp_y, 1, inverse_covariance_matrix, qFile->size);
    }
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    double t = mytime::timer.getTime();

    mxhelp::LAPACKE_InvertMatrixLU(covariance_matrix, qFile->size);

    inverse_covariance_matrix = covariance_matrix;

    isCovInverted = true;

    if (specifics::CONT_MARG_ORDER >= 0)
        _addMarginalizations();

    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_c_inv += t;
}

void Chunk::_getWeightedMatrix(double *m)
{
    double t = mytime::timer.getTime();

    //C-1 . Q
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
        qFile->size, qFile->size, 1., inverse_covariance_matrix, qFile->size,
        m, qFile->size, 0, temp_matrix[1], qFile->size);

    //C-1 . Q . C-1
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper,
        qFile->size, qFile->size, 1., inverse_covariance_matrix, qFile->size,
        temp_matrix[1], qFile->size, 0, m, qFile->size);

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_modqs += t;
}

void Chunk::_getFisherMatrix(const double *Qw_ikz_matrix, int i_kz)
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

        temp = 0.5 * mxhelp::trace_dsymm(Qw_ikz_matrix, Q_jkz_matrix, qFile->size);

        int ind_ij = (i_kz + fisher_index_start) 
                + bins::TOTAL_KZ_BINS * (j_kz + fisher_index_start),
            ind_ji = (j_kz + fisher_index_start) 
                + bins::TOTAL_KZ_BINS * (i_kz + fisher_index_start);

        *(fisher_matrix + ind_ij) = temp;
        *(fisher_matrix + ind_ji) = temp;
    }

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_fisher += t;
}

void Chunk::computePSbeforeFvector()
{
    double *weighted_data_vector = new double[qFile->size];

    cblas_dsymv(CblasRowMajor, CblasUpper, qFile->size, 1., inverse_covariance_matrix, 
        qFile->size, qFile->delta, 1, 0, weighted_data_vector, 1);

    // #pragma omp parallel for
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        double *Q_ikz_matrix = temp_matrix[0], *Sfid_matrix = temp_matrix[1], 
            temp_tk = 0;

        // Set derivative matrix ikz
        _setQiMatrix(Q_ikz_matrix, i_kz);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        double temp_dk = mxhelp::my_cblas_dsymvdot(weighted_data_vector, 
            Q_ikz_matrix, qFile->size);

        // Get weighted derivative matrix ikz: C-1 Qi C-1
        _getWeightedMatrix(Q_ikz_matrix);

        // Get Noise contribution: Tr(C-1 Qi C-1 N)
        double temp_bk = mxhelp::trace_ddiagmv(Q_ikz_matrix, qFile->noise, 
            qFile->size);

        // Set Fiducial Signal Matrix
        if (!specifics::TURN_OFF_SFID)
        {
            _setFiducialSignalMatrix(Sfid_matrix, false);

            // Tr(C-1 Qi C-1 Sfid)
            temp_tk = mxhelp::trace_dsymm(Q_ikz_matrix, Sfid_matrix, qFile->size);
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

void Chunk::oneQSOiteration(const double *ps_estimate, 
    double *dbt_sum_vector[3], double *fisher_sum)
{
    _allocateMatrices();

    // This function allocates new signal & deriv matrices 
    // if process::SAVE_ALL_SQ_FILES=false 
    // i.e., no caching of SQ files
    // If all tables are cached, then this function simply points 
    // to those in process:sq_private_table
    process::sq_private_table->readSQforR(RES_INDEX, interp2d_signal_matrix, 
        interp_derivative_matrix);

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

        mxhelp::vector_add(fisher_sum, fisher_matrix, FISHER_SIZE);
        
        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            mxhelp::vector_add(dbt_sum_vector[dbt_i], 
                dbt_estimate_before_fisher_vector[dbt_i], bins::TOTAL_KZ_BINS);

        // // Write results to file with their qso filename as base
        // if (process::SAVE_EACH_SPEC_RESULT)
        //     _saveIndividualResult();
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR("ERROR %s: Covariance matrix is not invertable. %s\n",
            e.what(), qFile->fname.c_str());

        LOG::LOGGER.ERR("Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
            qFile->size, MEDIAN_REDSHIFT, qFile->dv_kms, qFile->R_fwhm);
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

void Chunk::_allocateMatrices()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector[dbt_i] = new double[bins::TOTAL_KZ_BINS]();

    fisher_matrix = new double[FISHER_SIZE]();

    covariance_matrix = new double[DATA_SIZE_2];

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = new double[DATA_SIZE_2];
    
    for (int i = 0; i < nqj_eff; ++i)
        stored_qj[i] = new double[DATA_SIZE_2];
    
    if (isSfidStored)
        stored_sfid = new double[DATA_SIZE_2];

    // Create a temp highres lambda array
    if (specifics::USE_RESOLUTION_MATRIX && !qFile->Rmat->isDiaMatrix())
    {
        highres_lambda = qFile->Rmat->allocWaveGrid(qFile->wave[0]);
        qFile->Rmat->allocateTempHighRes();
    }
    else
        highres_lambda = qFile->wave;

    isQjSet   = false;
    isSfidSet = false;
}

void Chunk::_freeMatrices()
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

    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile->Rmat->freeBuffers();
        if (!qFile->Rmat->isDiaMatrix())
            delete [] highres_lambda;
    }

    isQjSet   = false;
    isSfidSet = false;
}

// void Chunk::_saveIndividualResult()
// {
//     mxhelp::vector_sub(dbt_estimate_before_fisher_vector[0], 
//         dbt_estimate_before_fisher_vector[1], bins::TOTAL_KZ_BINS);
//     mxhelp::vector_sub(dbt_estimate_before_fisher_vector[0], 
//         dbt_estimate_before_fisher_vector[2], bins::TOTAL_KZ_BINS);

//     try
//     {
//         ioh::boot_saver->writeBoot(qFile->id, 
//             dbt_estimate_before_fisher_vector[0], fisher_matrix);
//     }
//     catch (std::exception& e)
//     {
//         LOG::LOGGER.ERR("ERROR: Saving individual results: %s\n", 
//             qFile->fname.c_str());
//     }
// }

void Chunk::fprintfMatrices(const char *fname_base)
{
    char buf[1024];

    if (isSfidStored)
    {
        sprintf(buf, "%s-signal.txt", fname_base);
        mxhelp::fprintfMatrix(buf, stored_sfid, qFile->size, qFile->size);
    }

    if (qFile->Rmat != NULL)
    {
        sprintf(buf, "%s-resolution.txt", fname_base);
        qFile->Rmat->fprintfMatrix(buf);
    }
}

#undef DATA_SIZE_2





