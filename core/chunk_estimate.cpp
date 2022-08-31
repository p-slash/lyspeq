#include "core/chunk_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/sq_table.hpp"

#include "mathtools/smoother.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

void _check_isnan(double *mat, int size, std::string msg)
{
    #ifdef CHECK_NAN
    if (std::any_of(mat, mat+size, [](double x) {return std::isnan(x);}))
        throw std::runtime_error(msg);
    #else
        double *tmat __attribute__((unused)) = mat;
        int tsize __attribute__((unused)) = size;
        std::string tmsg __attribute__((unused)) = msg;
    #endif
}

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

inline
int _getMaxKindex(double knyq)
{
    auto it = std::lower_bound(bins::KBAND_CENTERS.begin(), bins::KBAND_CENTERS.end(), knyq);
    return std::distance(bins::KBAND_CENTERS.begin(), it);
}

Chunk::Chunk(const qio::QSOFile &qmaster, int i1, int i2)
{
    temp_matrix.reserve(2);
    dbt_estimate_before_fisher_vector.reserve(3);
    isCovInverted = false;
    _copyQSOFile(qmaster, i1, i2);

    qFile->readMinMaxMedRedshift(LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT);

    _kncut = _getMaxKindex(PI / qFile->dv_kms);

    _findRedshiftBin();

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(qFile->wave(), qFile->delta(), qFile->noise(), qFile->size);

    // Keep noise as error squared (variance)
    LOG::LOGGER.DEB("qFile->noise()[10]: %.5f --> ", qFile->noise()[10]);
    std::for_each(qFile->noise(), qFile->noise()+qFile->size, [](double &n) { n*=n; });
    LOG::LOGGER.DEB("%.5f\n", qFile->noise()[10]);

    nqj_eff = 0;

    // Set up number of matrices, index for Fisher matrix
    _setNQandFisherIndex();

    _setStoredMatrices();
    process::updateMemory(-getMinMemUsage());
}

void Chunk::_copyQSOFile(const qio::QSOFile &qmaster, int i1, int i2)
{
    qFile = std::make_unique<qio::QSOFile>(qmaster, i1, i2);

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

    interp_derivative_matrix.reserve(bins::NUMBER_OF_K_BANDS);
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
    else              stored_qj.reserve(nqj_eff);

    if (nqj_eff != N_Q_MATRICES)
        LOG::LOGGER.ERR("===============\n""Not all matrices are stored: %s\n"
            "#stored: %d vs #required:%d.\n""ND: %d, M1: %.1f MB. "
            "Avail mem after R & SQ subtracted: %.1lf MB\n""===============\n", 
            qFile->fname.c_str(), nqj_eff, N_Q_MATRICES, qFile->size, size_m1, 
            remain_mem);

    isQjSet   = false;
    isSfidSet = false;
}

bool Chunk::_isAboveNyquist(int i_kz)
{
    int kn, zm;
    bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);
    return kn > _kncut;
}

Chunk::~Chunk()
{
    process::updateMemory(getMinMemUsage());
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

void Chunk::_setFiducialSignalMatrix(double *sm)
{
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::timer.getTime();
    double v_ij, z_ij;

    if (isSfidSet)
    {
        std::copy(stored_sfid.get(), stored_sfid.get() + DATA_SIZE_2, sm);
    }
    else
    {
        double *inter_mat = (_finer_matrix) ? _finer_matrix.get() : sm;
        double *ptr = inter_mat, *li=_matrix_lambda;

        for (int row = 0; row < _matrix_n; ++row, ++li)
        {
            ptr += row;

            for (double *lj=li; lj != (_matrix_lambda+_matrix_n); ++lj, ++ptr)
            {
                _getVandZ(*li, *lj, v_ij, z_ij);

                *ptr = interp2d_signal_matrix->evaluate(z_ij, v_ij);
            }
        }

        mxhelp::copyUpperToLower(inter_mat, _matrix_n);

        if (specifics::USE_RESOLUTION_MATRIX)
            qFile->Rmat->sandwich(sm, inter_mat);
    }
    
    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}

void Chunk::_setQiMatrix(double *qi, int i_kz)
{
    ++mytime::number_of_times_called_setq;
    double t = mytime::timer.getTime(), t_interp;
    int kn, zm;
    double v_ij, z_ij;

    if (_isQikzStored(i_kz))
    {
        t_interp = 0;
        double *ptr = _getStoredQikz(i_kz);
        std::copy(ptr, ptr + DATA_SIZE_2, qi);
    }
    else
    {
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);
        bins::setRedshiftBinningFunction(zm);
        _setL2Limits(zm);

        double *inter_mat = (_finer_matrix) ? _finer_matrix.get() : qi;
        double *ptr = inter_mat, *li=_matrix_lambda, 
            *highres_l_end = _matrix_lambda + _matrix_n;

        shared_interp_1d interp_deriv_kn = interp_derivative_matrix[kn];

        for (int row = 0; row < _matrix_n; ++row, ++li)
        {
            ptr += row;
            double *lstart = std::lower_bound(li, highres_l_end, _L2MIN/(*li));
            double *lend   = std::upper_bound(li, highres_l_end, _L2MAX/(*li));

            // printf("i: %d \t %ld - %ld \n", row, lstart-highres_lambda, lend-highres_lambda);
            double *lj;
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
            qFile->Rmat->sandwich(qi, inter_mat);
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
        _setFiducialSignalMatrix(covariance_matrix.get());
    else
        std::fill_n(covariance_matrix.get(), DATA_SIZE_2, 0);

    double *Q_ikz_matrix;
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        if (_isAboveNyquist(i_kz)) continue;

        if (_isQikzStored(i_kz))
            Q_ikz_matrix = _getStoredQikz(i_kz);
        else
        {
            Q_ikz_matrix = temp_matrix[0].get();
            _setQiMatrix(Q_ikz_matrix, i_kz);
        }

        cblas_daxpy(DATA_SIZE_2, ps_estimate[i_kz + fisher_index_start], 
            Q_ikz_matrix, 1, covariance_matrix.get(), 1);
    }

    // add noise matrix diagonally
    // but smooth before adding
    Smoother::smoothNoise(qFile->noise(), temp_vector.get(), qFile->size);
    cblas_daxpy(qFile->size, 1., temp_vector.get(), 1,
        covariance_matrix.get(), qFile->size+1);

    isCovInverted = false;

    // When compiled with debugging feature
    // save matrices to files, break
    // #ifdef DEBUG_MATRIX_OUT
    // it->fprintfMatrices(fname_base);
    // throw std::runtime_error("DEBUGGING QUIT.");
    // #endif
}

void _getUnitVectorLogLam(const double *w, int size, int cmo, double *out)
{
    std::transform(w, w+size, out, [cmo](const double &l) { return pow(log(l/LYA_REST), cmo); });
    double norm = sqrt(cblas_dnrm2(size, out, 1));
    cblas_dscal(size, 1./norm, out, 1);
}

void _getUnitVectorLam(const double *w, int size, int cmo, double *out)
{
    std::transform(w, w+size, out, [cmo](const double &l) { return pow(l/LYA_REST, cmo); });
    double norm = sqrt(cblas_dnrm2(size, out, 1));
    cblas_dscal(size, 1./norm, out, 1);
}

void _remShermanMorrison(const double *v, int size, double *y, double *cinv)
{
    std::fill_n(y, size, 0);
    cblas_dsymv(CblasRowMajor, CblasUpper, size, 1., cinv, size, v, 1, 0, y, 1);
    double norm = cblas_ddot(size, v, 1, y, 1);
    cblas_dger(CblasRowMajor, size, size, -1./norm, y, 1, y, 1, cinv, size);
}

void Chunk::_addMarginalizations()
{
    double *temp_v = temp_matrix[0].get(), *temp_y = temp_matrix[1].get();

    // Zeroth order
    std::fill_n(temp_v, qFile->size, 1./sqrt(qFile->size));
    temp_v += qFile->size;
    // Log lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LOGLAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLogLam(qFile->wave(), qFile->size, cmo, temp_v);
        temp_v += qFile->size;
    }
    // Lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLam(qFile->wave(), qFile->size, cmo, temp_v);
        temp_v += qFile->size;
    }

    LOG::LOGGER.DEB("nvecs %d\n", specifics::CONT_NVECS);

    // Roll back to initial position
    temp_v = temp_matrix[0].get();
    static auto svals = std::make_unique<double[]>(specifics::CONT_NVECS);
    // SVD to get orthogonal marg vectors
    mxhelp::LAPACKE_svd(temp_v, svals.get(), qFile->size, specifics::CONT_NVECS);
    LOG::LOGGER.DEB("SVD'ed\n");

    // Remove each 
    for (int i = 0; i < specifics::CONT_NVECS; ++i, temp_v += qFile->size)
    {
        LOG::LOGGER.DEB("i: %d, s: %.2e\n", i, svals[i]);
        // skip if this vector is degenerate
        if (svals[i]<1e-6)  continue;

        _remShermanMorrison(temp_v, qFile->size, temp_y, inverse_covariance_matrix.get());
    }
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    double t = mytime::timer.getTime();

    mxhelp::LAPACKE_InvertMatrixLU(covariance_matrix.get(), qFile->size);

    inverse_covariance_matrix = std::move(covariance_matrix);

    isCovInverted = true;

    if (specifics::CONT_NVECS > 0)
        _addMarginalizations();

    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_c_inv += t;
}

void Chunk::_getWeightedMatrix(double *m)
{
    double t = mytime::timer.getTime();

    //C-1 . Q
    // std::fill_n(temp_matrix[1], DATA_SIZE_2, 0);
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
        qFile->size, qFile->size, 1., inverse_covariance_matrix.get(), qFile->size,
        m, qFile->size, 0, temp_matrix[1].get(), qFile->size);

    //C-1 . Q . C-1
    // std::fill_n(m, DATA_SIZE_2, 0);
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper,
        qFile->size, qFile->size, 1., inverse_covariance_matrix.get(), qFile->size,
        temp_matrix[1].get(), qFile->size, 0, m, qFile->size);

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_modqs += t;
}

void Chunk::_getFisherMatrix(const double *Qw_ikz_matrix, int i_kz)
{
    double temp, *Q_jkz_matrix, t = mytime::timer.getTime();
    
    // Now compute Fisher Matrix
    for (int j_kz = i_kz; j_kz < N_Q_MATRICES; ++j_kz)
    {
        if (_isAboveNyquist(j_kz)) continue;

        #ifdef FISHER_OPTIMIZATION
        int diff_ji = j_kz - i_kz;

        if ((diff_ji != 0) && (diff_ji != 1) && (diff_ji != bins::NUMBER_OF_K_BANDS))
            continue;
        #endif

        if (_isQikzStored(j_kz))
            Q_jkz_matrix = _getStoredQikz(j_kz);
        else
        {
            Q_jkz_matrix = temp_matrix[1].get();
            _setQiMatrix(Q_jkz_matrix, j_kz);
        }

        temp = 0.5 * mxhelp::trace_dsymm(Qw_ikz_matrix, Q_jkz_matrix, qFile->size);

        int ind_ij = (i_kz + fisher_index_start) 
                + bins::TOTAL_KZ_BINS * (j_kz + fisher_index_start),
            ind_ji = (j_kz + fisher_index_start) 
                + bins::TOTAL_KZ_BINS * (i_kz + fisher_index_start);

        fisher_matrix[ind_ij] = temp;
        fisher_matrix[ind_ji] = temp;
    }

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_fisher += t;
}

void Chunk::computePSbeforeFvector()
{
    double *Q_ikz_matrix = temp_matrix[0].get(), *Sfid_matrix, temp_tk = 0;

    if (isSfidSet)
        Sfid_matrix = stored_sfid.get();
    else
        Sfid_matrix = temp_matrix[1].get();

    LOG::LOGGER.DEB("PSb4F -> weighted data\n");
    cblas_dsymv(CblasRowMajor, CblasUpper, qFile->size, 1.,
        inverse_covariance_matrix.get(), 
        qFile->size, qFile->delta(), 1, 0, weighted_data_vector.get(), 1);

    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz)
    {
        LOG::LOGGER.DEB("PSb4F -> loop %d\n", i_kz);
        if (_isAboveNyquist(i_kz)) continue;

        LOG::LOGGER.DEB("   -> set qi   ");
        // Set derivative matrix ikz
        _setQiMatrix(Q_ikz_matrix, i_kz);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        double temp_dk = mxhelp::my_cblas_dsymvdot(weighted_data_vector.get(), 
            Q_ikz_matrix, temp_vector.get(), qFile->size);
         LOG::LOGGER.DEB("-> dk (%.1e)   ", temp_dk);

        LOG::LOGGER.DEB("-> weighted Q   ");
        // Get weighted derivative matrix ikz: C-1 Qi C-1
        _getWeightedMatrix(Q_ikz_matrix);

        LOG::LOGGER.DEB("-> nk   ");
        // Get Noise contribution: Tr(C-1 Qi C-1 N)
        double temp_bk = mxhelp::trace_ddiagmv(Q_ikz_matrix, qFile->noise(), 
            qFile->size);

        // Set Fiducial Signal Matrix
        if (!specifics::TURN_OFF_SFID)
        {
            if (!isSfidSet)
                _setFiducialSignalMatrix(Sfid_matrix);

            LOG::LOGGER.DEB("-> tk   ");
            // Tr(C-1 Qi C-1 Sfid)
            temp_tk = mxhelp::trace_dsymm(Q_ikz_matrix, Sfid_matrix, qFile->size);
        }
        
        dbt_estimate_before_fisher_vector[0][i_kz + fisher_index_start] = temp_dk;
        dbt_estimate_before_fisher_vector[1][i_kz + fisher_index_start] = temp_bk;
        dbt_estimate_before_fisher_vector[2][i_kz + fisher_index_start] = temp_tk;

        // Do not compute fisher matrix if it is precomputed
        if (!specifics::USE_PRECOMPUTED_FISHER)
            _getFisherMatrix(Q_ikz_matrix, i_kz);

        LOG::LOGGER.DEB("\n");
    }
}

void Chunk::oneQSOiteration(const double *ps_estimate, 
    std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
    double *fisher_sum)
{
    LOG::LOGGER.DEB("File %s\n", qFile->fname.c_str());
    LOG::LOGGER.DEB("TargetID %ld\n", qFile->id);
    LOG::LOGGER.DEB("Size %d\n", qFile->size);
    LOG::LOGGER.DEB("ncols: %d\n", _matrix_n);
    LOG::LOGGER.DEB("fisher_index_start: %d\n", fisher_index_start);
    LOG::LOGGER.DEB("Allocating matrices\n");

    _allocateMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    // i_kz = N_Q_MATRICES - j_kz - 1
    LOG::LOGGER.DEB("Setting qi matrices\n");

    for (int j_kz = 0; j_kz < nqj_eff; ++j_kz)
    {
        if (_isAboveNyquist(N_Q_MATRICES - j_kz - 1)) continue;

        _setQiMatrix(stored_qj[j_kz].get(), N_Q_MATRICES - j_kz - 1);
    }

    if (nqj_eff > 0)    isQjSet = true;

    // Preload fiducial signal matrix if memory allows
    if (isSfidStored)
    {
        _setFiducialSignalMatrix(stored_sfid.get());
        isSfidSet = true;
    }

    LOG::LOGGER.DEB("Setting cov matrix\n");

    setCovarianceMatrix(ps_estimate);
    _check_isnan(covariance_matrix.get(), DATA_SIZE_2, "NaN: covariance");

    try
    {
        LOG::LOGGER.DEB("Inverting cov matrix\n");
        invertCovarianceMatrix();
        _check_isnan(inverse_covariance_matrix.get(), DATA_SIZE_2,
            "NaN: inverse cov");

        LOG::LOGGER.DEB("PS before Fisher\n");
        computePSbeforeFvector();

        _check_isnan(fisher_matrix.get(), bins::FISHER_SIZE,
            "NaN: chunk fisher");

        mxhelp::vector_add(fisher_sum, fisher_matrix.get(), bins::FISHER_SIZE);

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            mxhelp::vector_add(dbt_sum_vector[dbt_i].get(), 
                dbt_estimate_before_fisher_vector[dbt_i].get(),
                bins::TOTAL_KZ_BINS);

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

    LOG::LOGGER.DEB("Freeing matrices\n");
    _freeMatrices();
}

void Chunk::_allocateMatrices()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(bins::TOTAL_KZ_BINS));

    fisher_matrix = std::make_unique<double[]>(bins::FISHER_SIZE);

    covariance_matrix = std::make_unique<double[]>(DATA_SIZE_2);

    for (int i = 0; i < 2; ++i)
        temp_matrix.push_back(
            std::make_unique<double[]>(DATA_SIZE_2));

    temp_vector = std::make_unique<double[]>(qFile->size);
    weighted_data_vector = std::make_unique<double[]>(qFile->size);
    
    for (int i = 0; i < nqj_eff; ++i)
        stored_qj.push_back(
            std::make_unique<double[]>(DATA_SIZE_2));

    if (isSfidStored)
        stored_sfid = std::make_unique<double[]>(DATA_SIZE_2);

    // Create a temp highres lambda array
    if (specifics::USE_RESOLUTION_MATRIX && !qFile->Rmat->isDiaMatrix())
    {
        unsigned long highsize = qFile->Rmat->getNCols();
        _finer_lambda   = std::make_unique<double[]>(highsize);

        double fine_dlambda = qFile->dlambda/specifics::OVERSAMPLING_FACTOR;
        int disp = qFile->Rmat->getNElemPerRow()/2;
        for (unsigned long i = 0; i < highsize; ++i)
            _finer_lambda[i] = qFile->wave()[0] + (i - disp)*fine_dlambda;

        _matrix_lambda = _finer_lambda.get();

        highsize *= highsize;
        _finer_matrix = std::make_unique<double[]>(highsize);
    }
    else
        _matrix_lambda = qFile->wave();

    isQjSet   = false;
    isSfidSet = false;

    // This function allocates new signal & deriv matrices 
    // if process::SAVE_ALL_SQ_FILES=false 
    // i.e., no caching of SQ files
    // If all tables are cached, then this function simply points 
    // to those in process:sq_private_table
    process::sq_private_table->readSQforR(RES_INDEX, interp2d_signal_matrix, 
        interp_derivative_matrix);
}

void Chunk::_freeMatrices()
{
    dbt_estimate_before_fisher_vector.clear();

    LOG::LOGGER.DEB("Free fisher\n");
    fisher_matrix.reset();

    LOG::LOGGER.DEB("Free cov\n");
    covariance_matrix.reset();
    inverse_covariance_matrix.reset();

    LOG::LOGGER.DEB("Free temps\n");
    temp_matrix.clear();
    temp_vector.reset();
    weighted_data_vector.reset();

    LOG::LOGGER.DEB("Free storedqj\n");
    stored_qj.clear();

    LOG::LOGGER.DEB("Free stored sfid\n");
    stored_sfid.reset();

    LOG::LOGGER.DEB("Free resomat related\n");
    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->freeBuffers();

    _finer_matrix.reset();
    _finer_lambda.reset();
    _matrix_lambda = NULL;

    isQjSet   = false;
    isSfidSet = false;

    // Do not delete if these are pointers to process::sq_private_table
    if (!process::SAVE_ALL_SQ_FILES)
    {
        if (interp2d_signal_matrix)
            interp2d_signal_matrix.reset();
        interp_derivative_matrix.clear();
    }
}

void Chunk::fprintfMatrices(const char *fname_base)
{
    char buf[1024];

    if (isSfidStored)
    {
        sprintf(buf, "%s-signal.txt", fname_base);
        mxhelp::fprintfMatrix(buf, stored_sfid.get(), qFile->size, qFile->size);
    }

    if (qFile->Rmat != NULL)
    {
        sprintf(buf, "%s-resolution.txt", fname_base);
        qFile->Rmat->fprintfMatrix(buf);
    }
}

#undef DATA_SIZE_2






