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

double _L2MAX, _L2MIN;
inline
void _setL2Limits(int zm)
{
    #if defined(TOPHAT_Z_BINNING_FN)
    const double
    ZSTART = bins::ZBIN_CENTERS[zm] - bins::Z_BIN_WIDTH/2,
    ZEND   = bins::ZBIN_CENTERS[zm] + bins::Z_BIN_WIDTH/2;
    #elif defined(TRIANGLE_Z_BINNING_FN)
    const double
    ZSTART = bins::ZBIN_CENTERS[zm] - bins::Z_BIN_WIDTH,
    ZEND   = bins::ZBIN_CENTERS[zm] + bins::Z_BIN_WIDTH;
    #endif
    _L2MAX  = (1 + ZEND) * LYA_REST;
    _L2MAX *=_L2MAX;
    _L2MIN  = (1 + ZSTART) * LYA_REST;
    _L2MIN *=_L2MIN;
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
    isCovInverted = false;
    _copyQSOFile(qmaster, i1, i2);

    qFile->readMinMaxMedRedshift(LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT);

    _findRedshiftBin();

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(qFile->wave(), qFile->delta(), qFile->noise(), size());

    // Keep noise as error squared (variance)
    std::for_each(qFile->noise(), qFile->noise()+size(), [](double &n) { n*=n; });

    // Set up number of matrices, index for Fisher matrix
    _setNQandFisherIndex();

    i_kz_vector.reserve(N_Q_MATRICES);
    int _kncut = _getMaxKindex(MY_PI / qFile->dv_kms);
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
        int kn, zm;
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        if (kn < _kncut)
            i_kz_vector.push_back(i_kz);
    }

    _setStoredMatrices();

    interp_derivative_matrix.reserve(bins::NUMBER_OF_K_BANDS);

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
        _matrix_n = size();
    }

    DATA_SIZE_2 = size()*size();
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
    double remain_mem = process::MEMORY_ALLOC,
           needed_mem = (3+i_kz_vector.size()+(!specifics::TURN_OFF_SFID))*size_m1;

    // Resolution matrix needs another temp storage.
    if (specifics::USE_RESOLUTION_MATRIX)
        remain_mem -= qFile->Rmat->getBufMemUsage();

    // Need at least 3 matrices as temp one for sfid
    if (remain_mem < needed_mem) {
        LOG::LOGGER.ERR("===============\n""Not all matrices are stored: %s\n"
            "#required:%d.\n""ND: %d, M1: %.1f MB. "
            "Avail mem after R & SQ subtracted: %.1lf MB\n""===============\n", 
            qFile->fname.c_str(), i_kz_vector.size(), size(), size_m1, 
            remain_mem);
        throw std::runtime_error("Not all matrices are stored.\n");
    }
}

Chunk::~Chunk()
{
    process::updateMemory(getMinMemUsage());
}

double Chunk::getMinMemUsage()
{
    double minmem = (double)sizeof(double) * size() * 3 / 1048576.; // in MB

    if (specifics::USE_RESOLUTION_MATRIX)
        minmem += qFile->Rmat->getMinMemUsage();

    return minmem;
}

double Chunk::getComputeTimeEst(const qio::QSOFile &qmaster, int i1, int i2)
{
    try
    {
        qio::QSOFile qtemp(qmaster, i1, i2);
        if (qtemp.realSize() < MIN_PIXELS_IN_CHUNK)
            return 0;

        double z1, z2, zm; 
        qtemp.readMinMaxMedRedshift(z1, z2, zm);

        int ZBIN_LOW = bins::findRedshiftBin(z1), ZBIN_UPP = bins::findRedshiftBin(z2);

        if ((ZBIN_LOW > (bins::NUMBER_OF_Z_BINS-1)) || (ZBIN_UPP < 0))
            return 0;

        int N_Q_MATRICES = ZBIN_UPP - ZBIN_LOW + 1;
        double res = std::pow(qtemp.size()/100., 3);

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
        const int N_M_COMBO = 3;
        #else
        const int N_M_COMBO = N_Q_MATRICES + 1;
        #endif

        return res * N_Q_MATRICES * N_M_COMBO;
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

    double *inter_mat = (_finer_matrix != NULL) ? _finer_matrix : sm;
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
    
    t = mytime::timer.getTime() - t;

    mytime::time_spent_on_set_sfid += t;
}

void Chunk::_setQiMatrix(double *qi, int i_kz)
{
    ++mytime::number_of_times_called_setq;
    double t = mytime::timer.getTime(), t_interp;
    int kn, zm;
    double v_ij, z_ij;

    bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);
    bins::setRedshiftBinningFunction(zm);
    _setL2Limits(zm);

    double *inter_mat = (_finer_matrix != NULL) ? _finer_matrix : qi;
    double *ptr = inter_mat, *li = _matrix_lambda, 
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
        std::copy(stored_sfid, stored_sfid + DATA_SIZE_2, covariance_matrix);
    else
        std::fill_n(covariance_matrix, DATA_SIZE_2, 0);

    const double *alpha = ps_estimate + fisher_index_start;
    for (int idx = 0; idx < i_kz_vector.size(); ++idx) {
        int i_kz = i_kz_vector[idx];
        double *Q_ikz_matrix = _getStoredQikz(idx);

        cblas_daxpy(DATA_SIZE_2, alpha[i_kz], 
            Q_ikz_matrix, 1, covariance_matrix, 1);
    }

    // add noise matrix diagonally
    // but smooth before adding
    process::noise_smoother->smoothNoise(qFile->noise(), temp_vector, size());
    cblas_daxpy(size(), 1., temp_vector, 1,
        covariance_matrix, size()+1);

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
    double *temp_v = temp_matrix[0], *temp_y = temp_matrix[1];

    // Zeroth order
    std::fill_n(temp_v, size(), 1./sqrt(size()));
    temp_v += size();
    // Log lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LOGLAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLogLam(qFile->wave(), size(), cmo, temp_v);
        temp_v += size();
    }
    // Lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLam(qFile->wave(), size(), cmo, temp_v);
        temp_v += size();
    }

    LOG::LOGGER.DEB("nvecs %d\n", specifics::CONT_NVECS);

    // Roll back to initial position
    temp_v = temp_matrix[0];
    static auto svals = std::make_unique<double[]>(specifics::CONT_NVECS);
    // SVD to get orthogonal marg vectors
    mxhelp::LAPACKE_svd(temp_v, svals.get(), size(), specifics::CONT_NVECS);
    LOG::LOGGER.DEB("SVD'ed\n");

    // Remove each 
    for (int i = 0; i < specifics::CONT_NVECS; ++i, temp_v += size())
    {
        LOG::LOGGER.DEB("i: %d, s: %.2e\n", i, svals[i]);
        // skip if this vector is degenerate
        if (svals[i]<1e-6)  continue;

        _remShermanMorrison(temp_v, size(), temp_y, inverse_covariance_matrix);
    }
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    double t = mytime::timer.getTime();

    mxhelp::LAPACKE_InvertMatrixLU(covariance_matrix, size());

    inverse_covariance_matrix = covariance_matrix;

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
        size(), size(), 1., inverse_covariance_matrix, size(),
        m, size(), 0, temp_matrix[1], size());

    //C-1 . Q . C-1
    // std::fill_n(m, DATA_SIZE_2, 0);
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper,
        size(), size(), 1., inverse_covariance_matrix, size(),
        temp_matrix[1], size(), 0, m, size());

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_modqs += t;
}

void Chunk::_getFisherMatrix(const double *Qw_ikz_matrix, int idx)
{
    double temp, *Q_jkz_matrix, t = mytime::timer.getTime();
    int i_kz = i_kz_vector[idx];

    // Now compute Fisher Matrix
    for (int jdx = idx; jdx < i_kz_vector.size(); ++jdx) {
        int j_kz = i_kz_vector[jdx];
        #ifdef FISHER_OPTIMIZATION
        int diff_ji = j_kz - i_kz;

        if ((diff_ji != 0) && (diff_ji != 1) && (diff_ji != bins::NUMBER_OF_K_BANDS))
            continue;
        #endif

        double *Q_jkz_matrix = _getStoredQikz(jdx);

        temp = 0.5 * mxhelp::trace_dsymm(Qw_ikz_matrix, Q_jkz_matrix, size());

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
    double *Q_ikz_matrix = temp_matrix[0];
    std::vector<double> dbt_vec(3, 0);

    LOG::LOGGER.DEB("PSb4F -> weighted data\n");
    cblas_dsymv(CblasRowMajor, CblasUpper, size(), 1.,
        inverse_covariance_matrix, 
        size(), qFile->delta(), 1, 0, weighted_data_vector, 1);

    for (int idx = 0; idx < i_kz_vector.size(); ++idx) {
        int i_kz = i_kz_vector[idx];
        LOG::LOGGER.DEB("PSb4F -> loop %d\n", i_kz);

        LOG::LOGGER.DEB("   -> set qi   ");
        // Set derivative matrix ikz
        double *ptr = _getStoredQikz(idx);
        std::copy(ptr, ptr + DATA_SIZE_2, Q_ikz_matrix);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        dbt_vec[0] = mxhelp::my_cblas_dsymvdot(weighted_data_vector, 
            Q_ikz_matrix, temp_vector, size());
         LOG::LOGGER.DEB("-> dk (%.1e)   ", dbt_vec[0]);

        LOG::LOGGER.DEB("-> weighted Q   ");
        // Get weighted derivative matrix ikz: C-1 Qi C-1
        _getWeightedMatrix(Q_ikz_matrix);

        LOG::LOGGER.DEB("-> nk   ");
        // Get Noise contribution: Tr(C-1 Qi C-1 N)
        dbt_vec[1] = mxhelp::trace_ddiagmv(Q_ikz_matrix, qFile->noise(), 
            size());

        // Set Fiducial Signal Matrix
        if (!specifics::TURN_OFF_SFID)
        {
            LOG::LOGGER.DEB("-> tk   ");
            // Tr(C-1 Qi C-1 Sfid)
            dbt_vec[2] = mxhelp::trace_dsymm(Q_ikz_matrix, stored_sfid, size());
        }

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            dbt_estimate_before_fisher_vector[dbt_i][i_kz + fisher_index_start] = dbt_vec[dbt_i];

        // Do not compute fisher matrix if it is precomputed
        if (!specifics::USE_PRECOMPUTED_FISHER)
            _getFisherMatrix(Q_ikz_matrix, idx);

        LOG::LOGGER.DEB("\n");
    }
}

void Chunk::oneQSOiteration(const double *ps_estimate, 
    std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
    double *fisher_sum)
{
    LOG::LOGGER.DEB("File %s\n", qFile->fname.c_str());
    LOG::LOGGER.DEB("TargetID %ld\n", qFile->id);
    LOG::LOGGER.DEB("Size %d\n", size());
    LOG::LOGGER.DEB("ncols: %d\n", _matrix_n);
    LOG::LOGGER.DEB("fisher_index_start: %d\n", fisher_index_start);
    LOG::LOGGER.DEB("Allocating matrices\n");

    _allocateMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    // i_kz = N_Q_MATRICES - j_kz - 1
    LOG::LOGGER.DEB("Setting qi matrices\n");

    for (int jdx = 0; jdx < i_kz_vector.size(); ++jdx) {
        int j_kz = i_kz_vector[jdx];
        _setQiMatrix(stored_qj + jdx*DATA_SIZE_2, N_Q_MATRICES-j_kz-1);
    }

    // Preload fiducial signal matrix if memory allows
    if (!specifics::TURN_OFF_SFID)
        _setFiducialSignalMatrix(stored_sfid);

    LOG::LOGGER.DEB("Setting cov matrix\n");

    setCovarianceMatrix(ps_estimate);
    _check_isnan(covariance_matrix, DATA_SIZE_2, "NaN: covariance");

    try
    {
        LOG::LOGGER.DEB("Inverting cov matrix\n");
        invertCovarianceMatrix();
        _check_isnan(inverse_covariance_matrix, DATA_SIZE_2,
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
            size(), MEDIAN_REDSHIFT, qFile->dv_kms, qFile->R_fwhm);
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

    covariance_matrix = new double[DATA_SIZE_2];

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = new double[DATA_SIZE_2];

    temp_vector = new double[size()];
    weighted_data_vector = new double[size()];
    
    stored_qj = new double[i_kz_vector.size()*DATA_SIZE_2];

    if (!specifics::TURN_OFF_SFID)
        stored_sfid = new double[DATA_SIZE_2];

    // Create a temp highres lambda array
    if (specifics::USE_RESOLUTION_MATRIX && !qFile->Rmat->isDiaMatrix())
    {
        int ncols = qFile->Rmat->getNCols();
        _finer_lambda = new double[ncols];

        double fine_dlambda = qFile->dlambda/specifics::OVERSAMPLING_FACTOR;
        int disp = qFile->Rmat->getNElemPerRow()/2;
        for (int i = 0; i < ncols; ++i)
            _finer_lambda[i] = qFile->wave()[0] + (i - disp)*fine_dlambda;

        _matrix_lambda = _finer_lambda;

        long highsize = (long)(ncols) * (long)(ncols);
        _finer_matrix = new double[highsize];
    }
    else
    {
        _matrix_lambda = qFile->wave();
        _finer_lambda  = NULL;
        _finer_matrix  = NULL;
    }

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
    delete [] covariance_matrix;

    LOG::LOGGER.DEB("Free temps\n");
    for (int i = 0; i < 2; ++i)
        delete [] temp_matrix[i];

    delete [] temp_vector;
    delete [] weighted_data_vector;

    LOG::LOGGER.DEB("Free storedqj\n");
    delete [] stored_qj;

    LOG::LOGGER.DEB("Free stored sfid\n");
    if (!specifics::TURN_OFF_SFID)
        delete [] stored_sfid;

    LOG::LOGGER.DEB("Free resomat related\n");
    if (specifics::USE_RESOLUTION_MATRIX)
    {
        qFile->Rmat->freeBuffer();
        if (!qFile->Rmat->isDiaMatrix())
        {
            delete [] _finer_matrix;
            delete [] _finer_lambda;
        }
    }

    if (interp2d_signal_matrix)
        interp2d_signal_matrix.reset();
    interp_derivative_matrix.clear();
}

void Chunk::fprintfMatrices(const char *fname_base)
{
    char buf[1024];

    if (!specifics::TURN_OFF_SFID)
    {
        sprintf(buf, "%s-signal.txt", fname_base);
        mxhelp::fprintfMatrix(buf, stored_sfid, size(), size());
    }

    if (qFile->Rmat != NULL)
    {
        sprintf(buf, "%s-resolution.txt", fname_base);
        qFile->Rmat->fprintfMatrix(buf);
    }
}






