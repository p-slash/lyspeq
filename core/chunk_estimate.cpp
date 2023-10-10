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

#if defined(ENABLE_OMP)
#include "omp.h"
#endif

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
    on_oversampling = specifics::USE_RESOLUTION_MATRIX && !qFile->Rmat->isDiaMatrix();

    qFile->readMinMaxMedRedshift(LOWER_REDSHIFT, UPPER_REDSHIFT, MEDIAN_REDSHIFT);

    _findRedshiftBin();

    // Convert flux to fluctuations around the mean flux of the chunk
    // Otherwise assume input data is fluctuations
    conv::convertFluxToDeltaF(qFile->wave(), qFile->delta(), qFile->noise(), size());

    // Keep noise as error squared (variance)
    std::for_each(
        qFile->noise(), qFile->noise() + size(),
        [](double &n) { n *= n; });

    // Divide wave by LYA_REST
    std::for_each(
        qFile->wave(), qFile->wave() + size(),
        [](double &w) { w /= LYA_REST; });

    // Set up number of matrices, index for Fisher matrix
    _setNQandFisherIndex();

    stored_ikz_qi.reserve(N_Q_MATRICES);
    int _kncut = _getMaxKindex(MY_PI / qFile->dv_kms);
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
        int kn, zm;
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        if (kn < _kncut)
            stored_ikz_qi.push_back(std::make_pair(i_kz, nullptr));
    }

    _setStoredMatrices();

    interp_derivative_matrix.reserve(bins::NUMBER_OF_K_BANDS);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(N_Q_MATRICES));

    fisher_matrix = std::make_unique<double[]>(N_Q_MATRICES * N_Q_MATRICES);

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
    double size_m1 = process::getMemoryMB(DATA_SIZE_2);

    int n_spec_mat = 3 + stored_ikz_qi.size() + (!specifics::TURN_OFF_SFID);
    double remain_mem = process::MEMORY_ALLOC,
           needed_mem = n_spec_mat * size_m1;

    // Resolution matrix needs another temp storage.
    if (specifics::USE_RESOLUTION_MATRIX)
        needed_mem += qFile->Rmat->getBufMemUsage();
    if (on_oversampling)
        needed_mem += process::getMemoryMB(_matrix_n * (3 * _matrix_n + 1));

    // Need at least 3 matrices as temp one for sfid
    if (remain_mem < needed_mem) {
        LOG::LOGGER.ERR("===============\n""Not all matrices are stored: %s\n"
            "#required:%d.\n""ND: %d, M1: %.1f MB. "
            "Avail mem after R & SQ subtracted: %.1lf MB\n""===============\n", 
            qFile->fname.c_str(), stored_ikz_qi.size(), size(), size_m1, 
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
    double minmem = process::getMemoryMB((N_Q_MATRICES + 1) * N_Q_MATRICES);
    if (qFile) {
        minmem += process::getMemoryMB(size() * 3);    
        if (specifics::USE_RESOLUTION_MATRIX)
            minmem += qFile->Rmat->getMinMemUsage();
    }

    return minmem;
}

void Chunk::releaseFile() {
    if (!qFile)
        return;

    double released_mem = process::getMemoryMB(size() * 3);
    if (specifics::USE_RESOLUTION_MATRIX)
        released_mem += qFile->Rmat->getMinMemUsage();
    process::updateMemory(released_mem);
    qFile.reset();
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
        const int N_M_COMBO = 2 * bins::NUMBER_OF_K_BANDS;
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

void Chunk::_setVZMatrices() {
    LOG::LOGGER.DEB("Setting v & z matrices\n");

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < _matrix_n; ++i)
    {
        for (int j = i; j < _matrix_n; ++j)
        {
            double li = _matrix_lambda[i], lj = _matrix_lambda[j];

            _vmatrix[j + i * _matrix_n] = SPEED_OF_LIGHT * log(lj / li);
            _zmatrix[j + i * _matrix_n] = sqrt(li * lj) - 1.;
        }
    }

    for (int i = 0; i < _matrix_n; ++i)
        _matrix_lambda[i] -= 1;
}

void Chunk::_setFiducialSignalMatrix(double *sm)
{
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::timer.getTime();
    double *inter_mat = (on_oversampling) ? _finer_matrix : sm;

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < _matrix_n; ++i) {
        for (int j = i; j < _matrix_n; ++j) {
            int idx = j + i * _matrix_n;
            inter_mat[idx] = interp2d_signal_matrix->evaluate(
                _zmatrix[idx], _vmatrix[idx]);
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

    bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);
    bins::setRedshiftBinningFunction(zm);

    double *inter_mat = (on_oversampling) ? _finer_matrix : qi;
    shared_interp_1d interp_deriv_kn = interp_derivative_matrix[kn];

    int low, up;
    bins::redshiftBinningFunction(
        _matrix_lambda, _matrix_n, zm,
        inter_mat, low, up);

    std::fill_n(inter_mat, _matrix_n * _matrix_n, 0);

    #pragma omp parallel for schedule(static, 1)
    for (int i = low; i < up; ++i) {
        int idx = i * (1 + _matrix_n), l1, u1;

        bins::redshiftBinningFunction(
            _zmatrix + idx, _matrix_n - i, zm,
            inter_mat + idx, l1, u1);

        #pragma omp simd
        for (int j = 0; j < u1; ++j)
            inter_mat[j + idx] *= interp_deriv_kn->evaluate(_vmatrix[j + idx]);
    }

    t_interp = mytime::timer.getTime() - t;
    mxhelp::copyUpperToLower(inter_mat, _matrix_n);

    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->sandwich(qi, inter_mat);

    t = mytime::timer.getTime() - t; 

    mytime::time_spent_set_qs += t;
    mytime::time_spent_on_q_interp += t_interp;
    mytime::time_spent_on_q_copy += t - t_interp;
}

void Chunk::setCovarianceMatrix(const double *ps_estimate)
{
    LOG::LOGGER.DEB("Setting cov matrix\n");

    // Set fiducial signal matrix
    if (!specifics::TURN_OFF_SFID)
        std::copy(stored_sfid, stored_sfid + DATA_SIZE_2, covariance_matrix);
    else
        std::fill_n(covariance_matrix, DATA_SIZE_2, 0);

    const double *alpha = ps_estimate + fisher_index_start;
    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        cblas_daxpy(
            DATA_SIZE_2, alpha[i_kz], 
            Q_ikz_matrix, 1, covariance_matrix, 1
        );

    // add noise matrix diagonally
    // but smooth before adding
    process::noise_smoother->smoothNoise(qFile->noise(), temp_vector, size());
    cblas_daxpy(
        size(), 1., temp_vector, 1,
        covariance_matrix, size() + 1
    );

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
    std::transform(
        w, w+size, out,
        [cmo](const double &l) { return pow(log(l), cmo); }
    );
    double norm = sqrt(cblas_dnrm2(size, out, 1));
    cblas_dscal(size, 1./norm, out, 1);
}

void _getUnitVectorLam(const double *w, int size, int cmo, double *out)
{
    std::transform(
        w, w+size, out,
        [cmo](const double &l) { return pow(l, cmo); }
    );
    double norm = sqrt(cblas_dnrm2(size, out, 1));
    cblas_dscal(size, 1./norm, out, 1);
}

void _remShermanMorrison(const double *v, int size, double *y, double *cinv)
{
    cblas_dsymv(CblasRowMajor, CblasUpper, size, 1., cinv, size, v, 1, 0, y, 1);
    double norm = cblas_ddot(size, v, 1, y, 1);
    cblas_dger(CblasRowMajor, size, size, -1. / norm, y, 1, y, 1, cinv, size);
}

void Chunk::_addMarginalizations()
{
    double *temp_v = temp_matrix[0], *temp_y = temp_matrix[1];

    // Zeroth order
    std::fill_n(temp_v, size(), 1. / sqrt(size()));
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
        if (svals[i] < 1e-6)  continue;

        _remShermanMorrison(temp_v, size(), temp_y, inverse_covariance_matrix);
    }
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    LOG::LOGGER.DEB("Inverting cov matrix\n");

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
    //C-1 . Q
    cblas_dsymm(
        CblasRowMajor, CblasLeft, CblasUpper,
        size(), size(), 1., inverse_covariance_matrix, size(),
        m, size(), 0, temp_matrix[0], size());

    std::copy(temp_matrix[0], temp_matrix[0] + DATA_SIZE_2, m);
}

void Chunk::_getFisherMatrix(const double *Q_ikz_matrix_T, int idx)
{
    int i_kz = stored_ikz_qi[idx].first;
    int idx_fji_0 = N_Q_MATRICES * i_kz;

    // Now compute Fisher Matrix
    for (auto jq = stored_ikz_qi.begin() + idx; jq != stored_ikz_qi.end(); ++jq)
    {
        #ifdef FISHER_OPTIMIZATION
        if ((jq->first - i_kz > 2 * bins::NUMBER_OF_K_BANDS))
            continue;
        #endif

        fisher_matrix[jq->first + idx_fji_0] = 
            cblas_ddot(DATA_SIZE_2, Q_ikz_matrix_T, 1, jq->second, 1);
    }
}

void Chunk::computePSbeforeFvector()
{
    LOG::LOGGER.DEB("PS before Fisher\n");

    double *dk0 = dbt_estimate_before_fisher_vector[0].get(),
           *nk0 = dbt_estimate_before_fisher_vector[1].get(),
           *tk0 = dbt_estimate_before_fisher_vector[2].get(),
           *Q_ikz_matrix;

    // C-1 . flux
    cblas_dsymv(
        CblasRowMajor, CblasUpper, size(), 1.,
        inverse_covariance_matrix, 
        size(), qFile->delta(), 1, 0, weighted_data_vector, 1);

    double t = mytime::timer.getTime();

    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi) {
        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        dk0[i_kz] = mxhelp::my_cblas_dsymvdot(
            weighted_data_vector, 
            Q_ikz_matrix, temp_vector, size());
        // Transform q matrices to weighted matrices inplace
        // Get weighted derivative matrix ikz: C-1 Qi
        _getWeightedMatrix(Q_ikz_matrix);
    }

    mytime::time_spent_set_modqs += mytime::timer.getTime() - t;

    // N C-1
    double *weighted_noise_matrix = temp_matrix[0];
    std::fill_n(weighted_noise_matrix, DATA_SIZE_2, 0);
    for (int i = 0; i < size(); ++i)
        cblas_daxpy(
            size(), qFile->noise()[i],
            inverse_covariance_matrix + i * size(), 1,
            weighted_noise_matrix + i * size(), 1);

    // Get Noise contribution: Tr(C-1 Qi C-1 N)
    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        nk0[i_kz] = cblas_ddot(DATA_SIZE_2, Q_ikz_matrix, 1, weighted_noise_matrix, 1);

    // Fiducial matrix, Sfid C-1
    if (!specifics::TURN_OFF_SFID) {
        double *weighted_sfid_matrix = temp_matrix[0];
        cblas_dsymm(
            CblasRowMajor, CblasLeft, CblasUpper,
            size(), size(), 1., stored_sfid, size(),
            inverse_covariance_matrix, size(), 0, weighted_sfid_matrix, size());

        for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
            tk0[i_kz] = cblas_ddot(DATA_SIZE_2, Q_ikz_matrix, 1, weighted_sfid_matrix, 1);
    }

    // Do not compute fisher matrix if it is precomputed
    if (specifics::USE_PRECOMPUTED_FISHER)
        return;

    t = mytime::timer.getTime();

    double *Q_ikz_matrix_T = temp_matrix[0];
    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi) {
        mxhelp::transpose_copy(Q_ikz_matrix, Q_ikz_matrix_T, size());

        _getFisherMatrix(Q_ikz_matrix_T, i_kz);
    }

    mytime::time_spent_set_fisher += mytime::timer.getTime() - t;
}

void Chunk::oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum
) {
    LOG::LOGGER.DEB("File %s\n", qFile->fname.c_str());
    LOG::LOGGER.DEB("TargetID %ld\n", qFile->id);
    LOG::LOGGER.DEB("Size %d\n", size());
    LOG::LOGGER.DEB("ncols: %d\n", _matrix_n);
    LOG::LOGGER.DEB("fisher_index_start: %d\n", fisher_index_start);
    LOG::LOGGER.DEB("Allocating matrices\n");

    _allocateMatrices();

    _setVZMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    // i_kz = N_Q_MATRICES - j_kz - 1
    LOG::LOGGER.DEB("Setting qi matrices\n");

    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        _setQiMatrix(Q_ikz_matrix, i_kz);

    for (int i = 0; i < _matrix_n; ++i)
        _matrix_lambda[i] += 1;

    // Preload fiducial signal matrix if memory allows
    if (!specifics::TURN_OFF_SFID)
        _setFiducialSignalMatrix(stored_sfid);

    setCovarianceMatrix(ps_estimate);

    try
    {
        invertCovarianceMatrix();

        computePSbeforeFvector();

        double *outfisher = fisher_sum + (bins::TOTAL_KZ_BINS + 1) * fisher_index_start;

        for (int i = 0; i < N_Q_MATRICES; ++i) {
            for (int j = i; j < N_Q_MATRICES; ++j) {
                outfisher[j + i * bins::TOTAL_KZ_BINS] += fisher_matrix[j + i * N_Q_MATRICES];
            } 
        }

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            mxhelp::vector_add(
                dbt_sum_vector[dbt_i].get() + fisher_index_start, 
                dbt_estimate_before_fisher_vector[dbt_i].get(),
                N_Q_MATRICES);
    }
    catch (std::exception& e) {
        LOG::LOGGER.ERR(
            "ERROR %s: Covariance matrix is not invertable. %s\n",
            e.what(), qFile->fname.c_str());

        LOG::LOGGER.ERR(
            "Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
            size(), MEDIAN_REDSHIFT, qFile->dv_kms, qFile->R_fwhm);
    }

    _freeMatrices();
}

void Chunk::addBoot(int p, double *temppower, double* tempfisher) {
    double *outfisher = tempfisher + (bins::TOTAL_KZ_BINS + 1) * fisher_index_start;

    for (int i = 0; i < N_Q_MATRICES; ++i) {
        for (int j = i; j < N_Q_MATRICES; ++j) {
            outfisher[j + i * bins::TOTAL_KZ_BINS] +=
                p * fisher_matrix[j + i * N_Q_MATRICES];
        } 
    }

    cblas_daxpy(
        N_Q_MATRICES,
        p, dbt_estimate_before_fisher_vector[0].get(), 1,
        temppower + fisher_index_start, 1);
}

void Chunk::_allocateMatrices()
{
    covariance_matrix = new double[DATA_SIZE_2];

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = new double[DATA_SIZE_2];

    temp_vector = new double[size()];
    weighted_data_vector = new double[size()];
    
    for (auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        Q_ikz_matrix = new double[DATA_SIZE_2];

    if (!specifics::TURN_OFF_SFID)
        stored_sfid = new double[DATA_SIZE_2];

    // Create a temp highres lambda array
    if (on_oversampling)
    {
        _finer_lambda = new double[_matrix_n];

        double fine_dlambda =
            qFile->dlambda / LYA_REST / specifics::OVERSAMPLING_FACTOR;
        int disp = qFile->Rmat->getNElemPerRow()/2;
        for (int i = 0; i < _matrix_n; ++i)
            _finer_lambda[i] = qFile->wave()[0] + (i - disp)*fine_dlambda;

        _matrix_lambda = _finer_lambda;

        long highsize = (long)(_matrix_n) * (long)(_matrix_n);
        _finer_matrix = new double[highsize];
        _vmatrix = new double[highsize];
        _zmatrix = new double[highsize];
    }
    else
    {
        _matrix_lambda = qFile->wave();
        _finer_lambda  = NULL;
        _finer_matrix  = NULL;
        _vmatrix = temp_matrix[0];
        _zmatrix = temp_matrix[1];
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
    delete [] covariance_matrix;

    for (int i = 0; i < 2; ++i)
        delete [] temp_matrix[i];

    delete [] temp_vector;
    delete [] weighted_data_vector;
    for (auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        delete [] Q_ikz_matrix;

    if (!specifics::TURN_OFF_SFID)
        delete [] stored_sfid;

    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->freeBuffer();

    if (on_oversampling)
    {
        delete [] _finer_matrix;
        delete [] _finer_lambda;
        delete [] _vmatrix;
        delete [] _zmatrix;
    }

    if (interp2d_signal_matrix)
        interp2d_signal_matrix.reset();
    interp_derivative_matrix.clear();
}

void Chunk::fprintfMatrices(const char *fname_base)
{
    std::string buf;

    if (!specifics::TURN_OFF_SFID)
    {
        buf = std::string(fname_base) + "-signal.txt";
        mxhelp::fprintfMatrix(buf.c_str(), stored_sfid, size(), size());
    }

    if (qFile->Rmat != NULL)
    {
        buf = std::string(fname_base) + "-resolution.txt";
        qFile->Rmat->fprintfMatrix(buf.c_str());
    }
}






