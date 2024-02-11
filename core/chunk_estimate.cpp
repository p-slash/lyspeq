#include "core/chunk_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/omp_manager.hpp"
#include "core/fiducial_cosmology.hpp"
#include "core/sq_table.hpp"

#include "mathtools/smoother.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each & transform & lower(upper)_bound
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>


#ifdef DEBUG
void CHECK_ISNAN(double *mat, int size, std::string msg)
{
    if (std::any_of(mat, mat+size, [](double x) {return std::isnan(x);}))
        throw std::runtime_error(std::string("NAN in ") + msg);
    std::string line = std::string("No nans in ") + msg + '\n';
    DEBUG_LOG(line.c_str());
}
#else
#define CHECK_ISNAN(X, Y, Z)
#endif

inline
int _getMaxKindex(double knyq)
{
    auto it = std::lower_bound(
        bins::KBAND_CENTERS.begin(), bins::KBAND_CENTERS.end(),
        0.85 * knyq);
    return std::distance(bins::KBAND_CENTERS.begin(), it);
}

namespace glmemory {
    int max_size = 0, max_size_2 = 0, max_matrix_n = 0, max_nqdim = 0;
    bool on_oversampling = false;
    double memUsed = 0;
    std::unique_ptr<double[]>
        covariance_matrix, temp_matrix[2], temp_vector, weighted_data_vector,
        stored_sfid, matrix_lambda, finer_matrix, v_matrix, z_matrix;
    std::vector<std::unique_ptr<double[]>> stored_ikz_qi;

    shared_interp_2d interp2d_signal_matrix;
    std::vector<shared_interp_1d> interp_derivative_matrix;

    void setMaxSizes(int size, int matrix_n, int nqdim, bool onsamp) {
        max_size = std::max(size, max_size);
        max_size_2 = max_size * max_size;
        max_matrix_n = std::max(matrix_n, max_matrix_n);
        max_nqdim = std::max(nqdim, max_nqdim);
        on_oversampling |= onsamp;
    }

    void allocMemory() {
        int highsize = 0;

        memUsed += process::getMemoryMB(
            2 * max_size + (3 + !specifics::TURN_OFF_SFID + max_nqdim) * max_size_2
        );

        if (on_oversampling) {
            highsize = max_matrix_n * max_matrix_n;
            memUsed += process::getMemoryMB(max_matrix_n + 3 * highsize);
        }

        process::updateMemory(-memUsed);

        temp_vector = std::make_unique<double[]>(max_size);
        weighted_data_vector = std::make_unique<double[]>(max_size);

        covariance_matrix = std::make_unique<double[]>(max_size_2);
        temp_matrix[0] = std::make_unique<double[]>(max_size_2);
        temp_matrix[1] = std::make_unique<double[]>(max_size_2);

        if (!specifics::TURN_OFF_SFID)
            stored_sfid = std::make_unique<double[]>(max_size_2);

        for (int i = 0; i < max_nqdim; ++i)
            stored_ikz_qi.push_back(std::make_unique<double[]>(max_size_2));

        if (on_oversampling) {
            matrix_lambda = std::make_unique<double[]>(max_matrix_n);
            finer_matrix = std::make_unique<double[]>(highsize);
            v_matrix = std::make_unique<double[]>(highsize);
            z_matrix = std::make_unique<double[]>(highsize);
        }

        interp_derivative_matrix.resize(bins::NUMBER_OF_K_BANDS);
    }

    void dealloc() {
        temp_vector.reset();
        weighted_data_vector.reset();
        covariance_matrix.reset();
        temp_matrix[0].reset();
        temp_matrix[1].reset();
        stored_sfid.reset();
        matrix_lambda.reset();
        finer_matrix.reset();
        v_matrix.reset();
        z_matrix.reset();

        interp2d_signal_matrix.reset();
        interp_derivative_matrix.clear();

        process::updateMemory(memUsed);
    }
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
    process::updateMemory(-getMinMemUsage());

    stored_ikz_qi.reserve(N_Q_MATRICES);
    int _kncut = _getMaxKindex(MY_PI / qFile->dv_kms);
    for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
        int kn, zm;
        bins::getFisherMatrixBinNoFromIndex(i_kz + fisher_index_start, kn, zm);

        if (kn < _kncut)
            stored_ikz_qi.push_back(std::make_pair(i_kz, nullptr));
    }

    _setStoredMatrices();
    glmemory::setMaxSizes(size(), _matrix_n, stored_ikz_qi.size(), on_oversampling);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(N_Q_MATRICES));

    fisher_matrix = std::make_unique<double[]>(N_Q_MATRICES * N_Q_MATRICES);
}

void Chunk::_copyQSOFile(const qio::QSOFile &qmaster, int i1, int i2)
{
    qFile = std::make_unique<qio::QSOFile>(qmaster, i1, i2);

    if (qFile->realSize() < MIN_PIXELS_IN_CHUNK)
    {
        std::ostringstream err_msg;
        err_msg << "Short chunk with realsize "
            << qFile->realSize() << '/'  << qFile->size() << '.';

        throw std::runtime_error(err_msg.str());
    }

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

    DATA_SIZE_2 = size() * size();
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
void Chunk::_setNQandFisherIndex() {   
    N_Q_MATRICES = ZBIN_UPP - ZBIN_LOW + 1;
    fisher_index_start = bins::getFisherMatrixIndex(0, ZBIN_LOW);

    if (bins::Z_BINNING_METHOD == bins::TriangleBinningMethod) {
        // If we need to distribute low end to a lefter bin
        if ((LOWER_REDSHIFT < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0))
        {
            ++N_Q_MATRICES;
            fisher_index_start -= bins::NUMBER_OF_K_BANDS;
        }
        // If we need to distribute high end to righter bin
        if ((bins::ZBIN_CENTERS[ZBIN_UPP] < UPPER_REDSHIFT) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
            ++N_Q_MATRICES;
    }

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
        throw std::runtime_error("Not all matrices are stored.");
    }
}

Chunk::~Chunk()
{
    process::updateMemory(
        process::getMemoryMB((N_Q_MATRICES + 1) * N_Q_MATRICES)
    );
}

double Chunk::getMinMemUsage()
{
    double minmem = process::getMemoryMB((N_Q_MATRICES + 1) * N_Q_MATRICES);
    if (qFile)
        minmem += qFile->getMinMemUsage();

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
        // NERSC Perlmutter scaling relation for -c 2
        const double agemm = 19., asymv = 0.45, adot = 1.0;
        double one_dgemm = agemm * std::pow(qtemp.size() / 100., 3),
               one_dsymv = asymv * std::pow(qtemp.size() / 100., 2),
               one_ddot = adot * std::pow(qtemp.size() / 100., 2);

        int fidxlocal = bins::getFisherMatrixIndex(0, ZBIN_LOW);

        if (bins::Z_BINNING_METHOD == bins::TriangleBinningMethod) {
            if ((z1 < bins::ZBIN_CENTERS[ZBIN_LOW]) && (ZBIN_LOW != 0)) {
                ++N_Q_MATRICES;
                fidxlocal -= bins::NUMBER_OF_K_BANDS;
            }

            if ((bins::ZBIN_CENTERS[ZBIN_UPP] < z2) && (ZBIN_UPP != (bins::NUMBER_OF_Z_BINS-1)))
                ++N_Q_MATRICES;
        }

        N_Q_MATRICES *= bins::NUMBER_OF_K_BANDS;

        int _kncut = _getMaxKindex(MY_PI / qtemp.dv_kms), real_nq_mat = 0;
        for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
            int kn, zm;
            bins::getFisherMatrixBinNoFromIndex(i_kz + fidxlocal, kn, zm);

            if (kn < _kncut) ++real_nq_mat;
        }

        // +1 for noise matrix dot product
        #ifdef FISHER_OPTIMIZATION
        const int N_M_COMBO = 11 + 1;
        #else
        int N_M_COMBO = real_nq_mat + 1 + 1;
        #endif

        double res = (
            real_nq_mat * (one_dgemm + one_ddot * N_M_COMBO)
            + (real_nq_mat + 1) * one_dsymv // dsymv contributions
        );

        if (!specifics::TURN_OFF_SFID)
            res += one_dgemm + real_nq_mat * one_ddot;

        if (specifics::USE_RESOLUTION_MATRIX) {
            const int ndiags = 11;
            double extra_ctime = ndiags * one_ddot * real_nq_mat;

            int osamp = specifics::OVERSAMPLING_FACTOR;
            if (osamp > 0)
                extra_ctime *= osamp * (osamp + 1);
            else
                extra_ctime *= 2;

            res += extra_ctime;
        }

        return res;
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR(
            "Chunk::getComputeTimeEst::%s. Skipping %s.\n",
            e.what(), qmaster.fname.c_str());
        return 0;
    }
}

void Chunk::_setVZMatrices() {
    DEBUG_LOG("Setting v & z matrices\n");

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
    double *inter_mat = (on_oversampling) ? glmemory::finer_matrix.get() : sm;

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < _matrix_n; ++i) {
        for (int j = i; j < _matrix_n; ++j) {
            int idx = j + i * _matrix_n;
            inter_mat[idx] = glmemory::interp2d_signal_matrix->evaluate(
                _zmatrix[idx], _vmatrix[idx]);
        }
    }

    mxhelp::copyUpperToLower(inter_mat, _matrix_n);

    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->sandwich(inter_mat, sm);

    CHECK_ISNAN(sm, DATA_SIZE_2, "Sfid");

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

    std::function<double(double)> eval_deriv_kn;

    if (specifics::USE_RESOLUTION_MATRIX) {
        double kc = bins::KBAND_CENTERS[kn],
               dk = bins::KBAND_EDGES[kn + 1] - bins::KBAND_EDGES[kn];

        eval_deriv_kn = [kc, dk](double v) {
            double x = dk * v / 2;
            if (x == 0)
                x = 1.0;
            else
                x = sin(x) / x;

            return dk / MY_PI * x * cos(kc * v);
        };
    } else {
        eval_deriv_kn = [idkn = glmemory::interp_derivative_matrix[kn]](double v) {
            return idkn->evaluate(v);
        };
    }

    int low, up;
    double *inter_mat = (on_oversampling) ? glmemory::finer_matrix.get() : qi;
    bins::redshiftBinningFunction(
        _matrix_lambda, _matrix_n, zm,
        inter_mat, low, up);

    std::fill_n(inter_mat, _matrix_n * _matrix_n, 0);

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < up; ++i) {
        int idx = i * (1 + _matrix_n), l1, u1;

        bins::redshiftBinningFunction(
            _zmatrix + idx, _matrix_n - i, zm,
            inter_mat + idx, l1, u1);

        #pragma omp simd
        for (int j = l1; j < u1; ++j)
            inter_mat[j + idx] *= eval_deriv_kn(_vmatrix[j + idx]);
    }

    t_interp = mytime::timer.getTime() - t;
    mxhelp::copyUpperToLower(inter_mat, _matrix_n);

    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->sandwich(inter_mat, qi);

    t = mytime::timer.getTime() - t; 

    mytime::time_spent_set_qs += t;
    mytime::time_spent_on_q_interp += t_interp;
    mytime::time_spent_on_q_copy += t - t_interp;
}

void Chunk::setCovarianceMatrix(const double *ps_estimate)
{
    DEBUG_LOG("Setting cov matrix\n");

    // Set fiducial signal matrix
    if (!specifics::TURN_OFF_SFID)
        std::copy_n(glmemory::stored_sfid.get(), DATA_SIZE_2, covariance_matrix);
    else
        std::fill_n(covariance_matrix, DATA_SIZE_2, 0);

    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        cblas_daxpy(
            DATA_SIZE_2, ps_estimate[fisher_index_start + i_kz], 
            Q_ikz_matrix, 1, covariance_matrix, 1
        );

    // add noise matrix diagonally
    // but smooth before adding
    double *nvec = qFile->noise();
    if (process::smoother->isSmoothingOn()) {
        process::smoother->smoothNoise(
            qFile->noise(), glmemory::temp_vector.get(), size());
        nvec = glmemory::temp_vector.get();
    }

    cblas_daxpy(size(), 1., nvec, 1, covariance_matrix, size() + 1);

    isCovInverted = false;
    CHECK_ISNAN(covariance_matrix, DATA_SIZE_2, "CovMat");

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
    cblas_dsymv(
        CblasRowMajor, CblasUpper, size, 1.,
        cinv, size, v, 1, 0, y, 1);
    double norm = cblas_ddot(size, v, 1, y, 1);
    cblas_dger(CblasRowMajor, size, size, -1. / norm, y, 1, y, 1, cinv, size);
}

void Chunk::_addMarginalizations() {
    DEBUG_LOG("Adding marginalizations...\n");
    static auto svals = std::make_unique<double[]>(specifics::CONT_NVECS);
    double *marg_mat = temp_matrix[0];

    std::fill_n(marg_mat, size(), 1. / sqrt(size()));  // Zeroth order
    marg_mat += size();

    for (int cmo = 1; cmo <= specifics::CONT_LOGLAM_MARG_ORDER; ++cmo) {
        _getUnitVectorLogLam(qFile->wave(), size(), cmo, marg_mat);
        marg_mat += size();
    }

    for (int cmo = 1; cmo <= specifics::CONT_LAM_MARG_ORDER; ++cmo) {
        _getUnitVectorLam(qFile->wave(), size(), cmo, marg_mat);
        marg_mat += size();
    }

    // Roll back to initial position
    marg_mat = temp_matrix[0];

    #ifdef DEBUG
    DEBUG_LOG("Mags before:");
    for (int i = 0; i < specifics::CONT_NVECS; ++i) {
        double tt = mxhelp::my_cblas_dsymvdot(
            marg_mat + i * size(), inverse_covariance_matrix,
            glmemory::temp_vector.get(), size());
        DEBUG_LOG("  %.3e", tt);
    } DEBUG_LOG("\n");
    std::copy_n(marg_mat, size() * specifics::CONT_NVECS, temp_matrix[1]);
    #endif

    // SVD to get orthogonal marg vectors
    mxhelp::LAPACKE_svd(marg_mat, svals.get(), size(), specifics::CONT_NVECS);

    int nvecs_to_use = specifics::CONT_NVECS;
    while (nvecs_to_use > 0) {
        if ((svals[nvecs_to_use - 1] / svals[0]) > DOUBLE_EPSILON)
            break;
        --nvecs_to_use;
    }

    for (int i = 0; i < nvecs_to_use; ++i)
        _remShermanMorrison(
            marg_mat + i * size(), size(),
            glmemory::temp_vector.get(), inverse_covariance_matrix);

    #ifdef DEBUG
    DEBUG_LOG("SVD:");
    for (int i = 0; i < specifics::CONT_NVECS; ++i)
        DEBUG_LOG("  %.3e", svals[i]);
    DEBUG_LOG("\nUsing first %d/%d vectors.\nMags after:",
              nvecs_to_use, specifics::CONT_NVECS);
    for (int i = 0; i < specifics::CONT_NVECS; ++i) {
        double tt = mxhelp::my_cblas_dsymvdot(
            temp_matrix[1] + i * size(), inverse_covariance_matrix,
            glmemory::temp_vector.get(), size());
        DEBUG_LOG("  %.3e", tt);
    } DEBUG_LOG("\n");
    #endif
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    DEBUG_LOG("Inverting cov matrix\n");

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
    // cblas_dsymm(
    //     CblasRowMajor, CblasLeft, CblasUpper,
    //     size(), size(), 1., inverse_covariance_matrix, size(),
    //     m, size(), 0, temp_matrix[0], size());

    // NERSC Perlmutter has faster dgemm
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        size(), size(), size(), 1., inverse_covariance_matrix, size(),
        m, size(), 0, temp_matrix[0], size());

    std::copy(temp_matrix[0], temp_matrix[0] + DATA_SIZE_2, m);
}

void Chunk::computePSbeforeFvector()
{
    DEBUG_LOG("PS before Fisher\n");

    double *dk0 = dbt_estimate_before_fisher_vector[0].get(),
           *nk0 = dbt_estimate_before_fisher_vector[1].get(),
           *tk0 = dbt_estimate_before_fisher_vector[2].get();

    // C-1 . flux
    cblas_dsymv(
        CblasRowMajor, CblasUpper, size(), 1.,
        inverse_covariance_matrix, size(), qFile->delta(), 1,
        0, glmemory::weighted_data_vector.get(), 1);

    double t = mytime::timer.getTime();

    for (auto iqt = stored_ikz_qi.begin(); iqt != stored_ikz_qi.end(); ++iqt) {
        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        dk0[iqt->first] = mxhelp::my_cblas_dsymvdot(
            glmemory::weighted_data_vector.get(), 
            iqt->second, glmemory::temp_vector.get(), size());
        // Transform q matrices to weighted matrices inplace
        // Get weighted derivative matrix ikz: C-1 Qi
        _getWeightedMatrix(iqt->second);
    }

    mytime::time_spent_set_modqs += mytime::timer.getTime() - t;

    // N C-1
    double *weighted_noise_matrix = temp_matrix[0];
    std::fill_n(weighted_noise_matrix, DATA_SIZE_2, 0);
    #pragma omp parallel for
    for (int i = 0; i < size(); ++i)
        cblas_daxpy(
            size(), qFile->noise()[i],
            inverse_covariance_matrix + i * size(), 1,
            weighted_noise_matrix + i * size(), 1);

    // Get Noise contribution: Tr(C-1 Qi C-1 N)
    #pragma omp parallel for
    for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
        nk0[i_kz] = cblas_ddot(DATA_SIZE_2, Q_ikz_matrix, 1, weighted_noise_matrix, 1);

    // Fiducial matrix, Sfid C-1
    if (!specifics::TURN_OFF_SFID) {
        double *weighted_sfid_matrix = temp_matrix[0];
        // cblas_dsymm(
        //     CblasRowMajor, CblasLeft, CblasUpper,
        //     size(), size(), 1., stored_sfid, size(),
        //     inverse_covariance_matrix, size(), 0, weighted_sfid_matrix, size());
        // NERSC Perlmutter has faster dgemm
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            size(), size(), size(), 1., glmemory::stored_sfid.get(), size(),
            inverse_covariance_matrix, size(), 0, weighted_sfid_matrix, size());

        #pragma omp parallel for
        for (const auto &[i_kz, Q_ikz_matrix] : stored_ikz_qi)
            tk0[i_kz] = cblas_ddot(DATA_SIZE_2, Q_ikz_matrix, 1, weighted_sfid_matrix, 1);
    }

    // Do not compute fisher matrix if it is precomputed
    if (specifics::USE_PRECOMPUTED_FISHER)
        return;

    t = mytime::timer.getTime();

    // I am not sure if we can parallize the outer loop here.
    // need to limit num_threads=2, but will that mess up inner loops?
    // Seems to speed things up though.
    #pragma omp parallel for num_threads(2) schedule(static, 1)
    for (auto iqt = stored_ikz_qi.cbegin(); iqt != stored_ikz_qi.cend(); ++iqt) {
        int idx_fji_0 = N_Q_MATRICES * iqt->first;
        double *Q_ikz_matrix_T = temp_matrix[myomp::getThreadNum()];

        mxhelp::transpose_copy(iqt->second, Q_ikz_matrix_T, size(), size());

        #pragma omp parallel for
        for (auto jqt = iqt; jqt != stored_ikz_qi.cend(); ++jqt) {
            #ifdef FISHER_OPTIMIZATION
            int diff_ji = jqt->first - iqt->first;
            if ((diff_ji > 5) && (abs(diff_ji - bins::NUMBER_OF_K_BANDS) > 2))
                continue;
            #endif

            fisher_matrix[jqt->first + idx_fji_0] = 
                cblas_ddot(DATA_SIZE_2, Q_ikz_matrix_T, 1, jqt->second, 1);
        }
    }

    mytime::time_spent_set_fisher += mytime::timer.getTime() - t;
}

void Chunk::oneQSOiteration(
        const double *ps_estimate, 
        std::vector<std::unique_ptr<double[]>> &dbt_sum_vector,
        double *fisher_sum
) {
    DEBUG_LOG("File %s\n", qFile->fname.c_str());
    DEBUG_LOG("TargetID %ld\n", qFile->id);
    DEBUG_LOG("Size %d\n", size());
    DEBUG_LOG("ncols: %d\n", _matrix_n);
    DEBUG_LOG("fisher_index_start: %d\n", fisher_index_start);
    DEBUG_LOG("N_Q_MATRICES: %d\n", N_Q_MATRICES);

    CHECK_ISNAN(qFile->wave(), size(), "qFile->wave");
    CHECK_ISNAN(qFile->delta(), size(), "qFile->delta");
    CHECK_ISNAN(qFile->noise(), size(), "qFile->noise");

    if (qFile->Rmat) {
        CHECK_ISNAN(qFile->Rmat->matrix(), qFile->Rmat->getSize(), "Rmat");
    }

    DEBUG_LOG("Allocating matrices\n");

    _initMatrices();

    _setVZMatrices();

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    // i_kz = N_Q_MATRICES - j_kz - 1
    DEBUG_LOG("Setting qi matrices\n");

    for (auto iqt = stored_ikz_qi.begin(); iqt != stored_ikz_qi.end(); ++iqt)
        _setQiMatrix(iqt->second, iqt->first);

    for (int i = 0; i < _matrix_n; ++i)
        _matrix_lambda[i] += 1;

    // Preload fiducial signal matrix if memory allows
    if (!specifics::TURN_OFF_SFID)
        _setFiducialSignalMatrix(glmemory::stored_sfid.get());

    setCovarianceMatrix(ps_estimate);

    try
    {
        invertCovarianceMatrix();

        computePSbeforeFvector();

        double *outfisher = fisher_sum + (bins::TOTAL_KZ_BINS + 1) * fisher_index_start;

        for (int i = 0; i < N_Q_MATRICES; ++i)
            for (int j = i; j < N_Q_MATRICES; ++j)
                outfisher[j + i * bins::TOTAL_KZ_BINS] += fisher_matrix[j + i * N_Q_MATRICES];

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            cblas_daxpy(
                N_Q_MATRICES,
                1, dbt_estimate_before_fisher_vector[dbt_i].get(), 1,
                dbt_sum_vector[dbt_i].get() + fisher_index_start, 1);
    }
    catch (std::exception& e) {
        LOG::LOGGER.ERR(
            "ERROR %s: Covariance matrix is not invertable. %s\n",
            e.what(), qFile->fname.c_str());

        LOG::LOGGER.ERR(
            "Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
            size(), MEDIAN_REDSHIFT, qFile->dv_kms, qFile->R_fwhm);
    }
}


void Chunk::_initMatrices()
{
    covariance_matrix = glmemory::covariance_matrix.get();

    for (int i = 0; i < 2; ++i)
        temp_matrix[i] = glmemory::temp_matrix[i].get();
    
    for (int i = 0; i < stored_ikz_qi.size(); ++i)
        stored_ikz_qi[i].second = glmemory::stored_ikz_qi[i].get();

    // Create a temp highres lambda array
    if (on_oversampling)
    {
        _matrix_lambda = glmemory::matrix_lambda.get();

        double fine_dlambda =
            qFile->dlambda / LYA_REST / specifics::OVERSAMPLING_FACTOR;
        int disp = qFile->Rmat->getNElemPerRow() / 2;
        for (int i = 0; i < _matrix_n; ++i)
            _matrix_lambda[i] = qFile->wave()[0] + (i - disp) * fine_dlambda;

        _vmatrix = glmemory::v_matrix.get();
        _zmatrix = glmemory::z_matrix.get();
    }
    else
    {
        _matrix_lambda = qFile->wave();
        _vmatrix = temp_matrix[0];
        _zmatrix = temp_matrix[1];
    }

    // This function allocates new signal & deriv matrices 
    // if process::SAVE_ALL_SQ_FILES=false 
    // i.e., no caching of SQ files
    // If all tables are cached, then this function simply points 
    // to those in process:sq_private_table
    process::sq_private_table->readSQforR(
        RES_INDEX, glmemory::interp2d_signal_matrix, 
        glmemory::interp_derivative_matrix);
}

void Chunk::fprintfMatrices(const char *fname_base)
{
    std::string buf;

    if (!specifics::TURN_OFF_SFID)
    {
        buf = std::string(fname_base) + "-signal.txt";
        mxhelp::fprintfMatrix(
            buf.c_str(), glmemory::stored_sfid.get(), size(), size());
    }

    if (qFile->Rmat != NULL)
    {
        buf = std::string(fname_base) + "-resolution.txt";
        qFile->Rmat->fprintfMatrix(buf.c_str());
    }
}






