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

CuBlasHelper cublas_helper;
CuSolverHelper cusolver_helper;


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
    on_oversampling = specifics::USE_RESOLUTION_MATRIX && !qFile->Rmat->isDiaMatrix();

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

    // host covariance and temp_matrix[2] are no longer needed
    // maybe still check for gpu memory alloc?
    double remain_mem = process::MEMORY_ALLOC,
           needed_mem = (3+i_kz_vector.size()+(!specifics::TURN_OFF_SFID))*size_m1;

    // Resolution matrix needs another temp storage.
    if (specifics::USE_RESOLUTION_MATRIX)
        needed_mem += qFile->Rmat->getBufMemUsage();
    if (on_oversampling) {
        double ncols = (double) qFile->Rmat->getNCols();
        needed_mem += sizeof(double) * ncols * (3 * ncols+1) / 1048576;
    }

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

void Chunk::_setVZMatrices() {
    for (int i = 0; i < _matrix_n; ++i)
    {
        double li = _matrix_lambda[i];

        for (int j = i; j < _matrix_n; ++j)
        {
            double lj = _matrix_lambda[j];
            int idx = j + i * _matrix_n;

            _getVandZ(li, lj, _vmatrix[idx], _zmatrix[idx]);
        }
    }
}

void Chunk::_setFiducialSignalMatrix(double *sm)
{
    ++mytime::number_of_times_called_setsfid;

    double t = mytime::timer.getTime();
    double *inter_mat = (on_oversampling) ? _finer_matrix : sm;

    #pragma omp parallel for
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

    #pragma omp parallel for
    for (int i = 0; i < _matrix_n; ++i) {
        for (int j = i; j < _matrix_n; ++j) {
            int idx = j + i * _matrix_n;
            inter_mat[idx] = 
                interp_deriv_kn->evaluate(_vmatrix[idx])
                * bins::redshiftBinningFunction(_zmatrix[idx], zm);
        }
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
        cublas_helper.dcopy(
            dev_sfid.get(), covariance_matrix.get(), DATA_SIZE_2);
    else
        cudaMemset(covariance_matrix.get(), 0, DATA_SIZE_2*sizeof(double));

    const double *alpha = ps_estimate + fisher_index_start;
    for (int idx = 0; idx < i_kz_vector.size(); ++idx) {
        int i_kz = i_kz_vector[idx];

        double *Q_ikz_matrix = _getDevQikz(idx);

        cublas_helper.daxpy(
            alpha[i_kz], Q_ikz_matrix, covariance_matrix.get(), DATA_SIZE_2);
    }

    // add noise matrix diagonally
    // but smooth before adding
    // process::noise_smoother->smoothNoise(qFile->noise(), temp_vector, size());
    cublas_helper.daxpy(
        1., dev_smnoise.get(), covariance_matrix.get(), size(), 1, size()+1);

    isCovInverted = false;
}

// CUDA Kernels to set marginalization vectors
__global__
void _setZerothOrder(int size, double alpha, double *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        out[i] = alpha;
}

__global__
void _getUnitVectorLogLam(const double *w, int size, int cmo, double *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        out[i] = pow(log(w[i]/LYA_REST), cmo);
    // double alpha = rnorm(size, out);
    // for (int i = index; i < size; i += stride)
    //     out[i] /= alpha;
}

__global__
void _getUnitVectorLam(const double *w, int size, int cmo, double *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        out[i] = pow(w[i]/LYA_REST, cmo);
    // double alpha = rnorm(size, out);
    // for (int i = index; i < size; i += stride)
    //     out[i] /= alpha;
}
// End CUDA Kernels ------------------------------

void _remShermanMorrison(const double *v, int size, double *y, double *cinv)
{
    // cudaMemset(y, 0, size*sizeof(double));
    cublas_helper.dsmyv(size, 1., cinv, size, v, 1, 0, y, 1);
    double alpha;
    cublasDdot(cublas_helper.blas_handle, size, v, 1, y, 1, &alpha);
    alpha = -1./alpha;
    cublasDsyr(
        cublas_helper.blas_handle, CUBLAS_FILL_MODE_LOWER,
        size, &alpha, y, 1, cinv, size);
    // cublasDger(
    // cuhelper.blas_handle, size, size, &norm, y, 1, y, 1, cinv, size);
}

void Chunk::_addMarginalizations()
{
    std::vector<MyCuStream> streams(specifics::CONT_NVECS);
    int num_blocks = (size() + MYCU_BLOCK_SIZE - 1) / MYCU_BLOCK_SIZE,
        vidx = 1;

    double  *temp_v = temp_matrix[0].get(),
            *temp_y = temp_matrix[1].get();

    // Zeroth order
    _setZerothOrder<<<num_blocks, MYCU_BLOCK_SIZE, 0, streams[0].get()>>>(
        size(), 1/sqrt(size()), temp_v);

    // Log lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LOGLAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLogLam<<<num_blocks, MYCU_BLOCK_SIZE, 0, streams[vidx].get()>>>(
            dev_wave.get(), size(), cmo, temp_v + vidx * size());
        ++vidx;
    }
    // Lambda polynomials
    for (int cmo = 1; cmo <= specifics::CONT_LAM_MARG_ORDER; ++cmo)
    {
        _getUnitVectorLam<<<num_blocks, MYCU_BLOCK_SIZE, 0, streams[vidx].get()>>>(
            dev_wave.get(), size(), cmo, temp_v + vidx * size());
        ++vidx;
    }

    // for (auto &stream : streams)
    //     stream.sync();

    LOG::LOGGER.DEB("nvecs %d\n", specifics::CONT_NVECS);

    static MyCuPtr<double> dev_svals(specifics::CONT_NVECS);
    static std::vector<double> cpu_svals(specifics::CONT_NVECS);

    // SVD to get orthogonal marg vectors
    cublas_helper.resetStream();
    cusolver_helper.svd(temp_v, dev_svals.get(), size(), specifics::CONT_NVECS);
    dev_svals.syncDownload(cpu_svals.data(), cpu_svals.size());
    LOG::LOGGER.DEB("SVD'ed\n");

    // Remove each 
    for (int i = 0; i < specifics::CONT_NVECS; ++i, temp_v += size())
    {
        LOG::LOGGER.DEB("i: %d, s: %.2e\n", i, cpu_svals[i]);
        // skip if this vector is degenerate
        if (cpu_svals[i] < 1e-6)  continue;

        _remShermanMorrison(temp_v, size(), temp_y, inverse_covariance_matrix);
    }
}

// Calculate the inverse into temp_matrix[0]
// Then swap the pointer with covariance matrix
void Chunk::invertCovarianceMatrix()
{
    LOG::LOGGER.DEB("Inverting cov matrix\n");
    double t = mytime::timer.getTime();

    cusolver_helper.invert_cholesky(covariance_matrix.get(), size());

    inverse_covariance_matrix = covariance_matrix.get();

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
    cublas_helper.dsymm(CUBLAS_SIDE_LEFT,
        size(), size(), 1., inverse_covariance_matrix, size(),
        m, size(), 0, temp_matrix[1].get(), size());

    //C-1 . Q . C-1
    cublas_helper.dsymm(CUBLAS_SIDE_RIGHT,
        size(), size(), 1., inverse_covariance_matrix, size(),
        temp_matrix[1].get(), size(), 0, m, size());

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_modqs += t;
}

void Chunk::_getFisherMatrix(const double *Qw_ikz_matrix, int idx)
{
    double t = mytime::timer.getTime();
    int i_kz = i_kz_vector[idx];
    int idx_fji_0 =
        bins::TOTAL_KZ_BINS * (i_kz + fisher_index_start)
        + fisher_index_start;
    cublas_helper.setPointerMode2Device();
    std::vector<std::unique_ptr<MyCuStream>> streams;

    // Now compute Fisher Matrix
    for (int jdx = idx; jdx < i_kz_vector.size(); ++jdx) {
        int j_kz = i_kz_vector[jdx];

        #ifdef FISHER_OPTIMIZATION
        int diff_ji = j_kz - i_kz;

        if ((diff_ji != 0) && (diff_ji != 1) && (diff_ji != bins::NUMBER_OF_K_BANDS))
            continue;
        #endif

        streams.push_back(std::make_unique<MyCuStream>());
        cublas_helper.setStream(*streams.back());

        double *Q_jkz_matrix = _getDevQikz(jdx);
        double *fij = dev_fisher.get() + j_kz + idx_fji_0;

        cublas_helper.trace_dsymm(Qw_ikz_matrix, Q_jkz_matrix, size(), fij);
    }

    // for (auto &stream : streams)
    //     stream.sync();

    cublas_helper.resetStream();
    cublas_helper.setPointerMode2Host();

    t = mytime::timer.getTime() - t;

    mytime::time_spent_set_fisher += t;
}

void Chunk::computePSbeforeFvector()
{
    LOG::LOGGER.DEB("PS before Fisher\n");

    double
        *Q_ikz_matrix = temp_matrix[0].get(),
        *dk0 = dbt_estimate_before_fisher_vector[0].get() + fisher_index_start,
        *nk0 = dbt_estimate_before_fisher_vector[1].get() + fisher_index_start,
        *tk0 = dbt_estimate_before_fisher_vector[2].get() + fisher_index_start;

    LOG::LOGGER.DEB("PSb4F -> weighted data\n");
    cublas_helper.dsmyv(
        size(), 1., inverse_covariance_matrix, 
        size(), dev_delta.get(), 1, 0, weighted_data_vector.get(), 1);

    for (int idx = 0; idx < i_kz_vector.size(); ++idx) {
        int i_kz = i_kz_vector[idx];
        double *dk = dk0 + i_kz, *nk = nk0 + i_kz, *tk = tk0 + i_kz;

        LOG::LOGGER.DEB("PSb4F -> loop %d\n", i_kz);
        LOG::LOGGER.DEB("   -> set qi   ");
        // Set derivative matrix ikz
        cublas_helper.dcopy(_getDevQikz(idx), Q_ikz_matrix, DATA_SIZE_2);

        // Find data contribution to ps before F vector
        // (C-1 . flux)T . Q . (C-1 . flux)
        cublas_helper.my_cublas_dsymvdot(
            weighted_data_vector.get(), 
            Q_ikz_matrix, temp_vector.get(), size(), dk);
        LOG::LOGGER.DEB("-> dk (%.1e)   ", *dk);

        LOG::LOGGER.DEB("-> weighted Q   ");
        // Get weighted derivative matrix ikz: C-1 Qi C-1
        _getWeightedMatrix(Q_ikz_matrix);

        LOG::LOGGER.DEB("-> nk   ");
        // Get Noise contribution: Tr(C-1 Qi C-1 N)
        cublas_helper.trace_ddiagmv(Q_ikz_matrix, dev_noise.get(), size(), nk);

        // Set Fiducial Signal Matrix
        if (!specifics::TURN_OFF_SFID)
        {
            LOG::LOGGER.DEB("-> tk   ");
            // Tr(C-1 Qi C-1 Sfid)
            cublas_helper.trace_dsymm(Q_ikz_matrix, dev_sfid.get(), size(), tk);
        }

        // Do not compute fisher matrix if it is precomputed
        if (!specifics::USE_PRECOMPUTED_FISHER)
            _getFisherMatrix(Q_ikz_matrix, idx);

        LOG::LOGGER.DEB("\n");
    }
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

    _initIteration();

    setCovarianceMatrix(ps_estimate);

    try {
        invertCovarianceMatrix();

        computePSbeforeFvector();
        dev_fisher.syncDownload(fisher_sum, bins::FISHER_SIZE);

        mxhelp::vector_add(fisher_sum, fisher_matrix.get(), bins::FISHER_SIZE);

        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            mxhelp::vector_add(
                dbt_sum_vector[dbt_i].get(), 
                dbt_estimate_before_fisher_vector[dbt_i].get(),
                bins::TOTAL_KZ_BINS);
    }
    catch (std::exception& e)
    {
        LOG::LOGGER.ERR(
            "ERROR %s: Covariance matrix is not invertable. %s\n"
            "Npixels: %d, Median z: %.2f, dv: %.2f, R=%d\n",
            e.what(), qFile->fname.c_str(),
            size(), MEDIAN_REDSHIFT, qFile->dv_kms, qFile->R_fwhm);
    }

    _freeMatrices();
}

void Chunk::_initIteration() {
    _allocateCuda();
    _allocateCpu();

    // but smooth noise add save dev_smnoise
    process::noise_smoother->smoothNoise(qFile->noise(), cpu_qj, size());
    dev_smnoise.asyncCpy(cpu_qj, size());

    LOG::LOGGER.DEB("Setting v & z matrices\n");
    _setVZMatrices();

    MyCuStream::syncMainStream();

    // Preload fiducial signal matrix if memory allows
    if (!specifics::TURN_OFF_SFID) {
        _setFiducialSignalMatrix(cpu_sfid);
        dev_sfid.asyncCpy(cpu_sfid, DATA_SIZE_2);
    }

    // Preload last nqj_eff matrices
    // 0 is the last matrix
    // i_kz = N_Q_MATRICES - j_kz - 1
    LOG::LOGGER.DEB("Setting qi matrices\n");

    for (int jdx = 0; jdx < i_kz_vector.size(); ++jdx) {
        int j_kz = i_kz_vector[jdx];
        _setQiMatrix(cpu_qj + jdx * DATA_SIZE_2, N_Q_MATRICES - j_kz - 1);
        dev_qj.asyncCpy(
            cpu_qj + jdx * DATA_SIZE_2, DATA_SIZE_2, jdx * DATA_SIZE_2);
    }
}

void Chunk::_allocateCuda() {
    // Move qfile to gpu
    dev_wave.realloc(size(), qFile->wave());
    dev_delta.realloc(size(), qFile->delta());
    dev_noise.realloc(size(), qFile->noise());
    dev_smnoise.realloc(size());

    dev_fisher.realloc(bins::FISHER_SIZE);
    cudaMemset(dev_fisher.get(), 0, bins::FISHER_SIZE * sizeof(double));

    covariance_matrix.realloc(DATA_SIZE_2);

    for (int i = 0; i < 2; ++i)
        temp_matrix[i].realloc(DATA_SIZE_2);

    temp_vector.realloc(size());
    weighted_data_vector.realloc(size());

    dev_qj.realloc(i_kz_vector.size()*DATA_SIZE_2);

    if (!specifics::TURN_OFF_SFID)
        dev_sfid.realloc(DATA_SIZE_2);
}

void Chunk::_allocateCpu() {
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        dbt_estimate_before_fisher_vector.push_back(
            std::make_unique<double[]>(bins::TOTAL_KZ_BINS));

    fisher_matrix = std::make_unique<double[]>(bins::FISHER_SIZE);

    cpu_qj = new double[i_kz_vector.size() * DATA_SIZE_2];

    if (!specifics::TURN_OFF_SFID)
        cpu_sfid = new double[DATA_SIZE_2];

    // Create a temp highres lambda array
    if (on_oversampling)
    {
        int ncols = qFile->Rmat->getNCols();
        _finer_lambda = new double[ncols];

        double fine_dlambda = qFile->dlambda / specifics::OVERSAMPLING_FACTOR;
        int disp = qFile->Rmat->getNElemPerRow() / 2;
        for (int i = 0; i < ncols; ++i)
            _finer_lambda[i] = qFile->wave()[0] + (i - disp) * fine_dlambda;

        _matrix_lambda = _finer_lambda;

        long highsize = (long)(ncols) * (long)(ncols);
        _finer_matrix = new double[highsize];
        _vmatrix = new double[highsize];
        _zmatrix = new double[highsize];
    }
    else
    {
        _matrix_lambda = qFile->wave();
        _finer_lambda  = NULL;
        _finer_matrix  = NULL;
        _vmatrix = new double[DATA_SIZE_2];
        _zmatrix = new double[DATA_SIZE_2];
    }

    // This function allocates new signal & deriv matrices 
    // if process::SAVE_ALL_SQ_FILES=false 
    // i.e., no caching of SQ files
    // If all tables are cached, then this function simply points 
    // to those in process:sq_private_table
    process::sq_private_table->readSQforR(
        RES_INDEX, interp2d_signal_matrix, interp_derivative_matrix);
}

void Chunk::_freeCuda() {
    dev_wave.reset();
    dev_delta.reset();
    dev_noise.reset();
    dev_smnoise.reset();

    dev_fisher.reset();
    LOG::LOGGER.DEB("Free cov\n");
    covariance_matrix.reset();

    LOG::LOGGER.DEB("Free temps\n");
    for (int i = 0; i < 2; ++i)
        temp_matrix[i].reset();

    temp_vector.reset();
    weighted_data_vector.reset();
    dev_qj.reset();

    if (!specifics::TURN_OFF_SFID)
        dev_sfid.reset();
}

void Chunk::_freeCpu() {
    dbt_estimate_before_fisher_vector.clear();

    LOG::LOGGER.DEB("Free fisher\n");
    fisher_matrix.reset();

    LOG::LOGGER.DEB("Free storedqj\n");
    delete [] cpu_qj;

    LOG::LOGGER.DEB("Free stored sfid\n");
    if (!specifics::TURN_OFF_SFID)
        delete [] cpu_sfid;

    LOG::LOGGER.DEB("Free resomat related\n");
    if (specifics::USE_RESOLUTION_MATRIX)
        qFile->Rmat->freeBuffer();
    if (on_oversampling)
    {
        delete [] _finer_matrix;
        delete [] _finer_lambda;
    }
    delete [] _vmatrix;
    delete [] _zmatrix;
}

void Chunk::_freeMatrices()
{
    LOG::LOGGER.DEB("Freeing matrices\n");

    _freeCuda();
    _freeCpu();

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
        mxhelp::fprintfMatrix(buf.c_str(), cpu_sfid, size(), size());
    }

    if (qFile->Rmat != NULL)
    {
        buf = std::string(fname_base) + "-resolution.txt";
        qFile->Rmat->fprintfMatrix(buf.c_str());
    }
}






