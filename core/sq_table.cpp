#include "core/sq_table.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm> // std::copy
#include <stdexcept>

#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "mathtools/matrix_helper.hpp"
#include "mathtools/fourier_integrator.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"
#include "io/logger.hpp"

std::unique_ptr<SQLookupTable> process::sq_private_table;

std::unique_ptr<double[]>
LINEAR_V_ARRAY, LINEAR_Z_ARRAY, signal_array, derivative_array;

void deallocateSignalAndDerivArrays()
{
    derivative_array.reset();
    signal_array.reset();

    if (!specifics::TURN_OFF_SFID)
        signal_array.reset();
}

void deallocateVAndZArrays()
{
    LINEAR_V_ARRAY.reset();

    if (!specifics::TURN_OFF_SFID)
        LINEAR_Z_ARRAY.reset();
}


// Internal Functions and Variables
namespace sqhelper
{
    double getLinearSpacing(double length, int N)
    {
        double r = (N==1) ? 0 : (length / (N - 1.));
        return r;
    }

    // Allocates a double array with size N
    // Must deallocate after!!
    std::unique_ptr<double[]> allocLinearlySpacedArray(double x0, double lengthx, int N)
    {
        auto resulting_array = std::make_unique<double[]>(N);
        double dx = getLinearSpacing(lengthx, N);

        for (int j = 0; j < N; ++j)
            resulting_array[j] = x0 + dx * j;

        return resulting_array;
    }

    inline
    bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8)
    {
        double mag = std::max(fabs(a),fabs(b));
        return fabs(a-b) < (abserr + relerr * mag);
    }

    void _checkHeaderConsistency(const SQ_IO_Header &hdr,
        int nvpts, int nzpts, double vlen, double z1, double zlen,
        int Rthis, double dvthis, double k1, double k2)
    {
        std::vector<std::string> inconsistencies;
        if (hdr.nvpoints != nvpts)
            inconsistencies.push_back("nvpoints ");
        if (hdr.nzpoints != nzpts)
            inconsistencies.push_back("nzpoints ");
        if (hdr.v_length != vlen)
            inconsistencies.push_back("v_length ");
        if (hdr.z1 != z1)
            inconsistencies.push_back("z1 ");
        if (hdr.z_length != zlen)
            inconsistencies.push_back("z_length ");
        if (hdr.spectrograph_resolution != Rthis)
            inconsistencies.push_back("spectrograph_resolution ");
        if (not isClose(hdr.pixel_width, dvthis))
            inconsistencies.push_back("pixel_width ");
        if (not isClose(hdr.k1, k1))
            inconsistencies.push_back("k1 ");
        if (not isClose(hdr.k2, k2))
            inconsistencies.push_back("k2 ");

        if (not inconsistencies.empty())
        {
            int total_chars = 1;
            for (const auto &s : inconsistencies)
                total_chars += s.size();

            std::string msg("Header is inconsistent at ");
            msg.reserve(msg.size()+total_chars);
            for_each(inconsistencies.begin(), inconsistencies.end(),
                [&msg] (const std::string& s) { msg += s; });
            
            throw std::runtime_error(msg);
        }
    }
}
// ---------------------------------------

/* This function reads following keys from config file:
NumberVPoints: int
NumberZPoints: int
VelocityLength: double
LookUpTableDir: string
    Saves into directory. Default is current dir.
SignalLookUpTableBase: string
    Default is signal.
DerivativeSLookUpTableBase: string
    Default is deriv.
FileNameRList: string
*/
SQLookupTable::SQLookupTable(ConfigFile &config)
{
    N_V_POINTS = config.getInteger("NumberVPoints");
    N_Z_POINTS_OF_S = config.getInteger("NumberZPoints");
    LENGTH_V = config.getDouble("VelocityLength");
    LENGTH_Z_OF_S = bins::Z_BIN_WIDTH * (bins::NUMBER_OF_Z_BINS+1);
    itp_v1 = 0;
    itp_z1 = bins::ZBIN_CENTERS[0] - bins::Z_BIN_WIDTH;

    DIR = config.get("LookUpTableDir", ".");
    S_BASE = config.get("SignalLookUpTableBase", "signal");
    Q_BASE = config.get("DerivativeSLookUpTableBase", "deriv");

    std::string frlist = config.get("FileNameRList");
    if (frlist.empty())
        throw std::invalid_argument("Must pass FileNameRList.");

    NUMBER_OF_R_VALUES = ioh::readListRdv(frlist.c_str(), R_DV_VALUES);

    LOG::LOGGER.STD("Number of R-dv pairs: %d\n", NUMBER_OF_R_VALUES);

    LINEAR_V_ARRAY = sqhelper::allocLinearlySpacedArray(
        itp_v1, LENGTH_V, N_V_POINTS);
    
    if (!specifics::TURN_OFF_SFID)
        LINEAR_Z_ARRAY = sqhelper::allocLinearlySpacedArray(
            itp_z1, LENGTH_Z_OF_S, N_Z_POINTS_OF_S);
}

void SQLookupTable::readTables()
{
    allocateSignalAndDerivArrays();
    LOG::LOGGER.STD("Setting tables.\n");

    if (!specifics::TURN_OFF_SFID)
        interp2d_signal_matrices.reserve(NUMBER_OF_R_VALUES);
    interp_derivative_matrices.reserve(NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS);

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
    {
        if (!specifics::TURN_OFF_SFID)
            interp2d_signal_matrices.push_back(_allocReadSFile(r));
        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
            interp_derivative_matrices.push_back(_allocReadQFile(kn, r));
    }

    deallocateSignalAndDerivArrays();
}

void SQLookupTable::computeTables(bool force_rewrite)
{       
    std::string buf_fnames;
    double time_spent_table_sfid, time_spent_table_q;

    struct spectrograph_windowfn_params 
        win_params = {0, 0, 0, 0};
    struct sq_integrand_params
        integration_parameters = {&fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};
    sqhelper::SQ_IO_Header tmp_hdr = { N_V_POINTS, 0, LENGTH_V, itp_z1, bins::Z_BIN_WIDTH, 
                0, 0, 0, 0};

    allocateSignalAndDerivArrays();

    // Initialize loop for parallel computing
    int delta_nr     = NUMBER_OF_R_VALUES / process::total_pes, 
        r_start_this = delta_nr * process::this_pe, 
        r_end_this   = delta_nr * (process::this_pe+1);

    if (process::this_pe == process::total_pes-1)
        r_end_this = NUMBER_OF_R_VALUES;

    // Integrate derivative matrices
    // int subNz = Nz / NUMBER_OF_Z_BINS;
    FourierIntegrator q_integrator(GSL_INTEG_COSINE, 
        q_matrix_integrand, &integration_parameters);

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_q = mytime::timer.getTime();

        int Rthis = R_DV_VALUES[r].first;
        double dvthis = R_DV_VALUES[r].second;

        win_params.spectrograph_res = SPEED_OF_LIGHT / Rthis / ONE_SIGMA_2_FWHM;
        win_params.pixel_width = dvthis;
        tmp_hdr.spectrograph_resolution = Rthis;
        tmp_hdr.pixel_width = dvthis;

        LOG::LOGGER.STD("Creating look up tables for derivative signal matrices."
            " R=%d (%.2f km/s), dv=%.1f.\n", Rthis, win_params.spectrograph_res, dvthis);

        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        {
            tmp_hdr.k1 = bins::KBAND_EDGES[kn];
            tmp_hdr.k2 = bins::KBAND_EDGES[kn+1];

            LOG::LOGGER.STD("Q matrix for k=[%.1e - %.1e] s/km.\n",
                tmp_hdr.k1, tmp_hdr.k2);

            buf_fnames = sqhelper::QTableFileNameConvention(DIR, Q_BASE, Rthis, 
                dvthis, tmp_hdr.k1, tmp_hdr.k2);
            
            if (!force_rewrite && ioh::file_exists(buf_fnames.c_str()))
            {
                LOG::LOGGER.STD("File %s already exists. Skip to next.\n", 
                    buf_fnames.c_str());
                continue;
            }

            for (int nv = 0; nv < N_V_POINTS; ++nv)
            {
                win_params.delta_v_ij = LINEAR_V_ARRAY[nv];

                derivative_array[nv] = q_integrator.evaluate(tmp_hdr.k1, tmp_hdr.k2, 
                    win_params.delta_v_ij, 0);
            }

            sqhelper::SQLookupTableFile derivative_signal_table(buf_fnames, 'w');
            derivative_signal_table.setHeader(tmp_hdr);
            derivative_signal_table.writeData(derivative_array.get());
        }
        
        time_spent_table_q = mytime::timer.getTime() - time_spent_table_q;
        LOG::LOGGER.STD("Time spent on derivative matrix table R=%d, dv=%.1f is %.2f mins.\n",
            Rthis, dvthis, time_spent_table_q);
    }
    // Q matrices are written.
    // ---------------------

    // Integrate fiducial signal matrix
    FourierIntegrator s_integrator(GSL_INTEG_COSINE, signal_matrix_integrand, 
        &integration_parameters);
    
    // Skip this section if fiducial signal matrix is turned off.
    if (specifics::TURN_OFF_SFID) return;

    tmp_hdr = {
        N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, itp_z1, LENGTH_Z_OF_S, 0, 0, 
        fidcosmo::FID_LOWEST_K, fidcosmo::FID_HIGHEST_K};

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_sfid = mytime::timer.getTime();
        int Rthis = R_DV_VALUES[r].first;
        double dvthis = R_DV_VALUES[r].second;

        // Convert integer FWHM to 1 sigma km/s
        win_params.spectrograph_res = SPEED_OF_LIGHT / Rthis / ONE_SIGMA_2_FWHM;
        win_params.pixel_width = dvthis;
        tmp_hdr.spectrograph_resolution = Rthis;
        tmp_hdr.pixel_width = dvthis;

        LOG::LOGGER.STD("Creating look up table for signal matrix R=%d (%.2f km/s), dv=%.1f.\n",
            Rthis, win_params.spectrograph_res, dvthis);

        buf_fnames = sqhelper::STableFileNameConvention(DIR, S_BASE, Rthis, dvthis);
        
        if (!force_rewrite && ioh::file_exists(buf_fnames.c_str()))
        {
            LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf_fnames.c_str());
            continue;
        }

        for (int nv = 0; nv < N_V_POINTS; ++nv)
        {
            win_params.delta_v_ij = LINEAR_V_ARRAY[nv];  // 0 + LENGTH_V * nv / (Nv - 1.);
            s_integrator.setTableParameters(
                win_params.delta_v_ij, tmp_hdr.k2 - tmp_hdr.k1);

            for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz)
            {
                int xy = nz + N_Z_POINTS_OF_S * nv;
                win_params.z_ij = LINEAR_Z_ARRAY[nz];   // z_first + z_length * nz / (Nz - 1.);
                
                // 1E-15 gave roundoff error for smoothing with 20.8 km/s
                // Correlation at dv=0 is between 0.01 and 1. 
                // Giving room for 7 decades, absolute error can be 1e-9
                signal_array[xy] = s_integrator.evaluate(
                    tmp_hdr.k1, tmp_hdr.k2, -1, 1E-9);
            }
        }
        
        sqhelper::SQLookupTableFile signal_table(buf_fnames, 'w');

        signal_table.setHeader(tmp_hdr);
        signal_table.writeData(signal_array.get());

        time_spent_table_sfid = mytime::timer.getTime() - time_spent_table_sfid;

        LOG::LOGGER.STD("Time spent on fiducial signal matrix table R=%d, dv=%.1f is %.2f mins.\n",
            Rthis, dvthis, time_spent_table_sfid);
    }

    LOG::LOGGER.STD("Deallocating arrays...\n");
    deallocateSignalAndDerivArrays();
}

shared_interp_2d SQLookupTable::_allocReadSFile(int r_index)
{
    std::string buf_fnames;
    sqhelper::SQ_IO_Header tmp_hdr;

    int Rthis = R_DV_VALUES[r_index].first;
    double dvthis = R_DV_VALUES[r_index].second;

    // Read S table.
    buf_fnames = sqhelper::STableFileNameConvention(DIR, S_BASE, Rthis, dvthis);
    LOG::LOGGER.STD("Reading sq_lookup_table_file %s.\n", buf_fnames.c_str());
    
    sqhelper::SQLookupTableFile s_table_file(buf_fnames, 'r');
    
    tmp_hdr = s_table_file.readHeader();
    sqhelper::_checkHeaderConsistency(tmp_hdr, N_V_POINTS, N_Z_POINTS_OF_S, 
        LENGTH_V, itp_z1, LENGTH_Z_OF_S, Rthis, dvthis,
        fidcosmo::FID_LOWEST_K, fidcosmo::FID_HIGHEST_K);

    // Allocate memory before reading further
    if (!derivative_array)
        allocateSignalAndDerivArrays();

    // Start reading data and set to intep pointer
    s_table_file.readData(signal_array.get());
    itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);
    itp_dz = sqhelper::getLinearSpacing(LENGTH_Z_OF_S, N_Z_POINTS_OF_S);

    return std::make_shared<DiscreteInterpolation2D>(
        itp_z1, itp_dz, itp_v1, itp_dv,
        signal_array.get(), N_Z_POINTS_OF_S, N_V_POINTS);
}

shared_interp_1d SQLookupTable::_allocReadQFile(int kn, int r_index)
{
    std::string buf_fnames;
    sqhelper::SQ_IO_Header tmp_hdr;
    int Rthis = R_DV_VALUES[r_index].first;
    double dvthis = R_DV_VALUES[r_index].second;
    double kvalue_1, kvalue_2;
    
    kvalue_1 = bins::KBAND_EDGES[kn];
    kvalue_2 = bins::KBAND_EDGES[kn + 1];

    buf_fnames = sqhelper::QTableFileNameConvention(DIR, Q_BASE, Rthis, dvthis, 
        kvalue_1, kvalue_2);

    sqhelper::SQLookupTableFile q_table_file(buf_fnames, 'r');

    tmp_hdr = q_table_file.readHeader();
    sqhelper::_checkHeaderConsistency(tmp_hdr, N_V_POINTS, 0, LENGTH_V,
        itp_z1, bins::Z_BIN_WIDTH, Rthis, dvthis, kvalue_1, kvalue_2);

    if (!derivative_array)
        allocateSignalAndDerivArrays();
    
    q_table_file.readData(derivative_array.get());

    itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);

    return std::make_shared<DiscreteInterpolation1D>(
        itp_v1, itp_dv, N_V_POINTS, derivative_array.get());
}

void SQLookupTable::readSQforR(
        int r_index, shared_interp_2d &s,
        std::vector<shared_interp_1d> &q, bool alloc
) {
    // Skip this section if fiducial signal matrix is turned off.
    if (!specifics::TURN_OFF_SFID)
    {
        if (alloc || interp2d_signal_matrices.empty())
            s = _allocReadSFile(r_index);
        else
            s = getSignalMatrixInterp(r_index);
    }

    q.clear();
    // Read Q tables. 
    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {            
        if (alloc || interp_derivative_matrices.empty())
            q.push_back(_allocReadQFile(kn, r_index));
        else
            q.push_back(getDerivativeMatrixInterp(kn, r_index));
    }
}

void SQLookupTable::computeDerivativeMatrices(
        int r_index,
        DiscreteInterpolation1D *interpLnW2,
        shared_interp_2d &s,
        std::vector<shared_interp_1d>  &q
) {
    // signal matrix can be calculated as well in the future.
    if (!specifics::TURN_OFF_SFID)
    {
        if (interp2d_signal_matrices.empty())
            s = _allocReadSFile(r_index);
        else
            s = getSignalMatrixInterp(r_index);
    }

    q.clear();
    double kcenter = 0, itp_dv = LINEAR_V_ARRAY[1] - LINEAR_V_ARRAY[0];
    static struct new_q_integrand_params integration_parameters = { nullptr, 0. };
    static FourierIntegrator q_integrator(
        GSL_INTEG_COSINE, new_q_matrix_integrand, &integration_parameters);

    integration_parameters.interpLnW2 = interpLnW2;

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        shared_interp_1d qkn = std::make_shared<DiscreteInterpolation1D>(
            0, itp_dv, N_V_POINTS);
        double *derivative_array = qkn->get();
        kcenter = bins::KBAND_CENTERS[kn];
        integration_parameters.lnW2kc = interpLnW2->evaluate(kcenter);

        for (int nv = 0; nv < N_V_POINTS; ++nv)
            derivative_array[nv] = q_integrator.evaluate(
                bins::KBAND_EDGES[kn], bins::KBAND_EDGES[kn + 1], 
                LINEAR_V_ARRAY[nv], 0);

        kcenter = exp(integration_parameters.lnW2kc);
        cblas_dscal(N_V_POINTS, kcenter, derivative_array, 1);
        q.push_back(qkn);
    }
}

int SQLookupTable::findSpecResIndex(int spec_res, double dv) const
{
    double rounded_dv = round(dv/5)*5;
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        if (R_DV_VALUES[r].first == spec_res && fabs(rounded_dv - R_DV_VALUES[r].second)<0.01)
            return r;

    return -1;
}

double SQLookupTable::getOneSetMemUsage()
{
    double s = process::getMemoryMB(N_V_POINTS * N_Z_POINTS_OF_S),
           q = process::getMemoryMB(N_V_POINTS * bins::NUMBER_OF_K_BANDS);

    if (specifics::TURN_OFF_SFID) s = 0;

    return s+q;
}

double SQLookupTable::getMaxMemUsage()
{
    double rdvsize = 1.5 * process::getMemoryMB(NUMBER_OF_R_VALUES);

    return getOneSetMemUsage() * NUMBER_OF_R_VALUES + rdvsize;
}

void SQLookupTable::allocateSignalAndDerivArrays()
{
    derivative_array = std::make_unique<double[]>(N_V_POINTS);

    // Allocate N_Z_POINTS_OF_S dependent arrays
    if (!specifics::TURN_OFF_SFID)
        signal_array = std::make_unique<double[]>(N_V_POINTS * N_Z_POINTS_OF_S);
}










