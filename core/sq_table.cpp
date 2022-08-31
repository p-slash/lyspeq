#include "core/sq_table.hpp"

#include <cstdio>
#include <algorithm> // std::copy
#include <stdexcept>

#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "mathtools/fourier_integrator.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"
#include "io/logger.hpp"

#define ONE_SIGMA_2_FWHM 2.35482004503

std::unique_ptr<SQLookupTable> process::sq_private_table;

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
            resulting_array.get()[j] = x0 + dx * j;

        return resulting_array;
    }
}
// ---------------------------------------

/* This function reads following keys from config file:
NumberVPoints: int
NumberZPoints: int
VelocityLength: double
OutputDir: string
    Saves into directory. Default is current dir.
SignalLookUpTableBase: string
    Default is signal.
DerivativeSLookUpTableBase: string
    Default is deriv.
FileNameRList: string
*/
SQLookupTable::SQLookupTable(const ConfigFile &config)
{
    N_V_POINTS = config.getInteger("NumberVPoints");
    N_Z_POINTS_OF_S = config.getInteger("NumberZPoints");
    LENGTH_V = config.getDouble("VelocityLength");
    DIR = config.get("OutputDir", ".");
    S_BASE = config.get("SignalLookUpTableBase", "signal");
    Q_BASE = config.get("DerivativeSLookUpTableBase", "deriv");
    itp_v1 = 0;

    itp_z1 = bins::ZBIN_CENTERS[0] - bins::Z_BIN_WIDTH;

    std::string frlist = config.get("FileNameRList");
    if (frlist.empty())
        throw std::invalid_argument("Must pass FileNameRList.");

    NUMBER_OF_R_VALUES = ioh::readListRdv(frlist.c_str(), R_DV_VALUES);

    LOG::LOGGER.STD("Number of R-dv pairs: %d\n", NUMBER_OF_R_VALUES);}

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
    LENGTH_Z_OF_S = bins::Z_BIN_WIDTH * (bins::NUMBER_OF_Z_BINS+1);
           
    std::string buf_fnames;
    double time_spent_table_sfid, time_spent_table_q;

    struct spectrograph_windowfn_params 
        win_params = {0, 0, 0, 0};
    struct sq_integrand_params
        integration_parameters = {&fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};

    allocateSignalAndDerivArrays();
    allocateVAndZArrays();

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

        LOG::LOGGER.STD("Creating look up tables for derivative signal matrices."
            " R=%d (%.2f km/s), dv=%.1f.\n", Rthis, win_params.spectrograph_res, dvthis);

        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        {
            double kvalue_1 = bins::KBAND_EDGES[kn];
            double kvalue_2 = bins::KBAND_EDGES[kn + 1];

            LOG::LOGGER.STD("Q matrix for k=[%.1e - %.1e] s/km.\n", kvalue_1, kvalue_2);

            buf_fnames = sqhelper::QTableFileNameConvention(DIR, Q_BASE, Rthis, 
                dvthis, kvalue_1, kvalue_2);
            
            if (!force_rewrite && ioh::file_exists(buf_fnames.c_str()))
            {
                LOG::LOGGER.STD("File %s already exists. Skip to next.\n", 
                    buf_fnames.c_str());
                continue;
            }

            for (int nv = 0; nv < N_V_POINTS; ++nv)
            {
                win_params.delta_v_ij = LINEAR_V_ARRAY.get()[nv];

                derivative_array[nv] = q_integrator.evaluate(kvalue_1, kvalue_2, 
                    win_params.delta_v_ij, 0);
            }

            SQLookupTableFile derivative_signal_table(buf_fnames, 'w');

            derivative_signal_table.setHeader(N_V_POINTS, 0, LENGTH_V, bins::Z_BIN_WIDTH, 
                Rthis, dvthis, kvalue_1, kvalue_2);
            
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

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_sfid = mytime::timer.getTime();
        int Rthis = R_DV_VALUES[r].first;
        double dvthis = R_DV_VALUES[r].second;

        // Convert integer FWHM to 1 sigma km/s
        win_params.spectrograph_res = SPEED_OF_LIGHT / Rthis / ONE_SIGMA_2_FWHM;
        win_params.pixel_width = dvthis;

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
            win_params.delta_v_ij = LINEAR_V_ARRAY.get()[nv];  // 0 + LENGTH_V * nv / (Nv - 1.);
            s_integrator.setTableParameters(win_params.delta_v_ij, 
                fidcosmo::FID_HIGHEST_K - fidcosmo::FID_LOWEST_K);

            for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz)
            {
                int xy = nz + N_Z_POINTS_OF_S * nv;
                win_params.z_ij = LINEAR_Z_ARRAY.get()[nz];   // z_first + z_length * nz / (Nz - 1.);
                
                // 1E-15 gave roundoff error for smoothing with 20.8 km/s
                // Correlation at dv=0 is between 0.01 and 1. 
                // Giving room for 7 decades, absolute error can be 1e-9
                signal_array[xy] = s_integrator.evaluate(fidcosmo::FID_LOWEST_K, 
                    fidcosmo::FID_HIGHEST_K, -1, 1E-9);
            }
        }
        
        SQLookupTableFile signal_table(buf_fnames, 'w');

        signal_table.setHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S, Rthis, dvthis, 
            0, bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS]);

        signal_table.writeData(signal_array.get());

        time_spent_table_sfid = mytime::timer.getTime() - time_spent_table_sfid;

        LOG::LOGGER.STD("Time spent on fiducial signal matrix table R=%d, dv=%.1f is %.2f mins.\n",
            Rthis, dvthis, time_spent_table_sfid);
    }

    LOG::LOGGER.STD("Deallocating arrays...\n");
    deallocateSignalAndDerivArrays();
    deallocateVAndZArrays();
}

shared_interp_2d SQLookupTable::_allocReadSFile(int r_index)
{
    std::string buf_fnames;
    int dummy_R;
    double temp_px_width, temp_ki, temp_kf;

    int Rthis = R_DV_VALUES[r_index].first;
    double dvthis = R_DV_VALUES[r_index].second;

    // Read S table.
    buf_fnames = sqhelper::STableFileNameConvention(DIR, S_BASE, Rthis, dvthis);
    LOG::LOGGER.STD("Reading sq_lookup_table_file %s.\n", buf_fnames.c_str());
    
    SQLookupTableFile s_table_file(buf_fnames, 'r');
    
    s_table_file.readHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S, dummy_R, 
        temp_px_width, temp_ki, temp_kf);

    // Allocate memory before reading further
    if (!derivative_array)
        allocateSignalAndDerivArrays();

    // Start reading data and set to intep pointer
    s_table_file.readData(signal_array.get());
    itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);
    itp_dz = sqhelper::getLinearSpacing(LENGTH_Z_OF_S, N_Z_POINTS_OF_S);

    return std::make_shared<DiscreteInterpolation2D>(itp_z1, itp_dz, itp_v1, itp_dv,
        signal_array.get(), N_Z_POINTS_OF_S, N_V_POINTS);
}

shared_interp_1d SQLookupTable::_allocReadQFile(int kn, int r_index)
{
    std::string buf_fnames;
    int dummy_R, dummy_Nz;
    double temp_px_width, temp_ki, temp_kf;

    int Rthis = R_DV_VALUES[r_index].first;
    double dvthis = R_DV_VALUES[r_index].second;
    double kvalue_1, kvalue_2, dummy_lzq;
    
    kvalue_1 = bins::KBAND_EDGES[kn];
    kvalue_2 = bins::KBAND_EDGES[kn + 1];

    buf_fnames = sqhelper::QTableFileNameConvention(DIR, Q_BASE, Rthis, dvthis, 
        kvalue_1, kvalue_2);

    SQLookupTableFile q_table_file(buf_fnames, 'r');

    q_table_file.readHeader(N_V_POINTS, dummy_Nz, LENGTH_V, dummy_lzq, dummy_R, 
        temp_px_width, temp_ki, temp_kf);

    if (!derivative_array)
        allocateSignalAndDerivArrays();
    
    q_table_file.readData(derivative_array.get());

    itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);

    return std::make_shared<DiscreteInterpolation1D>(itp_v1, itp_dv, derivative_array.get(), 
        N_V_POINTS);
}

void SQLookupTable::readSQforR(int r_index, shared_interp_2d &s,
        std::vector<shared_interp_1d> &q, bool alloc)
{
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

shared_interp_1d SQLookupTable::getDerivativeMatrixInterp(int kn, 
    int r_index) const
{
    return interp_derivative_matrices[getIndex4DerivativeInterpolation(kn ,r_index)];
}

shared_interp_2d SQLookupTable::getSignalMatrixInterp(int r_index) const
{
    return interp2d_signal_matrices[r_index];
}

int SQLookupTable::getIndex4DerivativeInterpolation(int kn, int r_index) const
{
    return kn + bins::NUMBER_OF_K_BANDS * r_index;
}

int SQLookupTable::findSpecResIndex(int spec_res, double dv) const
{
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        if (R_DV_VALUES[r].first == spec_res && fabs(dv - R_DV_VALUES[r].second)<0.01)
            return r;

    return -1;
}

void SQLookupTable::allocateSignalAndDerivArrays()
{
    derivative_array = std::make_unique<double[]>(N_V_POINTS);

    // Allocate N_Z_POINTS_OF_S dependent arrays
    if (!specifics::TURN_OFF_SFID)
        signal_array = std::make_unique<double[]>(N_V_POINTS * N_Z_POINTS_OF_S);
}

void SQLookupTable::allocateVAndZArrays()
{
    LINEAR_V_ARRAY = sqhelper::allocLinearlySpacedArray(itp_v1, 
        LENGTH_V, N_V_POINTS);
    
    if (!specifics::TURN_OFF_SFID)
        LINEAR_Z_ARRAY = sqhelper::allocLinearlySpacedArray(itp_z1, 
            LENGTH_Z_OF_S, N_Z_POINTS_OF_S);
}

void SQLookupTable::deallocateSignalAndDerivArrays()
{
    derivative_array.reset();
    signal_array.reset();
}

void SQLookupTable::deallocateVAndZArrays()
{
    LINEAR_V_ARRAY.reset();
    LINEAR_Z_ARRAY.reset();  
}

double SQLookupTable::getOneSetMemUsage()
{
    double s = (double)sizeof(double) * N_V_POINTS * N_Z_POINTS_OF_S  / 1048576.,
    q = (double)sizeof(double) * N_V_POINTS * bins::NUMBER_OF_K_BANDS / 1048576.;

    if (specifics::TURN_OFF_SFID) s = 0;

    return s+q;
}

double SQLookupTable::getMaxMemUsage()
{
    double rdvsize = (double)sizeof(double) * NUMBER_OF_R_VALUES * 1.5 / 1048576.;

    return getOneSetMemUsage() * NUMBER_OF_R_VALUES + rdvsize;
}












