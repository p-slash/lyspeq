#include "core/sq_table.hpp"

#include <cstdio>
#include <algorithm> // std::copy
#include <stdexcept>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "gsltools/fourier_integrator.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"
#include "io/logger.hpp"

#define ONE_SIGMA_2_FWHM 2.35482004503

// Internal Functions and Variables
namespace sqhelper
{
    // Temporary arrays. They are not stored after construction!
    double *LINEAR_V_ARRAY, *LINEAR_Z_ARRAY, *signal_array, *derivative_array;

    // return y = c + deltaY / (N-1) * n
    double getLinearlySpacedValue(double c, double delta_y, int N, int n)
    {
        if (N == 1)     return c + delta_y/2;

        return c + delta_y / (N - 1.) * n;
    }

    // Allocates a double array with size N
    // Must delocate after!!
    double *allocLinearlySpacedArray(double x0, double lengthx, int N)
    {
        double *resulting_array = new double[N];
        for (int j = 0; j < N; ++j)
            resulting_array[j] = sqhelper::getLinearlySpacedValue(x0, lengthx, N, j);

        return resulting_array;
    }
}
// ---------------------------------------

SQLookupTable::SQLookupTable(const char *dir, const char *s_base, const char *q_base, const char *fname_rlist)
    : DIR(dir), S_BASE(s_base), Q_BASE(q_base)
{
    NUMBER_OF_R_VALUES = ioh::readList(fname_rlist, R_VALUES);

    LOG::LOGGER.STD("Number of R values: %d\n", NUMBER_OF_R_VALUES);

    std::for_each(R_VALUES.begin(), R_VALUES.end(), [&](int &R) { LOG::LOGGER.STD("%d\n", R); } );
}

void SQLookupTable::readTables()
{
    LOG::LOGGER.STD("Setting tables.\n");

    interp2d_signal_matrices     = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices   = new Interpolation*[NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        readSQforR(r);

    deallocateTmpArrays();
}

void SQLookupTable::computeTables(double PIXEL_WIDTH, int Nv, int Nz, double Lv, bool force_rewrite)
{
    N_V_POINTS      = Nv;
    N_Z_POINTS_OF_S = Nz;
    LENGTH_V        = Lv;
    LENGTH_Z_OF_S   = bins::Z_BIN_WIDTH * (bins::NUMBER_OF_Z_BINS+1);
           
    std::string buf_fnames;
    double time_spent_table_sfid, time_spent_table_q;

    struct spectrograph_windowfn_params     win_params             = {0, 0, PIXEL_WIDTH, 0};
    struct sq_integrand_params              integration_parameters = {&fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};
    
    allocateTmpArrays();

    // Initialize loop for parallel computing
    int delta_nr     = NUMBER_OF_R_VALUES / process::total_pes, 
        r_start_this = delta_nr * process::this_pe, 
        r_end_this   = delta_nr * (process::this_pe+1);

    if (process::this_pe == process::total_pes-1)
        r_end_this = NUMBER_OF_R_VALUES;

    // Integrate fiducial signal matrix
    FourierIntegrator s_integrator(GSL_INTEG_COSINE, signal_matrix_integrand, &integration_parameters);
    
    // Skip this section if fiducial signal matrix is turned off.
    if (specifics::TURN_OFF_SFID) goto DERIVATIVE;

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_sfid = mytime::getTime();

        // Convert integer FWHM to 1 sigma km/s
        win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
        
        LOG::LOGGER.STD("T%d/%d - Creating look up table for signal matrix. R = %d : %.2f km/s.\n",
            process::this_pe, process::total_pes, R_VALUES[r], win_params.spectrograph_res);

        buf_fnames = sqhelper::STableFileNameConvention(DIR, S_BASE, R_VALUES[r]);
        
        if (!force_rewrite && ioh::file_exists(buf_fnames.c_str()))
        {
            LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf_fnames.c_str());
            continue;
        }

        for (int nv = 0; nv < N_V_POINTS; ++nv)
        {
            win_params.delta_v_ij = sqhelper::LINEAR_V_ARRAY[nv];  // 0 + LENGTH_V * nv / (Nv - 1.);
            s_integrator.setTableParameters(win_params.delta_v_ij, fidcosmo::FID_HIGHEST_K - fidcosmo::FID_LOWEST_K);

            for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz)
            {
                int xy = nz + N_Z_POINTS_OF_S * nv;
                win_params.z_ij = sqhelper::LINEAR_Z_ARRAY[nz];   // z_first + z_length * nz / (Nz - 1.);
                
                // 1E-15 gave roundoff error for smoothing with 20.8 km/s
                // Correlation at dv=0 is between 0.01 and 1. 
                // Giving room for 7 decades, absolute error can be 1e-9
                sqhelper::signal_array[xy] = s_integrator.evaluate(fidcosmo::FID_LOWEST_K, fidcosmo::FID_HIGHEST_K, -1, 1E-9);
            }
        }
        
        SQLookupTableFile signal_table(buf_fnames, 'w');

        signal_table.setHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S, R_VALUES[r], PIXEL_WIDTH, 0, bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS]);

        signal_table.writeData(sqhelper::signal_array);

        time_spent_table_sfid = mytime::getTime() - time_spent_table_sfid;

        LOG::LOGGER.STD("T:%d/%d - Time spent on fiducial signal matrix table R %d is %.2f mins.\n",
            process::this_pe, process::total_pes, R_VALUES[r], time_spent_table_sfid);
    }

DERIVATIVE:
    // Integrate derivative matrices
    // int subNz = Nz / NUMBER_OF_Z_BINS;
    FourierIntegrator q_integrator(GSL_INTEG_COSINE, q_matrix_integrand, &integration_parameters);

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_q = mytime::getTime();

        win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
        LOG::LOGGER.STD("T:%d/%d - Creating look up tables for derivative signal matrices. R = %d : %.2f km/s.\n",
            process::this_pe, process::total_pes, R_VALUES[r], win_params.spectrograph_res);

        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        {
            double kvalue_1 = bins::KBAND_EDGES[kn];
            double kvalue_2 = bins::KBAND_EDGES[kn + 1];

            LOG::LOGGER.STD("Q matrix for k = [%.1e - %.1e] s/km.\n", kvalue_1, kvalue_2);

            buf_fnames =  sqhelper::QTableFileNameConvention(DIR, Q_BASE, R_VALUES[r], kvalue_1, kvalue_2);
            
            if (!force_rewrite && ioh::file_exists(buf_fnames.c_str()))
            {
                LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf_fnames.c_str());
                continue;
            }

            for (int nv = 0; nv < N_V_POINTS; ++nv)
            {
                win_params.delta_v_ij = sqhelper::LINEAR_V_ARRAY[nv];

                sqhelper::derivative_array[nv] = q_integrator.evaluate(kvalue_1, kvalue_2, win_params.delta_v_ij, 0);
            }

            SQLookupTableFile derivative_signal_table(buf_fnames, 'w');

            derivative_signal_table.setHeader(N_V_POINTS, 0, LENGTH_V, bins::Z_BIN_WIDTH, R_VALUES[r], PIXEL_WIDTH, kvalue_1, kvalue_2);
            
            derivative_signal_table.writeData(sqhelper::derivative_array);
        }
        
        time_spent_table_q = mytime::getTime() - time_spent_table_q;
        LOG::LOGGER.STD("T:%d/%d - Time spent on derivative matrix table R %d is %.2f mins.\n",
                process::this_pe, process::total_pes, R_VALUES[r], time_spent_table_q);
    }
    // Q matrices are written.
    // ---------------------
    deallocateTmpArrays();
}

SQLookupTable::SQLookupTable(const SQLookupTable &sq)
{
    LOG::LOGGER.STD("Copying SQ table.\n");
    
    NUMBER_OF_R_VALUES = sq.NUMBER_OF_R_VALUES;

    N_V_POINTS      = sq.N_V_POINTS;
    N_Z_POINTS_OF_S = sq.N_Z_POINTS_OF_S;

    LENGTH_V      = sq.LENGTH_V;
    LENGTH_Z_OF_S = sq.LENGTH_Z_OF_S;

    R_VALUES = sq.R_VALUES;

    interp2d_signal_matrices   = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices = new Interpolation*[NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        interp2d_signal_matrices[r] = new Interpolation2D(*sq.interp2d_signal_matrices[r]);
        
    for (int q = 0; q < NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS; ++q)
        interp_derivative_matrices[q] = new Interpolation(*sq.interp_derivative_matrices[q]);
}

SQLookupTable::~SQLookupTable()
{
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        delete interp2d_signal_matrices[r];

    for (int kr = 0; kr < NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS; ++kr)
        delete interp_derivative_matrices[kr];

    delete [] interp2d_signal_matrices;
    delete [] interp_derivative_matrices;
}

void SQLookupTable::allocateTmpArrays()
{
    double z1 = bins::ZBIN_CENTERS[0] - bins::Z_BIN_WIDTH;

    // Allocate, set velocity and redshift arrays
    sqhelper::LINEAR_V_ARRAY = sqhelper::allocLinearlySpacedArray(0, LENGTH_V, N_V_POINTS);
    sqhelper::LINEAR_Z_ARRAY = sqhelper::allocLinearlySpacedArray(z1, LENGTH_Z_OF_S, N_Z_POINTS_OF_S);

    // Allocate signal and derivative arrays
    sqhelper::signal_array     = new double[N_V_POINTS * N_Z_POINTS_OF_S];
    sqhelper::derivative_array = new double[N_V_POINTS];
}

void SQLookupTable::deallocateTmpArrays()
{
    delete [] sqhelper::LINEAR_V_ARRAY;
    delete [] sqhelper::LINEAR_Z_ARRAY;

    delete [] sqhelper::signal_array;
    delete [] sqhelper::derivative_array;
}

void SQLookupTable::readSQforR(int r_index)
{
    std::string buf_fnames;
    int dummy_R, dummy_Nz;
    double temp_px_width, temp_ki, temp_kf;

    // Skip this section if fiducial signal matrix is turned off.
    if (specifics::TURN_OFF_SFID)
    {
        // Read S table.
        buf_fnames = sqhelper::STableFileNameConvention(DIR, S_BASE, R_VALUES[r_index]);
        LOG::LOGGER.IO("Reading sq_lookup_table_file %s.\n", buf_fnames.c_str());
        
        SQLookupTableFile s_table_file(buf_fnames, 'r');
        
        s_table_file.readHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S, dummy_R, temp_px_width, temp_ki, temp_kf);

        // Allocate memory before reading further
        if (sqhelper::LINEAR_V_ARRAY == NULL)
            allocateTmpArrays();

        // Start reading data and interpolating
        s_table_file.readData(sqhelper::signal_array);

        // Interpolate
        interp2d_signal_matrices[r_index] = new Interpolation2D(INTERP_2D_TYPE, sqhelper::LINEAR_Z_ARRAY, sqhelper::LINEAR_V_ARRAY, 
            sqhelper::signal_array, N_Z_POINTS_OF_S, N_V_POINTS);
    }

    // Read Q tables. 
    double kvalue_1, kvalue_2, dummy_lzq;

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        kvalue_1 = bins::KBAND_EDGES[kn];
        kvalue_2 = bins::KBAND_EDGES[kn + 1];

        buf_fnames = sqhelper::QTableFileNameConvention(DIR, Q_BASE, R_VALUES[r_index], kvalue_1, kvalue_2);

        SQLookupTableFile q_table_file(buf_fnames, 'r');

        q_table_file.readHeader(N_V_POINTS, dummy_Nz, LENGTH_V, dummy_lzq, dummy_R, temp_px_width, temp_ki, temp_kf);

        q_table_file.readData(sqhelper::derivative_array);

        // Interpolate
        int i = getIndex4DerivativeInterpolation(kn, r_index);

        interp_derivative_matrices[i] = new Interpolation(INTERP_1D_TYPE, sqhelper::LINEAR_V_ARRAY, sqhelper::derivative_array, N_V_POINTS);
    }
}

double SQLookupTable::getSignalMatrixValue(double v_ij, double z_ij, int r_index) const 
{
    return interp2d_signal_matrices[r_index]->evaluate(z_ij, v_ij);
}

double SQLookupTable::getDerivativeMatrixValue(double v_ij, int kn, int r_index) const
{
    return interp_derivative_matrices[getIndex4DerivativeInterpolation(kn ,r_index)]->evaluate(v_ij);
}

Interpolation* SQLookupTable::getDerivativeMatrixInterp(int kn, int r_index) const
{
    return interp_derivative_matrices[getIndex4DerivativeInterpolation(kn ,r_index)];
}

Interpolation2D* SQLookupTable::getSignalMatrixInterp(int r_index) const
{
    return interp2d_signal_matrices[r_index];
}

int SQLookupTable::getIndex4DerivativeInterpolation(int kn, int r_index) const
{
    return kn + bins::NUMBER_OF_K_BANDS * r_index;
}

int SQLookupTable::findSpecResIndex(int spec_res) const
{
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        if (R_VALUES[r] == spec_res)    return r;

    return -1;
}














