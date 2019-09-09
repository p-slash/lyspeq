#include "core/sq_table.hpp"

#include <cstdio>
#include <algorithm> // std::copy
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "gsltools/fourier_integrator.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"
#include "io/logger.hpp"

#define ONE_SIGMA_2_FWHM 2.35482004503

SQLookupTable::SQLookupTable(const char *dir, const char *s_base, const char *q_base, const char *fname_rlist)
:
DIR(dir), S_BASE(s_base), Q_BASE(q_base)
{
    LINEAR_V_ARRAY   = NULL;
    LINEAR_Z_ARRAY   = NULL;
    signal_array     = NULL;
    derivative_array = NULL;

    NUMBER_OF_R_VALUES = ioh::readList(fname_rlist, R_VALUES);

    LOG::LOGGER.STD("Number of R values: %d\n", NUMBER_OF_R_VALUES);

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        LOG::LOGGER.STD("%d\n", R_VALUES[r]);
}

void SQLookupTable::readTables()
{
    LOG::LOGGER.STD("Setting tables..\n");

    interp2d_signal_matrices     = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices   = new Interpolation*[NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        readSQforR(r);

    deallocateTmpArrays();
}

void SQLookupTable::computeTables(double PIXEL_WIDTH, int Nv, int Nz, double Lv, bool force_rewrite)
{
    N_V_POINTS = Nv;
    N_Z_POINTS_OF_S = Nz;
    LENGTH_V = Lv;
    double z_length = bins::Z_BIN_WIDTH * (bins::NUMBER_OF_Z_BINS+1);
    
#pragma omp parallel private(LINEAR_V_ARRAY, LINEAR_Z_ARRAY, signal_array, derivative_array)
{       
    char buf[700];
    double time_spent_table_sfid, time_spent_table_q;

    #if defined(_OPENMP)
    t_rank = omp_get_thread_num();
    #endif

    struct spectrograph_windowfn_params     win_params             = {0, 0, PIXEL_WIDTH, 0};
    struct sq_integrand_params              integration_parameters = {&fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};
    
    allocateTmpArrays();

    // Integrate fiducial signal matrix
    FourierIntegrator s_integrator(GSL_INTEG_COSINE, signal_matrix_integrand, &integration_parameters);

    // Skip this section if fiducial signal matrix is turned off.
    if (TURN_OFF_SFID) goto DERIVATIVE;

    #pragma omp for nowait
    for (int r = 0; r < NUMBER_OF_R_VALUES; r++)
    {
        time_spent_table_sfid = mytime::getTime();

        // Convert integer FWHM to 1 sigma km/s
        win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
        
        LOG::LOGGER.STD("T%d/%d - Creating look up table for signal matrix. R = %d : %.2f km/s.\n", \
                t_rank, numthreads, R_VALUES[r], win_params.spectrograph_res);

        STableFileNameConvention(buf, DIR.c_str(), S_BASE.c_str(), R_VALUES[r]);
        
        if (!force_rewrite && ioh::file_exists(buf))
        {
            LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf);
            continue;
        }

        for (int xy = 0; xy < N_V_POINTS*N_Z_POINTS_OF_S; ++xy)
        {
            // xy = nv + Nv * nz
            int nz = xy / Nv, nv = xy % Nv;

            win_params.delta_v_ij = LINEAR_V_ARRAY[nv];         // 0 + LENGTH_V * nv / (Nv - 1.);
            win_params.z_ij       = LINEAR_Z_ARRAY[nz];         // z_first + z_length * nz / (Nz - 1.);  
            
            s_integrator.setTableParameters(win_params.delta_v_ij, 10.);
            // 1E-15 gave roundoff error for smoothing with 20.8 km/s
            signal_array[xy]    = s_integrator.evaluate0ToInfty(1E-14);
        }

        SQLookupTableFile signal_table(buf, 'w');

        signal_table.setHeader( N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, z_length, \
                                R_VALUES[r], PIXEL_WIDTH, \
                                0, bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS]);

        signal_table.writeData(signal_array);

        time_spent_table_sfid = mytime::getTime() - time_spent_table_sfid;

        LOG::LOGGER.STD("T:%d/%d - Time spent on fiducial signal matrix table R %d is %.2f mins.\n", \
                t_rank, numthreads, R_VALUES[r], time_spent_table_sfid);
    }

DERIVATIVE:
    // Integrate derivative matrices
    // int subNz = Nz / NUMBER_OF_Z_BINS;
    FourierIntegrator q_integrator(GSL_INTEG_COSINE, q_matrix_integrand, &integration_parameters);

    #pragma omp for
    for (int r = 0; r < NUMBER_OF_R_VALUES; r++)
    {
        time_spent_table_q = mytime::getTime();

        win_params.spectrograph_res = SPEED_OF_LIGHT / R_VALUES[r] / ONE_SIGMA_2_FWHM;
        LOG::LOGGER.STD("T:%d/%d - Creating look up tables for derivative signal matrices. R = %d : %.2f km/s.\n", \
                t_rank, numthreads, R_VALUES[r], win_params.spectrograph_res);

        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        {
            double kvalue_1 = bins::KBAND_EDGES[kn];
            double kvalue_2 = bins::KBAND_EDGES[kn + 1];

            LOG::LOGGER.STD("Q matrix for k = [%.1e - %.1e] s/km.\n", kvalue_1, kvalue_2);

            QTableFileNameConvention(buf, DIR.c_str(), Q_BASE.c_str(), R_VALUES[r], kvalue_1, kvalue_2);
            
            if (!force_rewrite && ioh::file_exists(buf))
            {
                LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf);
                continue;
            }

            for (int nv = 0; nv < N_V_POINTS; ++nv)
            {
                win_params.delta_v_ij = LINEAR_V_ARRAY[nv];

                derivative_array[nv] = q_integrator.evaluate(win_params.delta_v_ij, kvalue_1, kvalue_2, 0);
            }

            SQLookupTableFile derivative_signal_table(buf, 'w');

            derivative_signal_table.setHeader(  N_V_POINTS, 0, LENGTH_V, bins::Z_BIN_WIDTH, \
                                                R_VALUES[r], PIXEL_WIDTH, \
                                                kvalue_1, kvalue_2);
            
            derivative_signal_table.writeData(derivative_array);
        }
        
        time_spent_table_q = mytime::getTime() - time_spent_table_q;
        LOG::LOGGER.STD("T:%d/%d - Time spent on derivative matrix table R %d is %.2f mins.\n", \
                t_rank, numthreads, R_VALUES[r], time_spent_table_q);
    }
    // Q matrices are written.
    // ---------------------
    deallocateTmpArrays();

}
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
    // Allocate and set v array
    LINEAR_V_ARRAY = new double[N_V_POINTS];
    for (int nv = 0; nv < N_V_POINTS; ++nv)
        LINEAR_V_ARRAY[nv] = getLinearlySpacedValue(0, LENGTH_V, N_V_POINTS, nv);

    // Allocate and set redshift array
    LINEAR_Z_ARRAY = new double[N_Z_POINTS_OF_S];
    double zfirst  = bins::ZBIN_CENTERS[0] - bins::Z_BIN_WIDTH;

    for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz)
        LINEAR_Z_ARRAY[nz] = getLinearlySpacedValue(zfirst, LENGTH_Z_OF_S, N_Z_POINTS_OF_S, nz);

    // Allocate signal and derivative arrays
    signal_array     = new double[N_V_POINTS * N_Z_POINTS_OF_S];
    derivative_array = new double[N_V_POINTS];
}

void SQLookupTable::deallocateTmpArrays()
{
    delete [] LINEAR_V_ARRAY;
    delete [] LINEAR_Z_ARRAY;

    delete [] signal_array;
    delete [] derivative_array;
}

void SQLookupTable::readSQforR(int r_index)
{
    char buf[700];

    // Read S table.
    STableFileNameConvention(buf, DIR.c_str(), S_BASE.c_str(), R_VALUES[r_index]);
    LOG::LOGGER.IO("Reading sq_lookup_table_file %s.\n", buf);
    SQLookupTableFile s_table_file(buf, 'r');
    
    int dummy_R, dummy_Nz;
    double temp_px_width, temp_ki, temp_kf;

    s_table_file.readHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S,
                            dummy_R, temp_px_width,
                            temp_ki, temp_kf);

    // Allocate memory before reading further
    if (LINEAR_V_ARRAY == NULL)     allocateTmpArrays();

    // Start reading data and interpolating
    s_table_file.readData(signal_array);

    // Interpolate
    interp2d_signal_matrices[r_index] = new Interpolation2D(INTERP_2D_TYPE,
                                                            LINEAR_V_ARRAY, LINEAR_Z_ARRAY,
                                                            signal_array,
                                                            N_V_POINTS, N_Z_POINTS_OF_S);

    // Read Q tables. 
    double kvalue_1, kvalue_2, dummy_lzq;

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        kvalue_1 = bins::KBAND_EDGES[kn];
        kvalue_2 = bins::KBAND_EDGES[kn + 1];

        QTableFileNameConvention(buf, DIR.c_str(), Q_BASE.c_str(), R_VALUES[r_index], kvalue_1, kvalue_2);

        SQLookupTableFile q_table_file(buf, 'r');

        q_table_file.readHeader(N_V_POINTS, dummy_Nz, LENGTH_V, dummy_lzq,
                                dummy_R, temp_px_width,
                                temp_ki, temp_kf);

        q_table_file.readData(derivative_array);

        // Interpolate
        int i = getIndex4DerivativeInterpolation(kn, r_index);
        interp_derivative_matrices[i] = new Interpolation(INTERP_1D_TYPE, LINEAR_V_ARRAY, derivative_array, N_V_POINTS);
    }
}

double SQLookupTable::getSignalMatrixValue(double v_ij, double z_ij, int r_index) const 
{
    return interp2d_signal_matrices[r_index]->evaluate(v_ij, z_ij);
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














