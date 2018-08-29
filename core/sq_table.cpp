#include "sq_table.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/sq_lookup_table_file.hpp"

#include <cstdio>

double triangular_z_bin(double z, double zm, double deltaz)
{
    return 1;
    
    if (zm - deltaz < z && z < zm)
    {
        return (z - zm + deltaz) / deltaz;
    }
    else if (zm < z && z < zm + deltaz)
    {
        return (zm + deltaz - z) / deltaz;
    }

    return 0;
}

SQLookupTable::SQLookupTable(   const char *dir, const char *s_base, const char *q_base, \
                                const char *fname_rlist, \
                                const double *k_edges, int nkbins, \
                                const double *z_centers, int nzbins)
{
    KBAND_EDGES  = k_edges;
    ZBIN_CENTERS = z_centers;
    N_K_BINS     = nkbins;
    N_Z_BINS     = nzbins;

    LINEAR_V_ARRAY   = NULL;
    LINEAR_Z_ARRAY   = NULL;
    signal_array     = NULL;
    derivative_array = NULL;

    // Read R values
    FILE *toRead = open_file(fname_rlist, "r");
    fscanf(toRead, "%d\n", &NUMBER_OF_R_VALUES);

    printf("Number of R values: %d\n", NUMBER_OF_R_VALUES);

    R_VALUES = new int[NUMBER_OF_R_VALUES];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
    {
        fscanf(toRead, "%d\n", &R_VALUES[r]);
    }

    fclose(toRead);
    // Reading R values done
    // ---------------------

    printf("Setting tables..\n");
    fflush(stdout);

    interp2d_signal_matrices     = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices   = new Interpolation*[NUMBER_OF_R_VALUES * N_K_BINS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
    {
        readSQforR(r, dir, s_base, q_base);
    }
}


SQLookupTable::~SQLookupTable()
{
    delete [] R_VALUES;

    for (int r = 0; r < NUMBER_OF_R_VALUES; r++)
    {
        delete interp2d_signal_matrices[r];
    }

    for (int kr = 0; kr < NUMBER_OF_R_VALUES * N_K_BINS; ++kr)
    {
        delete interp_derivative_matrices[kr];
    }

    delete [] LINEAR_V_ARRAY;
    delete [] LINEAR_Z_ARRAY;

    delete [] signal_array;
    delete [] derivative_array;

    delete [] interp2d_signal_matrices;
    delete [] interp_derivative_matrices;
}

void SQLookupTable::allocate()
{
    // Allocate and set v array
    LINEAR_V_ARRAY = new double[N_V_POINTS];
    for (int nv = 0; nv < N_V_POINTS; ++nv)
    {
        LINEAR_V_ARRAY[nv] = getLinearlySpacedValue(0, LENGTH_V, N_V_POINTS, nv);
        printf("%.2e ", LINEAR_V_ARRAY[nv]);
    }

    // Allocate and set redshift array
    LINEAR_Z_ARRAY = new double[N_Z_POINTS_OF_S];
    double zbinwidth = ZBIN_CENTERS[1] - ZBIN_CENTERS[0];
    double zfirst    = ZBIN_CENTERS[0] - zbinwidth / 2.;

    for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz)
    {
        LINEAR_Z_ARRAY[nz] = getLinearlySpacedValue(zfirst, LENGTH_Z_OF_S, N_Z_POINTS_OF_S, nz);
    }

    // Allocate signal and derivative arrays
    // index = nv + Nv * (nz + Nz * r)
    signal_array     = new double[N_V_POINTS * N_Z_POINTS_OF_S * NUMBER_OF_R_VALUES];

    // index = nv + Nv * (r + Nr * kn)
    derivative_array = new double[N_V_POINTS * NUMBER_OF_R_VALUES * N_K_BINS];
}

void SQLookupTable::readSQforR(int r_index, const char *dir, const char *s_base, const char *q_base)
{
    char buf[500];

    // Read S table.
    STableFileNameConvention(buf, dir, s_base, R_VALUES[r_index]);
    SQLookupTableFile s_table_file(buf, 'r');
    
    int dummy_R, dummy_Nz;
    double temp_px_width, temp_ki, temp_kf;

    s_table_file.readHeader(N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, LENGTH_Z_OF_S, \
                            dummy_R, temp_px_width, \
                            temp_ki, temp_kf);

    N_Z_POINTS_OF_Q = N_Z_POINTS_OF_S / N_Z_BINS;

    if (LINEAR_V_ARRAY == NULL)
    {
        allocate();
    }

    double *sub_signal_array = &signal_array[getIndex4SignalMatrix(0, 0, r_index)];
    s_table_file.readData(sub_signal_array);

    // Interpolate
    interp2d_signal_matrices[r_index] = new Interpolation2D(LINEAR_V_ARRAY, LINEAR_Z_ARRAY, \
                                                            sub_signal_array, \
                                                            N_V_POINTS, N_Z_POINTS_OF_S);

    // Read Q tables. 
    double kvalue_1, kvalue_2;

    for (int kn = 0; kn < N_K_BINS; ++kn)
    {
        kvalue_1 = KBAND_EDGES[kn];
        kvalue_2 = KBAND_EDGES[kn + 1];

        QTableFileNameConvention(buf, dir, q_base, R_VALUES[r_index], kvalue_1, kvalue_2);

        SQLookupTableFile q_table_file(buf, 'r');

        q_table_file.readHeader(N_V_POINTS, dummy_Nz, LENGTH_V, LENGTH_Z_OF_Q, \
                                dummy_R, temp_px_width, \
                                temp_ki, temp_kf);

        double *sub_q_array = &derivative_array[getIndex4DerivativeMatrix(0, kn, r_index)];
        q_table_file.readData(sub_q_array);

        // Interpolate
        interp_derivative_matrices[getIndex4DerivativeInterpolation(kn, r_index)] = new Interpolation(LINEAR_V_ARRAY, sub_q_array, N_V_POINTS);
    }
}

double SQLookupTable::getSignalMatrixValue(double v_ij, double z_ij, int r_index) const 
{
    return interp2d_signal_matrices[r_index]->evaluate(v_ij, z_ij);
}

double SQLookupTable::getDerivativeMatrixValue(double v_ij, double z_ij, int zm, int kn, int r_index) const
{
    double v_result = interp_derivative_matrices[getIndex4DerivativeInterpolation(kn ,r_index)]->evaluate(v_ij);
    double z_result = triangular_z_bin(z_ij, ZBIN_CENTERS[zm], LENGTH_Z_OF_Q);

    return z_result * v_result;
}

int SQLookupTable::getIndex4SignalMatrix(int nv, int nz, int r_index) const
{
    return nv + N_V_POINTS * (nz + N_Z_POINTS_OF_S * r_index);
}
int SQLookupTable::getIndex4DerivativeInterpolation(int kn, int r_index) const
{
    return r_index + NUMBER_OF_R_VALUES * kn;
}
int SQLookupTable::getIndex4DerivativeMatrix(int nv, int kn, int r_index) const
{
    return nv + N_V_POINTS * getIndex4DerivativeInterpolation(kn, r_index);
}

int SQLookupTable::findSpecResIndex(int spec_res) const
{
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
    {
        if (R_VALUES[r] == spec_res)
        {
            return r;
        }
    }

    return -1;
}














