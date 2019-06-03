#include "sq_table.hpp"
#include "global_numbers.hpp"

#include "../io/io_helper_functions.hpp"
#include "../io/sq_lookup_table_file.hpp"

#include <cstdio>
#include <algorithm> // std::copy

double tophat_z_bin(double z, int zm)
{
    double z_center = ZBIN_CENTERS[zm];

    if (z_center - Z_BIN_WIDTH/2 < z && z < z_center + Z_BIN_WIDTH/2)
        return 1.;

    return 0;
}

double triangular_z_bin(double z, int zm)
{
    double z_center = ZBIN_CENTERS[zm];
    
    if (z_center - Z_BIN_WIDTH < z && z <= z_center)
    {
        if (zm == 0)    return 1.;
        else            return (z - z_center + Z_BIN_WIDTH) / Z_BIN_WIDTH;
    }
    else if (z_center < z && z < z_center + Z_BIN_WIDTH)
    {
        if (zm == NUMBER_OF_Z_BINS - 1)     return 1.;
        else                                return (z_center + Z_BIN_WIDTH - z) / Z_BIN_WIDTH;
    }

    return 0;
}

double redshift_binning_function(double z, int zm)
{
    #ifdef TOPHAT_Z_BINNING_FN
    return tophat_z_bin(z, zm);
    #endif

    #ifdef TRIANGLE_Z_BINNING_FN
    return triangular_z_bin(z, zm);
    #endif
}

SQLookupTable::SQLookupTable(const char *dir, const char *s_base, const char *q_base, const char *fname_rlist)
{
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
        fscanf(toRead, "%d\n", &R_VALUES[r]);

    fclose(toRead);
    // Reading R values done
    // ---------------------

    printf("Setting tables..\n");
    fflush(stdout);

    interp2d_signal_matrices     = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices   = new Interpolation*[NUMBER_OF_R_VALUES * NUMBER_OF_K_BANDS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        readSQforR(r, dir, s_base, q_base);

    deallocateTmpArrays();
}

SQLookupTable::SQLookupTable(const SQLookupTable &sq)
{
    printf("Copying SQ table.\n");
    fflush(stdout);
    
    NUMBER_OF_R_VALUES = sq.NUMBER_OF_R_VALUES;

    N_V_POINTS      = sq.N_V_POINTS;
    N_Z_POINTS_OF_S = sq.N_Z_POINTS_OF_S;

    LENGTH_V      = sq.LENGTH_V;
    LENGTH_Z_OF_S = sq.LENGTH_Z_OF_S;

    R_VALUES = copyArrayAlloc<int>(sq.R_VALUES, NUMBER_OF_R_VALUES);

    interp2d_signal_matrices   = new Interpolation2D*[NUMBER_OF_R_VALUES];
    interp_derivative_matrices = new Interpolation*[NUMBER_OF_R_VALUES * NUMBER_OF_K_BANDS];

    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        interp2d_signal_matrices[r] = new Interpolation2D(*sq.interp2d_signal_matrices[r]);
        
    for (int q = 0; q < NUMBER_OF_R_VALUES * NUMBER_OF_K_BANDS; ++q)
        interp_derivative_matrices[q] = new Interpolation(*sq.interp_derivative_matrices[q]);
}

SQLookupTable::~SQLookupTable()
{
    delete [] R_VALUES;

    for (int r = 0; r < NUMBER_OF_R_VALUES; r++)
        delete interp2d_signal_matrices[r];

    for (int kr = 0; kr < NUMBER_OF_R_VALUES * NUMBER_OF_K_BANDS; ++kr)
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
    double zfirst  = ZBIN_CENTERS[0] - Z_BIN_WIDTH;

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

    // Allocate memory before reading further
    if (LINEAR_V_ARRAY == NULL)     allocateTmpArrays();

    // Start reading data and interpolating
    s_table_file.readData(signal_array);

    // Interpolate
    interp2d_signal_matrices[r_index] = new Interpolation2D(INTERP_2D_TYPE, \
                                                            LINEAR_V_ARRAY, LINEAR_Z_ARRAY, \
                                                            signal_array, \
                                                            N_V_POINTS, N_Z_POINTS_OF_S);

    // Read Q tables. 
    double kvalue_1, kvalue_2, dummy_lzq;

    for (int kn = 0; kn < NUMBER_OF_K_BANDS; ++kn)
    {
        kvalue_1 = KBAND_EDGES[kn];
        kvalue_2 = KBAND_EDGES[kn + 1];

        QTableFileNameConvention(buf, dir, q_base, R_VALUES[r_index], kvalue_1, kvalue_2);

        SQLookupTableFile q_table_file(buf, 'r');

        q_table_file.readHeader(N_V_POINTS, dummy_Nz, LENGTH_V, dummy_lzq, \
                                dummy_R, temp_px_width, \
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

double SQLookupTable::getDerivativeMatrixValue(double v_ij, double z_ij, int zm, int kn, int r_index) const
{
    double z_result = redshift_binning_function(z_ij, zm);
    
    if (z_result == 0)      return 0;

    double v_result = interp_derivative_matrices[getIndex4DerivativeInterpolation(kn ,r_index)]->evaluate(v_ij);

    return z_result * v_result;
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
    return kn + NUMBER_OF_K_BANDS * r_index;
}

int SQLookupTable::findSpecResIndex(int spec_res) const
{
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        if (R_VALUES[r] == spec_res)    return r;

    return -1;
}














