#ifndef SQ_TABLE_H
#define SQ_TABLE_H

#include "../gsltools/interpolation.hpp"
#include "../gsltools/interpolation_2d.hpp"

class SQLookupTable
{
    int NUMBER_OF_R_VALUES, N_K_BINS, N_Z_BINS, \
        N_V_POINTS, N_Z_POINTS_OF_S, N_Z_POINTS_OF_Q;

    int *R_VALUES;

    double LENGTH_V, LENGTH_Z_OF_S, LENGTH_Z_OF_Q;

    double  *LINEAR_V_ARRAY, *LINEAR_Z_ARRAY, \
            *signal_array, *derivative_array;

    const double *KBAND_EDGES, *ZBIN_CENTERS;

    Interpolation2D **interp2d_signal_matrices;
    Interpolation   **interp_derivative_matrices;

    int findSpecResIndex(int spec_res);

    int getIndex4SignalMatrix(int nv, int nz, int r);

    int getIndex4DerivativeInterpolation(int kn, int r);
    int getIndex4DerivativeMatrix(int nv, int kn, int r);

    void allocate();
    void readSQforR(int r_index, const char *dir, const char *s_base, const char *q_base);
public:
    SQLookupTable(  const char *dir, const char *s_base, const char *q_base, \
                    const char *fname_rlist, \
                    const double *k_edges, int nkbins, \
                    const double *z_centers, int nzbins);
    ~SQLookupTable();
    
    double getSignalMatrixValue(double v_ij, double z_ij, int spec_res);
    double getDerivativeMatrixValue(double v_ij, double z_ij, int zm, int kn, int spec_res);
};

#endif
