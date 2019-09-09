#ifndef SQ_TABLE_H
#define SQ_TABLE_H

#include <vector>
#include <string>

#include "gsltools/interpolation.hpp"
#include "gsltools/interpolation_2d.hpp"

// This table read, stores and interpolates pre-evaluated S and Q matrices for different 
// spectral resolution (R) values. Here R is assumend to be an integer where c / R is in km/s.
// NOT thread safe because of Interpolations. Create local copies per thread

// dir            : The directory where files live
// s_base, q_base : File basenames. Signal matrices (S) as well as derivative matrices (Q) follows 
//               the conventions in ../io/sq_lookup_table_file.
// fname_rlist    : full path of spectral resolution list file. This files starts with 
//               NUMBER_OF_R_VALUES

// v and z values are linearly spaced using the function again in ../io/sq_lookup_table_file. They 
// are stored temporarily in LINEAR_V_ARRAY and LINEAR_Z_ARRAY respectively.

// Q matrices are not scaled with redshift binning function. N_K_BINS many files for each R only 
// have values for v_ij, i.e. 1D arrays in v. These values are multiplied by binning function when 
// evaluated getDerivativeMatrixValue called.

// LENGTH_V, N_V_POINTS and N_Z_POINTS_OF_S are read from SQLookupTableFiles.
// LENGTH_Z_OF_S is equal to N_Z_BINS times redshift bin width, whereas 

// All values are stored in 1D arrays such as signal_array and derivative_array. The indexing 
// conventions have their own functions.
class SQLookupTable
{
    int NUMBER_OF_R_VALUES, N_V_POINTS, N_Z_POINTS_OF_S;

    std::string DIR, S_BASE, Q_BASE;

    std::vector<int> R_VALUES;

    double LENGTH_V, LENGTH_Z_OF_S;

    Interpolation2D **interp2d_signal_matrices;
    Interpolation   **interp_derivative_matrices;

    int getIndex4DerivativeInterpolation(int kn, int r_index) const;

    void allocateTmpArrays();
    void deallocateTmpArrays();

    void readSQforR(int r_index);

    void computeSignalTables();
    // void computeDerivativeTables();

public:
    SQLookupTable(const char *dir, const char *s_base, const char *q_base, const char *fname_rlist);
    SQLookupTable(const SQLookupTable &sq);

    ~SQLookupTable();
    
    void readTables();
    void computeTables(double PIXEL_WIDTH, int Nv, int Nz, double Lv, bool force_rewrite);

    int findSpecResIndex(int spec_res) const;

    double getSignalMatrixValue(double v_ij, double z_ij, int r_index) const;
    double getDerivativeMatrixValue(double v_ij, int kn, int r_index) const;

    Interpolation*   getDerivativeMatrixInterp(int kn, int r_index) const;
    Interpolation2D* getSignalMatrixInterp(int r_index) const;
};

#endif
