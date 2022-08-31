#ifndef SQ_TABLE_H
#define SQ_TABLE_H

#include <vector>
#include <string>

#include "mathtools/discrete_interpolation.hpp"
#include "io/config_file.hpp"

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
    double LENGTH_V, LENGTH_Z_OF_S;
    std::vector<std::pair<int, double>> R_DV_VALUES;
    
    std::string DIR, S_BASE, Q_BASE;

    // Temporary arrays. They are not stored after construction!
    std::unique_ptr<double[]>  LINEAR_V_ARRAY, LINEAR_Z_ARRAY,
        signal_array, derivative_array;

    std::vector<shared_interp_2d> interp2d_signal_matrices;
    std::vector<shared_interp_1d> interp_derivative_matrices;

    double itp_v1, itp_dv, itp_z1, itp_dz; 

    int getIndex4DerivativeInterpolation(int kn, int r_index) const;

    void allocateSignalAndDerivArrays();
    void allocateVAndZArrays();
    void deallocateSignalAndDerivArrays();
    void deallocateVAndZArrays();

    shared_interp_1d _allocReadQFile(int kn, int r_index);
    shared_interp_2d _allocReadSFile(int r_index);

public:
    SQLookupTable(const ConfigFile &config);

    void readSQforR(int r_index, shared_interp_2d &s,
        std::vector<shared_interp_1d>  &q, 
        bool alloc=false);

    void readTables();
    // runs with omp parallel
    void computeTables(bool force_rewrite);

    int findSpecResIndex(int spec_res, double dv) const;

    shared_interp_1d getDerivativeMatrixInterp(int kn, int r_index) const;
    shared_interp_2d getSignalMatrixInterp(int r_index) const;

    double getOneSetMemUsage();
    double getMaxMemUsage();
};

namespace process
{
    extern std::unique_ptr<SQLookupTable> sq_private_table;
}

#endif
