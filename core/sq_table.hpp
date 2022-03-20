#ifndef SQ_TABLE_H
#define SQ_TABLE_H

#include <vector>
#include <string>

#include "gsltools/discrete_interpolation.hpp"

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

    DiscreteInterpolation2D **interp2d_signal_matrices;
    DiscreteInterpolation1D **interp_derivative_matrices;
    double itp_v1, itp_dv, itp_z1, itp_dz; 

    int getIndex4DerivativeInterpolation(int kn, int r_index) const;

    void allocateSignalAndDerivArrays();
    void allocateVAndZArrays();
    void deallocateSignalAndDerivArrays();
    void deallocateVAndZArrays();

    DiscreteInterpolation1D* _allocReadQFile(int kn, int r_index);
    DiscreteInterpolation2D* _allocReadSFile(int r_index);

public:
    SQLookupTable(const char *dir, const char *s_base, const char *q_base, 
        const char *fname_rlist, int Nv=0, int Nz=0, double Lv=0);
    // SQLookupTable(const SQLookupTable &sq);

    ~SQLookupTable();
    
    void readSQforR(int r_index, DiscreteInterpolation2D*& s, DiscreteInterpolation1D**& q, 
        bool alloc=false);

    void readTables();
    // runs with omp parallel
    void computeTables(bool force_rewrite);

    int findSpecResIndex(int spec_res, double dv) const;

    DiscreteInterpolation1D* getDerivativeMatrixInterp(int kn, int r_index) const;
    DiscreteInterpolation2D* getSignalMatrixInterp(int r_index) const;

    double getOneSetMemUsage();
    double getMaxMemUsage();
};

#endif
