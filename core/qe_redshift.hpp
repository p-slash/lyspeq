#ifndef QE_REDSHIFT_H
#define QE_REDSHIFT_H

class OneDQPSERedshiftBins
{
public:
    OneDQPSERedshiftBins(   int full_data_size, \
                            const double *xspace, \
                            const double *delta, \
                            const double *noise, \
                            int no_kbands, \
                            double linear_kband_length, \
                            double first_k, \
                            int no_zbins, \
                            double linear_zbin_width, \
                            double initial_z);
    ~OneDQPSERedshiftBins();
    
};



#endif
