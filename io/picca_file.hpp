#ifndef PICCA_FILE_H
#define PICCA_FILE_H

#include <fitsio.h>

class PiccaFile
{
    fitsfile *fits_file;
    char file_name[256];
    int no_spectra, status;
    int curr_spec_index;
    int curr_N, curr_M;
    
    void _move(int index);

public:
    PiccaFile(const char *fname);
    ~PiccaFile();
    
    int getNumberSpectra() const {return no_spectra;};

    void readParameters(int index, int &N, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms);

    void readData(int index, int N, double *lambda, double *delta, double *noise);
    void readResolutionMatrix(int index, int N, double *Rmat, int &mdim);
};

#endif
