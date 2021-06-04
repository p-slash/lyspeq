#ifndef PICCA_FILE_H
#define PICCA_FILE_H

#include <fitsio.h>
#include <string>

class PiccaFile
{
    fitsfile *fits_file;
    std::string file_name;
    // int hdunum;
    int no_spectra, status;
    int curr_spec_index;
    int curr_N, curr_M;
    
    void _move(int index);

public:
    PiccaFile(std::string fname_qso);
    ~PiccaFile();
    
    int getNumberSpectra() const {return no_spectra;};

    void readParameters(int &N, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms, int newhdu=-1);

    void readData(double *lambda, double *delta, double *noise, int newhdu=-1);
    void readResolutionMatrix(double *Rmat, int &mdim, int newhdu=-1);
};

#endif
