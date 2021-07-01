#ifndef PICCA_FILE_H
#define PICCA_FILE_H

#include <fitsio.h>
#include <string>
#include "io/qso_file.hpp"
#include "core/matrix_helper.hpp"

class PiccaFile: public QSOFile
{
    fitsfile *fits_file;
    std::string file_name;
    // int hdunum;
    int no_spectra, status;
    int curr_spec_index;
    int curr_N, curr_ndiags;
    int ic_llam, ic_delta, ic_ivar, ic_reso;
    void _move(int index);

public:
    // Assume fname to be ..fits.gz[1]
    PiccaFile(std::string fname_qso);
    ~PiccaFile();
    
    int getNumberSpectra() const {return no_spectra;};

    void readParameters(int &N, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *delta, double *noise);
    void readAllocResolutionMatrix(mxhelp::Resolution *& Rmat);
};

#endif
