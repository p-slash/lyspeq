#ifndef QSO_FILE_H
#define QSO_FILE_H

#include <cstdio>
#include <string>
#include <fitsio.h>
#include "core/matrix_helper.hpp"

namespace qio
{
enum ifileformat {Binary, Picca};

class BQFile
{
    FILE *qso_file;
    // char read_write[3];

    struct qso_io_header
    {
        int data_size;

        double redshift;
        double declination;
        double right_ascension;

        int spectrograph_resolution_fwhm;
        double signal_to_noise;
        double pixel_width;

        double lower_observed_lambda;
        double upper_observed_lambda;

        double lower_rest_lambda;
        double upper_rest_lambda;
    } header;

public:
    BQFile(std::string fname_qso);
    ~BQFile();

    void readParameters(int &data_number, double &z, int &fwhm_resolution, double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *fluxfluctuations, double *noise);
};

class PiccaFile
{
    fitsfile *fits_file;
    // int hdunum;
    int no_spectra, status;
    int curr_spec_index;
    int curr_N, curr_ndiags;
    void _move(int index);
    int _getColNo(char *tmplt);
    void _checkStatus();

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

class QSOFile
{
    std::string file_name;
    ifileformat PB;
    PiccaFile *pfile;
    BQFile *bqfile;

public:
    QSOFile(std::string fname_qso, ifileformat p_or_b);
    ~QSOFile();

    void readParameters(int &data_number, double &z, int &fwhm_resolution, double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *fluxfluctuations, double *noise);
    void readAllocResolutionMatrix(mxhelp::Resolution *& Rmat);
};

}

#endif
