#ifndef QSO_FILE_H
#define QSO_FILE_H

#include <cstdio>
#include <string>
#include <map>
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
    BQFile(const std::string &fname_qso);
    ~BQFile();

    void readParameters(int &data_number, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *fluxfluctuations, double *noise);
};

class PiccaFile
{
    static std::map<std::string, fitsfile*> cache;

    fitsfile *fits_file;
    // int hdunum;
    int no_spectra, status;
    int curr_spec_index;

    int curr_N, curr_elem_per_row;
    int _getColNo(char *tmplt);
    void _checkStatus();

public:
    static bool compareFnames(const std::string &s1, const std::string &s2);
    static void clearCache();

    // Assume fname to be ..fits.gz[1]
    PiccaFile(const std::string &fname_qso);
    ~PiccaFile();

    int getNumberSpectra() const {return no_spectra;};

    void readParameters(long &thid, int &N, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms, double &dlambda, int &oversampling);

    void readData(double *lambda, double *delta, double *noise);
    mxhelp::Resolution* readAllocResolutionMatrix(int oversampling, double dlambda);
};

class QSOFile
{
    ifileformat PB;
    PiccaFile *pfile;
    BQFile *bqfile;

    double *wave_head, *delta_head, *noise_head;
public:
    std::string fname;
    double z_qso, snr, dv_kms, dlambda;
    long id;
    int size, R_fwhm, oversampling;
    double *wave, *delta, *noise;
    mxhelp::Resolution *Rmat;

    QSOFile(const std::string &fname_qso, ifileformat p_or_b);
    QSOFile(const qio::QSOFile &qmaster, int i1, int i2);
    void closeFile();
    ~QSOFile();

    void recalcDvDLam();
    void readParameters();
    void readData();

    // This is just a pointer shift for w,d,e. Rmat is copied
    // returns new size
    int cutBoundary(double z_lower_edge, double z_upper_edge);

    void readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed);
    void readAllocResolutionMatrix();
};

}

#endif
