#ifndef QSO_FILE_H
#define QSO_FILE_H

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <fitsio.h>

#include "core/global_numbers.hpp"
#include "mathtools/matrix_helper.hpp"

namespace qio
{
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
    BQFile(BQFile &&rhs) = default;
    BQFile(const BQFile &rhs) = delete;
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
    int curr_N, curr_elem_per_row;

    std::vector<std::string> header_keys, colnames;

    int _getColNo(char *tmplt);
    void _checkStatus();

    void _setHeaderKeys();
    void _setColumnNames();
    bool _isHeaderKey(const std::string &key);
    bool _isColumnName(const std::string &key);

public:
    static bool compareFnames(const std::string &s1, const std::string &s2);
    static void clearCache();

    // Assume fname to be ..fits.gz[1]
    PiccaFile(const std::string &fname_qso);
    PiccaFile(PiccaFile &&rhs) = default;
    PiccaFile(const PiccaFile &rhs) = delete;

    int getNumberSpectra() const {return no_spectra;};

    void readParameters(long &thid, int &N, double &z, int &fwhm_resolution, 
        double &sig2noi, double &dv_kms, double &dlambda, int &oversampling);

    void readData(double *lambda, double *delta, double *noise);
    std::unique_ptr<mxhelp::Resolution> readAllocResolutionMatrix(int oversampling, double dlambda);
};

class QSOFile
{
    ifileformat PB;
    std::unique_ptr<PiccaFile> pfile;
    std::unique_ptr<BQFile> bqfile;

    double *wave_head, *delta_head, *noise_head;
    int arr_size, shift;
public:
    std::string fname;
    double z_qso, snr, dv_kms, dlambda;
    long id;
    int R_fwhm, oversampling;
    std::unique_ptr<mxhelp::Resolution> Rmat;

    QSOFile(const std::string &fname_qso, ifileformat p_or_b);
    QSOFile(const qio::QSOFile &qmaster, int i1, int i2);
    QSOFile(QSOFile &&rhs) = delete;
    QSOFile(const QSOFile &rhs) = delete;

    void closeFile();
    ~QSOFile();

    int size() const { return arr_size; };
    double* wave() const  { return wave_head+shift; };
    double* delta() const { return delta_head+shift; };
    double* noise() const { return noise_head+shift; };

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
