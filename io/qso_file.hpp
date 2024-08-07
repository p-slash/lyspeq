#ifndef QSO_FILE_H
#define QSO_FILE_H

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include <fitsio.h>

#include "core/global_numbers.hpp"
#include "io/io_helper_functions.hpp"
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
    BQFile(const std::string &fname_qso) {
        qso_file = ioh::open_file(fname_qso.c_str(), "rb");
    };

    BQFile(BQFile &&rhs) = default;
    BQFile(const BQFile &rhs) = delete;
    ~BQFile() { fclose(qso_file); };

    void readParameters(
        int &data_number, double &z, double &dec, double &ra,
        int &fwhm_resolution,  double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *fluxfluctuations, double *ivar);
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
    void _handleStatus();

    void _setHeaderKeys();
    void _setColumnNames();
    bool _isHeaderKey(const std::string &key);
    bool _isColumnName(const std::string &key);
    void _readOptionalInt(const std::string &key, int &output);

public:
    static bool compareFnames(const std::string &s1, const std::string &s2);
    static void clearCache();

    // Assume fname to be ..fits.gz[1]
    PiccaFile(const std::string &fname_qso);
    PiccaFile(PiccaFile &&rhs) = default;
    PiccaFile(const PiccaFile &rhs) = delete;

    int getNumberSpectra() const {return no_spectra;};

    void readParameters(
        long &thid, int &N, double &z, double &dec, double &ra,
        int &fwhm_resolution, double &sig2noi, double &dv_kms, double &dlambda,
        int &expid, int &night, int &fiber, int& petal);

    void readData(double *lambda, double *delta, double *ivar);
    std::unique_ptr<mxhelp::Resolution> readAllocResolutionMatrix();
};

class QSOFile
{
    ifileformat PB;
    std::unique_ptr<PiccaFile> pfile;
    std::unique_ptr<BQFile> bqfile;

    double *wave_head, *delta_head, *ivar_head;
    int arr_size, _fullsize, shift, num_masked_pixels;
    // count num_masked_pixels after cutting
    void _cutMaskedBoundary();
    void _countMaskedPixels();

public:
    std::string fname;
    double z_qso, snr, dv_kms, dlambda, ra, dec;
    long id;
    int R_fwhm, expid, night, fiber, petal;
    std::unique_ptr<mxhelp::Resolution> Rmat;

    QSOFile(const std::string &fname_qso, ifileformat p_or_b);
    // The "copy" constructor below also cuts masked boundaries.
    QSOFile(const qio::QSOFile &qmaster, int i1, int i2);
    QSOFile(QSOFile &&rhs) = delete;
    QSOFile(const QSOFile &rhs) = delete;
    ~QSOFile() {
        process::updateMemory(getMinMemUsage());

        closeFile();
        delete [] wave_head;
        delete [] delta_head;
        delete [] ivar_head;
    };

    void closeFile() { pfile.reset(); bqfile.reset(); };

    int size() const { return arr_size; };
    int realSize() const { return arr_size-num_masked_pixels; };
    double* wave() const  { return wave_head+shift; };
    double* delta() const { return delta_head+shift; };
    double* ivar() const { return ivar_head+shift; };

    double getMinMemUsage() {
        double mem = 0;
        if (wave_head != nullptr)
            mem += process::getMemoryMB(_fullsize * 3);
        if (Rmat)
            mem += Rmat->getMinMemUsage();
        return mem;
    }

    void recalcDvDLam();
    void readParameters();
    void readData();

    // This is just a pointer shift for w,d,e. Rmat is copied
    // Cuts masked boundaries as well
    // Counts the num_masked_pixels
    void cutBoundary(double z_lower_edge, double z_upper_edge);
    int maskOutliers(double factor=5.0);

    void readMinMaxMedRedshift(double &zmin, double &zmax, double &zmed);
    void readAllocResolutionMatrix();
};

}

#endif
