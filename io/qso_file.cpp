#include "io/qso_file.hpp"
#include "io/io_helper_functions.hpp"
#include <stdexcept>

QSOFile::QSOFile(const char *fname)
{
    sprintf(file_name, "%s", fname);

    qso_file = ioh::open_file(file_name, "rb");
}

QSOFile::~QSOFile()
{
    fclose(qso_file);
}

void QSOFile::readParameters(int &data_number, double &z, int &fwhm_resolution, double &sig2noi, double &dv_kms)
{
    rewind(qso_file);

    if (fread(&header, sizeof(qso_io_header), 1, qso_file) != 1)
        std::runtime_error("fread error in header QSOFile!");

    data_number      = header.data_size;
    z                = header.redshift;
    fwhm_resolution  = header.spectrograph_resolution_fwhm;
    sig2noi          = header.signal_to_noise;
    dv_kms           = header.pixel_width;
}

void QSOFile::readData(double *lambda, double *flux, double *noise)
{
    int rl, rf, rn;
    fseek(qso_file, sizeof(qso_io_header), SEEK_SET);

    rl = fread(lambda, sizeof(double), header.data_size, qso_file);
    rf = fread(flux,   sizeof(double), header.data_size, qso_file);
    rn = fread(noise,  sizeof(double), header.data_size, qso_file);

    if (rl != header.data_size || rf != header.data_size || rn != header.data_size)
        std::runtime_error("fread error in data QSOFile!");
}
