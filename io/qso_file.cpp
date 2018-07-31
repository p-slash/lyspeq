#include "qso_file.hpp"
#include "io_helper_functions.hpp"

QSOFile::QSOFile(const char *fname)
{
    sprintf(file_name, "%s", fname);

    qso_file = open_file(file_name, "rb");
}

QSOFile::~QSOFile()
{
    fclose(qso_file);
}

void QSOFile::readParameters(int &data_number, double &z, double &resolution, double &sig2noi, double &dv_kms)
{
    #define SPEED_OF_LIGHT 299792.458

    rewind(qso_file);

    fread(&header, sizeof(qso_io_header), 1, qso_file);

    data_number = header.data_size;
    z           = header.redshift;
    resolution  = SPEED_OF_LIGHT  / (1.0 * header.spectrograph_resolution);
    sig2noi     = header.signal_to_noise;
    dv_kms      = header.pixel_width;
}

void QSOFile::readData(double *lambda, double *flux, double *noise)
{
    fseek(qso_file, sizeof(qso_io_header), SEEK_SET);

    fread(lambda, header.data_size * sizeof(double), 1, qso_file);
    fread(flux,   header.data_size * sizeof(double), 1, qso_file);
    fread(noise,  header.data_size * sizeof(double), 1, qso_file);
}