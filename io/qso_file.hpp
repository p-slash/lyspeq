#ifndef QSO_FILE_H
#define QSO_FILE_H

#include <cstdio>

class QSOFile
{
    FILE *qso_file;
    char file_name[256];
    // char read_write[3];

    struct qso_io_header
    {
        int data_size;

        double redshift;
        double declination;
        double right_ascension;

        int spectrograph_resolution;
        double signal_to_noise;
        double pixel_width;

        double lower_observed_lambda;
        double upper_observed_lambda;

        double lower_rest_lambda;
        double upper_rest_lambda;
    } header;

public:
    QSOFile(const char *fname);
    ~QSOFile();

    void readParameters(int &data_number, double &z, int &resolution, double &sig2noi, double &dv_kms);

    void readData(double *lambda, double *flux, double *noise);
    
};

#endif
