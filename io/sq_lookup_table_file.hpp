#ifndef SQ_LOOKUP_TABLE_FILE_H
#define SQ_LOOKUP_TABLE_FILE_H

#include <cstdio>

void QTableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_Q, \
                                int r, double k1, double k2);

void STableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_S, \
                                int r);

// Returns y = c + deltaY / (N-1) * n
double getLinearlySpacedValue(double c, double delta_y, int N, int n);

// This file object reads and writes evaluated S and Q matrices in a standard file format.
// spectrograph_resolution, pixel_width and k values can be used to check consistency.
class SQLookupTableFile
{
    FILE *sq_file;
    char file_name[256];
    char read_write[3];

    struct sq_io_header
    {
        int vpoints;
        int zpoints;

        double v_length;
        double z_length;

        int spectrograph_resolution;
        double pixel_width;

        double initial_k;
        double final_k;
    } header;
    
    bool isHeaderSet;

    void readHeader();
    
public:
    SQLookupTableFile(const char *fname, char rw);
    ~SQLookupTableFile();

    void setHeader( int nv, int nz, double len_v, double len_z, \
                    int R, double dv, \
                    double ki, double kf);

    void readHeader(int &nv, int &nz, double &len_v, double &len_z, \
                    int &R, double &dv, \
                    double &ki, double &kf);

    void readData(double *data);
    void writeData(double *data);
    
};

#endif
