#ifndef SQ_LOOKUP_TABLE_H
#define SQ_LOOKUP_TABLE_H

#include <cstdio>

void QTableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_Q, \
                                int r, double k1, double k2, double z);

void STableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_S, \
                                int r);

// Returns y = c + deltaY / (N-1) * n
double getLinearlySpacedValue(double c, double delta_y, int N, int n);

class SQLookupTable
{
    FILE *sq_file;
    char file_name[256];
    char read_write[3];

    struct sq_io_header
    {
        int vpoints;
        int zpoints;

        int spectrograph_resolution;
        double pixel_width;

        double redshift;
        double initial_k;
        double final_k;
    } header;
    
    bool isHeaderSet;

    void readHeader();
    
public:
    SQLookupTable(const char *fname, char rw);
    ~SQLookupTable();

    void setHeader(int nv, int nz, int R, double dv, double z, double ki, double kf);

    void readHeader(int &nv, int &nz, int &R, double &dv, double &z, double &ki, double &kf);

    void readData(double *data);
    void writeData(double *data);
    
};

#endif
