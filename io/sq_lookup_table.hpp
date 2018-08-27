#ifndef SQ_LOOKUP_TABLE_H
#define SQ_LOOKUP_TABLE_H

#include <cstdio>

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

public:
    SQLookupTable(const char *fname, char rw);
    ~SQLookupTable();

    void setHeader(int nv, int nz, int R, double dv, double z, double ki, double kf);

    void readData(double *data);
    void writeData(double *data);
    
};

#endif
