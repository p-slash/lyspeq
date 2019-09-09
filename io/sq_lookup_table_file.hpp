#ifndef SQ_LOOKUP_TABLE_FILE_H
#define SQ_LOOKUP_TABLE_FILE_H

#include <cstdio>
#include <string>

namespace sqhelper
{
    std::string QTableFileNameConvention(const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_Q, int r, double k1, double k2);

    std::string STableFileNameConvention(const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_S, int r);
}

// This file object reads and writes evaluated S and Q matrices in a standard file format.
// spectrograph_resolution, pixel_width and k values can be used to check consistency.
class SQLookupTableFile
{
    FILE *sq_file;
    std::string file_name;
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
    SQLookupTableFile(std::string fname, char rw);
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
