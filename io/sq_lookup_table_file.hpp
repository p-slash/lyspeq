#ifndef SQ_LOOKUP_TABLE_FILE_H
#define SQ_LOOKUP_TABLE_FILE_H

#include <cstdio>
#include <string>

namespace sqhelper
{
    std::string QTableFileNameConvention(const std::string &OUTPUT_DIR, const std::string &OUTPUT_FILEBASE_Q, 
        int r, double dv, double k1, double k2);

    std::string STableFileNameConvention(const std::string &OUTPUT_DIR, const std::string &OUTPUT_FILEBASE_S, 
        int r, double dv);

    typedef struct
    {
        int nvpoints;
        int nzpoints;

        double v_length;
        double z1;
        double z_length;

        int spectrograph_resolution;
        double pixel_width;

        double k1;
        double k2;
    } SQ_IO_Header;

    // This file object reads and writes evaluated S and Q matrices in a standard file format.
    // spectrograph_resolution, pixel_width and k values can be used to check consistency.
    class SQLookupTableFile
    {
        FILE *sq_file;
        std::string file_name;
        char read_write[3];

        SQ_IO_Header header;
        
        bool isHeaderSet;

        void _readHeader();
        
    public:
        SQLookupTableFile(std::string fname, char rw);
        ~SQLookupTableFile();

        void setHeader(const SQ_IO_Header hdr);

        SQ_IO_Header readHeader();

        void readData(double *data);
        void writeData(const double *data);
        
    };
}

#endif
