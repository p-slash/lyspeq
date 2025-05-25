#ifndef SQ_LOOKUP_TABLE_FILE_H
#define SQ_LOOKUP_TABLE_FILE_H

#include <cstdio>
#include <string>

#include "io/myfitsio.hpp"


namespace sqhelper
{
    std::string SQTableFileNameConvention(
        const std::string &OUTPUT_DIR,
        const std::string &OUTPUT_FILEBASE_S, 
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

        double k1, dklin, dklog;
        int nklin, nklog;
    } SQ_IO_Header;


    class SQLookupTableFile {
        ioh::unique_fitsfile_ptr fitsfile_ptr;
        fitsfile *fits_file;

        std::string file_name;
        SQ_IO_Header meta;
        
    public:
        SQLookupTableFile(const std::string &fname, bool towrite);

        SQ_IO_Header readMeta();
        void writeMeta(SQ_IO_Header &hdr);

        // Call only after setting Metadata;
        // data size should be nktot * nvpoints
        void readDeriv(double *data);
        void writeDeriv(const double *data);
        // data size should be nzpoints * nvpoints
        void readSignal(double *data);
        void writeSignal(const double *data);
    };
}

#endif
