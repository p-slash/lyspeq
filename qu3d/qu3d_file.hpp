#ifndef QU3D_FILE_H
#define QU3D_FILE_H

#include "io/myfitsio.hpp"


namespace ioh {
    class Qu3dFile {
    public:
        Qu3dFile(const std::string &base, int thispe) {
            status = 0; fits_file = nullptr;

            if (thispe != 0)
                return;

            std::string out_fname = "!" + base + "-qu3d.fits";

            fitsfile_ptr = create_unique_fitsfile_ptr(out_fname);
            fits_file = fitsfile_ptr.get();
        }

        void write(
                const double *data, int ndata, const std::string ext,
                int num_mc=-1
        ) {
            if (fits_file == nullptr)
                return;

            int bitpix = DOUBLE_IMG;
            long naxis = 1, naxes[1] = {ndata};
            fits_create_img(fits_file, bitpix, naxis, naxes, &status);
            checkFitsStatus(status);

            fits_update_key_str(
                fits_file, "EXTNAME", ext.c_str(), nullptr, &status);

            if (num_mc > 0)
                fits_write_key(
                    fits_file, TUINT, "NUM_MCS", &num_mc, nullptr, &status
                );

            fits_write_img(
                fits_file, TDOUBLE, 1, ndata, (void *) data, &status);
            checkFitsStatus(status);
        }

        void flush() {
            if (fits_file == nullptr)
                return;

            fits_flush_buffer(fits_file, 0, &status);
            checkFitsStatus(status);
        }
    private:
        int status;
        unique_fitsfile_ptr fitsfile_ptr;
        fitsfile *fits_file;
    };
}

#endif
