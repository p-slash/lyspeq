#ifndef QU3D_FILE_H
#define QU3D_FILE_H

#include <memory>
#include <string>

#include <fitsio.h>


namespace ioh {
    inline void checkFitsStatus(int status) {
        if (status == 0)
            return;

        char fits_msg[80];
        fits_get_errstatus(status, fits_msg);
        std::string error_msg =
            std::string("FITS ERROR ") + std::string(fits_msg);

        throw std::runtime_error(error_msg);
    }

    class Qu3dFile {
    public:
        Qu3dFile(const std::string &base, int thispe) {
            status = 0; fits_file = nullptr;

            if (thispe != 0)
                return;

            std::string out_fname = "!" + base + "-qu3d.fits";

            fits_create_file(&fits_file, out_fname.c_str(), &status);
            checkFitsStatus(status);
        }
        ~Qu3dFile() {
            if (fits_file != nullptr)
                fits_close_file(fits_file, &status);
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
            fits_flush_buffer(fits_file, 0, &status);
            checkFitsStatus(status);
        }
    private:
        int status;
        fitsfile *fits_file;
    };
}

#endif
