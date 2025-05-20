#ifndef MYFITSIO_H
#define MYFITSIO_H

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

    struct fitsfile_deleter {
        void operator()(fitsfile* fptr) {
            int status = 0;
            if (fptr != nullptr)
                fits_close_file(fptr, &status);
            checkFitsStatus(status);
        }
    };

    using unique_fitsfile_ptr = std::unique_ptr<fitsfile, fitsfile_deleter>;

    inline unique_fitsfile_ptr create_unique_fitsfile_ptr(
            const std::string &fname
    ) {
        int status = 0;
        fitsfile *fits_file = nullptr;
        fits_create_file(&fits_file, fname.c_str(), &status);
        checkFitsStatus(status);
        unique_fitsfile_ptr fptr(fits_file);
        return fptr;
    }

    inline unique_fitsfile_ptr open_unique_fitsfile_ptr(
            const std::string &fname, int mode
    ) {
        int status = 0;
        fitsfile *fits_file = nullptr;
        fits_open_file(&fits_file, fname.c_str(), mode, &status);
        checkFitsStatus(status);
        unique_fitsfile_ptr fptr(fits_file);
        return fptr;
    }
}

#endif
