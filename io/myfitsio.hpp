#ifndef MYFITSIO_H
#define MYFITSIO_H

#include <memory>
#include <string>
#include <vector>

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

    static std::vector<std::string> readHeaderKeys(
            unique_fitsfile_ptr &fts, int &status
    ) {
        int nkeys;
        char keyname[FLEN_KEYWORD], value[FLEN_VALUE];
        std::vector<std::string> header_keys;

        fits_get_hdrspace(fts.get(), &nkeys, NULL, &status);
        header_keys.reserve(nkeys);

        for (int i = 1; i <= nkeys; ++i) {
            fits_read_keyn(fts.get(), i, keyname, value, NULL, &status);
            header_keys.push_back(std::string(keyname));
        }
        return header_keys;
    }

    static std::vector<std::string> readColumnNAmes(
            unique_fitsfile_ptr &fts, int &status
    ) {
        int ncols;
        char keyname[FLEN_KEYWORD], colname[FLEN_VALUE];
        std::vector<std::string> colnames;

        fits_get_num_cols(fts.get(), &ncols, &status);
        colnames.reserve(ncols);

        for (int i = 1; i <= ncols; i++) {
            fits_make_keyn("TTYPE", i, keyname, &status); /* make keyword */
            fits_read_key(fts.get(), TSTRING, keyname, colname, NULL, &status);

            colnames.push_back(std::string(colname));
        }
        return colnames;
    }
}

#endif
