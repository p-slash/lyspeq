#ifndef MYFITSIO_H
#define MYFITSIO_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <fitsio.h>


namespace ioh {
    template<class T> inline int fitsioDataType();
    template<> inline int fitsioDataType<double>() { return TDOUBLE; }
    template<> inline int fitsioDataType<int>() { return TINT; }
    template<> inline int fitsioDataType<bool>() { return TLOGICAL; }
    template<> inline int fitsioDataType<long>() { return TLONG; }

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
            fitsfile *fts, int &status
    ) {
        int nkeys;
        char keyname[FLEN_KEYWORD], value[FLEN_VALUE];
        std::vector<std::string> header_keys;

        fits_get_hdrspace(fts, &nkeys, NULL, &status);
        header_keys.reserve(nkeys);

        for (int i = 1; i <= nkeys; ++i) {
            fits_read_keyn(fts, i, keyname, value, NULL, &status);
            header_keys.push_back(std::string(keyname));
        }
        return header_keys;
    }

    static std::vector<std::string> readColumnNAmes(
            fitsfile *fts, int &status
    ) {
        int ncols;
        char keyname[FLEN_KEYWORD], colname[FLEN_VALUE];
        std::vector<std::string> colnames;

        fits_get_num_cols(fts, &ncols, &status);
        colnames.reserve(ncols);

        for (int i = 1; i <= ncols; i++) {
            fits_make_keyn("TTYPE", i, keyname, &status); /* make keyword */
            fits_read_key(fts, TSTRING, keyname, colname, NULL, &status);

            colnames.push_back(std::string(colname));
        }
        return colnames;
    }

    template<class T>
    inline void readFitsKey(fitsfile *fts, const char *key, T &value, int &status) {
        fits_read_key(fts, fitsioDataType<T>(), key, &value, NULL, &status);
    }
    template<class T>
    inline void writeFitsKey(fitsfile *fts, const char *key, T value, int &status) {
        fits_write_key(fts, fitsioDataType<T>(), key, &value, NULL, &status);
    }

    class FitsFileSlim {
        fitsfile *fits_file;
        int status;
        std::vector<std::string> header_keys, colnames;
    public:
        FitsFileSlim() : fits_file(nullptr), status(0) {};
        FitsFileSlim(fitsfile *fptr) : fits_file(fptr), status(0) {};
        void set(fitsfile *fptr) { fits_file = fptr; status = 0; }
        fitsfile* get() const { return fits_file; }
        void open(const std::string &fname, int mode=READONLY) {
            fits_open_file(&fits_file, fname.c_str(), mode, &status);
        }
        void close() { fits_close_file(fits_file, &status); }
        void check() const { checkFitsStatus(status); }
        int getNumHdus() {
            int nhdus;
            fits_get_num_hdus(fits_file, &nhdus, &status);
            return nhdus;
        }
        void moveToHdu(int hdunum, int &hdutype) {
            fits_movabs_hdu(fits_file, hdunum, &hdutype, &status);
        }
        void moveToHdu(char *extname, int hdutype=BINARY_TBL) {
            fits_movnam_hdu(fits_file, hdutype, extname, 0, &status);
        }
        void readHeaderKeys() { header_keys = ioh::readHeaderKeys(fits_file, status); }
        void readColumnNames() { colnames = ioh::readColumnNAmes(fits_file, status); }
        int getColumnNumber(char *colname) {
            int colnum;
            fits_get_colnum(fits_file, CASEINSEN, colname, &colnum, &status);
            return colnum;
        }
        const std::vector<std::string>& getHeaderKeys() const { return header_keys; }
        const std::vector<std::string>& getColumnNames() const { return colnames; }

        bool isHeaderKey(const std::string &key) const {
            return std::any_of(header_keys.begin(), header_keys.end(),
                [key](const std::string &elem) { return elem == key; });
        }
        bool isColumnName(const std::string &key) const {
            return std::any_of(colnames.begin(), colnames.end(),
                [key](const std::string &elem) { return elem == key; });
        }
        template<class T>
        void readKey(const char *key, T &value) {
            readFitsKey(fits_file, key, value, status);
        }
        template<class T>
        void readKeyFallback(const char *key, T &value, int fallback=-1) {
            if (isHeaderKey(key))  readKey(key, value);
            else value = fallback;
        }
        template<class T>
        void readColumn(int colnum, long nelements, T *data) {
            fits_read_col(fits_file, fitsioDataType<T>(), colnum, 1, 1, nelements,
                0, data, 0, &status);
        }
        void readTDIM(int colnum, long nelements, int &naxis, long *naxes) {
            fits_read_tdim(fits_file, colnum, nelements, &naxis, naxes, &status);
        }

        /* Writing */
        void createTable(
                int nrows, int ncols, char **colnames, char **coltypes,
                char **colunits=nullptr, const char *extname=nullptr
        ) {
            fits_create_tbl(
                fits_file, BINARY_TBL, nrows, ncols,
                colnames, coltypes, colunits, extname, &status);
        }
        template<class T>
        void writeKey(const char *key, T value) {
            writeFitsKey(fits_file, key, value, status);
        }
        template<class T>
        void writeColumn(int colnum, long nelements, T *data) {
            fits_write_col(fits_file, fitsioDataType<T>(), colnum, 1, 1,
                           nelements, data, &status);
        }
        void createImage(int naxis, long *naxes, const char *extname=nullptr) {
            fits_create_img(fits_file, DOUBLE_IMG, naxis, naxes, &status);
            if (extname != nullptr)
                fits_update_key_str(fits_file, "EXTNAME", extname, nullptr, &status);
        }
        void writeImage(long nelements, const double *data) {
            fits_write_img(fits_file, TDOUBLE, 1, nelements, (void *) data, &status);
        }
        void flush() { fits_flush_file(fits_file, &status); }
    };
}

#endif
