#ifndef MYFITSIO_H
#define MYFITSIO_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <fitsio.h>


namespace ioh {
    template<typename T> inline constexpr int fitsioDataType = -1;
    template<> inline constexpr int fitsioDataType<double> = TDOUBLE;
    template<> inline constexpr int fitsioDataType<int> = TINT;
    template<> inline constexpr int fitsioDataType<bool> = TLOGICAL;
    template<> inline constexpr int fitsioDataType<long> = TLONG;

    template<typename T> inline constexpr const char* fitsioTForm = "";
    template<> inline constexpr const char* fitsioTForm<double> = "1D";
    template<> inline constexpr const char* fitsioTForm<int> = "1J";
    template<> inline constexpr const char* fitsioTForm<bool> = "1L";
    template<> inline constexpr const char* fitsioTForm<long> = "1K";

    struct FitsColumn { const char *name, *tform, *unit; };

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
    static fitsfile_deleter safeFitsClose;

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
        fits_read_key(fts, fitsioDataType<T>, key, &value, NULL, &status);
    }
    template<class T>
    inline void writeFitsKey(fitsfile *fts, const char *key, T value, int &status) {
        fits_write_key(fts, fitsioDataType<T>, key, &value, NULL, &status);
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
        void moveToHdu(int hdunum, int *hdutype) {
            fits_movabs_hdu(fits_file, hdunum, hdutype, &status);
        }
        void moveToHdu(const char *extname, int hdutype=BINARY_TBL) {
            fits_movnam_hdu(fits_file, hdutype, const_cast<char*>(extname), 0, &status);
        }
        void readHeaderKeys() { header_keys = ioh::readHeaderKeys(fits_file, status); }
        void readColumnNames() { colnames = ioh::readColumnNAmes(fits_file, status); }
        int getColumnNumber(const char *colname) {
            int colnum;
            fits_get_colnum(fits_file, CASEINSEN, const_cast<char*>(colname),
                            &colnum, &status);
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
        void readLongKey(std::vector<long> &naxes, int nfound, int nstart=1, int nmax=3, const char *key="NAXIS") {
            if (naxes.empty())  naxes.resize(nmax);
            nmax = naxes.size();
            fits_read_keys_lng(fits_file, key, nstart, nmax, naxes.data(), &nfound, &status);
            naxes.resize(nfound);
        }
        template<class T>
        void readKeyFallback(const char *key, T &value, int fallback=-1) {
            if (isHeaderKey(key))  readKey(key, value);
            else value = fallback;
        }
        template<class T>
        void readColumn(int colnum, long nelements, T *data) {
            int nonull;
            fits_read_col(fits_file, fitsioDataType<T>, colnum, 1, 1, nelements,
                0, data, &nonull, &status);
        }
        template<class T>
        void readImage(T *data, long nelements, long firstelem=1, double nullval=0) {
            fits_read_img(fits_file, fitsioDataType<T>, firstelem, nelements, &nullval, data, nullptr, &status);
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
        void createTable(
                int nrows, const std::vector<ioh::FitsColumn> &columns,
                const char *extname=nullptr
        ) {
            int ncols = columns.size();
            char **colnames = new char*[ncols];
            char **coltypes = new char*[ncols];
            char **colunits = new char*[ncols];
            for (int i = 0; i < ncols; i++) {
                colnames[i] = const_cast<char*>(columns[i].name);
                coltypes[i] = const_cast<char*>(columns[i].tform);
                colunits[i] = const_cast<char*>(columns[i].unit);
            }
            fits_create_tbl(
                fits_file, BINARY_TBL, nrows, ncols,
                colnames, coltypes, colunits, extname, &status);
            delete[] colnames;
            delete[] coltypes;
            delete[] colunits;
        }
        template<class T>
        void writeKey(const char *key, T value) {
            writeFitsKey(fits_file, key, value, status);
        }
        template<class T>
        void writeColumn(int colnum, long nelements, const T *data) {
            fits_write_col(fits_file, fitsioDataType<T>, colnum, 1, 1,
                           nelements, const_cast<T*>(data), &status);
        }
        template<class T>
        void writeColumn(const char *colname, long nelements, const T *data) {
            int colnum = getColumnNumber(colname);
            writeColumn(colnum, nelements, data);
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
