#ifndef CONTMARG_FILE_H
#define CONTMARG_FILE_H

#include <algorithm>
#include <cstdio>
#include <memory>
#include <unordered_map>

#include "core/omp_manager.hpp"
#include "io/io_helper_functions.hpp"

namespace ioh {
    class ContMargFile {
    public:
        ContMargFile(const std::string &base) : tmpfolder(base) {};

        std::string write(
                double *data, int N, long int targetid, double *evecs=nullptr
        ) {
            FILE *fptr;
            std::string fname = _openFile(targetid, fptr);

            size_t Min = N * N,
                   Mout = fwrite(data, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            if (evecs != nullptr)
                if (N != fwrite(evecs, sizeof(double), N, fptr))
                    throw std::runtime_error("ERROR in ContMargFile::write");

            fclose(fptr);

            return std::move(fname);
        }

        void read(const char *fname, int N, double *out) {
            FILE *fptr = open_file(fname, "rb");

            size_t Min = N * N, 
                   Mout = fread(out, sizeof(double), Min, fptr);
            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

        std::string getFname(long int targetid) {
            return tmpfolder + "/qu3d-rdmat-" + std::to_string(targetid) + ".dat";
        }
    private:
        std::string tmpfolder;

        std::string _openFile(long int targetid, FILE *fptr) {
            std::string fname = getFname(targetid);
            fptr = open_file(fname.c_str(), "wb");
            return std::move(fname);
        }
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
