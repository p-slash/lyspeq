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
            std::string fname = getFname(targetid);
            std::unique_ptr<FILE, decltype(&fclose)> fptr(
                fopen(fname.c_str(), "wb"), &fclose);

            if (!fptr)
                throw std::runtime_error(std::string("Cannot open file: ") + fname);

            size_t Min = N * N,
                   Mout = fwrite(data, sizeof(double), Min, fptr.get());

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            if (evecs != nullptr)
                if (N != fwrite(evecs, sizeof(double), N, fptr.get()))
                    throw std::runtime_error("ERROR in ContMargFile::write");

            return fname;
        }

        void read(const char *fname, int N, double *out) {
            std::unique_ptr<FILE, decltype(&fclose)> fptr(
                fopen(fname, "rb"), &fclose);

            if (!fptr)
                throw std::runtime_error(std::string("Cannot open file: ") + fname);

            size_t Min = N * N, 
                   Mout = fread(out, sizeof(double), Min, fptr.get());

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

        std::string getFname(long int targetid) {
            return tmpfolder + "/qu3d-rdmat-" + std::to_string(targetid) + ".dat";
        }
    private:
        std::string tmpfolder;
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
