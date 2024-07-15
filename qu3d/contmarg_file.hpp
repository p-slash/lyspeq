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
        ContMargFile(const std::string &base) : fbase(base) {};
        ~ContMargFile() {
            std::for_each(
                file_handlers.begin(), file_handlers.end(),
                [](const auto &x) { fclose(x.second); }
            );
        }

        size_t write(double *data, int N, int fidx) {
            auto it = file_handlers.find(fidx);
            FILE *fptr;

            if (it == file_handlers.end())
                fptr = _openFile(fidx);
            else
                fptr = it->second;

            size_t Min = N * N,
                   Mout = fwrite(data, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            return ftell(fptr);
        }

        void read(int fidx, size_t pos, int N, double *out) {
            FILE *fptr = file_handlers[fidx];
            if (fseek(fptr, pos, SEEK_SET) != 0)
                throw std::runtime_error("ERROR in ContMargFile::read::fseek");

            size_t Min = N * N, 
                   Mout = fread(out, sizeof(double), Min, fptr);
            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

    private:
        std::string fbase;
        std::unordered_map<int, FILE*> file_handlers;

        FILE* _openFile(int fidx) {
            std::string fname = fbase + "/qu3d-rmat-" + std::to_string(fidx) + ".dat";
            FILE *fptr = open_file(fname.c_str(), "wb");
            file_handlers[fidx] = fptr;
            return fptr;
        }
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
