#ifndef CONTMARG_FILE_H
#define CONTMARG_FILE_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <unordered_map>

#include "core/omp_manager.hpp"
#include "io/io_helper_functions.hpp"


namespace ioh {
    struct file_deleter {
        void operator()(FILE* fptr) { fclose(fptr); }
    };

    using unique_file_ptr = std::unique_ptr<FILE, file_deleter>;


    class ContMargFile {
    public:
        ContMargFile(const std::string &base)
        : tmpfolder(base) {
            for (int i = 0; i < myomp::getMaxNumThreads(); ++i)
                file_writers[i] = _openFile(i, "wb");
        };

        size_t write(
                double *data, size_t N, int &fidx, double *evecs=nullptr
        ) {
            fidx = myomp::getThreadNum();
            FILE *fptr = file_writers[fidx].get();

            size_t fpos = ftell(fptr);
            size_t Min = N * N,
                   Mout = fwrite(data, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            if (evecs != nullptr)
                if (N != fwrite(evecs, sizeof(double), N, fptr))
                    throw std::runtime_error("ERROR in ContMargFile::write");

            return fpos;
        }

        void openAllReaders() {
            std::vector<int> all_fidcs;
            all_fidcs.resize(file_writers.size());

            for (const auto &[fidx, uptr] : file_writers)
                all_fidcs.push_back(fidx);

            file_writers.clear();
            file_readers_vec.resize(myomp::getMaxNumThreads());

            for (auto &rdr : file_readers_vec)
                for (const int &fidx : all_fidcs)
                    rdr.emplace(fidx, _openFile(fidx, "rb"));
        }

        void read(int fidx, size_t fpos, size_t N, double *out) {
            FILE *fptr = file_readers_vec[myomp::getThreadNum()][fidx].get();

            if (fseek(fptr, fpos, SEEK_SET) != 0)
                throw std::runtime_error("ERROR in ContMargFile::read::fseek");

            size_t Min = N * N, 
                   Mout = fread(out, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

        std::string getFname(int fidx) {
            return tmpfolder + "/qu3d-rdmat-" + std::to_string(fidx) + ".dat";
        }
    private:
        std::string tmpfolder;

        std::unordered_map<int, unique_file_ptr> file_writers;
        std::vector<
            std::unordered_map<int, unique_file_ptr>
        > file_readers_vec;

        unique_file_ptr _openFile(int fidx, const char *mode) {
            std::string fname = getFname(fidx);

            unique_file_ptr fptr(fopen(fname.c_str(), mode));

            if (!fptr)
                throw std::runtime_error(
                    std::string("Cannot open file: ") + fname);   

            return fptr;
        }
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
