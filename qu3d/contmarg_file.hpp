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
        void operator()(std::FILE* fptr) { std::fclose(fptr); }
    };

    using unique_file_ptr = std::unique_ptr<std::FILE, file_deleter>;


    class ContMargFile {
    public:
        ContMargFile(const std::string &base)
        : tmpfolder(base) {
            file_writers.reserve(myomp::getMaxNumThreads());
            for (int i = 0; i < myomp::getMaxNumThreads(); ++i)
                file_writers.push_back(_openFile(i, "wb"));
        };

        void write(
                double *data, int N, int &fidx, double *evecs=nullptr
        ) {
            fidx = myomp::getThreadNum();
            std::FILE *fptr = file_writers[fidx].get();

            if (std::fwrite(&N, sizeof(int), 1, fptr) != 1)
                throw std::runtime_error("ERROR in ContMargFile::write");

            size_t Min = N * N,
                   Mout = std::fwrite(data, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::write");

            if (evecs != nullptr)
                if (size_t(N) != std::fwrite(evecs, sizeof(double), N, fptr))
                    throw std::runtime_error("ERROR in ContMargFile::write");
        }

        void openAllReaders() {
            file_readers.reserve(file_writers.size());
            for (size_t i = 0; i < file_writers.size(); ++i)
                file_readers.push_back(_openFile(i, "rb"));
            file_writers.clear();
        }

        void rewind() {
            for (auto &fptr : file_readers)
                std::rewind(fptr.get());
        }

        void read(int N, double *out) {
            std::FILE *fptr = file_readers[myomp::getThreadNum()].get();
            int n;
            if (std::fread(&n, sizeof(int), 1, fptr) != 0 || n != N)
                throw std::runtime_error("ERROR in ContMargFile::read::fseek");

            size_t Min = size_t(N) * N, 
                   Mout = std::fread(out, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
        }

        std::string getFname(int fidx) {
            return tmpfolder + "/qu3d-rdmat-" + std::to_string(fidx) + ".dat";
        }
    private:
        std::string tmpfolder;
        std::vector<unique_file_ptr> file_writers, file_readers;

        unique_file_ptr _openFile(int fidx, const char *mode) {
            std::string fname = getFname(fidx);

            unique_file_ptr fptr(std::fopen(fname.c_str(), mode));

            if (!fptr)
                throw std::runtime_error(
                    std::string("Cannot open file: ") + fname);   

            return fptr;
        }
    };

    extern std::unique_ptr<ContMargFile> continuumMargFileHandler;
}

#endif
