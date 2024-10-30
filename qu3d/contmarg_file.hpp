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
        ContMargFile(const std::string &base, const std::string &prefix)
        : tmpfolder(base), unique_prefix(prefix) {};

        void write(
                double *data, int N, long targetid, int &fidx,
                double *evecs=nullptr
        ) {
            fidx = myomp::getThreadNum();
            std::FILE *fptr = file_writers[fidx].get();

            if (std::fwrite(&N, sizeof(int), 1, fptr) != 1)
                throw std::runtime_error("ERROR in ContMargFile::write");
            if (std::fwrite(&targetid, sizeof(long), 1, fptr) != 1)
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
            file_readers.reserve(myomp::getMaxNumThreads());
            for (int i = 0; i < myomp::getMaxNumThreads(); ++i)
                file_readers.push_back(_openFile(i, "rb"));
        }

        void openAllWriters() {
            file_writers.reserve(myomp::getMaxNumThreads());
            for (int i = 0; i < myomp::getMaxNumThreads(); ++i)
                file_writers.push_back(_openFile(i, "wb"));
        }

        void closeAllWriters() { file_writers.clear(); }

        void rewind() {
            for (auto &fptr : file_readers)
                std::rewind(fptr.get());
        }

        void read(int N, long targetid, double *out) {
            std::FILE *fptr = file_readers[myomp::getThreadNum()].get();
            int n;  long id;

            #ifdef DEBUG_IO
            if (std::fread(&n, sizeof(int), 1, fptr) != 1)
                throw std::runtime_error("ERROR in ContMargFile::read::N");

            if (std::fread(&id, sizeof(long), 1, fptr) != 1)
                throw std::runtime_error("ERROR in ContMargFile::read::targetid");

            if ((n != N) || (id != targetid)) {
                std::string err =
                    "ERROR in ContMargFile::read::n("
                    + std::to_string(n) + ") vs N(" + std::to_string(N) + ") "
                    + "and id(" + std::to_string(id) + ") vs targetid("
                    + std::to_string(targetid) + ")";
                throw std::runtime_error(err);
            }

            size_t Min = size_t(N) * N, 
                   Mout = std::fread(out, sizeof(double), Min, fptr);

            if (Min != Mout)
                throw std::runtime_error("ERROR in ContMargFile::read::fread");
            #else
            std::fread(&n, sizeof(int), 1, fptr);
            std::fread(&id, sizeof(long), 1, fptr);
            std::fread(out, sizeof(double), size_t(N) * N, fptr);
            #endif
        }

        std::string getFname(int fidx) {
            return tmpfolder + "/qu3d-rdmat" + unique_prefix + "-"
                   + std::to_string(fidx) + ".dat";
        }
    private:
        std::string tmpfolder, unique_prefix;
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
