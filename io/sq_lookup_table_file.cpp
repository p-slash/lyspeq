#include "io/sq_lookup_table_file.hpp"
#include "io/io_helper_functions.hpp"

#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace sqhelper
{
std::string QTableFileNameConvention(
        const std::string &OUTPUT_DIR, const std::string &OUTPUT_FILEBASE_Q,
        int r, double dv, double k1, double k2
) {
    std::ostringstream qt_fname;

    qt_fname << OUTPUT_DIR << "/" << OUTPUT_FILEBASE_Q  << "_R" << r 
        << std::fixed << std::setprecision(1) << "_dv" << dv
        << std::scientific << std::setprecision(5) << "_k" << k1 << "_" << k2
        << ".dat";

    return qt_fname.str();
}

std::string STableFileNameConvention(
        const std::string &OUTPUT_DIR, const std::string &OUTPUT_FILEBASE_S,
        int r, double dv
) {
    std::ostringstream st_fname;
    st_fname << OUTPUT_DIR << "/" << OUTPUT_FILEBASE_S << "_R" << r 
        << std::fixed << std::setprecision(1) << "_dv" << dv << ".dat";

    return st_fname.str();
}

SQLookupTableFile::SQLookupTableFile(std::string fname, char rw)
    : file_name(fname)
{
    read_write[0] = rw;
    read_write[1] = 'b';
    read_write[2] = '\0';

    sq_file = ioh::open_file(file_name.c_str(), read_write);

    isHeaderSet = false;
}

SQLookupTableFile::~SQLookupTableFile()
{
    fclose(sq_file);
}

void SQLookupTableFile::setHeader(const SQ_IO_Header hdr)
{
    if (read_write[0] == 'r')
        throw std::runtime_error("SQLookupTableFile::setHeader() in reading mode!\n");

    header = hdr;
    isHeaderSet = true;
}

void SQLookupTableFile::_readHeader()
{
    if (!isHeaderSet)
    {
        rewind(sq_file);
        if (fread(&header, sizeof(SQ_IO_Header), 1, sq_file) != 1)
            throw std::runtime_error("fread error in header SQLookupTableFile!");

        isHeaderSet = true;
    }
}

SQ_IO_Header SQLookupTableFile::readHeader()
{
    if (read_write[0] == 'w')
        throw std::runtime_error("SQLookupTableFile::readHeader() in writing mode!");

    _readHeader();
    return header;
}

void SQLookupTableFile::writeData(const double *data)
{
    if (!isHeaderSet)
        throw std::runtime_error("SQLookupTableFile::writeData() before header is set!");

    int size = header.nvpoints * header.nzpoints;

    if (size == 0)
        size = header.nvpoints;

    int fw;

    fw = fwrite(&header, sizeof(SQ_IO_Header), 1, sq_file);
    if (fw != 1)
        throw std::runtime_error( "fwrite error in header SQLookupTableFile!");

    fw = fwrite(data, sizeof(double), size, sq_file);
    if (fw != size)
        throw std::runtime_error( "fwrite error in data SQLookupTableFile!");
}

void SQLookupTableFile::readData(double *data)
{
    readHeader();

    fseek(sq_file, sizeof(SQ_IO_Header), SEEK_SET);

    size_t size = header.nvpoints * header.nzpoints;

    if (size == 0)
        size = header.nvpoints;

    if (fread(data, sizeof(double), size, sq_file) != size)
        throw std::runtime_error("fread error in data SQLookupTableFile!");
}

}



