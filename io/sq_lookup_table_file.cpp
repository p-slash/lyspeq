#include "io/sq_lookup_table_file.hpp"
#include "io/io_helper_functions.hpp"

#include <stdexcept>
#include <sstream>
#include <iomanip>

std::string sqhelper::QTableFileNameConvention(const std::string &OUTPUT_DIR, 
    const std::string &OUTPUT_FILEBASE_Q, int r, double dv, double k1, double k2)
{
    std::ostringstream qt_fname;

    qt_fname << OUTPUT_DIR << "/" << OUTPUT_FILEBASE_Q  << "_R" << r 
        << std::fixed << std::setprecision(1) << "_dv" << dv
        << std::scientific << std::setprecision(1) << "_k" << k1 << "_" << k2
        << ".dat";

    return qt_fname.str();
}

std::string sqhelper::STableFileNameConvention(const std::string &OUTPUT_DIR, 
    const std::string &OUTPUT_FILEBASE_S, int r, double dv)
{
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

void SQLookupTableFile::setHeader(  int nv, int nz, double len_v, double len_z, \
                                    int R, double dv, \
                                    double ki, double kf)
{
    if (read_write[0] == 'r')
    {
        printf("WARNING: Setting header while reading SQLookupTableFile does nothing!\n");
        return;
    }

    header.vpoints = nv;
    header.zpoints = nz;

    header.v_length = len_v;
    header.z_length = len_z;

    header.spectrograph_resolution = R;
    header.pixel_width = dv;

    header.initial_k = ki;
    header.final_k = kf;

    isHeaderSet = true;
}

void SQLookupTableFile::readHeader()
{
    if (!isHeaderSet)
    {
        rewind(sq_file);
        if (fread(&header, sizeof(sq_io_header), 1, sq_file) != 1)
            throw std::runtime_error("fread error in header SQLookupTableFile!");
        
        isHeaderSet = true;
    }
}

void SQLookupTableFile::readHeader( int &nv, int &nz, double &len_v, double &len_z, \
                                    int &R, double &dv, \
                                    double &ki, double &kf)
{

    if (read_write[0] == 'w')
    {
        printf("WARNING: Reading header while writing SQLookupTableFile does nothing!\n");
        return;
    }
    
    readHeader();

    nv = header.vpoints;
    nz = header.zpoints;

    len_v = header.v_length;
    len_z = header.z_length;

    R  = header.spectrograph_resolution;
    dv = header.pixel_width;
    
    ki = header.initial_k;
    kf = header.final_k;
}

void SQLookupTableFile::writeData(double *data)
{
    if (!isHeaderSet)
    {
        printf("WARNING: Set header first before writing SQLookupTableFile!\n");
        return;
    }
    
    int size = header.vpoints * header.zpoints;
    
    if (size == 0)
        size = header.vpoints;

    int fw;

    fw = fwrite(&header, sizeof(sq_io_header), 1, sq_file);
    if (fw != 1)
        throw std::runtime_error( "fwrite error in header SQLookupTableFile!");

    fw = fwrite(data, sizeof(double), size, sq_file);
    if (fw != 1)
        throw std::runtime_error( "fwrite error in data SQLookupTableFile!");
}

void SQLookupTableFile::readData(double *data)
{
    readHeader();
    
    fseek(sq_file, sizeof(sq_io_header), SEEK_SET);
    
    size_t size = header.vpoints * header.zpoints;
    
    if (size == 0)
        size = header.vpoints;

    if (fread(data, sizeof(double), size, sq_file) != size)
        throw std::runtime_error("fread error in data SQLookupTableFile!");
}




