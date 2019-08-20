#include "io/sq_lookup_table_file.hpp"
#include "io/io_helper_functions.hpp"

void QTableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_Q, \
                                int r, double k1, double k2)
{
    sprintf(fname, "%s/%s_R%d_k%.1e_%.1e.dat", OUTPUT_DIR, OUTPUT_FILEBASE_Q, r, k1, k2);
}

void STableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_S, \
                                int r)
{
    sprintf(fname, "%s/%s_R%d.dat", OUTPUT_DIR, OUTPUT_FILEBASE_S, r);
}

// return y = c + deltaY / (N-1) * n
double getLinearlySpacedValue(double c, double delta_y, int N, int n)
{
    if (N == 1)     return c + delta_y/2;

    return c + delta_y / (N - 1.) * n;
}


SQLookupTableFile::SQLookupTableFile(const char *fname, char rw)
{
    sprintf(file_name, "%s", fname);

    read_write[0] = rw;
    read_write[1] = 'b';
    read_write[2] = '\0';

    sq_file = ioh::open_file(file_name, read_write);

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
            throw "ERROR: fread error in header SQLookupTableFile!\n";
        
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
    {
        size = header.vpoints;
    }

    int fw;

    fw = fwrite(&header, sizeof(sq_io_header), 1, sq_file);
    if (fw != 1)
        throw "ERROR: fwrite error in header SQLookupTableFile!\n";
    fw = fwrite(data, sizeof(double), size, sq_file);
    if (fw != 1)
        throw "ERROR: fwrite error in data SQLookupTableFile!\n";
}

void SQLookupTableFile::readData(double *data)
{
    readHeader();
    
    fseek(sq_file, sizeof(sq_io_header), SEEK_SET);
    
    size_t size = header.vpoints * header.zpoints;
    
    if (size == 0)
    {
        size = header.vpoints;
    }

    if (fread(data, sizeof(double), size, sq_file) != size)
        throw "ERROR: fread error in data SQLookupTableFile!\n";
}




