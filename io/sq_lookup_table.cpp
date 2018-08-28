#include "sq_lookup_table.hpp"
#include "io_helper_functions.hpp"

void QTableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_Q, \
                                int r, double k1, double k2, double z)
{
    sprintf(fname, "%s/%s_R%d_k%.1e_%.1e_z%.1f.dat", OUTPUT_DIR, OUTPUT_FILEBASE_Q, r, k1, k2, z);
}

void STableFileNameConvention(  char *fname, const char *OUTPUT_DIR, const char *OUTPUT_FILEBASE_S, \
                                int r)
{
    sprintf(fname, "%s/%s_R%d.dat", OUTPUT_DIR, OUTPUT_FILEBASE_S, r);
}

// return y = c + deltaY / (N-1) * n
double getLinearlySpacedValue(double c, double delta_y, int N, int n)
{
    return c + delta_y / (N - 1.) * n;
}


SQLookupTable::SQLookupTable(const char *fname, char rw)
{
    sprintf(file_name, "%s", fname);

    read_write[0] = rw;
    read_write[1] = 'b';
    read_write[2] = '\0';

    sq_file = open_file(file_name, read_write);

    isHeaderSet = false;
}

SQLookupTable::~SQLookupTable()
{
    fclose(sq_file);
}

void SQLookupTable::setHeader(int nv, int nz, int R, double dv, double z, double ki, double kf)
{
    if (read_write[0] == 'r')
    {
        printf("WARNING: Setting header while reading SQLookupTable does nothing!\n");
        return;
    }

    header.vpoints = nv;
    header.zpoints = nz;
    header.spectrograph_resolution = R;
    header.pixel_width = dv;
    header.redshift = z;
    header.initial_k = ki;
    header.final_k = kf;

    isHeaderSet = true;
}

void SQLookupTable::readHeader()
{
    if (!isHeaderSet)
    {
        rewind(sq_file);
        fread(&header, sizeof(sq_io_header), 1, sq_file);
        
        isHeaderSet = true;
    }
}

void SQLookupTable::readHeader(int &nv, int &nz, int &R, double &dv, double &z, double &ki, double &kf)
{

    if (read_write[0] == 'w')
    {
        printf("WARNING: Reading header while writing SQLookupTable does nothing!\n");
        return;
    }
    
    readHeader();

    nv = header.vpoints;
    nz = header.zpoints;
    R  = header.spectrograph_resolution;
    dv = header.pixel_width;
    z  = header.redshift;
    ki = header.initial_k;
    kf = header.final_k;
}

void SQLookupTable::writeData(double *data)
{
    if (!isHeaderSet)
    {
        printf("WARNING: Set header first before writing SQLookupTable!\n");
        return;
    }

    printf("Saving SQLookupTable as %s.\n", file_name);
    
    int size = header.vpoints * header.zpoints;

    fwrite(&header, sizeof(sq_io_header), 1, sq_file);
    fwrite(data, size, sizeof(double), sq_file);
}

void SQLookupTable::readData(double *data)
{
    readHeader();
    
    fseek(sq_file, sizeof(sq_io_header), SEEK_SET);

    printf("Reading SQLookupTable %s.\n", file_name);
    
    int size = header.vpoints * header.zpoints;

    fread(data, size, sizeof(double), sq_file);
}




