#include "sq_lookup_table.hpp"
#include "io_helper_functions.hpp"

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

void readData(double *v_ij, double *z_ij, double *data)
{

}




