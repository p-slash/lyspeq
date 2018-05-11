#include "config_file.hpp"
#include "io_helper_functions.hpp"

#include <cstring>
#include <cstdlib>
#include <cstdio>

ConfigFile::ConfigFile(const char *fname)
{
    strcpy(file_name, fname);
    
    no_params = 0;
}

int ConfigFile::getIndex(const char *key)
{
    for (int i = 0; i < no_params; i++)
    {
        if (strcmp(keys[i], key) == 0)
        {
            return i;
        }
    }

    return -1;
}

void ConfigFile::addKey(const char *key, void *variable, VariableType vt)
{
    if (no_params == ConfigFile::MAX_PARAMETERS)
    {
        printf("WARNING: REACHED MAXIMUM NUMBER OF PARAMETERS THAT CAN BE READ IN A CONFIG FILE!!\n");

        return;
    }

    types[no_params] = vt;
    address[no_params] = variable;

    strcpy(keys[no_params], key);

    no_params++;
}

void ConfigFile::readAll()
{
    FILE *config_file = open_file(file_name, "r");
    
    char line[1024], buffer_key[200], buffer_value[200];
    int index;

    while (!feof(config_file))
    {
        fgets(line, 1024, config_file);

        if (line[0] == '%' || line[0] == '#')
            continue;

        if (sscanf(line, "%s %s", buffer_key, buffer_value) < 2)
            continue;

        index = getIndex(buffer_key);

        if (index == -1)
        {
            printf("WARNING: %s NOT FOUND!\n", buffer_key);
        }
        else
        {
            switch (types[index])
            {
                case INTEGER:
                    *((int *) address[index]) = atoi(buffer_value);
                    break;

                case DOUBLE:
                    *((double *) address[index]) = atof(buffer_value);
                    break;

                case STRING:
                    strcpy((char *) address[index], buffer_value);
                    break;
            }
        }
    }
    
    fclose(config_file);
}
