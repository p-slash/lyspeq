#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

#include <cstring>
#include <cstdlib>
#include <cstdio>

ConfigFile::ConfigFile(const char *fname)
{
    strcpy(file_name, fname);
    
    no_params = 0;
}

void ConfigFile::addKey(const std::string key, void *variable, VariableType vt)
{
    if (variable == NULL)   return;
    
    vpair new_pair = {variable, vt};

    key_umap[key] = new_pair;

    no_params++;
}

void ConfigFile::readAll()
{
    FILE *config_file = open_file(file_name, "r");
    
    char line[1024], buffer_key[200], buffer_value[200];

    while (!feof(config_file))
    {
        fgets(line, 1024, config_file);

        if (line[0] == '%' || line[0] == '#')
            continue;

        if (sscanf(line, "%s %s", buffer_key, buffer_value) < 2)
            continue;

        kumap_itr = key_umap.find(buffer_key);
        
        if (kumap_itr == key_umap.end())
        {
            printf("WARNING: %s NOT FOUND!\n", buffer_key);
        }
        else
        {
            vpair *tmp_vp = &(*kumap_itr).second;

            switch (tmp_vp->vt)
            {
                case INTEGER:
                    *((int *) tmp_vp->address) = atoi(buffer_value);
                    break;

                case DOUBLE:
                    *((double *) tmp_vp->address) = atof(buffer_value);
                    break;

                case STRING:
                    strcpy((char *) tmp_vp->address, buffer_value);
                    break;
            }
        }
    }
    
    fclose(config_file);
}
