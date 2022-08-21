#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

#include <cstring> // strcpy
#include <cstdlib>
#include <cstdio>

ConfigFile::ConfigFile(const std::string &fname)
{
    file_name = fname;
    
    no_params = 0;
}

// void ConfigFile::addKey(const std::string key, void *variable, VariableType vt)
// {
//     if (variable == NULL)   return;
    
//     vpair new_pair = {variable, vt};

//     key_umap[key] = new_pair;

//     no_params++;
// }

void ConfigFile::readAll()
{
    FILE *config_file = ioh::open_file(file_name.c_str(), "r");
    
    char line[1024], buffer_key[200], buffer_value[500];

    while (!feof(config_file))
    {
        if (fgets(line, 1024, config_file) == NULL)
            continue;

        if (line[0] == '%' || line[0] == '#')
            continue;

        if (sscanf(line, "%s %s", buffer_key, buffer_value) < 2)
            continue;

        key_umap[std::string(buffer_key)] = std::string(buffer_value);
    }
    
    fclose(config_file);
}

std::string ConfigFile::get(const std::string &key, const std::string &fallback) const
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
        return kumap_itr->second;
    else
        return fallback;
}

double ConfigFile::getDouble(const std::string &key, double fallback) const
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
        return std::stod(kumap_itr->second);
    else
        return fallback;
}

int ConfigFile::getInteger(const std::string &key, int fallback) const
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
        return std::stoi(kumap_itr->second);
    else
        return fallback;
}



