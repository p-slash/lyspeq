#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

#include <cstring> // strcpy
#include <cstdlib>
#include <cstdio>
#include <stdexcept> // std::invalid_argument

// ConfigFile::ConfigFile(const config_map &default_config) : 
// {
//     key_umap = default_config;
// }

// void ConfigFile::addKey(const std::string key, void *variable, VariableType vt)
// {
//     if (variable == NULL)   return;
    
//     vpair new_pair = {variable, vt};

//     key_umap[key] = new_pair;

//     no_params++;
// }

void ConfigFile::addDefaults(const config_map &default_config)
{
    for (auto entry : default_config)
    {
        auto kumap_itr = key_umap.find(entry.first);
        if (kumap_itr == key_umap.end())
            key_umap[entry.first] = entry.second;
    }
}

void ConfigFile::readFile(const std::string &fname)
{
    if (fname.empty())
        throw std::invalid_argument("ConfigFile::readAll() fname is empty.");

    FILE *config_file = ioh::open_file(fname.c_str(), "r");
    
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

void ConfigFile::writeConfig(FILE *toWrite, std::string prefix) const
{
    const char *c_prefix = prefix.c_str();
    fprintf(toWrite, "%sUsing following configuration parameters:\n",
        c_prefix);
    for (auto entry : key_umap)
        fprintf(toWrite, "%s%s %s\n", c_prefix,
            entry.first.c_str(), entry.second.c_str());
}

// bool ConfigFile::getBool(const std::string &key, bool fallback=false) const
// {
//     const std::string true_values[] = {"ON", "on", "true", "True", "TRUE"};
//     auto kumap_itr = key_umap.find(key);

//     bool result = fallback;
//     if (kumap_itr != key_umap.end())
//     {
//         std::string value = kumap_itr->second;
//         if (value == "ON" || value == "on" || value == "true")
//         return std::stoi(kumap_itr->second);
//     }
//     return result;
// }


