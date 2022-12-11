#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

#include <cstdlib>
#include <cstdio>
#include <stdexcept> // std::invalid_argument

ConfigFile::ConfigFile(const config_map &default_config) :
key_umap (default_config)
{
    for (const auto &entry : key_umap)
        key_call_counter[entry.first] = 0;
}

void ConfigFile::addDefaults(const config_map &default_config)
{
    for (const auto &entry : default_config)
    {
        auto kumap_itr = key_umap.find(entry.first);
        if (kumap_itr == key_umap.end())
        {
            key_umap[entry.first] = entry.second;
            key_call_counter[entry.first] = 0;
        }
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

        std::string curr_key(buffer_key);
        key_umap[curr_key] = std::string(buffer_value);
        key_call_counter[curr_key] = 0;
    }
    
    fclose(config_file);
}

std::string ConfigFile::get(const std::string &key, const std::string &fallback)
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
    {
        ++key_call_counter[key];
        return kumap_itr->second;
    }
    else
        return fallback;
}

double ConfigFile::getDouble(const std::string &key, double fallback)
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
    {
        ++key_call_counter[key];
        return std::stod(kumap_itr->second);
    }
    else
        return fallback;
}

int ConfigFile::getInteger(const std::string &key, int fallback)
{
    auto kumap_itr = key_umap.find(key);

    if (kumap_itr != key_umap.end())
    {
        ++key_call_counter[key];
        return std::stoi(kumap_itr->second);
    }
    else
        return fallback;
}

void ConfigFile::writeConfig(FILE *toWrite, std::string prefix) const
{
    const char *c_prefix = prefix.c_str();
    fprintf(toWrite, "%sUsing following configuration parameters:\n",
        c_prefix);
    for (const auto &entry : key_umap)
        fprintf(toWrite, "%s%s %s\n", c_prefix,
            entry.first.c_str(), entry.second.c_str());
}

bool _isNOTin(const std::vector<std::string> &ignored_keys, const std::string &elem)
{
    for (const auto &s : ignored_keys)
        if (s == elem)
            return false;
    return true;
}

void ConfigFile::checkUnusedKeys(const std::vector<std::string> &ignored_keys) const
{
    std::vector<std::string> uncalled;

    for (const auto &entry : key_call_counter)
        if ((entry.second == 0) && _isNOTin(ignored_keys, entry.first))
            uncalled.push_back(entry.first);

    if (!uncalled.empty())
        fprintf(stderr, "ERROR: unused keys in config:\n");
    for (const auto &entry : uncalled)
        fprintf(stderr, "--  %s\n", entry.c_str());

    if (!uncalled.empty())
        throw std::invalid_argument("Unused keys in config.");
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


