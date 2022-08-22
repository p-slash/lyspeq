#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <unordered_map>
#include <string>

typedef std::unordered_map<std::string, std::string> config_map;

class ConfigFile
{
    config_map key_umap;

public:
    ConfigFile(const config_map &default_config={}): key_umap (default_config) {};
    ~ConfigFile() {};

    std::string get(const std::string &key, const std::string &fallback="") const;
    double getDouble(const std::string &key, double fallback=0) const;
    int getInteger(const std::string &key, int fallback=0) const;
    // bool getBool(const std::string &key, bool fallback=false) const;

    // add map in source only if it does not exist in target.
    void addDefaults(const config_map &default_config);
    void readFile(const std::string &fname);

    void writeConfig(FILE *toWrite, std::string prefix="# ") const;
};

#endif
