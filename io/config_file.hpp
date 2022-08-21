#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <unordered_map>
#include <string>

class ConfigFile
{
    std::string file_name;

    std::unordered_map<std::string, std::string> key_umap;

    int no_params;
    
public:
    ConfigFile(const std::string &fname);
    ~ConfigFile() {};

    std::string get(const std::string &key, const std::string &fallback="") const;
    double getDouble(const std::string &key, double fallback=0) const;
    int getInteger(const std::string &key, int fallback=0) const;

    void readAll();
};

#endif
