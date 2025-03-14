#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <unordered_map>
#include <string>
#include <vector>

typedef std::unordered_map<std::string, std::string> config_map;
typedef std::unordered_map<std::string, int> call_counts;
class ConfigFile
{
    config_map key_umap;
    call_counts key_call_counter;

public:
    ConfigFile(const config_map &default_config={});
    ~ConfigFile() {};

    std::string get(const std::string &key, const std::string &fallback="");
    double getDouble(const std::string &key, double fallback=0);
    int getInteger(const std::string &key, int fallback=0);
    // bool getBool(const std::string &key, bool fallback=false) const;

    void update(const std::string &key, const std::string &value) {
        key_umap[key] = value;
    }

    // add map in source only if it does not exist in target.
    void addDefaults(const config_map &default_config);
    void addDefaults(const ConfigFile &config) { addDefaults(config.key_umap); };
    void readFile(const std::string &fname);

    void writeConfig(FILE *toWrite, std::string prefix="# ") const;
    void checkUnusedKeys(const std::vector<std::string> &ignored_keys={}) const;
};

#endif
