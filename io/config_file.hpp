#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <unordered_map>
#include <string>

enum VariableType
{
    INTEGER,
    DOUBLE,
    STRING
};

typedef struct
{
    void *address;
    VariableType vt;
} vpair;

class ConfigFile
{
    char file_name[300];

    std::unordered_map<std::string, vpair> key_umap;
    std::unordered_map<std::string, vpair>::iterator kumap_itr;

    int no_params;
    
public:
    ConfigFile(const char *fname);
    ~ConfigFile() {};

    void addKey(const std::string key, void *variable, VariableType vt);

    // If key is not found in file, variable is untouched.
    void readAll();
};

#endif
