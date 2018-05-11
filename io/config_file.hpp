#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

enum VariableType
{
    INTEGER,
    DOUBLE,
    STRING
};
    
class ConfigFile
{
    char file_name[300];

    static const int MAX_PARAMETERS = 100;
    static const int MAX_KEY_LENGTH = 50;

    VariableType types[ConfigFile::MAX_PARAMETERS];
    void *address[ConfigFile::MAX_PARAMETERS];
    char keys[ConfigFile::MAX_PARAMETERS][ConfigFile::MAX_KEY_LENGTH];

    int no_params;

    int getIndex(const char *key);
    
public:
    ConfigFile(const char *fname);
    ~ConfigFile() {};

    void addKey(const char *key, void *variable, VariableType vt);
    void readAll();
};

#endif
