#ifndef LOGGER_H
#define LOGGER_H

#include <cstdio>
#include <string>

enum LOG_TYPE
{
    STD,
    ERR,
    IO
};

class Logger
{
    std::string std_fname, err_fname, io_fname;
    FILE *stdfile, *errfile, *iofile;
public:
    Logger();
    ~Logger();
    
    std::string getFileName(LOG_TYPE lt) const;

    void open(const char *outdir);
    void close();
    void reopen();
    
    void log(LOG_TYPE lt, const char *fmt, ...);
    // void log(LOG_TYPE lt, std::string str);
};
#endif
