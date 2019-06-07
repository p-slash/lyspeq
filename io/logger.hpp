#ifndef LOGGER_H
#define LOGGER_H

enum LOG_TYPE
{
    STD,
    ERR,
    IO
};

class Logger
{
    FILE *stdfile, *errfile, *iofile;
public:
    Logger();
    ~Logger();
    
    void open(const char *outdir);
    void log(LOG_TYPE lt, const char *fmt, ...);
};
#endif
