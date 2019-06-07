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
    Logger(const char *outdir);
    ~Logger();
    
    log(LOG_TYPE lt, const char *fmt, ...);
};
#endif
