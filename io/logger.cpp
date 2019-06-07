#include "io/logger.hpp"

Logger::Logger(const char *outdir)
{
    char buf[700];
    sprintf(buf, "%s/log.txt", outdir);
    stdfile = open_file(buf, 'w');

    sprintf(buf, "%s/error_log.txt", outdir);
    errfile = open_file(buf, 'w');

    sprintf(buf, "%s/io_log.txt", outdir);
    iofile = open_file(buf, 'w');
}

Logger::~Logger()
{
    fclose(stdfile);
    fclose(errfile);
    fclose(iofile);
}

Logger::log(LOG_TYPE lt, const char *fmt, ...)
{
    assert(lt == STD || lt == ERR || lt == IO);

    va_list args;
    va_start(args, fmt);

    switch(lt)
    {
        case STD:
            vfprintf(stdfile, fmt, args);
            fflush(stdfile);
            break;
        case ERR:
            vfprintf(errfile, fmt, args);
            fflush(errfile);
            break;
        case IO:
            vfprintf(iofile, fmt, args);
            fflush(iofile);
            break;
    }

    va_end(args);
}





