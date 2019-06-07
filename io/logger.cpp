#include "io/logger.hpp"

#include <cassert>
#include <cstdarg>

#include "io/io_helper_functions.hpp"

Logger::Logger()
{
    stdfile = stdout;
    errfile = NULL;
    iofile  = NULL;
}

Logger::~Logger()
{
    if (stdfile != stdout)    fclose(stdfile);
    if (errfile != NULL)    fclose(errfile);
    if (iofile  != NULL)    fclose(iofile);
}

void Logger::open(const char *outdir)
{
    char buf[700];
    sprintf(buf, "%s/log.txt", outdir);
    stdfile = open_file(buf, "wa");
    freopen(buf, "wa", stdout);

    sprintf(buf, "%s/error_log.txt", outdir);
    errfile = open_file(buf, "wa");

    sprintf(buf, "%s/io_log.txt", outdir);
    iofile = open_file(buf, "wa");
}

void Logger::log(LOG_TYPE lt, const char *fmt, ...)
{
    assert(lt == STD || lt == ERR || lt == IO);

    va_list args;
    va_start(args, fmt);

    switch(lt)
    {
        case STD:
            // if (stdfile == NULL) stdfile=stdout;
            vfprintf(stdfile, fmt, args);
            fflush(stdfile);
            break;
        case ERR:
            if (errfile == NULL) throw "error_log.txt is not open.\n";
            vfprintf(errfile, fmt, args);
            vfprintf(stderr, fmt, args);
            fflush(errfile);
            break;
        case IO:
            if (iofile  == NULL) throw "io_log.txt is not open.\n";
            vfprintf(iofile, fmt, args);
            fflush(iofile);
            break;
    }

    va_end(args);
}

void Logger::copy()
{
    // f = fopen(stdout, "r");

    char buffer;

    while(!feof(stdout))
    {
        buffer = fgetc(stdout);
        fputc(buffer, stdfile);
        // printf("%c", buffer);
    }
    // fclose(stdout);
}





