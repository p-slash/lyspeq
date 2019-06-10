#include "io/logger.hpp"

#include <cassert>
#include <cstdarg>
#include <sstream> // std::ostringstream

#include "io/io_helper_functions.hpp"

Logger::Logger()
{
    stdfile = stdout;
    errfile = NULL;
    iofile  = NULL;
}

Logger::~Logger()
{
    close();
}

void Logger::open(const char *outdir)
{
    std_fname = outdir;
    std_fname += "/log.txt";
    stdfile = open_file(std_fname.c_str(), "w");

    err_fname = outdir;
    err_fname += "/error_log.txt";
    errfile = open_file(err_fname.c_str(), "w");

    io_fname = outdir;
    io_fname += "/io_log.txt";
    iofile = open_file(io_fname.c_str(), "w");
}

void Logger::close()
{
    if (stdfile != stdout)  fclose(stdfile);
    if (errfile != NULL)    fclose(errfile);
    if (iofile  != NULL)    fclose(iofile);
}

void Logger::reopen()
{
    stdfile = open_file(std_fname.c_str(), "a");
    errfile = open_file(err_fname.c_str(), "a");
    iofile  = open_file(io_fname.c_str(),  "a");
}

void Logger::log(LOG_TYPE lt, const char *fmt, ...)
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
            vfprintf(stderr, fmt, args);
            if (errfile == NULL) throw "error_log.txt is not open.\n";
            vfprintf(errfile, fmt, args);
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

std::string Logger::getFileName(LOG_TYPE lt) const
{
    switch(lt)
    {
        case STD:   return std_fname;
        case ERR:   return err_fname;
        case IO:    return io_fname;
    }

    return NULL;
}






