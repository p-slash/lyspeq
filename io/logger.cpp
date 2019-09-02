#include "io/logger.hpp"

#include <cassert>
#include <cstdarg>
#include <sstream> // std::ostringstream
#include <stdexcept>

#include "io/io_helper_functions.hpp"

namespace LOG
{
    Logger LOGGER;
}

void write_log(FILE *f1, FILE *f2, const char *fmt, va_list &args1)
{
    va_list args2;
    va_copy(args2, args1);

    vfprintf(f1, fmt, args1);
    fflush(f1);

    if (f1 != f2)
    {
        vfprintf(f2, fmt, args2);
        fflush(f2);
    }

    va_end(args2);
}

using namespace LOG;

Logger::Logger()
{
    std_fname  = "";
    err_fname  = "";
    io_fname   = "";
    time_fname = "";

    stdfile  = stdout;
    errfile  = stderr;
    iofile   = stdout;
    timefile = NULL;
}

Logger::~Logger()
{
    close();
}

void Logger::open(const char *outdir)
{
    std_fname = outdir;
    std_fname += "/log.txt";
    stdfile = ioh::open_file(std_fname.c_str(), "w");

    err_fname = outdir;
    err_fname += "/error_log.txt";
    errfile = ioh::open_file(err_fname.c_str(), "w");

    io_fname = outdir;
    io_fname += "/io_log.txt";
    iofile = ioh::open_file(io_fname.c_str(), "w");

    time_fname = outdir;
    time_fname += "/time_log.txt";
    timefile = ioh::open_file(time_fname.c_str(), "w");
}

void Logger::close()
{
    if (stdfile  != stdout)  fclose(stdfile);
    if (errfile  != stderr)  fclose(errfile);
    if (iofile   != stdout)  fclose(iofile);
    if (timefile != NULL)    fclose(timefile);
}

void Logger::reopen()
{
    stdfile   = ioh::open_file(std_fname.c_str(), "a");
    errfile   = ioh::open_file(err_fname.c_str(), "a");
    iofile    = ioh::open_file(io_fname.c_str(),  "a");
    timefile  = ioh::open_file(time_fname.c_str(),  "a");
}

std::string Logger::getFileName(TYPE::LOG_TYPE lt) const
{
    switch(lt)
    {
        case TYPE::STD:   return std_fname;
        case TYPE::ERR:   return err_fname;
        case TYPE::IO:    return io_fname;
        case TYPE::TIME:  return time_fname;
    }

    return NULL;
}

void Logger::IO(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    write_log(iofile, stdout, fmt, args);

    va_end(args);
}

void Logger::STD(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    write_log(stdfile, stdout, fmt, args);

    va_end(args);
}

void Logger::ERR(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    write_log(errfile, stderr, fmt, args);

    va_end(args);
}

void Logger::TIME(const char *fmt, ...)
{
    if (timefile == NULL)
        throw std::runtime_error("timelog file is not open");
    
    va_list args;
    va_start(args, fmt);

    write_log(timefile, timefile, fmt, args);

    va_end(args);
}










