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

void write_log(FILE *ff, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(ff, fmt, args);
    fflush(ff);
    va_end(args);
}

using namespace LOG;

Logger::Logger()
{
    std_fname = "";
    err_fname = "";
    io_fname  = "";
    
    stdfile = stdout;
    errfile = stderr;
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
    stdfile = ioh::open_file(std_fname.c_str(), "w");

    err_fname = outdir;
    err_fname += "/error_log.txt";
    errfile = ioh::open_file(err_fname.c_str(), "w");

    io_fname = outdir;
    io_fname += "/io_log.txt";
    iofile = ioh::open_file(io_fname.c_str(), "w");
}

void Logger::close()
{
    if (stdfile != stdout)  fclose(stdfile);
    if (errfile != stderr)  fclose(errfile);
    if (iofile  != NULL)    fclose(iofile);
}

void Logger::reopen()
{
    stdfile = ioh::open_file(std_fname.c_str(), "a");
    errfile = ioh::open_file(err_fname.c_str(), "a");
    iofile  = ioh::open_file(io_fname.c_str(),  "a");
}

std::string Logger::getFileName(TYPE::LOG_TYPE lt) const
{
    switch(lt)
    {
        case TYPE::STD:   return std_fname;
        case TYPE::ERR:   return err_fname;
        case TYPE::IO:    return io_fname;
    }

    return NULL;
}

void Logger::IO(const char *fmt, ...)
{
    if (iofile  == NULL)
        throw std::runtime_error("io_log.txt is not open.");

    write_log(iofile, fmt);
}

void Logger::STD(const char *fmt, ...)
{
    if (stdfile != stdout)
        write_log(stdout, fmt);

    write_log(stdfile, fmt);
}
void Logger::ERR(const char *fmt, ...)
{
    if (errfile != stderr)
        write_log(stderr, fmt);

    write_log(errfile, fmt);
}




