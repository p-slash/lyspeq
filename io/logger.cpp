#include "io/logger.hpp"

#include <cstdarg>
#include <sstream> // std::ostringstream
#include <stdexcept>

#include "io/io_helper_functions.hpp"
// #include "core/global_numbers.hpp"

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
    stdfile  = stdout;
    errfile  = stderr;
    timefile = NULL;

    errfileOpened = false;
}

Logger::~Logger()
{
    close();
}

void Logger::open(const char *outdir, int np)
{
    this_pe = np;
    std::ostringstream oss_fname(outdir, std::ostringstream::ate);
    
    oss_fname << "/error_log" << this_pe << ".txt";
    err_fname = oss_fname.str();
    // only open if error message exists
    // errfile = ioh::open_file(err_fname.c_str(), "w");

    if (this_pe != 0) return;

    oss_fname.str(outdir);
    oss_fname.clear();
    oss_fname << "/log" << this_pe << ".txt";
    std_fname = oss_fname.str();
    stdfile   = ioh::open_file(std_fname.c_str(), "w");

    oss_fname.str(outdir);
    oss_fname.clear();
    oss_fname << "/time_log" << this_pe << ".txt";
    time_fname = oss_fname.str();
    timefile   = ioh::open_file(time_fname.c_str(), "w");
}

void Logger::close()
{
    if (stdfile  != stdout)  fclose(stdfile);
    if (errfile  != stderr)  fclose(errfile);
    if (timefile != NULL)    fclose(timefile);
}

void Logger::reopen()
{
    if (errfileOpened)
        errfile = ioh::open_file(err_fname.c_str(), "a");

    if (this_pe != 0) return;
    stdfile   = ioh::open_file(std_fname.c_str(), "a");
    timefile  = ioh::open_file(time_fname.c_str(),  "a");
}

std::string Logger::getFileName(TYPE::LOG_TYPE lt) const
{
    switch(lt)
    {
        case TYPE::STD:   return std_fname;
        case TYPE::ERR:   return err_fname;
        case TYPE::TIME:  return time_fname;
    }

    return NULL;
}

void Logger::STD(const char *fmt, ...)
{
    if (this_pe != 0) return;

    va_list args;
    va_start(args, fmt);

    write_log(stdfile, stdout, fmt, args);

    va_end(args);
}

void Logger::ERR(const char *fmt, ...)
{
    if (!errfileOpened && !err_fname.empty())
    {
        errfile = ioh::open_file(err_fname.c_str(), "w");
        errfileOpened = true;
    }

    va_list args;
    va_start(args, fmt);

    write_log(errfile, stderr, fmt, args);

    va_end(args);
}

void Logger::TIME(const char *fmt, ...)
{    
    if (this_pe != 0) return;

    va_list args;
    va_start(args, fmt);

    write_log(timefile, timefile, fmt, args);

    va_end(args);
}










