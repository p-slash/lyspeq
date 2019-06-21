#include "io/logger.hpp"

#include <cassert>
#include <cstdarg>
#include <sstream> // std::ostringstream

#include "io/io_helper_functions.hpp"

namespace LOG
{
    Logger LOGGER;
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

// void Logger::log(LOG_TYPE lt, const char *fmt, ...)
// {
//     assert(lt == STD || lt == ERR || lt == IO);

//     va_list args;
//     va_start(args, fmt);

//     switch(lt)
//     {
//         case STD:
//             vfprintf(stdfile, fmt, args);
//             //if (stdfile != stdout)  vfprintf(stdout, fmt, args);
//             fflush(stdfile);
//             break;
//         case ERR:
//             vfprintf(errfile, fmt, args);
//             //if (errfile != stderr) vfprintf(stderr, fmt, args);
//             fflush(errfile);
//             break;
//         case IO:
//             if (iofile  == NULL) throw "io_log.txt is not open.\n";
//             vfprintf(iofile, fmt, args);
//             fflush(iofile);
//             break;
//     }

//     va_end(args);
// }

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
    va_list args;
    va_start(args, fmt);
    if (iofile  == NULL) throw "io_log.txt is not open.\n";
    vfprintf(iofile, fmt, args);
    fflush(iofile);
    va_end(args);
}

void Logger::STD(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stdfile, fmt, args);
    fflush(stdfile);
    va_end(args);
}
void Logger::ERR(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(errfile, fmt, args);
    fflush(errfile);
    va_end(args);
}




