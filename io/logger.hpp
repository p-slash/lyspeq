#ifndef LOGGER_H
#define LOGGER_H

#include <cstdio>
#include <string>

namespace LOG
{
    namespace TYPE
    {
        enum LOG_TYPE
        {
            STD,
            ERR,
            TIME
        };
    }

    // Simple logging into a given directory. Only writes on master process.
    // Writes STD to log.txt, ERR to error_log.txt, IO to io_log.txt
    // Default of STD is stdout, ERR is stderr.
    // Keeps files open and flushes immediately after logging.

    // Example for Logger:
    //     LOGGER.open("./");
    //     LOGGER.STD("This goes into ./log.txt and stdout.\n");
    class Logger
    {
        int this_pe;
        std::string std_fname, err_fname, time_fname;
        FILE *stdfile, *errfile, *timefile;
    public:
        Logger();
        ~Logger();

        std::string getFileName(TYPE::LOG_TYPE lt) const;

        void open(const char *outdir, int np);
        void close();
        void reopen();

        void STD(const char *fmt, ...);
        void ERR(const char *fmt, ...);
        void TIME(const char *fmt, ...);
    };

    extern Logger LOGGER;
}
#endif
