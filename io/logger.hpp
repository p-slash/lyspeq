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
            IO
        };
    }

    // Simple logging into a given directory. NOT thread safe. Only access on master process.
    // Writes STD to log.txt, ERR to error_log.txt, IO to io_log.txt
    // Default of STD is stdout, ERR is stderr.
    // Keeps files open and flushes immediately after logging.

    // Example for Logger:
    //     LOGGER.open("./");
    //     LOGGER.log(STD, "This goes into ./log.txt and stdout.\n");
    class Logger
    {
        std::string std_fname, err_fname, io_fname;
        FILE *stdfile, *errfile, *iofile;
    public:
        Logger();
        ~Logger();
        
        std::string getFileName(TYPE::LOG_TYPE lt) const;

        void open(const char *outdir);
        void close();
        void reopen();
        
        // void log(LOG_TYPE lt, const char *fmt, ...);
        void IO(const char *fmt, ...);
        void STD(const char *fmt, ...);
        void ERR(const char *fmt, ...);
    };

    extern Logger LOGGER;
}
#endif
