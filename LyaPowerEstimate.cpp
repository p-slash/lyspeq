#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "core/quadratic_estimate.hpp"
#include "io/config_file.hpp"
#include "io/io_helper_functions.hpp"

void convert_lambda2v(double *lambda, int size)
{
    #define SPEED_OF_LIGHT 299792.458
    #define LYA_REST 1215.67

    double mean_lambda = 0;

    for (int i = 0; i < size; i++)
    {
        mean_lambda += lambda[i] / size;
    }

    for (int i = 0; i < size; i++)
    {
        lambda[i] = 2. * SPEED_OF_LIGHT * (1 - sqrt(mean_lambda / lambda[i]));
    }
}

int main(int argc, char const *argv[])
{
    const char *FNAME_CONFIG = argv[1];
    
    char FNAME_MOCK[300], \
         OUTPUT_DIR[300], \
         OUTPUT_FILEBASE[300],\
         buf[500];

    int DATA_SIZE, NBin, NUMBER_OF_ITERATIONS, LINEAR_LOG;
    double K_0, K_1, *k_edges;

    double  *lambda, \
            *delta_f, \
            *noise;

    try
    {
        // Set up config file to read variables.
        ConfigFile cFile(FNAME_CONFIG);

        cFile.addKey("K0", &K_0, DOUBLE);
        cFile.addKey("K1", &K_1, DOUBLE);
        cFile.addKey("NumberOfBins", &NBin, INTEGER);
        cFile.addKey("LinLog", &LINEAR_LOG, INTEGER);

        cFile.addKey("FileName", FNAME_MOCK, STRING);

        cFile.addKey("OutputDir", OUTPUT_DIR, STRING);
        cFile.addKey("OutputFileBase", OUTPUT_FILEBASE, STRING);

        cFile.addKey("NumberOfIterations", &NUMBER_OF_ITERATIONS, INTEGER);

        cFile.readAll();

        // Construct k edges
        k_edges = new double[NBin + 1];
        for (int i = 0; i < NBin + 1; i++)
        {
            k_edges[i] = K_0 + K_1 * i;

            if (LINEAR_LOG == 10)
            {
                k_edges[i] = pow(10., k_edges[i]);
            }
        }

        // Construct and read data arrays
        FILE *toRead = open_file(FNAME_MOCK, "r");
        fscanf(toRead, "%d\n", &DATA_SIZE);

        printf("DATA size: %d\n", DATA_SIZE);

        lambda  = new double[DATA_SIZE];
        delta_f = new double[DATA_SIZE];
        noise   = new double[DATA_SIZE];

        double mean_f = 0;
        for (int i = 0; i < DATA_SIZE; i++)
        {
            fscanf(toRead, "%le %le\n", &lambda[i], &delta_f[i]);
            noise[i] = 0.02;

            mean_f += delta_f[i] / DATA_SIZE;
        }

        // Convert to mean flux
        printf("Mean flux: %lf\n", mean_f);

        for (int i = 0; i < DATA_SIZE; i++)
        {
            delta_f[i] = (delta_f[i] / mean_f) - 1.;
        }

        convert_lambda2v(lambda, DATA_SIZE);
        
        OneDQuadraticPowerEstimate qps(DATA_SIZE, lambda, delta_f, noise, NBin, k_edges);

        qps.iterate(NUMBER_OF_ITERATIONS);

        sprintf(buf, "%s/%s_qso_power_estimate.dat", OUTPUT_DIR, OUTPUT_FILEBASE);
        qps.write_spectrum_estimate(buf);

        delete [] lambda;
        delete [] noise;
        delete [] delta_f;
    }
    catch (std::exception& e)
    {
        printf("%s\n", e.what());
        return -1;
    }
    catch (const char* msg)
    {
        printf("%s\n", msg);
        return -1;
    }

    return 0;
}










