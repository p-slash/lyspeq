#include <algorithm>
#include <cstdio>

#include "mathtools/real_field.hpp"
#include "tests/test_utils.hpp"
#include "io/qso_file.hpp"


void write2VectorsToFile(
        const char *fname,
        const std::vector<double> &x,
        const std::vector<double> &y
) {
    FILE *toWrite;
        
    toWrite = fopen(fname, "w");

    for (int i = 0; i < x.size(); ++i)
        fprintf(toWrite, "%.14le %.14e\n", x[i], y[i]);

    fclose(toWrite);
}


void write2VectorsToFile(
        const char *fname,
        const std::vector<double> &x,
        const std::vector<std::complex<double>> &y
) {
    FILE *toWrite;
        
    toWrite = fopen(fname, "w");

    for (int i = 0; i < x.size(); ++i)
        fprintf(toWrite, "%.14le %.14e\n", x[i], y[i]);

    fclose(toWrite);
}

void writeArrayToFile(
        const char *fname,
        const double *x, int N
) {
    FILE *toWrite;
        
    toWrite = fopen(fname, "w");

    for (int i = 0; i < N; ++i)
        fprintf(toWrite, "%.14le\n", x[i]);

    fclose(toWrite);
}


int main(int argc, char *argv[])
{
    if (argc<2)
    {
        fprintf(stderr, "Missing fits file!\n");
        return -1;
    }

    specifics::OVERSAMPLING_FACTOR = 4;

    auto fname = std::string(argv[1]);

    qio::QSOFile qFile(fname, qio::Picca);
    qFile.readParameters();
    qFile.readData();

    qFile.readAllocResolutionMatrix();
    qFile.recalcDvDLam();  // meansnr: 0.67616031758204
    qFile.calcRkmsFromRMat();
    printf("Rkms: %.5e\n", qFile.R_kms);  // 47.5047929893881

    RealField r2c(1024, 1);

    qFile.setRealField(r2c);
    qFile.calcAverageWindowFunctionFromRMat(r2c);

    std::string window_fname = std::string(SRCDIR) + "/tests/output/window2_k_dia.txt";
    write2VectorsToFile(window_fname.c_str(), r2c.k, r2c.field_x);
    for (int i = 0; i < r2c.size_k(); i += 20)
        printf("%.5e  %.5e\n", r2c.k[i], r2c.field_x[i]);

    std::vector<std::complex<double>> temp(r2c.size_k());
    std::transform(
        r2c.field_x.begin(), r2c.field_x.end(),
        temp.begin(),
        [](double d) { return exp(d / 2); });
    r2c.zero_field_x();
    std::copy(temp.begin(), temp.end(), r2c.field_k.begin());
    r2c.fftK2X();

    window_fname = std::string(SRCDIR) + "/tests/output/window2_x_dia.txt";
    write2VectorsToFile(window_fname.c_str(), r2c.x, r2c.field_x);



    // Ivar: [0.51403648, 0.03894964, 0.54754096, 0.89146243, 0.98388764]
    qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    qFile.recalcDvDLam();
    qFile.calcRkmsFromRMat();
    printf("Rkms: %.5e\n", qFile.R_kms);  // 47.5047929893881
    // qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");


    qFile.setRealField(r2c);
    qFile.calcAverageWindowFunctionFromRMat(r2c);
    window_fname = std::string(SRCDIR) + "/tests/output/window2_k_osamp.txt";
    write2VectorsToFile(window_fname.c_str(), r2c.k, r2c.field_x);
    for (int i = 0; i < r2c.size_k(); i += 20)
        printf("%.5e  %.5e\n", r2c.k[i], r2c.field_x[i]);

    temp.resize(r2c.size_k());
    std::fill(temp.begin(), temp.end(), 0);
    std::transform(
        r2c.field_x.begin(), r2c.field_x.end(),
        temp.begin(),
        [](double d) { return exp(d / 2); });
    r2c.zero_field_x();
    std::copy(temp.begin(), temp.end(), r2c.field_k.begin());
    r2c.fftK2X();
    window_fname = std::string(SRCDIR) + "/tests/output/window2_x_osamp.txt";
    write2VectorsToFile(window_fname.c_str(), r2c.x, r2c.field_x);
    return 0;
}

