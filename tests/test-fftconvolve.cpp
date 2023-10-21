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
    printf("Rkms: %.2f\n", qFile.R_kms);  // 47.5047929893881

    std::vector<double> k_A, window2, lambda_vec;
    double dl_reso;

    qFile.calcAverageWindowFunctionFromRMat(k_A, window2, dl_reso);

    std::string window_fname = std::string(SRCDIR) + "/tests/output/window2_k_dia.txt";
    write2VectorsToFile(window_fname.c_str(), k_A, window2);
    for (int i = 0; i < window2.size(); i += 20)
        printf("%.5e  %.5e\n", k_A[i], window2[i]);

    RealFieldR2R r2r(k_A.size(), dl_reso);
    std::transform(
        window2.begin(), window2.end(),
        r2r.field_x.begin(),
        [](double d) { return sqrt(d); });
    r2r.fftK2X();
    lambda_vec.resize(k_A.size());
    for (int i = 0; i < k_A.size(); ++i)
        lambda_vec[i] = i * dl_reso;
    window_fname = std::string(SRCDIR) + "/tests/output/window2_x_dia.txt";
    write2VectorsToFile(window_fname.c_str(), lambda_vec, r2r.field_x);




    // Ivar: [0.51403648, 0.03894964, 0.54754096, 0.89146243, 0.98388764]
    qFile.Rmat->oversample(specifics::OVERSAMPLING_FACTOR, qFile.dlambda);
    qFile.recalcDvDLam();
    printf("Rkms: %.2f\n", qFile.R_kms);  // 47.5047929893881
    // qFile.Rmat->fprintfMatrix("debugoutput-oversampled-resomat.txt");


    qFile.calcAverageWindowFunctionFromRMat(k_A, window2, dl_reso);
    window_fname = std::string(SRCDIR) + "/tests/output/window2_k_osamp.txt";
    write2VectorsToFile(window_fname.c_str(), k_A, window2);
    for (int i = 0; i < window2.size(); i += 20)
        printf("%.5e  %.5e\n", k_A[i], window2[i]);

    r2r.resize(k_A.size(), dl_reso);
    std::transform(
        window2.begin(), window2.end(),
        r2r.field_x.begin(),
        [](double d) { return sqrt(d); });
    r2r.fftK2X();
    lambda_vec.resize(k_A.size());
    for (int i = 0; i < k_A.size(); ++i)
        lambda_vec[i] = i * dl_reso;
    window_fname = std::string(SRCDIR) + "/tests/output/window2_x_osamp.txt";
    write2VectorsToFile(window_fname.c_str(), lambda_vec, r2r.field_x);
    return 0;
}

