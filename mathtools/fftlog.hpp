#ifndef FFTLOG_H
#define FFTLOG_H

#include <complex>
#include <memory>
#include <vector>

#include <fftw3.h>


class FFTLog {
    std::vector<std::complex<double>> _field, _u_m;
    int N;
    double mu, q, L, dlnr, lnkcrc;

    fftw_plan p_x2k;
    fftw_plan p_k2x;
public:
    double *field;
    std::unique_ptr<double[]> r, k;

    FFTLog(int n) : N(n), mu(0), q(0), L(0), p_x2k(nullptr), p_k2x(nullptr)
    {
        int Nimag = N / 2 + 1;
        _field.resize(Nimag);
        _u_m.resize(Nimag);
        r = std::make_unique<double[]>(N);
        k = std::make_unique<double[]>(N);

        fftw_complex *_fk = reinterpret_cast<fftw_complex*>(_field.data());
        field = reinterpret_cast<double*>(_field.data());
        p_x2k = fftw_plan_dft_r2c_1d(N, field, _fk, FFTW_MEASURE);
        p_k2x = fftw_plan_dft_c2r_1d(N, _fk, field, FFTW_MEASURE);
    };
    FFTLog(const FFTLog &rhs) = delete;
    FFTLog(FFTLog &&rhs) = delete;
    ~FFTLog() {
        fftw_destroy_plan(p_x2k);
        fftw_destroy_plan(p_k2x);
    };

    void construct(double _mu, double r1, double r2, double _q=0, double lnkr=0);
    void zero_field() { std::fill(_field.begin(), _field.end(), 0); }
    void transform();

    double getLnKcRc() const { return lnkcrc; }
    double getDLn() const { return dlnr; }
};


#endif
