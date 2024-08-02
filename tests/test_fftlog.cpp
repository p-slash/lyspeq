#include <algorithm>
#include <chrono>

#include "mathtools/fftlog.hpp"
#include "mathtools/matrix_helper.hpp"
#include "tests/test_utils.hpp"


namespace mytime {
    class Timer
    {
        using steady_c  = std::chrono::steady_clock;
        using seconds_t = std::chrono::duration<double>;

        std::chrono::time_point<steady_c> m0;
    public:
        Timer() : m0(steady_c::now()) {};
        ~Timer() {};

        double getTime() const
        {
            return std::chrono::duration_cast<seconds_t>(steady_c::now() - m0).count();
        } 
    };

    static Timer timer;
}


int test_rotate() {
    std::vector<int> v{0, 1, 2, 3, 4, -3, -2, -1};
    std::rotate(v.begin(), v.begin() + v.size() / 2 + 1, v.end());
    for (const int &a : v)
        printf("%d ", a);
    printf("\n");
    return 0;
}


int test_fftlog() {
    constexpr int N = 128;
    constexpr double r1 = 1e-7, r2 = 1e3;
    FFTLog fht(N);

    // r^(mu + 1) * exp(-r^2/2) -> k^mu+1 exp(-k^2/2)
    fht.construct(0, r1, r2, 0, log(r1 * r2));
    for (int i = 0; i < N; ++i)
        fht.field[i] = fht.r[i] * std::exp(-fht.r[i] * fht.r[i] / 2.0);

    fht.transform();

    printf("lnkcrc: %.5e\n", fht.getLnKcRc());
    printf("i \t k \t Tr \t FHT\n");
    for (int i = 0; i < N; ++i)
        printf(
            "%d\t%.3e\t%.3e\t%.3e\n",
               i, fht.k[i], fht.k[i] * std::exp(-fht.k[i] * fht.k[i] / 2.0),
               fht.field[i]);

    return 0;
}


static inline float 
fastlog2 (float x)
{
  union { float f; uint32_t i; } vx = { x };
  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f 
           - 1.72587999f / (0.3520887068f + mx.f);
}

static inline float
fasterlog2 (float x)
{
  union { float f; uint32_t i; } vx = { x };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;
  return y - 126.94269504f;
}

static inline float
fastlog (float x)
{
  return 0.69314718f * fastlog2 (x);
}


static inline float
fasterlog (float x)
{
//  return 0.69314718f * fasterlog2 (x);

  union { float f; uint32_t i; } vx = { x };
  float y = vx.i;
  y *= 8.2629582881927490e-8f;
  return y - 87.989971088f;
}


int time_log_log2() {
    int N = 1000;  int nloop = 100000;
    float dl = 12.0 / N, result = 0.0f;
    auto input = std::make_unique<float[]>(N);
    for (int i = 0; i < N; ++i)
        input[i] = expf(-6.0f + i * dl);

    double t1 = mytime::timer.getTime(), t2 = 0;
    for (int i = 0; i < nloop; ++i) {
        for (int j = 0; j < N; ++j) {
            result += logf(expf(-6.0f + j * dl));
        }
    }
    t2 = mytime::timer.getTime();

    printf("Time spent in logf: %.4f sec\t\tres %.1f\n", 60.0 * (t2 - t1), result);

    t1 = mytime::timer.getTime();
    for (int i = 0; i < nloop; ++i) {
        for (int j = 0; j < N; ++j) {
            result += log2f(input[j]);
        }
    }

    t2 = mytime::timer.getTime();
    printf("Time spent in log2f: %.4f sec\t\tres %.1f\n", 60.0 * (t2 - t1), result);


    t1 = mytime::timer.getTime();
    for (int i = 0; i < nloop; ++i) {
        for (int j = 0; j < N; ++j) {
            result += fastlog2(input[j]);
        }
    }

    t2 = mytime::timer.getTime();
    printf("Time spent in fastlog2: %.4f sec\t\tres %.1f\n", 60.0 * (t2 - t1), result);


    t1 = mytime::timer.getTime();
    for (int i = 0; i < nloop; ++i) {
        for (int j = 0; j < N; ++j) {
            result += fasterlog2(input[j]);
        }
    }

    t2 = mytime::timer.getTime();
    printf("Time spent in fasterlog2: %.4f sec\t\tres %.1f\n", 60.0 * (t2 - t1), result);

    t1 = mytime::timer.getTime();
    for (int i = 0; i < nloop; ++i) {
        for (int j = 0; j < N; ++j) {
            result = sqrtf(j * dl + i * dl);
        }
    }

    t2 = mytime::timer.getTime();
    printf("Time spent in sqrtf: %.4f sec\n", 60.0 * (t2 - t1));

    return 0;
}


int printFastlog() {
    int N = 10;
    float dl = 12.0 / N, result;
    printf("log2\t flog2 \t ftlog2\n");
    for (int j = 0; j < N; ++j) {
        float f = expf(-6.0f + j * dl);
        printf("%.4f\t%.4f\t%.4f\n", log2(f), fastlog2(f), fastlog2(f));
    }

    return 0;
}

int main() {
    int r = 0;

    r += test_rotate();
    r += test_fftlog();
    r += time_log_log2();
    r += printFastlog();

    return r;
}
