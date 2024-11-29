#ifndef MATHUTILS_H
#define MATHUTILS_H

static inline double trapz(const double *y, int N, double dx=1.0) {
    double result = y[N - 1] / 2;
    for (int i = N - 2; i > 0; --i)
        result += y[i];
    result += y[0] / 2;
    return result * dx;
}


static inline
bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8) {
    double mag = std::max(fabs(a), fabs(b));
    return fabs(a - b) < (abserr + relerr * mag);
}


static inline double legendre0(double x) { return 1.0; }
static inline double legendre1(double x) { return x; }
static inline double legendre2(double x) { return (1.5 * x * x - 0.5); }
static inline double legendre3(double x) { return 2.5 * x * (x * x - 0.6); }
static inline double legendre4(double x) {
    double x2 = x * x;
    return 4.375 * x2 * x2 - 3.75 * x2 + 0.375;
}
static inline double legendre6(double x) {
    double x2 = x * x, x4 = x2 * x2, x6 = x4 * x2;
    return 14.4375 * x6 - 19.6875 * x4 + 6.5625 * x2 - 0.3125;
}

static double legendre(int l, double x) {
    switch (l) {
    case 0: return 1.0;  break;
    case 1: return x;  break;
    case 2: return legendre2(x);  break;
    case 3: return legendre3(x);  break;
    case 4: return legendre4(x);  break;
    case 6: return legendre6(x);  break;
    default:
        double il = 1.0 / l;
        return (2.0 - il) * x * legendre(l - 1, x)
               - legendre(l - 2, x) * (1.0 - il);
    }
}

// Below functions can be found in https://github.com/romeric/fastapprox
/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

static inline float fastlog2 (float x) {
    union { float f; uint32_t i; } vx = { x };
    union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;

    return y - 124.22551499f - 1.498030302f * mx.f
           - 1.72587999f / (0.3520887068f + mx.f);
}


static inline float fasterlog2 (float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 126.94269504f;
}

// End of Paul Mineiro's code.

#endif
