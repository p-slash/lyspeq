#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>

#include "mathtools/simplesolver.hpp"


struct RootCase {
    std::string name;
    std::function<double(double)> func;
    double x1, x2, truth;
};


int test_case(const RootCase &c, double rtol=1e-6) {
    int status = -99, nevals = 0;
    auto counted = [&](double x) { ++nevals; return c.func(x); };

    double xsol = rootBrent(counted, c.x1, c.x2, &status, 10000, rtol);
    double tol = 4 * rtol * fabs(c.truth) + rtol;
    bool ok = (status == 0) && (fabs(xsol - c.truth) < tol);

    printf("%-6s %-28s x=%.10f truth=%.10f status=%d nevals=%d\n",
           ok ? "PASS" : "FAIL", c.name.c_str(), xsol, c.truth, status,
           nevals);
    return ok ? 0 : 1;
}


int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int nfails = 0;

    RootCase cases[] = {
        {"linear", [](double x) { return 2 * x - 3; }, 0, 5, 1.5},
        {"linear (reversed bracket)",
         [](double x) { return 2 * x - 3; }, 5, 0, 1.5},
        {"quadratic x^2-2",
         [](double x) { return x * x - 2; }, 0, 2, sqrt(2.0)},
        {"cubic x^3-x-2",
         [](double x) { return x * x * x - x - 2; }, 1, 2,
         1.5213797068045676},
        {"cos(x)-x",
         [](double x) { return cos(x) - x; }, 0, 1, 0.7390851332151607},
        {"exp(x)-10",
         [](double x) { return exp(x) - 10; }, 0, 5, log(10.0)},
        {"sin(x) near pi",
         [](double x) { return sin(x); }, 3, 4, M_PI},
        {"root at endpoint",
         [](double x) { return x - 1; }, 1, 3, 1.0},
        {"root at zero",
         [](double x) { return x * x * x; }, -1, 2, 0.0},
        {"large scale",
         [](double x) { return x - 1e6; }, 0, 1e7, 1e6},
        {"steep tanh",
         [](double x) { return tanh(50 * (x - 0.3)); }, -1, 1, 0.3},
    };

    for (const auto &c : cases)
        nfails += test_case(c);

    // No sign change in bracket -> status must be -1
    int status = -99;
    rootBrent([](double x) { return x * x + 1; }, -1, 1, &status);
    bool ok = status == -1;
    printf("%-6s %-28s status=%d\n", ok ? "PASS" : "FAIL",
           "no root in bracket", status);
    nfails += !ok;

    // status pointer is optional
    double x = rootBrent([](double x) { return x - 0.5; }, 0, 1);
    ok = fabs(x - 0.5) < 1e-5;
    printf("%-6s %-28s x=%.10f\n", ok ? "PASS" : "FAIL",
           "null status pointer", x);
    nfails += !ok;

    printf("%d failure(s).\n", nfails);
    return nfails != 0;
}
