#include <cstdio>
#include <algorithm>
#include <string>

int AA = 10, BB=15;
int *pAA = &AA;
int *pBB = &BB;

void testPointRefs(int *& a)
{
    a = pAA;
}

void copyBtoa(int *& a)
{
    std::copy(pBB, pBB+1, a);
}

int main()
{
    int *pT = pBB;
    
    printf("Initial values:\n");
    printf("pBB: %d\n", pBB);
    printf("pAA: %d\n", pAA);
    printf("pTb: %d\n", pT);

    printf("------\nCall Function:\n");
    testPointRefs(pT);
    printf("pTa: %d\n", pT);

    printf("------\nCopy B to pT:\n");
    copyBtoa(pT);
    printf("pAA: %d\n", pAA);
    printf("pTa: %d\n", pT);
    printf("AA : %d\n", AA);
    printf("*pT: %d\n", *pT);

    printf("------\nSet pT=pBB:\n");
    pT = pBB;
    printf("pAA: %d\n", pAA);
    printf("pTa: %d\n", pT);

    std::string buf = std::string("BASED") + "_error_dump_quadratic_power_estimate_detailed.dat";
    printf("Test output: %s\n", buf.c_str());

    int jj = 3;
    buf = std::string("TEMP") + "/tmp-power-" + std::to_string(jj) + ".txt.",
    printf("Test output: %s\n", buf.c_str());

    return 0;
}
