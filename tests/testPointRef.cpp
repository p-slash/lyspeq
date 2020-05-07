#include <cstdio>
#include <algorithm>

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

    return 0;
}
