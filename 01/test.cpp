#include <time.h>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
using namespace std;
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
void fun()
{
    printf("fun() starts \n");
    printf("Press enter to stop fun \n");
    while(1)
    {
        if (getchar())
            break;
    }
    printf("fun() ends \n");
}

int main()
{
    int j[(int)1e4];
    clock_t t;
    t = clock();
    fun();
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    cout << "time elapsed " << time_taken << "s\n";




    double t1 = cpuSecond();
    fun();
    cout << "time elapsed " << cpuSecond() - t1 << "s\n";
}