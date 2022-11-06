#ifndef WHATEVER_H_INCLUDED
#define WHATEVER_H_INCLUDED
#define TILE_WIDTH 2
#define widthField 8
#define precisionField 0
class matrix{
    public:
    int rows, cols;
    double *host_pointer, *device_pointer;
    matrix();
	matrix(int r, int c);
    ~matrix();
    void memAllocInBoth();
    void display();
    void sdisplay();
    void init_rand();
    void H2D();
	void D2H();
};
double cpuSecond();
__device__ void withTILE(double *d_MatA,double *d_MatB,double *d_MatC, int Width);
__device__ void withoutTILE(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int width, int cols);
#endif