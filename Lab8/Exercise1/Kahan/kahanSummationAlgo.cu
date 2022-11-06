#include <bits/stdc++.h>
#define N 10
using namespace std;

// Function to implement the Kahan
// summation algorithm
__global__ void kahanSum(double *no)
{
	__shared__ double sum;

	__shared__ double c;
	sum = c = 0.0;
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
	{
		double y = no[i] - c;
		double t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	__syncthreads();
	if(i == 0) printf("Kahan sum: %lf\n", sum);
}

__global__ void init(double *no)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < N)
		no[i] = 0.1;
}

double sum(vector<double> &fa)
{
	double sum = 0.0;

	for (int i = 0; i < N; i++)
	{
		sum = sum + fa[i];
	}
	return sum;
}

int main()
{
	vector<double> no(N);
	double *d_no;
	cudaMalloc((void **)&d_no, N * sizeof(double));
	init<<<1, 10>>>(d_no);
	for (int i = 0; i < N; i++)
	{
		no[i] = 0.1;
	}

	cout << setprecision(16);
	cout << "Normal sum: " << sum(no) << "\n";
	kahanSum<<<1, 10>>>(d_no);
}