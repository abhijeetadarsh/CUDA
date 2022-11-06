#include <stdio.h>
__global__ void printMyName()
{
	printf("Abhijeet Adarsh by GPU\n");
}

int main(){
    printf("PART 1\n");
	for(int i=0;i<10;i++) printf("Abhijeet Adarsh by CPU\n");
	printMyName<<<1, 10>>>();
	cudaDeviceReset();
	return 0;
}