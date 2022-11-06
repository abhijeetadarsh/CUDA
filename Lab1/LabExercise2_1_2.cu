#include <stdio.h>
__global__ void printCourseDetails()
{
	printf("Course Name: GPU Computing Lab\n");
	printf("Name of Experiment: Programs -Hello world, a Kernel Call and Passing Parameters\n");
	printf("Date: 13 Aug 2022\nby GPU\n");
}
int main(){
    printf("PART 2\n");
	for(int i=0;i<4;i++){
		printf("Course Name: GPU Computing Lab\n");
		printf("Name of Experiment: Programs -Hello world, a Kernel Call and Passing Parameters\n");
		printf("Date: 13 Aug 2022\nby CPU\n");
	}
	printCourseDetails<<<1, 4>>>();
	cudaDeviceReset();
	return 0;
}
