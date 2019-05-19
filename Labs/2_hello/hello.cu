#include <stdio.h>
/*
//=========== PART-1 ========================
__global__ void hello()
{

}

int main(void)
{

	hello<<< 1, 1 >>>();
	cudaDeviceSynchronize();

	printf("Hello World\n");
	return 0;
}
*/

//=========== PART-2 ========================
__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__global__ void hello()
{
	//every thread prints one character
	printf("%c\n", STR[threadIdx.x % STR_LENGTH]);
}

int main(void)
{

	hello<<< 1, 16>>>();
	cudaDeviceSynchronize();

	return 0;
}

