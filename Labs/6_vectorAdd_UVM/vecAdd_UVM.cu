// This example demonstrates parallel floating point vector
// addition with a simple __global__ function.

#include <stdlib.h>
#include <stdio.h>


// this kernel computes the vector sum c = a + b
// each thread performs one pair-wise addition
__global__ void vector_add(const float *a,
                           const float *b,
                           float *c,
                           const size_t n)
{
  // compute the global element index this thread should process
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

  // avoid accessing out of bounds elements
  if(index < n)
  {
    // sum elements
    c[index] = a[index] + b[index];
  }
}


int main(void)
{
  // create arrays of 1M elements
  const int num_elements = 1<<20;

  // compute the size of the arrays in bytes
  const int num_bytes = num_elements * sizeof(float);

  // points to host & device arrays
  float *array_a = NULL;
  float *array_b = NULL;
  float *array_c = NULL;
  float *host_array_a   = NULL;
  float *host_array_b   = NULL;
  float *host_array_c   = NULL;

  // // malloc the host arrays
  // host_array_a = (float*)malloc(num_bytes);
  // host_array_b = (float*)malloc(num_bytes);
  // host_array_c = (float*)malloc(num_bytes);

  // cudaMalloc the device arrays
  cudaMallocManaged((void**)&array_a, num_bytes);
  cudaMallocManaged((void**)&array_b, num_bytes);
  cudaMallocManaged((void**)&array_c, num_bytes);

  // if any memory allocation failed, report an error message
  if(
     array_a == 0 || array_b == 0 || array_c == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // initialize host_array_a & host_array_b
  for(int i = 0; i < num_elements; ++i)
  {
    // make array a a linear ramp
    array_a[i] = (float)i;

    // make array b random
    array_b[i] = (float)rand() / RAND_MAX;
  }

  // // copy arrays a & b to the device memory space
  // cudaMemcpy(device_array_a, host_array_a, num_bytes, cudaMemcpyHostToDevice);
  // cudaMemcpy(device_array_b, host_array_b, num_bytes, cudaMemcpyHostToDevice);

  // compute c = a + b on the device
  const size_t nThreads = 256;
  size_t nBlocks = num_elements / nThreads;

  // deal with a possible partial final block
  if(num_elements % nThreads) ++nBlocks;

  // launch the kernel
  vector_add<<<nBlocks, nThreads>>>(array_a, array_b, array_c, num_elements);

  // copy the result back to the host memory space
  // cudaMemcpy(host_array_c, device_array_c, num_bytes, cudaMemcpyDeviceToHost);

  // print out the first 10 results
  for(int i = 0; i < 10; ++i)
  {
    printf("result %d: %1.1f + %7.1f = %7.1f\n", i, array_a[i], array_b[i], array_c[i]);
  }

  // // deallocate memory
  // free(host_array_a);
  // free(host_array_b);
  // free(host_array_c);

  cudaFree(array_a);
  cudaFree(array_b);
  cudaFree(array_c);

  return 0;
}

