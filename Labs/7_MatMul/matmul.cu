// This example demonstrates the use of shared per-block arrays
// implement an optimized dense matrix multiplication algorithm.
// Like the shared_variables.cu example, a per-block __shared__
// array acts as a "bandwidth multiplier" by eliminating redundant
// loads issued by neighboring threads.

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#define TILE_DIM 16

// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(float *A, float *B, float *C, size_t N)
{
  // calculate the row & column index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  // create a new variable to hold the sum
  // FIX ME 
  double sum = 0; 

  // perform dot product in a loop
  for (int k=0 ; k < N ; ++k) {
  // FIX ME 
    double a = A[row * N + k];
    double b = B[k* N + col]
    //sum = A[row][k] * B[k][col];
    sum += a * b;
  }
  
  // write out this thread's result
  // FIX ME 
  //C[row][col] = sum;
  C[row * N + col] = sum;
}

// an optimized version of matrix_multiplication which eliminates redundant loads
__global__ void matrix_multiply(float *A, float *B, float *C, size_t N)
{
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x,  by = blockIdx.y;

  // allocate 2D tiles in __shared__ memory
  __shared__ float ATile[TILE_DIM][TILE_DIM]; //FIXME 
  // cudaMalloc((void**)&ATile, sizeof(float) * n * n);
  
  __shared__ float BTile[TILE_DIM][TILE_DIM]; //FIXME 
  // cudaMalloc((void**)&BTile, sizeof(float) * n * n);
  
  // calculate the row & column index of the element
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx; // FIXME

  float result = 0;

  // loop over the tiles of the input in phases
  for(int p = 0; p < N/TILE_DIM; ++p)
  {
    // collaboratively load tiles into __shared__
    ATile[ty][tx] = A[N*row + (p*TILE_DIM + tx)] ;// FIXME
    BTile[ty][tx] = B[(p*TILE_DIM + ty)*N + col] ;// FIXME

    // wait until all data is loaded before allowing
    // any thread in this block to continue
    //FIXME 
    _syncthreads();
    // do dot product between row of ATile and column of BTile
    for(int k = 0; k < TILE_DIM; ++k)
    {
      
      result += ATile[ty][k] * BTile[k][tx];//FIXME
    }
    _syncthreads();
    // wait until all threads are finished with the data
    // before allowing any thread in this block to continue

  }
  C[ty * N +tx] = result;
  // write out this thread's result
  //FIX ME
}


int main(void)
{
  // create a large workload so we can easily measure the
  // performance difference of both implementations

  // note that n measures the width of the matrix, not the number of total elements
  const size_t n = 2048;
  const dim3 block_size(TILE_DIM,TILE_DIM);
  const dim3 num_blocks(n / block_size.x, n / block_size.y);

  // generate random input on the host
  std::vector<float> h_A(n*n), h_B(n*n), h_C(n*n);
  const float valB = 0.01f;  

  for(int i = 0; i < n*n; ++i)
  {
    h_A[i] = 1.0f; //static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = valB; //static_cast<float>(rand()) / RAND_MAX;
  }

  // allocate storage for the device
  float *d_A = 0, *d_B = 0, *d_C = 0;
  cudaMalloc((void**)&d_A, sizeof(float) * n * n);
  cudaMalloc((void**)&d_B, sizeof(float) * n * n);
  cudaMalloc((void**)&d_C, sizeof(float) * n * n);

  // copy input to the device
  cudaMemcpy(d_A, &h_A[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, &h_B[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);

  // time the kernel launches using CUDA events
  cudaEvent_t launch_begin, launch_end;
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);

  // to get accurate timings, launch a single "warm-up" kernel
  matrix_multiply_simple<<<num_blocks,block_size>>>(d_A, d_B, d_C, n);

  // time many kernel launches and take the average time
  const size_t num_launches = 100;
  float average_simple_time = 0;
  std::cout << "Timing simple implementation...";
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply_simple<<<num_blocks,block_size>>>(d_A, d_B, d_C, n);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_simple_time += time;
  }
  average_simple_time /= num_launches;
  std::cout << " done." << std::endl;

  // now time the optimized kernel

  // again, launch a single "warm-up" kernel
  matrix_multiply<<<num_blocks,block_size>>>(d_A, d_B, d_C, n);

  // time many kernel launches and take the average time
  float average_optimized_time = 0;
  std::cout << "Timing optimized implementation...";
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply<<<num_blocks,block_size>>>(d_A, d_B, d_C, n);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_optimized_time += time;
  }
  average_optimized_time /= num_launches;
  std::cout << " done." << std::endl;

  // Copy result from device to host
  cudaMemcpy(&h_C[0], d_C, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

  //host matrix multiply
  //check for correctness

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero
    bool correct = true; 
    for (int i = 0; i < n*n; i++)
    {
        double abs_err = fabs(h_C[i] - (n * valB));
        double dot_length = n;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], n*valB, eps);
            correct = false;
	    break;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // report the effective throughput of each kernel in GFLOPS
  // the effective throughput is measured as the number of floating point operations performed per second:
  // (one mul + one add) * N^3

  float simple_throughput = static_cast<float>(2 * n * n * n) / (average_simple_time / 1000.0f) / 1000000000.0f;
  float optimized_throughput = static_cast<float>(2 * n * n * n) / (average_optimized_time / 1000.0f) / 1000000000.0f;

  std::cout << "Matrix size: " << n << "x" << n << std::endl;
  std::cout << "Tile size: " << TILE_DIM << "x" << TILE_DIM << std::endl;

  std::cout << "Throughput of simple kernel: " << simple_throughput << " GFLOPS" << std::endl;
  std::cout << "Throughput of optimized kernel: " << optimized_throughput << " GFLOPS" << std::endl;
  std::cout << "Performance improvement: " << optimized_throughput / simple_throughput << "x" << std::endl;
  std::cout << std::endl;

  // destroy the CUDA events
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

