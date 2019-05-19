/**
* jacobi1D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
*
* Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
* Will Killian <killian@udel.edu>
* Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
* Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
*
* 04.2019
* Modified by Burak Bastem and Didem Unat
*/

#include <stdio.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define N 1024 // 2 of them are halos, does not work with odd size
#define TSTEPS 10000
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

void init_array(int n, double* A, double* B) {
  for (int i = 0; i < n; ++i) {
    A[i] = ((double) 4 * i + 10) / n;
    B[i] = ((double) 7 * i + 11) / n;
  }
}

void runJacobi1DCpu(int tsteps, int n, double*& A, double*& B) {
  for (int t = 0; t < tsteps; ++t) {
    for (int i = 1; i < n - 1; ++i) {
      B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
    }
    double* tmp = A; A = B; B = tmp; // swap
  }
}

__global__ void runJacobiCUDA_kernel(int n, double* A, double* B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i > 0) && (i < (n-1))) {
    B[i] = 0.33333f * (A[i-1] + A[i] + A[i + 1]);
  }
}

float absVal(float a) {
  if(a < 0)	{
    return (a * -1);
  } else {
    return a;
  }
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
    return 0.0f;
  }	else	{
    return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + 0.00000001f)));
  }
}

/* DCE code. Must scan the entire live-out data.
Can be used also to check the correctness of the output. */
static void print_array(int n, double* A) {
  for (int i = 0; i < n; ++i) {
    fprintf(stderr, "%0.2lf ", A[i]);
    if (i % 20 == 0) fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void compareResults(int n, double* a, double* a_outputFromGpu, double* b, double* b_outputFromGpu) {
  int fail = 0;
  // Compare a and c
  for (int i=0; i < n; ++i) {
    if (percentDiff(a[i], a_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  for (int i=0; i < n; ++i) {
    if (percentDiff(b[i], b_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
  if (fail == 0) {
    printf("Test PASSED\n");
    printf("Done\n");
  }
  else 
    printf("Test FAILED\n");

  // print_array(n, a);
  // print_array(n, a_outputFromGpu);
  // print_array(n, b);
  // print_array(n, b_outputFromGpu);
}

void runJacobi1DCUDA(int tsteps, int n, double* A, double* B, double* A_outputFromGpu, double* B_outputFromGpu) {

  double* d_A0;
  double* d_A1;
  double* d_B0;
  double* d_B1;

  cudaSetDevice(0);
  cudaMalloc(&d_A0, (n/2+1) * sizeof(double));
  cudaMalloc(&d_B0, (n/2+1) * sizeof(double));
  cudaMemcpy(d_A0, A, (n/2+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B0, B, (n/2+1) * sizeof(double), cudaMemcpyHostToDevice);

  cudaSetDevice(1);
  //FIXME 
  cudaMalloc(&d_A1, (n/2+1) * sizeof(double));
  cudaMalloc(&d_B1, (n/2+1) * sizeof(double));
  cudaMemcpy(d_A1, A, (n/2+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B1, B, (n/2+1) * sizeof(double), cudaMemcpyHostToDevice);
  

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((unsigned int)ceil( ((float)n) / ((float)block.x) ), 1);

  for (int t = 0; t < tsteps ; ++t) {
    cudaSetDevice(0);
    runJacobiCUDA_kernel <<< grid, block >>> (n/2+1, d_A0, d_B0);

    cudaSetDevice(1);
    runJacobiCUDA_kernel <<< grid, block >>> (n/2+1, d_A1, d_B1); //FIXME 
    

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    double* tmp0 = d_A0; d_A0 = d_B0; d_B0 = tmp0; // swap
    cudaSetDevice(1);
    //FIXME
    cudaDeviceSynchronize();
    double* tmp1 = d_A1; d_A1 = d_B1; d_B1 = tmp1; // swap

    // ---> update halos directly using cudaMemcpyPeer peer to peer transfers version 1 <--- //
    //FIXME 
    cudaMemcpyPeer()
    // ---> update halos directly using cudaMemcpy transfers version 2 <--- //
    //FIXME 
    cudaMemcpy(d_A1,d_A0,(n/2+1) * sizeof(double), cudaMemcpyHostToDevice);
    // ---> update through host version 3<--- //
    //FIXME 

  }

  cudaSetDevice(0);
  cudaMemcpy(A_outputFromGpu, d_A0, sizeof(double) * n/2, cudaMemcpyDeviceToHost);
  cudaMemcpy(B_outputFromGpu, d_B0, sizeof(double) * n/2, cudaMemcpyDeviceToHost);
  cudaFree(d_A0);
  cudaFree(d_B0);

  cudaSetDevice(1);
  cudaMemcpy(&A_outputFromGpu[n/2], &d_A1[1], sizeof(double) * n/2, cudaMemcpyDeviceToHost);
  cudaMemcpy(&B_outputFromGpu[n/2], &d_B1[1], sizeof(double) * n/2, cudaMemcpyDeviceToHost);
  cudaFree(d_A1);
  cudaFree(d_B1);
}

int main(int argc, char** argv) {

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;
  double* a = new double[n];
  double* b = new double[n];
  double* a_outputFromGpu = new double[n];
  double* b_outputFromGpu = new double[n];
  init_array(n, a, b);

  runJacobi1DCUDA(tsteps, n, a, b, a_outputFromGpu, b_outputFromGpu);
  runJacobi1DCpu(tsteps, n, a, b);

  compareResults(n, a, a_outputFromGpu, b, b_outputFromGpu);

  cudaFree(a_outputFromGpu);
  cudaFree(b_outputFromGpu);
  free(a);
  free(b);
  return 0;
}
