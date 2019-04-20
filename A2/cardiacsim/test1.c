#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>

#define N 3000

int main(int argc, char *argv[])
{
  int P, myrank, M, myrow, mycol;
  int i, j, sqrtP;
  double t_start, t_end;
  double **my_A, *my_x, *my_y, *x, *y;
  MPI_Comm  rowcomm, colcomm; 

  /* Initializations */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  sqrtP = sqrt(P); // Assuming P is a perfect square 
  M = N / P; // Assuming N is a multiple of P
  bigM = N / sqrtP; 
  my_x = (double*) malloc(M * sizeof(double));
  my_y = (double*) malloc(M * sizeof(double));
  x = (double*) malloc(bigM * sizeof(double));
  y = (double*) malloc(bigM * sizeof(double));
  
  A = (double**) malloc(bigM * sizeof(double*));

  for (i = 0; i < bigM; ++i)
    A[i] = (double*) malloc(bigM * sizeof(double));

  //initialize local arrays (e.g. from a file or randomly)
  //...
  
  /* setup communication groups */ 
  myrow = myrank / sqrtP; 
  mycol = myrank % sqrtP; 
  
  MPI_Comm_split(MPI_COMM_WORLD, myrow, myrank, &rowcomm); //
  MPI_Comm_split(MPI_COMM_WORLD, mycol, myrank, &colcomm); //
      
  /* collect x vectors along column comms, y vector along row comms */ 
  MPI_Allgather(my_x, M, MPI_DOUBLE, x, bigM, MPI_DOUBLE, colcomm);  
  MPI_Allgather(my_y, M, MPI_DOUBLE, y, bigM, MPI_DOUBLE, rowcomm); 
  
  /* local computations */ 
  for (i = 0; i < bigM; ++i) 
    for (j = 0; j < bigM; ++j) 
      y[i] += A[i][j] * x[j]; 
  
/* collect partial results along rows */ 
  for (i = 0; i < sqrtP; ++i) 
    MPI_Reduce(&(y[M*i]), my_y, M, MPI_DOUBLE, MPI_SUM, i, rowcomm); 

  MPI_Finalize();
  return 0;
}