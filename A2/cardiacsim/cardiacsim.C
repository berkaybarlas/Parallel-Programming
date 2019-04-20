/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
using namespace std;

// Utilities
// 

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            cerr << "ERROR: Bad call to gettimeofday" << endl;
            return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
   double **E;
   int nx=n, ny=m;
   E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
   assert(E);
   int j;
   for(j=0;j<ny;j++) 
     E[j] = (double*)(E+ny) + j*nx;
   return(E);
}
    
// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
 double stats(double **E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
	   l2norm += E[j][i]*E[j][i];
	   if (E[j][i] > mx)
	       mx = E[j][i];
      if (E[j][i] == 1) {
       // printf("\n found 1: %d %d \n", i , j);
      }
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }

// External functions
extern "C" {
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);

void addPadding(double** src, double** target, int m, int n) {
  int i, j;
for (j=1; j<=m; j++){
      for (i=1; i<=n; i++) {
	target[j][i] = src[j-1][i-1];
      }
    }
}

void removePadding(double** src, double** target, int m, int n) {
  int i, j;
for (j=1; j<=m; j++){
      for (i=1; i<=n; i++) {
	target[j-1][i-1] = src[j][i];
      }
    }
}

void simulate (double** E,  double** E_prev,double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b, int x_pos, int y_pos, int px, int py)
{
  int i, j; 
    /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
    
   if(x_pos == 0)
    for (j=1; j<=m; j++) 
      E_prev[j][0] = E_prev[j][2];
    if(x_pos == px-1)
    for (j=1; j<=m; j++) 
      E_prev[j][n+1] = E_prev[j][n-1];
 
    if(y_pos == 0) 
    {
    for (i=1; i<=n; i++) 
      E_prev[0][i] = E_prev[2][i];
    }
    if(y_pos == py-1)
    for (i=1; i<=n; i++) 
      E_prev[m+1][i] = E_prev[m-1][i];
    
    // Solve for the excitation, the PDE
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++) {
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);
      }
    }
    
    /* 
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);
    }
    
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));
    }
    
}

void divideOrCollectData(int collect, double** src, double** dst, int x, int y, int n, int m, int rank) 
{
double **s_dst;
double **s_src;
s_src = alloc2D(m, n);
s_dst = alloc2D(y, x);
if(rank == 0) 
{
  if(collect == 0) // divide 
    removePadding(src, s_src, m, n);
}

if(collect == 1) // collect
  removePadding(dst, s_dst, y, x);

int sizes[2]    = {m, n};         /* global size */
int subsizes[2] = {y, x};     /* local size */
int starts[2]   = {0,0};    
MPI_Datatype type, box;
MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
MPI_Type_create_resized(type, 0, x*sizeof(double), &box);
MPI_Type_commit(&box);

  /* scatter the array to all processors */
  int px = n / x; // px 
  int py = n / y; // py
  int sendcounts[px*py];
  int displs[px*py];

  if (rank == 0) 
  {
      for (int i=0; i<px*py; i++) 
        sendcounts[i] = 1;
      int disp = 0;
      for (int i=0; i<py; i++) {
          for (int j=0; j<px; j++) {
              disp = j + y * i * px;
              displs[i*px+j] = disp;
              //printf("rank: %d, disp: %d\n",rank,disp);
              //disp += 1;
          }
          //disp += px * (y - 1);
      }
  }

if(collect == 0) // divide
{
  MPI_Scatterv(s_src[0], sendcounts, displs, box, 
                s_dst[0], x*y, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  addPadding(s_dst, dst, y, x);
}
if(collect == 1) // collect
{
  MPI_Gatherv(s_dst[0], x*y, MPI_DOUBLE, 
                s_src[0], sendcounts, displs, box, 0, MPI_COMM_WORLD);
  if(rank == 0)
  addPadding(s_src, src , m ,n);              
}
}

void collectData(double** src, double* dst, int x, int y) {
  int box_size = x * y ;
  //MPI_Scatterv(src, box_size, box_size, MPI_DOUBLE, 
  //            dst, box_size, MPI_DOULBE, 0, MPI_COMM_WORLD);
   MPI_Gather(dst, box_size, MPI_DOUBLE, 
              &src[1][0], box_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   
}

// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double **E, **R, **E_prev;
  int P, rank;
  MPI_Init(&argc, &argv); /// should give process number 
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  
  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
  
  double T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 1;
  int no_comm = 0;
  int num_threads=1; 


  // For time integration, these values shouldn't change 
  double dx = 1.0/n;
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads);
  m = n;  

  //if (rank == 0 ) // to uncomment this add master if statement to divide and collect
  {
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the 
  // boundaries of the computation box
  E = alloc2D(m+2,n+2);
  E_prev = alloc2D(m+2,n+2);
  R = alloc2D(m+2,n+2);
  
  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j][i] = R[j][i] = 0;
  
  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j][i] = 1.0;
  
  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j][i] = 1.0;
  
  cout << "Grid Size       : " << n << endl; 
  cout << "Duration of Sim : " << T << endl; 
  cout << "Time step dt    : " << dt << endl; 
  cout << "Process geometry: " << px << " x " << py << endl;
  if (no_comm)
    cout << "Communication   : DISABLED" << endl;
  
  cout << endl;
  }
  // Start the timer
  double t0 = getTime();
  
 
  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;

  /// create sub array 
  double **my_E;
  double **my_E_prev;
  double **my_R;
/**
 * Missing PART !!
 * not save case 
 * and not fully dividebly 
 **/
  int x_size = n / px; 
  int y_size = m / py;
  int x_pos = rank % px;
  int y_pos = rank / px;



  printf("\nx_size: %d, y_size: %d xpos:%d ypos:%d \n", x_size, y_size,x_pos,y_pos); 
  my_E = alloc2D(y_size + 2, x_size + 2);
  my_E_prev = alloc2D(y_size + 2, x_size + 2);
  my_R = alloc2D(y_size + 2, x_size + 2);

  /// send required data to processes
  // E[3][3] = 3.2;
  // E[3][203] = 3.2;
  printf("xsize %d ysize %d, n %d m %d\n",x_size, y_size, n, m);
  divideOrCollectData(0, E, my_E, x_size, y_size, n, m, rank);
  divideOrCollectData(0, E_prev, my_E_prev, x_size, y_size, n, m, rank);
  divideOrCollectData(0, R, my_R, x_size, y_size, n, m, rank);
  
  // printf("my_E[3][3]: %f\n" , my_E[3][3]);
  //convert small data to my_E
  //add padding

  double *leftSend, *rightSend, *left, *right;
  int upperRank = rank - px;
  int bottomRank = rank + px;
  int leftRank = rank + 1;
  int rightRank = rank - 1;
  right = (double*) malloc(y_size * sizeof(double));
  left = (double*) malloc(y_size * sizeof(double));
  rightSend = (double*) malloc(y_size * sizeof(double));
  leftSend = (double*) malloc(y_size * sizeof(double));
  
  int UPPERTAG = 1;
  int BOTTOMTAG = 3; 
  int LEFTTAG = 0;
  int RIGHTTAG = 2; 

  MPI_Request reqs[8];
  MPI_Status status[8];
  while (t<T) {
    int requestCount  = 0;
    /// send ghost cells to neighbor processes 
    
    // recieve up 
    if(y_pos != 0){
      MPI_Irecv(my_E_prev[0], x_size+2, MPI_DOUBLE, upperRank, BOTTOMTAG, MPI_COMM_WORLD, &reqs[requestCount++]);
      MPI_Isend(my_E_prev[1], x_size+2, MPI_DOUBLE, upperRank, UPPERTAG, MPI_COMM_WORLD, &reqs[requestCount++]); 
    }
      
    
    // recieve down 
    if(y_pos != ((P - 1) / px) ) {
      MPI_Irecv(my_E_prev[y_size+1], x_size+2, MPI_DOUBLE, bottomRank, UPPERTAG, MPI_COMM_WORLD, &reqs[requestCount++]);
      MPI_Isend(my_E_prev[y_size], x_size+2, MPI_DOUBLE, bottomRank, BOTTOMTAG, MPI_COMM_WORLD, &reqs[requestCount++]); 
    }
    
    // left side
    if(x_pos != 0 ) {
      int i; 
      for(i = 0; i < y_size; i++) {
       leftSend[i] = my_E_prev[i+1][1]; 
      }
      MPI_Irecv(&left[0], y_size, MPI_DOUBLE, rightRank, LEFTTAG, MPI_COMM_WORLD, &reqs[requestCount++]);
      MPI_Isend(&leftSend[0], y_size, MPI_DOUBLE, rightRank, RIGHTTAG, MPI_COMM_WORLD, &reqs[requestCount++]); 
  
    }
    // right side
    if(x_pos != ((P-1) % px)) {
      int i; 
      for(i = 0; i < y_size; i++) {
       rightSend[i] = my_E_prev[i+1][x_size];
      }
      MPI_Irecv(&right[0], y_size, MPI_DOUBLE, leftRank, RIGHTTAG, MPI_COMM_WORLD, &reqs[requestCount++]);
      MPI_Isend(&rightSend[0], y_size, MPI_DOUBLE, leftRank, LEFTTAG, MPI_COMM_WORLD, &reqs[requestCount++]); 
  
    } 
    
    //
  
    MPI_Waitall(requestCount, reqs, status);
    // left side
    int i;
    if(x_pos != 0 ) {
    for(i = 0; i < y_size; i++) {
      my_E_prev[i+1][0] = left[i];
    }
    }
    // right side 
    if(x_pos != ((P-1) % px)) {
      for(i = 0; i < y_size; i++) {
        my_E_prev[i+1][x_size+1] = right[i];
      }
    }

    t += dt;
    niter++;

    simulate(my_E, my_E_prev, my_R, alpha, x_size,  y_size, kk, dt, a, epsilon, M1, M2, b, x_pos, y_pos, px, py); 
    
    //swap current E with previous E
    double **tmp = my_E; 
    my_E = my_E_prev; 
    my_E_prev = tmp;
    
    /// take all the required data to plot 
    if (plot_freq){
      int k = (int)(t/plot_freq);
      if ((t - k * plot_freq) < dt){
	//splot(E,t,niter,m+2,n+2);
      }
    }
  }//end of while loop

  divideOrCollectData(1, E_prev, my_E_prev, x_size, y_size, m, n, rank);
  if(rank == 0) { // master
    double time_elapsed = getTime() - t0;

    double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;

    cout << "Number of Iterations        : " << niter << endl;
    cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
    cout << "Sustained Gflops Rate       : " << Gflops << endl; 
    cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl; 

    double mx;
    double l2norm = stats(E_prev,m,n,&mx);
    cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

    if (plot_freq){
      cout << "\n\nEnter any input to close the program and the plot..." << endl;
      getchar();
    }
  }
  free (E);
  free (E_prev);
  free (R);
  free (my_E);
  free (my_E_prev);
  free (my_R);

  MPI_Finalize();
  return 0;
}
