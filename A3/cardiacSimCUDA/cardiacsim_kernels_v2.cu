/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 * Refer to "Detailed Numerical Analyses of the Aliev-Panfilov Model on GPGPU"
 * https://www.simula.no/publications/detailed-numerical-analyses-aliev-panfilov-model-gpgpu
 * by Xing Cai, Didem Unat and Scott Baden
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
 #include <getopt.h>
 
 using namespace std;
 
 // External functions
 extern "C" void splot(double *E, double T, int niter, int m, int n);
 
 void
 cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads);
 
 // Utilities
 //
 // Timer
 // Make successive calls and take a difference to get the elapsed time.
 static const double kMicro = 1.0e-6;
 
 double getTime() {
     struct timeval TV;
     struct timezone TZ;
 
     const int RC = gettimeofday(&TV, &TZ);
     if (RC == -1) {
         cerr << "ERROR: Bad call to gettimeofday" << endl;
         return (-1);
     }
 
     return (((double) TV.tv_sec) + kMicro * ((double) TV.tv_usec));
 
 }  // end getTime()
 
 // Reports statistics about the computation
 // These values should not vary (except to within roundoff)
 // when we use different numbers of  processes to solve the problem
 double stats(double *E, int m, int n, double *_mx) {
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j = 1; j <= m; j++) {
             for (i = 1; i <= n; i++) {
                     l2norm += E[j * (n+2) + i] * E[j * (n+2) + i];
                     if (E[j * (n+2) + i] > mx)
                             mx = E[j * (n+2) + i];
             }
     }
     *_mx = mx;
     l2norm /= (double) ((m) * (n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }
 
 __global__ void ghosts(const int n, const int m, double *E_prev) {
     int j = threadIdx.x + 1;
 
     E_prev[j * (n+2)] = E_prev[j * (n+2) + 2];
     E_prev[j * (n+2) + (n + 1)] = E_prev[j * (n + 2) + (n - 1)];
 
     E_prev[j] = E_prev[2 * (n + 2) + j];
     E_prev[(m + 1) * (n + 2) + j] = E_prev[(m - 1) * (n + 2) + j];
 }

 __global__ void pde_ode(const double a, const double kk, const double dt, const int n, const int m, double *E, double *E_prev, double *R, 
    const double epsilon, const double M1, const double M2, const double b, const double alpha) {
    
    int i = threadIdx.x + 1;
    int j = blockIdx.x + 1;
    int index = j * (n + 2) + i;
     
    E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + m + 2] + E_prev[index - (m + 2)]);
    E[index] = E[index] - dt * (kk * E[index] * (E[index] - a) * (E[index] - 1) + E[index] * R[index]);
    R[index] = R[index] + dt * (epsilon + M1 * R[index] / (E[index] + M2)) * (-R[index] - kk * E[index] * (E[index] - b - 1));
 }
 
 void simulate(double *E, double *E_prev, double *R,
               const double alpha, const int n, const int m, const double kk,
               const double dt, const double a, const double epsilon,
               const double M1, const double M2, const double b) {
     /*
      * Copy data from boundary of the computational box
      * to the padding region, set up for differencing
      * on the boundary of the computational box
      * Using mirror boundaries
      */
 
     ghosts<<<1, n>>>(n, m, E_prev); 
     pde_ode<<<m, n>>>(a, kk, dt, n, m, E, E_prev, R, epsilon, M1, M2, b, alpha);
 }
 // Define Kernels
 
 // __global__ void 
 // __device__ 
 // Main program
 int main(int argc, char **argv) {
     /*
      *  Solution arrays
      *   E is the "Excitation" variable, a voltage
      *   R is the "Recovery" variable
      *   E_prev is the Excitation variable for the previous timestep,
      *      and is used in time integration
      */
     double *E, *R, *E_prev;
 
     // Various constants - these definitions shouldn't change
     const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;
 
     double T = 1000.0;
     int m = 200, n = 200;
     int plot_freq = 0;
     int px = 1, py = 1;
     int no_comm = 0;
     int num_threads = 1;
 
     cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
     m = n;
     // Allocate contiguous memory for solution arrays
     // The computational box is defined on [1:m+1,1:n+1]
     // We pad the arrays in order to facilitate differencing on the
     // boundaries of the computation box
         E = (double *) malloc(sizeof(double) * size_t((m + 2) * (n + 2)));
     E_prev = (double *) malloc(sizeof(double) * size_t((m + 2) * (n + 2)));
     R = (double *) malloc(sizeof(double) * size_t((m + 2) * (n + 2)));
 
     int i, j;
     // Initialization
     for (j = 1; j <= m; j++)
         for (i = 1; i <= n; i++)
             E_prev[j * (n+2) + i] = R[j * (n+2) + i] = 0;
 
     for (j = 1; j <= m; j++)
         for (i = n / 2 + 1; i <= n; i++)
             E_prev[j * (n+2) + i] = 1.0;
 
     for (j = m / 2 + 1; j <= m; j++)
         for (i = 1; i <= n; i++)
             R[j * (n+2) + i] = 1.0;
 
         double dx = 1.0 / n;
 
     // For time integration, these values shouldn't change
     double rp = kk * (b + 1) * (b + 1) / 4;
     double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
     double dtr = 1 / (epsilon + ((M1 / M2) * rp));
     double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
     double alpha = d * dt / (dx * dx);
 
     cout << "Grid Size       : " << n << endl;
     cout << "Duration of Sim : " << T << endl;
     cout << "Time step dt    : " << dt << endl;
     cout << "Process geometry: " << px << " x " << py << endl;
     if (no_comm)
         cout << "Communication   : DISABLED" << endl;
 
     cout << endl;
 
     // Start the timer
     double t0 = getTime();
 
 
     // Simulated time is different from the integer timestep number
     // Simulated time
     double t = 0.0;
     // Integer timestep number
     int niter = 0;
 
     double *d_E, *d_E_prev, *d_R;
 
     cudaMalloc((void **) &d_E, sizeof(double) * (m + 2) * (n + 2));
     cudaMalloc((void **) &d_E_prev, sizeof(double) * (m + 2) * (n + 2));
     cudaMalloc((void **) &d_R, sizeof(double) * (m + 2) * (n + 2));
 
     cudaMemcpy(d_E, E, sizeof(double) * (m + 2) * (n + 2), cudaMemcpyHostToDevice);
     cudaMemcpy(d_E_prev, E_prev, sizeof(double) * (m + 2) * (n + 2), cudaMemcpyHostToDevice);
     cudaMemcpy(d_R, R, sizeof(double) * (m + 2) * (n + 2), cudaMemcpyHostToDevice);
 
     while (t < T) {
 
         t += dt;
         niter++;
 
         simulate(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
 
         //swap current E with previous E
         double *tmp = d_E;
         d_E = d_E_prev;
         d_E_prev = tmp;
 
         if (plot_freq) {
             int k = (int) (t / plot_freq);
             if ((t - k * plot_freq) < dt) {
                 splot(E, t, niter, m + 2, n + 2);
             }
         }
     }//end of while loop
 
     cudaMemcpy(E_prev, d_E_prev, sizeof(double) * (m + 2) * (n + 2), cudaMemcpyDeviceToHost);
 
     cudaFree(d_E);
     cudaFree(d_E_prev);
     cudaFree(d_R);  
 
     double time_elapsed = getTime() - t0;
 
     double Gflops = (double) (niter * (1E-9 * n * n) * 28.0) / time_elapsed;
     double BW = (double) (niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;
 
     cout << "Number of Iterations        : " << niter << endl;
     cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
     cout << "Sustained Gflops Rate       : " << Gflops << endl;
     cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;
 
     double mx;
     double l2norm = stats(E_prev, m, n, &mx);
     cout << "Max: " << mx << " L2norm: " << l2norm << endl;
 
     if (plot_freq) {
         cout << "\n\nEnter any input to close the program and the plot..." << endl;
         getchar();
     }
 
     free(E);
     free(E_prev);
     free(R);
 
     return 0;
 }
 
 void
 cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads) {
     /// Command line arguments
     // Default value of the domain sizes
     static struct option long_options[] = {
             {"n",          required_argument, 0, 'n'},
             {"px",         required_argument, 0, 'x'},
             {"py",         required_argument, 0, 'y'},
             {"tfinal",     required_argument, 0, 't'},
             {"plot",       required_argument, 0, 'p'},
             {"nocomm",     no_argument,       0, 'k'},
             {"numthreads", required_argument, 0, 'o'},
     };
     // Process command line arguments
     int ac;
     for (ac = 1; ac < argc; ac++) {
         int c;
         while ((c = getopt_long(argc, argv, "n:x:y:t:kp:o:", long_options, NULL)) != -1) {
             switch (c) {
 
                 // Size of the computational box
                 case 'n':
                     n = atoi(optarg);
                     break;
 
                     // X processor geometry
                 case 'x':
                     px = atoi(optarg);
 
                     // Y processor geometry
                 case 'y':
                     py = atoi(optarg);
 
                     // Length of simulation, in simulated time units
                 case 't':
                     T = atof(optarg);
                     break;
                     // Turn off communication
                 case 'k':
                     no_comm = 1;
                     break;
 
                     // Plot the excitation variable
                 case 'p':
                     plot_freq = atoi(optarg);
                     break;
 
                     // Plot the excitation variable
                 case 'o':
                     num_threads = atoi(optarg);
                     break;
 
                     // Error
                 default:
                     printf("Usage: a.out [-n <domain size>] [-t <final time >]\n\t [-p <plot frequency>]\n\t[-px <x processor geometry> [-py <y proc. geometry] [-k turn off communication] [-o <Number of OpenMP threads>]\n");
                     exit(-1);
             }
         }
     }
 }
 /* **********************************************************
  *  Author : Urvashi R.V. [04/06/2004]
  *      Modified by Didem Unat [03/23/18]
  *************************************************************/
 
 #include <stdio.h>
 
 /* Function to plot the 2D array
  * 'gnuplot' is instantiated via a pipe and
  * the values to be plotted are passed through, along
  * with gnuplot commands */
 
 FILE *gnu = NULL;
 
 void splot(double *U, double T, int niter, int m, int n) {
     int i, j;
     if (gnu == NULL) gnu = popen("gnuplot", "w");
 
     double mx = -1, mn = 32768;
     for (j = 0; j < m; j++)
         for (i = 0; i < n; i++) {
             if (U[j * m + i] > mx)
                 mx = U[j * m + i];
             if (U[j * m + i] < mn)
                 mn = U[j * m + i];
         }
 
     fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
     fprintf(gnu, "set size square\n");
     fprintf(gnu, "set key off\n");
     fprintf(gnu, "set pm3d map\n");
     // Various color schemes
     fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");
 
     //    fprintf(gnu,"set palette rgbformulae 22, 13, 31\n");
     //    fprintf(gnu,"set palette rgbformulae 30, 31, 32\n");
 
     fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);
     for (j = 0; j < m; j++) {
         for (i = 0; i < n; i++) {
             fprintf(gnu, "%d %d %f\n", i, j, U[i * m + j]);
         }
         fprintf(gnu, "\n");
     }
     fprintf(gnu, "e\n");
     fflush(gnu);
     return;
 }
  