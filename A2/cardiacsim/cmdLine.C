#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>
#include <string.h>
//using namespace std;

void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int& num_threads){
/// Command line arguments
 // Default value of the domain sizes
 static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"px", required_argument, 0, 'x'},
        {"py", required_argument, 0, 'y'},
        {"tfinal", required_argument, 0, 't'},
        {"plot", required_argument, 0, 'p'},
        {"nocomm", no_argument, 0, 'k'},
        {"numthreads", required_argument, 0, 'o'},
 };
    // Process command line arguments
 int ac;
 for(ac=1;ac<argc;ac++) {
    int c;
    while ((c=getopt_long(argc,argv,"n:x:y:t:kp:o:",long_options,NULL)) != -1){
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
