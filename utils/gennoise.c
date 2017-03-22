// From Paul Bourke - paulbourke.net / 20140825
// http://paulbourke.net/fractals/noise/gennoise.c

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <sys/types.h>
#include <time.h>

#include "paulslib.h"

#define N 8192
#define TWOPOWER 13
#define TWOPI 6.283185307179586476925287

/*
	Create a noise signal using fBm
*/

int main(int argc,char **argv) 
{
	int i;
	double beta,seed;
	double mag,pha;
	double real[N],imag[N];

	if (argc < 3) {
		fprintf(stderr,"Usage: %s beta seed\n",argv[0]);
		exit(0);
	}
	if ((beta = atof(argv[1])) < 1 || beta > 3) {
		fprintf(stderr,"Beta must be between 1 and 3\n");
		exit(0);
	}
	if ((seed = atof(argv[2])) <= 0) {
		seed = time(NULL) % 30000;
		RandomInitialise(seed,seed+100);
	}

	real[0] = 0;
	imag[0] = 0;
	for (i=1;i<=N/2;i++) {
		mag = pow(i+1.0,-beta/2) * RandomGaussian(0.0,1.0); // Note to self a number of years later, why "i+1"
		pha = TWOPI * RandomUniform();
		real[i] = mag * cos(pha);
		imag[i] = mag * sin(pha);
		real[N-i] =  real[i];
		imag[N-i] = -imag[i];
	}
	imag[N/2] = 0;

	FFT(-1,TWOPOWER,real,imag);

	for (i=0;i<N;i++) 
		printf("%d %g\n",i,real[i]);
}


