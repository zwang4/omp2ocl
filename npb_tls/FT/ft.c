/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 2.3 OpenMP C versions - FT

  This benchmark is an OpenMP C version of the NPB FT code.

  The OpenMP C versions are developed by RWCP and derived from the serial
  Fortran versions in "NPB 2.3-serial" developed by NAS.

  Permission to use, copy, distribute and modify this software for any
  purpose with or without fee is hereby granted.
  This software is provided "as is" without express or implied warranty.

  Send comments on the OpenMP C versions to pdp-openmp@rwcp.or.jp

  Information on OpenMP activities at RWCP is available at:

http://pdplab.trc.rwcp.or.jp/pdperf/Omni/

Information on NAS Parallel Benchmarks 2.3 is available at:

http://www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------*/
/*--------------------------------------------------------------------

Authors: D. Bailey
W. Saphir

OpenMP C version: S. Satoh

--------------------------------------------------------------------*/

#include "npb-C.h"
/* global variables */
#include "global.h"

static double yy0_real[NX][FFTBLOCKPAD];
static double yy0_imag[NX][FFTBLOCKPAD];
static double yy1_real[NX][FFTBLOCKPAD];
static double yy1_imag[NX][FFTBLOCKPAD];
#if 0
#pragma omp threadprivate(yy0_real) __global
#pragma omp threadprivate(yy0_imag) __global
#pragma omp threadprivate(yy1_real) __global
#pragma omp threadprivate(yy1_imag) __global
#endif

#pragma omp threadprivate(yy0_real)
#pragma omp threadprivate(yy0_imag)
#pragma omp threadprivate(yy1_real)
#pragma omp threadprivate(yy1_imag)

/* function declarations */
static void evolve(double u0_real[NZ][NY][NX], double u0_imag[NZ][NY][NX], double u1_real[NZ][NY][NX], double u1_imag[NZ][NY][NX],
		int t, int indexmap[NZ][NY][NX], int d[3]);
static void compute_initial_conditions(double u0_real[NZ][NY][NX], double u0_imag[NZ][NY][NX], int d[3]);
static void ipow46(double a, int exponent, double *result);
static void setup(void);
static void compute_indexmap(int indexmap[NZ][NY][NX], int d[3]);
static void print_timers(void);
static void fft(int dir, double x1_real[NZ][NY][NX], double x1_imag[NZ][NY][NX], double x2_real[NZ][NY][NX], double x2_imag[NZ][NY][NX]);
static void cffts1(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX]);
static void cffts2(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX]);
static void cffts3(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX]);

static void fft_init (int n);
static void cfftz (int is, int m, int n, double x_real[NX][FFTBLOCKPAD], double x_imag[NX][FFTBLOCKPAD],
		double y_real[NX][FFTBLOCKPAD], double y_imag[NX][FFTBLOCKPAD]);
static void fftz2 (int is, int l, int m, int n, int ny, int ny1,
		double u_real[NX], double u_imag[NX], double x_real[NX][FFTBLOCKPAD], double x_imag[NX][FFTBLOCKPAD],
		double y_real[NX][FFTBLOCKPAD], double y_imag[NX][FFTBLOCKPAD]);
static int ilog2(int n);
static void checksum(int i, double u1_real[NZ][NY][NX], double u1_imag[NZ][NY][NX], int d[3]);
static void verify (int d1, int d2, int d3, int nt,
		boolean *verified, char *class);


#if defined(USE_POW)
#define r23 pow(0.5, 23.0)
#define r46 (r23*r23)
#define t23 pow(2.0, 23.0)
#define t46 (t23*t23)
#else
#define r23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define r46 (r23*r23)
#define t23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define t46 (t23*t23)
#endif

/*c---------------------------------------------------------------------
  c---------------------------------------------------------------------*/

double randlc (double *x, double a) {

	/*c---------------------------------------------------------------------
	  c---------------------------------------------------------------------*/

	/*c---------------------------------------------------------------------
	  c
	  c   This routine returns a uniform pseudorandom double precision number in the
	  c   range (0, 1) by using the linear congruential generator
	  c
	  c   x_{k+1} = a x_k  (mod 2^46)
	  c
	  c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
	  c   before repeating.  The argument A is the same as 'a' in the above formula,
	  c   and X is the same as x_0.  A and X must be odd double precision integers
	  c   in the range (1, 2^46).  The returned value RANDLC is normalized to be
	  c   between 0 and 1, i.e. RANDLC = 2^(-46) * x_1.  X is updated to contain
	  c   the new seed x_1, so that subsequent calls to RANDLC using the same
	  c   arguments will generate a continuous sequence.
	  c
	  c   This routine should produce the same results on any computer with at least
	  c   48 mantissa bits in double precision floating point data.  On 64 bit
	  c   systems, double precision should be disabled.
	  c
	  c   David H. Bailey     October 26, 1990
	  c
	  c---------------------------------------------------------------------*/

	double t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*c---------------------------------------------------------------------
	  c   Break A into two parts such that A = 2^23 * A1 + A2.
	  c---------------------------------------------------------------------*/
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;

	/*c---------------------------------------------------------------------
	  c   Break X into two parts such that X = 2^23 * X1 + X2, compute
	  c   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	  c   X = 2^23 * Z + A2 * X2  (mod 2^46).
	  c---------------------------------------------------------------------*/
	t1 = r23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - t23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(r23 * t1);
	z = t1 - t23 * t2;
	t3 = t23 * z + a2 * x2;
	t4 = (int)(r46 * t3);
	(*x) = t3 - t46 * t4;

	return (r46 * (*x));
}

void vranlc (int n, double *x_seed, double a, double y[]) {

	/*c---------------------------------------------------------------------
	  c---------------------------------------------------------------------*/

	/*c---------------------------------------------------------------------
	  c
	  c   This routine generates N uniform pseudorandom double precision numbers in
	  c   the range (0, 1) by using the linear congruential generator
	  c
	  c   x_{k+1} = a x_k  (mod 2^46)
	  c
	  c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
	  c   before repeating.  The argument A is the same as 'a' in the above formula,
	  c   and X is the same as x_0.  A and X must be odd double precision integers
	  c   in the range (1, 2^46).  The N results are placed in Y and are normalized
	  c   to be between 0 and 1.  X is updated to contain the new seed, so that
	  c   subsequent calls to VRANLC using the same arguments will generate a
	  c   continuous sequence.  If N is zero, only initialization is performed, and
	  c   the variables X, A and Y are ignored.
	  c
	  c   This routine is the standard version designed for scalar or RISC systems.
	  c   However, it should produce the same results on any single processor
	  c   computer with at least 48 mantissa bits in double precision floating point
	  c   data.  On 64 bit systems, double precision should be disabled.
	  c
	  c---------------------------------------------------------------------*/

	int i;
	double x,t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*c---------------------------------------------------------------------
	  c   Break A into two parts such that A = 2^23 * A1 + A2.
	  c---------------------------------------------------------------------*/
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;
	x = *x_seed;

	/*c---------------------------------------------------------------------
	  c   Generate N results.   This loop is not vectorizable.
	  c---------------------------------------------------------------------*/
	for (i = 1; i <= n; i++) {

		/*c---------------------------------------------------------------------
		  c   Break X into two parts such that X = 2^23 * X1 + X2, compute
		  c   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
		  c   X = 2^23 * Z + A2 * X2  (mod 2^46).
		  c---------------------------------------------------------------------*/
		t1 = r23 * x;
		x1 = (int)t1;
		x2 = x - t23 * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)(r23 * t1);
		z = t1 - t23 * t2;
		t3 = t23 * z + a2 * x2;
		t4 = (int)(r46 * t3);
		x = t3 - t46 * t4;
		y[i] = r46 * x;
	}
	*x_seed = x;
}
/*--------------------------------------------------------------------
  c FT benchmark
  c-------------------------------------------------------------------*/

int main(int argc, char **argv) {

	/*c-------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int i, ierr;
#pragma omp2ocl init

	/*------------------------------------------------------------------
	  c u0, u1, u2 are the main arrays in the problem. 
	  c Depending on the decomposition, these arrays will have different 
	  c dimensions. To accomodate all possibilities, we allocate them as 
	  c one-dimensional arrays and pass them to subroutines for different 
	  c views
	  c  - u0 contains the initial (transformed) initial condition
	  c  - u1 and u2 are working arrays
	  c  - indexmap maps i,j,k of u0 to the correct i^2+j^2+k^2 for the
	  c    time evolution operator. 
	  c-----------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c Large arrays are in common so that they are allocated on the
	  c heap rather than the stack. This common block is not
	  c referenced directly anywhere else. Padding is to avoid accidental 
	  c cache problems, since all array sizes are powers of two.
	  c-------------------------------------------------------------------*/
	static double u0_real[NZ][NY][NX], u0_imag[NZ][NY][NX];
	static double pad1_real[3], pad1_imag[3];
	static double u1_real[NZ][NY][NX], u1_imag[NZ][NY][NX];
	static double pad2_real[3], pad2_imag[3];
	static double u2_real[NZ][NY][NX], u2_imag[NZ][NY][NX];
	static double pad3_real[3], pad3_imag[3];
	static int indexmap[NZ][NY][NX];

	int iter;
	int nthreads = 1;
	double total_time, mflops;
	boolean verified;
	char class;

	/*--------------------------------------------------------------------
	  c Run the entire problem once to make sure all data is touched. 
	  c This reduces variable startup costs, which is important for such a 
	  c short benchmark. The other NPB 2 implementations are similar. 
	  c-------------------------------------------------------------------*/
	for (i = 0; i < T_MAX; i++) {
		timer_clear(i);
	}
	setup();
	////#pragma omp parallel
	{
		compute_indexmap(indexmap, dims[2]);
		////#pragma omp single
		{
			compute_initial_conditions(u1_real, u1_imag, dims[0]);
			fft_init (dims[0][0]);
		}
		fft(1, u1_real, u1_imag, u0_real, u0_imag);
	} /* end parallel */

	/*--------------------------------------------------------------------
	  c Start over from the beginning. Note that all operations must
	  c be timed, in contrast to other benchmarks. 
	  c-------------------------------------------------------------------*/
	for (i = 0; i < T_MAX; i++) {
		timer_clear(i);
	}

#pragma omp2ocl flush
	timer_start(T_TOTAL);
	if (TIMERS_ENABLED == TRUE) timer_start(T_SETUP);

	////#pragma omp parallel private(iter) firstprivate(niter)
	{
		compute_indexmap(indexmap, dims[2]);

		////#pragma omp single
		{
			compute_initial_conditions(u1_real, u1_imag, dims[0]);

			fft_init (dims[0][0]);
		}

		if (TIMERS_ENABLED == TRUE) {
			////#pragma omp master
			timer_stop(T_SETUP);
		}
		if (TIMERS_ENABLED == TRUE) {
			////#pragma omp master   
			timer_start(T_FFT);
		}
		fft(1, u1_real, u1_imag, u0_real, u0_imag);
		if (TIMERS_ENABLED == TRUE) {
			////#pragma omp master      
			timer_stop(T_FFT);
		}

		for (iter = 1; iter <= niter; iter++) {
			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_start(T_EVOLVE);
			}

			evolve(u0_real, u0_imag, u1_real, u1_imag, iter, indexmap, dims[0]);

			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_stop(T_EVOLVE);
			}
			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_start(T_FFT);
			}

			fft(-1, u1_real, u1_imag, u2_real, u2_imag);

			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_stop(T_FFT);
			}
			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_start(T_CHECKSUM);
			}

			checksum(iter, u2_real, u2_imag, dims[0]);

			if (TIMERS_ENABLED == TRUE) {
				////#pragma omp master      
				timer_stop(T_CHECKSUM);
			}
		}

		////#pragma omp single
		verify(NX, NY, NZ, niter, &verified, &class);

#if defined(_OPENMP)
		////#pragma omp master    
		nthreads = omp_get_num_threads();
#endif /* _OPENMP */    
	} /* end parallel */

#pragma omp2ocl sync
	timer_stop(T_TOTAL);
	total_time = timer_read(T_TOTAL);

	if( total_time != 0.0) {
		mflops = 1.0e-6*(double)(NTOTAL) *
			(14.8157+7.19641*log((double)(NTOTAL))
			 +  (5.23518+7.21113*log((double)(NTOTAL)))*niter)
			/total_time;
	} else {
		mflops = 0.0;
	}
	c_print_results("FT", class, NX, NY, NZ, niter, nthreads,
			total_time, mflops, "          floating point", verified, 
			NPBVERSION, COMPILETIME,
			CS1, CS2, CS3, CS4, CS5, CS6, CS7);
	if (TIMERS_ENABLED == TRUE) print_timers();
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void evolve(double u0_real[NZ][NY][NX], double u0_imag[NZ][NY][NX], double u1_real[NZ][NY][NX], double u1_imag[NZ][NY][NX],
		int t, int indexmap[NZ][NY][NX], int d[3]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c evolve u0 -> u1 (t time steps) in fourier space
	  c-------------------------------------------------------------------*/

	int i, j, k;
#pragma omp parallel for private(k, j, i)
	for (k = 0; k < d[2]; k++) {
		for (j = 0; j < d[1]; j++) {
			for (i = 0; i < d[0]; i++) {
				u1_real[k][j][i] = u0_real[k][j][i] * ex[t*indexmap[k][j][i]];
				u1_imag[k][j][i] = u0_imag[k][j][i] * ex[t*indexmap[k][j][i]];
			}
		}
	}
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void compute_initial_conditions(double u0_real[NZ][NY][NX], double u0_imag[NZ][NY][NX], int d[3]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c Fill in array u0 with initial conditions from 
	  c random number generator 
	  c-------------------------------------------------------------------*/

	int k;
	double x0, start, an, dummy;
	static double tmp[NX*2*MAXDIM+1];
	int i,j,t;

	start = SEED;
	/*--------------------------------------------------------------------
	  c Jump to the starting element for our first plane.
	  c-------------------------------------------------------------------*/
	ipow46(A, (zstart[0]-1)*2*NX*NY + (ystart[0]-1)*2*NX, &an);
	dummy = randlc(&start, an);
	ipow46(A, 2*NX*NY, &an);

	/*--------------------------------------------------------------------
	  c Go through by z planes filling in one square at a time.
	  c-------------------------------------------------------------------*/
	for (k = 0; k < dims[0][2]; k++) {
		x0 = start;
		vranlc(2*NX*dims[0][1], &x0, A, tmp);

		t = 1;
		for (j = 0; j < dims[0][1]; j++)
			for (i = 0; i < NX; i++) {
				u0_real[k][j][i] = tmp[t++];
				u0_imag[k][j][i] = tmp[t++];
			}

		if (k != dims[0][2]) dummy = randlc(&start, an);
	}
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void ipow46(double a, int exponent, double *result) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c compute a^exponent mod 2^46
	  c-------------------------------------------------------------------*/

	double dummy, q, r;
	int n, n2;

	/*--------------------------------------------------------------------
	  c Use
	  c   a^n = a^(n/2)*a^(n/2) if n even else
	  c   a^n = a*a^(n-1)       if n odd
	  c-------------------------------------------------------------------*/
	*result = 1;
	if (exponent == 0) return;
	q = a;
	r = 1;
	n = exponent;

	while (n > 1) {
		n2 = n/2;
		if (n2 * 2 == n) {
			dummy = randlc(&q, q);
			n = n2;
		} else {
			dummy = randlc(&r, q);
			n = n-1;
		}
	}
	dummy = randlc(&r, q);
	*result = r;
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void setup(void) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int ierr, i, j, fstatus;

	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
			" - FT Benchmark\n\n");

	niter = NITER_DEFAULT;

	printf(" Size                : %3dx%3dx%3d\n", NX, NY, NZ);
	printf(" Iterations          :     %7d\n", niter);

	/* 1004 format(' Number of processes :     ', i7)
	   1005 format(' Processor array     :     ', i3, 'x', i3)
	   1006 format(' WARNING: compiled for ', i5, ' processes. ',
	   >       ' Will not verify. ')*/

	for (i = 0;i < 3 ; i++) {
		dims[i][0] = NX;
		dims[i][1] = NY;
		dims[i][2] = NZ;
	}


	for (i = 0; i < 3; i++) {
		xstart[i] = 1;
		xend[i]   = NX;
		ystart[i] = 1;
		yend[i]   = NY;
		zstart[i] = 1;
		zend[i]   = NZ;
	}

	/*--------------------------------------------------------------------
	  c Set up info for blocking of ffts and transposes.  This improves
	  c performance on cache-based systems. Blocking involves
	  c working on a chunk of the problem at a time, taking chunks
	  c along the first, second, or third dimension. 
	  c
	  c - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
	  c - In cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

	  c Since 1st dim is always in processor, we'll assume it's long enough 
	  c (default blocking factor is 16 so min size for 1st dim is 16)
	  c The only case we have to worry about is cffts1 in a 2d decomposition. 
	  c so the blocking factor should not be larger than the 2nd dimension. 
	  c-------------------------------------------------------------------*/

	fftblock = FFTBLOCK_DEFAULT;
	fftblockpad = FFTBLOCKPAD_DEFAULT;

	if (fftblock != FFTBLOCK_DEFAULT) fftblockpad = fftblock+3;
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void compute_indexmap(int indexmap[NZ][NY][NX], int d[3]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
	  c for time evolution exponent. 
	  c-------------------------------------------------------------------*/

	int i, j, k, ii, ii2, jj, ij2, kk;
	double ap;
	int xstart_i, ystart_i, zstart_i;
	/*--------------------------------------------------------------------
	  c basically we want to convert the fortran indices 
	  c   1 2 3 4 5 6 7 8 
	  c to 
	  c   0 1 2 3 -4 -3 -2 -1
	  c The following magic formula does the trick:
	  c mod(i-1+n/2, n) - n/2
	  c-------------------------------------------------------------------*/

	xstart_i = xstart[2];
	ystart_i = ystart[2];
	zstart_i = zstart[2];

#pragma omp parallel for private(i, ii, ii2, j, jj, ij2, k, kk) schedule(static)
	for (i = 0; i < dims[2][0]; i++) {
		for (j = 0; j < dims[2][1]; j++) {
			for (k = 0; k < dims[2][2]; k++) {
				ii =  (i+1+xstart_i-2+NX/2)%NX - NX/2;
				ii2 = ii*ii;
				jj = (j+1+ystart_i-2+NY/2)%NY - NY/2;
				ij2 = jj*jj+ii2;

				kk = (k+1+zstart_i-2+NZ/2)%NZ - NZ/2;
				indexmap[k][j][i] = kk*kk+ij2;
			}
		}
	}

	/*--------------------------------------------------------------------
	  c compute array of exponentials for time evolution. 
	  c-------------------------------------------------------------------*/
	////#pragma omp single
	{
		ap = - 4.0 * ALPHA * PI * PI;

		ex[0] = 1.0;
		ex[1] = exp(ap);
		for (i = 2; i <= EXPMAX; i++) {
			ex[i] = ex[i-1]*ex[1];
		}
	} /* end single */
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void print_timers(void) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int i;
	char *tstrings[] = { "          total ",
		"          setup ", 
		"            fft ", 
		"         evolve ", 
		"       checksum ", 
		"         fftlow ", 
		"        fftcopy " };

	for (i = 0; i < T_MAX; i++) {
		if (timer_read(i) != 0.0) {
			printf("timer %2d(%16s( :%10.6f\n", i, tstrings[i], timer_read(i));
		}
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void fft(int dir, double x1_real[NZ][NY][NX], double x1_imag[NZ][NY][NX], double x2_real[NZ][NY][NX], double x2_imag[NZ][NY][NX]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	//dcomplex y0[NX][FFTBLOCKPAD];
	//dcomplex y0[NX][FFTBLOCKPAD];
	//dcomplex y1[NX][FFTBLOCKPAD];

	/*--------------------------------------------------------------------
	  c note: args x1, x2 must be different arrays
	  c note: args for cfftsx are (direction, layout, xin, xout, scratch)
	  c       xin/xout may be the same and it can be somewhat faster
	  c       if they are
	  c-------------------------------------------------------------------*/

	if (dir == 1) {
		//cffts1(1, dims[0], x1, x1, y0, y1);	/* x1 -> x1 */
		cffts1(1, dims[0], x1_real, x1_imag, x1_real, x1_imag);	/* x1 -> x1 */
		//cffts2(1, dims[1], x1, x1, y0, y1);	/* x1 -> x1 */
		cffts2(1, dims[1], x1_real, x1_imag, x1_real, x1_imag);	/* x1 -> x1 */
		//cffts3(1, dims[2], x1, x2, y0, y1);	/* x1 -> x2 */
		cffts3(1, dims[2], x1_real, x1_imag, x2_real, x2_imag);	/* x1 -> x1 */
	} else {
		//cffts3(-1, dims[2], x1, x1, y0, y1);	/* x1 -> x1 */
		cffts3(-1, dims[2], x1_real, x1_imag, x1_real, x1_imag);	/* x1 -> x1 */
		//cffts2(-1, dims[1], x1, x1, y0, y1);	/* x1 -> x1 */
		cffts2(-1, dims[1], x1_real, x1_imag, x1_real, x1_imag);	/* x1 -> x1 */
		//cffts1(-1, dims[0], x1, x2, y0, y1);	/* x1 -> x2 */
		cffts1(-1, dims[0], x1_real, x1_imag, x2_real, x2_imag);	/* x1 -> x2 */
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void cffts1(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX])
{

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int logd[3];
	int i, j, k, jj;

	for (i = 0; i < 3; i++) {
		logd[i] = ilog2(d[i]);
	}

	int logd_0 = logd[0];

#pragma omp parallel for private(k, jj, j, i) schedule(static) //mult_iterations(32,32)
	for (k = 0; k < d[2]; k++) {
		for (jj = 0; jj <= d[1] - fftblock; jj+=fftblock) {
			for (j = 0; j < fftblock; j++) {
				for (i = 0; i < d[0]; i++) {
					yy0_real[i][j] = x_real[k][j+jj][i];
					yy0_imag[i][j] = x_imag[k][j+jj][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */

			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz (is, logd_0,
					d[0], yy0_real, yy0_imag, yy1_real, yy1_imag);

			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			for (j = 0; j < fftblock; j++) {
				for (i = 0; i < d[0]; i++) {
					xout_real[k][j+jj][i] = yy0_real[i][j];
					xout_imag[k][j+jj][i] = yy0_imag[i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void cffts2(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX])
{

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int logd[3];
	int i, j, k, ii;

	for (i = 0; i < 3; i++) {
		logd[i] = ilog2(d[i]);
	}

	int logd_1 = logd[1];

#pragma omp parallel for private(k, ii, j, i) schedule(static) //mult_iterations(32,32)
	for (k = 0; k < d[2]; k++) {
		for (ii = 0; ii <= d[0] - fftblock; ii+=fftblock) {
			/*	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			for (j = 0; j < d[1]; j++) {
				for (i = 0; i < fftblock; i++) {
					yy0_real[j][i] = x_real[k][j][i+ii];
					yy0_imag[j][i] = x_imag[k][j][i+ii];
				}
			}
			/*	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz (is, logd_1, 
					d[1], yy0_real, yy0_imag, yy1_real, yy1_imag);

			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			for (j = 0; j < d[1]; j++) {
				for (i = 0; i < fftblock; i++) {
					xout_real[k][j][i+ii] = yy0_real[j][i];
					xout_imag[k][j][i+ii] = yy0_imag[j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void cffts3(int is, int d[3], double x_real[NZ][NY][NX], double x_imag[NZ][NY][NX],
		double xout_real[NZ][NY][NX], double xout_imag[NZ][NY][NX])
{

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int logd[3];
	int i, j, k, ii;

	for (i = 0;i < 3; i++) {
		logd[i] = ilog2(d[i]);
	}

	int logd_2 = logd[2];

#pragma omp parallel for private(j, ii, k, i) schedule(static) //mult_iterations(32)
	for (j = 0; j < d[1]; j++) {
		for (ii = 0; ii <= d[0] - fftblock; ii+=fftblock) {
			/*	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			for (k = 0; k < d[2]; k++) {
				for (i = 0; i < fftblock; i++) {
					yy0_real[k][i] = x_real[k][j][i+ii];
					yy0_imag[k][i] = x_imag[k][j][i+ii];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz (is, logd_2,
					d[2], yy0_real, yy0_imag, yy1_real, yy1_imag);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			for (k = 0; k < d[2]; k++) {
				for (i = 0; i < fftblock; i++) {
					xout_real[k][j][i+ii] = yy0_real[k][i];
					xout_imag[k][j][i+ii] = yy0_imag[k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void fft_init (int n) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c compute the roots-of-unity array that will be used for subsequent FFTs. 
	  c-------------------------------------------------------------------*/

	int m,nu,ku,i,j,ln;
	double t, ti;


	/*--------------------------------------------------------------------
	  c   Initialize the U array with sines and cosines in a manner that permits
	  c   stride one access at each FFT iteration.
	  c-------------------------------------------------------------------*/
	nu = n;
	m = ilog2(n);
	u_real[0] = (double)m;
	u_imag[0] = 0.0;
	ku = 1;
	ln = 1;

	for (j = 1; j <= m; j++) {
		t = PI / ln;

		for (i = 0; i <= ln - 1; i++) {
			ti = i * t;
			u_real[i+ku] = cos(ti);
			u_imag[i+ku] = sin(ti);
		}

		ku = ku + ln;
		ln = 2 * ln;
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void cfftz (int is, int m, int n, double x_real[NX][FFTBLOCKPAD], double x_imag[NX][FFTBLOCKPAD],
		double y_real[NX][FFTBLOCKPAD], double y_imag[NX][FFTBLOCKPAD]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c   Computes NY N-point complex-to-complex FFTs of X using an algorithm due
	  c   to Swarztrauber.  X is both the input and the output array, while Y is a 
	  c   scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to 
	  c   perform FFTs, the array U must be initialized by calling CFFTZ with IS 
	  c   set to 0 and M set to MX, where MX is the maximum value of M for any 
	  c   subsequent call.
	  c-------------------------------------------------------------------*/

	int i,j,l,mx;

	/*--------------------------------------------------------------------
	  c   Check if input parameters are invalid.
	  c-------------------------------------------------------------------*/
	mx = (int)(u_real[0]);

	/*--------------------------------------------------------------------
	  c   Perform one variant of the Stockham FFT.
	  c-------------------------------------------------------------------*/
	for (l = 1; l <= m; l+=2) {
		fftz2 (is, l, m, n, fftblock, fftblockpad, u_real, u_imag, x_real, x_imag, y_real, y_imag);
		if (l == m) break;
		fftz2 (is, l + 1, m, n, fftblock, fftblockpad, u_real, u_imag, y_real, y_imag, x_real, x_imag);
	}

	/*--------------------------------------------------------------------
	  c   Copy Y to X.
	  c-------------------------------------------------------------------*/
	if (m % 2 == 1) {
		for (j = 0; j < n; j++) {
			for (i = 0; i < fftblock; i++) {
				x_real[j][i] = y_real[j][i];
				x_imag[j][i] = y_imag[j][i];
			}
		}
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void fftz2 (int is, int l, int m, int n, int ny, int ny1,
		double u_real[NX], double u_imag[NX], double x_real[NX][FFTBLOCKPAD], double x_imag[NX][FFTBLOCKPAD],
		double y_real[NX][FFTBLOCKPAD], double y_imag[NX][FFTBLOCKPAD]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c   Performs the L-th iteration of the second variant of the Stockham FFT.
	  c-------------------------------------------------------------------*/

	int k,n1,li,lj,lk,ku,i,j,i11,i12,i21,i22;
	double u1_real,x11_real,x21_real;
	double u1_imag,x11_imag,x21_imag;

	/*--------------------------------------------------------------------
	  c   Set initial parameters.
	  c-------------------------------------------------------------------*/

	n1 = n / 2;
	if (l-1 == 0) {
		lk = 1;
	} else {
		lk = 2 << ((l - 1)-1);
	}
	if (m-l == 0) {
		li = 1;
	} else {
		li = 2 << ((m - l)-1);
	}
	lj = 2 * lk;
	ku = li;

	for (i = 0; i < li; i++) {

		i11 = i * lk;
		i12 = i11 + n1;
		i21 = i * lj;
		i22 = i21 + lk;
		if (is >= 1) {
			u1_real = u_real[ku+i];
			u1_imag = u_imag[ku+i];
		} else {
			u1_real =  u_real[ku+i];
			u1_imag = -u_imag[ku+i];
		}

		/*--------------------------------------------------------------------
		  c   This loop is vectorizable.
		  c-------------------------------------------------------------------*/
		for (k = 0; k < lk; k++) {
			for (j = 0; j < ny; j++) {
				double x11real, x11imag;
				double x21real, x21imag;
				x11real = x_real[i11+k][j];
				x11imag = x_imag[i11+k][j];
				x21real = x_real[i12+k][j];
				x21imag = x_imag[i12+k][j];
				y_real[i21+k][j] = x11real + x21real;
				y_imag[i21+k][j] = x11imag + x21imag;
				y_real[i22+k][j] = u1_real * (x11real - x21real)
					- u1_imag * (x11imag - x21imag);
				y_imag[i22+k][j] = u1_real * (x11imag - x21imag)
					+ u1_imag * (x11real - x21real);
			}
		}
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static int ilog2(int n) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int nn, lg;

	if (n == 1) {
		return 0;
	}
	lg = 1;
	nn = 2;
	while (nn < n) {
		nn = nn << 1;
		lg++;
	}

	return lg;
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void checksum(int i, double u1_real[NZ][NY][NX], double u1_imag[NZ][NY][NX], int d[3]) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int j, q,r,s, ierr;
	double _chk_real,_allchk_real;
	double _chk_imag,_allchk_imag;

	_chk_real = 0.0;
	_chk_imag = 0.0;

	{
		double chk_real = _chk_real;
		double chk_imag = _chk_imag;
		////#pragma omp for nowait
#pragma omp parallel for reduction(+: chk_real, chk_imag) private(j, q, r, s) schedule(static)
		for (j = 1; j <= 1024; j++) {
			q = j%NX+1;
			if (q >= xstart[0] && q <= xend[0]) {
				r = (3*j)%NY+1;
				if (r >= ystart[0] && r <= yend[0]) {
					s = (5*j)%NZ+1;
					if (s >= zstart[0] && s <= zend[0]) {
						// cadd is a macro in npb-C.h adding the real and imaginary
						// component. So, the preprocessed statement still follows the
						// reduction pattern
						//cadd(chk,chk,u1[s-zstart[0]][r-ystart[0]][q-xstart[0]]);
						chk_real = chk_real + u1_real[s-zstart[0]][r-ystart[0]][q-xstart[0]];
						chk_imag = chk_imag + u1_imag[s-zstart[0]][r-ystart[0]][q-xstart[0]];
					}
				}
			}
		}

		_chk_real = chk_real;
		_chk_imag = chk_imag;
	}
	////#pragma omp critical
	{
		sums_real[i] += _chk_real;
		sums_imag[i] += _chk_imag;
	}
	////#pragma omp barrier
	////#pragma omp single
	{    
		/* complex % real */
		sums_real[i] = sums_real[i]/(double)(NTOTAL);
		sums_imag[i] = sums_imag[i]/(double)(NTOTAL);

		printf("T = %5d     Checksum = %22.12e %22.12e\n",
				i, sums_real[i], sums_imag[i]);
	}
}


/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void verify (int d1, int d2, int d3, int nt,
		boolean *verified, char *class) {

	/*--------------------------------------------------------------------
	  c-------------------------------------------------------------------*/

	int ierr, size, i;
	double err, epsilon;

	/*--------------------------------------------------------------------
	  c   Sample size reference checksums
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c   Class S size reference checksums
	  c-------------------------------------------------------------------*/
	double vdata_real_s[6+1] = { 0.0,
		5.546087004964e+02,
		5.546385409189e+02,
		5.546148406171e+02,
		5.545423607415e+02,
		5.544255039624e+02,
		5.542683411902e+02 };
	double vdata_imag_s[6+1] = { 0.0,
		4.845363331978e+02,
		4.865304269511e+02,
		4.883910722336e+02,
		4.901273169046e+02,
		4.917475857993e+02,
		4.932597244941e+02 };
	/*--------------------------------------------------------------------
	  c   Class W size reference checksums
	  c-------------------------------------------------------------------*/
	double vdata_real_w[6+1] = { 0.0,
		5.673612178944e+02,
		5.631436885271e+02,
		5.594024089970e+02,
		5.560698047020e+02,
		5.530898991250e+02,
		5.504159734538e+02 };
	double vdata_imag_w[6+1] = { 0.0,
		5.293246849175e+02,
		5.282149986629e+02,
		5.270996558037e+02, 
		5.260027904925e+02, 
		5.249400845633e+02,
		5.239212247086e+02 };
	/*--------------------------------------------------------------------
	  c   Class A size reference checksums
	  c-------------------------------------------------------------------*/
	double vdata_real_a[6+1] = { 0.0,
		5.046735008193e+02,
		5.059412319734e+02,
		5.069376896287e+02,
		5.077892868474e+02,
		5.085233095391e+02,
		5.091487099959e+02 };
	double vdata_imag_a[6+1] = { 0.0,
		5.114047905510e+02,
		5.098809666433e+02,
		5.098144042213e+02,
		5.101336130759e+02,
		5.104914655194e+02,
		5.107917842803e+02 };
	/*--------------------------------------------------------------------
	  c   Class B size reference checksums
	  c-------------------------------------------------------------------*/
	double vdata_real_b[20+1] = { 0.0,
		5.177643571579e+02,
		5.154521291263e+02,
		5.146409228649e+02,
		5.142378756213e+02,
		5.139626667737e+02,
		5.137423460082e+02,
		5.135547056878e+02,
		5.133910925466e+02,
		5.132470705390e+02,
		5.131197729984e+02,
		5.130070319283e+02,
		5.129070537032e+02,
		5.128182883502e+02,
		5.127393733383e+02,
		5.126691062020e+02,
		5.126064276004e+02,
		5.125504076570e+02,
		5.125002331720e+02,
		5.124551951846e+02,
		5.124146770029e+02 };
	double vdata_imag_b[20+1] = { 0.0,
		5.077803458597e+02,
		5.088249431599e+02,                  
		5.096208912659e+02,
		5.101023387619e+02,                  
		5.103976610617e+02,                  
		5.105948019802e+02,                  
		5.107404165783e+02,                  
		5.108576573661e+02,                  
		5.109577278523e+02,
		5.110460304483e+02,                  
		5.111252433800e+02,                  
		5.111968077718e+02,                  
		5.112616233064e+02,                  
		5.113203605551e+02,                  
		5.113735928093e+02,                  
		5.114218460548e+02,
		5.114656139760e+02,
		5.115053595966e+02,
		5.115415130407e+02,
		5.115744692211e+02 };
	/*--------------------------------------------------------------------
	  c   Class C size reference checksums
	  c-------------------------------------------------------------------*/
	double vdata_real_c[20+1] = { 0.0,
		5.195078707457e+02,
		5.155422171134e+02,
		5.144678022222e+02,
		5.140150594328e+02,
		5.137550426810e+02,
		5.135811056728e+02,
		5.134569343165e+02,
		5.133651975661e+02,
		5.132955192805e+02,
		5.132410471738e+02,
		5.131971141679e+02,
		5.131605205716e+02,
		5.131290734194e+02,
		5.131012720314e+02,
		5.130760908195e+02,
		5.130528295923e+02,
		5.130310107773e+02,
		5.130103090133e+02,
		5.129905029333e+02,
		5.129714421109e+02 };
	double vdata_imag_c[20+1] = { 0.0,
		5.149019699238e+02,
		5.127578201997e+02,
		5.122251847514e+02,
		5.121090289018e+02,
		5.121143685824e+02,
		5.121496764568e+02,
		5.121870921893e+02,
		5.122193250322e+02,
		5.122454735794e+02,
		5.122663649603e+02,
		5.122830879827e+02,
		5.122965869718e+02,
		5.123075927445e+02,
		5.123166486553e+02,
		5.123241541685e+02,
		5.123304037599e+02,
		5.123356167976e+02,
		5.123399592211e+02,
		5.123435588985e+02,
		5.123465164008e+02 };

	epsilon = 1.0e-12;
	*verified = TRUE;
	*class = 'U';

	if (d1 == 64 &&
			d2 == 64 &&
			d3 == 64 &&
			nt == 6) {
		*class = 'S';
		for (i = 1; i <= nt; i++) {
			err = ((sums_real[i]) - vdata_real_s[i]) / vdata_real_s[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
			err = ((sums_imag[i]) - vdata_imag_s[i]) / vdata_imag_s[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
		}
	} else if (d1 == 128 &&
			d2 == 128 &&
			d3 == 32 &&
			nt == 6) {
		*class = 'W';
		for (i = 1; i <= nt; i++) {
			err = ((sums_real[i]) - vdata_real_w[i]) / vdata_real_w[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
			err = ((sums_imag[i]) - vdata_imag_w[i]) / vdata_imag_w[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
		}
	} else if (d1 == 256 &&
			d2 == 256 &&
			d3 == 128 &&
			nt == 6) {
		*class = 'A';
		for (i = 1; i <= nt; i++) {
			err = (sums_real[i] - vdata_real_a[i]) / vdata_real_a[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
			err = ((sums_imag[i]) - vdata_imag_a[i]) / vdata_imag_a[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
		}
	} else if (d1 == 512 &&
			d2 == 256 &&
			d3 == 256 &&
			nt == 20) {
		*class = 'B';
		for (i = 1; i <= nt; i++) {
			err = ((sums_real[i]) - vdata_real_b[i]) / vdata_real_b[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
			err = ((sums_imag[i]) - vdata_imag_b[i]) / vdata_imag_b[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
		}
	} else if (d1 == 512 &&
			d2 == 512 &&
			d3 == 512 &&
			nt == 20) {
		*class = 'C';
		for (i = 1; i <= nt; i++) {
			err = ((sums_real[i]) - vdata_real_c[i]) / vdata_real_c[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
			err = ((sums_imag[i]) - vdata_imag_c[i]) / vdata_imag_c[i];
			if (fabs(err) > epsilon) {
				*verified = FALSE;
				break;
			}
		}
	}

	if (*class != 'U') {
		printf("Result verification successful\n");
	} else {
		printf("Result verification failed\n");
	}
	printf("class = %1c\n", *class);
}

