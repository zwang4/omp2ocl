/*
   --------------------------------------------------------------------

   NAS Parallel Benchmarks 2.3 OpenMP C versions - FT

   This benchmark is an OpenMP C version of the NPB FT code.

   The OpenMP C versions are developed by RWCP and derived from the serial
   Fortran versions in "NPB 2.3-serial" developed by NAS.

   Permission to use, copy, distribute and modify this software for any
   purpose with or without fee is hereby granted.
   This software is provided "as is" without express or implied warranty.

   Send comments on the OpenMP C versions to pdp-openmp@rwcp.or.jp

   Information on OpenMP activities at RWCP is available at:

http:pdplab.trc.rwcp.or.jppdperf/Omni/

Information on NAS Parallel Benchmarks 2.3 is available at:

http:www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------
 */
/*
   --------------------------------------------------------------------

Authors: D. Bailey
W. Saphir

OpenMP C version: S. Satoh

--------------------------------------------------------------------
 */
#include "npb-C.h"
/* global variables */
#include "global.h"

#ifndef __O2G_HEADER__ 

#define __O2G_HEADER__ 

/******************************************/
/* Added codes for OpenMP2GPU translation */
/******************************************/
#include <cutil.h>
#include <math.h>
#define MAX(a,b) (((a) > (b)) ? (a) : (b))


static int gpuNumThreads = BLOCK_SIZE;
static int gpuNumBlocks;
static int gpuNumBlocks1;
static int gpuNumBlocks2;
static int totalNumThreads;
unsigned int gpuGmemSize = 0;
unsigned int gpuSmemSize = 0;
static unsigned int gpuBytes = 0;

#endif 
/* End of __O2G_HEADER__ */



int * gpu__fftblock;
int * gpu__fftblockpad;
double * gpu__u_imag;
double * gpu__u_real;
double * gpu__u1_imag__main;
double * gpu__u1_real__main;
double * gpu__u0_imag__main;
double * gpu__u0_real__main;
double * gpu__u2_imag__main;
double * gpu__u2_real__main;
int * gpu__xend;
int * gpu__xstart;
int * gpu__yend;
int * gpu__ystart;
int * gpu__zend;
int * gpu__zstart;
int * gpu__dims;
size_t pitch__dims;
int * gpu__indexmap__main;
double * gpu__ex;
static double tmp__compute_initial_conditions[(((512*2)*512)+1)];
static double yy0_real[512][18];
static double yy0_imag[512][18];
static double yy1_real[512][18];
static double yy1_imag[512][18];
#pragma omp threadprivate(yy0_real)
#pragma omp threadprivate(yy0_imag)
#pragma omp threadprivate(yy1_real)
#pragma omp threadprivate(yy1_imag)
/* function declarations */
static void evolve_cloned0(double u0_real[256][256][512], double u0_imag[256][256][512], double u1_real[256][256][512], double u1_imag[256][256][512], int t, int indexmap[256][256][512], int d[3]);
static void compute_initial_conditions(double u0_real[256][256][512], double u0_imag[256][256][512], int d[3]);
static void ipow46(double a, int exponent, double * result);
static void setup(void );
static void compute_indexmap(int indexmap[256][256][512], int d[3]);
static void compute_indexmap_clnd1(int indexmap[256][256][512], int d[3]);
static void print_timers(void );
static void fft(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512]);
static void fft_clnd2_cloned0(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512]);
static void fft_clnd1(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512]);
static void cffts1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts1_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts1_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts1_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts1_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts1_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts2_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void cffts3_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18]);
static void fft_init(int n);
__device__ static void dev_cfftz(int is, int m, int n, double x_real[][512][18], double x_imag[][512][18], double y_real[][512][18], double y_imag[][512][18], int * fftblock, int * fftblockpad, double u_imag[512], double u_real[512], int _gtid);
__device__ static void dev_fftz2(int is, int l, int m, int n, int ny, int ny1, double u_real[512], double u_imag[512], double x_real[][512][18], double x_imag[][512][18], double y_real[][512][18], double y_imag[][512][18], int _gtid);
static int ilog2(int n);
static void checksum(int i, double u1_real[256][256][512], double u1_imag[256][256][512], int d[3]);
static void verify(int d1, int d2, int d3, int nt, int * verified, char * cclass);
/*  */
/*          E  L  A  P  S  E  D  _  T  I  M  E */
/*  */
double elapsed_time(void )
{
	double t;
	wtime(( & t));
	return t;
}

double start[64];
double elapsed[64];
/*  */
/*             T  I  M  E  R  _  C  L  E  A  R */
/*  */
void timer_clear(int n)
{
	elapsed[n]=0.0;
	return ;
}

/*  */
/*             T  I  M  E  R  _  S  T  A  R  T */
/*  */
void timer_start(int n)
{
	start[n]=elapsed_time();
	return ;
}

/*  */
/*             T  I  M  E  R  _  S  T  O  P */
/*  */
void timer_stop(int n)
{
	double t;
	double now;
	now=elapsed_time();
	t=(now-start[n]);
	elapsed[n]+=t;
	return ;
}

/*  */
/*             T  I  M  E  R  _  R  E  A  D */
/*  */
double timer_read(int n)
{
	double _ret_val_0;
	_ret_val_0=elapsed[n];
	return _ret_val_0;
}

static void c_print_results(char * name, char cccccclass, int n1, int n2, int n3, int niter, int nthreads, double t, double mops, char * optype, int passed_verification, char * npbversion, char * compiletime, char * cc, char * clink, char * c_lib, char * c_inc, char * cflags, char * clinkflags, char * rand)
{
	printf("\n\n %s Benchmark Completed\n", name);
	printf(" Class           =                        %c\n", cccccclass);
	/* as in IS */
	if (((n2==0)&&(n3==0)))
	{
		printf(" Size            =             %12d\n", n1);
	}
	else
	{
		printf(" Size            =              %3dx%3dx%3d\n", n1, n2, n3);
	}
	printf(" Iterations      =             %12d\n", niter);
	printf(" Threads         =             %12d\n", nthreads);
	printf(" Time in seconds =             %12.2f\n", t);
	printf(" Mop/s total     =             %12.2f\n", mops);
	printf(" Operation type  = %24s\n", optype);
	if (passed_verification)
	{
		printf(" Verification    =               SUCCESSFUL\n");
	}
	else
	{
		printf(" Verification    =             UNSUCCESSFUL\n");
	}
	printf(" Version         =             %12s\n", npbversion);
	printf(" Compile date    =             %12s\n", compiletime);
	printf("\n Compile options:\n");
	printf("    CC           = %s\n", cc);
	printf("    CLINK        = %s\n", clink);
	printf("    C_LIB        = %s\n", c_lib);
	printf("    C_INC        = %s\n", c_inc);
	printf("    CFLAGS       = %s\n", cflags);
	printf("    CLINKFLAGS   = %s\n", clinkflags);
	printf("    RAND         = %s\n", rand);
	/*
	   printf( "\n\n" );
	   printf( " Please send the results of this run to:\n\n" );
	   printf( " NPB Development Team\n" );
	   printf( " Internet: npb@nas.nasa.gov\n \n" );
	   printf( " If email is not available, send this to:\n\n" );
	   printf( " MS T27A-1\n" );
	   printf( " NASA Ames Research Center\n" );
	   printf( " Moffett Field, CA  94035-1000\n\n" );
	   printf( " Fax: 415-604-3957\n\n" );
	 */
	return ;
}

/*
   c---------------------------------------------------------------------
   c---------------------------------------------------------------------
 */
double randlc(double * x, double a)
{
	/*
	   c---------------------------------------------------------------------
	   c---------------------------------------------------------------------
	 */
	/*
	   c---------------------------------------------------------------------
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
	   c   between 0 and 1, i.e. RANDLC = 2^(-46) x_1.  X is updated to contain
	   c   the new seed x_1, so that subsequent calls to RANDLC using the same
	   c   arguments will generate a continuous sequence.
	   c
	   c   This routine should produce the same results on any computer with at least
	   c   48 mantissa bits in double precision floating point data.  On 64 bit
	   c   systems, double precision should be disabled.
	   c
	   c   David H. Bailey     October 26, 1990
	   c
	   c---------------------------------------------------------------------
	 */
	double t1;
	double t2;
	double t3;
	double t4;
	double a1;
	double a2;
	double x1;
	double x2;
	double z;
	/*
	   c---------------------------------------------------------------------
	   c   Break A into two parts such that A = 2^23 A1 + A2.
	   c---------------------------------------------------------------------
	 */
	double _ret_val_0;
	t1=(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*a);
	a1=((int)t1);
	a2=(a-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*a1));
	/*
	   c---------------------------------------------------------------------
	   c   Break X into two parts such that X = 2^23 X1 + X2, compute
	   c   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	   c   X = 2^23 * Z + A2 * X2  (mod 2^46).
	   c---------------------------------------------------------------------
	 */
	t1=(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*( * x));
	x1=((int)t1);
	x2=(( * x)-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*x1));
	t1=((a1*x2)+(a2*x1));
	t2=((int)(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*t1));
	z=(t1-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*t2));
	t3=((((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*z)+(a2*x2));
	t4=((int)((((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5))*t3));
	( * x)=(t3-((((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0))*t4));
	_ret_val_0=((((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5))*( * x));
	return _ret_val_0;
}

/*
   c---------------------------------------------------------------------
   c---------------------------------------------------------------------
 */
void vranlc(int n, double * x_seed, double a, double y[])
{
	/*
	   c---------------------------------------------------------------------
	   c---------------------------------------------------------------------
	 */
	/*
	   c---------------------------------------------------------------------
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
	   c---------------------------------------------------------------------
	 */
	int i;
	double x;
	double t1;
	double t2;
	double t3;
	double t4;
	double a1;
	double a2;
	double x1;
	double x2;
	double z;
	/*
	   c---------------------------------------------------------------------
	   c   Break A into two parts such that A = 2^23 A1 + A2.
	   c---------------------------------------------------------------------
	 */
	t1=(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*a);
	a1=((int)t1);
	a2=(a-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*a1));
	x=( * x_seed);
	/*
	   c---------------------------------------------------------------------
	   c   Generate N results.   This loop is not vectorizable.
	   c---------------------------------------------------------------------
	 */
#pragma loop name vranlc#0 
	for (i=1; i<=n; i ++ )
	{
		/*
		   c---------------------------------------------------------------------
		   c   Break X into two parts such that X = 2^23 X1 + X2, compute
		   c   Z = A1 * X2 + A2 * X1  (mod 2^23), and then
		   c   X = 2^23 * Z + A2 * X2  (mod 2^46).
		   c---------------------------------------------------------------------
		 */
		t1=(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*x);
		x1=((int)t1);
		x2=(x-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*x1));
		t1=((a1*x2)+(a2*x1));
		t2=((int)(((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*t1));
		z=(t1-(((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*t2));
		t3=((((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*z)+(a2*x2));
		t4=((int)((((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5))*t3));
		x=(t3-((((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*((((((((((((((((((((((2.0*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0)*2.0))*t4));
		y[i]=((((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5))*x);
	}
	( * x_seed)=x;
	return ;
}

int main(int argc, char *  * argv)
{
	/*
	   c-------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int i;
	/*
	   ------------------------------------------------------------------
	   c u0, u1, u2 are the main arrays in the problem. 
	   c Depending on the decomposition, these arrays will have different 
	   c dimensions. To accomodate all possibilities, we allocate them as 
	   c one-dimensional arrays and pass them to subroutines for different 
	   c views
	   c  - u0 contains the initial (transformed) initial condition
	   c  - u1 and u2 are working arrays
	   c  - indexmap maps i,j,k of u0 to the correct i^2+j^2+k^2 for the
	   c    time evolution operator. 
	   c-----------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c Large arrays are in common so that they are allocated on the
	   c heap rather than the stack. This common block is not
	   c referenced directly anywhere else. Padding is to avoid accidental 
	   c cache problems, since all array sizes are powers of two.
	   c-------------------------------------------------------------------
	 */
	static double u0_real[256][256][512];
	static double u0_imag[256][256][512];
	static double u1_real[256][256][512];
	static double u1_imag[256][256][512];
	static double u2_real[256][256][512];
	static double u2_imag[256][256][512];
	static int indexmap[256][256][512];
	int iter;
	int nthreads = 1;
	double total_time;
	double mflops;
	int verified;
	char cclass;
	/*
	   --------------------------------------------------------------------
	   c Run the entire problem once to make sure all data is touched. 
	   c This reduces variable startup costs, which is important for such a 
	   c short benchmark. The other NPB 2 implementations are similar. 
	   c-------------------------------------------------------------------
	 */
	int _ret_val_0;

	////////////////////////////////
	// CUDA Device Initialization //
	////////////////////////////////
	int deviceCount;
	CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "cutil error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	int dev = 0;
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));
	fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
	CUDA_SAFE_CALL(cudaSetDevice(dev));


	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__fftblock)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__fftblockpad)), gpuBytes));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u_imag)), gpuBytes));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u_real)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u1_imag__main)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u1_real__main)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u0_imag__main)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u0_real__main)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u2_imag__main)), gpuBytes));
	gpuBytes=(((256*256)*512)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u2_real__main)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xend)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xstart)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yend)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ystart)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zend)), gpuBytes));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zstart)), gpuBytes));
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__dims)), ( & pitch__dims), (3*sizeof (int)), 3));
	gpuBytes=(((256*256)*512)*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__indexmap__main)), gpuBytes));
	gpuBytes=(((20*((((512*512)/4)+((256*256)/4))+((256*256)/4)))+1)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ex)), gpuBytes));
#pragma loop name main#0 
	for (i=0; i<7; i ++ )
	{
		timer_clear(i);
	}
	setup();
	/* #pragma omp parallel */
	{
		compute_indexmap(indexmap, dims[2]);
		/* #pragma omp single */
		{
			compute_initial_conditions(u1_real, u1_imag, dims[0]);
			fft_init(dims[0][0]);
		}
		fft(1, u1_real, u1_imag, u0_real, u0_imag);
	}
	/* end parallel */
	/*
	   --------------------------------------------------------------------
	   c Start over from the beginning. Note that all operations must
	   c be timed, in contrast to other benchmarks. 
	   c-------------------------------------------------------------------
	 */
#pragma loop name main#1 
	for (i=0; i<7; i ++ )
	{
		timer_clear(i);
	}
	timer_start(0);
	if ((0==1))
	{
		timer_start(1);
	}
	/* #pragma omp parallel private(iter) firstprivate(niter) */
	{
		compute_indexmap_clnd1(indexmap, dims[2]);
		/* #pragma omp single */
		{
			compute_initial_conditions(u1_real, u1_imag, dims[0]);
			fft_init(dims[0][0]);
		}
		if ((0==1))
		{
			/* #pragma omp master */
			timer_stop(1);
		}
		if ((0==1))
		{
			/* #pragma omp master    */
			timer_start(2);
		}
		fft_clnd1(1, u1_real, u1_imag, u0_real, u0_imag);
		if ((0==1))
		{
			/* #pragma omp master       */
			timer_stop(2);
		}
#pragma loop name main#2 
		for (iter=1; iter<=niter; iter ++ )
		{
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_start(3);
			}
			evolve_cloned0(u0_real, u0_imag, u1_real, u1_imag, iter, indexmap, dims[0]);
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_stop(3);
			}
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_start(2);
			}
			fft_clnd2_cloned0(( - 1), u1_real, u1_imag, u2_real, u2_imag);
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_stop(2);
			}
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_start(4);
			}
			checksum(iter, u2_real, u2_imag, dims[0]);
			if ((0==1))
			{
				/* #pragma omp master       */
				timer_stop(4);
			}
		}
		/* #pragma omp single */
		verify(512, 256, 256, niter, ( & verified), ( & cclass));
	}
	/* end parallel */
	timer_stop(0);
	total_time=timer_read(0);
	if ((total_time!=0.0))
	{
		mflops=(((1.0E-6*((double)33554432))*((14.8157+(7.19641*log(((double)33554432))))+((5.23518+(7.21113*log(((double)33554432))))*niter)))/total_time);
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("FT", cclass, 512, 256, 256, niter, nthreads, total_time, mflops, "          floating point", verified, "2.3", "20 Feb 2012", "gcc", "gcc", "-lm", "-I../common", "-O3 ", "(none)", "randdp");
	if ((0==1))
	{
		print_timers();
	}
	printf("/***********************/ \n/* Input Configuration */ \n/***********************/ \n");
	printf("====> GPU Block Size: 1024 \n");
	printf("/**********************/ \n/* Used Optimizations */ \n/**********************/ \n");
	printf("====> MallocPitch Opt is used.\n");
	printf("====> MatrixTranspose Opt is used.\n");
	printf("====> ParallelLoopSwap Opt is used.\n");
	printf("====> LoopCollapse Opt is used.\n");
	printf("====> Unrolling-on-reduction Opt is used.\n");
	printf("====> Allocate GPU variables as global ones.\n");
	printf("====> Optimize globally allocated GPU variables .\n");
	printf("====> CPU-GPU Mem Transfer Opt Level: 4\n");
	printf("====> Cuda Malloc Opt Level: 1\n");
	printf("====> Assume that all loops have non-zero iterations.\n");
	printf("====> Cache shared scalar variables onto GPU registers.\n");
	printf("====> Cache shared array elements onto GPU registers.\n");
	printf("====> Cache private array variables onto GPU shared memory.\n");
	printf("====> local array reduction variable configuration = 1\n");
	CUDA_SAFE_CALL(cudaFree(gpu__fftblock));
	CUDA_SAFE_CALL(cudaFree(gpu__fftblockpad));
	CUDA_SAFE_CALL(cudaFree(gpu__u_imag));
	CUDA_SAFE_CALL(cudaFree(gpu__u_real));
	CUDA_SAFE_CALL(cudaFree(gpu__u1_imag__main));
	CUDA_SAFE_CALL(cudaFree(gpu__u1_real__main));
	CUDA_SAFE_CALL(cudaFree(gpu__u0_imag__main));
	CUDA_SAFE_CALL(cudaFree(gpu__u0_real__main));
	CUDA_SAFE_CALL(cudaFree(gpu__u2_imag__main));
	CUDA_SAFE_CALL(cudaFree(gpu__u2_real__main));
	CUDA_SAFE_CALL(cudaFree(gpu__xend));
	CUDA_SAFE_CALL(cudaFree(gpu__xstart));
	CUDA_SAFE_CALL(cudaFree(gpu__yend));
	CUDA_SAFE_CALL(cudaFree(gpu__ystart));
	CUDA_SAFE_CALL(cudaFree(gpu__zend));
	CUDA_SAFE_CALL(cudaFree(gpu__zstart));
	CUDA_SAFE_CALL(cudaFree(gpu__dims));
	CUDA_SAFE_CALL(cudaFree(gpu__indexmap__main));
	CUDA_SAFE_CALL(cudaFree(gpu__ex));
	fflush(stdout);
	fflush(stderr);
	return _ret_val_0;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void evolve_cloned0_kernel0(int * d, double * ex, int indexmap[256][256][512], int * t, double u0_imag[256][256][512], double u0_real[256][256][512], double u1_imag[256][256][512], double u1_real[256][256][512])
{
	double ex_0;
	int t_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=_gtid;
	t_0=( * t);
	if (i<d[0])
	{
#pragma loop name evolve#0#0 
		for (j=0; j<d[1]; j ++ )
		{
#pragma loop name evolve#0#0#0 
			for (k=0; k<d[2]; k ++ )
			{
				ex_0=ex[(t_0*indexmap[k][j][i])];
				u1_real[k][j][i]=(u0_real[k][j][i]*ex_0);
				u1_imag[k][j][i]=(u0_imag[k][j][i]*ex_0);
			}
		}
	}
}

static void evolve_cloned0(double u0_real[256][256][512], double u0_imag[256][256][512], double u1_real[256][256][512], double u1_imag[256][256][512], int t, int indexmap[256][256][512], int d[3])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c evolve u0 -> u1 (t time steps) in fourier space
	   c-------------------------------------------------------------------
	 */
	int * gpu__d;
	int * gpu__t;
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[0])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(((20*((((512*512)/4)+((256*256)/4))+((256*256)/4)))+1)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__ex, ex, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__t)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__t, ( & t), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(d, ex, indexmap, t, u0_imag, u0_real, u1_imag, u1_real) private(i, j, k)
#pragma cuda gpurun noc2gmemtr(indexmap, u0_imag, u0_real, u1_imag, u1_real) 
#pragma cuda gpurun nocudamalloc(indexmap, u0_imag, u0_real, u1_imag, u1_real) 
#pragma cuda gpurun nocudafree(ex, indexmap, u0_imag, u0_real, u1_imag, u1_real) 
#pragma cuda gpurun multisrccg(ex) 
#pragma cuda gpurun nog2cmemtr(d, ex, indexmap, t, u0_imag, u0_real, u1_imag, u1_real) 
#pragma cuda ainfo kernelid(0) procname(evolve_cloned0) 
#pragma cuda gpurun registerRO(ex[(t*indexmap[k][j][i])], t) 
#pragma cuda gpurun cudafree(d, t) 
	evolve_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__ex, ((int (*)[256][512])gpu__indexmap__main), gpu__t, ((double (*)[256][512])gpu__u0_imag__main), ((double (*)[256][512])gpu__u0_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__t));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void compute_initial_conditions(double u0_real[256][256][512], double u0_imag[256][256][512], int d[3])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c Fill in array u0 with initial conditions from 
	   c random number generator 
	   c-------------------------------------------------------------------
	 */
	int k;
	double x0;
	double start;
	double an;
	double dummy;
	int i;
	int j;
	int t;
	start=3.14159265E8;
	/*
	   --------------------------------------------------------------------
	   c Jump to the starting element for our first plane.
	   c-------------------------------------------------------------------
	 */
	ipow46(1.220703125E9, (((((zstart[0]-1)*2)*512)*256)+(((ystart[0]-1)*2)*512)), ( & an));
	dummy=randlc(( & start), an);
	ipow46(1.220703125E9, ((2*512)*256), ( & an));
	/*
	   --------------------------------------------------------------------
	   c Go through by z planes filling in one square at a time.
	   c-------------------------------------------------------------------
	 */
#pragma loop name compute_initial_conditions#0 
	for (k=0; k<dims[0][2]; k ++ )
	{
		x0=start;
		vranlc(((2*512)*dims[0][1]), ( & x0), 1.220703125E9, tmp__compute_initial_conditions);
		t=1;
#pragma loop name compute_initial_conditions#0#0 
		for (j=0; j<dims[0][1]; j ++ )
		{
#pragma loop name compute_initial_conditions#0#0#0 
			for (i=0; i<512; i ++ )
			{
				u0_real[k][j][i]=tmp__compute_initial_conditions[(t ++ )];
				u0_imag[k][j][i]=tmp__compute_initial_conditions[(t ++ )];
			}
		}
		if ((k!=dims[0][2]))
		{
			dummy=randlc(( & start), an);
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void ipow46(double a, int exponent, double * result)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c compute a^exponent mod 2^46
	   c-------------------------------------------------------------------
	 */
	double dummy;
	double q;
	double r;
	int n;
	int n2;
	/*
	   --------------------------------------------------------------------
	   c Use
	   c   a^n = a^(n2)*a^(n/2) if n even else
	   c   a^n = a*a^(n-1)       if n odd
	   c-------------------------------------------------------------------
	 */
	( * result)=1;
	if ((exponent==0))
	{
		return ;
	}
	q=a;
	r=1;
	n=exponent;
	while (n>1)
	{
		n2=(n/2);
		if (((n2*2)==n))
		{
			dummy=randlc(( & q), q);
			n=n2;
		}
		else
		{
			dummy=randlc(( & r), q);
			n=(n-1);
		}
	}
	dummy=randlc(( & r), q);
	( * result)=r;
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void setup(void )
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int i;
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - FT Benchmark\n\n");
	niter=20;
	printf(" Size                : %3dx%3dx%3d\n", 512, 256, 256);
	printf(" Iterations          :     %7d\n", niter);
	/*
	   1004 format(' Number of processes :     ', i7)
	   1005 format(' Processor array     :     ', i3, 'x', i3)
	   1006 format(' WARNING: compiled for ', i5, ' processes. ',
	   >       ' Will not verify. ')
	 */
#pragma loop name setup#0 
	for (i=0; i<3; i ++ )
	{
		dims[i][0]=512;
		dims[i][1]=256;
		dims[i][2]=256;
	}
#pragma loop name setup#1 
	for (i=0; i<3; i ++ )
	{
		xstart[i]=1;
		xend[i]=512;
		ystart[i]=1;
		yend[i]=256;
		zstart[i]=1;
		zend[i]=256;
	}
	/*
	   --------------------------------------------------------------------
	   c Set up info for blocking of ffts and transposes.  This improves
	   c performance on cache-based systems. Blocking involves
	   c working on a chunk of the problem at a time, taking chunks
	   c along the first, second, or third dimension. 
	   c
	   c - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
	   c - In cffts23 blocking is on 1st dimension (with fft on 2nd and 3rd dims)

	   c Since 1st dim is always in processor, we'll assume it's long enough 
	   c (default blocking factor is 16 so min size for 1st dim is 16)
	   c The only case we have to worry about is cffts1 in a 2d decomposition. 
	   c so the blocking factor should not be larger than the 2nd dimension. 
	   c-------------------------------------------------------------------
	 */
	fftblock=16;
	fftblockpad=18;
	if ((fftblock!=16))
	{
		fftblockpad=(fftblock+3);
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void compute_indexmap_kernel0(int * dims, size_t pitch__dims, int indexmap[256][256][512], int * xstart_i, int * ystart_i, int * zstart_i)
{
	int i;
	int ii;
	int ii2;
	int ij2;
	int j;
	int jj;
	int k;
	int kk;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=_gtid;
	if (i<( * (((int * )(((char * )dims)+(2*pitch__dims)))+0)))
	{
#pragma loop name compute_indexmap#0#0 
		for (j=0; j<( * (((int * )(((char * )dims)+(2*pitch__dims)))+1)); j ++ )
		{
#pragma loop name compute_indexmap#0#0#0 
			for (k=0; k<( * (((int * )(((char * )dims)+(2*pitch__dims)))+2)); k ++ )
			{
				ii=((((((i+1)+( * xstart_i))-2)+(512/2))%512)-(512/2));
				ii2=(ii*ii);
				jj=((((((j+1)+( * ystart_i))-2)+(256/2))%256)-(256/2));
				ij2=((jj*jj)+ii2);
				kk=((((((k+1)+( * zstart_i))-2)+(256/2))%256)-(256/2));
				indexmap[k][j][i]=((kk*kk)+ij2);
			}
		}
	}
}

static void compute_indexmap(int indexmap[256][256][512], int d[3])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
	   c for time evolution exponent. 
	   c-------------------------------------------------------------------
	 */
	int i;
	double ap;
	int xstart_i;
	int ystart_i;
	int zstart_i;
	/*
	   --------------------------------------------------------------------
	   c basically we want to convert the fortran indices 
	   c   1 2 3 4 5 6 7 8 
	   c to 
	   c   0 1 2 3 -4 -3 -2 -1
	   c The following magic formula does the trick:
	   c mod(i-1+n2, n) - n/2
	   c-------------------------------------------------------------------
	 */
	int * gpu__xstart_i;
	int * gpu__ystart_i;
	int * gpu__zstart_i;
	xstart_i=xstart[2];
	ystart_i=ystart[2];
	zstart_i=zstart[2];
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)dims[2][0])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy2D(gpu__dims, pitch__dims, dims, (3*sizeof (int)), (3*sizeof (int)), 3, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xstart_i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xstart_i, ( & xstart_i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ystart_i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__ystart_i, ( & ystart_i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zstart_i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zstart_i, ( & zstart_i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dims, indexmap, xstart_i, ystart_i, zstart_i) private(i, ii, ii2, ij2, j, jj, k, kk) schedule(static)
#pragma cuda gpurun nocudafree(dims, indexmap) 
#pragma cuda gpurun nog2cmemtr(dims, indexmap, xstart_i, ystart_i, zstart_i) 
#pragma cuda ainfo kernelid(0) procname(compute_indexmap) 
#pragma cuda gpurun cudafree(xstart_i, ystart_i, zstart_i) 
#pragma cuda gpurun noc2gmemtr(indexmap) 
	compute_indexmap_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__dims, pitch__dims, ((int (*)[256][512])gpu__indexmap__main), gpu__xstart_i, gpu__ystart_i, gpu__zstart_i);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__zstart_i));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__ystart_i));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__xstart_i));
	/*
	   --------------------------------------------------------------------
	   c compute array of exponentials for time evolution. 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp single */
	{
		ap=(((( - 4.0)*1.0E-6)*3.141592653589793)*3.141592653589793);
		ex[0]=1.0;
		ex[1]=exp(ap);
#pragma loop name compute_indexmap#1 
		for (i=2; i<=(20*((((512*512)/4)+((256*256)/4))+((256*256)/4))); i ++ )
		{
			ex[i]=(ex[(i-1)]*ex[1]);
		}
	}
	/* end single */
	return ;
}

__global__ void compute_indexmap_clnd1_kernel0(int * dims, size_t pitch__dims, int indexmap[256][256][512], int * xstart_i, int * ystart_i, int * zstart_i)
{
	int i;
	int ii;
	int ii2;
	int ij2;
	int j;
	int jj;
	int k;
	int kk;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=_gtid;
	if (i<( * (((int * )(((char * )dims)+(2*pitch__dims)))+0)))
	{
#pragma loop name compute_indexmap#0#0 
		for (j=0; j<( * (((int * )(((char * )dims)+(2*pitch__dims)))+1)); j ++ )
		{
#pragma loop name compute_indexmap#0#0#0 
			for (k=0; k<( * (((int * )(((char * )dims)+(2*pitch__dims)))+2)); k ++ )
			{
				ii=((((((i+1)+( * xstart_i))-2)+(512/2))%512)-(512/2));
				ii2=(ii*ii);
				jj=((((((j+1)+( * ystart_i))-2)+(256/2))%256)-(256/2));
				ij2=((jj*jj)+ii2);
				kk=((((((k+1)+( * zstart_i))-2)+(256/2))%256)-(256/2));
				indexmap[k][j][i]=((kk*kk)+ij2);
			}
		}
	}
}

static void compute_indexmap_clnd1(int indexmap[256][256][512], int d[3])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
	   c for time evolution exponent. 
	   c-------------------------------------------------------------------
	 */
	int i;
	double ap;
	int xstart_i;
	int ystart_i;
	int zstart_i;
	/*
	   --------------------------------------------------------------------
	   c basically we want to convert the fortran indices 
	   c   1 2 3 4 5 6 7 8 
	   c to 
	   c   0 1 2 3 -4 -3 -2 -1
	   c The following magic formula does the trick:
	   c mod(i-1+n2, n) - n/2
	   c-------------------------------------------------------------------
	 */
	int * gpu__xstart_i;
	int * gpu__ystart_i;
	int * gpu__zstart_i;
	xstart_i=xstart[2];
	ystart_i=ystart[2];
	zstart_i=zstart[2];
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xstart_i)), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)dims[2][0])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xstart_i, ( & xstart_i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ystart_i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__ystart_i, ( & ystart_i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zstart_i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zstart_i, ( & zstart_i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dims, indexmap, xstart_i, ystart_i, zstart_i) private(i, ii, ii2, ij2, j, jj, k, kk) schedule(static)
#pragma cuda gpurun noc2gmemtr(dims, indexmap) 
#pragma cuda gpurun nocudamalloc(dims, indexmap) 
#pragma cuda gpurun nocudafree(dims, indexmap) 
#pragma cuda gpurun nog2cmemtr(dims, indexmap, xstart_i, ystart_i, zstart_i) 
#pragma cuda ainfo kernelid(0) procname(compute_indexmap_clnd1) 
#pragma cuda gpurun cudafree(xstart_i, ystart_i, zstart_i) 
	compute_indexmap_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__dims, pitch__dims, ((int (*)[256][512])gpu__indexmap__main), gpu__xstart_i, gpu__ystart_i, gpu__zstart_i);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__zstart_i));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__ystart_i));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__xstart_i));
	/*
	   --------------------------------------------------------------------
	   c compute array of exponentials for time evolution. 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp single */
	{
		ap=(((( - 4.0)*1.0E-6)*3.141592653589793)*3.141592653589793);
		ex[0]=1.0;
		ex[1]=exp(ap);
#pragma loop name compute_indexmap#1 
		for (i=2; i<=(20*((((512*512)/4)+((256*256)/4))+((256*256)/4))); i ++ )
		{
			ex[i]=(ex[(i-1)]*ex[1]);
		}
	}
	/* end single */
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void print_timers(void )
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int i;
	char * tstrings[] =  { "          total ", "          setup ", "            fft ", "         evolve ", "       checksum ", "         fftlow ", "        fftcopy " } ;
#pragma loop name print_timers#0 
	for (i=0; i<7; i ++ )
	{
		if ((timer_read(i)!=0.0))
		{
			printf("timer %2d(%16s( :%10.6f\n", i, tstrings[i], timer_read(i));
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void fft(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y1[NX][FFTBLOCKPAD]; */
	/*
	   --------------------------------------------------------------------
	   c note: args x1, x2 must be different arrays
	   c note: args for cfftsx are (direction, layout, xin, xout, scratch)
	   c       xinxout may be the same and it can be somewhat faster
	   c       if they are
	   c-------------------------------------------------------------------
	 */
	if ((dir==1))
	{
		/* cffts1(1, dims[0], x1, x1, y0, y1);	x1 -> x1 */
		cffts1(1, dims[0], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2(1, dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts3(1, dims[2], x1, x2, y0, y1);	x1 -> x2 */
		cffts3(1, dims[2], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
	}
	else
	{
		/* cffts3(-1, dims[2], x1, x1, y0, y1);	x1 -> x1 */
		cffts3_clnd1(( - 1), dims[2], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(-1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2_clnd1(( - 1), dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts1(-1, dims[0], x1, x2, y0, y1);	x1 -> x2 */
		cffts1_clnd1(( - 1), dims[0], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x2 */
	}
	return ;
}

static void fft_clnd2_cloned0(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y1[NX][FFTBLOCKPAD]; */
	/*
	   --------------------------------------------------------------------
	   c note: args x1, x2 must be different arrays
	   c note: args for cfftsx are (direction, layout, xin, xout, scratch)
	   c       xinxout may be the same and it can be somewhat faster
	   c       if they are
	   c-------------------------------------------------------------------
	 */
	if ((dir==1))
	{
		/* cffts1(1, dims[0], x1, x1, y0, y1);	x1 -> x1 */
		cffts1_clnd2_cloned0(1, dims[0], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2_clnd2_cloned0(1, dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts3(1, dims[2], x1, x2, y0, y1);	x1 -> x2 */
		cffts3_clnd2_cloned0(1, dims[2], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
	}
	else
	{
		/* cffts3(-1, dims[2], x1, x1, y0, y1);	x1 -> x1 */
		cffts3_clnd3_cloned0(( - 1), dims[2], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(-1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2_clnd3_cloned0(( - 1), dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts1(-1, dims[0], x1, x2, y0, y1);	x1 -> x2 */
		cffts1_clnd3_cloned0(( - 1), dims[0], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x2 */
	}
	return ;
}

static void fft_clnd1(int dir, double x1_real[256][256][512], double x1_imag[256][256][512], double x2_real[256][256][512], double x2_imag[256][256][512])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y0[NX][FFTBLOCKPAD]; */
	/* dcomplex y1[NX][FFTBLOCKPAD]; */
	/*
	   --------------------------------------------------------------------
	   c note: args x1, x2 must be different arrays
	   c note: args for cfftsx are (direction, layout, xin, xout, scratch)
	   c       xinxout may be the same and it can be somewhat faster
	   c       if they are
	   c-------------------------------------------------------------------
	 */
	if ((dir==1))
	{
		/* cffts1(1, dims[0], x1, x1, y0, y1);	x1 -> x1 */
		cffts1_clnd4(1, dims[0], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2_clnd4(1, dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts3(1, dims[2], x1, x2, y0, y1);	x1 -> x2 */
		cffts3_clnd4(1, dims[2], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
	}
	else
	{
		/* cffts3(-1, dims[2], x1, x1, y0, y1);	x1 -> x1 */
		cffts3_clnd5(( - 1), dims[2], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts2(-1, dims[1], x1, x1, y0, y1);	x1 -> x1 */
		cffts2_clnd5(( - 1), dims[1], x1_real, x1_imag, x1_real, x1_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x1 */
		/* cffts1(-1, dims[0], x1, x2, y0, y1);	x1 -> x2 */
		cffts1_clnd5(( - 1), dims[0], x1_real, x1_imag, x2_real, x2_imag, NULL, NULL, NULL, NULL);
		/* x1 -> x2 */
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void cffts1_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__fftblock, ( & fftblock), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__fftblockpad, ( & fftblockpad), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_imag, u_imag, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_real, u_real, gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts1_clnd5_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1_clnd5) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_clnd5_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u0_imag__main), ((double (*)[256][512])gpu__u0_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts1_clnd4_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_imag, u_imag, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_real, u_real, gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1_clnd4) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_clnd4_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts1_clnd3_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun multisrccg(xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1_clnd3_cloned0) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_clnd3_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u2_imag__main), ((double (*)[256][512])gpu__u2_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts1_clnd2_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1_clnd2_cloned0) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_clnd2_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts1_clnd1_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_0, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int j;
	int jj;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[0];
#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock_0); jj+=fftblock_0)
		{
#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d_0; i ++ )
				{
					yy0_real[_gtid][i][j]=x_real[k][(j+jj)][i];
					yy0_imag[_gtid][i][j]=x_imag[k][(j+jj)][i];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_0), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock_0; j ++ )
			{
#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d_0; i ++ )
				{
					xout_real[k][(j+jj)][i]=yy0_real[_gtid][i][j];
					xout_imag[k][(j+jj)][i]=yy0_imag[_gtid][i][j];
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts1_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_0;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_0;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_0=logd[0];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_0)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_0, ( & logd_0), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, j, jj, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_0, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts1_clnd1) 
#pragma cuda gpurun registerRO(d[0], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_0) 
	cffts1_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_0, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u0_imag__main), ((double (*)[256][512])gpu__u0_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_0));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void cffts2_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts2_clnd5_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2_clnd5) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_clnd5_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts2_clnd4_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2_clnd4) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_clnd4_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts2_clnd3_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2_clnd3_cloned0) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_clnd3_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts2_clnd2_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2_clnd2_cloned0) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_clnd2_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts2_clnd1_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_1, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	fftblock_0=( * fftblock);
	if (k<d[2])
	{
		d_0=d[1];
#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#0 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][j][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][j][i]=x_imag[k][j][(i+ii)];
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_1), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts2#1#0#1 
			for (j=0; j<d_0; j ++ )
			{
#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][j][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][j][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts2_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_1;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_1;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_1=logd[1];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_1, ( & logd_1), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[2])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_1, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts2_clnd1) 
#pragma cuda gpurun registerRO(d[1], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_1) 
	cffts2_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_1, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void cffts3_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u0_imag__main), ((double (*)[256][512])gpu__u0_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts3_clnd5_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3_clnd5(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_imag, u_imag, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_real, u_real, gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3_clnd5) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_clnd5_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts3_clnd4_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3_clnd4(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3_clnd4) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_clnd4_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u0_imag__main), ((double (*)[256][512])gpu__u0_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts3_clnd3_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3_clnd3_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3_clnd3_cloned0) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_clnd3_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts3_clnd2_cloned0_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3_clnd2_cloned0(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun multisrccg(xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3_clnd2_cloned0) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_clnd2_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u2_imag__main), ((double (*)[256][512])gpu__u2_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

__global__ void cffts3_clnd1_kernel0(int * d, int * fftblock, int * fftblockpad, int * is, int * logd_2, double * u_imag, double * u_real, double x_imag[256][256][512], double x_real[256][256][512], double xout_imag[256][256][512], double xout_real[256][256][512], double yy0_imag[][512][18], double yy0_real[][512][18], double yy1_imag[][512][18], double yy1_real[][512][18])
{
	int d_0;
	int fftblock_0;
	int i;
	int ii;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=_gtid;
	fftblock_0=( * fftblock);
	if (j<d[1])
	{
		d_0=d[2];
#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock_0); ii+=fftblock_0)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#0 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					yy0_real[_gtid][k][i]=x_real[k][j][(i+ii)];
					yy0_imag[_gtid][k][i]=x_imag[k][j][(i+ii)];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			dev_cfftz(( * is), ( * logd_2), d_0, yy0_real, yy0_imag, yy1_real, yy1_imag, fftblock, fftblockpad, u_imag, u_real, _gtid);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
#pragma loop name cffts3#1#0#1 
			for (k=0; k<d_0; k ++ )
			{
#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock_0; i ++ )
				{
					xout_real[k][j][(i+ii)]=yy0_real[_gtid][k][i];
					xout_imag[k][j][(i+ii)]=yy0_imag[_gtid][k][i];
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
}

static void cffts3_clnd1(int is, int d[3], double x_real[256][256][512], double x_imag[256][256][512], double xout_real[256][256][512], double xout_imag[256][256][512], double y0_real[512][18], double y0_imag[512][18], double y1_real[512][18], double y1_imag[512][18])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int logd[3];
	int i;
	int logd_2;
	int * gpu__d;
	int * gpu__is;
	int * gpu__logd_2;
	double * gpu__yy0_imag;
	double * gpu__yy0_real;
	double * gpu__yy1_imag;
	double * gpu__yy1_real;
#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	logd_2=logd[2];
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__d)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__d, d, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__fftblock, ( & fftblock), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__fftblockpad, ( & fftblockpad), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__is)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__is, ( & is), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__logd_2)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__logd_2, ( & logd_2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_imag, u_imag, gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=(512*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__u_real, u_real, gpuBytes, cudaMemcpyHostToDevice));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)d[1])/1024.0F)));
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=((int)ceil((((float)gpuNumBlocks)/10000.0F)));
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy0_real)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_imag)), gpuBytes));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yy1_real)), gpuBytes));
#pragma omp parallel for threadprivate(yy0_imag, yy0_real, yy1_imag, yy1_real) shared(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) private(i, ii, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudafree(fftblock, fftblockpad, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda gpurun nog2cmemtr(d, fftblock, fftblockpad, is, logd_2, u_imag, u_real, x_imag, x_real, xout_imag, xout_real) 
#pragma cuda ainfo kernelid(0) procname(cffts3_clnd1) 
#pragma cuda gpurun registerRO(d[2], fftblock) 
#pragma cuda gpurun cudafree(d, is, logd_2) 
	cffts3_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__d, gpu__fftblock, gpu__fftblockpad, gpu__is, gpu__logd_2, gpu__u_imag, gpu__u_real, ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[256][512])gpu__u1_imag__main), ((double (*)[256][512])gpu__u1_real__main), ((double (*)[512][18])gpu__yy0_imag), ((double (*)[512][18])gpu__yy0_real), ((double (*)[512][18])gpu__yy1_imag), ((double (*)[512][18])gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_real, gpu__yy1_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy1_imag, gpu__yy1_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy1_imag));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_real, gpu__yy0_real, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_real));
	gpuBytes=((512*18)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(yy0_imag, gpu__yy0_imag, gpuBytes, cudaMemcpyDeviceToHost));
	gpuBytes=(totalNumThreads*((512*18)*sizeof (double)));
	CUDA_SAFE_CALL(cudaFree(gpu__yy0_imag));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__logd_2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__is));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaFree(gpu__d));
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void fft_init(int n)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c compute the roots-of-unity array that will be used for subsequent FFTs. 
	   c-------------------------------------------------------------------
	 */
	int m;
	int nu;
	int ku;
	int i;
	int j;
	int ln;
	double t;
	double ti;
	/*
	   --------------------------------------------------------------------
	   c   Initialize the U array with sines and cosines in a manner that permits
	   c   stride one access at each FFT iteration.
	   c-------------------------------------------------------------------
	 */
	nu=n;
	m=ilog2(n);
	u_real[0]=((double)m);
	u_imag[0]=0.0;
	ku=1;
	ln=1;
#pragma loop name fft_init#0 
	for (j=1; j<=m; j ++ )
	{
		t=(3.141592653589793/ln);
#pragma loop name fft_init#0#0 
		for (i=0; i<=(ln-1); i ++ )
		{
			ti=(i*t);
			u_real[(i+ku)]=cos(ti);
			u_imag[(i+ku)]=sin(ti);
		}
		ku=(ku+ln);
		ln=(2*ln);
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__device__ static void dev_cfftz(int is, int m, int n, double x_real[][512][18], double x_imag[][512][18], double y_real[][512][18], double y_imag[][512][18], int * fftblock, int * fftblockpad, double u_imag[512], double u_real[512], int _gtid)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   Computes NY N-point complex-to-complex FFTs of X using an algorithm due
	   c   to Swarztrauber.  X is both the input and the output array, while Y is a 
	   c   scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to 
	   c   perform FFTs, the array U must be initialized by calling CFFTZ with IS 
	   c   set to 0 and M set to MX, where MX is the maximum value of M for any 
	   c   subsequent call.
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int l;
	int mx;
	/*
	   --------------------------------------------------------------------
	   c   Check if input parameters are invalid.
	   c-------------------------------------------------------------------
	 */
	int fftblock_0;
	fftblock_0=( * fftblock);
	mx=((int)u_real[0]);
	/*
	   --------------------------------------------------------------------
	   c   Perform one variant of the Stockham FFT.
	   c-------------------------------------------------------------------
	 */
#pragma loop name cfftz#0 
	for (l=1; l<=m; l+=2)
	{
		dev_fftz2(is, l, m, n, fftblock_0, ( * fftblockpad), u_real, u_imag, x_real, x_imag, y_real, y_imag, _gtid);
		if ((l==m))
		{
			break;
		}
		dev_fftz2(is, (l+1), m, n, fftblock_0, ( * fftblockpad), u_real, u_imag, y_real, y_imag, x_real, x_imag, _gtid);
	}
	/*
	   --------------------------------------------------------------------
	   c   Copy Y to X.
	   c-------------------------------------------------------------------
	 */
	if (((m%2)==1))
	{
#pragma loop name cfftz#1 
		for (j=0; j<n; j ++ )
		{
#pragma loop name cfftz#1#0 
			for (i=0; i<fftblock_0; i ++ )
			{
				x_real[_gtid][j][i]=y_real[_gtid][j][i];
				x_imag[_gtid][j][i]=y_imag[_gtid][j][i];
			}
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__device__ static void dev_fftz2(int is, int l, int m, int n, int ny, int ny1, double u_real[512], double u_imag[512], double x_real[][512][18], double x_imag[][512][18], double y_real[][512][18], double y_imag[][512][18], int _gtid)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   Performs the L-th iteration of the second variant of the Stockham FFT.
	   c-------------------------------------------------------------------
	 */
	int k;
	int n1;
	int li;
	int lj;
	int lk;
	int ku;
	int i;
	int j;
	int i11;
	int i12;
	int i21;
	int i22;
	double u1_real;
	double u1_imag;
	/*
	   --------------------------------------------------------------------
	   c   Set initial parameters.
	   c-------------------------------------------------------------------
	 */
	n1=(n/2);
	if (((l-1)==0))
	{
		lk=1;
	}
	else
	{
		lk=(2<<((l-1)-1));
	}
	if (((m-l)==0))
	{
		li=1;
	}
	else
	{
		li=(2<<((m-l)-1));
	}
	lj=(2*lk);
	ku=li;
#pragma loop name fftz2#0 
	for (i=0; i<li; i ++ )
	{
		i11=(i*lk);
		i12=(i11+n1);
		i21=(i*lj);
		i22=(i21+lk);
		if ((is>=1))
		{
			u1_real=u_real[(ku+i)];
			u1_imag=u_imag[(ku+i)];
		}
		else
		{
			u1_real=u_real[(ku+i)];
			u1_imag=( - u_imag[(ku+i)]);
		}
		/*
		   --------------------------------------------------------------------
		   c   This loop is vectorizable.
		   c-------------------------------------------------------------------
		 */
#pragma loop name fftz2#0#0 
		for (k=0; k<lk; k ++ )
		{
#pragma loop name fftz2#0#0#0 
			for (j=0; j<ny; j ++ )
			{
				double x11real;
				double x11imag;
				double x21real;
				double x21imag;
				x11real=x_real[_gtid][(i11+k)][j];
				x11imag=x_imag[_gtid][(i11+k)][j];
				x21real=x_real[_gtid][(i12+k)][j];
				x21imag=x_imag[_gtid][(i12+k)][j];
				y_real[_gtid][(i21+k)][j]=(x11real+x21real);
				y_imag[_gtid][(i21+k)][j]=(x11imag+x21imag);
				y_real[_gtid][(i22+k)][j]=((u1_real*(x11real-x21real))-(u1_imag*(x11imag-x21imag)));
				y_imag[_gtid][(i22+k)][j]=((u1_real*(x11imag-x21imag))+(u1_imag*(x11real-x21real)));
			}
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static int ilog2(int n)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int nn;
	int lg;
	int _ret_val_0;
	if ((n==1))
	{
		_ret_val_0=0;
		return _ret_val_0;
	}
	lg=1;
	nn=2;
	while (nn<n)
	{
		nn=(nn<<1);
		lg ++ ;
	}
	return lg;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void checksum_kernel0(double * red__chk_imag, double * red__chk_real, double u1_imag[256][256][512], double u1_real[256][256][512], int * xend, int * xstart, int * yend, int * ystart, int * zend, int * zstart)
{
	__shared__ double sh__chk_imag[BLOCK_SIZE];
	__shared__ double sh__chk_real[BLOCK_SIZE];
	int xstart_0;
	int ystart_0;
	int zstart_0;
	int j;
	int q;
	int r;
	int s;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	zstart_0=zstart[0];
	ystart_0=ystart[0];
	xstart_0=xstart[0];
	sh__chk_real[threadIdx.x]=0.0F;
	sh__chk_imag[threadIdx.x]=0.0F;
	j=(_gtid+1);
#pragma omp for nowait
	if (j<=1024)
	{
		q=((j%512)+1);
		if (((q>=xstart_0)&&(q<=xend[0])))
		{
			r=(((3*j)%256)+1);
			if (((r>=ystart_0)&&(r<=yend[0])))
			{
				s=(((5*j)%256)+1);
				if (((s>=zstart_0)&&(s<=zend[0])))
				{
					/* cadd is a macro in npb-C.h adding the real and imaginary */
					/* component. So, the preprocessed statement still follows the */
					/* reduction pattern */
					/* cadd(chk,chk,u1[s-zstart[0]][r-ystart[0]][q-xstart[0]]); */
					sh__chk_real[threadIdx.x]=(sh__chk_real[threadIdx.x]+u1_real[(s-zstart_0)][(r-ystart_0)][(q-xstart_0)]);
					sh__chk_imag[threadIdx.x]=(sh__chk_imag[threadIdx.x]+u1_imag[(s-zstart_0)][(r-ystart_0)][(q-xstart_0)]);
				}
			}
		}
	}
	__syncthreads();
	if ((threadIdx.x<256))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+256)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+256)];
	}
	__syncthreads();
	if ((threadIdx.x<128))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+128)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+128)];
	}
	__syncthreads();
	if ((threadIdx.x<64))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+64)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+64)];
	}
	__syncthreads();
	if ((threadIdx.x<32))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+32)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+32)];
	}
	if ((threadIdx.x<16))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+16)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+16)];
	}
	if ((threadIdx.x<8))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+8)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+8)];
	}
	if ((threadIdx.x<4))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+4)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+4)];
	}
	if ((threadIdx.x<2))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+2)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+2)];
	}
	if ((threadIdx.x<1))
	{
		sh__chk_imag[threadIdx.x]+=sh__chk_imag[(threadIdx.x+1)];
		sh__chk_real[threadIdx.x]+=sh__chk_real[(threadIdx.x+1)];
	}
	if ((threadIdx.x==0))
	{
		red__chk_imag[_bid]=sh__chk_imag[0];
		red__chk_real[_bid]=sh__chk_real[0];
	}
}

static void checksum(int i, double u1_real[256][256][512], double u1_imag[256][256][512], int d[3])
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	double _chk_real;
	double _chk_imag;
	double * red__chk_imag;
	double * chk_imag__extended;
	int _ti_100_0;
	double * red__chk_real;
	double * chk_real__extended;
	_chk_real=0.0;
	_chk_imag=0.0;
	{
		double chk_real = _chk_real;
		double chk_imag = _chk_imag;
		/* #pragma omp for nowait */
		dim3 dimBlock0(gpuNumThreads, 1, 1);
		gpuNumBlocks=1;
		if ((gpuNumBlocks>MAX_GDIMENSION))
		{
			gpuNumBlocks2=1;
			gpuNumBlocks1=MAX_NDIMENSION;
		}
		else
		{
			gpuNumBlocks2=1;
			gpuNumBlocks1=gpuNumBlocks;
		}
		dim3 dimGrid0(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=(gpuNumBlocks*sizeof (double));
		CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & red__chk_imag)), gpuBytes));
		chk_imag__extended=((double * )malloc(gpuBytes));
		gpuBytes=(gpuNumBlocks*sizeof (double));
		CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & red__chk_real)), gpuBytes));
		chk_real__extended=((double * )malloc(gpuBytes));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__xend, xend, gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__xstart, xstart, gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__yend, yend, gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__ystart, ystart, gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__zend, zend, gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=(3*sizeof (int));
		CUDA_SAFE_CALL(cudaMemcpy(gpu__zstart, zstart, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel shared(u1_imag, u1_real, xend, xstart, yend, ystart, zend, zstart) private(j, q, r, s) reduction(+: chk_imag, chk_real) schedule(static)
#pragma cuda gpurun noc2gmemtr(u1_imag, u1_real, x_imag, x_real, xout_imag, xout_real, yy0_imag, yy0_real) 
#pragma cuda gpurun nocudamalloc(u1_imag, u1_real) 
#pragma cuda gpurun nocudafree(u1_imag, u1_real, xend, xstart, yend, ystart, zend, zstart) 
#pragma cuda gpurun multisrccg(xend, xstart, yend, ystart, zend, zstart) 
#pragma cuda gpurun nog2cmemtr(u1_imag, u1_real, xend, xstart, yend, ystart, zend, zstart) 
#pragma cuda ainfo kernelid(0) procname(checksum) 
#pragma cuda gpurun registerRO(xstart[0], ystart[0], zstart[0]) 
		checksum_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(red__chk_imag, red__chk_real, ((double (*)[256][512])gpu__u2_imag__main), ((double (*)[256][512])gpu__u2_real__main), gpu__xend, gpu__xstart, gpu__yend, gpu__ystart, gpu__zend, gpu__zstart);
		gpuBytes=(gpuNumBlocks*sizeof (double));
		CUDA_SAFE_CALL(cudaMemcpy(chk_real__extended, red__chk_real, gpuBytes, cudaMemcpyDeviceToHost));
		for (_ti_100_0=0; _ti_100_0<gpuNumBlocks; _ti_100_0 ++ )
		{
			chk_real+=chk_real__extended[_ti_100_0];
		}
		free(chk_real__extended);
		CUDA_SAFE_CALL(cudaFree(red__chk_real));
		gpuBytes=(gpuNumBlocks*sizeof (double));
		CUDA_SAFE_CALL(cudaMemcpy(chk_imag__extended, red__chk_imag, gpuBytes, cudaMemcpyDeviceToHost));
		for (_ti_100_0=0; _ti_100_0<gpuNumBlocks; _ti_100_0 ++ )
		{
			chk_imag+=chk_imag__extended[_ti_100_0];
		}
		free(chk_imag__extended);
		CUDA_SAFE_CALL(cudaFree(red__chk_imag));
		_chk_real=chk_real;
		_chk_imag=chk_imag;
	}
	/* #pragma omp critical */
	{
		sums_real[i]+=_chk_real;
		sums_imag[i]+=_chk_imag;
	}
	/* #pragma omp barrier */
	/* #pragma omp single */
	{
		/* complex % real */
		sums_real[i]=(sums_real[i]/((double)33554432));
		sums_imag[i]=(sums_imag[i]/((double)33554432));
		printf("T = %5d     Checksum = %22.12e %22.12e\n", i, sums_real[i], sums_imag[i]);
	}
	return ;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
static void verify(int d1, int d2, int d3, int nt, int * verified, char * cclass)
{
	/*
	   --------------------------------------------------------------------
	   c-------------------------------------------------------------------
	 */
	int i;
	double err;
	double epsilon;
	/*
	   --------------------------------------------------------------------
	   c   Sample size reference checksums
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   Class S size reference checksums
	   c-------------------------------------------------------------------
	 */
	double vdata_real_s[(6+1)] =  { 0.0, 554.6087004964, 554.6385409189, 554.6148406171, 554.5423607415, 554.4255039624, 554.2683411902 } ;
	double vdata_imag_s[(6+1)] =  { 0.0, 484.5363331978, 486.5304269511, 488.3910722336, 490.1273169046, 491.7475857993, 493.2597244941 } ;
	/*
	   --------------------------------------------------------------------
	   c   Class W size reference checksums
	   c-------------------------------------------------------------------
	 */
	double vdata_real_w[(6+1)] =  { 0.0, 567.3612178944, 563.1436885271, 559.402408997, 556.069804702, 553.089899125, 550.4159734538 } ;
	double vdata_imag_w[(6+1)] =  { 0.0, 529.3246849175, 528.2149986629, 527.0996558037, 526.0027904925, 524.9400845633, 523.9212247086 } ;
	/*
	   --------------------------------------------------------------------
	   c   Class A size reference checksums
	   c-------------------------------------------------------------------
	 */
	double vdata_real_a[(6+1)] =  { 0.0, 504.6735008193, 505.9412319734, 506.9376896287, 507.7892868474, 508.5233095391, 509.1487099959 } ;
	double vdata_imag_a[(6+1)] =  { 0.0, 511.404790551, 509.8809666433, 509.8144042213, 510.1336130759, 510.4914655194, 510.7917842803 } ;
	/*
	   --------------------------------------------------------------------
	   c   Class B size reference checksums
	   c-------------------------------------------------------------------
	 */
	double vdata_real_b[(20+1)] =  { 0.0, 517.7643571579, 515.4521291263, 514.6409228649, 514.2378756213, 513.9626667737, 513.7423460082, 513.5547056878, 513.3910925466, 513.247070539, 513.1197729984, 513.0070319283, 512.9070537032, 512.8182883502, 512.7393733383, 512.669106202, 512.6064276004, 512.550407657, 512.500233172, 512.4551951846, 512.4146770029 } ;
	double vdata_imag_b[(20+1)] =  { 0.0, 507.7803458597, 508.8249431599, 509.6208912659, 510.1023387619, 510.3976610617, 510.5948019802, 510.7404165783, 510.8576573661, 510.9577278523, 511.0460304483, 511.12524338, 511.1968077718, 511.2616233064, 511.3203605551, 511.3735928093, 511.4218460548, 511.465613976, 511.5053595966, 511.5415130407, 511.5744692211 } ;
	/*
	   --------------------------------------------------------------------
	   c   Class C size reference checksums
	   c-------------------------------------------------------------------
	 */
	double vdata_real_c[(20+1)] =  { 0.0, 519.5078707457, 515.5422171134, 514.4678022222, 514.0150594328, 513.755042681, 513.5811056728, 513.4569343165, 513.3651975661, 513.2955192805, 513.2410471738, 513.1971141679, 513.1605205716, 513.1290734194, 513.1012720314, 513.0760908195, 513.0528295923, 513.0310107773, 513.0103090133, 512.9905029333, 512.9714421109 } ;
	double vdata_imag_c[(20+1)] =  { 0.0, 514.9019699238, 512.7578201997, 512.2251847514, 512.1090289018, 512.1143685824, 512.1496764568, 512.1870921893, 512.2193250322, 512.2454735794, 512.2663649603, 512.2830879827, 512.2965869718, 512.3075927445, 512.3166486553, 512.3241541685, 512.3304037599, 512.3356167976, 512.3399592211, 512.3435588985, 512.3465164008 } ;
	epsilon=1.0E-12;
	( * verified)=1;
	( * cclass)='U';
	if (((((d1==64)&&(d2==64))&&(d3==64))&&(nt==6)))
	{
		( * cclass)='S';
#pragma loop name verify#0 
		for (i=1; i<=nt; i ++ )
		{
			err=((sums_real[i]-vdata_real_s[i])/vdata_real_s[i]);
			if ((fabs(err)>epsilon))
			{
				( * verified)=0;
				break;
			}
			err=((sums_imag[i]-vdata_imag_s[i])/vdata_imag_s[i]);
			if ((fabs(err)>epsilon))
			{
				( * verified)=0;
				break;
			}
		}
	}
	else
	{
		if (((((d1==128)&&(d2==128))&&(d3==32))&&(nt==6)))
		{
			( * cclass)='W';
#pragma loop name verify#1 
			for (i=1; i<=nt; i ++ )
			{
				err=((sums_real[i]-vdata_real_w[i])/vdata_real_w[i]);
				if ((fabs(err)>epsilon))
				{
					( * verified)=0;
					break;
				}
				err=((sums_imag[i]-vdata_imag_w[i])/vdata_imag_w[i]);
				if ((fabs(err)>epsilon))
				{
					( * verified)=0;
					break;
				}
			}
		}
		else
		{
			if (((((d1==256)&&(d2==256))&&(d3==128))&&(nt==6)))
			{
				( * cclass)='A';
#pragma loop name verify#2 
				for (i=1; i<=nt; i ++ )
				{
					err=((sums_real[i]-vdata_real_a[i])/vdata_real_a[i]);
					if ((fabs(err)>epsilon))
					{
						( * verified)=0;
						break;
					}
					err=((sums_imag[i]-vdata_imag_a[i])/vdata_imag_a[i]);
					if ((fabs(err)>epsilon))
					{
						( * verified)=0;
						break;
					}
				}
			}
			else
			{
				if (((((d1==512)&&(d2==256))&&(d3==256))&&(nt==20)))
				{
					( * cclass)='B';
#pragma loop name verify#3 
					for (i=1; i<=nt; i ++ )
					{
						err=((sums_real[i]-vdata_real_b[i])/vdata_real_b[i]);
						if ((fabs(err)>epsilon))
						{
							( * verified)=0;
							break;
						}
						err=((sums_imag[i]-vdata_imag_b[i])/vdata_imag_b[i]);
						if ((fabs(err)>epsilon))
						{
							( * verified)=0;
							break;
						}
					}
				}
				else
				{
					if (((((d1==512)&&(d2==512))&&(d3==512))&&(nt==20)))
					{
						( * cclass)='C';
#pragma loop name verify#4 
						for (i=1; i<=nt; i ++ )
						{
							err=((sums_real[i]-vdata_real_c[i])/vdata_real_c[i]);
							if ((fabs(err)>epsilon))
							{
								( * verified)=0;
								break;
							}
							err=((sums_imag[i]-vdata_imag_c[i])/vdata_imag_c[i]);
							if ((fabs(err)>epsilon))
							{
								( * verified)=0;
								break;
							}
						}
					}
				}
			}
		}
	}
	if ((( * cclass)!='U'))
	{
		printf("Result verification successful\n");
	}
	else
	{
		printf("Result verification failed\n");
	}
	printf("cclass = %1c\n", ( * cclass));
	return ;
}

