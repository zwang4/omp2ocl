/*
--------------------------------------------------------------------
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - EP

  This benchmark is an OpenMP C version of the NPB EP code.
  
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

  Author: P. O. Frederickson 
          D. H. Bailey
          A. C. Woo

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------
*/
#include "npb-C.h"
#include "npbparams.h"

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



float * gpu__an__main;
int * gpu__k_offset__main;
int * gpu__l__main;
int * gpu__np__main;
/* parameters */
/* global variables */
/* commonstorage */
/* static float qq[NQ];		private copy of q[0:NQ-1] */
static float x[(2*(1<<1))];
#pragma omp threadprivate(x)
static float q[10];
/*  */
/*          E  L  A  P  S  E  D  _  T  I  M  E */
/*  */
float elapsed_time(void )
{
float t;
wtime(( & t));
return t;
}

float start[64];
float elapsed[64];
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
float t;
float now;
now=elapsed_time();
t=(now-start[n]);
elapsed[n]+=t;
return ;
}

/*  */
/*             T  I  M  E  R  _  R  E  A  D */
/*  */
float timer_read(int n)
{
float _ret_val_0;
_ret_val_0=elapsed[n];
return _ret_val_0;
}

static void c_print_results(char * name, char ccclass, int n1, int n2, int n3, int niter, int nthreads, float t, float mops, char * optype, int passed_verification, char * npbversion, char * compiletime, char * cc, char * clink, char * c_lib, char * c_inc, char * cflags, char * clinkflags, char * rand)
{
printf("\n\n %s Benchmark Completed\n", name);
printf(" Class           =                        %c\n", ccclass);
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
__device__ static float dev_randlc(float * x, float a)
{
/*
c---------------------------------------------------------------------
c---------------------------------------------------------------------
*/
/*
c---------------------------------------------------------------------
c
c   This routine returns a uniform pseudorandom float precision number in the
c   range (0, 1) by using the linear congruential generator
c
c   x_{k+1} = a x_k  (mod 2^46)
c
c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
c   before repeating.  The argument A is the same as 'a' in the above formula,
c   and X is the same as x_0.  A and X must be odd float precision integers
c   in the range (1, 2^46).  The returned value RANDLC is normalized to be
c   between 0 and 1, i.e. RANDLC = 2^(-46) x_1.  X is updated to contain
c   the new seed x_1, so that subsequent calls to RANDLC using the same
c   arguments will generate a continuous sequence.
c
c   This routine should produce the same results on any computer with at least
c   48 mantissa bits in float precision floating point data.  On 64 bit
c   systems, float precision should be disabled.
c
c   David H. Bailey     October 26, 1990
c
c---------------------------------------------------------------------
*/
float t1;
float t2;
float t3;
float t4;
float a1;
float a2;
float x1;
float x2;
float z;
/*
c---------------------------------------------------------------------
c   Break A into two parts such that A = 2^23 A1 + A2.
c---------------------------------------------------------------------
*/
float _ret_val_0;
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

float randlc(float * x, float a)
{
/*
c---------------------------------------------------------------------
c---------------------------------------------------------------------
*/
/*
c---------------------------------------------------------------------
c
c   This routine returns a uniform pseudorandom float precision number in the
c   range (0, 1) by using the linear congruential generator
c
c   x_{k+1} = a x_k  (mod 2^46)
c
c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
c   before repeating.  The argument A is the same as 'a' in the above formula,
c   and X is the same as x_0.  A and X must be odd float precision integers
c   in the range (1, 2^46).  The returned value RANDLC is normalized to be
c   between 0 and 1, i.e. RANDLC = 2^(-46) x_1.  X is updated to contain
c   the new seed x_1, so that subsequent calls to RANDLC using the same
c   arguments will generate a continuous sequence.
c
c   This routine should produce the same results on any computer with at least
c   48 mantissa bits in float precision floating point data.  On 64 bit
c   systems, float precision should be disabled.
c
c   David H. Bailey     October 26, 1990
c
c---------------------------------------------------------------------
*/
float t1;
float t2;
float t3;
float t4;
float a1;
float a2;
float x1;
float x2;
float z;
/*
c---------------------------------------------------------------------
c   Break A into two parts such that A = 2^23 A1 + A2.
c---------------------------------------------------------------------
*/
float _ret_val_0;
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
__device__ static void dev_vranlc_n(int n, float * x_seed, float a, float * y, size_t pitch__x)
{
/*
c---------------------------------------------------------------------
c---------------------------------------------------------------------
*/
/*
c---------------------------------------------------------------------
c
c   This routine generates N uniform pseudorandom float precision numbers in
c   the range (0, 1) by using the linear congruential generator
c
c   x_{k+1} = a x_k  (mod 2^46)
c
c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
c   before repeating.  The argument A is the same as 'a' in the above formula,
c   and X is the same as x_0.  A and X must be odd float precision integers
c   in the range (1, 2^46).  The N results are placed in Y and are normalized
c   to be between 0 and 1.  X is updated to contain the new seed, so that
c   subsequent calls to VRANLC using the same arguments will generate a
c   continuous sequence.  If N is zero, only initialization is performed, and
c   the variables X, A and Y are ignored.
c
c   This routine is the standard version designed for scalar or RISC systems.
c   However, it should produce the same results on any single processor
c   computer with at least 48 mantissa bits in float precision floating point
c   data.  On 64 bit systems, float precision should be disabled.
c
c---------------------------------------------------------------------
*/
int i;
float x;
float t1;
float t2;
float t3;
float t4;
float a1;
float a2;
float x1;
float x2;
float z;
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
#pragma loop name vranlc_n#0 
for (i=0; i<n; i ++ )
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
( * ((float * )(((char * )y)+(i*pitch__x))))=((((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*((((((((((((((((((((((0.5*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5)*0.5))*x);
}
( * x_seed)=x;
return ;
}

void vranlc(int n, float * x_seed, float a, float y[])
{
/*
c---------------------------------------------------------------------
c---------------------------------------------------------------------
*/
/*
c---------------------------------------------------------------------
c
c   This routine generates N uniform pseudorandom float precision numbers in
c   the range (0, 1) by using the linear congruential generator
c
c   x_{k+1} = a x_k  (mod 2^46)
c
c   where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
c   before repeating.  The argument A is the same as 'a' in the above formula,
c   and X is the same as x_0.  A and X must be odd float precision integers
c   in the range (1, 2^46).  The N results are placed in Y and are normalized
c   to be between 0 and 1.  X is updated to contain the new seed, so that
c   subsequent calls to VRANLC using the same arguments will generate a
c   continuous sequence.  If N is zero, only initialization is performed, and
c   the variables X, A and Y are ignored.
c
c   This routine is the standard version designed for scalar or RISC systems.
c   However, it should produce the same results on any single processor
c   computer with at least 48 mantissa bits in float precision floating point
c   data.  On 64 bit systems, float precision should be disabled.
c
c---------------------------------------------------------------------
*/
int i;
float x;
float t1;
float t2;
float t3;
float t4;
float a1;
float a2;
float x1;
float x2;
float z;
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

/*
--------------------------------------------------------------------
      program EMBAR
c-------------------------------------------------------------------
*/
/*

c   This is the serial version of the APP Benchmark 1,
c   the "embarassingly parallel" benchmark.
c
c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
c   numbers.  MK is the Log_2 of the size of each batch of uniform random
c   numbers.  MK can be set for convenience on a given system, since it does
c   not affect the results.

*/
__global__ void main_kernel0(float * red__sx, float * red__sy, float * an, int * k_offset, int * l, int * np, float * x, size_t pitch__x)
{
__shared__ float sh__sx[BLOCK_SIZE];
__shared__ float sh__sy[BLOCK_SIZE];
float * x_0;
int i;
int ik;
int k;
int kk;
float t1;
float t2;
float t3;
float t4;
float x1;
float x2;
int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
int _gtid = (threadIdx.x+(_bid*blockDim.x));
x_0=(((float * )x)+_gtid);
sh__sy[threadIdx.x]=0.0F;
sh__sx[threadIdx.x]=0.0F;
k=(_gtid+1);
#pragma omp for nowait
if (k<=( * np))
{
kk=(( * k_offset)+k);
t1=2.71828183E8;
t2=( * an);
/*      Find starting seed t1 for this kk. */
#pragma loop name main#4#0 
for (i=1; i<=100; i ++ )
{
ik=(kk/2);
if (((2*ik)!=kk))
{
t3=dev_randlc(( & t1), t2);
}
if ((ik==0))
{
break;
}
t3=dev_randlc(( & t2), t2);
kk=ik;
}
/*      Compute uniform pseudorandom numbers. */
dev_vranlc_n((2*(1<<1)), ( & t1), 1.220703125E9, x_0, pitch__x);
/* if (TIMERS_ENABLED == TRUE) timer_stop(3); */
/*

c       Compute Gaussian deviates by acceptance-rejection method and 
c       tally counts in concentric square annuli.  This loop is not 
c       vectorizable.

*/
/* if (TIMERS_ENABLED == TRUE) timer_start(2); */
#pragma loop name main#4#1 
for (i=0; i<(1<<1); i ++ )
{
x1=((2.0*( * ((float * )(((char * )x_0)+((2*i)*pitch__x)))))-1.0);
x2=((2.0*( * ((float * )(((char * )x_0)+(((2*i)+1)*pitch__x)))))-1.0);
t1=((x1*x1)+(x2*x2));
if ((t1<=1.0))
{
t2=sqrt(((( - 2.0)*log(t1))/t1));
t3=(x1*t2);
/* Xi */
t4=(x2*t2);
/* Yi */
( * l)=((fabs(t3)>fabs(t4)) ? fabs(t3) : fabs(t4));
/* qq[l] += 1.0;				counts */
sh__sx[threadIdx.x]=(sh__sx[threadIdx.x]+t3);
/* sum of Xi */
sh__sy[threadIdx.x]=(sh__sy[threadIdx.x]+t4);
/* sum of Yi */
}
}
/* if (TIMERS_ENABLED == TRUE) timer_stop(2); */
}
__syncthreads();
if ((threadIdx.x<256))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+256)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+256)];
}
__syncthreads();
if ((threadIdx.x<128))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+128)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+128)];
}
__syncthreads();
if ((threadIdx.x<64))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+64)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+64)];
}
__syncthreads();
if ((threadIdx.x<32))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+32)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+32)];
}
if ((threadIdx.x<16))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+16)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+16)];
}
if ((threadIdx.x<8))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+8)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+8)];
}
if ((threadIdx.x<4))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+4)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+4)];
}
if ((threadIdx.x<2))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+2)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+2)];
}
if ((threadIdx.x<1))
{
sh__sx[threadIdx.x]+=sh__sx[(threadIdx.x+1)];
sh__sy[threadIdx.x]+=sh__sy[(threadIdx.x+1)];
}
if ((threadIdx.x==0))
{
red__sx[_bid]=sh__sx[0];
red__sy[_bid]=sh__sy[0];
}
}

int main(int argc, char *  * argv)
{
float Mops;
float t1;
float t2;
float sx;
float sy;
float tm;
float an;
float tt;
float gc;
float dum[3] =  { 1.0, 1.0, 1.0 } ;
int np;
int i;
int nit;
int k_offset;
int j;
int nthreads = 1;
int verified;
char size[(13+1)];
/* character13 */
/*

c   Because the size of the problem is too large to store in a 32-bit
c   integer for some classes, we put it into a string (for printing).
c   Have to strip off the decimal point put in there by the floating
c   point print statement (internal file)

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


float * red__sx;
float * sx__extended;
int _ti_100_0;
float * red__sy;
float * sy__extended;
float * gpu__x;
size_t pitch__x;
float * x__extended;
gpuBytes=sizeof (float);
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__an__main)), gpuBytes));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k_offset__main)), gpuBytes));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__l__main)), gpuBytes));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__np__main)), gpuBytes));
printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - EP Benchmark\n");
sprintf(size, "%12.0f", pow(2.0, (32+1)));
#pragma loop name main#0 
for (j=13; j>=1; j -- )
{
if ((size[j]=='.'))
{
size[j]=' ';
}
}
printf(" Number of random numbers generated: %13s\n", size);
verified=0;
/*

c   Compute the number of "batches" of random number pairs generated 
c   per processor. Adjust if the number of processors does not evenly 
c   divide the total number

*/
np=(1<<(32-1));
/*

c   Call the random number generator functions and initialize
c   the x-array to reduce the effects of paging on the timings.
c   Also, call all mathematical functions that are used. Make
c   sure these initializations cannot be eliminated as dead code.

*/
vranlc(0, ( & dum[0]), dum[1], ( & dum[2]));
dum[0]=randlc(( & dum[1]), dum[2]);
#pragma loop name main#1 
for (i=0; i<(2*(1<<1)); i ++ )
{
x[i]=( - 1.0E99);
}
Mops=log(sqrt(fabs(((1.0>1.0) ? 1.0 : 1.0))));
timer_clear(1);
timer_clear(2);
timer_clear(3);
timer_start(1);
vranlc(0, ( & t1), 1.220703125E9, x);
/*   Compute AN = A ^ (2 NK) (mod 2^46). */
t1=1.220703125E9;
#pragma loop name main#2 
for (i=1; i<=(1+1); i ++ )
{
t2=randlc(( & t1), t1);
}
an=t1;
tt=2.71828183E8;
gc=0.0;
sx=0.0;
sy=0.0;
#pragma loop name main#3 
for (i=0; i<=(10-1); i ++ )
{
q[i]=0.0;
}
/*

c   Each instance of this loop may be performed independently. We compute
c   the k offsets separately to take into account the fact that some nodes
c   have more numbers to generate than others

*/
k_offset=( - 1);
{
float t1;
float t2;
float t3;
float t4;
float x1;
float x2;
int kk;
int i;
int ik;
int l;
/* #pragma omp parallel for reduction(+:sx,sy) */
dim3 dimBlock0(gpuNumThreads, 1, 1);
gpuNumBlocks=((int)ceil((((float)np)/1024.0F)));
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
gpuBytes=(gpuNumBlocks*sizeof (float));
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & red__sx)), gpuBytes));
sx__extended=((float * )malloc(gpuBytes));
gpuBytes=(gpuNumBlocks*sizeof (float));
CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & red__sy)), gpuBytes));
sy__extended=((float * )malloc(gpuBytes));
gpuBytes=sizeof (float);
CUDA_SAFE_CALL(cudaMemcpy(gpu__an__main, ( & an), gpuBytes, cudaMemcpyHostToDevice));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMemcpy(gpu__k_offset__main, ( & k_offset), gpuBytes, cudaMemcpyHostToDevice));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMemcpy(gpu__l__main, ( & l), gpuBytes, cudaMemcpyHostToDevice));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMemcpy(gpu__np__main, ( & np), gpuBytes, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__x)), ( & pitch__x), (totalNumThreads*sizeof (float)), (2*(1<<1))));
gpuBytes=(pitch__x*(2*(1<<1)));
x__extended=((float * )malloc(gpuBytes));
#pragma omp parallel threadprivate(x) shared(an, k_offset, l, np) private(i, ik, k, kk, t1, t2, t3, t4, x1, x2) reduction(+: sx, sy)
#pragma cuda ainfo kernelid(0) procname(main) 
#pragma cuda gpurun cudafree(an, k_offset, l, np) 
#pragma cuda gpurun nog2cmemtr(an, k_offset, np) 
main_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(red__sx, red__sy, gpu__an__main, gpu__k_offset__main, gpu__l__main, gpu__np__main, gpu__x, pitch__x);
gpuBytes=(pitch__x*(2*(1<<1)));
CUDA_SAFE_CALL(cudaMemcpy2D(x__extended, (totalNumThreads*sizeof (float)), gpu__x, pitch__x, (totalNumThreads*sizeof (float)), (2*(1<<1)), cudaMemcpyDeviceToHost));
for (_ti_100_0=0; _ti_100_0<(2*(1<<1)); _ti_100_0 ++ )
{
x[_ti_100_0]=( * ((float * )(((char * )x__extended)+(_ti_100_0*pitch__x))));
}
free(x__extended);
CUDA_SAFE_CALL(cudaFree(gpu__x));
gpuBytes=sizeof (int);
CUDA_SAFE_CALL(cudaMemcpy(( & l), gpu__l__main, gpuBytes, cudaMemcpyDeviceToHost));
gpuBytes=(gpuNumBlocks*sizeof (float));
CUDA_SAFE_CALL(cudaMemcpy(sy__extended, red__sy, gpuBytes, cudaMemcpyDeviceToHost));
for (_ti_100_0=0; _ti_100_0<gpuNumBlocks; _ti_100_0 ++ )
{
sy+=sy__extended[_ti_100_0];
}
free(sy__extended);
CUDA_SAFE_CALL(cudaFree(red__sy));
gpuBytes=(gpuNumBlocks*sizeof (float));
CUDA_SAFE_CALL(cudaMemcpy(sx__extended, red__sx, gpuBytes, cudaMemcpyDeviceToHost));
for (_ti_100_0=0; _ti_100_0<gpuNumBlocks; _ti_100_0 ++ )
{
sx+=sx__extended[_ti_100_0];
}
free(sx__extended);
CUDA_SAFE_CALL(cudaFree(red__sx));
}
/* end of parallel region */
#pragma loop name main#5 
for (i=0; i<=(10-1); i ++ )
{
gc=(gc+q[i]);
}
timer_stop(1);
tm=timer_read(1);
nit=0;
if ((32==24))
{
if (((fabs(((sx-( - 3247.83465203474))/sx))<=1.0E-8)&&(fabs(((sy-( - 6958.407078382297))/sy))<=1.0E-8)))
{
verified=1;
}
}
else
{
if ((32==25))
{
if (((fabs(((sx-( - 2863.319731645753))/sx))<=1.0E-8)&&(fabs(((sy-( - 6320.053679109499))/sy))<=1.0E-8)))
{
verified=1;
}
}
else
{
if ((32==28))
{
if (((fabs(((sx-( - 4295.875165629892))/sx))<=1.0E-8)&&(fabs(((sy-( - 15807.32573678431))/sy))<=1.0E-8)))
{
verified=1;
}
}
else
{
if ((32==30))
{
if (((fabs(((sx-40338.15542441498)/sx))<=1.0E-8)&&(fabs(((sy-( - 26606.69192809235))/sy))<=1.0E-8)))
{
verified=1;
}
}
else
{
if ((32==32))
{
if (((fabs(((sx-47643.67927995374)/sx))<=1.0E-8)&&(fabs(((sy-( - 80840.72988043731))/sy))<=1.0E-8)))
{
verified=1;
}
}
}
}
}
}
Mops=((pow(2.0, (32+1))/tm)/1000000.0);
printf("EP Benchmark Results: \n""CPU Time = %10.4f\n""N = 2^%5d\n""No. Gaussian Pairs = %15.0f\n""Sums = %25.15e %25.15e\n""Counts:\n", tm, 32, gc, sx, sy);
#pragma loop name main#6 
for (i=0; i<=(10-1); i ++ )
{
printf("%3d %15.0f\n", i, q[i]);
}
c_print_results("EP", 'C', (32+1), 0, 0, nit, nthreads, tm, Mops, "Random numbers generated", verified, "2.3", "23 Aug 2012", "gcc", "gcc", "-lm", "-I../common", "-O3 ", "(none)", "randdp");
if ((0==1))
{
printf("Total time:     %f", timer_read(1));
printf("Gaussian pairs: %f", timer_read(2));
printf("Random numbers: %f", timer_read(3));
}
printf("/***********************/ \n/* Input Configuration */ \n/***********************/ \n");
printf("====> GPU Block Size: 1024 \n");
printf("/**********************/ \n/* Used Optimizations */ \n/**********************/ \n");
printf("====> MatrixTranspose Opt is used.\n");
printf("====> ParallelLoopSwap Opt is used.\n");
printf("====> LoopCollapse Opt is used.\n");
printf("====> Unrolling-on-reduction Opt is used.\n");
printf("====> Allocate GPU variables as global ones.\n");
printf("====> CPU-GPU Mem Transfer Opt Level: 4\n");
printf("====> Cuda Malloc Opt Level: 1\n");
printf("====> Assume that all loops have non-zero iterations.\n");
printf("====> Cache shared array elements onto GPU registers.\n");
printf("====> local array reduction variable configuration = 1\n");
CUDA_SAFE_CALL(cudaFree(gpu__an__main));
CUDA_SAFE_CALL(cudaFree(gpu__k_offset__main));
CUDA_SAFE_CALL(cudaFree(gpu__l__main));
CUDA_SAFE_CALL(cudaFree(gpu__np__main));
fflush(stdout);
fflush(stderr);
return _ret_val_0;
}

