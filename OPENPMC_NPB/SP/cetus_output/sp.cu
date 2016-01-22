/*
   --------------------------------------------------------------------

   NAS Parallel Benchmarks 2.3 OpenMP C versions - SP

   This benchmark is an OpenMP C version of the NPB SP code.

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

Author: R. Van der Wijngaart
W. Saphir

OpenMP C version: S. Satoh

--------------------------------------------------------------------
 */
#include "npb-C.h"
/* global variables */
#include "header.h"

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



double * gpu__rhs;
double * gpu__u;
double * gpu__ainv;
double * gpu__c1c2;
double * gpu__qs;
double * gpu__rho_i;
double * gpu__speed;
double * gpu__square;
double * gpu__us;
double * gpu__vs;
double * gpu__ws;
double * gpu__forcing;
double * gpu__c1;
double * gpu__c2;
double * gpu__con43;
double * gpu__dx1tx1;
double * gpu__dx2tx1;
double * gpu__dx3tx1;
double * gpu__dx4tx1;
double * gpu__dx5tx1;
double * gpu__tx2;
double * gpu__xxcon2;
double * gpu__xxcon3;
double * gpu__xxcon4;
double * gpu__xxcon5;
double * gpu__dssp;
double * gpu__dy1ty1;
double * gpu__dy2ty1;
double * gpu__dy3ty1;
double * gpu__dy4ty1;
double * gpu__dy5ty1;
double * gpu__ty2;
double * gpu__yycon2;
double * gpu__yycon3;
double * gpu__yycon4;
double * gpu__yycon5;
double * gpu__dz1tz1;
double * gpu__dz2tz1;
double * gpu__dz3tz1;
double * gpu__dz4tz1;
double * gpu__dz5tz1;
double * gpu__tz2;
double * gpu__zzcon2;
double * gpu__zzcon3;
double * gpu__zzcon4;
double * gpu__zzcon5;
double * gpu__dt;
double * gpu__lhs;
double * gpu__comz1;
double * gpu__comz4;
double * gpu__comz5;
double * gpu__comz6;
double * gpu__dttx2;
double * gpu__dtty2;
double * gpu__dttz2;
double * gpu__bt;
double * gpu__c2iv;
/* function declarations */
static void add(void );
static void add_clnd1_cloned1(void );
static void adi(void );
static void adi_clnd1_cloned1(void );
static void error_norm(double rms[5]);
static void rhs_norm(double rms[5]);
static void exact_rhs(void );
static void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
static void initialize(void );
static void lhsinit(void );
static void lhsx(void );
static void lhsx_clnd1_cloned1(void );
static void lhsy(void );
static void lhsy_clnd1_cloned1(void );
static void lhsz(void );
static void lhsz_clnd1_cloned1(void );
static void ninvr(void );
static void ninvr_clnd1_cloned1(void );
static void pinvr(void );
static void pinvr_clnd1_cloned1(void );
static void compute_rhs(void );
static void compute_rhs_clnd2(void );
static void compute_rhs_clnd1_cloned0(void );
static void set_constants(void );
static void txinvr(void );
static void txinvr_clnd1_cloned0(void );
static void tzetar(void );
static void tzetar_clnd1_cloned1(void );
static void verify(int no_time_steps, char * cclass, int * verified);
static void x_solve(void );
static void x_solve_clnd1_cloned1(void );
static void y_solve(void );
static void y_solve_clnd1_cloned1(void );
static void z_solve(void );
static void z_solve_clnd1_cloned1(void );
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

static void c_print_results(char * name, char cccclass, int n1, int n2, int n3, int niter, int nthreads, double t, double mops, char * optype, int passed_verification, char * npbversion, char * compiletime, char * cc, char * clink, char * c_lib, char * c_inc, char * cflags, char * clinkflags, char * rand)
{
	printf("\n\n %s Benchmark Completed\n", name);
	printf(" Class           =                        %c\n", cccclass);
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
   --------------------------------------------------------------------
   program SP
   c-------------------------------------------------------------------
 */
int main(int argc, char *  * argv)
{
	int niter;
	int step;
	double mflops;
	double tmax;
	int nthreads = 1;
	int verified;
	char cclass;
	FILE * fp;
	/*
	   --------------------------------------------------------------------
	   c      Read input file (if it exists), else take
	   c      defaults from parameters
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


	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__rhs)), gpuBytes));
	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ainv)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__c1c2)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__qs)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__rho_i)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__speed)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__square)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__us)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__vs)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ws)), gpuBytes));
	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__forcing)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__c1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__c2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__con43)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dx1tx1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dx2tx1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dx3tx1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dx4tx1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dx5tx1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__tx2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xxcon2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xxcon3)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xxcon4)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__xxcon5)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dssp)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dy1ty1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dy2ty1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dy3ty1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dy4ty1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dy5ty1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ty2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yycon2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yycon3)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yycon4)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__yycon5)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dz1tz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dz2tz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dz3tz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dz4tz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dz5tz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__tz2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zzcon2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zzcon3)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zzcon4)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__zzcon5)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dt)), gpuBytes));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__lhs)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__comz1)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__comz4)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__comz5)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__comz6)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dttx2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dtty2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__dttz2)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__bt)), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__c2iv)), gpuBytes));
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - SP Benchmark\n\n");
	fp=fopen("inputsp.data", "r");
	if ((fp!=((void * )0)))
	{
		printf(" Reading from input file inputsp.data\n");
		fscanf(fp, "%d", ( & niter));
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%lf", ( & dt));
		while (fgetc(fp)!='\n')
		{
			;
		}
	}
	else
	{
		printf(" No input file inputsp.data. Using compiled defaults");
		niter=400;
		dt=6.7E-4;
	}
	printf(" Size: %3dx%3dx%3d\n", 162, 162, 162);
	printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);
	if ((((162>162)||(162>162))||(162>162)))
	{
		printf("%d, %d, %d\n", 162, 162, 162);
		printf(" Problem size too big for compiled array sizes\n");
		exit(1);
	}
	set_constants();
	initialize();
	lhsinit();
	exact_rhs();
	/*
	   --------------------------------------------------------------------
	   c      do one time step to touch all code, and reinitialize
	   c-------------------------------------------------------------------
	 */
	{
		adi();
	}
	initialize();
	timer_clear(1);
	timer_start(1);
	{
#pragma loop name main#0 
		for (step=1; step<(1+niter); step ++ )
		{
			if ((((step%20)==0)||(step==1)))
			{
				printf(" Time step %4d\n", step);
			}
			adi_clnd1_cloned1();
		}
	}
	CUDA_SAFE_CALL(cudaMemcpy(rhs, gpu__rhs, gpuBytes, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(u, gpu__u, gpuBytes, cudaMemcpyDeviceToHost));
	/* end parallel */
	timer_stop(1);
	tmax=timer_read(1);
	verify(niter, ( & cclass), ( & verified));
	if ((tmax!=0))
	{
		mflops=((((((881.174*pow(((double)162), 3.0))-(4683.91*(((double)162)*((double)162))))+(11484.5*((double)162)))-19272.4)*((double)niter))/(tmax*1000000.0));
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("SP", cclass, 162, 162, 162, niter, nthreads, tmax, mflops, "          floating point", verified, "2.3", "17 Feb 2012", "gcc", "gcc", "-lm", "-I../common", "-O3 ", "(none)", "(none)");
	printf("/***********************/ \n/* Input Configuration */ \n/***********************/ \n");
	printf("====> GPU Block Size: 1024 \n");
	printf("/**********************/ \n/* Used Optimizations */ \n/**********************/ \n");
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
	CUDA_SAFE_CALL(cudaFree(gpu__rhs));
	CUDA_SAFE_CALL(cudaFree(gpu__u));
	CUDA_SAFE_CALL(cudaFree(gpu__ainv));
	CUDA_SAFE_CALL(cudaFree(gpu__c1c2));
	CUDA_SAFE_CALL(cudaFree(gpu__qs));
	CUDA_SAFE_CALL(cudaFree(gpu__rho_i));
	CUDA_SAFE_CALL(cudaFree(gpu__speed));
	CUDA_SAFE_CALL(cudaFree(gpu__square));
	CUDA_SAFE_CALL(cudaFree(gpu__us));
	CUDA_SAFE_CALL(cudaFree(gpu__vs));
	CUDA_SAFE_CALL(cudaFree(gpu__ws));
	CUDA_SAFE_CALL(cudaFree(gpu__forcing));
	CUDA_SAFE_CALL(cudaFree(gpu__c1));
	CUDA_SAFE_CALL(cudaFree(gpu__c2));
	CUDA_SAFE_CALL(cudaFree(gpu__con43));
	CUDA_SAFE_CALL(cudaFree(gpu__dx1tx1));
	CUDA_SAFE_CALL(cudaFree(gpu__dx2tx1));
	CUDA_SAFE_CALL(cudaFree(gpu__dx3tx1));
	CUDA_SAFE_CALL(cudaFree(gpu__dx4tx1));
	CUDA_SAFE_CALL(cudaFree(gpu__dx5tx1));
	CUDA_SAFE_CALL(cudaFree(gpu__tx2));
	CUDA_SAFE_CALL(cudaFree(gpu__xxcon2));
	CUDA_SAFE_CALL(cudaFree(gpu__xxcon3));
	CUDA_SAFE_CALL(cudaFree(gpu__xxcon4));
	CUDA_SAFE_CALL(cudaFree(gpu__xxcon5));
	CUDA_SAFE_CALL(cudaFree(gpu__dssp));
	CUDA_SAFE_CALL(cudaFree(gpu__dy1ty1));
	CUDA_SAFE_CALL(cudaFree(gpu__dy2ty1));
	CUDA_SAFE_CALL(cudaFree(gpu__dy3ty1));
	CUDA_SAFE_CALL(cudaFree(gpu__dy4ty1));
	CUDA_SAFE_CALL(cudaFree(gpu__dy5ty1));
	CUDA_SAFE_CALL(cudaFree(gpu__ty2));
	CUDA_SAFE_CALL(cudaFree(gpu__yycon2));
	CUDA_SAFE_CALL(cudaFree(gpu__yycon3));
	CUDA_SAFE_CALL(cudaFree(gpu__yycon4));
	CUDA_SAFE_CALL(cudaFree(gpu__yycon5));
	CUDA_SAFE_CALL(cudaFree(gpu__dz1tz1));
	CUDA_SAFE_CALL(cudaFree(gpu__dz2tz1));
	CUDA_SAFE_CALL(cudaFree(gpu__dz3tz1));
	CUDA_SAFE_CALL(cudaFree(gpu__dz4tz1));
	CUDA_SAFE_CALL(cudaFree(gpu__dz5tz1));
	CUDA_SAFE_CALL(cudaFree(gpu__tz2));
	CUDA_SAFE_CALL(cudaFree(gpu__zzcon2));
	CUDA_SAFE_CALL(cudaFree(gpu__zzcon3));
	CUDA_SAFE_CALL(cudaFree(gpu__zzcon4));
	CUDA_SAFE_CALL(cudaFree(gpu__zzcon5));
	CUDA_SAFE_CALL(cudaFree(gpu__dt));
	CUDA_SAFE_CALL(cudaFree(gpu__lhs));
	CUDA_SAFE_CALL(cudaFree(gpu__comz1));
	CUDA_SAFE_CALL(cudaFree(gpu__comz4));
	CUDA_SAFE_CALL(cudaFree(gpu__comz5));
	CUDA_SAFE_CALL(cudaFree(gpu__comz6));
	CUDA_SAFE_CALL(cudaFree(gpu__dttx2));
	CUDA_SAFE_CALL(cudaFree(gpu__dtty2));
	CUDA_SAFE_CALL(cudaFree(gpu__dttz2));
	CUDA_SAFE_CALL(cudaFree(gpu__bt));
	CUDA_SAFE_CALL(cudaFree(gpu__c2iv));
	fflush(stdout);
	fflush(stderr);
	return _ret_val_0;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void add_kernel0(double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name add#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name add#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name add#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					u[m][i][j][k]=(u[m][i][j][k]+rhs[m][i][j][k]);
				}
			}
		}
	}
}

static void add(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c addition of update to the vector u
	   c-------------------------------------------------------------------
	 */
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
#pragma omp parallel for shared(rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(rhs, u) 
#pragma cuda gpurun nocudamalloc(rhs, u) 
#pragma cuda gpurun nocudafree(rhs, u) 
#pragma cuda gpurun nog2cmemtr(rhs, u) 
#pragma cuda ainfo kernelid(0) procname(add) 
	add_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	return ;
}

__global__ void add_clnd1_cloned1_kernel0(double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name add#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name add#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name add#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					u[m][i][j][k]=(u[m][i][j][k]+rhs[m][i][j][k]);
				}
			}
		}
	}
}

static void add_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c addition of update to the vector u
	   c-------------------------------------------------------------------
	 */
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
#pragma omp parallel for shared(rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(rhs, u) 
#pragma cuda gpurun nocudamalloc(rhs, u) 
#pragma cuda gpurun nocudafree(rhs, u) 
#pragma cuda gpurun nog2cmemtr(rhs) 
#pragma cuda ainfo kernelid(0) procname(add_clnd1_cloned1) 
	add_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void adi(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	compute_rhs();
	txinvr();
	x_solve();
	y_solve();
	z_solve();
	add();
	return ;
}

static void adi_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	compute_rhs_clnd1_cloned0();
	txinvr_clnd1_cloned0();
	x_solve_clnd1_cloned1();
	y_solve_clnd1_cloned1();
	z_solve_clnd1_cloned1();
	add_clnd1_cloned1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void error_norm(double rms[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function computes the norm of the difference between the
	   c computed solution and the exact solution
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int d;
	double xi;
	double eta;
	double zeta;
	double u_exact[5];
	double add;
#pragma loop name error_norm#0 
	for (m=0; m<5; m ++ )
	{
		rms[m]=0.0;
	}
#pragma loop name error_norm#1 
	for (i=0; i<((1+162)-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name error_norm#1#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
			eta=(((double)j)*dnym1);
#pragma loop name error_norm#1#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, u_exact);
#pragma loop name error_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=(u[m][i][j][k]-u_exact[m]);
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
#pragma loop name error_norm#2 
	for (m=0; m<5; m ++ )
	{
#pragma loop name error_norm#2#0 
		for (d=0; d<3; d ++ )
		{
			rms[m]=(rms[m]/((double)(grid_points[d]-2)));
		}
		rms[m]=sqrt(rms[m]);
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void rhs_norm(double rms[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int d;
	int m;
	double add;
#pragma loop name rhs_norm#0 
	for (m=0; m<5; m ++ )
	{
		rms[m]=0.0;
	}
#pragma loop name rhs_norm#1 
	for (i=0; i<((1+162)-2); i ++ )
	{
#pragma loop name rhs_norm#1#0 
		for (j=0; j<((1+162)-2); j ++ )
		{
#pragma loop name rhs_norm#1#0#0 
			for (k=0; k<((1+162)-2); k ++ )
			{
#pragma loop name rhs_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=rhs[m][i][j][k];
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
#pragma loop name rhs_norm#2 
	for (m=0; m<5; m ++ )
	{
#pragma loop name rhs_norm#2#0 
		for (d=0; d<3; d ++ )
		{
			rms[m]=(rms[m]/((double)(grid_points[d]-2)));
		}
		rms[m]=sqrt(rms[m]);
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void exact_rhs(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c compute the right hand side based on exact solution
	   c-------------------------------------------------------------------
	 */
	double dtemp[5];
	double xi;
	double eta;
	double zeta;
	double dtpp;
	int m;
	int i;
	int j;
	int k;
	int ip1;
	int im1;
	int jp1;
	int jm1;
	int km1;
	int kp1;
	/*
	   --------------------------------------------------------------------
	   c      initialize                                  
	   c-------------------------------------------------------------------
	 */
#pragma loop name exact_rhs#0 
	for (m=0; m<5; m ++ )
	{
#pragma loop name exact_rhs#0#0 
		for (i=0; i<((1+162)-1); i ++ )
		{
#pragma loop name exact_rhs#0#0#0 
			for (j=0; j<((1+162)-1); j ++ )
			{
#pragma loop name exact_rhs#0#0#0#0 
				for (k=0; k<((1+162)-1); k ++ )
				{
					forcing[m][i][j][k]=0.0;
				}
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      xi-direction flux differences                      
	   c-------------------------------------------------------------------
	 */
#pragma loop name exact_rhs#1 
	for (k=1; k<((1+162)-2); k ++ )
	{
		zeta=(((double)k)*dnzm1);
#pragma loop name exact_rhs#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			eta=(((double)j)*dnym1);
#pragma loop name exact_rhs#1#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				xi=(((double)i)*dnxm1);
				exact_solution(xi, eta, zeta, dtemp);
#pragma loop name exact_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][i]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
#pragma loop name exact_rhs#1#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][i]=(dtpp*dtemp[m]);
				}
				cuf[i]=(buf[1][i]*buf[1][i]);
				buf[0][i]=((cuf[i]+(buf[2][i]*buf[2][i]))+(buf[3][i]*buf[3][i]));
				q[i]=(0.5*(((buf[1][i]*ue[1][i])+(buf[2][i]*ue[2][i]))+(buf[3][i]*ue[3][i])));
			}
#pragma loop name exact_rhs#1#0#1 
			for (i=1; i<((1+162)-2); i ++ )
			{
				im1=(i-1);
				ip1=(i+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(tx2*(ue[1][ip1]-ue[1][im1])))+(dx1tx1*((ue[0][ip1]-(2.0*ue[0][i]))+ue[0][im1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(tx2*(((ue[1][ip1]*buf[1][ip1])+(c2*(ue[4][ip1]-q[ip1])))-((ue[1][im1]*buf[1][im1])+(c2*(ue[4][im1]-q[im1]))))))+(xxcon1*((buf[1][ip1]-(2.0*buf[1][i]))+buf[1][im1])))+(dx2tx1*((ue[1][ip1]-(2.0*ue[1][i]))+ue[1][im1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(tx2*((ue[2][ip1]*buf[1][ip1])-(ue[2][im1]*buf[1][im1]))))+(xxcon2*((buf[2][ip1]-(2.0*buf[2][i]))+buf[2][im1])))+(dx3tx1*((ue[2][ip1]-(2.0*ue[2][i]))+ue[2][im1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(tx2*((ue[3][ip1]*buf[1][ip1])-(ue[3][im1]*buf[1][im1]))))+(xxcon2*((buf[3][ip1]-(2.0*buf[3][i]))+buf[3][im1])))+(dx4tx1*((ue[3][ip1]-(2.0*ue[3][i]))+ue[3][im1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(tx2*((buf[1][ip1]*((c1*ue[4][ip1])-(c2*q[ip1])))-(buf[1][im1]*((c1*ue[4][im1])-(c2*q[im1]))))))+((0.5*xxcon3)*((buf[0][ip1]-(2.0*buf[0][i]))+buf[0][im1])))+(xxcon4*((cuf[ip1]-(2.0*cuf[i]))+cuf[im1])))+(xxcon5*((buf[4][ip1]-(2.0*buf[4][i]))+buf[4][im1])))+(dx5tx1*((ue[4][ip1]-(2.0*ue[4][i]))+ue[4][im1])));
			}
			/*
			   --------------------------------------------------------------------
			   c            Fourth-order dissipation                         
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#1#0#2 
			for (m=0; m<5; m ++ )
			{
				i=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][i])-(4.0*ue[m][(i+1)]))+ue[m][(i+2)])));
				i=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][(i-1)])+(6.0*ue[m][i]))-(4.0*ue[m][(i+1)]))+ue[m][(i+2)])));
			}
#pragma loop name exact_rhs#1#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#1#0#3#0 
				for (i=3; i<((1+162)-4); i ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][(i-2)]-(4.0*ue[m][(i-1)]))+(6.0*ue[m][i]))-(4.0*ue[m][(i+1)]))+ue[m][(i+2)])));
				}
			}
#pragma loop name exact_rhs#1#0#4 
			for (m=0; m<5; m ++ )
			{
				i=(162-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][(i-2)]-(4.0*ue[m][(i-1)]))+(6.0*ue[m][i]))-(4.0*ue[m][(i+1)]))));
				i=(162-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][(i-2)]-(4.0*ue[m][(i-1)]))+(5.0*ue[m][i]))));
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c  eta-direction flux differences             
	   c-------------------------------------------------------------------
	 */
#pragma loop name exact_rhs#2 
	for (k=1; k<((1+162)-2); k ++ )
	{
		zeta=(((double)k)*dnzm1);
#pragma loop name exact_rhs#2#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			xi=(((double)i)*dnxm1);
#pragma loop name exact_rhs#2#0#0 
			for (j=0; j<((1+162)-1); j ++ )
			{
				eta=(((double)j)*dnym1);
				exact_solution(xi, eta, zeta, dtemp);
#pragma loop name exact_rhs#2#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][j]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
#pragma loop name exact_rhs#2#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][j]=(dtpp*dtemp[m]);
				}
				cuf[j]=(buf[2][j]*buf[2][j]);
				buf[0][j]=((cuf[j]+(buf[1][j]*buf[1][j]))+(buf[3][j]*buf[3][j]));
				q[j]=(0.5*(((buf[1][j]*ue[1][j])+(buf[2][j]*ue[2][j]))+(buf[3][j]*ue[3][j])));
			}
#pragma loop name exact_rhs#2#0#1 
			for (j=1; j<((1+162)-2); j ++ )
			{
				jm1=(j-1);
				jp1=(j+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(ty2*(ue[2][jp1]-ue[2][jm1])))+(dy1ty1*((ue[0][jp1]-(2.0*ue[0][j]))+ue[0][jm1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(ty2*((ue[1][jp1]*buf[2][jp1])-(ue[1][jm1]*buf[2][jm1]))))+(yycon2*((buf[1][jp1]-(2.0*buf[1][j]))+buf[1][jm1])))+(dy2ty1*((ue[1][jp1]-(2.0*ue[1][j]))+ue[1][jm1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(ty2*(((ue[2][jp1]*buf[2][jp1])+(c2*(ue[4][jp1]-q[jp1])))-((ue[2][jm1]*buf[2][jm1])+(c2*(ue[4][jm1]-q[jm1]))))))+(yycon1*((buf[2][jp1]-(2.0*buf[2][j]))+buf[2][jm1])))+(dy3ty1*((ue[2][jp1]-(2.0*ue[2][j]))+ue[2][jm1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(ty2*((ue[3][jp1]*buf[2][jp1])-(ue[3][jm1]*buf[2][jm1]))))+(yycon2*((buf[3][jp1]-(2.0*buf[3][j]))+buf[3][jm1])))+(dy4ty1*((ue[3][jp1]-(2.0*ue[3][j]))+ue[3][jm1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(ty2*((buf[2][jp1]*((c1*ue[4][jp1])-(c2*q[jp1])))-(buf[2][jm1]*((c1*ue[4][jm1])-(c2*q[jm1]))))))+((0.5*yycon3)*((buf[0][jp1]-(2.0*buf[0][j]))+buf[0][jm1])))+(yycon4*((cuf[jp1]-(2.0*cuf[j]))+cuf[jm1])))+(yycon5*((buf[4][jp1]-(2.0*buf[4][j]))+buf[4][jm1])))+(dy5ty1*((ue[4][jp1]-(2.0*ue[4][j]))+ue[4][jm1])));
			}
			/*
			   --------------------------------------------------------------------
			   c            Fourth-order dissipation                      
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#2#0#2 
			for (m=0; m<5; m ++ )
			{
				j=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][j])-(4.0*ue[m][(j+1)]))+ue[m][(j+2)])));
				j=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][(j-1)])+(6.0*ue[m][j]))-(4.0*ue[m][(j+1)]))+ue[m][(j+2)])));
			}
#pragma loop name exact_rhs#2#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#2#0#3#0 
				for (j=3; j<((1+162)-4); j ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][(j-2)]-(4.0*ue[m][(j-1)]))+(6.0*ue[m][j]))-(4.0*ue[m][(j+1)]))+ue[m][(j+2)])));
				}
			}
#pragma loop name exact_rhs#2#0#4 
			for (m=0; m<5; m ++ )
			{
				j=(162-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][(j-2)]-(4.0*ue[m][(j-1)]))+(6.0*ue[m][j]))-(4.0*ue[m][(j+1)]))));
				j=(162-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][(j-2)]-(4.0*ue[m][(j-1)]))+(5.0*ue[m][j]))));
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      zeta-direction flux differences                      
	   c-------------------------------------------------------------------
	 */
#pragma loop name exact_rhs#3 
	for (j=1; j<((1+162)-2); j ++ )
	{
		eta=(((double)j)*dnym1);
#pragma loop name exact_rhs#3#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			xi=(((double)i)*dnxm1);
#pragma loop name exact_rhs#3#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, dtemp);
#pragma loop name exact_rhs#3#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][k]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
#pragma loop name exact_rhs#3#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][k]=(dtpp*dtemp[m]);
				}
				cuf[k]=(buf[3][k]*buf[3][k]);
				buf[0][k]=((cuf[k]+(buf[1][k]*buf[1][k]))+(buf[2][k]*buf[2][k]));
				q[k]=(0.5*(((buf[1][k]*ue[1][k])+(buf[2][k]*ue[2][k]))+(buf[3][k]*ue[3][k])));
			}
#pragma loop name exact_rhs#3#0#1 
			for (k=1; k<((1+162)-2); k ++ )
			{
				km1=(k-1);
				kp1=(k+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(tz2*(ue[3][kp1]-ue[3][km1])))+(dz1tz1*((ue[0][kp1]-(2.0*ue[0][k]))+ue[0][km1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(tz2*((ue[1][kp1]*buf[3][kp1])-(ue[1][km1]*buf[3][km1]))))+(zzcon2*((buf[1][kp1]-(2.0*buf[1][k]))+buf[1][km1])))+(dz2tz1*((ue[1][kp1]-(2.0*ue[1][k]))+ue[1][km1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(tz2*((ue[2][kp1]*buf[3][kp1])-(ue[2][km1]*buf[3][km1]))))+(zzcon2*((buf[2][kp1]-(2.0*buf[2][k]))+buf[2][km1])))+(dz3tz1*((ue[2][kp1]-(2.0*ue[2][k]))+ue[2][km1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(tz2*(((ue[3][kp1]*buf[3][kp1])+(c2*(ue[4][kp1]-q[kp1])))-((ue[3][km1]*buf[3][km1])+(c2*(ue[4][km1]-q[km1]))))))+(zzcon1*((buf[3][kp1]-(2.0*buf[3][k]))+buf[3][km1])))+(dz4tz1*((ue[3][kp1]-(2.0*ue[3][k]))+ue[3][km1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(tz2*((buf[3][kp1]*((c1*ue[4][kp1])-(c2*q[kp1])))-(buf[3][km1]*((c1*ue[4][km1])-(c2*q[km1]))))))+((0.5*zzcon3)*((buf[0][kp1]-(2.0*buf[0][k]))+buf[0][km1])))+(zzcon4*((cuf[kp1]-(2.0*cuf[k]))+cuf[km1])))+(zzcon5*((buf[4][kp1]-(2.0*buf[4][k]))+buf[4][km1])))+(dz5tz1*((ue[4][kp1]-(2.0*ue[4][k]))+ue[4][km1])));
			}
			/*
			   --------------------------------------------------------------------
			   c            Fourth-order dissipation                        
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#3#0#2 
			for (m=0; m<5; m ++ )
			{
				k=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][k])-(4.0*ue[m][(k+1)]))+ue[m][(k+2)])));
				k=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][(k-1)])+(6.0*ue[m][k]))-(4.0*ue[m][(k+1)]))+ue[m][(k+2)])));
			}
#pragma loop name exact_rhs#3#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#3#0#3#0 
				for (k=3; k<((1+162)-4); k ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][(k-2)]-(4.0*ue[m][(k-1)]))+(6.0*ue[m][k]))-(4.0*ue[m][(k+1)]))+ue[m][(k+2)])));
				}
			}
#pragma loop name exact_rhs#3#0#4 
			for (m=0; m<5; m ++ )
			{
				k=(162-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][(k-2)]-(4.0*ue[m][(k-1)]))+(6.0*ue[m][k]))-(4.0*ue[m][(k+1)]))));
				k=(162-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][(k-2)]-(4.0*ue[m][(k-1)]))+(5.0*ue[m][k]))));
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c now change the sign of the forcing function, 
	   c-------------------------------------------------------------------
	 */
#pragma loop name exact_rhs#4 
	for (m=0; m<5; m ++ )
	{
#pragma loop name exact_rhs#4#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name exact_rhs#4#0#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
#pragma loop name exact_rhs#4#0#0#0 
				for (k=1; k<((1+162)-2); k ++ )
				{
					forcing[m][i][j][k]=(( - 1.0)*forcing[m][i][j][k]);
				}
			}
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void exact_solution(double xi, double eta, double zeta, double dtemp[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function returns the exact solution at point xi, eta, zeta  
	   c-------------------------------------------------------------------
	 */
	int m;
#pragma loop name exact_solution#0 
	for (m=0; m<5; m ++ )
	{
		dtemp[m]=(((ce[0][m]+(xi*(ce[1][m]+(xi*(ce[4][m]+(xi*(ce[7][m]+(xi*ce[10][m]))))))))+(eta*(ce[2][m]+(eta*(ce[5][m]+(eta*(ce[8][m]+(eta*ce[11][m]))))))))+(zeta*(ce[3][m]+(zeta*(ce[6][m]+(zeta*(ce[9][m]+(zeta*ce[12][m]))))))));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void initialize(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This subroutine initializes the field variable u using 
	   c tri-linear transfinite interpolation of the boundary values     
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int ix;
	int iy;
	int iz;
	double xi;
	double eta;
	double zeta;
	double Pface[2][3][5];
	double Pxi;
	double Peta;
	double Pzeta;
	double temp[5];
	/*
	   --------------------------------------------------------------------
	   c  Later (in compute_rhs) we compute 1u for every element. A few of 
	   c  the corner elements are not used, but it convenient (and faster) 
	   c  to compute the whole thing with a simple loop. Make sure those 
	   c  values are nonzero by initializing the whole thing here. 
	   c-------------------------------------------------------------------
	 */
#pragma loop name initialize#0 
	for (i=0; i<((1+162)-1); i ++ )
	{
#pragma loop name initialize#0#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name initialize#0#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				u[0][i][j][k]=1.0;
				u[1][i][j][k]=0.0;
				u[2][i][j][k]=0.0;
				u[3][i][j][k]=0.0;
				u[4][i][j][k]=1.0;
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c first store the "interpolated" values everywhere on the grid    
	   c-------------------------------------------------------------------
	 */
#pragma loop name initialize#1 
	for (i=0; i<((1+162)-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name initialize#1#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
			eta=(((double)j)*dnym1);
#pragma loop name initialize#1#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
#pragma loop name initialize#1#0#0#0 
				for (ix=0; ix<2; ix ++ )
				{
					exact_solution(((double)ix), eta, zeta, ( & Pface[ix][0][0]));
				}
#pragma loop name initialize#1#0#0#1 
				for (iy=0; iy<2; iy ++ )
				{
					exact_solution(xi, ((double)iy), zeta, ( & Pface[iy][1][0]));
				}
#pragma loop name initialize#1#0#0#2 
				for (iz=0; iz<2; iz ++ )
				{
					exact_solution(xi, eta, ((double)iz), ( & Pface[iz][2][0]));
				}
#pragma loop name initialize#1#0#0#3 
				for (m=0; m<5; m ++ )
				{
					Pxi=((xi*Pface[1][0][m])+((1.0-xi)*Pface[0][0][m]));
					Peta=((eta*Pface[1][1][m])+((1.0-eta)*Pface[0][1][m]));
					Pzeta=((zeta*Pface[1][2][m])+((1.0-zeta)*Pface[0][2][m]));
					u[m][i][j][k]=((((((Pxi+Peta)+Pzeta)-(Pxi*Peta))-(Pxi*Pzeta))-(Peta*Pzeta))+((Pxi*Peta)*Pzeta));
				}
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c now store the exact values on the boundaries        
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c west face                                                  
	   c-------------------------------------------------------------------
	 */
	xi=0.0;
	i=0;
#pragma loop name initialize#2 
	for (j=0; j<162; j ++ )
	{
		eta=(((double)j)*dnym1);
#pragma loop name initialize#2#0 
		for (k=0; k<162; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#2#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c east face                                                      
	   c-------------------------------------------------------------------
	 */
	xi=1.0;
	i=(162-1);
#pragma loop name initialize#3 
	for (j=0; j<162; j ++ )
	{
		eta=(((double)j)*dnym1);
#pragma loop name initialize#3#0 
		for (k=0; k<162; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#3#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c south face                                                 
	   c-------------------------------------------------------------------
	 */
	eta=0.0;
	j=0;
#pragma loop name initialize#4 
	for (i=0; i<162; i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name initialize#4#0 
		for (k=0; k<162; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#4#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c north face                                    
	   c-------------------------------------------------------------------
	 */
	eta=1.0;
	j=(162-1);
#pragma loop name initialize#5 
	for (i=0; i<162; i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name initialize#5#0 
		for (k=0; k<162; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#5#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c bottom face                                       
	   c-------------------------------------------------------------------
	 */
	zeta=0.0;
	k=0;
#pragma loop name initialize#6 
	for (i=0; i<162; i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name initialize#6#0 
		for (j=0; j<162; j ++ )
		{
			eta=(((double)j)*dnym1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#6#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c top face     
	   c-------------------------------------------------------------------
	 */
	zeta=1.0;
	k=(162-1);
#pragma loop name initialize#7 
	for (i=0; i<162; i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name initialize#7#0 
		for (j=0; j<162; j ++ )
		{
			eta=(((double)j)*dnym1);
			exact_solution(xi, eta, zeta, temp);
#pragma loop name initialize#7#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsinit_kernel0(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<162)
	{
#pragma loop name lhsinit#0#0 
		for (j=0; j<162; j ++ )
		{
#pragma loop name lhsinit#0#0#0 
			for (i=0; i<162; i ++ )
			{
#pragma loop name lhsinit#0#0#0#0 
				for (n=0; n<15; n ++ )
				{
					lhs[n][i][j][k]=0.0;
				}
			}
		}
	}
}

__global__ void lhsinit_kernel1(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<162)
	{
#pragma loop name lhsinit#1#0 
		for (j=0; j<162; j ++ )
		{
#pragma loop name lhsinit#1#0#0 
			for (i=0; i<162; i ++ )
			{
#pragma loop name lhsinit#1#0#0#0 
				for (n=0; n<3; n ++ )
				{
					lhs[((5*n)+2)][i][j][k]=1.0;
				}
			}
		}
	}
}

static void lhsinit(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     zap the whole left hand side for starters
	   c-------------------------------------------------------------------
	 */
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
#pragma omp parallel for shared(lhs) private(i, j, k, n)
#pragma cuda gpurun noc2gmemtr(lhs) nog2cmemtr(lhs) 
#pragma cuda gpurun nocudafree(lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsinit) 
	lhsinit_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* #pragma omp barrier   */
	/*
	   --------------------------------------------------------------------
	   c      next, set all diagonal values to 1. This is overkill, but 
	   c      convenient
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(lhs) private(i, j, k, n)
#pragma cuda gpurun noc2gmemtr(lhs) nog2cmemtr(lhs) 
#pragma cuda gpurun nocudamalloc(lhs) 
#pragma cuda gpurun nocudafree(lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsinit) 
	lhsinit_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsx_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * i, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[2][i_0][j][k]=(lhs[2][i_0][j][k]+( * comz5));
			lhs[3][i_0][j][k]=(lhs[3][i_0][j][k]-comz4_0);
			lhs[4][i_0][j][k]=(lhs[4][i_0][j][k]+comz1_0);
			lhs[1][(i_0+1)][j][k]=(lhs[1][(i_0+1)][j][k]-comz4_0);
			lhs[2][(i_0+1)][j][k]=(lhs[2][(i_0+1)][j][k]+( * comz6));
			lhs[3][(i_0+1)][j][k]=(lhs[3][(i_0+1)][j][k]-comz4_0);
			lhs[4][(i_0+1)][j][k]=(lhs[4][(i_0+1)][j][k]+comz1_0);
		}
	}
}

__global__ void lhsx_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsx#2#0#0 
			for (i=3; i<((1+162)-4); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsx_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * i, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[0][i_0][j][k]=(lhs[0][i_0][j][k]+comz1_0);
			lhs[1][i_0][j][k]=(lhs[1][i_0][j][k]-comz4_0);
			lhs[2][i_0][j][k]=(lhs[2][i_0][j][k]+( * comz6));
			lhs[3][i_0][j][k]=(lhs[3][i_0][j][k]-comz4_0);
			lhs[0][(i_0+1)][j][k]=(lhs[0][(i_0+1)][j][k]+comz1_0);
			lhs[1][(i_0+1)][j][k]=(lhs[1][(i_0+1)][j][k]-comz4_0);
			lhs[2][(i_0+1)][j][k]=(lhs[2][(i_0+1)][j][k]+( * comz5));
		}
	}
}

__global__ void lhsx_kernel3(double * dttx2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dttx2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dttx2_0=( * dttx2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsx#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[(i+1)][j][k];
				speed_0=speed[(i-1)][j][k];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dttx2_0*speed_0));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dttx2_0*speed_1));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dttx2_0*speed_0));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dttx2_0*speed_1));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsx(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three x-factors  
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      first fill the lhs for the u-eigenvalue                   
	   c-------------------------------------------------------------------
	 */
	int * gpu__i;
#pragma loop name lhsx#0 
	for (j=1; j<((1+162)-2); j ++ )
	{
#pragma loop name lhsx#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			/* 			#pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsx#0#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[i]=us[i][j][k];
				rhon[i]=(((dx2+(con43*ru1))>(((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1))) ? (dx2+(con43*ru1)) : (((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)));
			}
			/* trace_stop("lhsx", 1); */
			/* trace_start("lhsx", 2); */
			/* 			#pragma omp parallel for private(i) schedule(static) */
#pragma loop name lhsx#0#0#1 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttx2)*cv[(i-1)])-(dttx1*rhon[(i-1)]));
				lhs[2][i][j][k]=(1.0+(c2dttx1*rhon[i]));
				lhs[3][i][j][k]=((dttx2*cv[(i+1)])-(dttx1*rhon[(i+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsx", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                             
	   c-------------------------------------------------------------------
	 */
	i=1;
	/* #pragma omp for nowait */
	/* trace_start("lhsx", 3); */
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
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__comz1, ( & comz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__comz4, ( & comz4), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__comz5, ( & comz5), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__comz6, ( & comz6), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, i, lhs) private(j, k) schedule(static)
#pragma cuda gpurun nocudamalloc(lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsx) 
#pragma cuda gpurun registerRO(comz1, comz4, i) 
	lhsx_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsx", 3); */
	/* #pragma omp for nowait */
	/* trace_start("lhsx", 4); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsx) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsx_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsx", 4); */
	i=(162-3);
	/* #pragma omp for   */
	/* trace_start("lhsx", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, i, lhs) private(j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsx) 
#pragma cuda gpurun registerRO(comz1, comz4, i) 
#pragma cuda gpurun cudafree(i) 
	lhsx_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/*  trace_stop("lhsx", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, fill the other factors (u+c), (u-c) by adding to 
	   c      the first  
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsx", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dttx2, ( & dttx2), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dttx2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, speed) 
#pragma cuda gpurun nocudamalloc(lhs, speed) 
#pragma cuda gpurun nocudafree(dttx2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dttx2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsx) 
#pragma cuda gpurun registerRO(dttx2, speed[(i+1)][j][k], speed[(i-1)][j][k]) 
	lhsx_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dttx2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsx", 6); */
	return ;
}

__global__ void lhsx_clnd1_cloned1_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * i, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[2][i_0][j][k]=(lhs[2][i_0][j][k]+( * comz5));
			lhs[3][i_0][j][k]=(lhs[3][i_0][j][k]-comz4_0);
			lhs[4][i_0][j][k]=(lhs[4][i_0][j][k]+comz1_0);
			lhs[1][(i_0+1)][j][k]=(lhs[1][(i_0+1)][j][k]-comz4_0);
			lhs[2][(i_0+1)][j][k]=(lhs[2][(i_0+1)][j][k]+( * comz6));
			lhs[3][(i_0+1)][j][k]=(lhs[3][(i_0+1)][j][k]-comz4_0);
			lhs[4][(i_0+1)][j][k]=(lhs[4][(i_0+1)][j][k]+comz1_0);
		}
	}
}

__global__ void lhsx_clnd1_cloned1_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsx#2#0#0 
			for (i=3; i<((1+162)-4); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsx_clnd1_cloned1_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * i, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	comz1_0=( * comz1);
	comz4_0=( * comz4);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[0][i_0][j][k]=(lhs[0][i_0][j][k]+comz1_0);
			lhs[1][i_0][j][k]=(lhs[1][i_0][j][k]-comz4_0);
			lhs[2][i_0][j][k]=(lhs[2][i_0][j][k]+( * comz6));
			lhs[3][i_0][j][k]=(lhs[3][i_0][j][k]-comz4_0);
			lhs[0][(i_0+1)][j][k]=(lhs[0][(i_0+1)][j][k]+comz1_0);
			lhs[1][(i_0+1)][j][k]=(lhs[1][(i_0+1)][j][k]-comz4_0);
			lhs[2][(i_0+1)][j][k]=(lhs[2][(i_0+1)][j][k]+( * comz5));
		}
	}
}

__global__ void lhsx_clnd1_cloned1_kernel3(double * dttx2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dttx2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dttx2_0=( * dttx2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsx#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsx#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[(i+1)][j][k];
				speed_0=speed[(i-1)][j][k];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dttx2_0*speed_0));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dttx2_0*speed_1));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dttx2_0*speed_0));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dttx2_0*speed_1));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsx_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three x-factors  
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      first fill the lhs for the u-eigenvalue                   
	   c-------------------------------------------------------------------
	 */
	int * gpu__i;
#pragma loop name lhsx#0 
	for (j=1; j<((1+162)-2); j ++ )
	{
#pragma loop name lhsx#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			/* 			#pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsx#0#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[i]=us[i][j][k];
				rhon[i]=(((dx2+(con43*ru1))>(((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1))) ? (dx2+(con43*ru1)) : (((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)));
			}
			/* trace_stop("lhsx", 1); */
			/* trace_start("lhsx", 2); */
			/* 			#pragma omp parallel for private(i) schedule(static) */
#pragma loop name lhsx#0#0#1 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttx2)*cv[(i-1)])-(dttx1*rhon[(i-1)]));
				lhs[2][i][j][k]=(1.0+(c2dttx1*rhon[i]));
				lhs[3][i][j][k]=((dttx2*cv[(i+1)])-(dttx1*rhon[(i+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsx", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                             
	   c-------------------------------------------------------------------
	 */
	i=1;
	/* #pragma omp for nowait */
	/* trace_start("lhsx", 3); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, i, lhs) private(j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsx_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, i) 
	lhsx_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsx", 3); */
	/* #pragma omp for nowait */
	/* trace_start("lhsx", 4); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsx_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsx_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsx", 4); */
	i=(162-3);
	/* #pragma omp for   */
	/* trace_start("lhsx", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, i, lhs) private(j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, i, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsx_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, i) 
#pragma cuda gpurun cudafree(i) 
	lhsx_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/*  trace_stop("lhsx", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, fill the other factors (u+c), (u-c) by adding to 
	   c      the first  
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsx", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dttx2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(dttx2, lhs, speed) 
#pragma cuda gpurun nocudamalloc(dttx2, lhs, speed) 
#pragma cuda gpurun nocudafree(dttx2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dttx2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsx_clnd1_cloned1) 
#pragma cuda gpurun registerRO(dttx2, speed[(i+1)][j][k], speed[(i-1)][j][k]) 
	lhsx_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dttx2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsx", 6); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsy_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * j, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int j_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#1#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			lhs[2][i][j_0][k]=(lhs[2][i][j_0][k]+( * comz5));
			lhs[3][i][j_0][k]=(lhs[3][i][j_0][k]-comz4_0);
			lhs[4][i][j_0][k]=(lhs[4][i][j_0][k]+comz1_0);
			lhs[1][i][(j_0+1)][k]=(lhs[1][i][(j_0+1)][k]-comz4_0);
			lhs[2][i][(j_0+1)][k]=(lhs[2][i][(j_0+1)][k]+( * comz6));
			lhs[3][i][(j_0+1)][k]=(lhs[3][i][(j_0+1)][k]-comz4_0);
			lhs[4][i][(j_0+1)][k]=(lhs[4][i][(j_0+1)][k]+comz1_0);
		}
	}
}

__global__ void lhsy_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#2#0 
		for (j=3; j<((1+162)-4); j ++ )
		{
#pragma loop name lhsy#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsy_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * j, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int j_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#3#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			lhs[0][i][j_0][k]=(lhs[0][i][j_0][k]+comz1_0);
			lhs[1][i][j_0][k]=(lhs[1][i][j_0][k]-comz4_0);
			lhs[2][i][j_0][k]=(lhs[2][i][j_0][k]+( * comz6));
			lhs[3][i][j_0][k]=(lhs[3][i][j_0][k]-comz4_0);
			lhs[0][i][(j_0+1)][k]=(lhs[0][i][(j_0+1)][k]+comz1_0);
			lhs[1][i][(j_0+1)][k]=(lhs[1][i][(j_0+1)][k]-comz4_0);
			lhs[2][i][(j_0+1)][k]=(lhs[2][i][(j_0+1)][k]+( * comz5));
		}
	}
}

__global__ void lhsy_kernel3(double * dtty2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dtty2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dtty2_0=( * dtty2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsy#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[i][(j+1)][k];
				speed_0=speed[i][(j-1)][k];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dtty2_0*speed_0));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dtty2_0*speed_1));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dtty2_0*speed_0));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dtty2_0*speed_1));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsy(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three y-factors   
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      first fill the lhs for the u-eigenvalue         
	   c-------------------------------------------------------------------
	 */
	int * gpu__j;
#pragma loop name lhsy#0 
	for (i=1; i<((1+162)-2); i ++ )
	{
#pragma loop name lhsy#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			/* trace_start("lhsy", 1); */
			/* #pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsy#0#0#0 
			for (j=0; j<((1+162)-1); j ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[j]=vs[i][j][k];
				rhoq[j]=(((dy3+(con43*ru1))>(((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1))) ? (dy3+(con43*ru1)) : (((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)));
			}
			/* trace_stop("lhsy", 1); */
			/* trace_start("lhsy", 2); */
			/* #pragma omp parallel for private(j) schedule(static) */
#pragma loop name lhsy#0#0#1 
			for (j=1; j<((1+162)-2); j ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dtty2)*cv[(j-1)])-(dtty1*rhoq[(j-1)]));
				lhs[2][i][j][k]=(1.0+(c2dtty1*rhoq[j]));
				lhs[3][i][j][k]=((dtty2*cv[(j+1)])-(dtty1*rhoq[(j+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsy", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                             
	   c-------------------------------------------------------------------
	 */
	j=1;
	/* #pragma omp for nowait */
	/* trace_start("lhsy", 3); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, j, lhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsy) 
#pragma cuda gpurun registerRO(comz1, comz4, j) 
	lhsy_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsy", 3); */
	/* trace_start("lhsy", 4); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsy) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsy_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsy", 4); */
	j=(162-3);
	/* trace_start("lhsy", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, j, lhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsy) 
#pragma cuda gpurun registerRO(comz1, comz4, j) 
#pragma cuda gpurun cudafree(j) 
	lhsy_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	/* trace_stop("lhsy", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, do the other two factors                    
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsy", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dtty2, ( & dtty2), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dtty2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, speed) 
#pragma cuda gpurun nocudamalloc(lhs, speed) 
#pragma cuda gpurun nocudafree(dtty2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dtty2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsy) 
#pragma cuda gpurun registerRO(dtty2, speed[i][(j+1)][k], speed[i][(j-1)][k]) 
	lhsy_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dtty2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsy", 6); */
	return ;
}

__global__ void lhsy_clnd1_cloned1_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * j, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int j_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#1#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			lhs[2][i][j_0][k]=(lhs[2][i][j_0][k]+( * comz5));
			lhs[3][i][j_0][k]=(lhs[3][i][j_0][k]-comz4_0);
			lhs[4][i][j_0][k]=(lhs[4][i][j_0][k]+comz1_0);
			lhs[1][i][(j_0+1)][k]=(lhs[1][i][(j_0+1)][k]-comz4_0);
			lhs[2][i][(j_0+1)][k]=(lhs[2][i][(j_0+1)][k]+( * comz6));
			lhs[3][i][(j_0+1)][k]=(lhs[3][i][(j_0+1)][k]-comz4_0);
			lhs[4][i][(j_0+1)][k]=(lhs[4][i][(j_0+1)][k]+comz1_0);
		}
	}
}

__global__ void lhsy_clnd1_cloned1_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz1_0=( * comz1);
	comz4_0=( * comz4);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#2#0 
		for (j=3; j<((1+162)-4); j ++ )
		{
#pragma loop name lhsy#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsy_clnd1_cloned1_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * j, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int j_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	comz1_0=( * comz1);
	j_0=( * j);
	comz4_0=( * comz4);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#3#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			lhs[0][i][j_0][k]=(lhs[0][i][j_0][k]+comz1_0);
			lhs[1][i][j_0][k]=(lhs[1][i][j_0][k]-comz4_0);
			lhs[2][i][j_0][k]=(lhs[2][i][j_0][k]+( * comz6));
			lhs[3][i][j_0][k]=(lhs[3][i][j_0][k]-comz4_0);
			lhs[0][i][(j_0+1)][k]=(lhs[0][i][(j_0+1)][k]+comz1_0);
			lhs[1][i][(j_0+1)][k]=(lhs[1][i][(j_0+1)][k]-comz4_0);
			lhs[2][i][(j_0+1)][k]=(lhs[2][i][(j_0+1)][k]+( * comz5));
		}
	}
}

__global__ void lhsy_clnd1_cloned1_kernel3(double * dtty2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dtty2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dtty2_0=( * dtty2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsy#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsy#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[i][(j+1)][k];
				speed_0=speed[i][(j-1)][k];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dtty2_0*speed_0));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dtty2_0*speed_1));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dtty2_0*speed_0));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dtty2_0*speed_1));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsy_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three y-factors   
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      first fill the lhs for the u-eigenvalue         
	   c-------------------------------------------------------------------
	 */
	int * gpu__j;
#pragma loop name lhsy#0 
	for (i=1; i<((1+162)-2); i ++ )
	{
#pragma loop name lhsy#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			/* trace_start("lhsy", 1); */
			/* #pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsy#0#0#0 
			for (j=0; j<((1+162)-1); j ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[j]=vs[i][j][k];
				rhoq[j]=(((dy3+(con43*ru1))>(((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1))) ? (dy3+(con43*ru1)) : (((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)));
			}
			/* trace_stop("lhsy", 1); */
			/* trace_start("lhsy", 2); */
			/* #pragma omp parallel for private(j) schedule(static) */
#pragma loop name lhsy#0#0#1 
			for (j=1; j<((1+162)-2); j ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dtty2)*cv[(j-1)])-(dtty1*rhoq[(j-1)]));
				lhs[2][i][j][k]=(1.0+(c2dtty1*rhoq[j]));
				lhs[3][i][j][k]=((dtty2*cv[(j+1)])-(dtty1*rhoq[(j+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsy", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                             
	   c-------------------------------------------------------------------
	 */
	j=1;
	/* #pragma omp for nowait */
	/* trace_start("lhsy", 3); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, j, lhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsy_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, j) 
	lhsy_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsy", 3); */
	/* trace_start("lhsy", 4); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsy_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsy_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsy", 4); */
	j=(162-3);
	/* trace_start("lhsy", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, j, lhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, j, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsy_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, j) 
#pragma cuda gpurun cudafree(j) 
	lhsy_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	/* trace_stop("lhsy", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, do the other two factors                    
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsy", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dtty2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(dtty2, lhs, speed) 
#pragma cuda gpurun nocudamalloc(dtty2, lhs, speed) 
#pragma cuda gpurun nocudafree(dtty2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dtty2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsy_clnd1_cloned1) 
#pragma cuda gpurun registerRO(dtty2, speed[i][(j+1)][k], speed[i][(j-1)][k]) 
	lhsy_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dtty2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsy", 6); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsz_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * k, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int k_0;
	int i;
	int j;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	comz1_0=( * comz1);
	comz4_0=( * comz4);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name lhsz#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[2][i][j][k_0]=(lhs[2][i][j][k_0]+( * comz5));
			lhs[3][i][j][k_0]=(lhs[3][i][j][k_0]-comz4_0);
			lhs[4][i][j][k_0]=(lhs[4][i][j][k_0]+comz1_0);
			lhs[1][i][j][(k_0+1)]=(lhs[1][i][j][(k_0+1)]-comz4_0);
			lhs[2][i][j][(k_0+1)]=(lhs[2][i][j][(k_0+1)]+( * comz6));
			lhs[3][i][j][(k_0+1)]=(lhs[3][i][j][(k_0+1)]-comz4_0);
			lhs[4][i][j][(k_0+1)]=(lhs[4][i][j][(k_0+1)]+comz1_0);
		}
	}
}

__global__ void lhsz_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+3);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-4))
	{
#pragma loop name lhsz#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsz#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsz_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * k, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int k_0;
	int i;
	int j;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (i<((1+162)-2))
	{
#pragma loop name lhsz#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[0][i][j][k_0]=(lhs[0][i][j][k_0]+comz1_0);
			lhs[1][i][j][k_0]=(lhs[1][i][j][k_0]-comz4_0);
			lhs[2][i][j][k_0]=(lhs[2][i][j][k_0]+( * comz6));
			lhs[3][i][j][k_0]=(lhs[3][i][j][k_0]-comz4_0);
			lhs[0][i][j][(k_0+1)]=(lhs[0][i][j][(k_0+1)]+comz1_0);
			lhs[1][i][j][(k_0+1)]=(lhs[1][i][j][(k_0+1)]-comz4_0);
			lhs[2][i][j][(k_0+1)]=(lhs[2][i][j][(k_0+1)]+( * comz5));
		}
	}
}

__global__ void lhsz_kernel3(double * dttz2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dttz2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dttz2_0=( * dttz2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsz#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsz#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[i][j][(k-1)];
				speed_0=speed[i][j][(k+1)];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dttz2_0*speed_1));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dttz2_0*speed_0));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dttz2_0*speed_1));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dttz2_0*speed_0));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsz(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three z-factors   
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c first fill the lhs for the u-eigenvalue                          
	   c-------------------------------------------------------------------
	 */
	int * gpu__k;
#pragma loop name lhsz#0 
	for (i=1; i<((1+162)-2); i ++ )
	{
#pragma loop name lhsz#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			/* trace_start("lhsz", 1); */
			/* #pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsz#0#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[k]=ws[i][j][k];
				rhos[k]=(((dz4+(con43*ru1))>(((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1))) ? (dz4+(con43*ru1)) : (((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)));
			}
			/* trace_stop("lhsz", 1); */
			/* trace_start("lhsz", 2); */
			/* #pragma omp for   */
			/* #pragma omp parallel for private(k) schedule(static) */
#pragma loop name lhsz#0#0#1 
			for (k=1; k<((1+162)-2); k ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttz2)*cv[(k-1)])-(dttz1*rhos[(k-1)]));
				lhs[2][i][j][k]=(1.0+(c2dttz1*rhos[k]));
				lhs[3][i][j][k]=((dttz2*cv[(k+1)])-(dttz1*rhos[(k+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsz", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                                  
	   c-------------------------------------------------------------------
	 */
	k=1;
	/* trace_start("lhsz", 3); */
	/* #pragma omp for nowait */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, k, lhs) private(i, j) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsz) 
#pragma cuda gpurun registerRO(comz1, comz4, k) 
	lhsz_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsz", 3); */
	/* trace_start("lhsz", 4); */
	/* #pragma omp for nowait */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsz) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsz_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsz", 4); */
	k=(162-3);
	/* trace_start("lhsz", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, k, lhs) private(i, j) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsz) 
#pragma cuda gpurun registerRO(comz1, comz4, k) 
#pragma cuda gpurun cudafree(k) 
	lhsz_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("lhsz", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, fill the other factors (u+c), (u-c) 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsz", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dttz2, ( & dttz2), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dttz2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, speed) 
#pragma cuda gpurun nocudamalloc(lhs, speed) 
#pragma cuda gpurun nocudafree(dttz2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dttz2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsz) 
#pragma cuda gpurun registerRO(dttz2, speed[i][j][(k+1)], speed[i][j][(k-1)]) 
	lhsz_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dttz2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsz", 6); */
	return ;
}

__global__ void lhsz_clnd1_cloned1_kernel0(double * comz1, double * comz4, double * comz5, double * comz6, int * k, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int k_0;
	int i;
	int j;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (i<((1+162)-2))
	{
#pragma loop name lhsz#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[2][i][j][k_0]=(lhs[2][i][j][k_0]+( * comz5));
			lhs[3][i][j][k_0]=(lhs[3][i][j][k_0]-comz4_0);
			lhs[4][i][j][k_0]=(lhs[4][i][j][k_0]+comz1_0);
			lhs[1][i][j][(k_0+1)]=(lhs[1][i][j][(k_0+1)]-comz4_0);
			lhs[2][i][j][(k_0+1)]=(lhs[2][i][j][(k_0+1)]+( * comz6));
			lhs[3][i][j][(k_0+1)]=(lhs[3][i][j][(k_0+1)]-comz4_0);
			lhs[4][i][j][(k_0+1)]=(lhs[4][i][j][(k_0+1)]+comz1_0);
		}
	}
}

__global__ void lhsz_clnd1_cloned1_kernel1(double * comz1, double * comz4, double * comz6, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+3);
	comz4_0=( * comz4);
	comz1_0=( * comz1);
	if (k<((1+162)-4))
	{
#pragma loop name lhsz#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsz#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1_0);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4_0);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+( * comz6));
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4_0);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1_0);
			}
		}
	}
}

__global__ void lhsz_clnd1_cloned1_kernel2(double * comz1, double * comz4, double * comz5, double * comz6, int * k, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double comz1_0;
	double comz4_0;
	int k_0;
	int i;
	int j;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	comz1_0=( * comz1);
	k_0=( * k);
	comz4_0=( * comz4);
	if (i<((1+162)-2))
	{
#pragma loop name lhsz#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			lhs[0][i][j][k_0]=(lhs[0][i][j][k_0]+comz1_0);
			lhs[1][i][j][k_0]=(lhs[1][i][j][k_0]-comz4_0);
			lhs[2][i][j][k_0]=(lhs[2][i][j][k_0]+( * comz6));
			lhs[3][i][j][k_0]=(lhs[3][i][j][k_0]-comz4_0);
			lhs[0][i][j][(k_0+1)]=(lhs[0][i][j][(k_0+1)]+comz1_0);
			lhs[1][i][j][(k_0+1)]=(lhs[1][i][j][(k_0+1)]-comz4_0);
			lhs[2][i][j][(k_0+1)]=(lhs[2][i][j][(k_0+1)]+( * comz5));
		}
	}
}

__global__ void lhsz_clnd1_cloned1_kernel3(double * dttz2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double dttz2_0;
	double speed_0;
	double speed_1;
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	dttz2_0=( * dttz2);
	if (k<((1+162)-2))
	{
#pragma loop name lhsz#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name lhsz#4#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				speed_1=speed[i][j][(k-1)];
				speed_0=speed[i][j][(k+1)];
				lhs[(0+5)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+5)][i][j][k]=(lhs[1][i][j][k]-(dttz2_0*speed_1));
				lhs[(2+5)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+5)][i][j][k]=(lhs[3][i][j][k]+(dttz2_0*speed_0));
				lhs[(4+5)][i][j][k]=lhs[4][i][j][k];
				lhs[(0+10)][i][j][k]=lhs[0][i][j][k];
				lhs[(1+10)][i][j][k]=(lhs[1][i][j][k]+(dttz2_0*speed_1));
				lhs[(2+10)][i][j][k]=lhs[2][i][j][k];
				lhs[(3+10)][i][j][k]=(lhs[3][i][j][k]-(dttz2_0*speed_0));
				lhs[(4+10)][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
}

static void lhsz_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c This function computes the left hand side for the three z-factors   
	   c-------------------------------------------------------------------
	 */
	double ru1;
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c first fill the lhs for the u-eigenvalue                          
	   c-------------------------------------------------------------------
	 */
	int * gpu__k;
#pragma loop name lhsz#0 
	for (i=1; i<((1+162)-2); i ++ )
	{
#pragma loop name lhsz#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			/* trace_start("lhsz", 1); */
			/* #pragma omp parallel for private(i, j, k, ru1) schedule(static) */
#pragma loop name lhsz#0#0#0 
			for (k=0; k<((1+162)-1); k ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[k]=ws[i][j][k];
				rhos[k]=(((dz4+(con43*ru1))>(((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1))) ? (dz4+(con43*ru1)) : (((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)));
			}
			/* trace_stop("lhsz", 1); */
			/* trace_start("lhsz", 2); */
			/* #pragma omp for   */
			/* #pragma omp parallel for private(k) schedule(static) */
#pragma loop name lhsz#0#0#1 
			for (k=1; k<((1+162)-2); k ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttz2)*cv[(k-1)])-(dttz1*rhos[(k-1)]));
				lhs[2][i][j][k]=(1.0+(c2dttz1*rhos[k]));
				lhs[3][i][j][k]=((dttz2*cv[(k+1)])-(dttz1*rhos[(k+1)]));
				lhs[4][i][j][k]=0.0;
			}
			/* trace_stop("lhsz", 2); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      add fourth order dissipation                                  
	   c-------------------------------------------------------------------
	 */
	k=1;
	/* trace_start("lhsz", 3); */
	/* #pragma omp for nowait */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=((((15*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__lhs, lhs, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, k, lhs) private(i, j) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda ainfo kernelid(0) procname(lhsz_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, k) 
	lhsz_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsz", 3); */
	/* trace_start("lhsz", 4); */
	/* #pragma omp for nowait */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(comz1, comz4, comz6, lhs) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz6, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsz_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4) 
	lhsz_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz6, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	/* trace_stop("lhsz", 4); */
	k=(162-3);
	/* trace_start("lhsz", 5); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(comz1, comz4, comz5, comz6, k, lhs) private(i, j) schedule(static)
#pragma cuda gpurun noc2gmemtr(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nocudamalloc(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda gpurun nocudafree(comz1, comz4, comz5, comz6, lhs) 
#pragma cuda gpurun nog2cmemtr(comz1, comz4, comz5, comz6, k, lhs) 
#pragma cuda ainfo kernelid(2) procname(lhsz_clnd1_cloned1) 
#pragma cuda gpurun registerRO(comz1, comz4, k) 
#pragma cuda gpurun cudafree(k) 
	lhsz_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__comz1, gpu__comz4, gpu__comz5, gpu__comz6, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("lhsz", 5); */
	/*
	   --------------------------------------------------------------------
	   c      subsequently, fill the other factors (u+c), (u-c) 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for   */
	/* trace_start("lhsz", 6); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dttz2, lhs, speed) private(i, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(dttz2, lhs, speed) 
#pragma cuda gpurun nocudamalloc(dttz2, lhs, speed) 
#pragma cuda gpurun nocudafree(dttz2, lhs, speed) 
#pragma cuda gpurun nog2cmemtr(dttz2, lhs, speed) 
#pragma cuda ainfo kernelid(3) procname(lhsz_clnd1_cloned1) 
#pragma cuda gpurun registerRO(dttz2, speed[i][j][(k+1)], speed[i][j][(k-1)]) 
	lhsz_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dttz2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed));
	/* trace_stop("lhsz", 6); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void ninvr_kernel0(double * bt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name ninvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name ninvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt_0*r3);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=( - r2);
				rhs[1][i][j][k]=r1;
				rhs[2][i][j][k]=(bt_0*(r4-r5));
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
}

static void ninvr(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication              
	   c-------------------------------------------------------------------
	 */
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
#pragma omp parallel for shared(bt, rhs) private(i, j, k, r1, r2, r3, r4, r5, t1, t2) schedule(static)
#pragma cuda gpurun noc2gmemtr(bt, rhs) 
#pragma cuda gpurun nocudamalloc(bt, rhs) 
#pragma cuda gpurun nocudafree(bt, rhs) 
#pragma cuda gpurun nog2cmemtr(bt, rhs) 
#pragma cuda ainfo kernelid(0) procname(ninvr) 
#pragma cuda gpurun registerRO(bt) 
	ninvr_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__bt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("ninvr", 1); */
	return ;
}

__global__ void ninvr_clnd1_cloned1_kernel0(double * bt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name ninvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name ninvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt_0*r3);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=( - r2);
				rhs[1][i][j][k]=r1;
				rhs[2][i][j][k]=(bt_0*(r4-r5));
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
}

static void ninvr_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication              
	   c-------------------------------------------------------------------
	 */
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
#pragma omp parallel for shared(bt, rhs) private(i, j, k, r1, r2, r3, r4, r5, t1, t2) schedule(static)
#pragma cuda gpurun noc2gmemtr(bt, rhs) 
#pragma cuda gpurun nocudamalloc(bt, rhs) 
#pragma cuda gpurun nocudafree(bt, rhs) 
#pragma cuda gpurun nog2cmemtr(bt, rhs) 
#pragma cuda ainfo kernelid(0) procname(ninvr_clnd1_cloned1) 
#pragma cuda gpurun registerRO(bt) 
	ninvr_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__bt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("ninvr", 1); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void pinvr_kernel0(double * bt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name pinvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name pinvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt_0*r1);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=(bt_0*(r4-r5));
				rhs[1][i][j][k]=( - r3);
				rhs[2][i][j][k]=r2;
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
}

static void pinvr(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication                       
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("pinvr", 1); */
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
#pragma omp parallel for shared(bt, rhs) private(i, j, k, r1, r2, r3, r4, r5, t1, t2) schedule(static)
#pragma cuda gpurun noc2gmemtr(bt, rhs) 
#pragma cuda gpurun nocudamalloc(bt, rhs) 
#pragma cuda gpurun nocudafree(bt, rhs) 
#pragma cuda gpurun nog2cmemtr(bt, rhs) 
#pragma cuda ainfo kernelid(0) procname(pinvr) 
#pragma cuda gpurun registerRO(bt) 
	pinvr_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__bt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("pinvr", 1); */
	return ;
}

__global__ void pinvr_clnd1_cloned1_kernel0(double * bt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name pinvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name pinvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt_0*r1);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=(bt_0*(r4-r5));
				rhs[1][i][j][k]=( - r3);
				rhs[2][i][j][k]=r2;
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
}

static void pinvr_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication                       
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("pinvr", 1); */
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
#pragma omp parallel for shared(bt, rhs) private(i, j, k, r1, r2, r3, r4, r5, t1, t2) schedule(static)
#pragma cuda gpurun noc2gmemtr(bt, rhs) 
#pragma cuda gpurun nocudamalloc(bt, rhs) 
#pragma cuda gpurun nocudafree(bt, rhs) 
#pragma cuda gpurun nog2cmemtr(bt, rhs) 
#pragma cuda ainfo kernelid(0) procname(pinvr_clnd1_cloned1) 
#pragma cuda gpurun registerRO(bt) 
	pinvr_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__bt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("pinvr", 1); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void compute_rhs_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * c1c2, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double square_0;
	double u_0;
	double u_1;
	double u_2;
	double aux;
	int i;
	int j;
	int k;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				u_2=u[2][i][j][k];
				u_1=u[3][i][j][k];
				u_0=u[1][i][j][k];
				square_0=square[i][j][k];
				rho_inv=(1.0/u[0][i][j][k]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_0*rho_inv);
				vs[i][j][k]=(u_2*rho_inv);
				ws[i][j][k]=(u_1*rho_inv);
				square_0=((0.5*(((u_0*u_0)+(u_2*u_2))+(u_1*u_1)))*rho_inv);
				qs[i][j][k]=(square_0*rho_inv);
				/*
				   --------------------------------------------------------------------
				   c               (do not need speed and ainx until the lhs computation)
				   c-------------------------------------------------------------------
				 */
				aux=((( * c1c2)*rho_inv)*(u[4][i][j][k]-square_0));
				/* aux = sqrt(aux); */
				speed[i][j][k]=aux;
				ainv[i][j][k]=(1.0/aux);
				square[i][j][k]=square_0;
			}
		}
	}
}

__global__ void compute_rhs_kernel1(double forcing[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#1#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=forcing[m][i][j][k];
				}
			}
		}
	}
}

__global__ void compute_rhs_kernel2(double * c1, double * c2, double * con43, double * dx1tx1, double * dx2tx1, double * dx3tx1, double * dx4tx1, double * dx5tx1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tx2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * xxcon2, double * xxcon3, double * xxcon4, double * xxcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tx2_0;
	double u_0;
	double u_5;
	double u_7;
	double u_8;
	double xxcon2_0;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	c2_0=( * c2);
	xxcon2_0=( * xxcon2);
	c1_0=( * c1);
	tx2_0=( * tx2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[1][(i-1)][j][k];
				u_7=u[4][(i+1)][j][k];
				u_5=u[4][(i-1)][j][k];
				u_0=u[1][(i+1)][j][k];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dx1tx1)*((u[0][(i+1)][j][k]-(2.0*u[0][i][j][k]))+u[0][(i-1)][j][k])))-(tx2_0*(u_0-u_8)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dx2tx1)*((u_0-(2.0*u[1][i][j][k]))+u_8)))+((xxcon2_0*( * con43))*((up1-(2.0*uijk))+um1)))-(tx2_0*(((u_0*up1)-(u_8*um1))+((((u_7-square_1)-u_5)+square_0)*c2_0))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dx3tx1)*((u[2][(i+1)][j][k]-(2.0*u[2][i][j][k]))+u[2][(i-1)][j][k])))+(xxcon2_0*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(tx2_0*((u[2][(i+1)][j][k]*up1)-(u[2][(i-1)][j][k]*um1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dx4tx1)*((u[3][(i+1)][j][k]-(2.0*u[3][i][j][k]))+u[3][(i-1)][j][k])))+(xxcon2_0*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(tx2_0*((u[3][(i+1)][j][k]*up1)-(u[3][(i-1)][j][k]*um1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dx5tx1)*((u_7-(2.0*u[4][i][j][k]))+u_5)))+(( * xxcon3)*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(( * xxcon4)*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(( * xxcon5)*(((u_7*rho_i[(i+1)][j][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[(i-1)][j][k]))))-(tx2_0*((((c1_0*u_7)-(c2_0*square_1))*up1)-(((c1_0*u_5)-(c2_0*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel3(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
		/* #pragma omp for */
#pragma loop name compute_rhs#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((5.0*u[m][i_0][j][k])-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_kernel4(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((((( - 4.0)*u[m][(i_0-1)][j][k])+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_kernel5(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#5#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (i=(3*1); i<(((1+162)-(3*1))-1); i ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][(i-2)][j][k]-(4.0*u[m][(i-1)][j][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][(i+1)][j][k]))+u[m][(i+2)][j][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_kernel6(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#6#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel7(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#7#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(5.0*u[m][i_0][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel8(double * c1, double * c2, double * con43, double * dy1ty1, double * dy2ty1, double * dy3ty1, double * dy4ty1, double * dy5ty1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * ty2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * yycon2, double * yycon3, double * yycon4, double * yycon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double ty2_0;
	double u_5;
	double u_6;
	double u_7;
	double u_8;
	double yycon2_0;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	c1_0=( * c1);
	yycon2_0=( * yycon2);
	ty2_0=( * ty2);
	c2_0=( * c2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[4][i][(j-1)][k];
				u_7=u[4][i][(j+1)][k];
				u_6=u[2][i][(j+1)][k];
				u_5=u[2][i][(j-1)][k];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dy1ty1)*((u[0][i][(j+1)][k]-(2.0*u[0][i][j][k]))+u[0][i][(j-1)][k])))-(ty2_0*(u_6-u_5)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dy2ty1)*((u[1][i][(j+1)][k]-(2.0*u[1][i][j][k]))+u[1][i][(j-1)][k])))+(yycon2_0*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(ty2_0*((u[1][i][(j+1)][k]*vp1)-(u[1][i][(j-1)][k]*vm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dy3ty1)*((u_6-(2.0*u[2][i][j][k]))+u_5)))+((yycon2_0*( * con43))*((vp1-(2.0*vijk))+vm1)))-(ty2_0*(((u_6*vp1)-(u_5*vm1))+((((u_7-square_0)-u_8)+square_1)*c2_0))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dy4ty1)*((u[3][i][(j+1)][k]-(2.0*u[3][i][j][k]))+u[3][i][(j-1)][k])))+(yycon2_0*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(ty2_0*((u[3][i][(j+1)][k]*vp1)-(u[3][i][(j-1)][k]*vm1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dy5ty1)*((u_7-(2.0*u[4][i][j][k]))+u_8)))+(( * yycon3)*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(( * yycon4)*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(( * yycon5)*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_8*rho_i[i][(j-1)][k]))))-(ty2_0*((((c1_0*u_7)-(c2_0*square_0))*vp1)-(((c1_0*u_8)-(c2_0*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel9(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#9#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((5.0*u[m][i][j][k])-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_kernel10(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#10#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((( - 4.0)*u[m][i][(j-1)][k])+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_kernel11(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#11#0 
		for (j=(3*1); j<(((1+162)-(3*1))-1); j ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_kernel12(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#12#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel13(double * dssp, int * j, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#13#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(( * dssp)*((u[m][i][(j_0-2)][k]-(4.0*u[m][i][(j_0-1)][k]))+(5.0*u[m][i][j_0][k]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel14(double * c1, double * c2, double * con43, double * dz1tz1, double * dz2tz1, double * dz3tz1, double * dz4tz1, double * dz5tz1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tz2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * zzcon2, double * zzcon3, double * zzcon4, double * zzcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tz2_0;
	double u_3;
	double u_4;
	double u_5;
	double u_6;
	double zzcon2_0;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	c1_0=( * c1);
	c2_0=( * c2);
	zzcon2_0=( * zzcon2);
	tz2_0=( * tz2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_6=u[3][i][j][(k-1)];
				u_5=u[4][i][j][(k-1)];
				u_4=u[4][i][j][(k+1)];
				u_3=u[3][i][j][(k+1)];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dz1tz1)*((u[0][i][j][(k+1)]-(2.0*u[0][i][j][k]))+u[0][i][j][(k-1)])))-(tz2_0*(u_3-u_6)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dz2tz1)*((u[1][i][j][(k+1)]-(2.0*u[1][i][j][k]))+u[1][i][j][(k-1)])))+(zzcon2_0*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(tz2_0*((u[1][i][j][(k+1)]*wp1)-(u[1][i][j][(k-1)]*wm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dz3tz1)*((u[2][i][j][(k+1)]-(2.0*u[2][i][j][k]))+u[2][i][j][(k-1)])))+(zzcon2_0*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(tz2_0*((u[2][i][j][(k+1)]*wp1)-(u[2][i][j][(k-1)]*wm1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dz4tz1)*((u_3-(2.0*u[3][i][j][k]))+u_6)))+((zzcon2_0*( * con43))*((wp1-(2.0*wijk))+wm1)))-(tz2_0*(((u_3*wp1)-(u_6*wm1))+((((u_4-square_0)-u_5)+square_1)*c2_0))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dz5tz1)*((u_4-(2.0*u[4][i][j][k]))+u_5)))+(( * zzcon3)*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(( * zzcon4)*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(( * zzcon5)*(((u_4*rho_i[i][j][(k+1)])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[i][j][(k-1)]))))-(tz2_0*((((c1_0*u_4)-(c2_0*square_0))*wp1)-(((c1_0*u_5)-(c2_0*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel15(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<(162-1))
	{
#pragma loop name compute_rhs#15#0 
		for (j=1; j<(162-1); j ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*(((5.0*u[m][i][j][k_0])-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_kernel16(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#16#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((((( - 4.0)*u[m][i][j][(k_0-1)])+(6.0*u[m][i][j][k_0]))-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_kernel17(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+3);
	if (k<(((1+162)-(3*1))-1))
	{
#pragma loop name compute_rhs#17#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))+u[m][i][j][(k+2)])));
				}
			}
		}
	}
}

__global__ void compute_rhs_kernel18(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#18#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel19(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#19#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((u[m][i][j][(k_0-2)]-(4.0*u[m][i][j][(k_0-1)]))+(5.0*u[m][i][j][k_0]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel20(double * dt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#20#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]*( * dt));
				}
			}
		}
	}
}

static void compute_rhs(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      compute the reciprocal of density, and the kinetic energy, 
	   c      and the speed of sound. 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for nowait */
	/* trace_start("compute_rhs", 1); */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	int * gpu__i;
	int * gpu__j;
	int * gpu__k;
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
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__c1c2, ( & c1c2), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) private(aux, i, j, k, rho_inv) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, qs, rho_i, speed, spped, square, u, us, vs, ws) nog2cmemtr(ainv, c1c2, qs, rho_i, speed, spped, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs) 
#pragma cuda gpurun registerRO(u[1][i][j][k], u[2][i][j][k], u[3][i][j][k]) 
#pragma cuda gpurun registerRW(square[i][j][k]) 
	compute_rhs_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__c1c2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("compute_rhs", 1); */
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__forcing, forcing, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(forcing, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun nocudafree(forcing, rhs) 
#pragma cuda gpurun nog2cmemtr(forcing, rhs) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs) 
#pragma cuda gpurun noc2gmemtr(rhs) 
	compute_rhs_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__forcing), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("compute_rhs", 2); */
	/*
	   --------------------------------------------------------------------
	   c      compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 3); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__c1, ( & c1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__c2, ( & c2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__con43, ( & con43), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dx1tx1, ( & dx1tx1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dx2tx1, ( & dx2tx1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dx3tx1, ( & dx3tx1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dx4tx1, ( & dx4tx1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dx5tx1, ( & dx5tx1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__tx2, ( & tx2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xxcon2, ( & xxcon2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xxcon3, ( & xxcon3), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xxcon4, ( & xxcon4), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__xxcon5, ( & xxcon5), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1) schedule(static)
#pragma cuda gpurun noc2gmemtr(qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs) 
#pragma cuda gpurun registerRO(c1, c2, square[(i+1)][j][k], square[(i-1)][j][k], tx2, u[1][(i+1)][j][k], u[1][(i-1)][j][k], u[4][(i+1)][j][k], u[4][(i-1)][j][k], xxcon2) 
	compute_rhs_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dx1tx1, gpu__dx2tx1, gpu__dx3tx1, gpu__dx4tx1, gpu__dx5tx1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tx2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__xxcon2, gpu__xxcon3, gpu__xxcon4, gpu__xxcon5);
	/* trace_stop("compute_rhs", 3); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	i=1;
	/* trace_start("compute_rhs", 4); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dssp, ( & dssp), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(rhs, u) 
#pragma cuda gpurun nocudamalloc(rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 4); */
	i=2;
	/* trace_start("compute_rhs", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 5); */
	/* trace_start("compute_rhs", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs) 
	compute_rhs_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 6); */
	i=(162-3);
	/* trace_start("compute_rhs", 7); */
	dim3 dimBlock6(gpuNumThreads, 1, 1);
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
	dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 7); */
	i=(162-2);
	/* trace_start("compute_rhs", 8); */
	dim3 dimBlock7(gpuNumThreads, 1, 1);
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
	dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs) 
#pragma cuda gpurun registerRO(i) 
#pragma cuda gpurun cudafree(i) 
	compute_rhs_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/* trace_stop("compute_rhs", 8); */
	/* #pragma omp barrier */
	/*
	   --------------------------------------------------------------------
	   c      compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 9); */
	dim3 dimBlock8(gpuNumThreads, 1, 1);
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
	dim3 dimGrid8(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dy1ty1, ( & dy1ty1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dy2ty1, ( & dy2ty1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dy3ty1, ( & dy3ty1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dy4ty1, ( & dy4ty1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dy5ty1, ( & dy5ty1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__ty2, ( & ty2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__yycon2, ( & yycon2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__yycon3, ( & yycon3), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__yycon4, ( & yycon4), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__yycon5, ( & yycon5), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(8) procname(compute_rhs) 
#pragma cuda gpurun registerRO(c1, c2, square[i][(j+1)][k], square[i][(j-1)][k], ty2, u[2][i][(j+1)][k], u[2][i][(j-1)][k], u[4][i][(j+1)][k], u[4][i][(j-1)][k], yycon2) 
	compute_rhs_kernel8<<<dimGrid8, dimBlock8, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dy1ty1, gpu__dy2ty1, gpu__dy3ty1, gpu__dy4ty1, gpu__dy5ty1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__ty2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__yycon2, gpu__yycon3, gpu__yycon4, gpu__yycon5);
	/* trace_stop("compute_rhs", 9); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	j=1;
	/* trace_start("compute_rhs", 10); */
	dim3 dimBlock9(gpuNumThreads, 1, 1);
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
	dim3 dimGrid9(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(9) procname(compute_rhs) 
	compute_rhs_kernel9<<<dimGrid9, dimBlock9, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 10); */
	j=2;
	/* trace_start("compute_rhs", 11); */
	dim3 dimBlock10(gpuNumThreads, 1, 1);
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
	dim3 dimGrid10(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(10) procname(compute_rhs) 
	compute_rhs_kernel10<<<dimGrid10, dimBlock10, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 11); */
	/* trace_start("compute_rhs", 12); */
	dim3 dimBlock11(gpuNumThreads, 1, 1);
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
	dim3 dimGrid11(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(11) procname(compute_rhs) 
	compute_rhs_kernel11<<<dimGrid11, dimBlock11, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 12); */
	j=(162-3);
	/* trace_start("compute_rhs", 13); */
	dim3 dimBlock12(gpuNumThreads, 1, 1);
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
	dim3 dimGrid12(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(12) procname(compute_rhs) 
	compute_rhs_kernel12<<<dimGrid12, dimBlock12, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 13); */
	j=(162-2);
	/* trace_start("compute_rhs", 14); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
	dim3 dimBlock13(gpuNumThreads, 1, 1);
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
	dim3 dimGrid13(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, j, rhs, u) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, j, rhs, u) 
#pragma cuda ainfo kernelid(13) procname(compute_rhs) 
#pragma cuda gpurun registerRO(j) 
#pragma cuda gpurun cudafree(j) 
	compute_rhs_kernel13<<<dimGrid13, dimBlock13, 0, 0>>>(gpu__dssp, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	/* trace_stop("compute_rhs", 14); */
	/* #pragma omp barrier   */
	/*
	   --------------------------------------------------------------------
	   c      compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 15); */
	dim3 dimBlock14(gpuNumThreads, 1, 1);
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
	dim3 dimGrid14(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dz1tz1, ( & dz1tz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dz2tz1, ( & dz2tz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dz3tz1, ( & dz3tz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dz4tz1, ( & dz4tz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dz5tz1, ( & dz5tz1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__tz2, ( & tz2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zzcon2, ( & zzcon2), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zzcon3, ( & zzcon3), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zzcon4, ( & zzcon4), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__zzcon5, ( & zzcon5), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, j, qs, rho_i, rhs, square, u, us, vs, ws) nog2cmemtr(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(14) procname(compute_rhs) 
#pragma cuda gpurun registerRO(c1, c2, square[i][j][(k+1)], square[i][j][(k-1)], tz2, u[3][i][j][(k+1)], u[3][i][j][(k-1)], u[4][i][j][(k+1)], u[4][i][j][(k-1)], zzcon2) 
	compute_rhs_kernel14<<<dimGrid14, dimBlock14, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dz1tz1, gpu__dz2tz1, gpu__dz3tz1, gpu__dz4tz1, gpu__dz5tz1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tz2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__zzcon2, gpu__zzcon3, gpu__zzcon4, gpu__zzcon5);
	/* trace_stop("compute_rhs", 15); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	k=1;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
	dim3 dimBlock15(gpuNumThreads, 1, 1);
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
	dim3 dimGrid15(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(15) procname(compute_rhs) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_kernel15<<<dimGrid15, dimBlock15, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 16); */
	k=2;
	/* trace_start("compute_rhs", 17); */
	dim3 dimBlock16(gpuNumThreads, 1, 1);
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
	dim3 dimGrid16(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(16) procname(compute_rhs) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_kernel16<<<dimGrid16, dimBlock16, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 17); */
	/* trace_start("compute_rhs", 18); */
	dim3 dimBlock17(gpuNumThreads, 1, 1);
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
	dim3 dimGrid17(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(17) procname(compute_rhs) 
	compute_rhs_kernel17<<<dimGrid17, dimBlock17, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 18); */
	k=(162-3);
	/* trace_start("compute_rhs", 19); */
	dim3 dimBlock18(gpuNumThreads, 1, 1);
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
	dim3 dimGrid18(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(18) procname(compute_rhs) 
	compute_rhs_kernel18<<<dimGrid18, dimBlock18, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 19); */
	k=(162-2);
	/* trace_start("compute_rhs", 20); */
	dim3 dimBlock19(gpuNumThreads, 1, 1);
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
	dim3 dimGrid19(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(19) procname(compute_rhs) 
#pragma cuda gpurun registerRO(k) 
#pragma cuda gpurun cudafree(k) 
	compute_rhs_kernel19<<<dimGrid19, dimBlock19, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("compute_rhs", 20); */
	/* trace_start("compute_rhs", 21); */
	dim3 dimBlock20(gpuNumThreads, 1, 1);
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
	dim3 dimGrid20(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__dt, ( & dt), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(dt, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(rhs) 
#pragma cuda gpurun nocudamalloc(rhs) 
#pragma cuda gpurun nocudafree(dt, rhs) 
#pragma cuda gpurun nog2cmemtr(dt, rhs) 
#pragma cuda ainfo kernelid(20) procname(compute_rhs) 
	compute_rhs_kernel20<<<dimGrid20, dimBlock20, 0, 0>>>(gpu__dt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("compute_rhs", 21); */
	return ;
}

__global__ void compute_rhs_clnd2_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * c1c2, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double square_0;
	double u_0;
	double u_1;
	double u_2;
	double aux;
	int i;
	int j;
	int k;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				u_2=u[2][i][j][k];
				u_1=u[3][i][j][k];
				u_0=u[1][i][j][k];
				square_0=square[i][j][k];
				rho_inv=(1.0/u[0][i][j][k]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_0*rho_inv);
				vs[i][j][k]=(u_2*rho_inv);
				ws[i][j][k]=(u_1*rho_inv);
				square_0=((0.5*(((u_0*u_0)+(u_2*u_2))+(u_1*u_1)))*rho_inv);
				qs[i][j][k]=(square_0*rho_inv);
				/*
				   --------------------------------------------------------------------
				   c               (do not need speed and ainx until the lhs computation)
				   c-------------------------------------------------------------------
				 */
				aux=((( * c1c2)*rho_inv)*(u[4][i][j][k]-square_0));
				/* aux = sqrt(aux); */
				speed[i][j][k]=aux;
				ainv[i][j][k]=(1.0/aux);
				square[i][j][k]=square_0;
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel1(double forcing[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#1#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=forcing[m][i][j][k];
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel2(double * c1, double * c2, double * con43, double * dx1tx1, double * dx2tx1, double * dx3tx1, double * dx4tx1, double * dx5tx1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tx2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * xxcon2, double * xxcon3, double * xxcon4, double * xxcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tx2_0;
	double u_0;
	double u_5;
	double u_7;
	double u_8;
	double xxcon2_0;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	tx2_0=( * tx2);
	c1_0=( * c1);
	c2_0=( * c2);
	xxcon2_0=( * xxcon2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[1][(i-1)][j][k];
				u_7=u[4][(i+1)][j][k];
				u_5=u[4][(i-1)][j][k];
				u_0=u[1][(i+1)][j][k];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dx1tx1)*((u[0][(i+1)][j][k]-(2.0*u[0][i][j][k]))+u[0][(i-1)][j][k])))-(tx2_0*(u_0-u_8)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dx2tx1)*((u_0-(2.0*u[1][i][j][k]))+u_8)))+((xxcon2_0*( * con43))*((up1-(2.0*uijk))+um1)))-(tx2_0*(((u_0*up1)-(u_8*um1))+((((u_7-square_1)-u_5)+square_0)*c2_0))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dx3tx1)*((u[2][(i+1)][j][k]-(2.0*u[2][i][j][k]))+u[2][(i-1)][j][k])))+(xxcon2_0*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(tx2_0*((u[2][(i+1)][j][k]*up1)-(u[2][(i-1)][j][k]*um1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dx4tx1)*((u[3][(i+1)][j][k]-(2.0*u[3][i][j][k]))+u[3][(i-1)][j][k])))+(xxcon2_0*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(tx2_0*((u[3][(i+1)][j][k]*up1)-(u[3][(i-1)][j][k]*um1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dx5tx1)*((u_7-(2.0*u[4][i][j][k]))+u_5)))+(( * xxcon3)*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(( * xxcon4)*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(( * xxcon5)*(((u_7*rho_i[(i+1)][j][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[(i-1)][j][k]))))-(tx2_0*((((c1_0*u_7)-(c2_0*square_1))*up1)-(((c1_0*u_5)-(c2_0*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel3(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
		/* #pragma omp for */
#pragma loop name compute_rhs#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((5.0*u[m][i_0][j][k])-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel4(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((((( - 4.0)*u[m][(i_0-1)][j][k])+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel5(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#5#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (i=(3*1); i<(((1+162)-(3*1))-1); i ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][(i-2)][j][k]-(4.0*u[m][(i-1)][j][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][(i+1)][j][k]))+u[m][(i+2)][j][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel6(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#6#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel7(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#7#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(5.0*u[m][i_0][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel8(double * c1, double * c2, double * con43, double * dy1ty1, double * dy2ty1, double * dy3ty1, double * dy4ty1, double * dy5ty1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * ty2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * yycon2, double * yycon3, double * yycon4, double * yycon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double ty2_0;
	double u_5;
	double u_6;
	double u_7;
	double u_8;
	double yycon2_0;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	c1_0=( * c1);
	c2_0=( * c2);
	ty2_0=( * ty2);
	yycon2_0=( * yycon2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[4][i][(j-1)][k];
				u_7=u[4][i][(j+1)][k];
				u_6=u[2][i][(j+1)][k];
				u_5=u[2][i][(j-1)][k];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dy1ty1)*((u[0][i][(j+1)][k]-(2.0*u[0][i][j][k]))+u[0][i][(j-1)][k])))-(ty2_0*(u_6-u_5)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dy2ty1)*((u[1][i][(j+1)][k]-(2.0*u[1][i][j][k]))+u[1][i][(j-1)][k])))+(yycon2_0*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(ty2_0*((u[1][i][(j+1)][k]*vp1)-(u[1][i][(j-1)][k]*vm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dy3ty1)*((u_6-(2.0*u[2][i][j][k]))+u_5)))+((yycon2_0*( * con43))*((vp1-(2.0*vijk))+vm1)))-(ty2_0*(((u_6*vp1)-(u_5*vm1))+((((u_7-square_0)-u_8)+square_1)*c2_0))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dy4ty1)*((u[3][i][(j+1)][k]-(2.0*u[3][i][j][k]))+u[3][i][(j-1)][k])))+(yycon2_0*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(ty2_0*((u[3][i][(j+1)][k]*vp1)-(u[3][i][(j-1)][k]*vm1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dy5ty1)*((u_7-(2.0*u[4][i][j][k]))+u_8)))+(( * yycon3)*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(( * yycon4)*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(( * yycon5)*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_8*rho_i[i][(j-1)][k]))))-(ty2_0*((((c1_0*u_7)-(c2_0*square_0))*vp1)-(((c1_0*u_8)-(c2_0*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel9(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#9#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((5.0*u[m][i][j][k])-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel10(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#10#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((( - 4.0)*u[m][i][(j-1)][k])+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel11(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#11#0 
		for (j=(3*1); j<(((1+162)-(3*1))-1); j ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel12(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#12#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel13(double * dssp, int * j, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#13#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(( * dssp)*((u[m][i][(j_0-2)][k]-(4.0*u[m][i][(j_0-1)][k]))+(5.0*u[m][i][j_0][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel14(double * c1, double * c2, double * con43, double * dz1tz1, double * dz2tz1, double * dz3tz1, double * dz4tz1, double * dz5tz1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tz2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * zzcon2, double * zzcon3, double * zzcon4, double * zzcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tz2_0;
	double u_3;
	double u_4;
	double u_5;
	double u_6;
	double zzcon2_0;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	zzcon2_0=( * zzcon2);
	tz2_0=( * tz2);
	c2_0=( * c2);
	c1_0=( * c1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_6=u[3][i][j][(k-1)];
				u_5=u[4][i][j][(k-1)];
				u_4=u[4][i][j][(k+1)];
				u_3=u[3][i][j][(k+1)];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dz1tz1)*((u[0][i][j][(k+1)]-(2.0*u[0][i][j][k]))+u[0][i][j][(k-1)])))-(tz2_0*(u_3-u_6)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dz2tz1)*((u[1][i][j][(k+1)]-(2.0*u[1][i][j][k]))+u[1][i][j][(k-1)])))+(zzcon2_0*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(tz2_0*((u[1][i][j][(k+1)]*wp1)-(u[1][i][j][(k-1)]*wm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dz3tz1)*((u[2][i][j][(k+1)]-(2.0*u[2][i][j][k]))+u[2][i][j][(k-1)])))+(zzcon2_0*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(tz2_0*((u[2][i][j][(k+1)]*wp1)-(u[2][i][j][(k-1)]*wm1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dz4tz1)*((u_3-(2.0*u[3][i][j][k]))+u_6)))+((zzcon2_0*( * con43))*((wp1-(2.0*wijk))+wm1)))-(tz2_0*(((u_3*wp1)-(u_6*wm1))+((((u_4-square_0)-u_5)+square_1)*c2_0))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dz5tz1)*((u_4-(2.0*u[4][i][j][k]))+u_5)))+(( * zzcon3)*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(( * zzcon4)*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(( * zzcon5)*(((u_4*rho_i[i][j][(k+1)])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[i][j][(k-1)]))))-(tz2_0*((((c1_0*u_4)-(c2_0*square_0))*wp1)-(((c1_0*u_5)-(c2_0*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel15(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<(162-1))
	{
#pragma loop name compute_rhs#15#0 
		for (j=1; j<(162-1); j ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*(((5.0*u[m][i][j][k_0])-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel16(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#16#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((((( - 4.0)*u[m][i][j][(k_0-1)])+(6.0*u[m][i][j][k_0]))-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel17(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+3);
	if (k<(((1+162)-(3*1))-1))
	{
#pragma loop name compute_rhs#17#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))+u[m][i][j][(k+2)])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel18(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#18#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel19(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#19#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((u[m][i][j][(k_0-2)]-(4.0*u[m][i][j][(k_0-1)]))+(5.0*u[m][i][j][k_0]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel20(double * dt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#20#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]*( * dt));
				}
			}
		}
	}
}

static void compute_rhs_clnd2(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      compute the reciprocal of density, and the kinetic energy, 
	   c      and the speed of sound. 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for nowait */
	/* trace_start("compute_rhs", 1); */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	int * gpu__i;
	int * gpu__j;
	int * gpu__k;
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
#pragma omp parallel for shared(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) private(aux, i, j, k, rho_inv) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, c1c2, qs, rho_i, speed, spped, square, u, us, vs, ws) nog2cmemtr(ainv, c1c2, qs, rho_i, speed, spped, square, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(u[1][i][j][k], u[2][i][j][k], u[3][i][j][k]) 
#pragma cuda gpurun registerRW(square[i][j][k]) 
	compute_rhs_clnd2_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__c1c2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("compute_rhs", 1); */
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(forcing, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(forcing, rhs) 
#pragma cuda gpurun nocudamalloc(forcing, rhs) 
#pragma cuda gpurun nocudafree(forcing, rhs) 
#pragma cuda gpurun nog2cmemtr(forcing, rhs) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__forcing), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("compute_rhs", 2); */
	/*
	   --------------------------------------------------------------------
	   c      compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 3); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(c1, c2, square[(i+1)][j][k], square[(i-1)][j][k], tx2, u[1][(i+1)][j][k], u[1][(i-1)][j][k], u[4][(i+1)][j][k], u[4][(i-1)][j][k], xxcon2) 
	compute_rhs_clnd2_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dx1tx1, gpu__dx2tx1, gpu__dx3tx1, gpu__dx4tx1, gpu__dx5tx1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tx2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__xxcon2, gpu__xxcon3, gpu__xxcon4, gpu__xxcon5);
	/* trace_stop("compute_rhs", 3); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	i=1;
	/* trace_start("compute_rhs", 4); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd2_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 4); */
	i=2;
	/* trace_start("compute_rhs", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd2_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 5); */
	/* trace_start("compute_rhs", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 6); */
	i=(162-3);
	/* trace_start("compute_rhs", 7); */
	dim3 dimBlock6(gpuNumThreads, 1, 1);
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
	dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd2_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 7); */
	i=(162-2);
	/* trace_start("compute_rhs", 8); */
	dim3 dimBlock7(gpuNumThreads, 1, 1);
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
	dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(i) 
#pragma cuda gpurun cudafree(i) 
	compute_rhs_clnd2_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/* trace_stop("compute_rhs", 8); */
	/* #pragma omp barrier */
	/*
	   --------------------------------------------------------------------
	   c      compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 9); */
	dim3 dimBlock8(gpuNumThreads, 1, 1);
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
	dim3 dimGrid8(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(8) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(c1, c2, square[i][(j+1)][k], square[i][(j-1)][k], ty2, u[2][i][(j+1)][k], u[2][i][(j-1)][k], u[4][i][(j+1)][k], u[4][i][(j-1)][k], yycon2) 
	compute_rhs_clnd2_kernel8<<<dimGrid8, dimBlock8, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dy1ty1, gpu__dy2ty1, gpu__dy3ty1, gpu__dy4ty1, gpu__dy5ty1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__ty2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__yycon2, gpu__yycon3, gpu__yycon4, gpu__yycon5);
	/* trace_stop("compute_rhs", 9); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	j=1;
	/* trace_start("compute_rhs", 10); */
	dim3 dimBlock9(gpuNumThreads, 1, 1);
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
	dim3 dimGrid9(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(9) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel9<<<dimGrid9, dimBlock9, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 10); */
	j=2;
	/* trace_start("compute_rhs", 11); */
	dim3 dimBlock10(gpuNumThreads, 1, 1);
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
	dim3 dimGrid10(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(10) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel10<<<dimGrid10, dimBlock10, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 11); */
	/* trace_start("compute_rhs", 12); */
	dim3 dimBlock11(gpuNumThreads, 1, 1);
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
	dim3 dimGrid11(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(11) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel11<<<dimGrid11, dimBlock11, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 12); */
	j=(162-3);
	/* trace_start("compute_rhs", 13); */
	dim3 dimBlock12(gpuNumThreads, 1, 1);
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
	dim3 dimGrid12(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(12) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel12<<<dimGrid12, dimBlock12, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 13); */
	j=(162-2);
	/* trace_start("compute_rhs", 14); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
	dim3 dimBlock13(gpuNumThreads, 1, 1);
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
	dim3 dimGrid13(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, j, rhs, u) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, j, rhs, u) 
#pragma cuda ainfo kernelid(13) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(j) 
#pragma cuda gpurun cudafree(j) 
	compute_rhs_clnd2_kernel13<<<dimGrid13, dimBlock13, 0, 0>>>(gpu__dssp, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	/* trace_stop("compute_rhs", 14); */
	/* #pragma omp barrier   */
	/*
	   --------------------------------------------------------------------
	   c      compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 15); */
	dim3 dimBlock14(gpuNumThreads, 1, 1);
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
	dim3 dimGrid14(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, j, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nog2cmemtr(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(14) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(c1, c2, square[i][j][(k+1)], square[i][j][(k-1)], tz2, u[3][i][j][(k+1)], u[3][i][j][(k-1)], u[4][i][j][(k+1)], u[4][i][j][(k-1)], zzcon2) 
	compute_rhs_clnd2_kernel14<<<dimGrid14, dimBlock14, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dz1tz1, gpu__dz2tz1, gpu__dz3tz1, gpu__dz4tz1, gpu__dz5tz1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tz2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__zzcon2, gpu__zzcon3, gpu__zzcon4, gpu__zzcon5);
	/* trace_stop("compute_rhs", 15); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	k=1;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
	dim3 dimBlock15(gpuNumThreads, 1, 1);
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
	dim3 dimGrid15(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(15) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_clnd2_kernel15<<<dimGrid15, dimBlock15, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 16); */
	k=2;
	/* trace_start("compute_rhs", 17); */
	dim3 dimBlock16(gpuNumThreads, 1, 1);
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
	dim3 dimGrid16(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(16) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_clnd2_kernel16<<<dimGrid16, dimBlock16, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 17); */
	/* trace_start("compute_rhs", 18); */
	dim3 dimBlock17(gpuNumThreads, 1, 1);
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
	dim3 dimGrid17(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(17) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel17<<<dimGrid17, dimBlock17, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 18); */
	k=(162-3);
	/* trace_start("compute_rhs", 19); */
	dim3 dimBlock18(gpuNumThreads, 1, 1);
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
	dim3 dimGrid18(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(18) procname(compute_rhs_clnd2) 
	compute_rhs_clnd2_kernel18<<<dimGrid18, dimBlock18, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 19); */
	k=(162-2);
	/* trace_start("compute_rhs", 20); */
	dim3 dimBlock19(gpuNumThreads, 1, 1);
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
	dim3 dimGrid19(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(19) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(k) 
#pragma cuda gpurun cudafree(k) 
	compute_rhs_clnd2_kernel19<<<dimGrid19, dimBlock19, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("compute_rhs", 20); */
	/* trace_start("compute_rhs", 21); */
	dim3 dimBlock20(gpuNumThreads, 1, 1);
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
	dim3 dimGrid20(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dt, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dt, rhs) 
#pragma cuda gpurun nocudamalloc(dt, rhs) 
#pragma cuda gpurun nocudafree(dt, rhs) 
#pragma cuda ainfo kernelid(20) procname(compute_rhs_clnd2) 
#pragma cuda gpurun nog2cmemtr(dt) 
	compute_rhs_clnd2_kernel20<<<dimGrid20, dimBlock20, 0, 0>>>(gpu__dt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	gpuBytes=((((5*(((162/2)*2)+1))*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	/* trace_stop("compute_rhs", 21); */
	return ;
}

__global__ void compute_rhs_clnd1_cloned0_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * c1c2, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double square_0;
	double u_0;
	double u_1;
	double u_2;
	double aux;
	int i;
	int j;
	int k;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
				u_2=u[2][i][j][k];
				u_1=u[3][i][j][k];
				u_0=u[1][i][j][k];
				square_0=square[i][j][k];
				rho_inv=(1.0/u[0][i][j][k]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_0*rho_inv);
				vs[i][j][k]=(u_2*rho_inv);
				ws[i][j][k]=(u_1*rho_inv);
				square_0=((0.5*(((u_0*u_0)+(u_2*u_2))+(u_1*u_1)))*rho_inv);
				qs[i][j][k]=(square_0*rho_inv);
				/*
				   --------------------------------------------------------------------
				   c               (do not need speed and ainx until the lhs computation)
				   c-------------------------------------------------------------------
				 */
				aux=((( * c1c2)*rho_inv)*(u[4][i][j][k]-square_0));
				/* aux = sqrt(aux); */
				speed[i][j][k]=aux;
				ainv[i][j][k]=(1.0/aux);
				square[i][j][k]=square_0;
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel1(double forcing[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=_gtid;
	if (k<((1+162)-1))
	{
#pragma loop name compute_rhs#1#0 
		for (j=0; j<((1+162)-1); j ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (i=0; i<((1+162)-1); i ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=forcing[m][i][j][k];
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel2(double * c1, double * c2, double * con43, double * dx1tx1, double * dx2tx1, double * dx3tx1, double * dx4tx1, double * dx5tx1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tx2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * xxcon2, double * xxcon3, double * xxcon4, double * xxcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tx2_0;
	double u_0;
	double u_5;
	double u_7;
	double u_8;
	double xxcon2_0;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	c1_0=( * c1);
	c2_0=( * c2);
	xxcon2_0=( * xxcon2);
	tx2_0=( * tx2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[1][(i-1)][j][k];
				u_7=u[4][(i+1)][j][k];
				u_5=u[4][(i-1)][j][k];
				u_0=u[1][(i+1)][j][k];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dx1tx1)*((u[0][(i+1)][j][k]-(2.0*u[0][i][j][k]))+u[0][(i-1)][j][k])))-(tx2_0*(u_0-u_8)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dx2tx1)*((u_0-(2.0*u[1][i][j][k]))+u_8)))+((xxcon2_0*( * con43))*((up1-(2.0*uijk))+um1)))-(tx2_0*(((u_0*up1)-(u_8*um1))+((((u_7-square_1)-u_5)+square_0)*c2_0))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dx3tx1)*((u[2][(i+1)][j][k]-(2.0*u[2][i][j][k]))+u[2][(i-1)][j][k])))+(xxcon2_0*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(tx2_0*((u[2][(i+1)][j][k]*up1)-(u[2][(i-1)][j][k]*um1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dx4tx1)*((u[3][(i+1)][j][k]-(2.0*u[3][i][j][k]))+u[3][(i-1)][j][k])))+(xxcon2_0*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(tx2_0*((u[3][(i+1)][j][k]*up1)-(u[3][(i-1)][j][k]*um1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dx5tx1)*((u_7-(2.0*u[4][i][j][k]))+u_5)))+(( * xxcon3)*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(( * xxcon4)*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(( * xxcon5)*(((u_7*rho_i[(i+1)][j][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[(i-1)][j][k]))))-(tx2_0*((((c1_0*u_7)-(c2_0*square_1))*up1)-(((c1_0*u_5)-(c2_0*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel3(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
		/* #pragma omp for */
#pragma loop name compute_rhs#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((5.0*u[m][i_0][j][k])-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel4(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((((( - 4.0)*u[m][(i_0-1)][j][k])+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))+u[m][(i_0+2)][j][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel5(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#5#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (i=(3*1); i<(((1+162)-(3*1))-1); i ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][(i-2)][j][k]-(4.0*u[m][(i-1)][j][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][(i+1)][j][k]))+u[m][(i+2)][j][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel6(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#6#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*(((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(6.0*u[m][i_0][j][k]))-(4.0*u[m][(i_0+1)][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel7(double * dssp, int * i, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#7#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(( * dssp)*((u[m][(i_0-2)][j][k]-(4.0*u[m][(i_0-1)][j][k]))+(5.0*u[m][i_0][j][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel8(double * c1, double * c2, double * con43, double * dy1ty1, double * dy2ty1, double * dy3ty1, double * dy4ty1, double * dy5ty1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * ty2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * yycon2, double * yycon3, double * yycon4, double * yycon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double ty2_0;
	double u_5;
	double u_6;
	double u_7;
	double u_8;
	double yycon2_0;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	yycon2_0=( * yycon2);
	c2_0=( * c2);
	c1_0=( * c1);
	ty2_0=( * ty2);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_8=u[4][i][(j-1)][k];
				u_7=u[4][i][(j+1)][k];
				u_6=u[2][i][(j+1)][k];
				u_5=u[2][i][(j-1)][k];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dy1ty1)*((u[0][i][(j+1)][k]-(2.0*u[0][i][j][k]))+u[0][i][(j-1)][k])))-(ty2_0*(u_6-u_5)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dy2ty1)*((u[1][i][(j+1)][k]-(2.0*u[1][i][j][k]))+u[1][i][(j-1)][k])))+(yycon2_0*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(ty2_0*((u[1][i][(j+1)][k]*vp1)-(u[1][i][(j-1)][k]*vm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dy3ty1)*((u_6-(2.0*u[2][i][j][k]))+u_5)))+((yycon2_0*( * con43))*((vp1-(2.0*vijk))+vm1)))-(ty2_0*(((u_6*vp1)-(u_5*vm1))+((((u_7-square_0)-u_8)+square_1)*c2_0))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dy4ty1)*((u[3][i][(j+1)][k]-(2.0*u[3][i][j][k]))+u[3][i][(j-1)][k])))+(yycon2_0*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(ty2_0*((u[3][i][(j+1)][k]*vp1)-(u[3][i][(j-1)][k]*vm1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dy5ty1)*((u_7-(2.0*u[4][i][j][k]))+u_8)))+(( * yycon3)*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(( * yycon4)*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(( * yycon5)*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_8*rho_i[i][(j-1)][k]))))-(ty2_0*((((c1_0*u_7)-(c2_0*square_0))*vp1)-(((c1_0*u_8)-(c2_0*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel9(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#9#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((5.0*u[m][i][j][k])-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel10(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#10#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((( - 4.0)*u[m][i][(j-1)][k])+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel11(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#11#0 
		for (j=(3*1); j<(((1+162)-(3*1))-1); j ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))+u[m][i][(j+2)][k])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel12(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#12#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][(j-2)][k]-(4.0*u[m][i][(j-1)][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][(j+1)][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel13(double * dssp, int * j, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#13#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(( * dssp)*((u[m][i][(j_0-2)][k]-(4.0*u[m][i][(j_0-1)][k]))+(5.0*u[m][i][j_0][k]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel14(double * c1, double * c2, double * con43, double * dz1tz1, double * dz2tz1, double * dz3tz1, double * dz4tz1, double * dz5tz1, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * tz2, double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * zzcon2, double * zzcon3, double * zzcon4, double * zzcon5)
{
	double c1_0;
	double c2_0;
	double square_0;
	double square_1;
	double tz2_0;
	double u_3;
	double u_4;
	double u_5;
	double u_6;
	double zzcon2_0;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	tz2_0=( * tz2);
	zzcon2_0=( * zzcon2);
	c2_0=( * c2);
	c1_0=( * c1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				u_6=u[3][i][j][(k-1)];
				u_5=u[4][i][j][(k-1)];
				u_4=u[4][i][j][(k+1)];
				u_3=u[3][i][j][(k+1)];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(( * dz1tz1)*((u[0][i][j][(k+1)]-(2.0*u[0][i][j][k]))+u[0][i][j][(k-1)])))-(tz2_0*(u_3-u_6)));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(( * dz2tz1)*((u[1][i][j][(k+1)]-(2.0*u[1][i][j][k]))+u[1][i][j][(k-1)])))+(zzcon2_0*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(tz2_0*((u[1][i][j][(k+1)]*wp1)-(u[1][i][j][(k-1)]*wm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(( * dz3tz1)*((u[2][i][j][(k+1)]-(2.0*u[2][i][j][k]))+u[2][i][j][(k-1)])))+(zzcon2_0*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(tz2_0*((u[2][i][j][(k+1)]*wp1)-(u[2][i][j][(k-1)]*wm1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(( * dz4tz1)*((u_3-(2.0*u[3][i][j][k]))+u_6)))+((zzcon2_0*( * con43))*((wp1-(2.0*wijk))+wm1)))-(tz2_0*(((u_3*wp1)-(u_6*wm1))+((((u_4-square_0)-u_5)+square_1)*c2_0))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(( * dz5tz1)*((u_4-(2.0*u[4][i][j][k]))+u_5)))+(( * zzcon3)*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(( * zzcon4)*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(( * zzcon5)*(((u_4*rho_i[i][j][(k+1)])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u_5*rho_i[i][j][(k-1)]))))-(tz2_0*((((c1_0*u_4)-(c2_0*square_0))*wp1)-(((c1_0*u_5)-(c2_0*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel15(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<(162-1))
	{
#pragma loop name compute_rhs#15#0 
		for (j=1; j<(162-1); j ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*(((5.0*u[m][i][j][k_0])-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel16(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#16#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((((( - 4.0)*u[m][i][j][(k_0-1)])+(6.0*u[m][i][j][k_0]))-(4.0*u[m][i][j][(k_0+1)]))+u[m][i][j][(k_0+2)])));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel17(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+3);
	if (k<(((1+162)-(3*1))-1))
	{
#pragma loop name compute_rhs#17#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*((((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))+u[m][i][j][(k+2)])));
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel18(double * dssp, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#18#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(( * dssp)*(((u[m][i][j][(k-2)]-(4.0*u[m][i][j][(k-1)]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][(k+1)]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel19(double * dssp, int * k, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name compute_rhs#19#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(( * dssp)*((u[m][i][j][(k_0-2)]-(4.0*u[m][i][j][(k_0-1)]))+(5.0*u[m][i][j][k_0]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_cloned0_kernel20(double * dt, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name compute_rhs#20#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]*( * dt));
				}
			}
		}
	}
}

static void compute_rhs_clnd1_cloned0(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	/*
	   --------------------------------------------------------------------
	   c      compute the reciprocal of density, and the kinetic energy, 
	   c      and the speed of sound. 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for nowait */
	/* trace_start("compute_rhs", 1); */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	int * gpu__i;
	int * gpu__j;
	int * gpu__k;
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
#pragma omp parallel for shared(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) private(aux, i, j, k, rho_inv) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, c1c2, qs, rho_i, speed, spped, square, u, us, vs, ws) nog2cmemtr(ainv, c1c2, qs, rho_i, speed, spped, square, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, c1c2, qs, rho_i, speed, square, u, us, vs, ws) 
#pragma cuda gpurun multisrccg(u) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(u[1][i][j][k], u[2][i][j][k], u[3][i][j][k]) 
#pragma cuda gpurun registerRW(square[i][j][k]) 
	compute_rhs_clnd1_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__c1c2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("compute_rhs", 1); */
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(forcing, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(forcing, rhs) 
#pragma cuda gpurun nocudamalloc(forcing, rhs) 
#pragma cuda gpurun nocudafree(forcing, rhs) 
#pragma cuda gpurun nog2cmemtr(forcing, rhs) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__forcing), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("compute_rhs", 2); */
	/*
	   --------------------------------------------------------------------
	   c      compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 3); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(c1, c2, square[(i+1)][j][k], square[(i-1)][j][k], tx2, u[1][(i+1)][j][k], u[1][(i-1)][j][k], u[4][(i+1)][j][k], u[4][(i-1)][j][k], xxcon2) 
	compute_rhs_clnd1_cloned0_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dx1tx1, gpu__dx2tx1, gpu__dx3tx1, gpu__dx4tx1, gpu__dx5tx1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tx2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__xxcon2, gpu__xxcon3, gpu__xxcon4, gpu__xxcon5);
	/* trace_stop("compute_rhs", 3); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	i=1;
	/* trace_start("compute_rhs", 4); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd1_cloned0_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 4); */
	i=2;
	/* trace_start("compute_rhs", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd1_cloned0_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 5); */
	/* trace_start("compute_rhs", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 6); */
	i=(162-3);
	/* trace_start("compute_rhs", 7); */
	dim3 dimBlock6(gpuNumThreads, 1, 1);
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
	dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, i, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(i) 
	compute_rhs_clnd1_cloned0_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 7); */
	i=(162-2);
	/* trace_start("compute_rhs", 8); */
	dim3 dimBlock7(gpuNumThreads, 1, 1);
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
	dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, i, rhs, u) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, i, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, i, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, i, rhs, u) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(i) 
#pragma cuda gpurun cudafree(i) 
	compute_rhs_clnd1_cloned0_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__dssp, gpu__i, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/* trace_stop("compute_rhs", 8); */
	/* #pragma omp barrier */
	/*
	   --------------------------------------------------------------------
	   c      compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 9); */
	dim3 dimBlock8(gpuNumThreads, 1, 1);
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
	dim3 dimGrid8(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nog2cmemtr(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(8) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(c1, c2, square[i][(j+1)][k], square[i][(j-1)][k], ty2, u[2][i][(j+1)][k], u[2][i][(j-1)][k], u[4][i][(j+1)][k], u[4][i][(j-1)][k], yycon2) 
	compute_rhs_clnd1_cloned0_kernel8<<<dimGrid8, dimBlock8, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dy1ty1, gpu__dy2ty1, gpu__dy3ty1, gpu__dy4ty1, gpu__dy5ty1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__ty2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__yycon2, gpu__yycon3, gpu__yycon4, gpu__yycon5);
	/* trace_stop("compute_rhs", 9); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	j=1;
	/* trace_start("compute_rhs", 10); */
	dim3 dimBlock9(gpuNumThreads, 1, 1);
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
	dim3 dimGrid9(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(9) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel9<<<dimGrid9, dimBlock9, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 10); */
	j=2;
	/* trace_start("compute_rhs", 11); */
	dim3 dimBlock10(gpuNumThreads, 1, 1);
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
	dim3 dimGrid10(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(10) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel10<<<dimGrid10, dimBlock10, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 11); */
	/* trace_start("compute_rhs", 12); */
	dim3 dimBlock11(gpuNumThreads, 1, 1);
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
	dim3 dimGrid11(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(11) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel11<<<dimGrid11, dimBlock11, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 12); */
	j=(162-3);
	/* trace_start("compute_rhs", 13); */
	dim3 dimBlock12(gpuNumThreads, 1, 1);
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
	dim3 dimGrid12(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(12) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel12<<<dimGrid12, dimBlock12, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 13); */
	j=(162-2);
	/* trace_start("compute_rhs", 14); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
	dim3 dimBlock13(gpuNumThreads, 1, 1);
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
	dim3 dimGrid13(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for num_threads(5) shared(dssp, j, rhs, u) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, j, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, j, rhs, u) 
#pragma cuda ainfo kernelid(13) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(j) 
#pragma cuda gpurun cudafree(j) 
	compute_rhs_clnd1_cloned0_kernel13<<<dimGrid13, dimBlock13, 0, 0>>>(gpu__dssp, gpu__j, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	/* trace_stop("compute_rhs", 14); */
	/* #pragma omp barrier   */
	/*
	   --------------------------------------------------------------------
	   c      compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	/* #pragma omp for */
	/* trace_start("compute_rhs", 15); */
	dim3 dimBlock14(gpuNumThreads, 1, 1);
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
	dim3 dimGrid14(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1) schedule(static)
#pragma cuda gpurun noc2gmemtr(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, j, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nog2cmemtr(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudafree(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(14) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(c1, c2, square[i][j][(k+1)], square[i][j][(k-1)], tz2, u[3][i][j][(k+1)], u[3][i][j][(k-1)], u[4][i][j][(k+1)], u[4][i][j][(k-1)], zzcon2) 
	compute_rhs_clnd1_cloned0_kernel14<<<dimGrid14, dimBlock14, 0, 0>>>(gpu__c1, gpu__c2, gpu__con43, gpu__dz1tz1, gpu__dz2tz1, gpu__dz3tz1, gpu__dz4tz1, gpu__dz5tz1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), gpu__tz2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws), gpu__zzcon2, gpu__zzcon3, gpu__zzcon4, gpu__zzcon5);
	/* trace_stop("compute_rhs", 15); */
	/*
	   --------------------------------------------------------------------
	   c      add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	k=1;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
	dim3 dimBlock15(gpuNumThreads, 1, 1);
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
	dim3 dimGrid15(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(15) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_clnd1_cloned0_kernel15<<<dimGrid15, dimBlock15, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 16); */
	k=2;
	/* trace_start("compute_rhs", 17); */
	dim3 dimBlock16(gpuNumThreads, 1, 1);
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
	dim3 dimGrid16(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, k, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(16) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(k) 
	compute_rhs_clnd1_cloned0_kernel16<<<dimGrid16, dimBlock16, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 17); */
	/* trace_start("compute_rhs", 18); */
	dim3 dimBlock17(gpuNumThreads, 1, 1);
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
	dim3 dimGrid17(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(17) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel17<<<dimGrid17, dimBlock17, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 18); */
	k=(162-3);
	/* trace_start("compute_rhs", 19); */
	dim3 dimBlock18(gpuNumThreads, 1, 1);
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
	dim3 dimGrid18(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, rhs, u) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, rhs, u) 
#pragma cuda ainfo kernelid(18) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel18<<<dimGrid18, dimBlock18, 0, 0>>>(gpu__dssp, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	/* trace_stop("compute_rhs", 19); */
	k=(162-2);
	/* trace_start("compute_rhs", 20); */
	dim3 dimBlock19(gpuNumThreads, 1, 1);
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
	dim3 dimGrid19(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dssp, k, rhs, u) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dssp, k, rhs, u) 
#pragma cuda gpurun nocudamalloc(dssp, k, rhs, u) 
#pragma cuda gpurun nocudafree(dssp, rhs, u) 
#pragma cuda gpurun nog2cmemtr(dssp, k, rhs, u) 
#pragma cuda ainfo kernelid(19) procname(compute_rhs_clnd1_cloned0) 
#pragma cuda gpurun registerRO(k) 
#pragma cuda gpurun cudafree(k) 
	compute_rhs_clnd1_cloned0_kernel19<<<dimGrid19, dimBlock19, 0, 0>>>(gpu__dssp, gpu__k, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("compute_rhs", 20); */
	/* trace_start("compute_rhs", 21); */
	dim3 dimBlock20(gpuNumThreads, 1, 1);
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
	dim3 dimGrid20(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(dt, rhs) private(i, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(dt, rhs) 
#pragma cuda gpurun nocudamalloc(dt, rhs) 
#pragma cuda gpurun nocudafree(dt, rhs) 
#pragma cuda gpurun nog2cmemtr(dt, rhs) 
#pragma cuda ainfo kernelid(20) procname(compute_rhs_clnd1_cloned0) 
	compute_rhs_clnd1_cloned0_kernel20<<<dimGrid20, dimBlock20, 0, 0>>>(gpu__dt, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("compute_rhs", 21); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void set_constants(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	ce[0][0]=2.0;
	ce[1][0]=0.0;
	ce[2][0]=0.0;
	ce[3][0]=4.0;
	ce[4][0]=5.0;
	ce[5][0]=3.0;
	ce[6][0]=0.5;
	ce[7][0]=0.02;
	ce[8][0]=0.01;
	ce[9][0]=0.03;
	ce[10][0]=0.5;
	ce[11][0]=0.4;
	ce[12][0]=0.3;
	ce[0][1]=1.0;
	ce[1][1]=0.0;
	ce[2][1]=0.0;
	ce[3][1]=0.0;
	ce[4][1]=1.0;
	ce[5][1]=2.0;
	ce[6][1]=3.0;
	ce[7][1]=0.01;
	ce[8][1]=0.03;
	ce[9][1]=0.02;
	ce[10][1]=0.4;
	ce[11][1]=0.3;
	ce[12][1]=0.5;
	ce[0][2]=2.0;
	ce[1][2]=2.0;
	ce[2][2]=0.0;
	ce[3][2]=0.0;
	ce[4][2]=0.0;
	ce[5][2]=2.0;
	ce[6][2]=3.0;
	ce[7][2]=0.04;
	ce[8][2]=0.03;
	ce[9][2]=0.05;
	ce[10][2]=0.3;
	ce[11][2]=0.5;
	ce[12][2]=0.4;
	ce[0][3]=2.0;
	ce[1][3]=2.0;
	ce[2][3]=0.0;
	ce[3][3]=0.0;
	ce[4][3]=0.0;
	ce[5][3]=2.0;
	ce[6][3]=3.0;
	ce[7][3]=0.03;
	ce[8][3]=0.05;
	ce[9][3]=0.04;
	ce[10][3]=0.2;
	ce[11][3]=0.1;
	ce[12][3]=0.3;
	ce[0][4]=5.0;
	ce[1][4]=4.0;
	ce[2][4]=3.0;
	ce[3][4]=2.0;
	ce[4][4]=0.1;
	ce[5][4]=0.4;
	ce[6][4]=0.3;
	ce[7][4]=0.05;
	ce[8][4]=0.04;
	ce[9][4]=0.03;
	ce[10][4]=0.1;
	ce[11][4]=0.3;
	ce[12][4]=0.2;
	c1=1.4;
	c2=0.4;
	c3=0.1;
	c4=1.0;
	c5=1.4;
	bt=sqrt(0.5);
	dnxm1=(1.0/((double)(162-1)));
	dnym1=(1.0/((double)(162-1)));
	dnzm1=(1.0/((double)(162-1)));
	c1c2=(c1*c2);
	c1c5=(c1*c5);
	c3c4=(c3*c4);
	c1345=(c1c5*c3c4);
	conz1=(1.0-c1c5);
	tx1=(1.0/(dnxm1*dnxm1));
	tx2=(1.0/(2.0*dnxm1));
	tx3=(1.0/dnxm1);
	ty1=(1.0/(dnym1*dnym1));
	ty2=(1.0/(2.0*dnym1));
	ty3=(1.0/dnym1);
	tz1=(1.0/(dnzm1*dnzm1));
	tz2=(1.0/(2.0*dnzm1));
	tz3=(1.0/dnzm1);
	dx1=0.75;
	dx2=0.75;
	dx3=0.75;
	dx4=0.75;
	dx5=0.75;
	dy1=0.75;
	dy2=0.75;
	dy3=0.75;
	dy4=0.75;
	dy5=0.75;
	dz1=1.0;
	dz2=1.0;
	dz3=1.0;
	dz4=1.0;
	dz5=1.0;
	dxmax=((dx3>dx4) ? dx3 : dx4);
	dymax=((dy2>dy4) ? dy2 : dy4);
	dzmax=((dz2>dz3) ? dz2 : dz3);
	dssp=(0.25*((dx1>((dy1>dz1) ? dy1 : dz1)) ? dx1 : ((dy1>dz1) ? dy1 : dz1)));
	c4dssp=(4.0*dssp);
	c5dssp=(5.0*dssp);
	dttx1=(dt*tx1);
	dttx2=(dt*tx2);
	dtty1=(dt*ty1);
	dtty2=(dt*ty2);
	dttz1=(dt*tz1);
	dttz2=(dt*tz2);
	c2dttx1=(2.0*dttx1);
	c2dtty1=(2.0*dtty1);
	c2dttz1=(2.0*dttz1);
	dtdssp=(dt*dssp);
	comz1=dtdssp;
	comz4=(4.0*dtdssp);
	comz5=(5.0*dtdssp);
	comz6=(6.0*dtdssp);
	c3c4tx3=(c3c4*tx3);
	c3c4ty3=(c3c4*ty3);
	c3c4tz3=(c3c4*tz3);
	dx1tx1=(dx1*tx1);
	dx2tx1=(dx2*tx1);
	dx3tx1=(dx3*tx1);
	dx4tx1=(dx4*tx1);
	dx5tx1=(dx5*tx1);
	dy1ty1=(dy1*ty1);
	dy2ty1=(dy2*ty1);
	dy3ty1=(dy3*ty1);
	dy4ty1=(dy4*ty1);
	dy5ty1=(dy5*ty1);
	dz1tz1=(dz1*tz1);
	dz2tz1=(dz2*tz1);
	dz3tz1=(dz3*tz1);
	dz4tz1=(dz4*tz1);
	dz5tz1=(dz5*tz1);
	c2iv=2.5;
	con43=(4.0/3.0);
	con16=(1.0/6.0);
	xxcon1=((c3c4tx3*con43)*tx3);
	xxcon2=(c3c4tx3*tx3);
	xxcon3=((c3c4tx3*conz1)*tx3);
	xxcon4=((c3c4tx3*con16)*tx3);
	xxcon5=((c3c4tx3*c1c5)*tx3);
	yycon1=((c3c4ty3*con43)*ty3);
	yycon2=(c3c4ty3*ty3);
	yycon3=((c3c4ty3*conz1)*ty3);
	yycon4=((c3c4ty3*con16)*ty3);
	yycon5=((c3c4ty3*c1c5)*ty3);
	zzcon1=((c3c4tz3*con43)*tz3);
	zzcon2=(c3c4tz3*tz3);
	zzcon3=((c3c4tz3*conz1)*tz3);
	zzcon4=((c3c4tz3*con16)*tz3);
	zzcon5=((c3c4tz3*c1c5)*tz3);
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void txinvr_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * bt, double * c2, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	double ac;
	double ac2inv;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double ru1;
	double t1;
	double t2;
	double t3;
	double uu;
	double vv;
	double ww;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name txinvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name txinvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				ru1=rho_i[i][j][k];
				uu=us[i][j][k];
				vv=vs[i][j][k];
				ww=ws[i][j][k];
				ac=speed[i][j][k];
				ac2inv=(ainv[i][j][k]*ainv[i][j][k]);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=((( * c2)*ac2inv)*(((((qs[i][j][k]*r1)-(uu*r2))-(vv*r3))-(ww*r4))+r5));
				t2=((bt_0*ru1)*((uu*r1)-r2));
				t3=(((bt_0*ru1)*ac)*t1);
				rhs[0][i][j][k]=(r1-t1);
				rhs[1][i][j][k]=(( - ru1)*((ww*r1)-r4));
				rhs[2][i][j][k]=(ru1*((vv*r1)-r3));
				rhs[3][i][j][k]=(( - t2)+t3);
				rhs[4][i][j][k]=(t2+t3);
			}
		}
	}
}

static void txinvr(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c block-diagonal matrix-vector multiplication                  
	   --------------------------------------------------------------------
	 */
	/* trace_start("txinvr", 1); */
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
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__bt, ( & bt), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) private(ac, ac2inv, i, j, k, r1, r2, r3, r4, r5, ru1, t1, t2, t3, uu, vv, ww) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nog2cmemtr(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(txinvr) 
#pragma cuda gpurun registerRO(bt) 
	txinvr_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__bt, gpu__c2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("txinvr", 1); */
	return ;
}

__global__ void txinvr_clnd1_cloned0_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * bt, double * c2, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double bt_0;
	double ac;
	double ac2inv;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double ru1;
	double t1;
	double t2;
	double t3;
	double uu;
	double vv;
	double ww;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	bt_0=( * bt);
	if (k<((1+162)-2))
	{
#pragma loop name txinvr#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name txinvr#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				ru1=rho_i[i][j][k];
				uu=us[i][j][k];
				vv=vs[i][j][k];
				ww=ws[i][j][k];
				ac=speed[i][j][k];
				ac2inv=(ainv[i][j][k]*ainv[i][j][k]);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=((( * c2)*ac2inv)*(((((qs[i][j][k]*r1)-(uu*r2))-(vv*r3))-(ww*r4))+r5));
				t2=((bt_0*ru1)*((uu*r1)-r2));
				t3=(((bt_0*ru1)*ac)*t1);
				rhs[0][i][j][k]=(r1-t1);
				rhs[1][i][j][k]=(( - ru1)*((ww*r1)-r4));
				rhs[2][i][j][k]=(ru1*((vv*r1)-r3));
				rhs[3][i][j][k]=(( - t2)+t3);
				rhs[4][i][j][k]=(t2+t3);
			}
		}
	}
}

static void txinvr_clnd1_cloned0(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c block-diagonal matrix-vector multiplication                  
	   --------------------------------------------------------------------
	 */
	/* trace_start("txinvr", 1); */
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
#pragma omp parallel for shared(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) private(ac, ac2inv, i, j, k, r1, r2, r3, r4, r5, ru1, t1, t2, t3, uu, vv, ww) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda gpurun nog2cmemtr(ainv, bt, c2, qs, rho_i, rhs, speed, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(txinvr_clnd1_cloned0) 
#pragma cuda gpurun registerRO(bt) 
	txinvr_clnd1_cloned0_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__bt, gpu__c2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("txinvr", 1); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void tzetar_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * bt, double * c2iv, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double ac;
	double ac2u;
	double acinv;
	double btuz;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	double t3;
	double uzik1;
	double xvel;
	double yvel;
	double zvel;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name tzetar#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name tzetar#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				xvel=us[i][j][k];
				yvel=vs[i][j][k];
				zvel=ws[i][j][k];
				ac=speed[i][j][k];
				acinv=ainv[i][j][k];
				ac2u=(ac*ac);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				uzik1=u[0][i][j][k];
				btuz=(( * bt)*uzik1);
				t1=((btuz*acinv)*(r4+r5));
				t2=(r3+t1);
				t3=(btuz*(r4-r5));
				rhs[0][i][j][k]=t2;
				rhs[1][i][j][k]=((( - uzik1)*r2)+(xvel*t2));
				rhs[2][i][j][k]=((uzik1*r1)+(yvel*t2));
				rhs[3][i][j][k]=((zvel*t2)+t3);
				rhs[4][i][j][k]=((((uzik1*((( - xvel)*r2)+(yvel*r1)))+(qs[i][j][k]*t2))+((( * c2iv)*ac2u)*t1))+(zvel*t3));
			}
		}
	}
}

static void tzetar(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication                       
	   c-------------------------------------------------------------------
	 */
	/*  trace_start("tzetar", 1); */
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
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__c2iv, ( & c2iv), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) private(ac, ac2u, acinv, btuz, i, j, k, r1, r2, r3, r4, r5, t1, t2, t3, uzik1, xvel, yvel, zvel) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, bt, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, bt, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nog2cmemtr(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(tzetar) 
	tzetar_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__bt, gpu__c2iv, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("tzetar", 1); */
	return ;
}

__global__ void tzetar_clnd1_cloned1_kernel0(double ainv[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double * bt, double * c2iv, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double speed[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double ac;
	double ac2u;
	double acinv;
	double btuz;
	int i;
	int j;
	int k;
	double r1;
	double r2;
	double r3;
	double r4;
	double r5;
	double t1;
	double t2;
	double t3;
	double uzik1;
	double xvel;
	double yvel;
	double zvel;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	if (k<((1+162)-2))
	{
#pragma loop name tzetar#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name tzetar#0#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
				xvel=us[i][j][k];
				yvel=vs[i][j][k];
				zvel=ws[i][j][k];
				ac=speed[i][j][k];
				acinv=ainv[i][j][k];
				ac2u=(ac*ac);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				uzik1=u[0][i][j][k];
				btuz=(( * bt)*uzik1);
				t1=((btuz*acinv)*(r4+r5));
				t2=(r3+t1);
				t3=(btuz*(r4-r5));
				rhs[0][i][j][k]=t2;
				rhs[1][i][j][k]=((( - uzik1)*r2)+(xvel*t2));
				rhs[2][i][j][k]=((uzik1*r1)+(yvel*t2));
				rhs[3][i][j][k]=((zvel*t2)+t3);
				rhs[4][i][j][k]=((((uzik1*((( - xvel)*r2)+(yvel*r1)))+(qs[i][j][k]*t2))+((( * c2iv)*ac2u)*t1))+(zvel*t3));
			}
		}
	}
}

static void tzetar_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c   block-diagonal matrix-vector multiplication                       
	   c-------------------------------------------------------------------
	 */
	/*  trace_start("tzetar", 1); */
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
#pragma omp parallel for shared(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) private(ac, ac2u, acinv, btuz, i, j, k, r1, r2, r3, r4, r5, t1, t2, t3, uzik1, xvel, yvel, zvel) schedule(static)
#pragma cuda gpurun noc2gmemtr(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nocudafree(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda gpurun nog2cmemtr(ainv, bt, c2iv, qs, rhs, speed, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(tzetar_clnd1_cloned1) 
	tzetar_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ainv), gpu__bt, gpu__c2iv, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__speed), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	/* trace_stop("tzetar", 1); */
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void verify(int no_time_steps, char * cclass, int * verified)
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c  verification routine                         
	   --------------------------------------------------------------------
	 */
	double xcrref[5];
	double xceref[5];
	double xcrdif[5];
	double xcedif[5];
	double epsilon;
	double xce[5];
	double xcr[5];
	double dtref;
	int m;
	/*
	   --------------------------------------------------------------------
	   c   tolerance level
	   --------------------------------------------------------------------
	 */
	epsilon=1.0E-8;
	/*
	   --------------------------------------------------------------------
	   c   compute the error norm and the residual norm, and exit if not printing
	   --------------------------------------------------------------------
	 */
	error_norm(xce);
	compute_rhs_clnd2();
	CUDA_SAFE_CALL(cudaMemcpy(rhs, gpu__rhs, gpuBytes, cudaMemcpyDeviceToHost));
	rhs_norm(xcr);
#pragma loop name verify#0 
	for (m=0; m<5; m ++ )
	{
		xcr[m]=(xcr[m]/dt);
	}
	( * cclass)='U';
	( * verified)=1;
#pragma loop name verify#1 
	for (m=0; m<5; m ++ )
	{
		xcrref[m]=1.0;
		xceref[m]=1.0;
	}
	/*
	   --------------------------------------------------------------------
	   c    reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02
	   --------------------------------------------------------------------
	 */
	if (((((162==12)&&(162==12))&&(162==12))&&(no_time_steps==100)))
	{
		( * cclass)='S';
		dtref=0.015;
		/*
		   --------------------------------------------------------------------
		   c    Reference values of RMS-norms of residual.
		   --------------------------------------------------------------------
		 */
		xcrref[0]=0.02747031545133948;
		xcrref[1]=0.010360746705285417;
		xcrref[2]=0.016235745065095532;
		xcrref[3]=0.015840557224455615;
		xcrref[4]=0.03484904060936246;
		/*
		   --------------------------------------------------------------------
		   c    Reference values of RMS-norms of solution error.
		   --------------------------------------------------------------------
		 */
		xceref[0]=2.7289258557377225E-5;
		xceref[1]=1.0364446640837285E-5;
		xceref[2]=1.615479828716647E-5;
		xceref[3]=1.57507049944801E-5;
		xceref[4]=3.417766618339053E-5;
		/*
		   --------------------------------------------------------------------
		   c    reference data for 36X36X36 grids after 400 time steps, with DT = 1.5d-03
		   --------------------------------------------------------------------
		 */
	}
	else
	{
		if (((((162==36)&&(162==36))&&(162==36))&&(no_time_steps==400)))
		{
			( * cclass)='W';
			dtref=0.0015;
			/*
			   --------------------------------------------------------------------
			   c    Reference values of RMS-norms of residual.
			   --------------------------------------------------------------------
			 */
			xcrref[0]=0.001893253733584;
			xcrref[1]=1.717075447775E-4;
			xcrref[2]=2.778153350936E-4;
			xcrref[3]=2.887475409984E-4;
			xcrref[4]=0.003143611161242;
			/*
			   --------------------------------------------------------------------
			   c    Reference values of RMS-norms of solution error.
			   --------------------------------------------------------------------
			 */
			xceref[0]=7.542088599534E-5;
			xceref[1]=6.512852253086E-6;
			xceref[2]=1.049092285688E-5;
			xceref[3]=1.128838671535E-5;
			xceref[4]=1.212845639773E-4;
			/*
			   --------------------------------------------------------------------
			   c    reference data for 64X64X64 grids after 400 time steps, with DT = 1.5d-03
			   --------------------------------------------------------------------
			 */
		}
		else
		{
			if (((((162==64)&&(162==64))&&(162==64))&&(no_time_steps==400)))
			{
				( * cclass)='A';
				dtref=0.0015;
				/*
				   --------------------------------------------------------------------
				   c    Reference values of RMS-norms of residual.
				   --------------------------------------------------------------------
				 */
				xcrref[0]=2.4799822399300195;
				xcrref[1]=1.1276337964368832;
				xcrref[2]=1.5028977888770492;
				xcrref[3]=1.421781621169518;
				xcrref[4]=2.129211303513828;
				/*
				   --------------------------------------------------------------------
				   c    Reference values of RMS-norms of solution error.
				   --------------------------------------------------------------------
				 */
				xceref[0]=1.090014029782055E-4;
				xceref[1]=3.734395176928209E-5;
				xceref[2]=5.009278540654163E-5;
				xceref[3]=4.767109393952825E-5;
				xceref[4]=1.3621613399213E-4;
				/*
				   --------------------------------------------------------------------
				   c    reference data for 102X102X102 grids after 400 time steps,
				   c    with DT = 1.0d-03
				   --------------------------------------------------------------------
				 */
			}
			else
			{
				if (((((162==102)&&(162==102))&&(162==102))&&(no_time_steps==400)))
				{
					( * cclass)='B';
					dtref=0.001;
					/*
					   --------------------------------------------------------------------
					   c    Reference values of RMS-norms of residual.
					   --------------------------------------------------------------------
					 */
					xcrref[0]=69.03293579998;
					xcrref[1]=30.95134488084;
					xcrref[2]=41.03336647017;
					xcrref[3]=38.64769009604;
					xcrref[4]=56.43482272596;
					/*
					   --------------------------------------------------------------------
					   c    Reference values of RMS-norms of solution error.
					   --------------------------------------------------------------------
					 */
					xceref[0]=0.009810006190188;
					xceref[1]=0.00102282790567;
					xceref[2]=0.001720597911692;
					xceref[3]=0.001694479428231;
					xceref[4]=0.01847456263981;
					/*
					   --------------------------------------------------------------------
					   c    reference data for 162X162X162 grids after 400 time steps,
					   c    with DT = 0.67d-03
					   --------------------------------------------------------------------
					 */
				}
				else
				{
					if (((((162==162)&&(162==162))&&(162==162))&&(no_time_steps==400)))
					{
						( * cclass)='C';
						dtref=6.7E-4;
						/*
						   --------------------------------------------------------------------
						   c    Reference values of RMS-norms of residual.
						   --------------------------------------------------------------------
						 */
						xcrref[0]=588.1691581829;
						xcrref[1]=245.4417603569;
						xcrref[2]=329.3829191851;
						xcrref[3]=308.1924971891;
						xcrref[4]=459.7223799176;
						/*
						   --------------------------------------------------------------------
						   c    Reference values of RMS-norms of solution error.
						   --------------------------------------------------------------------
						 */
						xceref[0]=0.2598120500183;
						xceref[1]=0.02590888922315;
						xceref[2]=0.0513288641632;
						xceref[3]=0.04806073419454;
						xceref[4]=0.5483377491301;
					}
					else
					{
						( * verified)=0;
					}
				}
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c    verification test for residuals if gridsize is either 12X12X12 or 
	   c    64X64X64 or 102X102X102 or 162X162X162
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c    Compute the difference of solution values and the known reference values.
	   --------------------------------------------------------------------
	 */
#pragma loop name verify#2 
	for (m=0; m<5; m ++ )
	{
		xcrdif[m]=fabs(((xcr[m]-xcrref[m])/xcrref[m]));
		xcedif[m]=fabs(((xce[m]-xceref[m])/xceref[m]));
	}
	/*
	   --------------------------------------------------------------------
	   c    Output the comparison of computed results to known cases.
	   --------------------------------------------------------------------
	 */
	if ((( * cclass)!='U'))
	{
		printf(" Verification being performed for cclass %1c\n", ( * cclass));
		printf(" accuracy setting for epsilon = %20.13e\n", epsilon);
		if ((fabs((dt-dtref))>epsilon))
		{
			( * verified)=0;
			( * cclass)='U';
			printf(" DT does not match the reference value of %15.8e\n", dtref);
		}
	}
	else
	{
		printf(" Unknown cclass\n");
	}
	if ((( * cclass)!='U'))
	{
		printf(" Comparison of RMS-norms of residual\n");
	}
	else
	{
		printf(" RMS-norms of residual\n");
	}
#pragma loop name verify#3 
	for (m=0; m<5; m ++ )
	{
		if ((( * cclass)=='U'))
		{
			printf("          %2d%20.13e\n", m, xcr[m]);
		}
		else
		{
			if ((xcrdif[m]>epsilon))
			{
				( * verified)=0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n", m, xcr[m], xcrref[m], xcrdif[m]);
			}
			else
			{
				printf("          %2d%20.13e%20.13e%20.13e\n", m, xcr[m], xcrref[m], xcrdif[m]);
			}
		}
	}
	if ((( * cclass)!='U'))
	{
		printf(" Comparison of RMS-norms of solution error\n");
	}
	else
	{
		printf(" RMS-norms of solution error\n");
	}
#pragma loop name verify#4 
	for (m=0; m<5; m ++ )
	{
		if ((( * cclass)=='U'))
		{
			printf("          %2d%20.13e\n", m, xce[m]);
		}
		else
		{
			if ((xcedif[m]>epsilon))
			{
				( * verified)=0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n", m, xce[m], xceref[m], xcedif[m]);
			}
			else
			{
				printf("          %2d%20.13e%20.13e%20.13e\n", m, xce[m], xceref[m], xcedif[m]);
			}
		}
	}
	if ((( * cclass)=='U'))
	{
		printf(" No reference values provided\n");
		printf(" No verification performed\n");
	}
	else
	{
		if (( * verified))
		{
			printf(" Verification Successful\n");
		}
		else
		{
			printf(" Verification failed\n");
		}
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void x_solve_kernel0(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int i2_0;
	int n_0;
	double fac1;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
	i2_0=( * i2);
	n_0=( * n);
	i1_0=( * i1);
	i_0=( * i);
	if (j<((1+162)-2))
	{
#pragma loop name x_solve#0#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
#pragma loop name x_solve#0#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(fac1*rhs[m][i_0][j][k]);
			}
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#0#0#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(rhs[m][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m][i_0][j][k]));
			}
			lhs[(n_0+1)][i2_0][j][k]=(lhs[(n_0+1)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+2)][i2_0][j][k]=(lhs[(n_0+2)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#0#0#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i2_0][j][k]=(rhs[m][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*rhs[m][i_0][j][k]));
			}
		}
	}
}

__global__ void x_solve_kernel1(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int n_0;
	double fac1;
	double fac2;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
	i_0=( * i);
	n_0=( * n);
	i1_0=( * i1);
	if (j<((1+162)-2))
	{
#pragma loop name x_solve#1#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
#pragma loop name x_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(fac1*rhs[m][i_0][j][k]);
			}
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(rhs[m][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m][i_0][j][k]));
			}
			/*
			   --------------------------------------------------------------------
			   c            scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i1_0][j][k]);
#pragma loop name x_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(fac2*rhs[m][i1_0][j][k]);
			}
		}
	}
}

__global__ void x_solve_kernel2(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int i2_0;
	int m_0;
	int n_0;
	double fac1;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i2_0=( * i2);
	i_0=( * i);
	n_0=( * n);
	m_0=( * m);
	i1_0=( * i1);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#2#0#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
			rhs[m_0][i_0][j][k]=(fac1*rhs[m_0][i_0][j][k]);
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i1_0][j][k]=(rhs[m_0][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m_0][i_0][j][k]));
			lhs[(n_0+1)][i2_0][j][k]=(lhs[(n_0+1)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+2)][i2_0][j][k]=(lhs[(n_0+2)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i2_0][j][k]=(rhs[m_0][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*rhs[m_0][i_0][j][k]));
		}
	}
}

__global__ void x_solve_kernel3(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int m_0;
	int n_0;
	double fac1;
	double fac2;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	m_0=( * m);
	i1_0=( * i1);
	i_0=( * i);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#2#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
			rhs[m_0][i_0][j][k]=(fac1*rhs[m_0][i_0][j][k]);
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i1_0][j][k]=(rhs[m_0][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m_0][i_0][j][k]));
			/*
			   --------------------------------------------------------------------
			   c               Scale the last row immediately
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i1_0][j][k]);
			rhs[m_0][i1_0][j][k]=(fac2*rhs[m_0][i1_0][j][k]);
		}
	}
}

__global__ void x_solve_kernel4(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(lhs[(( * n)+3)][i_0][j][k]*rhs[m][( * i1)][j][k]));
			}
		}
	}
}

__global__ void x_solve_kernel5(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#4#0#0 
			for (m=3; m<5; m ++ )
			{
				( * n)=(((m-3)+1)*5);
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(lhs[(( * n)+3)][i_0][j][k]*rhs[m][( * i1)][j][k]));
			}
		}
	}
}

__global__ void x_solve_kernel6(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int n_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#5#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#5#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=((rhs[m][i_0][j][k]-(lhs[(n_0+3)][i_0][j][k]*rhs[m][( * i1)][j][k]))-(lhs[(n_0+4)][i_0][j][k]*rhs[m][( * i2)][j][k]));
			}
		}
	}
}

__global__ void x_solve_kernel7(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int m_0;
	int n_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	n_0=( * n);
	m_0=( * m);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#6#0#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			rhs[m_0][i_0][j][k]=((rhs[m_0][i_0][j][k]-(lhs[(n_0+3)][i_0][j][k]*rhs[m_0][( * i1)][j][k]))-(lhs[(n_0+4)][i_0][j][k]*rhs[m_0][( * i2)][j][k]));
		}
	}
}

static void x_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the x-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the x-lines. Boundary conditions are non-periodic
	   --------------------------------------------------------------------
	 */
	int i;
	int n;
	int i1;
	int i2;
	int m;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   --------------------------------------------------------------------
	 */
	int * gpu__i;
	int * gpu__i1;
	int * gpu__i2;
	int * gpu__n;
	int * gpu__m;
	lhsx();
	/*
	   --------------------------------------------------------------------
	   c      perform the Thomas algorithm; first, FORWARD ELIMINATION     
	   --------------------------------------------------------------------
	 */
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i1)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i2)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
#pragma loop name x_solve#0 
	for (i=0; i<((1+162)-3); i ++ )
	{
		i1=(i+1);
		i2=(i+2);
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
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, n, rhs) private(fac1, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_solve) 
#pragma cuda gpurun registerRO(i, i1, i2, n) 
		x_solve_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 1); */
	}
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   --------------------------------------------------------------------
	 */
	i=(162-2);
	i1=(162-1);
	/* #pragma omp for */
	/* trace_start("x_solve", 2); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(fac1, fac2, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(x_solve) 
#pragma cuda gpurun registerRO(i, i1, n) 
	x_solve_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("x_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors                 
	   --------------------------------------------------------------------
	 */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__m)), gpuBytes));
#pragma loop name x_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name x_solve#2#0 
		for (i=0; i<((1+162)-3); i ++ )
		{
			i1=(i+1);
			i2=(i+2);
			dim3 dimBlock2(gpuNumThreads, 1, 1);
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
			dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, m, n, rhs) private(fac1, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(2) procname(x_solve) 
#pragma cuda gpurun registerRO(i, i1, i2, m, n) 
			x_solve_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("x_solve", 3); */
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   --------------------------------------------------------------------
		 */
		i=(162-2);
		i1=(162-1);
		/* trace_start("x_solve", 4); */
		dim3 dimBlock3(gpuNumThreads, 1, 1);
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
		dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, lhs, m, n, rhs) private(fac1, fac2, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, m, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(x_solve) 
#pragma cuda gpurun registerRO(i, i1, m, n) 
		x_solve_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 4); */
	}
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   --------------------------------------------------------------------
	 */
	i=(162-2);
	i1=(162-1);
	n=0;
	/*    trace_start("x_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(4) procname(x_solve) 
#pragma cuda gpurun registerRO(i) 
	x_solve_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/*  trace_stop("x_solve", 5); */
	/* trace_start("x_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(x_solve) 
#pragma cuda gpurun registerRO(i) 
	x_solve_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("x_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   --------------------------------------------------------------------
	 */
	n=0;
#pragma loop name x_solve#5 
	for (i=(162-3); i>( - 1); i -- )
	{
		i1=(i+1);
		i2=(i+2);
		/* #pragma omp for */
		/* trace_start("x_solve", 7); */
		dim3 dimBlock6(gpuNumThreads, 1, 1);
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
		dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(6) procname(x_solve) 
#pragma cuda gpurun registerRO(i, n) 
		x_solve_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 7); */
	}
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   --------------------------------------------------------------------
	 */
#pragma loop name x_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name x_solve#6#0 
		for (i=(162-3); i>(( - 1)+0); i -- )
		{
			i1=(i+1);
			i2=(i+2);
			dim3 dimBlock7(gpuNumThreads, 1, 1);
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
			dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, m, n, rhs) private(j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, m, rhs) 
#pragma cuda ainfo kernelid(7) procname(x_solve) 
#pragma cuda gpurun registerRO(i, m, n) 
#pragma cuda gpurun cudafree(i, i1, i2, m, n) 
			x_solve_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(( & n), gpu__n, gpuBytes, cudaMemcpyDeviceToHost));
			/* trace_stop("x_solve", 8); */
		}
	}
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__m));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/*
	   --------------------------------------------------------------------
	   c      Do the block-diagonal inversion          
	   --------------------------------------------------------------------
	 */
	ninvr();
	return ;
}

__global__ void x_solve_clnd1_cloned1_kernel0(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int i2_0;
	int n_0;
	double fac1;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
	i_0=( * i);
	n_0=( * n);
	i1_0=( * i1);
	i2_0=( * i2);
	if (j<((1+162)-2))
	{
#pragma loop name x_solve#0#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
#pragma loop name x_solve#0#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(fac1*rhs[m][i_0][j][k]);
			}
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#0#0#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(rhs[m][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m][i_0][j][k]));
			}
			lhs[(n_0+1)][i2_0][j][k]=(lhs[(n_0+1)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+2)][i2_0][j][k]=(lhs[(n_0+2)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#0#0#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i2_0][j][k]=(rhs[m][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*rhs[m][i_0][j][k]));
			}
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel1(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int n_0;
	double fac1;
	double fac2;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
	n_0=( * n);
	i_0=( * i);
	i1_0=( * i1);
	if (j<((1+162)-2))
	{
#pragma loop name x_solve#1#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
#pragma loop name x_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(fac1*rhs[m][i_0][j][k]);
			}
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
#pragma loop name x_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(rhs[m][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m][i_0][j][k]));
			}
			/*
			   --------------------------------------------------------------------
			   c            scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i1_0][j][k]);
#pragma loop name x_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1_0][j][k]=(fac2*rhs[m][i1_0][j][k]);
			}
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel2(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int i2_0;
	int m_0;
	int n_0;
	double fac1;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	n_0=( * n);
	m_0=( * m);
	i2_0=( * i2);
	i1_0=( * i1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#2#0#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
			rhs[m_0][i_0][j][k]=(fac1*rhs[m_0][i_0][j][k]);
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i1_0][j][k]=(rhs[m_0][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m_0][i_0][j][k]));
			lhs[(n_0+1)][i2_0][j][k]=(lhs[(n_0+1)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+2)][i2_0][j][k]=(lhs[(n_0+2)][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i2_0][j][k]=(rhs[m_0][i2_0][j][k]-(lhs[(n_0+0)][i2_0][j][k]*rhs[m_0][i_0][j][k]));
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel3(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int i1_0;
	int m_0;
	int n_0;
	double fac1;
	double fac2;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	m_0=( * m);
	i1_0=( * i1);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#2#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i_0][j][k]);
			lhs[(n_0+3)][i_0][j][k]=(fac1*lhs[(n_0+3)][i_0][j][k]);
			lhs[(n_0+4)][i_0][j][k]=(fac1*lhs[(n_0+4)][i_0][j][k]);
			rhs[m_0][i_0][j][k]=(fac1*rhs[m_0][i_0][j][k]);
			lhs[(n_0+2)][i1_0][j][k]=(lhs[(n_0+2)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+3)][i_0][j][k]));
			lhs[(n_0+3)][i1_0][j][k]=(lhs[(n_0+3)][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*lhs[(n_0+4)][i_0][j][k]));
			rhs[m_0][i1_0][j][k]=(rhs[m_0][i1_0][j][k]-(lhs[(n_0+1)][i1_0][j][k]*rhs[m_0][i_0][j][k]));
			/*
			   --------------------------------------------------------------------
			   c               Scale the last row immediately
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i1_0][j][k]);
			rhs[m_0][i1_0][j][k]=(fac2*rhs[m_0][i1_0][j][k]);
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel4(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(lhs[(( * n)+3)][i_0][j][k]*rhs[m][( * i1)][j][k]));
			}
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel5(int * i, int * i1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#4#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#4#0#0 
			for (m=3; m<5; m ++ )
			{
				( * n)=(((m-3)+1)*5);
				rhs[m][i_0][j][k]=(rhs[m][i_0][j][k]-(lhs[(( * n)+3)][i_0][j][k]*rhs[m][( * i1)][j][k]));
			}
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel6(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int n_0;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	i_0=( * i);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#5#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name x_solve#5#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i_0][j][k]=((rhs[m][i_0][j][k]-(lhs[(n_0+3)][i_0][j][k]*rhs[m][( * i1)][j][k]))-(lhs[(n_0+4)][i_0][j][k]*rhs[m][( * i2)][j][k]));
			}
		}
	}
}

__global__ void x_solve_clnd1_cloned1_kernel7(int * i, int * i1, int * i2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i_0;
	int m_0;
	int n_0;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	n_0=( * n);
	i_0=( * i);
	m_0=( * m);
	if (k<((1+162)-2))
	{
#pragma loop name x_solve#6#0#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			rhs[m_0][i_0][j][k]=((rhs[m_0][i_0][j][k]-(lhs[(n_0+3)][i_0][j][k]*rhs[m_0][( * i1)][j][k]))-(lhs[(n_0+4)][i_0][j][k]*rhs[m_0][( * i2)][j][k]));
		}
	}
}

static void x_solve_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the x-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the x-lines. Boundary conditions are non-periodic
	   --------------------------------------------------------------------
	 */
	int i;
	int n;
	int i1;
	int i2;
	int m;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   --------------------------------------------------------------------
	 */
	int * gpu__i;
	int * gpu__i1;
	int * gpu__i2;
	int * gpu__n;
	int * gpu__m;
	lhsx_clnd1_cloned1();
	/*
	   --------------------------------------------------------------------
	   c      perform the Thomas algorithm; first, FORWARD ELIMINATION     
	   --------------------------------------------------------------------
	 */
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i1)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__i2)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
#pragma loop name x_solve#0 
	for (i=0; i<((1+162)-3); i ++ )
	{
		i1=(i+1);
		i2=(i+2);
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
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, n, rhs) private(fac1, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, i1, i2, n) 
		x_solve_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 1); */
	}
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   --------------------------------------------------------------------
	 */
	i=(162-2);
	i1=(162-1);
	/* #pragma omp for */
	/* trace_start("x_solve", 2); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(fac1, fac2, j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, i1, n) 
	x_solve_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("x_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors                 
	   --------------------------------------------------------------------
	 */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__m)), gpuBytes));
#pragma loop name x_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name x_solve#2#0 
		for (i=0; i<((1+162)-3); i ++ )
		{
			i1=(i+1);
			i2=(i+2);
			dim3 dimBlock2(gpuNumThreads, 1, 1);
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
			dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, m, n, rhs) private(fac1, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(2) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, i1, i2, m, n) 
			x_solve_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("x_solve", 3); */
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   --------------------------------------------------------------------
		 */
		i=(162-2);
		i1=(162-1);
		/* trace_start("x_solve", 4); */
		dim3 dimBlock3(gpuNumThreads, 1, 1);
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
		dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, lhs, m, n, rhs) private(fac1, fac2, j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, m, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, i1, m, n) 
		x_solve_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 4); */
	}
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   --------------------------------------------------------------------
	 */
	i=(162-2);
	i1=(162-1);
	n=0;
	/*    trace_start("x_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(4) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i) 
	x_solve_clnd1_cloned1_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/*  trace_stop("x_solve", 5); */
	/* trace_start("x_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(i, i1, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i) 
	x_solve_clnd1_cloned1_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__i, gpu__i1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("x_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   --------------------------------------------------------------------
	 */
	n=0;
#pragma loop name x_solve#5 
	for (i=(162-3); i>( - 1); i -- )
	{
		i1=(i+1);
		i2=(i+2);
		/* #pragma omp for */
		/* trace_start("x_solve", 7); */
		dim3 dimBlock6(gpuNumThreads, 1, 1);
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
		dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, n, rhs) private(j, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(fac1, lhs, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(i, i1, i2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(6) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, n) 
		x_solve_clnd1_cloned1_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("x_solve", 7); */
	}
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   --------------------------------------------------------------------
	 */
#pragma loop name x_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name x_solve#6#0 
		for (i=(162-3); i>(( - 1)+0); i -- )
		{
			i1=(i+1);
			i2=(i+2);
			dim3 dimBlock7(gpuNumThreads, 1, 1);
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
			dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i, ( & i), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i1, ( & i1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__i2, ( & i2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(i, i1, i2, lhs, m, n, rhs) private(j, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) noconstant(i1, i2) 
#pragma cuda gpurun nocudamalloc(i, i1, i2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(i, i1, i2, lhs, m, rhs) 
#pragma cuda ainfo kernelid(7) procname(x_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(i, m, n) 
#pragma cuda gpurun cudafree(i, i1, i2, m, n) 
			x_solve_clnd1_cloned1_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__i, gpu__i1, gpu__i2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(( & n), gpu__n, gpuBytes, cudaMemcpyDeviceToHost));
			/* trace_stop("x_solve", 8); */
		}
	}
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__m));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__i));
	/*
	   --------------------------------------------------------------------
	   c      Do the block-diagonal inversion          
	   --------------------------------------------------------------------
	 */
	ninvr_clnd1_cloned1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void y_solve_kernel0(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int j2_0;
	int n_0;
	double fac1;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	n_0=( * n);
	j2_0=( * j2);
	j1_0=( * j1);
	j_0=( * j);
	if (i<((1+162)-2))
	{
#pragma loop name y_solve#0#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
#pragma loop name y_solve#0#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(fac1*rhs[m][i][j_0][k]);
			}
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#0#0#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(rhs[m][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m][i][j_0][k]));
			}
			lhs[(n_0+1)][i][j2_0][k]=(lhs[(n_0+1)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+2)][i][j2_0][k]=(lhs[(n_0+2)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#0#0#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j2_0][k]=(rhs[m][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*rhs[m][i][j_0][k]));
			}
		}
	}
}

__global__ void y_solve_kernel1(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	n_0=( * n);
	j_0=( * j);
	j1_0=( * j1);
	if (i<((1+162)-2))
	{
#pragma loop name y_solve#1#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
#pragma loop name y_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(fac1*rhs[m][i][j_0][k]);
			}
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(rhs[m][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m][i][j_0][k]));
			}
			/*
			   --------------------------------------------------------------------
			   c            scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j1_0][k]);
#pragma loop name y_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(fac2*rhs[m][i][j1_0][k]);
			}
		}
	}
}

__global__ void y_solve_kernel2(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int j2_0;
	int m_0;
	int n_0;
	double fac1;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	m_0=( * m);
	j2_0=( * j2);
	j_0=( * j);
	j1_0=( * j1);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#2#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
			rhs[m_0][i][j_0][k]=(fac1*rhs[m_0][i][j_0][k]);
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j1_0][k]=(rhs[m_0][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m_0][i][j_0][k]));
			lhs[(n_0+1)][i][j2_0][k]=(lhs[(n_0+1)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+2)][i][j2_0][k]=(lhs[(n_0+2)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j2_0][k]=(rhs[m_0][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*rhs[m_0][i][j_0][k]));
		}
	}
}

__global__ void y_solve_kernel3(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int m_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j1_0=( * j1);
	m_0=( * m);
	j_0=( * j);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#2#1#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
			rhs[m_0][i][j_0][k]=(fac1*rhs[m_0][i][j_0][k]);
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j1_0][k]=(rhs[m_0][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m_0][i][j_0][k]));
			/*
			   --------------------------------------------------------------------
			   c               Scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j1_0][k]);
			rhs[m_0][i][j1_0][k]=(fac2*rhs[m_0][i][j1_0][k]);
		}
	}
}

__global__ void y_solve_kernel4(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#3#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name y_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(lhs[(( * n)+3)][i][j_0][k]*rhs[m][i][( * j1)][k]));
			}
		}
	}
}

__global__ void y_solve_kernel5(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#4#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name y_solve#4#0#0 
			for (m=3; m<5; m ++ )
			{
				( * n)=(((m-3)+1)*5);
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(lhs[(( * n)+3)][i][j_0][k]*rhs[m][i][( * j1)][k]));
			}
		}
	}
}

__global__ void y_solve_kernel6(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int m_0;
	int n_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	n_0=( * n);
	m_0=( * m);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#5#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			rhs[m_0][i][j_0][k]=((rhs[m_0][i][j_0][k]-(lhs[(n_0+3)][i][j_0][k]*rhs[m_0][i][( * j1)][k]))-(lhs[(n_0+4)][i][j_0][k]*rhs[m_0][i][( * j2)][k]));
		}
	}
}

__global__ void y_solve_kernel7(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int m_0;
	int n_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	n_0=( * n);
	m_0=( * m);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#6#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			rhs[m_0][i][j_0][k]=((rhs[m_0][i][j_0][k]-(lhs[(n_0+3)][i][j_0][k]*rhs[m_0][i][( * j1)][k]))-(lhs[(n_0+4)][i][j_0][k]*rhs[m_0][i][( * j2)][k]));
		}
	}
}

static void y_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the y-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the y-lines. Boundary conditions are non-periodic
	   --------------------------------------------------------------------
	 */
	int j;
	int n;
	int j1;
	int j2;
	int m;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   --------------------------------------------------------------------
	 */
	int * gpu__j;
	int * gpu__j1;
	int * gpu__j2;
	int * gpu__n;
	int * gpu__m;
	lhsy();
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j1)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j2)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
#pragma loop name y_solve#0 
	for (j=0; j<((1+162)-3); j ++ )
	{
		j1=(j+1);
		j2=(j+2);
		/* trace_start("y_solve", 1); */
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
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, n, rhs) private(fac1, i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_solve) 
#pragma cuda gpurun registerRO(j, j1, j2, n) 
		y_solve_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("y_solve", 1); */
	}
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   --------------------------------------------------------------------
	 */
	j=(162-2);
	j1=(162-1);
	/* trace_start("y_solve", 2); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(fac1, fac2, i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(y_solve) 
#pragma cuda gpurun registerRO(j, j1, n) 
	y_solve_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("y_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors                 
	   --------------------------------------------------------------------
	 */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__m)), gpuBytes));
#pragma loop name y_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name y_solve#2#0 
		for (j=0; j<((1+162)-3); j ++ )
		{
			j1=(j+1);
			j2=(j+2);
			/* trace_start("y_solve", 3); */
			dim3 dimBlock2(gpuNumThreads, 1, 1);
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
			dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(fac1, i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(2) procname(y_solve) 
#pragma cuda gpurun registerRO(j, j1, j2, m, n) 
			y_solve_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("y_solve", 3); */
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   --------------------------------------------------------------------
		 */
		j=(162-2);
		j1=(162-1);
		/* #pragma omp for       */
		/* trace_start("y_solve", 4); */
		dim3 dimBlock3(gpuNumThreads, 1, 1);
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
		dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, lhs, m, n, rhs) private(fac1, fac2, i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, m, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(y_solve) 
#pragma cuda gpurun registerRO(j, j1, m, n) 
		y_solve_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("y_solve", 4); */
	}
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   --------------------------------------------------------------------
	 */
	j=(162-2);
	j1=(162-1);
	n=0;
	/*    trace_start("y_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(4) procname(y_solve) 
#pragma cuda gpurun registerRO(j) 
	y_solve_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/*  trace_stop("y_solve", 5); */
	/* trace_start("y_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(y_solve) 
#pragma cuda gpurun registerRO(j) 
	y_solve_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("y_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   --------------------------------------------------------------------
	 */
	n=0;
#pragma loop name y_solve#5 
	for (m=0; m<3; m ++ )
	{
#pragma loop name y_solve#5#0 
		for (j=(162-3); j>(( - 1)+0); j -- )
		{
			j1=(j+1);
			j2=(j+2);
			/* trace_start("y_solve", 7); */
			dim3 dimBlock6(gpuNumThreads, 1, 1);
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
			dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(6) procname(y_solve) 
#pragma cuda gpurun registerRO(j, m, n) 
			y_solve_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("y_solve", 7); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   --------------------------------------------------------------------
	 */
#pragma loop name y_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name y_solve#6#0 
		for (j=(162-3); j>(( - 1)+0); j -- )
		{
			j1=(j+1);
			j2=(j1+1);
			/* trace_start("y_solve", 8); */
			dim3 dimBlock7(gpuNumThreads, 1, 1);
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
			dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, rhs) 
#pragma cuda ainfo kernelid(7) procname(y_solve) 
#pragma cuda gpurun registerRO(j, m, n) 
#pragma cuda gpurun cudafree(j, j1, j2, m, n) 
			y_solve_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(( & n), gpu__n, gpuBytes, cudaMemcpyDeviceToHost));
			/* trace_stop("y_solve", 8); */
		}
	}
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__m));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	pinvr();
	return ;
}

__global__ void y_solve_clnd1_cloned1_kernel0(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int j2_0;
	int n_0;
	double fac1;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	j2_0=( * j2);
	j_0=( * j);
	n_0=( * n);
	j1_0=( * j1);
	if (i<((1+162)-2))
	{
#pragma loop name y_solve#0#0#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
#pragma loop name y_solve#0#0#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(fac1*rhs[m][i][j_0][k]);
			}
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#0#0#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(rhs[m][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m][i][j_0][k]));
			}
			lhs[(n_0+1)][i][j2_0][k]=(lhs[(n_0+1)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+2)][i][j2_0][k]=(lhs[(n_0+2)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#0#0#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j2_0][k]=(rhs[m][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*rhs[m][i][j_0][k]));
			}
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel1(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	j_0=( * j);
	n_0=( * n);
	j1_0=( * j1);
	if (i<((1+162)-2))
	{
#pragma loop name y_solve#1#0 
		for (k=1; k<((1+162)-2); k ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
#pragma loop name y_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(fac1*rhs[m][i][j_0][k]);
			}
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
#pragma loop name y_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(rhs[m][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m][i][j_0][k]));
			}
			/*
			   --------------------------------------------------------------------
			   c            scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j1_0][k]);
#pragma loop name y_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1_0][k]=(fac2*rhs[m][i][j1_0][k]);
			}
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel2(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int j2_0;
	int m_0;
	int n_0;
	double fac1;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j1_0=( * j1);
	j_0=( * j);
	n_0=( * n);
	m_0=( * m);
	j2_0=( * j2);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#2#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
			rhs[m_0][i][j_0][k]=(fac1*rhs[m_0][i][j_0][k]);
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j1_0][k]=(rhs[m_0][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m_0][i][j_0][k]));
			lhs[(n_0+1)][i][j2_0][k]=(lhs[(n_0+1)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+2)][i][j2_0][k]=(lhs[(n_0+2)][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j2_0][k]=(rhs[m_0][i][j2_0][k]-(lhs[(n_0+0)][i][j2_0][k]*rhs[m_0][i][j_0][k]));
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel3(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int j1_0;
	int m_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j1_0=( * j1);
	j_0=( * j);
	n_0=( * n);
	m_0=( * m);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#2#1#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j_0][k]);
			lhs[(n_0+3)][i][j_0][k]=(fac1*lhs[(n_0+3)][i][j_0][k]);
			lhs[(n_0+4)][i][j_0][k]=(fac1*lhs[(n_0+4)][i][j_0][k]);
			rhs[m_0][i][j_0][k]=(fac1*rhs[m_0][i][j_0][k]);
			lhs[(n_0+2)][i][j1_0][k]=(lhs[(n_0+2)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+3)][i][j_0][k]));
			lhs[(n_0+3)][i][j1_0][k]=(lhs[(n_0+3)][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*lhs[(n_0+4)][i][j_0][k]));
			rhs[m_0][i][j1_0][k]=(rhs[m_0][i][j1_0][k]-(lhs[(n_0+1)][i][j1_0][k]*rhs[m_0][i][j_0][k]));
			/*
			   --------------------------------------------------------------------
			   c               Scale the last row immediately 
			   --------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j1_0][k]);
			rhs[m_0][i][j1_0][k]=(fac2*rhs[m_0][i][j1_0][k]);
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel4(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#3#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name y_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(lhs[(( * n)+3)][i][j_0][k]*rhs[m][i][( * j1)][k]));
			}
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel5(int * j, int * j1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int i;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	j_0=( * j);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#4#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name y_solve#4#0#0 
			for (m=3; m<5; m ++ )
			{
				( * n)=(((m-3)+1)*5);
				rhs[m][i][j_0][k]=(rhs[m][i][j_0][k]-(lhs[(( * n)+3)][i][j_0][k]*rhs[m][i][( * j1)][k]));
			}
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel6(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int m_0;
	int n_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	n_0=( * n);
	j_0=( * j);
	m_0=( * m);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#5#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			rhs[m_0][i][j_0][k]=((rhs[m_0][i][j_0][k]-(lhs[(n_0+3)][i][j_0][k]*rhs[m_0][i][( * j1)][k]))-(lhs[(n_0+4)][i][j_0][k]*rhs[m_0][i][( * j2)][k]));
		}
	}
}

__global__ void y_solve_clnd1_cloned1_kernel7(int * j, int * j1, int * j2, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * m, int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int j_0;
	int m_0;
	int n_0;
	int i;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
	m_0=( * m);
	j_0=( * j);
	n_0=( * n);
	if (k<((1+162)-2))
	{
#pragma loop name y_solve#6#0#0#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
			rhs[m_0][i][j_0][k]=((rhs[m_0][i][j_0][k]-(lhs[(n_0+3)][i][j_0][k]*rhs[m_0][i][( * j1)][k]))-(lhs[(n_0+4)][i][j_0][k]*rhs[m_0][i][( * j2)][k]));
		}
	}
}

static void y_solve_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the y-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the y-lines. Boundary conditions are non-periodic
	   --------------------------------------------------------------------
	 */
	int j;
	int n;
	int j1;
	int j2;
	int m;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   --------------------------------------------------------------------
	 */
	int * gpu__j;
	int * gpu__j1;
	int * gpu__j2;
	int * gpu__n;
	int * gpu__m;
	lhsy_clnd1_cloned1();
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j1)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__j2)), gpuBytes));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
#pragma loop name y_solve#0 
	for (j=0; j<((1+162)-3); j ++ )
	{
		j1=(j+1);
		j2=(j+2);
		/* trace_start("y_solve", 1); */
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
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, n, rhs) private(fac1, i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, j1, j2, n) 
		y_solve_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("y_solve", 1); */
	}
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   --------------------------------------------------------------------
	 */
	j=(162-2);
	j1=(162-1);
	/* trace_start("y_solve", 2); */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(fac1, fac2, i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, j1, n) 
	y_solve_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("y_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors                 
	   --------------------------------------------------------------------
	 */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__m)), gpuBytes));
#pragma loop name y_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name y_solve#2#0 
		for (j=0; j<((1+162)-3); j ++ )
		{
			j1=(j+1);
			j2=(j+2);
			/* trace_start("y_solve", 3); */
			dim3 dimBlock2(gpuNumThreads, 1, 1);
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
			dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(fac1, i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(2) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, j1, j2, m, n) 
			y_solve_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("y_solve", 3); */
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   --------------------------------------------------------------------
		 */
		j=(162-2);
		j1=(162-1);
		/* #pragma omp for       */
		/* trace_start("y_solve", 4); */
		dim3 dimBlock3(gpuNumThreads, 1, 1);
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
		dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
		gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
		totalNumThreads=(gpuNumBlocks*gpuNumThreads);
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
		gpuBytes=sizeof (int);
		CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, lhs, m, n, rhs) private(fac1, fac2, i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, m, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, j1, m, n) 
		y_solve_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
		/* trace_stop("y_solve", 4); */
	}
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   --------------------------------------------------------------------
	 */
	j=(162-2);
	j1=(162-1);
	n=0;
	/*    trace_start("y_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(4) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j) 
	y_solve_clnd1_cloned1_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/*  trace_stop("y_solve", 5); */
	/* trace_start("y_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(j, j1, lhs, n, rhs) private(i, k, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j) 
	y_solve_clnd1_cloned1_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__j, gpu__j1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("y_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   --------------------------------------------------------------------
	 */
	n=0;
#pragma loop name y_solve#5 
	for (m=0; m<3; m ++ )
	{
#pragma loop name y_solve#5#0 
		for (j=(162-3); j>(( - 1)+0); j -- )
		{
			j1=(j+1);
			j2=(j+2);
			/* trace_start("y_solve", 7); */
			dim3 dimBlock6(gpuNumThreads, 1, 1);
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
			dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda ainfo kernelid(6) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, m, n) 
			y_solve_clnd1_cloned1_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			/* trace_stop("y_solve", 7); */
		}
	}
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   --------------------------------------------------------------------
	 */
#pragma loop name y_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
#pragma loop name y_solve#6#0 
		for (j=(162-3); j>(( - 1)+0); j -- )
		{
			j1=(j+1);
			j2=(j1+1);
			/* trace_start("y_solve", 8); */
			dim3 dimBlock7(gpuNumThreads, 1, 1);
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
			dim3 dimGrid7(gpuNumBlocks1, gpuNumBlocks2, 1);
			gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
			totalNumThreads=(gpuNumBlocks*gpuNumThreads);
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j, ( & j), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j1, ( & j1), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__j2, ( & j2), gpuBytes, cudaMemcpyHostToDevice));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(gpu__m, ( & m), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(j, j1, j2, lhs, m, n, rhs) private(i, k) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(j, j1, j2, lhs, m, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(j, j1, j2, lhs, m, rhs) 
#pragma cuda ainfo kernelid(7) procname(y_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(j, m, n) 
#pragma cuda gpurun cudafree(j, j1, j2, m, n) 
			y_solve_clnd1_cloned1_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__j, gpu__j1, gpu__j2, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__m, gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
			gpuBytes=sizeof (int);
			CUDA_SAFE_CALL(cudaMemcpy(( & n), gpu__n, gpuBytes, cudaMemcpyDeviceToHost));
			/* trace_stop("y_solve", 8); */
		}
	}
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__m));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j2));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__j));
	pinvr_clnd1_cloned1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void z_solve_kernel0(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double lhs_0;
	double lhs_1;
	int n_0;
	double fac1;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	n_0=( * n);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#0#0#0 
			for (k=0; k<((1+162)-3); k ++ )
			{
				lhs_1=lhs[(n_0+3)][i][j][k];
				lhs_0=lhs[(n_0+4)][i][j][k];
				k1=(k+1);
				k2=(k+2);
				fac1=(1.0/lhs[(n_0+2)][i][j][k]);
				lhs_1=(fac1*lhs_1);
				lhs_0=(fac1*lhs_0);
#pragma loop name z_solve#0#0#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				}
				lhs[(n_0+2)][i][j][k1]=(lhs[(n_0+2)][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*lhs_1));
				lhs[(n_0+3)][i][j][k1]=(lhs[(n_0+3)][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*lhs_0));
#pragma loop name z_solve#0#0#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*rhs[m][i][j][k]));
				}
				lhs[(n_0+1)][i][j][k2]=(lhs[(n_0+1)][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*lhs_1));
				lhs[(n_0+2)][i][j][k2]=(lhs[(n_0+2)][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*lhs_0));
#pragma loop name z_solve#0#0#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*rhs[m][i][j][k]));
				}
				lhs[(n_0+4)][i][j][k]=lhs_0;
				lhs[(n_0+3)][i][j][k]=lhs_1;
			}
		}
	}
}

__global__ void z_solve_kernel1(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int k1_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	n_0=( * n);
	k1_0=( * k1);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j][k_0]);
			lhs[(n_0+3)][i][j][k_0]=(fac1*lhs[(n_0+3)][i][j][k_0]);
			lhs[(n_0+4)][i][j][k_0]=(fac1*lhs[(n_0+4)][i][j][k_0]);
#pragma loop name z_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k_0]=(fac1*rhs[m][i][j][k_0]);
			}
			lhs[(n_0+2)][i][j][k1_0]=(lhs[(n_0+2)][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*lhs[(n_0+3)][i][j][k_0]));
			lhs[(n_0+3)][i][j][k1_0]=(lhs[(n_0+3)][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*lhs[(n_0+4)][i][j][k_0]));
#pragma loop name z_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1_0]=(rhs[m][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*rhs[m][i][j][k_0]));
			}
			/*
			   --------------------------------------------------------------------
			   c               scale the last row immediately
			   c-------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j][k1_0]);
#pragma loop name z_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1_0]=(fac2*rhs[m][i][j][k1_0]);
			}
		}
	}
}

__global__ void z_solve_kernel2(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double lhs_0;
	double lhs_1;
	double lhs_2;
	double rhs_0;
	double rhs_1;
	double fac1;
	double fac2;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=(_gtid+3);
	if (m<5)
	{
		n=(((m-3)+1)*5);
#pragma loop name z_solve#2#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#2#0#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
#pragma loop name z_solve#2#0#0#0 
				for (k=0; k<((1+162)-3); k ++ )
				{
					rhs_1=rhs[m][i][j][k];
					lhs_2=lhs[(n+3)][i][j][k];
					lhs_1=lhs[(n+4)][i][j][k];
					k1=(k+1);
					k2=(k+2);
					fac1=(1.0/lhs[(n+2)][i][j][k]);
					lhs_2=(fac1*lhs_2);
					lhs_1=(fac1*lhs_1);
					rhs_1=(fac1*rhs_1);
					lhs[(n+2)][i][j][k1]=(lhs[(n+2)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_2));
					lhs[(n+3)][i][j][k1]=(lhs[(n+3)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_1));
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[(n+1)][i][j][k1]*rhs_1));
					lhs[(n+1)][i][j][k2]=(lhs[(n+1)][i][j][k2]-(lhs[(n+0)][i][j][k2]*lhs_2));
					lhs[(n+2)][i][j][k2]=(lhs[(n+2)][i][j][k2]-(lhs[(n+0)][i][j][k2]*lhs_1));
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[(n+0)][i][j][k2]*rhs_1));
					lhs[(n+4)][i][j][k]=lhs_1;
					lhs[(n+3)][i][j][k]=lhs_2;
					rhs[m][i][j][k]=rhs_1;
				}
			}
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   c-------------------------------------------------------------------
		 */
		k=(162-2);
		k1=(162-1);
		/* #pragma omp parallel for private(i, j, fac1, fac2) schedule(static) */
#pragma loop name z_solve#2#1 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#2#1#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
				rhs_1=rhs[m][i][j][k];
				rhs_0=rhs[m][i][j][k1];
				lhs_2=lhs[(n+3)][i][j][k];
				lhs_1=lhs[(n+4)][i][j][k];
				lhs_0=lhs[(n+2)][i][j][k1];
				fac1=(1.0/lhs[(n+2)][i][j][k]);
				lhs_2=(fac1*lhs_2);
				lhs_1=(fac1*lhs_1);
				rhs_1=(fac1*rhs_1);
				lhs_0=(lhs_0-(lhs[(n+1)][i][j][k1]*lhs_2));
				lhs[(n+3)][i][j][k1]=(lhs[(n+3)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_1));
				rhs_0=(rhs_0-(lhs[(n+1)][i][j][k1]*rhs_1));
				/*
				   --------------------------------------------------------------------
				   c               Scale the last row immediately (some of this is overkill
				   c               if this is the last cell)
				   c-------------------------------------------------------------------
				 */
				fac2=(1.0/lhs_0);
				rhs_0=(fac2*rhs_0);
				lhs[(n+2)][i][j][k1]=lhs_0;
				lhs[(n+4)][i][j][k]=lhs_1;
				lhs[(n+3)][i][j][k]=lhs_2;
				rhs[m][i][j][k1]=rhs_0;
				rhs[m][i][j][k]=rhs_1;
			}
		}
	}
}

__global__ void z_solve_kernel3(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(lhs[(( * n)+3)][i][j][k_0]*rhs[m][i][j][( * k1)]));
			}
		}
	}
}

__global__ void z_solve_kernel4(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=(_gtid+3);
	k_0=( * k);
	if (m<5)
	{
		n=(((m-3)+1)*5);
#pragma loop name z_solve#4#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#4#0#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(lhs[(n+3)][i][j][k_0]*rhs[m][i][j][( * k1)]));
			}
		}
	}
}

__global__ void z_solve_kernel5(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int n_0;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+159);
	n_0=( * n);
	if (k>( - 1))
	{
#pragma loop name z_solve#5#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#5#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name z_solve#5#0#0#0 
				for (m=0; m<3; m ++ )
				{
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[(n_0+3)][i][j][k]*rhs[m][i][j][k1]))-(lhs[(n_0+4)][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
}

__global__ void z_solve_kernel6(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+159);
	if (k>( - 1))
	{
#pragma loop name z_solve#6#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#6#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name z_solve#6#0#0#0 
				for (m=3; m<5; m ++ )
				{
					n=(((m-3)+1)*5);
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[(n+3)][i][j][k]*rhs[m][i][j][k1]))-(lhs[(n+4)][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
}

static void z_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the z-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the z-lines. Boundary conditions are non-periodic
	   c-------------------------------------------------------------------
	 */
	int k;
	int n;
	int k1;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   c-------------------------------------------------------------------
	 */
	int * gpu__n;
	int * gpu__k;
	int * gpu__k1;
	lhsz();
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(lhs, n, rhs) private(fac1, i, j, k, k1, k2, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_solve) 
#pragma cuda gpurun registerRO(n) 
#pragma cuda gpurun registerRW(lhs[(n+3)][i][j][k], lhs[(n+4)][i][j][k]) 
	z_solve_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 1); */
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   c-------------------------------------------------------------------
	 */
	k=(162-2);
	k1=(162-1);
	/* #pragma omp for */
	/* trace_start("z_solve", 2); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k1, ( & k1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(k, k1, lhs, n, rhs) private(fac1, fac2, i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, n, rhs) 
#pragma cuda gpurun nocudafree(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(z_solve) 
#pragma cuda gpurun registerRO(k, k1, n) 
	z_solve_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors               
	   c-------------------------------------------------------------------
	 */
	/* trace_start("z_solve", 3); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(lhs, rhs) private(fac1, fac2, i, j, k, k1, k2, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(z_solve) 
#pragma cuda gpurun registerRW(lhs[(n+2)][i][j][k1], lhs[(n+3)][i][j][k], lhs[(n+4)][i][j][k], rhs[m][i][j][k1], rhs[m][i][j][k]) 
	z_solve_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 3); */
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   c-------------------------------------------------------------------
	 */
	k=(162-2);
	k1=(162-1);
	n=0;
	/* trace_start("z_solve", 4); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k1, ( & k1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(k, k1, lhs, n, rhs) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(z_solve) 
#pragma cuda gpurun registerRO(k) 
	z_solve_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 4); */
	/* trace_start("z_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(k, k1, lhs, rhs) private(i, j, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(k, k1, lhs, rhs) 
#pragma cuda gpurun nocudamalloc(k, k1, lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, rhs) 
#pragma cuda ainfo kernelid(4) procname(z_solve) 
#pragma cuda gpurun registerRO(k) 
#pragma cuda gpurun cudafree(k, k1) 
	z_solve_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("z_solve", 5); */
	/*
	   --------------------------------------------------------------------
	   c      Whether or not this is the last processor, we always have
	   c      to complete the back-substitution 
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   c-------------------------------------------------------------------
	 */
	n=0;
	/* trace_start("z_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
	gpuNumBlocks=0;
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=0;
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(lhs, n, rhs) private(i, j, k, k1, k2, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(z_solve) 
#pragma cuda gpurun registerRO(n) 
#pragma cuda gpurun cudafree(n) 
	z_solve_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	/* trace_stop("z_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   c-------------------------------------------------------------------
	 */
	/* trace_start("z_solve", 7); */
	dim3 dimBlock6(gpuNumThreads, 1, 1);
	gpuNumBlocks=0;
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=0;
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(lhs, rhs) private(i, j, k, k1, k2, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, rhs) 
#pragma cuda ainfo kernelid(6) procname(z_solve) 
	z_solve_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 7); */
	tzetar();
	return ;
}

__global__ void z_solve_clnd1_cloned1_kernel0(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double lhs_0;
	double lhs_1;
	int n_0;
	double fac1;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	n_0=( * n);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#0#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#0#0#0 
			for (k=0; k<((1+162)-3); k ++ )
			{
				lhs_1=lhs[(n_0+3)][i][j][k];
				lhs_0=lhs[(n_0+4)][i][j][k];
				k1=(k+1);
				k2=(k+2);
				fac1=(1.0/lhs[(n_0+2)][i][j][k]);
				lhs_1=(fac1*lhs_1);
				lhs_0=(fac1*lhs_0);
#pragma loop name z_solve#0#0#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				}
				lhs[(n_0+2)][i][j][k1]=(lhs[(n_0+2)][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*lhs_1));
				lhs[(n_0+3)][i][j][k1]=(lhs[(n_0+3)][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*lhs_0));
#pragma loop name z_solve#0#0#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[(n_0+1)][i][j][k1]*rhs[m][i][j][k]));
				}
				lhs[(n_0+1)][i][j][k2]=(lhs[(n_0+1)][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*lhs_1));
				lhs[(n_0+2)][i][j][k2]=(lhs[(n_0+2)][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*lhs_0));
#pragma loop name z_solve#0#0#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[(n_0+0)][i][j][k2]*rhs[m][i][j][k]));
				}
				lhs[(n_0+4)][i][j][k]=lhs_0;
				lhs[(n_0+3)][i][j][k]=lhs_1;
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel1(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int k1_0;
	int n_0;
	double fac1;
	double fac2;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k1_0=( * k1);
	k_0=( * k);
	n_0=( * n);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#1#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
			fac1=(1.0/lhs[(n_0+2)][i][j][k_0]);
			lhs[(n_0+3)][i][j][k_0]=(fac1*lhs[(n_0+3)][i][j][k_0]);
			lhs[(n_0+4)][i][j][k_0]=(fac1*lhs[(n_0+4)][i][j][k_0]);
#pragma loop name z_solve#1#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k_0]=(fac1*rhs[m][i][j][k_0]);
			}
			lhs[(n_0+2)][i][j][k1_0]=(lhs[(n_0+2)][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*lhs[(n_0+3)][i][j][k_0]));
			lhs[(n_0+3)][i][j][k1_0]=(lhs[(n_0+3)][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*lhs[(n_0+4)][i][j][k_0]));
#pragma loop name z_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1_0]=(rhs[m][i][j][k1_0]-(lhs[(n_0+1)][i][j][k1_0]*rhs[m][i][j][k_0]));
			}
			/*
			   --------------------------------------------------------------------
			   c               scale the last row immediately
			   c-------------------------------------------------------------------
			 */
			fac2=(1.0/lhs[(n_0+2)][i][j][k1_0]);
#pragma loop name z_solve#1#0#2 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1_0]=(fac2*rhs[m][i][j][k1_0]);
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel2(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	double lhs_0;
	double lhs_1;
	double lhs_2;
	double rhs_0;
	double rhs_1;
	double fac1;
	double fac2;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=(_gtid+3);
	if (m<5)
	{
		n=(((m-3)+1)*5);
#pragma loop name z_solve#2#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#2#0#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
#pragma loop name z_solve#2#0#0#0 
				for (k=0; k<((1+162)-3); k ++ )
				{
					rhs_1=rhs[m][i][j][k];
					lhs_2=lhs[(n+3)][i][j][k];
					lhs_1=lhs[(n+4)][i][j][k];
					k1=(k+1);
					k2=(k+2);
					fac1=(1.0/lhs[(n+2)][i][j][k]);
					lhs_2=(fac1*lhs_2);
					lhs_1=(fac1*lhs_1);
					rhs_1=(fac1*rhs_1);
					lhs[(n+2)][i][j][k1]=(lhs[(n+2)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_2));
					lhs[(n+3)][i][j][k1]=(lhs[(n+3)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_1));
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[(n+1)][i][j][k1]*rhs_1));
					lhs[(n+1)][i][j][k2]=(lhs[(n+1)][i][j][k2]-(lhs[(n+0)][i][j][k2]*lhs_2));
					lhs[(n+2)][i][j][k2]=(lhs[(n+2)][i][j][k2]-(lhs[(n+0)][i][j][k2]*lhs_1));
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[(n+0)][i][j][k2]*rhs_1));
					lhs[(n+4)][i][j][k]=lhs_1;
					lhs[(n+3)][i][j][k]=lhs_2;
					rhs[m][i][j][k]=rhs_1;
				}
			}
		}
		/*
		   --------------------------------------------------------------------
		   c         And again the last two rows separately
		   c-------------------------------------------------------------------
		 */
		k=(162-2);
		k1=(162-1);
		/* #pragma omp parallel for private(i, j, fac1, fac2) schedule(static) */
#pragma loop name z_solve#2#1 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#2#1#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
				rhs_1=rhs[m][i][j][k];
				rhs_0=rhs[m][i][j][k1];
				lhs_2=lhs[(n+3)][i][j][k];
				lhs_1=lhs[(n+4)][i][j][k];
				lhs_0=lhs[(n+2)][i][j][k1];
				fac1=(1.0/lhs[(n+2)][i][j][k]);
				lhs_2=(fac1*lhs_2);
				lhs_1=(fac1*lhs_1);
				rhs_1=(fac1*rhs_1);
				lhs_0=(lhs_0-(lhs[(n+1)][i][j][k1]*lhs_2));
				lhs[(n+3)][i][j][k1]=(lhs[(n+3)][i][j][k1]-(lhs[(n+1)][i][j][k1]*lhs_1));
				rhs_0=(rhs_0-(lhs[(n+1)][i][j][k1]*rhs_1));
				/*
				   --------------------------------------------------------------------
				   c               Scale the last row immediately (some of this is overkill
				   c               if this is the last cell)
				   c-------------------------------------------------------------------
				 */
				fac2=(1.0/lhs_0);
				rhs_0=(fac2*rhs_0);
				lhs[(n+2)][i][j][k1]=lhs_0;
				lhs[(n+4)][i][j][k]=lhs_1;
				lhs[(n+3)][i][j][k]=lhs_2;
				rhs[m][i][j][k1]=rhs_0;
				rhs[m][i][j][k]=rhs_1;
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel3(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
	k_0=( * k);
	if (i<((1+162)-2))
	{
#pragma loop name z_solve#3#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#3#0#0 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(lhs[(( * n)+3)][i][j][k_0]*rhs[m][i][j][( * k1)]));
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel4(int * k, int * k1, double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int k_0;
	int i;
	int j;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=(_gtid+3);
	k_0=( * k);
	if (m<5)
	{
		n=(((m-3)+1)*5);
#pragma loop name z_solve#4#0 
		for (i=1; i<((1+162)-2); i ++ )
		{
#pragma loop name z_solve#4#0#0 
			for (j=1; j<((1+162)-2); j ++ )
			{
				rhs[m][i][j][k_0]=(rhs[m][i][j][k_0]-(lhs[(n+3)][i][j][k_0]*rhs[m][i][j][( * k1)]));
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel5(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], int * n, double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int n_0;
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+159);
	n_0=( * n);
	if (k>( - 1))
	{
#pragma loop name z_solve#5#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#5#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name z_solve#5#0#0#0 
				for (m=0; m<3; m ++ )
				{
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[(n_0+3)][i][j][k]*rhs[m][i][j][k1]))-(lhs[(n_0+4)][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
}

__global__ void z_solve_clnd1_cloned1_kernel6(double lhs[15][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[5][(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	int i;
	int j;
	int k;
	int k1;
	int k2;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+159);
	if (k>( - 1))
	{
#pragma loop name z_solve#6#0 
		for (j=1; j<((1+162)-2); j ++ )
		{
#pragma loop name z_solve#6#0#0 
			for (i=1; i<((1+162)-2); i ++ )
			{
#pragma loop name z_solve#6#0#0#0 
				for (m=3; m<5; m ++ )
				{
					n=(((m-3)+1)*5);
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[(n+3)][i][j][k]*rhs[m][i][j][k1]))-(lhs[(n+4)][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
}

static void z_solve_clnd1_cloned1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c this function performs the solution of the approximate factorization
	   c step in the z-direction for all five matrix components
	   c simultaneously. The Thomas algorithm is employed to solve the
	   c systems for the z-lines. Boundary conditions are non-periodic
	   c-------------------------------------------------------------------
	 */
	int k;
	int n;
	int k1;
	/*
	   --------------------------------------------------------------------
	   c                          FORWARD ELIMINATION  
	   c-------------------------------------------------------------------
	 */
	int * gpu__n;
	int * gpu__k;
	int * gpu__k1;
	lhsz_clnd1_cloned1();
	n=0;
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__n)), gpuBytes));
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
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(lhs, n, rhs) private(fac1, i, j, k, k1, k2, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, n, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(n) 
#pragma cuda gpurun registerRW(lhs[(n+3)][i][j][k], lhs[(n+4)][i][j][k]) 
	z_solve_clnd1_cloned1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 1); */
	/*
	   --------------------------------------------------------------------
	   c      The last two rows in this grid block are a bit different, 
	   c      since they do not have two more rows available for the
	   c      elimination of off-diagonal entries
	   c-------------------------------------------------------------------
	 */
	k=(162-2);
	k1=(162-1);
	/* #pragma omp for */
	/* trace_start("z_solve", 2); */
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k)), gpuBytes));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__k1)), gpuBytes));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k1, ( & k1), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(k, k1, lhs, n, rhs) private(fac1, fac2, i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, n, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, n, rhs) 
#pragma cuda gpurun nocudafree(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(1) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(k, k1, n) 
	z_solve_clnd1_cloned1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 2); */
	/*
	   --------------------------------------------------------------------
	   c      do the u+c and the u-c factors               
	   c-------------------------------------------------------------------
	 */
	/* trace_start("z_solve", 3); */
	dim3 dimBlock2(gpuNumThreads, 1, 1);
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(lhs, rhs) private(fac1, fac2, i, j, k, k1, k2, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRW(lhs[(n+2)][i][j][k1], lhs[(n+3)][i][j][k], lhs[(n+4)][i][j][k], rhs[m][i][j][k1], rhs[m][i][j][k]) 
	z_solve_clnd1_cloned1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 3); */
	/*
	   --------------------------------------------------------------------
	   c                         BACKSUBSTITUTION 
	   c-------------------------------------------------------------------
	 */
	k=(162-2);
	k1=(162-1);
	n=0;
	/* trace_start("z_solve", 4); */
	dim3 dimBlock3(gpuNumThreads, 1, 1);
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k, ( & k), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__k1, ( & k1), gpuBytes, cudaMemcpyHostToDevice));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(k, k1, lhs, n, rhs) private(i, j, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nocudafree(k, k1, lhs, n, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, n, rhs) 
#pragma cuda ainfo kernelid(3) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(k) 
	z_solve_clnd1_cloned1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 4); */
	/* trace_start("z_solve", 5); */
	dim3 dimBlock4(gpuNumThreads, 1, 1);
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(k, k1, lhs, rhs) private(i, j, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(k, k1, lhs, rhs) 
#pragma cuda gpurun nocudamalloc(k, k1, lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(k, k1, lhs, rhs) 
#pragma cuda ainfo kernelid(4) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(k) 
#pragma cuda gpurun cudafree(k, k1) 
	z_solve_clnd1_cloned1_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__k, gpu__k1, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k1));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__k));
	/* trace_stop("z_solve", 5); */
	/*
	   --------------------------------------------------------------------
	   c      Whether or not this is the last processor, we always have
	   c      to complete the back-substitution 
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c      The first three factors
	   c-------------------------------------------------------------------
	 */
	n=0;
	/* trace_start("z_solve", 6); */
	dim3 dimBlock5(gpuNumThreads, 1, 1);
	gpuNumBlocks=0;
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=0;
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaMemcpy(gpu__n, ( & n), gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel for shared(lhs, n, rhs) private(i, j, k, k1, k2, m) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, n, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, n, rhs) 
#pragma cuda ainfo kernelid(5) procname(z_solve_clnd1_cloned1) 
#pragma cuda gpurun registerRO(n) 
#pragma cuda gpurun cudafree(n) 
	z_solve_clnd1_cloned1_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), gpu__n, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	gpuBytes=sizeof (int);
	CUDA_SAFE_CALL(cudaFree(gpu__n));
	/* trace_stop("z_solve", 6); */
	/*
	   --------------------------------------------------------------------
	   c      And the remaining two
	   c-------------------------------------------------------------------
	 */
	/* trace_start("z_solve", 7); */
	dim3 dimBlock6(gpuNumThreads, 1, 1);
	gpuNumBlocks=0;
	if ((gpuNumBlocks>MAX_GDIMENSION))
	{
		gpuNumBlocks2=0;
		gpuNumBlocks1=MAX_NDIMENSION;
	}
	else
	{
		gpuNumBlocks2=1;
		gpuNumBlocks1=gpuNumBlocks;
	}
	dim3 dimGrid6(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel for shared(lhs, rhs) private(i, j, k, k1, k2, m, n) schedule(static)
#pragma cuda gpurun noc2gmemtr(lhs, rhs) 
#pragma cuda gpurun nocudamalloc(lhs, rhs) 
#pragma cuda gpurun nocudafree(lhs, rhs) 
#pragma cuda gpurun nog2cmemtr(lhs, rhs) 
#pragma cuda ainfo kernelid(6) procname(z_solve_clnd1_cloned1) 
	z_solve_clnd1_cloned1_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rhs));
	/* trace_stop("z_solve", 7); */
	tzetar_clnd1_cloned1();
	return ;
}

