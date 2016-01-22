/*
   --------------------------------------------------------------------

   NAS Parallel Benchmarks 2.3 OpenMP C versions - BT

   This benchmark is an OpenMP C version of the NPB BT code.

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

Authors: R. Van der Wijngaart
T. Harris
M. Yarrow

OpenMP C version: S. Satoh

--------------------------------------------------------------------
 */
#include <sys/time.h>
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

/**********************************************************/
/* Maximum width of linear memory bound to texture memory */
/**********************************************************/
/* width in bytes */
#define LMAX_WIDTH    134217728
/* width in words */
#define LMAX_WWIDTH  33554432
/**********************************/
/* Maximum memory pitch (in bytes)*/
/**********************************/
#define MAX_PITCH   262144
/****************************************/
/* Maximum allowed GPU global memory    */
/* (should be less than actual size ) */
/****************************************/
#define MAX_GMSIZE  6000000000
/****************************************/
/* Maximum allowed GPU shared memory    */
/****************************************/
#define MAX_SMSIZE  16384
/********************************************/
/* Maximum size of each dimension of a grid */
/********************************************/
#define MAX_GDIMENSION  65535
#define MAX_NDIMENSION  10000

#define BLOCK_SIZE  1024


static int gpuNumThreads = BLOCK_SIZE;
static int gpuNumBlocks;
static int gpuNumBlocks1;
static int gpuNumBlocks2;
static int totalNumThreads;
unsigned int gpuGmemSize = 0;
unsigned int gpuSmemSize = 0;
static unsigned long long gpuBytes = 0;

#endif 
/* End of __O2G_HEADER__ */



__constant__ double const__c1;
__constant__ double const__c2;
__constant__ double const__con43;
__constant__ double const__dx1tx1;
__constant__ double const__dx2tx1;
__constant__ double const__dx3tx1;
__constant__ double const__dx4tx1;
__constant__ double const__dx5tx1;
__constant__ double const__tx2;
__constant__ double const__xxcon2;
__constant__ double const__xxcon3;
__constant__ double const__xxcon4;
__constant__ double const__xxcon5;
__constant__ double const__dssp;
__constant__ double const__dy1ty1;
__constant__ double const__dy2ty1;
__constant__ double const__dy3ty1;
__constant__ double const__dy4ty1;
__constant__ double const__dy5ty1;
__constant__ double const__ty2;
__constant__ double const__yycon2;
__constant__ double const__yycon3;
__constant__ double const__yycon4;
__constant__ double const__yycon5;
__constant__ double const__dz1tz1;
__constant__ double const__dz2tz1;
__constant__ double const__dz3tz1;
__constant__ double const__dz4tz1;
__constant__ double const__dz5tz1;
__constant__ double const__tz2;
__constant__ double const__zzcon2;
__constant__ double const__zzcon3;
__constant__ double const__zzcon4;
__constant__ double const__zzcon5;
__constant__ double const__dt;
__constant__ double const__dnxm1;
__constant__ double const__dnym1;
__constant__ double const__dnzm1;
__constant__ double const__xxcon1;
__constant__ double const__yycon1;
__constant__ double const__zzcon1;
__constant__ double const__c1345;
__constant__ double const__c3c4;
__constant__ double const__dx1;
__constant__ double const__dx2;
__constant__ double const__dx3;
__constant__ double const__dx4;
__constant__ double const__dx5;
__constant__ double const__tx1;
__constant__ double const__dy1;
__constant__ double const__dy2;
__constant__ double const__dy3;
__constant__ double const__dy4;
__constant__ double const__dy5;
__constant__ double const__ty1;
__constant__ double const__c3;
__constant__ double const__c4;
__constant__ double const__dz1;
__constant__ double const__dz2;
__constant__ double const__dz3;
__constant__ double const__dz4;
__constant__ double const__dz5;
__constant__ double const__tz1;
int * gpu__grid_points;
double * gpu__rhs;
double * gpu__u;
double * gpu__forcing;
double * gpu__qs;
double * gpu__rho_i;
double * gpu__square;
double * gpu__us;
double * gpu__vs;
double * gpu__ws;
double * gpu__ce;
size_t pitch__ce;
double * gpu__lhs;
double * gpu__fjac;
double * gpu__njac;
/* function declarations */
static void add(void );
static void add_clnd1(void );
static void adi(void );
static void adi_clnd1(void );
static void error_norm(double rms[5]);
static void rhs_norm(double rms[5]);
static void exact_rhs(void );
static void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
__device__ static void dev_exact_solution(double xi, double eta, double zeta, double dtemp[5], double * ce, size_t pitch__ce);
static void initialize(void );
static void initialize_clnd1(void );
static void lhsinit(void );
static void lhsx(void );
static void lhsx_clnd1(void );
static void lhsy(void );
static void lhsy_clnd1(void );
static void lhsz(void );
static void lhsz_clnd1(void );
static void compute_rhs(void );
static void compute_rhs_clnd2(void );
static void compute_rhs_clnd1(void );
static void set_constants(void );
static void verify(int no_time_steps, char * cclass, int * verified);
static void x_solve(void );
static void x_solve_clnd1(void );
static void x_backsubstitute(void );
static void x_backsubstitute_clnd1(void );
static void x_solve_cell(void );
static void x_solve_cell_clnd1(void );
__device__ static void dev_matvec_sub(double ablock[5][5], double avec[5], double bvec[5]);
__device__ static void dev_matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5]);
__device__ static void dev_binvcrhs(double lhs[5][5], double c[5][5], double r[5]);
__device__ static void dev_binvrhs(double lhs[5][5], double r[5]);
static void y_solve(void );
static void y_solve_clnd1(void );
static void y_backsubstitute(void );
static void y_backsubstitute_clnd1(void );
static void y_solve_cell(void );
static void y_solve_cell_clnd1(void );
static void z_solve(void );
static void z_solve_clnd1(void );
static void z_backsubstitute(void );
static void z_backsubstitute_clnd1(void );
static void z_solve_cell(void );
static void z_solve_cell_clnd1(void );
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

static void c_print_results(char * name, char ccclass, int n1, int n2, int n3, int niter, int nthreads, double t, double mops, char * optype, int passed_verification, char * npbversion, char * compiletime, char * cc, char * clink, char * c_lib, char * c_inc, char * cflags, char * clinkflags, char * rand)
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
   --------------------------------------------------------------------
   program BT
   c-------------------------------------------------------------------
 */
int main(int argc, char *  * argv)
{
	int niter;
	int step;
	int n3;
	int nthreads = 1;
	double navg;
	double mflops;
	double tmax;
	int verified;
	char cclass;
	FILE * fp;
	/*
	   --------------------------------------------------------------------
	   c      Root node reads input file (if it exists) else takes
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


	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__grid_points)), gpuBytes));
	gpuBytes=(((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__rhs)), gpuBytes));
	gpuBytes=((((((((162+1)/2)*2)+1)*((((162+1)/2)*2)+1))*((((162+1)/2)*2)+1))*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__u)), gpuBytes));
	gpuBytes=(((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*(5+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__forcing)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__qs)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__rho_i)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__square)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__us)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__vs)), gpuBytes));
	gpuBytes=((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ws)), gpuBytes));
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__ce)), ( & pitch__ce), (13*sizeof (double)), 5));
	gpuBytes=(((((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*3)*5)*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__lhs)), gpuBytes));
	gpuBytes=((((((((162/2)*2)+1)*(((162/2)*2)+1))*((162-1)+1))*5)*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__fjac)), gpuBytes));
	gpuBytes=((((((((162/2)*2)+1)*(((162/2)*2)+1))*((162-1)+1))*5)*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__njac)), gpuBytes));
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - BT Benchmark\n\n");
	fp=fopen("inputbt.data", "r");
	if ((fp!=((void * )0)))
	{
		printf(" Reading from input file inputbt.data");
		fscanf(fp, "%d", ( & niter));
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%lg", ( & dt));
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%d%d%d", ( & grid_points[0]), ( & grid_points[1]), ( & grid_points[2]));
		fclose(fp);
	}
	else
	{
		printf(" No input file inputbt.data. Using compiled defaults\n");
		niter=200;
		dt=1.0E-4;
		grid_points[0]=162;
		grid_points[1]=162;
		grid_points[2]=162;
	}
	printf(" Size: %3dx%3dx%3d\n", grid_points[0], grid_points[1], grid_points[2]);
	printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);
	if ((((grid_points[0]>162)||(grid_points[1]>162))||(grid_points[2]>162)))
	{
		printf(" %dx%dx%d\n", grid_points[0], grid_points[1], grid_points[2]);
		printf(" Problem size too big for compiled array sizes\n");
		exit(1);
	}
	set_constants();
	/*
#pragma omp parallel threadprivate(buf, cuf, q, ue) shared(c1, c1345, c2, c3, c3c4, c4, ce, con43, dnxm1, dnym1, dnzm1, dssp, dt, dx1, dx1tx1, dx2, dx2tx1, dx3, dx3tx1, dx4, dx4tx1, dx5, dx5tx1, dy1, dy1ty1, dy2, dy2ty1, dy3, dy3ty1, dy4, dy4ty1, dy5, dy5ty1, dz1, dz1tz1, dz2, dz2tz1, dz3, dz3tz1, dz4, dz4tz1, dz5, dz5tz1, fjac, forcing, grid_points, lhs, njac, qs, rho_i, rhs, square, tmp1, tmp2, tmp3, tx1, tx2, ty1, ty2, tz1, tz2, u, us, vs, ws, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, yycon1, yycon2, yycon3, yycon4, yycon5, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) private() firstprivate()
	 */
	{
		initialize();
		lhsinit();
		exact_rhs();
		/*
		   --------------------------------------------------------------------
		   c      do one time step to touch all code, and reinitialize
		   c-------------------------------------------------------------------
		 */
		adi();
		initialize_clnd1();
	}
	/* end parallel */
	timer_clear(1);
	timer_start(1);
	/*
#pragma omp parallel threadprivate() shared(c1, c1345, c2, c3, c3c4, c4, con43, dssp, dt, dx1, dx1tx1, dx2, dx2tx1, dx3, dx3tx1, dx4, dx4tx1, dx5, dx5tx1, dy1, dy1ty1, dy2, dy2ty1, dy3, dy3ty1, dy4, dy4ty1, dy5, dy5ty1, dz1, dz1tz1, dz2, dz2tz1, dz3, dz3tz1, dz4, dz4tz1, dz5, dz5tz1, fjac, forcing, grid_points, lhs, njac, qs, rho_i, rhs, square, tmp1, tmp2, tmp3, tx1, tx2, ty1, ty2, tz1, tz2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5, yycon2, yycon3, yycon4, yycon5, zzcon2, zzcon3, zzcon4, zzcon5) private(step) firstprivate(niter)
	 */
	{
#pragma loop name main#0 
		for (step=1; step<=niter; step ++ )
		{
			if ((((step%20)==0)||(step==1)))
			{
#pragma omp parallel private(step)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, q, ue) nocudafree(buf, cuf, q, u, ue) nog2cmemtr(buf, cuf, q, ue) 
				{
#pragma omp master
					printf(" Time step %4d\n", step);
				}
			}
			adi_clnd1();
		}
	}
	/* end parallel */
	timer_stop(1);
	tmax=timer_read(1);
	verify(niter, ( & cclass), ( & verified));
	n3=((grid_points[0]*grid_points[1])*grid_points[2]);
	navg=(((grid_points[0]+grid_points[1])+grid_points[2])/3.0);
	if ((tmax!=0.0))
	{
		mflops=(((1.0E-6*((double)niter))*(((3478.8*((double)n3))-(17655.7*(navg*navg)))+(28023.7*navg)))/tmax);
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("BT", cclass, grid_points[0], grid_points[1], grid_points[2], niter, nthreads, tmax, mflops, "          floating point", verified, "2.3", "21 Feb 2012", "gcc", "gcc", "-lm", "-I../common", "-O3 ", "(none)", "(none)");
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
	printf("====> Cache R/O shared scalar variables onto GPU constant memory.\n");
	printf("====> local array reduction variable configuration = 1\n");
	CUDA_SAFE_CALL(cudaFree(gpu__grid_points));
	CUDA_SAFE_CALL(cudaFree(gpu__rhs));
	CUDA_SAFE_CALL(cudaFree(gpu__u));
	CUDA_SAFE_CALL(cudaFree(gpu__forcing));
	CUDA_SAFE_CALL(cudaFree(gpu__qs));
	CUDA_SAFE_CALL(cudaFree(gpu__rho_i));
	CUDA_SAFE_CALL(cudaFree(gpu__square));
	CUDA_SAFE_CALL(cudaFree(gpu__us));
	CUDA_SAFE_CALL(cudaFree(gpu__vs));
	CUDA_SAFE_CALL(cudaFree(gpu__ws));
	CUDA_SAFE_CALL(cudaFree(gpu__ce));
	CUDA_SAFE_CALL(cudaFree(gpu__lhs));
	CUDA_SAFE_CALL(cudaFree(gpu__fjac));
	CUDA_SAFE_CALL(cudaFree(gpu__njac));
	fflush(stdout);
	fflush(stderr);
	return _ret_val_0;
}

/*
   --------------------------------------------------------------------
   c-------------------------------------------------------------------
 */
__global__ void add_kernel0(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name add#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name add#0#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name add#0#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					u[i][j][k][m]=(u[i][j][k][m]+rhs[i][j][k][m]);
				}
			}
		}
	}
	return ;
}

static void add(void )
{
	/*
	   --------------------------------------------------------------------
	   c     addition of update to the vector u
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
#pragma omp parallel shared(grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, grid_points, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, rhs, u) 
#pragma cuda ainfo kernelid(0) procname(add) 
	add_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	return ;
}

__global__ void add_clnd1_kernel0(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name add#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name add#0#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name add#0#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					u[i][j][k][m]=(u[i][j][k][m]+rhs[i][j][k][m]);
				}
			}
		}
	}
	return ;
}

static void add_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   c     addition of update to the vector u
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
#pragma omp parallel shared(grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, q, rhs, ue) nocudafree(buf, cuf, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, rhs, u) 
#pragma cuda ainfo kernelid(0) procname(add_clnd1) 
	add_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	gpuBytes=((((((((162+1)/2)*2)+1)*((((162+1)/2)*2)+1))*((((162+1)/2)*2)+1))*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(u, gpu__u, gpuBytes, cudaMemcpyDeviceToHost));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void adi(void )
{
	compute_rhs();
	x_solve();
	y_solve();
	z_solve();
	add();
	return ;
}

static void adi_clnd1(void )
{
	compute_rhs_clnd1();
	x_solve_clnd1();
	y_solve_clnd1();
	z_solve_clnd1();
	add_clnd1();
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
	   c     this function computes the norm of the difference between the
	   c     computed solution and the exact solution
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
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
#pragma loop name error_norm#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			eta=(((double)j)*dnym1);
#pragma loop name error_norm#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, u_exact);
#pragma loop name error_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=(u[i][j][k][m]-u_exact[m]);
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
#pragma loop name error_norm#2 
	for (m=0; m<5; m ++ )
	{
#pragma loop name error_norm#2#0 
		for (d=0; d<=2; d ++ )
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
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
#pragma loop name rhs_norm#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name rhs_norm#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
#pragma loop name rhs_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=rhs[i][j][k][m];
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
#pragma loop name rhs_norm#2 
	for (m=0; m<5; m ++ )
	{
#pragma loop name rhs_norm#2#0 
		for (d=0; d<=2; d ++ )
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
__global__ void exact_rhs_kernel0(double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points)
{
	/*
	   --------------------------------------------------------------------
	   c     initialize                                  
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(forcing, grid_points) private(i)
	if (m<5)
	{
#pragma loop name exact_rhs#0#0 
		for (k=0; k<grid_points[2]; k ++ )
		{
#pragma loop name exact_rhs#0#0#0 
			for (j=0; j<grid_points[1]; j ++ )
			{
#pragma loop name exact_rhs#0#0#0#0 
				for (i=0; i<grid_points[0]; i ++ )
				{
					forcing[i][j][k][m]=0.0;
				}
			}
		}
	}
}

__global__ void exact_rhs_kernel1(double * ce, size_t pitch__ce, double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double buf[][162][5], double * cuf, size_t pitch__cuf, double * q, size_t pitch__q, double ue[][162][5])
{
	/*
	   --------------------------------------------------------------------
	   c     xi-direction flux differences                      
	   c-------------------------------------------------------------------
	 */
	double forcing_0;
	int grid_points_0;
	double * cuf_0;
	double * q_0;
	double dtpp;
	double eta;
	int i;
	int im1;
	int ip1;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__dtemp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	q_0=(((double * )q)+_gtid);
	cuf_0=(((double * )cuf)+_gtid);
	grid_points_0=grid_points[0];
	j=(_gtid+1);
#pragma omp for shared(c1, c2, dnxm1, dnym1, dnzm1, dssp, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, forcing, grid_points, tx2, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5) private(buf, cuf, j, ue)
	if (j<(grid_points[1]-1))
	{
#pragma loop name exact_rhs#1#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name exact_rhs#1#0#0 
			for (i=0; i<grid_points_0; i ++ )
			{
				zeta=(((double)k)*const__dnzm1);
				eta=(((double)j)*const__dnym1);
				xi=(((double)i)*const__dnxm1);
				dev_exact_solution(xi, eta, zeta, sh__dtemp[threadIdx.x], ce, pitch__ce);
#pragma loop name exact_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[_gtid][i][m]=sh__dtemp[threadIdx.x][m];
				}
				dtpp=(1.0/sh__dtemp[threadIdx.x][0]);
#pragma loop name exact_rhs#1#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[_gtid][i][m]=(dtpp*sh__dtemp[threadIdx.x][m]);
				}
				( * ((double * )(((char * )cuf_0)+(i*pitch__cuf))))=(buf[_gtid][i][1]*buf[_gtid][i][1]);
				buf[_gtid][i][0]=((( * ((double * )(((char * )cuf_0)+(i*pitch__cuf))))+(buf[_gtid][i][2]*buf[_gtid][i][2]))+(buf[_gtid][i][3]*buf[_gtid][i][3]));
				( * ((double * )(((char * )q_0)+(i*pitch__q))))=(0.5*(((buf[_gtid][i][1]*ue[_gtid][i][1])+(buf[_gtid][i][2]*ue[_gtid][i][2]))+(buf[_gtid][i][3]*ue[_gtid][i][3])));
			}
#pragma loop name exact_rhs#1#0#1 
			for (i=1; i<(grid_points_0-1); i ++ )
			{
				im1=(i-1);
				ip1=(i+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(const__tx2*(ue[_gtid][ip1][1]-ue[_gtid][im1][1])))+(const__dx1tx1*((ue[_gtid][ip1][0]-(2.0*ue[_gtid][i][0]))+ue[_gtid][im1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(const__tx2*(((ue[_gtid][ip1][1]*buf[_gtid][ip1][1])+(const__c2*(ue[_gtid][ip1][4]-( * ((double * )(((char * )q_0)+(ip1*pitch__q)))))))-((ue[_gtid][im1][1]*buf[_gtid][im1][1])+(const__c2*(ue[_gtid][im1][4]-( * ((double * )(((char * )q_0)+(im1*pitch__q))))))))))+(const__xxcon1*((buf[_gtid][ip1][1]-(2.0*buf[_gtid][i][1]))+buf[_gtid][im1][1])))+(const__dx2tx1*((ue[_gtid][ip1][1]-(2.0*ue[_gtid][i][1]))+ue[_gtid][im1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(const__tx2*((ue[_gtid][ip1][2]*buf[_gtid][ip1][1])-(ue[_gtid][im1][2]*buf[_gtid][im1][1]))))+(const__xxcon2*((buf[_gtid][ip1][2]-(2.0*buf[_gtid][i][2]))+buf[_gtid][im1][2])))+(const__dx3tx1*((ue[_gtid][ip1][2]-(2.0*ue[_gtid][i][2]))+ue[_gtid][im1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(const__tx2*((ue[_gtid][ip1][3]*buf[_gtid][ip1][1])-(ue[_gtid][im1][3]*buf[_gtid][im1][1]))))+(const__xxcon2*((buf[_gtid][ip1][3]-(2.0*buf[_gtid][i][3]))+buf[_gtid][im1][3])))+(const__dx4tx1*((ue[_gtid][ip1][3]-(2.0*ue[_gtid][i][3]))+ue[_gtid][im1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(const__tx2*((buf[_gtid][ip1][1]*((const__c1*ue[_gtid][ip1][4])-(const__c2*( * ((double * )(((char * )q_0)+(ip1*pitch__q)))))))-(buf[_gtid][im1][1]*((const__c1*ue[_gtid][im1][4])-(const__c2*( * ((double * )(((char * )q_0)+(im1*pitch__q))))))))))+((0.5*const__xxcon3)*((buf[_gtid][ip1][0]-(2.0*buf[_gtid][i][0]))+buf[_gtid][im1][0])))+(const__xxcon4*((( * ((double * )(((char * )cuf_0)+(ip1*pitch__cuf))))-(2.0*( * ((double * )(((char * )cuf_0)+(i*pitch__cuf))))))+( * ((double * )(((char * )cuf_0)+(im1*pitch__cuf)))))))+(const__xxcon5*((buf[_gtid][ip1][4]-(2.0*buf[_gtid][i][4]))+buf[_gtid][im1][4])))+(const__dx5tx1*((ue[_gtid][ip1][4]-(2.0*ue[_gtid][i][4]))+ue[_gtid][im1][4])));
			}
			/*
			   --------------------------------------------------------------------
			   c     Fourth-order dissipation                         
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#1#0#2 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				i=1;
				forcing_0=(forcing_0-(const__dssp*(((5.0*ue[_gtid][i][m])-(4.0*ue[_gtid][(i+1)][m]))+ue[_gtid][(i+2)][m])));
				i=2;
				forcing_0=(forcing_0-(const__dssp*((((( - 4.0)*ue[_gtid][(i-1)][m])+(6.0*ue[_gtid][i][m]))-(4.0*ue[_gtid][(i+1)][m]))+ue[_gtid][(i+2)][m])));
				forcing[i][j][k][m]=forcing_0;
			}
#pragma loop name exact_rhs#1#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#1#0#3#0 
				for (i=(1*3); i<=((grid_points_0-(3*1))-1); i ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(const__dssp*((((ue[_gtid][(i-2)][m]-(4.0*ue[_gtid][(i-1)][m]))+(6.0*ue[_gtid][i][m]))-(4.0*ue[_gtid][(i+1)][m]))+ue[_gtid][(i+2)][m])));
				}
			}
#pragma loop name exact_rhs#1#0#4 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				i=(grid_points_0-3);
				forcing_0=(forcing_0-(const__dssp*(((ue[_gtid][(i-2)][m]-(4.0*ue[_gtid][(i-1)][m]))+(6.0*ue[_gtid][i][m]))-(4.0*ue[_gtid][(i+1)][m]))));
				i=(grid_points_0-2);
				forcing_0=(forcing_0-(const__dssp*((ue[_gtid][(i-2)][m]-(4.0*ue[_gtid][(i-1)][m]))+(5.0*ue[_gtid][i][m]))));
				forcing[i][j][k][m]=forcing_0;
			}
		}
	}
}

__global__ void exact_rhs_kernel2(double * ce, size_t pitch__ce, double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double buf[][162][5], double * cuf, size_t pitch__cuf, double * q, size_t pitch__q, double ue[][162][5])
{
	/*
	   --------------------------------------------------------------------
	   c     eta-direction flux differences             
	   c-------------------------------------------------------------------
	 */
	double forcing_0;
	int grid_points_0;
	double * cuf_0;
	double * q_0;
	double dtpp;
	double eta;
	int i;
	int j;
	int jm1;
	int jp1;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__dtemp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	q_0=(((double * )q)+_gtid);
	cuf_0=(((double * )cuf)+_gtid);
	grid_points_0=grid_points[1];
	i=(_gtid+1);
#pragma omp for shared(c1, c2, dnxm1, dnym1, dnzm1, dssp, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, forcing, grid_points, ty2, yycon1, yycon2, yycon3, yycon4, yycon5) private(i, ue)
	if (i<(grid_points[0]-1))
	{
#pragma loop name exact_rhs#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name exact_rhs#2#0#0 
			for (j=0; j<grid_points_0; j ++ )
			{
				zeta=(((double)k)*const__dnzm1);
				xi=(((double)i)*const__dnxm1);
				eta=(((double)j)*const__dnym1);
				dev_exact_solution(xi, eta, zeta, sh__dtemp[threadIdx.x], ce, pitch__ce);
#pragma loop name exact_rhs#2#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[_gtid][j][m]=sh__dtemp[threadIdx.x][m];
				}
				dtpp=(1.0/sh__dtemp[threadIdx.x][0]);
#pragma loop name exact_rhs#2#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[_gtid][j][m]=(dtpp*sh__dtemp[threadIdx.x][m]);
				}
				( * ((double * )(((char * )cuf_0)+(j*pitch__cuf))))=(buf[_gtid][j][2]*buf[_gtid][j][2]);
				buf[_gtid][j][0]=((( * ((double * )(((char * )cuf_0)+(j*pitch__cuf))))+(buf[_gtid][j][1]*buf[_gtid][j][1]))+(buf[_gtid][j][3]*buf[_gtid][j][3]));
				( * ((double * )(((char * )q_0)+(j*pitch__q))))=(0.5*(((buf[_gtid][j][1]*ue[_gtid][j][1])+(buf[_gtid][j][2]*ue[_gtid][j][2]))+(buf[_gtid][j][3]*ue[_gtid][j][3])));
			}
#pragma loop name exact_rhs#2#0#1 
			for (j=1; j<(grid_points_0-1); j ++ )
			{
				jm1=(j-1);
				jp1=(j+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(const__ty2*(ue[_gtid][jp1][2]-ue[_gtid][jm1][2])))+(const__dy1ty1*((ue[_gtid][jp1][0]-(2.0*ue[_gtid][j][0]))+ue[_gtid][jm1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(const__ty2*((ue[_gtid][jp1][1]*buf[_gtid][jp1][2])-(ue[_gtid][jm1][1]*buf[_gtid][jm1][2]))))+(const__yycon2*((buf[_gtid][jp1][1]-(2.0*buf[_gtid][j][1]))+buf[_gtid][jm1][1])))+(const__dy2ty1*((ue[_gtid][jp1][1]-(2.0*ue[_gtid][j][1]))+ue[_gtid][jm1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(const__ty2*(((ue[_gtid][jp1][2]*buf[_gtid][jp1][2])+(const__c2*(ue[_gtid][jp1][4]-( * ((double * )(((char * )q_0)+(jp1*pitch__q)))))))-((ue[_gtid][jm1][2]*buf[_gtid][jm1][2])+(const__c2*(ue[_gtid][jm1][4]-( * ((double * )(((char * )q_0)+(jm1*pitch__q))))))))))+(const__yycon1*((buf[_gtid][jp1][2]-(2.0*buf[_gtid][j][2]))+buf[_gtid][jm1][2])))+(const__dy3ty1*((ue[_gtid][jp1][2]-(2.0*ue[_gtid][j][2]))+ue[_gtid][jm1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(const__ty2*((ue[_gtid][jp1][3]*buf[_gtid][jp1][2])-(ue[_gtid][jm1][3]*buf[_gtid][jm1][2]))))+(const__yycon2*((buf[_gtid][jp1][3]-(2.0*buf[_gtid][j][3]))+buf[_gtid][jm1][3])))+(const__dy4ty1*((ue[_gtid][jp1][3]-(2.0*ue[_gtid][j][3]))+ue[_gtid][jm1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(const__ty2*((buf[_gtid][jp1][2]*((const__c1*ue[_gtid][jp1][4])-(const__c2*( * ((double * )(((char * )q_0)+(jp1*pitch__q)))))))-(buf[_gtid][jm1][2]*((const__c1*ue[_gtid][jm1][4])-(const__c2*( * ((double * )(((char * )q_0)+(jm1*pitch__q))))))))))+((0.5*const__yycon3)*((buf[_gtid][jp1][0]-(2.0*buf[_gtid][j][0]))+buf[_gtid][jm1][0])))+(const__yycon4*((( * ((double * )(((char * )cuf_0)+(jp1*pitch__cuf))))-(2.0*( * ((double * )(((char * )cuf_0)+(j*pitch__cuf))))))+( * ((double * )(((char * )cuf_0)+(jm1*pitch__cuf)))))))+(const__yycon5*((buf[_gtid][jp1][4]-(2.0*buf[_gtid][j][4]))+buf[_gtid][jm1][4])))+(const__dy5ty1*((ue[_gtid][jp1][4]-(2.0*ue[_gtid][j][4]))+ue[_gtid][jm1][4])));
			}
			/*
			   --------------------------------------------------------------------
			   c     Fourth-order dissipation                      
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#2#0#2 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				j=1;
				forcing_0=(forcing_0-(const__dssp*(((5.0*ue[_gtid][j][m])-(4.0*ue[_gtid][(j+1)][m]))+ue[_gtid][(j+2)][m])));
				j=2;
				forcing_0=(forcing_0-(const__dssp*((((( - 4.0)*ue[_gtid][(j-1)][m])+(6.0*ue[_gtid][j][m]))-(4.0*ue[_gtid][(j+1)][m]))+ue[_gtid][(j+2)][m])));
				forcing[i][j][k][m]=forcing_0;
			}
#pragma loop name exact_rhs#2#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#2#0#3#0 
				for (j=(1*3); j<=((grid_points_0-(3*1))-1); j ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(const__dssp*((((ue[_gtid][(j-2)][m]-(4.0*ue[_gtid][(j-1)][m]))+(6.0*ue[_gtid][j][m]))-(4.0*ue[_gtid][(j+1)][m]))+ue[_gtid][(j+2)][m])));
				}
			}
#pragma loop name exact_rhs#2#0#4 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				j=(grid_points_0-3);
				forcing_0=(forcing_0-(const__dssp*(((ue[_gtid][(j-2)][m]-(4.0*ue[_gtid][(j-1)][m]))+(6.0*ue[_gtid][j][m]))-(4.0*ue[_gtid][(j+1)][m]))));
				j=(grid_points_0-2);
				forcing_0=(forcing_0-(const__dssp*((ue[_gtid][(j-2)][m]-(4.0*ue[_gtid][(j-1)][m]))+(5.0*ue[_gtid][j][m]))));
				forcing[i][j][k][m]=forcing_0;
			}
		}
	}
}

__global__ void exact_rhs_kernel3(double * ce, size_t pitch__ce, double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double buf[][162][5], double * cuf, size_t pitch__cuf, double * q, size_t pitch__q, double ue[][162][5])
{
	/*
	   --------------------------------------------------------------------
	   c     zeta-direction flux differences                      
	   c-------------------------------------------------------------------
	 */
	double forcing_0;
	int grid_points_0;
	double * cuf_0;
	double * q_0;
	double dtpp;
	double eta;
	int i;
	int j;
	int k;
	int km1;
	int kp1;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__dtemp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	q_0=(((double * )q)+_gtid);
	cuf_0=(((double * )cuf)+_gtid);
	grid_points_0=grid_points[2];
	i=(_gtid+1);
#pragma omp for shared(c1, c2, dnxm1, dnym1, dnzm1, dssp, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, forcing, grid_points, tz2, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) private(i, ue)
	if (i<(grid_points[0]-1))
	{
#pragma loop name exact_rhs#3#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name exact_rhs#3#0#0 
			for (k=0; k<grid_points_0; k ++ )
			{
				eta=(((double)j)*const__dnym1);
				xi=(((double)i)*const__dnxm1);
				zeta=(((double)k)*const__dnzm1);
				dev_exact_solution(xi, eta, zeta, sh__dtemp[threadIdx.x], ce, pitch__ce);
#pragma loop name exact_rhs#3#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[_gtid][k][m]=sh__dtemp[threadIdx.x][m];
				}
				dtpp=(1.0/sh__dtemp[threadIdx.x][0]);
#pragma loop name exact_rhs#3#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[_gtid][k][m]=(dtpp*sh__dtemp[threadIdx.x][m]);
				}
				( * ((double * )(((char * )cuf_0)+(k*pitch__cuf))))=(buf[_gtid][k][3]*buf[_gtid][k][3]);
				buf[_gtid][k][0]=((( * ((double * )(((char * )cuf_0)+(k*pitch__cuf))))+(buf[_gtid][k][1]*buf[_gtid][k][1]))+(buf[_gtid][k][2]*buf[_gtid][k][2]));
				( * ((double * )(((char * )q_0)+(k*pitch__q))))=(0.5*(((buf[_gtid][k][1]*ue[_gtid][k][1])+(buf[_gtid][k][2]*ue[_gtid][k][2]))+(buf[_gtid][k][3]*ue[_gtid][k][3])));
			}
#pragma loop name exact_rhs#3#0#1 
			for (k=1; k<(grid_points_0-1); k ++ )
			{
				km1=(k-1);
				kp1=(k+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(const__tz2*(ue[_gtid][kp1][3]-ue[_gtid][km1][3])))+(const__dz1tz1*((ue[_gtid][kp1][0]-(2.0*ue[_gtid][k][0]))+ue[_gtid][km1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(const__tz2*((ue[_gtid][kp1][1]*buf[_gtid][kp1][3])-(ue[_gtid][km1][1]*buf[_gtid][km1][3]))))+(const__zzcon2*((buf[_gtid][kp1][1]-(2.0*buf[_gtid][k][1]))+buf[_gtid][km1][1])))+(const__dz2tz1*((ue[_gtid][kp1][1]-(2.0*ue[_gtid][k][1]))+ue[_gtid][km1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(const__tz2*((ue[_gtid][kp1][2]*buf[_gtid][kp1][3])-(ue[_gtid][km1][2]*buf[_gtid][km1][3]))))+(const__zzcon2*((buf[_gtid][kp1][2]-(2.0*buf[_gtid][k][2]))+buf[_gtid][km1][2])))+(const__dz3tz1*((ue[_gtid][kp1][2]-(2.0*ue[_gtid][k][2]))+ue[_gtid][km1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(const__tz2*(((ue[_gtid][kp1][3]*buf[_gtid][kp1][3])+(const__c2*(ue[_gtid][kp1][4]-( * ((double * )(((char * )q_0)+(kp1*pitch__q)))))))-((ue[_gtid][km1][3]*buf[_gtid][km1][3])+(const__c2*(ue[_gtid][km1][4]-( * ((double * )(((char * )q_0)+(km1*pitch__q))))))))))+(const__zzcon1*((buf[_gtid][kp1][3]-(2.0*buf[_gtid][k][3]))+buf[_gtid][km1][3])))+(const__dz4tz1*((ue[_gtid][kp1][3]-(2.0*ue[_gtid][k][3]))+ue[_gtid][km1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(const__tz2*((buf[_gtid][kp1][3]*((const__c1*ue[_gtid][kp1][4])-(const__c2*( * ((double * )(((char * )q_0)+(kp1*pitch__q)))))))-(buf[_gtid][km1][3]*((const__c1*ue[_gtid][km1][4])-(const__c2*( * ((double * )(((char * )q_0)+(km1*pitch__q))))))))))+((0.5*const__zzcon3)*((buf[_gtid][kp1][0]-(2.0*buf[_gtid][k][0]))+buf[_gtid][km1][0])))+(const__zzcon4*((( * ((double * )(((char * )cuf_0)+(kp1*pitch__cuf))))-(2.0*( * ((double * )(((char * )cuf_0)+(k*pitch__cuf))))))+( * ((double * )(((char * )cuf_0)+(km1*pitch__cuf)))))))+(const__zzcon5*((buf[_gtid][kp1][4]-(2.0*buf[_gtid][k][4]))+buf[_gtid][km1][4])))+(const__dz5tz1*((ue[_gtid][kp1][4]-(2.0*ue[_gtid][k][4]))+ue[_gtid][km1][4])));
			}
			/*
			   --------------------------------------------------------------------
			   c     Fourth-order dissipation                        
			   c-------------------------------------------------------------------
			 */
#pragma loop name exact_rhs#3#0#2 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				k=1;
				forcing_0=(forcing_0-(const__dssp*(((5.0*ue[_gtid][k][m])-(4.0*ue[_gtid][(k+1)][m]))+ue[_gtid][(k+2)][m])));
				k=2;
				forcing_0=(forcing_0-(const__dssp*((((( - 4.0)*ue[_gtid][(k-1)][m])+(6.0*ue[_gtid][k][m]))-(4.0*ue[_gtid][(k+1)][m]))+ue[_gtid][(k+2)][m])));
				forcing[i][j][k][m]=forcing_0;
			}
#pragma loop name exact_rhs#3#0#3 
			for (m=0; m<5; m ++ )
			{
#pragma loop name exact_rhs#3#0#3#0 
				for (k=(1*3); k<=((grid_points_0-(3*1))-1); k ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(const__dssp*((((ue[_gtid][(k-2)][m]-(4.0*ue[_gtid][(k-1)][m]))+(6.0*ue[_gtid][k][m]))-(4.0*ue[_gtid][(k+1)][m]))+ue[_gtid][(k+2)][m])));
				}
			}
#pragma loop name exact_rhs#3#0#4 
			for (m=0; m<5; m ++ )
			{
				forcing_0=forcing[i][j][k][m];
				k=(grid_points_0-3);
				forcing_0=(forcing_0-(const__dssp*(((ue[_gtid][(k-2)][m]-(4.0*ue[_gtid][(k-1)][m]))+(6.0*ue[_gtid][k][m]))-(4.0*ue[_gtid][(k+1)][m]))));
				k=(grid_points_0-2);
				forcing_0=(forcing_0-(const__dssp*((ue[_gtid][(k-2)][m]-(4.0*ue[_gtid][(k-1)][m]))+(5.0*ue[_gtid][k][m]))));
				forcing[i][j][k][m]=forcing_0;
			}
		}
	}
}

__global__ void exact_rhs_kernel4(double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points)
{
	/*
	   --------------------------------------------------------------------
	   c     now change the sign of the forcing function, 
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(forcing, grid_points) private(i)
	if (m<5)
	{
#pragma loop name exact_rhs#4#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name exact_rhs#4#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name exact_rhs#4#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					forcing[i][j][k][m]=(( - 1.0)*forcing[i][j][k][m]);
				}
			}
		}
	}
}

static void exact_rhs(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     compute the right hand side based on exact solution
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	double * gpu__buf;
	double * gpu__cuf;
	size_t pitch__cuf;
	double * gpu__q;
	size_t pitch__q;
	double * gpu__ue;
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon5, ( & zzcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon4, ( & zzcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon3, ( & zzcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon2, ( & zzcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon1, ( & zzcon1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5tz1, ( & dz5tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4tz1, ( & dz4tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3tz1, ( & dz3tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2tz1, ( & dz2tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1tz1, ( & dz1tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon5, ( & yycon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon4, ( & yycon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon3, ( & yycon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon2, ( & yycon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon1, ( & yycon1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5ty1, ( & dy5ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4ty1, ( & dy4ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3ty1, ( & dy3ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2ty1, ( & dy2ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1ty1, ( & dy1ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon5, ( & xxcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon4, ( & xxcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon3, ( & xxcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon2, ( & xxcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon1, ( & xxcon1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5tx1, ( & dx5tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4tx1, ( & dx4tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3tx1, ( & dx3tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2tx1, ( & dx2tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1tx1, ( & dx1tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
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
#pragma omp parallel shared(forcing, grid_points) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, forcing, grid_points, q, ue) nocudafree(buf, cuf, forcing, grid_points, q, u, ue) nog2cmemtr(buf, cuf, forcing, grid_points, q, ue) 
#pragma cuda gpurun nocudamalloc(grid_points) 
#pragma cuda ainfo kernelid(0) procname(exact_rhs) 
	exact_rhs_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points);
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__buf)), gpuBytes));
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__cuf)), ( & pitch__cuf), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__cuf*162);
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__q)), ( & pitch__q), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__q*162);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ue)), gpuBytes));
#pragma omp parallel threadprivate(buf, cuf, q, ue) shared(c1, c2, ce, dnxm1, dnym1, dnzm1, dssp, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, forcing, grid_points, tx2, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5) private(dtemp, dtpp, eta, i, im1, ip1, j, k, m, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, ce, cuf, dnxm1, dnym1, dnzm1, forcing, grid_points, q, ue) nocudafree(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, forcing, grid_points, q, tx2, u, ue, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5) nog2cmemtr(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, forcing, grid_points, q, tx2, ue, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnym1, dnzm1, forcing, grid_points) 
#pragma cuda ainfo kernelid(1) procname(exact_rhs) 
#pragma cuda gpurun registerRO(grid_points[0]) 
#pragma cuda gpurun registerRW(forcing[i][j][k][m]) 
#pragma cuda gpurun sharedRW(dtemp) 
#pragma cuda gpurun constant(c1, c2, dnxm1, dnym1, dnzm1, dssp, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, tx2, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5) 
	exact_rhs_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__ce, pitch__ce, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[162][5])gpu__buf), gpu__cuf, pitch__cuf, gpu__q, pitch__q, ((double (*)[162][5])gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__q));
	CUDA_SAFE_CALL(cudaFree(gpu__cuf));
	CUDA_SAFE_CALL(cudaFree(gpu__buf));
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__buf)), gpuBytes));
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__cuf)), ( & pitch__cuf), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__cuf*162);
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__q)), ( & pitch__q), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__q*162);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ue)), gpuBytes));
#pragma omp parallel threadprivate(buf, cuf, q, ue) shared(c1, c2, ce, dnxm1, dnym1, dnzm1, dssp, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, forcing, grid_points, ty2, yycon1, yycon2, yycon3, yycon4, yycon5) private(dtemp, dtpp, eta, i, j, jm1, jp1, k, m, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, forcing, grid_points, q, ue) nocudafree(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, forcing, grid_points, q, ty2, u, ue, yycon1, yycon2, yycon3, yycon4, yycon5) nog2cmemtr(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, forcing, grid_points, q, ty2, ue, yycon1, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, ce, dnxm1, dnym1, dnzm1, dssp, forcing, grid_points) 
#pragma cuda ainfo kernelid(2) procname(exact_rhs) 
#pragma cuda gpurun registerRO(grid_points[1]) 
#pragma cuda gpurun registerRW(forcing[i][j][k][m]) 
#pragma cuda gpurun sharedRW(dtemp) 
#pragma cuda gpurun constant(c1, c2, dnxm1, dnym1, dnzm1, dssp, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, ty2, yycon1, yycon2, yycon3, yycon4, yycon5) 
	exact_rhs_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__ce, pitch__ce, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[162][5])gpu__buf), gpu__cuf, pitch__cuf, gpu__q, pitch__q, ((double (*)[162][5])gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__q));
	CUDA_SAFE_CALL(cudaFree(gpu__cuf));
	CUDA_SAFE_CALL(cudaFree(gpu__buf));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__buf)), gpuBytes));
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__cuf)), ( & pitch__cuf), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__cuf*162);
	CUDA_SAFE_CALL(cudaMallocPitch(((void *  * )( & gpu__q)), ( & pitch__q), (totalNumThreads*sizeof (double)), 162));
	gpuBytes=(pitch__q*162);
	gpuBytes=(totalNumThreads*((162*5)*sizeof (double)));
	CUDA_SAFE_CALL(cudaMalloc(((void *  * )( & gpu__ue)), gpuBytes));
#pragma omp parallel threadprivate(buf, cuf, q, ue) shared(c1, c2, ce, dnxm1, dnym1, dnzm1, dssp, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, forcing, grid_points, tz2, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) private(dtemp, dtpp, eta, i, j, k, km1, kp1, m, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, forcing, grid_points, q, ue) nocudafree(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, forcing, grid_points, q, tz2, u, ue, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) nog2cmemtr(buf, c1, c2, ce, cuf, dnxm1, dnym1, dnzm1, dssp, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, forcing, grid_points, q, tz2, ue, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, ce, dnxm1, dnym1, dnzm1, dssp, forcing, grid_points) 
#pragma cuda ainfo kernelid(3) procname(exact_rhs) 
#pragma cuda gpurun registerRO(grid_points[2]) 
#pragma cuda gpurun registerRW(forcing[i][j][k][m]) 
#pragma cuda gpurun sharedRW(dtemp) 
#pragma cuda gpurun constant(c1, c2, dnxm1, dnym1, dnzm1, dssp, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, tz2, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5) 
	exact_rhs_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__ce, pitch__ce, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[162][5])gpu__buf), gpu__cuf, pitch__cuf, gpu__q, pitch__q, ((double (*)[162][5])gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__ue));
	CUDA_SAFE_CALL(cudaFree(gpu__q));
	CUDA_SAFE_CALL(cudaFree(gpu__cuf));
	CUDA_SAFE_CALL(cudaFree(gpu__buf));
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
#pragma omp parallel shared(forcing, grid_points) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, forcing, grid_points, q, ue) nocudafree(buf, cuf, forcing, grid_points, q, u, ue) nog2cmemtr(buf, cuf, forcing, grid_points, q, ue) 
#pragma cuda gpurun nocudamalloc(forcing, grid_points) 
#pragma cuda ainfo kernelid(4) procname(exact_rhs) 
	exact_rhs_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points);
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__device__ static void dev_exact_solution(double xi, double eta, double zeta, double dtemp[5], double * ce, size_t pitch__ce)
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     this function returns the exact solution at point xi, eta, zeta  
	   c-------------------------------------------------------------------
	 */
	int m;
#pragma loop name exact_solution#0 
	for (m=0; m<5; m ++ )
	{
		dtemp[m]=(((( * (((double * )(((char * )ce)+(m*pitch__ce)))+0))+(xi*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+1))+(xi*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+4))+(xi*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+7))+(xi*( * (((double * )(((char * )ce)+(m*pitch__ce)))+10))))))))))+(eta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+2))+(eta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+5))+(eta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+8))+(eta*( * (((double * )(((char * )ce)+(m*pitch__ce)))+11))))))))))+(zeta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+3))+(zeta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+6))+(zeta*(( * (((double * )(((char * )ce)+(m*pitch__ce)))+9))+(zeta*( * (((double * )(((char * )ce)+(m*pitch__ce)))+12))))))))));
	}
	return ;
}

static void exact_solution(double xi, double eta, double zeta, double dtemp[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     this function returns the exact solution at point xi, eta, zeta  
	   c-------------------------------------------------------------------
	 */
	int m;
#pragma loop name exact_solution#0 
	for (m=0; m<5; m ++ )
	{
		dtemp[m]=(((ce[m][0]+(xi*(ce[m][1]+(xi*(ce[m][4]+(xi*(ce[m][7]+(xi*ce[m][10]))))))))+(eta*(ce[m][2]+(eta*(ce[m][5]+(eta*(ce[m][8]+(eta*ce[m][11]))))))))+(zeta*(ce[m][3]+(zeta*(ce[m][6]+(zeta*(ce[m][9]+(zeta*ce[m][12]))))))));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void initialize_kernel0(double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c  Later (in compute_rhs) we compute 1u for every element. A few of 
	   c  the corner elements are not used, but it convenient (and faster) 
	   c  to compute the whole thing with a simple loop. Make sure those 
	   c  values are nonzero by initializing the whole thing here. 
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(u) private(i)
	if (m<5)
	{
#pragma loop name initialize#0#0 
		for (k=0; k<162; k ++ )
		{
#pragma loop name initialize#0#0#0 
			for (j=0; j<162; j ++ )
			{
#pragma loop name initialize#0#0#0#0 
				for (i=0; i<162; i ++ )
				{
					u[i][j][k][m]=1.0;
				}
			}
		}
	}
}

__global__ void initialize_kernel1(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     first store the "interpolated" values everywhere on the grid    
	   c-------------------------------------------------------------------
	 */
	double Peta;
	double Pface[2][3][5];
	double Pxi;
	double Pzeta;
	double eta;
	int i;
	int ix;
	int iy;
	int iz;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, dnzm1, grid_points, u) private(Pface, i)
	if (i<grid_points[0])
	{
#pragma loop name initialize#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
#pragma loop name initialize#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				xi=(((double)i)*const__dnxm1);
				eta=(((double)j)*const__dnym1);
				zeta=(((double)k)*const__dnzm1);
#pragma loop name initialize#1#0#0#0 
				for (ix=0; ix<2; ix ++ )
				{
					dev_exact_solution(((double)ix), eta, zeta, ( & Pface[ix][0][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#1 
				for (iy=0; iy<2; iy ++ )
				{
					dev_exact_solution(xi, ((double)iy), zeta, ( & Pface[iy][1][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#2 
				for (iz=0; iz<2; iz ++ )
				{
					dev_exact_solution(xi, eta, ((double)iz), ( & Pface[iz][2][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#3 
				for (m=0; m<5; m ++ )
				{
					Pxi=((xi*Pface[1][0][m])+((1.0-xi)*Pface[0][0][m]));
					Peta=((eta*Pface[1][1][m])+((1.0-eta)*Pface[0][1][m]));
					Pzeta=((zeta*Pface[1][2][m])+((1.0-zeta)*Pface[0][2][m]));
					u[i][j][k][m]=((((((Pxi+Peta)+Pzeta)-(Pxi*Peta))-(Pxi*Pzeta))-(Peta*Pzeta))+((Pxi*Peta)*Pzeta));
				}
			}
		}
	}
}

__global__ void initialize_kernel2(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now store the exact values on the boundaries        
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     west face                                                  
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[1];
	grid_points_0=grid_points[2];
	i=0;
	xi=0.0;
	j=_gtid;
#pragma omp for shared(dnym1, dnzm1, grid_points, u) private(j) nowait
	if (j<grid_points_1)
	{
#pragma loop name initialize#2#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			eta=(((double)j)*const__dnym1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#2#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     east face                                                      
	   c-------------------------------------------------------------------
	 */
	i=(grid_points[0]-1);
	xi=1.0;
	j=_gtid;
#pragma omp for shared(dnym1, dnzm1, grid_points, u) private(j)
	if (j<grid_points_1)
	{
#pragma loop name initialize#3#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			eta=(((double)j)*const__dnym1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#3#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

__global__ void initialize_kernel3(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     south face                                                 
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	j=0;
	eta=0.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnzm1, grid_points, u) private(i) nowait
	if (i<grid_points_1)
	{
#pragma loop name initialize#4#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			xi=(((double)i)*const__dnxm1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#4#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     north face                                    
	   c-------------------------------------------------------------------
	 */
	j=(grid_points[1]-1);
	eta=1.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnzm1, grid_points, u) private(i)
	if (i<grid_points_1)
	{
#pragma loop name initialize#5#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			xi=(((double)i)*const__dnxm1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#5#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

__global__ void initialize_kernel4(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     bottom face                                       
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[1];
	grid_points_0=grid_points[0];
	k=0;
	zeta=0.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, grid_points, u) private(i) nowait
	if (i<grid_points_0)
	{
		xi=(((double)i)*const__dnxm1);
#pragma loop name initialize#6#0 
		for (j=0; j<grid_points_1; j ++ )
		{
			eta=(((double)j)*const__dnym1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#6#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     top face     
	   c-------------------------------------------------------------------
	 */
	k=(grid_points[2]-1);
	zeta=1.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, grid_points, u) private(i)
	if (i<grid_points_0)
	{
#pragma loop name initialize#7#0 
		for (j=0; j<grid_points_1; j ++ )
		{
			xi=(((double)i)*const__dnxm1);
			eta=(((double)j)*const__dnym1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#7#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

static void initialize(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     This subroutine initializes the field variable u using 
	   c     tri-linear transfinite interpolation of the boundary values     
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
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
#pragma omp parallel shared(u) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, q, u, ue) nocudafree(buf, cuf, q, u, ue) nog2cmemtr(buf, cuf, q, u, ue) 
#pragma cuda ainfo kernelid(0) procname(initialize) 
	initialize_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
	CUDA_SAFE_CALL(cudaMemcpy2D(gpu__ce, pitch__ce, ce, (13*sizeof (double)), (13*sizeof (double)), 5, cudaMemcpyHostToDevice));
	gpuBytes=(3*sizeof (int));
	CUDA_SAFE_CALL(cudaMemcpy(gpu__grid_points, grid_points, gpuBytes, cudaMemcpyHostToDevice));
#pragma omp parallel shared(ce, dnxm1, dnym1, dnzm1, grid_points, u) private(Peta, Pface, Pxi, Pzeta, eta, i, ix, iy, iz, j, k, m, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnym1, dnzm1, grid_points, q, u, ue) nog2cmemtr(buf, ce, cuf, dnxm1, dnym1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(u) 
#pragma cuda ainfo kernelid(1) procname(initialize) 
#pragma cuda gpurun constant(dnxm1, dnym1, dnzm1) 
	initialize_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[1])/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnym1, dnzm1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) nog2cmemtr(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnym1, dnzm1, grid_points, u) 
#pragma cuda ainfo kernelid(2) procname(initialize) 
#pragma cuda gpurun registerRO(grid_points[1], grid_points[2]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnym1, dnzm1) 
	initialize_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnxm1, dnzm1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) nog2cmemtr(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnzm1, grid_points, u) 
#pragma cuda ainfo kernelid(3) procname(initialize) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[2]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnxm1, dnzm1) 
	initialize_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock4(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnxm1, dnym1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) nog2cmemtr(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnym1, grid_points, u) 
#pragma cuda ainfo kernelid(4) procname(initialize) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnxm1, dnym1) 
	initialize_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	return ;
}

__global__ void initialize_clnd1_kernel0(double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c  Later (in compute_rhs) we compute 1u for every element. A few of 
	   c  the corner elements are not used, but it convenient (and faster) 
	   c  to compute the whole thing with a simple loop. Make sure those 
	   c  values are nonzero by initializing the whole thing here. 
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(u) private(i)
	if (m<5)
	{
#pragma loop name initialize#0#0 
		for (k=0; k<162; k ++ )
		{
#pragma loop name initialize#0#0#0 
			for (j=0; j<162; j ++ )
			{
#pragma loop name initialize#0#0#0#0 
				for (i=0; i<162; i ++ )
				{
					u[i][j][k][m]=1.0;
				}
			}
		}
	}
}

__global__ void initialize_clnd1_kernel1(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     first store the "interpolated" values everywhere on the grid    
	   c-------------------------------------------------------------------
	 */
	double Peta;
	double Pface[2][3][5];
	double Pxi;
	double Pzeta;
	double eta;
	int i;
	int ix;
	int iy;
	int iz;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, dnzm1, grid_points, u) private(Pface, i)
	if (i<grid_points[0])
	{
#pragma loop name initialize#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
#pragma loop name initialize#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				xi=(((double)i)*const__dnxm1);
				eta=(((double)j)*const__dnym1);
				zeta=(((double)k)*const__dnzm1);
#pragma loop name initialize#1#0#0#0 
				for (ix=0; ix<2; ix ++ )
				{
					dev_exact_solution(((double)ix), eta, zeta, ( & Pface[ix][0][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#1 
				for (iy=0; iy<2; iy ++ )
				{
					dev_exact_solution(xi, ((double)iy), zeta, ( & Pface[iy][1][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#2 
				for (iz=0; iz<2; iz ++ )
				{
					dev_exact_solution(xi, eta, ((double)iz), ( & Pface[iz][2][0]), ce, pitch__ce);
				}
#pragma loop name initialize#1#0#0#3 
				for (m=0; m<5; m ++ )
				{
					Pxi=((xi*Pface[1][0][m])+((1.0-xi)*Pface[0][0][m]));
					Peta=((eta*Pface[1][1][m])+((1.0-eta)*Pface[0][1][m]));
					Pzeta=((zeta*Pface[1][2][m])+((1.0-zeta)*Pface[0][2][m]));
					u[i][j][k][m]=((((((Pxi+Peta)+Pzeta)-(Pxi*Peta))-(Pxi*Pzeta))-(Peta*Pzeta))+((Pxi*Peta)*Pzeta));
				}
			}
		}
	}
}

__global__ void initialize_clnd1_kernel2(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now store the exact values on the boundaries        
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     west face                                                  
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[1];
	grid_points_0=grid_points[2];
	i=0;
	xi=0.0;
	j=_gtid;
#pragma omp for shared(dnym1, dnzm1, grid_points, u) private(j) nowait
	if (j<grid_points_1)
	{
#pragma loop name initialize#2#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			eta=(((double)j)*const__dnym1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#2#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     east face                                                      
	   c-------------------------------------------------------------------
	 */
	i=(grid_points[0]-1);
	xi=1.0;
	j=_gtid;
#pragma omp for shared(dnym1, dnzm1, grid_points, u) private(j)
	if (j<grid_points_1)
	{
#pragma loop name initialize#3#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			eta=(((double)j)*const__dnym1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#3#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

__global__ void initialize_clnd1_kernel3(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     south face                                                 
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	j=0;
	eta=0.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnzm1, grid_points, u) private(i) nowait
	if (i<grid_points_1)
	{
#pragma loop name initialize#4#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			xi=(((double)i)*const__dnxm1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#4#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     north face                                    
	   c-------------------------------------------------------------------
	 */
	j=(grid_points[1]-1);
	eta=1.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnzm1, grid_points, u) private(i)
	if (i<grid_points_1)
	{
#pragma loop name initialize#5#0 
		for (k=0; k<grid_points_0; k ++ )
		{
			xi=(((double)i)*const__dnxm1);
			zeta=(((double)k)*const__dnzm1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#5#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

__global__ void initialize_clnd1_kernel4(double * ce, size_t pitch__ce, int * grid_points, double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     bottom face                                       
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	double eta;
	int i;
	int j;
	int k;
	int m;
	double xi;
	double zeta;
	__shared__ double sh__temp[BLOCK_SIZE][5];
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_1=grid_points[1];
	grid_points_0=grid_points[0];
	k=0;
	zeta=0.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, grid_points, u) private(i) nowait
	if (i<grid_points_0)
	{
		xi=(((double)i)*const__dnxm1);
#pragma loop name initialize#6#0 
		for (j=0; j<grid_points_1; j ++ )
		{
			eta=(((double)j)*const__dnym1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#6#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c     top face     
	   c-------------------------------------------------------------------
	 */
	k=(grid_points[2]-1);
	zeta=1.0;
	i=_gtid;
#pragma omp for shared(dnxm1, dnym1, grid_points, u) private(i)
	if (i<grid_points_0)
	{
#pragma loop name initialize#7#0 
		for (j=0; j<grid_points_1; j ++ )
		{
			xi=(((double)i)*const__dnxm1);
			eta=(((double)j)*const__dnym1);
			dev_exact_solution(xi, eta, zeta, sh__temp[threadIdx.x], ce, pitch__ce);
#pragma loop name initialize#7#0#0 
			for (m=0; m<5; m ++ )
			{
				u[i][j][k][m]=sh__temp[threadIdx.x][m];
			}
		}
	}
}

static void initialize_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     This subroutine initializes the field variable u using 
	   c     tri-linear transfinite interpolation of the boundary values     
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnzm1, ( & dnzm1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnym1, ( & dnym1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dnxm1, ( & dnxm1), gpuBytes));
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
#pragma omp parallel shared(u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, q, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, q, u, ue) nocudafree(buf, cuf, q, u, ue) 
#pragma cuda gpurun nocudamalloc(u) 
#pragma cuda ainfo kernelid(0) procname(initialize_clnd1) 
	initialize_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnxm1, dnym1, dnzm1, grid_points, u) private(Peta, Pface, Pxi, Pzeta, eta, i, ix, iy, iz, j, k, m, xi, zeta)
#pragma cuda gpurun noc2gmemtr(buf, ce, cuf, dnxm1, dnym1, dnzm1, grid_points, q, u, ue) noshared(Pface) nog2cmemtr(buf, ce, cuf, dnxm1, dnym1, dnzm1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnym1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnym1, dnzm1, grid_points, u) 
#pragma cuda ainfo kernelid(1) procname(initialize_clnd1) 
#pragma cuda gpurun constant(dnxm1, dnym1, dnzm1) 
	initialize_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[1])/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnym1, dnzm1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noc2gmemtr(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) noshared(Pface) nog2cmemtr(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnym1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnym1, dnzm1, grid_points, u) 
#pragma cuda ainfo kernelid(2) procname(initialize_clnd1) 
#pragma cuda gpurun registerRO(grid_points[1], grid_points[2]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnym1, dnzm1) 
	initialize_clnd1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnxm1, dnzm1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noc2gmemtr(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) noshared(Pface) nog2cmemtr(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnzm1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnzm1, grid_points, u) 
#pragma cuda ainfo kernelid(3) procname(initialize_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[2]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnxm1, dnzm1) 
	initialize_clnd1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock4(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)grid_points[0])/1024.0F)));
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
	dim3 dimGrid4(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(ce, dnxm1, dnym1, grid_points, u) private(eta, i, j, k, m, temp, xi, zeta)
#pragma cuda gpurun noc2gmemtr(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) noshared(Pface) nog2cmemtr(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) nocudafree(buf, ce, cuf, dnxm1, dnym1, grid_points, q, u, ue) 
#pragma cuda gpurun nocudamalloc(ce, dnxm1, dnym1, grid_points, u) 
#pragma cuda ainfo kernelid(4) procname(initialize_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1]) 
#pragma cuda gpurun sharedRW(temp) 
#pragma cuda gpurun constant(dnxm1, dnym1) 
	initialize_clnd1_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__ce, pitch__ce, gpu__grid_points, ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsinit_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     zero the whole left hand side for starters
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	n=_gtid;
#pragma omp for shared(grid_points, lhs) private(i)
	if (n<5)
	{
#pragma loop name lhsinit#0#0 
		for (m=0; m<5; m ++ )
		{
#pragma loop name lhsinit#0#0#0 
			for (i=0; i<grid_points[0]; i ++ )
			{
#pragma loop name lhsinit#0#0#0#0 
				for (j=0; j<grid_points[1]; j ++ )
				{
#pragma loop name lhsinit#0#0#0#0#0 
					for (k=0; k<grid_points[2]; k ++ )
					{
						lhs[i][j][k][0][m][n]=0.0;
						lhs[i][j][k][1][m][n]=0.0;
						lhs[i][j][k][2][m][n]=0.0;
					}
				}
			}
		}
	}
}

__global__ void lhsinit_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])
{
	/*
	   --------------------------------------------------------------------
	   c     next, set all diagonal values to 1. This is overkill, but convenient
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, lhs) private(i)
	if (m<5)
	{
#pragma loop name lhsinit#1#0 
		for (i=0; i<grid_points[0]; i ++ )
		{
#pragma loop name lhsinit#1#0#0 
			for (j=0; j<grid_points[1]; j ++ )
			{
#pragma loop name lhsinit#1#0#0#0 
				for (k=0; k<grid_points[2]; k ++ )
				{
					lhs[i][j][k][1][m][m]=1.0;
				}
			}
		}
	}
}

static void lhsinit(void )
{
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
#pragma omp parallel shared(grid_points, lhs) private(i, j, k, m, n)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, ue) nocudafree(buf, cuf, grid_points, lhs, q, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, ue) 
#pragma cuda gpurun nocudamalloc(grid_points) 
#pragma cuda ainfo kernelid(0) procname(lhsinit) 
	lhsinit_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs));
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
#pragma omp parallel shared(grid_points, lhs) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, ue) nocudafree(buf, cuf, grid_points, lhs, q, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs) 
#pragma cuda ainfo kernelid(1) procname(lhsinit) 
	lhsinit_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsx_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     determine a (labeled f) and n jacobians
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	double u_4;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_0=grid_points[0];
	j=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, tx1, tx2, u) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name lhsx#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name lhsx#0#0#0 
			for (i=0; i<grid_points_0; i ++ )
			{
				u_4=u[i][j][k][0];
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u_4);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				/*
				   --------------------------------------------------------------------
				   c     
				   c-------------------------------------------------------------------
				 */
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=1.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - ((u_3*tmp2)*u_3))+(((const__c2*0.5)*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2));
				fjac[i][j][k][1][1]=((2.0-const__c2)*(u_3/u_4));
				fjac[i][j][k][1][2]=(( - const__c2)*(u_2*tmp1));
				fjac[i][j][k][1][3]=(( - const__c2)*(u_1*tmp1));
				fjac[i][j][k][1][4]=const__c2;
				fjac[i][j][k][2][0]=(( - (u_3*u_2))*tmp2);
				fjac[i][j][k][2][1]=(u_2*tmp1);
				fjac[i][j][k][2][2]=(u_3*tmp1);
				fjac[i][j][k][2][3]=0.0;
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - (u_3*u_1))*tmp2);
				fjac[i][j][k][3][1]=(u_1*tmp1);
				fjac[i][j][k][3][2]=0.0;
				fjac[i][j][k][3][3]=(u_3*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-(const__c1*(u_0*tmp1)))*(u_3*tmp1));
				fjac[i][j][k][4][1]=(((const__c1*u_0)*tmp1)-(((0.5*const__c2)*((((3.0*u_3)*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2));
				fjac[i][j][k][4][2]=((( - const__c2)*(u_2*u_3))*tmp2);
				fjac[i][j][k][4][3]=((( - const__c2)*(u_1*u_3))*tmp2);
				fjac[i][j][k][4][4]=(const__c1*(u_3*tmp1));
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=((const__con43*const__c3c4)*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(const__c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(const__c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - ((const__con43*const__c3c4)-const__c1345))*tmp3)*(u_3*u_3))-(((const__c3c4-const__c1345)*tmp3)*(u_2*u_2)))-(((const__c3c4-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=(((const__c3c4-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=(((const__c3c4-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
			/*
			   --------------------------------------------------------------------
			   c     now jacobians set, so form left hand side in x direction
			   c-------------------------------------------------------------------
			 */
#pragma loop name lhsx#0#0#1 
			for (i=1; i<(grid_points_0-1); i ++ )
			{
				tmp1=(const__dt*const__tx1);
				tmp2=(const__dt*const__tx2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[(i-1)][j][k][0][0])-(tmp1*njac[(i-1)][j][k][0][0]))-(tmp1*const__dx1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[(i-1)][j][k][0][1])-(tmp1*njac[(i-1)][j][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[(i-1)][j][k][0][2])-(tmp1*njac[(i-1)][j][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[(i-1)][j][k][0][3])-(tmp1*njac[(i-1)][j][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[(i-1)][j][k][0][4])-(tmp1*njac[(i-1)][j][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[(i-1)][j][k][1][0])-(tmp1*njac[(i-1)][j][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[(i-1)][j][k][1][1])-(tmp1*njac[(i-1)][j][k][1][1]))-(tmp1*const__dx2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[(i-1)][j][k][1][2])-(tmp1*njac[(i-1)][j][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[(i-1)][j][k][1][3])-(tmp1*njac[(i-1)][j][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[(i-1)][j][k][1][4])-(tmp1*njac[(i-1)][j][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[(i-1)][j][k][2][0])-(tmp1*njac[(i-1)][j][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[(i-1)][j][k][2][1])-(tmp1*njac[(i-1)][j][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[(i-1)][j][k][2][2])-(tmp1*njac[(i-1)][j][k][2][2]))-(tmp1*const__dx3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[(i-1)][j][k][2][3])-(tmp1*njac[(i-1)][j][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[(i-1)][j][k][2][4])-(tmp1*njac[(i-1)][j][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[(i-1)][j][k][3][0])-(tmp1*njac[(i-1)][j][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[(i-1)][j][k][3][1])-(tmp1*njac[(i-1)][j][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[(i-1)][j][k][3][2])-(tmp1*njac[(i-1)][j][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[(i-1)][j][k][3][3])-(tmp1*njac[(i-1)][j][k][3][3]))-(tmp1*const__dx4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[(i-1)][j][k][3][4])-(tmp1*njac[(i-1)][j][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[(i-1)][j][k][4][0])-(tmp1*njac[(i-1)][j][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[(i-1)][j][k][4][1])-(tmp1*njac[(i-1)][j][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[(i-1)][j][k][4][2])-(tmp1*njac[(i-1)][j][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[(i-1)][j][k][4][3])-(tmp1*njac[(i-1)][j][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[(i-1)][j][k][4][4])-(tmp1*njac[(i-1)][j][k][4][4]))-(tmp1*const__dx5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dx1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dx2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dx3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dx4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dx5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[(i+1)][j][k][0][0])-(tmp1*njac[(i+1)][j][k][0][0]))-(tmp1*const__dx1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[(i+1)][j][k][0][1])-(tmp1*njac[(i+1)][j][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[(i+1)][j][k][0][2])-(tmp1*njac[(i+1)][j][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[(i+1)][j][k][0][3])-(tmp1*njac[(i+1)][j][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[(i+1)][j][k][0][4])-(tmp1*njac[(i+1)][j][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[(i+1)][j][k][1][0])-(tmp1*njac[(i+1)][j][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[(i+1)][j][k][1][1])-(tmp1*njac[(i+1)][j][k][1][1]))-(tmp1*const__dx2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[(i+1)][j][k][1][2])-(tmp1*njac[(i+1)][j][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[(i+1)][j][k][1][3])-(tmp1*njac[(i+1)][j][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[(i+1)][j][k][1][4])-(tmp1*njac[(i+1)][j][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[(i+1)][j][k][2][0])-(tmp1*njac[(i+1)][j][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[(i+1)][j][k][2][1])-(tmp1*njac[(i+1)][j][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[(i+1)][j][k][2][2])-(tmp1*njac[(i+1)][j][k][2][2]))-(tmp1*const__dx3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[(i+1)][j][k][2][3])-(tmp1*njac[(i+1)][j][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[(i+1)][j][k][2][4])-(tmp1*njac[(i+1)][j][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[(i+1)][j][k][3][0])-(tmp1*njac[(i+1)][j][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[(i+1)][j][k][3][1])-(tmp1*njac[(i+1)][j][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[(i+1)][j][k][3][2])-(tmp1*njac[(i+1)][j][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[(i+1)][j][k][3][3])-(tmp1*njac[(i+1)][j][k][3][3]))-(tmp1*const__dx4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[(i+1)][j][k][3][4])-(tmp1*njac[(i+1)][j][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[(i+1)][j][k][4][0])-(tmp1*njac[(i+1)][j][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[(i+1)][j][k][4][1])-(tmp1*njac[(i+1)][j][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[(i+1)][j][k][4][2])-(tmp1*njac[(i+1)][j][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[(i+1)][j][k][4][3])-(tmp1*njac[(i+1)][j][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[(i+1)][j][k][4][4])-(tmp1*njac[(i+1)][j][k][4][4]))-(tmp1*const__dx5));
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
	   c     This function computes the left hand side in the xi-direction
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx1, ( & tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5, ( & dx5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4, ( & dx4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3, ( & dx3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2, ( & dx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1, ( & dx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, tx1, tx2, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, con43, cuf, dt, fjac, grid_points, lhs, njac, q, tx2, u, ue) nocudafree(buf, c1, c1345, c2, c3c4, con43, cuf, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, q, tx1, tx2, u, ue) nog2cmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, q, tx1, tx2, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dt, grid_points, lhs, tx2, u) 
#pragma cuda ainfo kernelid(0) procname(lhsx) 
#pragma cuda gpurun registerRO(grid_points[0], u[i][j][k][0], u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, tx1, tx2) 
	lhsx_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	return ;
}

__global__ void lhsx_clnd1_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     determine a (labeled f) and n jacobians
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	double u_4;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_0=grid_points[0];
	j=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, tx1, tx2, u) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name lhsx#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name lhsx#0#0#0 
			for (i=0; i<grid_points_0; i ++ )
			{
				u_4=u[i][j][k][0];
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u_4);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				/*
				   --------------------------------------------------------------------
				   c     
				   c-------------------------------------------------------------------
				 */
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=1.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - ((u_3*tmp2)*u_3))+(((const__c2*0.5)*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2));
				fjac[i][j][k][1][1]=((2.0-const__c2)*(u_3/u_4));
				fjac[i][j][k][1][2]=(( - const__c2)*(u_2*tmp1));
				fjac[i][j][k][1][3]=(( - const__c2)*(u_1*tmp1));
				fjac[i][j][k][1][4]=const__c2;
				fjac[i][j][k][2][0]=(( - (u_3*u_2))*tmp2);
				fjac[i][j][k][2][1]=(u_2*tmp1);
				fjac[i][j][k][2][2]=(u_3*tmp1);
				fjac[i][j][k][2][3]=0.0;
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - (u_3*u_1))*tmp2);
				fjac[i][j][k][3][1]=(u_1*tmp1);
				fjac[i][j][k][3][2]=0.0;
				fjac[i][j][k][3][3]=(u_3*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-(const__c1*(u_0*tmp1)))*(u_3*tmp1));
				fjac[i][j][k][4][1]=(((const__c1*u_0)*tmp1)-(((0.5*const__c2)*((((3.0*u_3)*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2));
				fjac[i][j][k][4][2]=((( - const__c2)*(u_2*u_3))*tmp2);
				fjac[i][j][k][4][3]=((( - const__c2)*(u_1*u_3))*tmp2);
				fjac[i][j][k][4][4]=(const__c1*(u_3*tmp1));
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=((const__con43*const__c3c4)*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(const__c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(const__c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - ((const__con43*const__c3c4)-const__c1345))*tmp3)*(u_3*u_3))-(((const__c3c4-const__c1345)*tmp3)*(u_2*u_2)))-(((const__c3c4-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=(((const__c3c4-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=(((const__c3c4-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
			/*
			   --------------------------------------------------------------------
			   c     now jacobians set, so form left hand side in x direction
			   c-------------------------------------------------------------------
			 */
#pragma loop name lhsx#0#0#1 
			for (i=1; i<(grid_points_0-1); i ++ )
			{
				tmp1=(const__dt*const__tx1);
				tmp2=(const__dt*const__tx2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[(i-1)][j][k][0][0])-(tmp1*njac[(i-1)][j][k][0][0]))-(tmp1*const__dx1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[(i-1)][j][k][0][1])-(tmp1*njac[(i-1)][j][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[(i-1)][j][k][0][2])-(tmp1*njac[(i-1)][j][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[(i-1)][j][k][0][3])-(tmp1*njac[(i-1)][j][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[(i-1)][j][k][0][4])-(tmp1*njac[(i-1)][j][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[(i-1)][j][k][1][0])-(tmp1*njac[(i-1)][j][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[(i-1)][j][k][1][1])-(tmp1*njac[(i-1)][j][k][1][1]))-(tmp1*const__dx2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[(i-1)][j][k][1][2])-(tmp1*njac[(i-1)][j][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[(i-1)][j][k][1][3])-(tmp1*njac[(i-1)][j][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[(i-1)][j][k][1][4])-(tmp1*njac[(i-1)][j][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[(i-1)][j][k][2][0])-(tmp1*njac[(i-1)][j][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[(i-1)][j][k][2][1])-(tmp1*njac[(i-1)][j][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[(i-1)][j][k][2][2])-(tmp1*njac[(i-1)][j][k][2][2]))-(tmp1*const__dx3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[(i-1)][j][k][2][3])-(tmp1*njac[(i-1)][j][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[(i-1)][j][k][2][4])-(tmp1*njac[(i-1)][j][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[(i-1)][j][k][3][0])-(tmp1*njac[(i-1)][j][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[(i-1)][j][k][3][1])-(tmp1*njac[(i-1)][j][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[(i-1)][j][k][3][2])-(tmp1*njac[(i-1)][j][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[(i-1)][j][k][3][3])-(tmp1*njac[(i-1)][j][k][3][3]))-(tmp1*const__dx4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[(i-1)][j][k][3][4])-(tmp1*njac[(i-1)][j][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[(i-1)][j][k][4][0])-(tmp1*njac[(i-1)][j][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[(i-1)][j][k][4][1])-(tmp1*njac[(i-1)][j][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[(i-1)][j][k][4][2])-(tmp1*njac[(i-1)][j][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[(i-1)][j][k][4][3])-(tmp1*njac[(i-1)][j][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[(i-1)][j][k][4][4])-(tmp1*njac[(i-1)][j][k][4][4]))-(tmp1*const__dx5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dx1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dx2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dx3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dx4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dx5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[(i+1)][j][k][0][0])-(tmp1*njac[(i+1)][j][k][0][0]))-(tmp1*const__dx1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[(i+1)][j][k][0][1])-(tmp1*njac[(i+1)][j][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[(i+1)][j][k][0][2])-(tmp1*njac[(i+1)][j][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[(i+1)][j][k][0][3])-(tmp1*njac[(i+1)][j][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[(i+1)][j][k][0][4])-(tmp1*njac[(i+1)][j][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[(i+1)][j][k][1][0])-(tmp1*njac[(i+1)][j][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[(i+1)][j][k][1][1])-(tmp1*njac[(i+1)][j][k][1][1]))-(tmp1*const__dx2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[(i+1)][j][k][1][2])-(tmp1*njac[(i+1)][j][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[(i+1)][j][k][1][3])-(tmp1*njac[(i+1)][j][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[(i+1)][j][k][1][4])-(tmp1*njac[(i+1)][j][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[(i+1)][j][k][2][0])-(tmp1*njac[(i+1)][j][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[(i+1)][j][k][2][1])-(tmp1*njac[(i+1)][j][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[(i+1)][j][k][2][2])-(tmp1*njac[(i+1)][j][k][2][2]))-(tmp1*const__dx3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[(i+1)][j][k][2][3])-(tmp1*njac[(i+1)][j][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[(i+1)][j][k][2][4])-(tmp1*njac[(i+1)][j][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[(i+1)][j][k][3][0])-(tmp1*njac[(i+1)][j][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[(i+1)][j][k][3][1])-(tmp1*njac[(i+1)][j][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[(i+1)][j][k][3][2])-(tmp1*njac[(i+1)][j][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[(i+1)][j][k][3][3])-(tmp1*njac[(i+1)][j][k][3][3]))-(tmp1*const__dx4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[(i+1)][j][k][3][4])-(tmp1*njac[(i+1)][j][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[(i+1)][j][k][4][0])-(tmp1*njac[(i+1)][j][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[(i+1)][j][k][4][1])-(tmp1*njac[(i+1)][j][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[(i+1)][j][k][4][2])-(tmp1*njac[(i+1)][j][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[(i+1)][j][k][4][3])-(tmp1*njac[(i+1)][j][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[(i+1)][j][k][4][4])-(tmp1*njac[(i+1)][j][k][4][4]))-(tmp1*const__dx5));
			}
		}
	}
}

static void lhsx_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     This function computes the left hand side in the xi-direction
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx1, ( & tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5, ( & dx5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4, ( & dx4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3, ( & dx3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2, ( & dx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1, ( & dx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, tx1, tx2, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noc2gmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, q, tx1, tx2, u, ue) noshared(Pface) nog2cmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, q, tx1, tx2, u, ue) nocudafree(buf, c1, c1345, c2, c3c4, con43, cuf, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, q, tx1, tx2, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, fjac, grid_points, lhs, njac, tx1, tx2, u) 
#pragma cuda ainfo kernelid(0) procname(lhsx_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], u[i][j][k][0], u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3c4, con43, dt, dx1, dx2, dx3, dx4, dx5, tx1, tx2) 
	lhsx_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsy_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Compute the indices for storing the tri-diagonal matrix;
	   c     determine a (labeled f) and n jacobians for cell c
	   c-------------------------------------------------------------------
	 */
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsy#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
#pragma loop name lhsy#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=1.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u_3*u_2))*tmp2);
				fjac[i][j][k][1][1]=(u_2*tmp1);
				fjac[i][j][k][1][2]=(u_3*tmp1);
				fjac[i][j][k][1][3]=0.0;
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - ((u_2*u_2)*tmp2))+((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][2][1]=((( - const__c2)*u_3)*tmp1);
				fjac[i][j][k][2][2]=(((2.0-const__c2)*u_2)*tmp1);
				fjac[i][j][k][2][3]=((( - const__c2)*u_1)*tmp1);
				fjac[i][j][k][2][4]=const__c2;
				fjac[i][j][k][3][0]=(( - (u_2*u_1))*tmp2);
				fjac[i][j][k][3][1]=0.0;
				fjac[i][j][k][3][2]=(u_1*tmp1);
				fjac[i][j][k][3][3]=(u_2*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=(((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-((const__c1*u_0)*tmp1))*u_2)*tmp1);
				fjac[i][j][k][4][1]=(((( - const__c2)*u_3)*u_2)*tmp2);
				fjac[i][j][k][4][2]=(((const__c1*u_0)*tmp1)-((0.5*const__c2)*((((u_3*u_3)+((3.0*u_2)*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][4][3]=((( - const__c2)*(u_2*u_1))*tmp2);
				fjac[i][j][k][4][4]=((const__c1*u_2)*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=(const__c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=((const__con43*const__c3c4)*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(const__c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (const__c3c4-const__c1345))*tmp3)*(u_3*u_3))-((((const__con43*const__c3c4)-const__c1345)*tmp3)*(u_2*u_2)))-(((const__c3c4-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=(((const__c3c4-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=(((const__c3c4-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
		}
	}
}

__global__ void lhsy_kernel1(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now joacobians set, so form left hand side in y direction
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, ty1, ty2) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsy#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsy#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(const__dt*const__ty1);
				tmp2=(const__dt*const__ty2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][(j-1)][k][0][0])-(tmp1*njac[i][(j-1)][k][0][0]))-(tmp1*const__dy1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][(j-1)][k][0][1])-(tmp1*njac[i][(j-1)][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][(j-1)][k][0][2])-(tmp1*njac[i][(j-1)][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][(j-1)][k][0][3])-(tmp1*njac[i][(j-1)][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][(j-1)][k][0][4])-(tmp1*njac[i][(j-1)][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][(j-1)][k][1][0])-(tmp1*njac[i][(j-1)][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][(j-1)][k][1][1])-(tmp1*njac[i][(j-1)][k][1][1]))-(tmp1*const__dy2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][(j-1)][k][1][2])-(tmp1*njac[i][(j-1)][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][(j-1)][k][1][3])-(tmp1*njac[i][(j-1)][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][(j-1)][k][1][4])-(tmp1*njac[i][(j-1)][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][(j-1)][k][2][0])-(tmp1*njac[i][(j-1)][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][(j-1)][k][2][1])-(tmp1*njac[i][(j-1)][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][(j-1)][k][2][2])-(tmp1*njac[i][(j-1)][k][2][2]))-(tmp1*const__dy3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][(j-1)][k][2][3])-(tmp1*njac[i][(j-1)][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][(j-1)][k][2][4])-(tmp1*njac[i][(j-1)][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][(j-1)][k][3][0])-(tmp1*njac[i][(j-1)][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][(j-1)][k][3][1])-(tmp1*njac[i][(j-1)][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][(j-1)][k][3][2])-(tmp1*njac[i][(j-1)][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][(j-1)][k][3][3])-(tmp1*njac[i][(j-1)][k][3][3]))-(tmp1*const__dy4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][(j-1)][k][3][4])-(tmp1*njac[i][(j-1)][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][(j-1)][k][4][0])-(tmp1*njac[i][(j-1)][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][(j-1)][k][4][1])-(tmp1*njac[i][(j-1)][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][(j-1)][k][4][2])-(tmp1*njac[i][(j-1)][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][(j-1)][k][4][3])-(tmp1*njac[i][(j-1)][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][(j-1)][k][4][4])-(tmp1*njac[i][(j-1)][k][4][4]))-(tmp1*const__dy5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dy1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dy2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dy3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dy4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dy5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][(j+1)][k][0][0])-(tmp1*njac[i][(j+1)][k][0][0]))-(tmp1*const__dy1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][(j+1)][k][0][1])-(tmp1*njac[i][(j+1)][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][(j+1)][k][0][2])-(tmp1*njac[i][(j+1)][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][(j+1)][k][0][3])-(tmp1*njac[i][(j+1)][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][(j+1)][k][0][4])-(tmp1*njac[i][(j+1)][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][(j+1)][k][1][0])-(tmp1*njac[i][(j+1)][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][(j+1)][k][1][1])-(tmp1*njac[i][(j+1)][k][1][1]))-(tmp1*const__dy2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][(j+1)][k][1][2])-(tmp1*njac[i][(j+1)][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][(j+1)][k][1][3])-(tmp1*njac[i][(j+1)][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][(j+1)][k][1][4])-(tmp1*njac[i][(j+1)][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][(j+1)][k][2][0])-(tmp1*njac[i][(j+1)][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][(j+1)][k][2][1])-(tmp1*njac[i][(j+1)][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][(j+1)][k][2][2])-(tmp1*njac[i][(j+1)][k][2][2]))-(tmp1*const__dy3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][(j+1)][k][2][3])-(tmp1*njac[i][(j+1)][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][(j+1)][k][2][4])-(tmp1*njac[i][(j+1)][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][(j+1)][k][3][0])-(tmp1*njac[i][(j+1)][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][(j+1)][k][3][1])-(tmp1*njac[i][(j+1)][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][(j+1)][k][3][2])-(tmp1*njac[i][(j+1)][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][(j+1)][k][3][3])-(tmp1*njac[i][(j+1)][k][3][3]))-(tmp1*const__dy4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][(j+1)][k][3][4])-(tmp1*njac[i][(j+1)][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][(j+1)][k][4][0])-(tmp1*njac[i][(j+1)][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][(j+1)][k][4][1])-(tmp1*njac[i][(j+1)][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][(j+1)][k][4][2])-(tmp1*njac[i][(j+1)][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][(j+1)][k][4][3])-(tmp1*njac[i][(j+1)][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][(j+1)][k][4][4])-(tmp1*njac[i][(j+1)][k][4][4]))-(tmp1*const__dy5));
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
	   c     This function computes the left hand side for the three y-factors   
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty1, ( & ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5, ( & dy5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4, ( & dy4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3, ( & dy3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2, ( & dy2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1, ( & dy1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nocudafree(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nog2cmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) 
#pragma cuda ainfo kernelid(0) procname(lhsy) 
#pragma cuda gpurun registerRO(u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3c4, con43) 
	lhsy_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, ty1, ty2) private(i, j, k, tmp1, tmp2)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, dt, fjac, grid_points, lhs, njac, q, ty2, ue) nocudafree(buf, cuf, dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, q, ty1, ty2, u, ue) nog2cmemtr(buf, cuf, dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, q, ty1, ty2, ue) 
#pragma cuda gpurun nocudamalloc(dt, fjac, grid_points, lhs, njac, ty2) 
#pragma cuda ainfo kernelid(1) procname(lhsy) 
#pragma cuda gpurun constant(dt, dy1, dy2, dy3, dy4, dy5, ty1, ty2) 
	lhsy_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac));
	return ;
}

__global__ void lhsy_clnd1_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Compute the indices for storing the tri-diagonal matrix;
	   c     determine a (labeled f) and n jacobians for cell c
	   c-------------------------------------------------------------------
	 */
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsy#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
#pragma loop name lhsy#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=1.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u_3*u_2))*tmp2);
				fjac[i][j][k][1][1]=(u_2*tmp1);
				fjac[i][j][k][1][2]=(u_3*tmp1);
				fjac[i][j][k][1][3]=0.0;
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - ((u_2*u_2)*tmp2))+((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][2][1]=((( - const__c2)*u_3)*tmp1);
				fjac[i][j][k][2][2]=(((2.0-const__c2)*u_2)*tmp1);
				fjac[i][j][k][2][3]=((( - const__c2)*u_1)*tmp1);
				fjac[i][j][k][2][4]=const__c2;
				fjac[i][j][k][3][0]=(( - (u_2*u_1))*tmp2);
				fjac[i][j][k][3][1]=0.0;
				fjac[i][j][k][3][2]=(u_1*tmp1);
				fjac[i][j][k][3][3]=(u_2*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=(((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-((const__c1*u_0)*tmp1))*u_2)*tmp1);
				fjac[i][j][k][4][1]=(((( - const__c2)*u_3)*u_2)*tmp2);
				fjac[i][j][k][4][2]=(((const__c1*u_0)*tmp1)-((0.5*const__c2)*((((u_3*u_3)+((3.0*u_2)*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][4][3]=((( - const__c2)*(u_2*u_1))*tmp2);
				fjac[i][j][k][4][4]=((const__c1*u_2)*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=(const__c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=((const__con43*const__c3c4)*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(const__c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (const__c3c4-const__c1345))*tmp3)*(u_3*u_3))-((((const__con43*const__c3c4)-const__c1345)*tmp3)*(u_2*u_2)))-(((const__c3c4-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=(((const__c3c4-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=(((const__c3c4-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
		}
	}
}

__global__ void lhsy_clnd1_kernel1(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now joacobians set, so form left hand side in y direction
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, ty1, ty2) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsy#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsy#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(const__dt*const__ty1);
				tmp2=(const__dt*const__ty2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][(j-1)][k][0][0])-(tmp1*njac[i][(j-1)][k][0][0]))-(tmp1*const__dy1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][(j-1)][k][0][1])-(tmp1*njac[i][(j-1)][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][(j-1)][k][0][2])-(tmp1*njac[i][(j-1)][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][(j-1)][k][0][3])-(tmp1*njac[i][(j-1)][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][(j-1)][k][0][4])-(tmp1*njac[i][(j-1)][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][(j-1)][k][1][0])-(tmp1*njac[i][(j-1)][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][(j-1)][k][1][1])-(tmp1*njac[i][(j-1)][k][1][1]))-(tmp1*const__dy2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][(j-1)][k][1][2])-(tmp1*njac[i][(j-1)][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][(j-1)][k][1][3])-(tmp1*njac[i][(j-1)][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][(j-1)][k][1][4])-(tmp1*njac[i][(j-1)][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][(j-1)][k][2][0])-(tmp1*njac[i][(j-1)][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][(j-1)][k][2][1])-(tmp1*njac[i][(j-1)][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][(j-1)][k][2][2])-(tmp1*njac[i][(j-1)][k][2][2]))-(tmp1*const__dy3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][(j-1)][k][2][3])-(tmp1*njac[i][(j-1)][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][(j-1)][k][2][4])-(tmp1*njac[i][(j-1)][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][(j-1)][k][3][0])-(tmp1*njac[i][(j-1)][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][(j-1)][k][3][1])-(tmp1*njac[i][(j-1)][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][(j-1)][k][3][2])-(tmp1*njac[i][(j-1)][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][(j-1)][k][3][3])-(tmp1*njac[i][(j-1)][k][3][3]))-(tmp1*const__dy4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][(j-1)][k][3][4])-(tmp1*njac[i][(j-1)][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][(j-1)][k][4][0])-(tmp1*njac[i][(j-1)][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][(j-1)][k][4][1])-(tmp1*njac[i][(j-1)][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][(j-1)][k][4][2])-(tmp1*njac[i][(j-1)][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][(j-1)][k][4][3])-(tmp1*njac[i][(j-1)][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][(j-1)][k][4][4])-(tmp1*njac[i][(j-1)][k][4][4]))-(tmp1*const__dy5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dy1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dy2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dy3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dy4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dy5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][(j+1)][k][0][0])-(tmp1*njac[i][(j+1)][k][0][0]))-(tmp1*const__dy1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][(j+1)][k][0][1])-(tmp1*njac[i][(j+1)][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][(j+1)][k][0][2])-(tmp1*njac[i][(j+1)][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][(j+1)][k][0][3])-(tmp1*njac[i][(j+1)][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][(j+1)][k][0][4])-(tmp1*njac[i][(j+1)][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][(j+1)][k][1][0])-(tmp1*njac[i][(j+1)][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][(j+1)][k][1][1])-(tmp1*njac[i][(j+1)][k][1][1]))-(tmp1*const__dy2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][(j+1)][k][1][2])-(tmp1*njac[i][(j+1)][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][(j+1)][k][1][3])-(tmp1*njac[i][(j+1)][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][(j+1)][k][1][4])-(tmp1*njac[i][(j+1)][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][(j+1)][k][2][0])-(tmp1*njac[i][(j+1)][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][(j+1)][k][2][1])-(tmp1*njac[i][(j+1)][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][(j+1)][k][2][2])-(tmp1*njac[i][(j+1)][k][2][2]))-(tmp1*const__dy3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][(j+1)][k][2][3])-(tmp1*njac[i][(j+1)][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][(j+1)][k][2][4])-(tmp1*njac[i][(j+1)][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][(j+1)][k][3][0])-(tmp1*njac[i][(j+1)][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][(j+1)][k][3][1])-(tmp1*njac[i][(j+1)][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][(j+1)][k][3][2])-(tmp1*njac[i][(j+1)][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][(j+1)][k][3][3])-(tmp1*njac[i][(j+1)][k][3][3]))-(tmp1*const__dy4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][(j+1)][k][3][4])-(tmp1*njac[i][(j+1)][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][(j+1)][k][4][0])-(tmp1*njac[i][(j+1)][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][(j+1)][k][4][1])-(tmp1*njac[i][(j+1)][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][(j+1)][k][4][2])-(tmp1*njac[i][(j+1)][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][(j+1)][k][4][3])-(tmp1*njac[i][(j+1)][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][(j+1)][k][4][4])-(tmp1*njac[i][(j+1)][k][4][4]))-(tmp1*const__dy5));
			}
		}
	}
}

static void lhsy_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     This function computes the left hand side for the three y-factors   
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty1, ( & ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5, ( & dy5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4, ( & dy4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3, ( & dy3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2, ( & dy2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1, ( & dy1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noc2gmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) noshared(Pface) nog2cmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nocudafree(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) 
#pragma cuda ainfo kernelid(0) procname(lhsy_clnd1) 
#pragma cuda gpurun registerRO(u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3c4, con43) 
	lhsy_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, ty1, ty2) private(i, j, k, tmp1, tmp2)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, q, ty1, ty2, ue) noshared(Pface) nog2cmemtr(buf, cuf, dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, q, ty1, ty2, ue) nocudafree(buf, cuf, dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, q, ty1, ty2, u, ue) 
#pragma cuda gpurun nocudamalloc(dt, dy1, dy2, dy3, dy4, dy5, fjac, grid_points, lhs, njac, ty1, ty2) 
#pragma cuda ainfo kernelid(1) procname(lhsy_clnd1) 
#pragma cuda gpurun constant(dt, dy1, dy2, dy3, dy4, dy5, ty1, ty2) 
	lhsy_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void lhsz_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Compute the indices for storing the block-diagonal matrix;
	   c     determine c (labeled f) and s jacobians
	   c---------------------------------------------------------------------
	 */
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3, c3c4, c4, con43, fjac, grid_points, njac, u) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsz#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsz#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=1.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u_3*u_1))*tmp2);
				fjac[i][j][k][1][1]=(u_1*tmp1);
				fjac[i][j][k][1][2]=0.0;
				fjac[i][j][k][1][3]=(u_3*tmp1);
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - (u_2*u_1))*tmp2);
				fjac[i][j][k][2][1]=0.0;
				fjac[i][j][k][2][2]=(u_1*tmp1);
				fjac[i][j][k][2][3]=(u_2*tmp1);
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - ((u_1*u_1)*tmp2))+((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][3][1]=((( - const__c2)*u_3)*tmp1);
				fjac[i][j][k][3][2]=((( - const__c2)*u_2)*tmp1);
				fjac[i][j][k][3][3]=(((2.0-const__c2)*u_1)*tmp1);
				fjac[i][j][k][3][4]=const__c2;
				fjac[i][j][k][4][0]=((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-(const__c1*(u_0*tmp1)))*(u_1*tmp1));
				fjac[i][j][k][4][1]=((( - const__c2)*(u_3*u_1))*tmp2);
				fjac[i][j][k][4][2]=((( - const__c2)*(u_2*u_1))*tmp2);
				fjac[i][j][k][4][3]=((const__c1*(u_0*tmp1))-((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+((3.0*u_1)*u_1))*tmp2)));
				fjac[i][j][k][4][4]=((const__c1*u_1)*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=(const__c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(const__c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(((const__con43*const__c3)*const__c4)*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (const__c3c4-const__c1345))*tmp3)*(u_3*u_3))-(((const__c3c4-const__c1345)*tmp3)*(u_2*u_2)))-((((const__con43*const__c3c4)-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=(((const__c3c4-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=(((const__c3c4-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
		}
	}
}

__global__ void lhsz_kernel1(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now jacobians set, so form left hand side in z direction
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, tz1, tz2) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsz#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsz#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(const__dt*const__tz1);
				tmp2=(const__dt*const__tz2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][j][(k-1)][0][0])-(tmp1*njac[i][j][(k-1)][0][0]))-(tmp1*const__dz1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][j][(k-1)][0][1])-(tmp1*njac[i][j][(k-1)][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][j][(k-1)][0][2])-(tmp1*njac[i][j][(k-1)][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][j][(k-1)][0][3])-(tmp1*njac[i][j][(k-1)][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][j][(k-1)][0][4])-(tmp1*njac[i][j][(k-1)][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][j][(k-1)][1][0])-(tmp1*njac[i][j][(k-1)][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][j][(k-1)][1][1])-(tmp1*njac[i][j][(k-1)][1][1]))-(tmp1*const__dz2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][j][(k-1)][1][2])-(tmp1*njac[i][j][(k-1)][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][j][(k-1)][1][3])-(tmp1*njac[i][j][(k-1)][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][j][(k-1)][1][4])-(tmp1*njac[i][j][(k-1)][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][j][(k-1)][2][0])-(tmp1*njac[i][j][(k-1)][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][j][(k-1)][2][1])-(tmp1*njac[i][j][(k-1)][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][j][(k-1)][2][2])-(tmp1*njac[i][j][(k-1)][2][2]))-(tmp1*const__dz3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][j][(k-1)][2][3])-(tmp1*njac[i][j][(k-1)][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][j][(k-1)][2][4])-(tmp1*njac[i][j][(k-1)][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][j][(k-1)][3][0])-(tmp1*njac[i][j][(k-1)][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][j][(k-1)][3][1])-(tmp1*njac[i][j][(k-1)][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][j][(k-1)][3][2])-(tmp1*njac[i][j][(k-1)][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][j][(k-1)][3][3])-(tmp1*njac[i][j][(k-1)][3][3]))-(tmp1*const__dz4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][j][(k-1)][3][4])-(tmp1*njac[i][j][(k-1)][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][j][(k-1)][4][0])-(tmp1*njac[i][j][(k-1)][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][j][(k-1)][4][1])-(tmp1*njac[i][j][(k-1)][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][j][(k-1)][4][2])-(tmp1*njac[i][j][(k-1)][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][j][(k-1)][4][3])-(tmp1*njac[i][j][(k-1)][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][j][(k-1)][4][4])-(tmp1*njac[i][j][(k-1)][4][4]))-(tmp1*const__dz5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dz1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dz2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dz3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dz4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dz5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][j][(k+1)][0][0])-(tmp1*njac[i][j][(k+1)][0][0]))-(tmp1*const__dz1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][j][(k+1)][0][1])-(tmp1*njac[i][j][(k+1)][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][j][(k+1)][0][2])-(tmp1*njac[i][j][(k+1)][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][j][(k+1)][0][3])-(tmp1*njac[i][j][(k+1)][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][j][(k+1)][0][4])-(tmp1*njac[i][j][(k+1)][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][j][(k+1)][1][0])-(tmp1*njac[i][j][(k+1)][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][j][(k+1)][1][1])-(tmp1*njac[i][j][(k+1)][1][1]))-(tmp1*const__dz2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][j][(k+1)][1][2])-(tmp1*njac[i][j][(k+1)][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][j][(k+1)][1][3])-(tmp1*njac[i][j][(k+1)][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][j][(k+1)][1][4])-(tmp1*njac[i][j][(k+1)][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][j][(k+1)][2][0])-(tmp1*njac[i][j][(k+1)][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][j][(k+1)][2][1])-(tmp1*njac[i][j][(k+1)][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][j][(k+1)][2][2])-(tmp1*njac[i][j][(k+1)][2][2]))-(tmp1*const__dz3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][j][(k+1)][2][3])-(tmp1*njac[i][j][(k+1)][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][j][(k+1)][2][4])-(tmp1*njac[i][j][(k+1)][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][j][(k+1)][3][0])-(tmp1*njac[i][j][(k+1)][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][j][(k+1)][3][1])-(tmp1*njac[i][j][(k+1)][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][j][(k+1)][3][2])-(tmp1*njac[i][j][(k+1)][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][j][(k+1)][3][3])-(tmp1*njac[i][j][(k+1)][3][3]))-(tmp1*const__dz4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][j][(k+1)][3][4])-(tmp1*njac[i][j][(k+1)][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][j][(k+1)][4][0])-(tmp1*njac[i][j][(k+1)][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][j][(k+1)][4][1])-(tmp1*njac[i][j][(k+1)][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][j][(k+1)][4][2])-(tmp1*njac[i][j][(k+1)][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][j][(k+1)][4][3])-(tmp1*njac[i][j][(k+1)][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][j][(k+1)][4][4])-(tmp1*njac[i][j][(k+1)][4][4]))-(tmp1*const__dz5));
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
	   c     This function computes the left hand side for the three z-factors   
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz1, ( & tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5, ( & dz5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4, ( & dz4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3, ( & dz3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2, ( & dz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1, ( & dz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c4, ( & c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3, ( & c3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3, c3c4, c4, con43, fjac, grid_points, njac, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c1345, c2, c3c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nocudafree(buf, c1, c1345, c2, c3, c3c4, c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nog2cmemtr(buf, c1, c1345, c2, c3, c3c4, c4, con43, cuf, fjac, grid_points, njac, q, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c1345, c2, c3c4, con43, fjac, grid_points, njac, u) 
#pragma cuda ainfo kernelid(0) procname(lhsz) 
#pragma cuda gpurun registerRO(u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3, c3c4, c4, con43) 
	lhsz_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, tz1, tz2) private(i, j, k, tmp1, tmp2)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, dt, fjac, grid_points, lhs, njac, q, tz2, ue) nocudafree(buf, cuf, dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, q, tz1, tz2, u, ue) nog2cmemtr(buf, cuf, dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, q, tz1, tz2, ue) 
#pragma cuda gpurun nocudamalloc(dt, fjac, grid_points, lhs, njac, tz2) 
#pragma cuda ainfo kernelid(1) procname(lhsz) 
#pragma cuda gpurun constant(dt, dz1, dz2, dz3, dz4, dz5, tz1, tz2) 
	lhsz_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac));
	return ;
}

__global__ void lhsz_clnd1_kernel0(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Compute the indices for storing the block-diagonal matrix;
	   c     determine c (labeled f) and s jacobians
	   c---------------------------------------------------------------------
	 */
	double u_0;
	double u_1;
	double u_2;
	double u_3;
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	double tmp3;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(c1, c1345, c2, c3, c3c4, c4, con43, fjac, grid_points, njac, u) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsz#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsz#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				u_3=u[i][j][k][1];
				u_2=u[i][j][k][2];
				u_1=u[i][j][k][3];
				u_0=u[i][j][k][4];
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=1.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u_3*u_1))*tmp2);
				fjac[i][j][k][1][1]=(u_1*tmp1);
				fjac[i][j][k][1][2]=0.0;
				fjac[i][j][k][1][3]=(u_3*tmp1);
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - (u_2*u_1))*tmp2);
				fjac[i][j][k][2][1]=0.0;
				fjac[i][j][k][2][2]=(u_1*tmp1);
				fjac[i][j][k][2][3]=(u_2*tmp1);
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - ((u_1*u_1)*tmp2))+((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+(u_1*u_1))*tmp2)));
				fjac[i][j][k][3][1]=((( - const__c2)*u_3)*tmp1);
				fjac[i][j][k][3][2]=((( - const__c2)*u_2)*tmp1);
				fjac[i][j][k][3][3]=(((2.0-const__c2)*u_1)*tmp1);
				fjac[i][j][k][3][4]=const__c2;
				fjac[i][j][k][4][0]=((((const__c2*(((u_3*u_3)+(u_2*u_2))+(u_1*u_1)))*tmp2)-(const__c1*(u_0*tmp1)))*(u_1*tmp1));
				fjac[i][j][k][4][1]=((( - const__c2)*(u_3*u_1))*tmp2);
				fjac[i][j][k][4][2]=((( - const__c2)*(u_2*u_1))*tmp2);
				fjac[i][j][k][4][3]=((const__c1*(u_0*tmp1))-((0.5*const__c2)*((((u_3*u_3)+(u_2*u_2))+((3.0*u_1)*u_1))*tmp2)));
				fjac[i][j][k][4][4]=((const__c1*u_1)*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - const__c3c4)*tmp2)*u_3);
				njac[i][j][k][1][1]=(const__c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - const__c3c4)*tmp2)*u_2);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(const__c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=(((( - const__con43)*const__c3c4)*tmp2)*u_1);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(((const__con43*const__c3)*const__c4)*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (const__c3c4-const__c1345))*tmp3)*(u_3*u_3))-(((const__c3c4-const__c1345)*tmp3)*(u_2*u_2)))-((((const__con43*const__c3c4)-const__c1345)*tmp3)*(u_1*u_1)))-((const__c1345*tmp2)*u_0));
				njac[i][j][k][4][1]=(((const__c3c4-const__c1345)*tmp2)*u_3);
				njac[i][j][k][4][2]=(((const__c3c4-const__c1345)*tmp2)*u_2);
				njac[i][j][k][4][3]=((((const__con43*const__c3c4)-const__c1345)*tmp2)*u_1);
				njac[i][j][k][4][4]=(const__c1345*tmp1);
			}
		}
	}
}

__global__ void lhsz_clnd1_kernel1(double fjac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5], int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double njac[(((162/2)*2)+1)][(((162/2)*2)+1)][((162-1)+1)][5][5])
{
	/*
	   --------------------------------------------------------------------
	   c     now jacobians set, so form left hand side in z direction
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int k;
	double tmp1;
	double tmp2;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, tz1, tz2) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name lhsz#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name lhsz#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(const__dt*const__tz1);
				tmp2=(const__dt*const__tz2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][j][(k-1)][0][0])-(tmp1*njac[i][j][(k-1)][0][0]))-(tmp1*const__dz1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][j][(k-1)][0][1])-(tmp1*njac[i][j][(k-1)][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][j][(k-1)][0][2])-(tmp1*njac[i][j][(k-1)][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][j][(k-1)][0][3])-(tmp1*njac[i][j][(k-1)][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][j][(k-1)][0][4])-(tmp1*njac[i][j][(k-1)][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][j][(k-1)][1][0])-(tmp1*njac[i][j][(k-1)][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][j][(k-1)][1][1])-(tmp1*njac[i][j][(k-1)][1][1]))-(tmp1*const__dz2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][j][(k-1)][1][2])-(tmp1*njac[i][j][(k-1)][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][j][(k-1)][1][3])-(tmp1*njac[i][j][(k-1)][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][j][(k-1)][1][4])-(tmp1*njac[i][j][(k-1)][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][j][(k-1)][2][0])-(tmp1*njac[i][j][(k-1)][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][j][(k-1)][2][1])-(tmp1*njac[i][j][(k-1)][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][j][(k-1)][2][2])-(tmp1*njac[i][j][(k-1)][2][2]))-(tmp1*const__dz3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][j][(k-1)][2][3])-(tmp1*njac[i][j][(k-1)][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][j][(k-1)][2][4])-(tmp1*njac[i][j][(k-1)][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][j][(k-1)][3][0])-(tmp1*njac[i][j][(k-1)][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][j][(k-1)][3][1])-(tmp1*njac[i][j][(k-1)][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][j][(k-1)][3][2])-(tmp1*njac[i][j][(k-1)][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][j][(k-1)][3][3])-(tmp1*njac[i][j][(k-1)][3][3]))-(tmp1*const__dz4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][j][(k-1)][3][4])-(tmp1*njac[i][j][(k-1)][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][j][(k-1)][4][0])-(tmp1*njac[i][j][(k-1)][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][j][(k-1)][4][1])-(tmp1*njac[i][j][(k-1)][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][j][(k-1)][4][2])-(tmp1*njac[i][j][(k-1)][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][j][(k-1)][4][3])-(tmp1*njac[i][j][(k-1)][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][j][(k-1)][4][4])-(tmp1*njac[i][j][(k-1)][4][4]))-(tmp1*const__dz5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*const__dz1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*const__dz2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*const__dz3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*const__dz4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*const__dz5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][j][(k+1)][0][0])-(tmp1*njac[i][j][(k+1)][0][0]))-(tmp1*const__dz1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][j][(k+1)][0][1])-(tmp1*njac[i][j][(k+1)][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][j][(k+1)][0][2])-(tmp1*njac[i][j][(k+1)][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][j][(k+1)][0][3])-(tmp1*njac[i][j][(k+1)][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][j][(k+1)][0][4])-(tmp1*njac[i][j][(k+1)][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][j][(k+1)][1][0])-(tmp1*njac[i][j][(k+1)][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][j][(k+1)][1][1])-(tmp1*njac[i][j][(k+1)][1][1]))-(tmp1*const__dz2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][j][(k+1)][1][2])-(tmp1*njac[i][j][(k+1)][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][j][(k+1)][1][3])-(tmp1*njac[i][j][(k+1)][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][j][(k+1)][1][4])-(tmp1*njac[i][j][(k+1)][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][j][(k+1)][2][0])-(tmp1*njac[i][j][(k+1)][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][j][(k+1)][2][1])-(tmp1*njac[i][j][(k+1)][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][j][(k+1)][2][2])-(tmp1*njac[i][j][(k+1)][2][2]))-(tmp1*const__dz3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][j][(k+1)][2][3])-(tmp1*njac[i][j][(k+1)][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][j][(k+1)][2][4])-(tmp1*njac[i][j][(k+1)][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][j][(k+1)][3][0])-(tmp1*njac[i][j][(k+1)][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][j][(k+1)][3][1])-(tmp1*njac[i][j][(k+1)][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][j][(k+1)][3][2])-(tmp1*njac[i][j][(k+1)][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][j][(k+1)][3][3])-(tmp1*njac[i][j][(k+1)][3][3]))-(tmp1*const__dz4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][j][(k+1)][3][4])-(tmp1*njac[i][j][(k+1)][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][j][(k+1)][4][0])-(tmp1*njac[i][j][(k+1)][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][j][(k+1)][4][1])-(tmp1*njac[i][j][(k+1)][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][j][(k+1)][4][2])-(tmp1*njac[i][j][(k+1)][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][j][(k+1)][4][3])-(tmp1*njac[i][j][(k+1)][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][j][(k+1)][4][4])-(tmp1*njac[i][j][(k+1)][4][4]))-(tmp1*const__dz5));
			}
		}
	}
}

static void lhsz_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     This function computes the left hand side for the three z-factors   
	   c-------------------------------------------------------------------
	 */
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz1, ( & tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5, ( & dz5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4, ( & dz4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3, ( & dz3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2, ( & dz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1, ( & dz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c4, ( & c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3c4, ( & c3c4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c3, ( & c3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1345, ( & c1345), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(c1, c1345, c2, c3, c3c4, c4, con43, fjac, grid_points, njac, u) private(i, j, k, tmp1, tmp2, tmp3)
#pragma cuda gpurun noc2gmemtr(buf, c1, c1345, c2, c3, c3c4, c4, con43, cuf, fjac, grid_points, njac, q, u, ue) noshared(Pface) nog2cmemtr(buf, c1, c1345, c2, c3, c3c4, c4, con43, cuf, fjac, grid_points, njac, q, u, ue) nocudafree(buf, c1, c1345, c2, c3, c3c4, c4, con43, cuf, fjac, grid_points, njac, q, u, ue) 
#pragma cuda gpurun nocudamalloc(c1, c1345, c2, c3, c3c4, c4, con43, fjac, grid_points, njac, u) 
#pragma cuda ainfo kernelid(0) procname(lhsz_clnd1) 
#pragma cuda gpurun registerRO(u[i][j][k][1], u[i][j][k][2], u[i][j][k][3], u[i][j][k][4]) 
#pragma cuda gpurun constant(c1, c1345, c2, c3, c3c4, c4, con43) 
	lhsz_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, tz1, tz2) private(i, j, k, tmp1, tmp2)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, q, tz1, tz2, ue) noshared(Pface) nog2cmemtr(buf, cuf, dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, q, tz1, tz2, ue) nocudafree(buf, cuf, dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, q, tz1, tz2, u, ue) 
#pragma cuda gpurun nocudamalloc(dt, dz1, dz2, dz3, dz4, dz5, fjac, grid_points, lhs, njac, tz1, tz2) 
#pragma cuda ainfo kernelid(1) procname(lhsz_clnd1) 
#pragma cuda gpurun constant(dt, dz1, dz2, dz3, dz4, dz5, tz1, tz2) 
	lhsz_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__fjac), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][((162-1)+1)][5][5])gpu__njac));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void compute_rhs_kernel0(double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute the reciprocal of density, and the kinetic energy, 
	   c     and the speed of sound.
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	double u_0;
	double u_1;
	double u_2;
	int i;
	int j;
	int k;
	int m;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=_gtid;
#pragma omp for shared(grid_points, qs, rho_i, square, u, us, vs, ws) private(i, rho_inv) nowait
	if (k<grid_points_0)
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<grid_points_2; j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<grid_points_1; i ++ )
			{
				u_2=u[i][j][k][1];
				u_1=u[i][j][k][2];
				u_0=u[i][j][k][3];
				rho_inv=(1.0/u[i][j][k][0]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_2*rho_inv);
				vs[i][j][k]=(u_1*rho_inv);
				ws[i][j][k]=(u_0*rho_inv);
				square[i][j][k]=((0.5*(((u_2*u_2)+(u_1*u_1))+(u_0*u_0)))*rho_inv);
				qs[i][j][k]=(square[i][j][k]*rho_inv);
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	m=_gtid;
#pragma omp for shared(forcing, grid_points, rhs) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#1#0 
		for (k=0; k<grid_points_0; k ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (j=0; j<grid_points_2; j ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (i=0; i<grid_points_1; i ++ )
				{
					rhs[i][j][k][m]=forcing[i][j][k][m];
				}
			}
		}
	}
}

__global__ void compute_rhs_kernel1(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_2;
	double u_3;
	double u_4;
	double u_6;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, uijk, um1, up1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_6=u[(i+1)][j][k][1];
				u_4=u[(i-1)][j][k][1];
				u_3=u[(i-1)][j][k][4];
				u_2=u[(i+1)][j][k][4];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dx1tx1*((u[(i+1)][j][k][0]-(2.0*u[i][j][k][0]))+u[(i-1)][j][k][0])))-(const__tx2*(u_6-u_4)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dx2tx1*((u_6-(2.0*u[i][j][k][1]))+u_4)))+((const__xxcon2*const__con43)*((up1-(2.0*uijk))+um1)))-(const__tx2*(((u_6*up1)-(u_4*um1))+((((u_2-square_1)-u_3)+square_0)*const__c2))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dx3tx1*((u[(i+1)][j][k][2]-(2.0*u[i][j][k][2]))+u[(i-1)][j][k][2])))+(const__xxcon2*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][2]*up1)-(u[(i-1)][j][k][2]*um1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dx4tx1*((u[(i+1)][j][k][3]-(2.0*u[i][j][k][3]))+u[(i-1)][j][k][3])))+(const__xxcon2*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][3]*up1)-(u[(i-1)][j][k][3]*um1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dx5tx1*((u_2-(2.0*u[i][j][k][4]))+u_3)))+(const__xxcon3*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(const__xxcon4*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(const__xxcon5*(((u_2*rho_i[(i+1)][j][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[(i-1)][j][k]))))-(const__tx2*((((const__c1*u_2)-(const__c2*square_1))*up1)-(((const__c1*u_3)-(const__c2*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel2(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	i=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#3#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	i=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#4#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[(i-1)][j][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#5#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (i=3; i<(grid_points_1-3); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
				}
			}
		}
	}
	i=(grid_points_1-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#6#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))));
			}
		}
	}
	i=(grid_points_1-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#7#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel3(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_1;
	double u_3;
	double u_5;
	double u_7;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, vijk, vm1, vp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_7=u[i][(j+1)][k][4];
				u_5=u[i][(j-1)][k][2];
				u_3=u[i][(j-1)][k][4];
				u_1=u[i][(j+1)][k][2];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dy1ty1*((u[i][(j+1)][k][0]-(2.0*u[i][j][k][0]))+u[i][(j-1)][k][0])))-(const__ty2*(u_1-u_5)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dy2ty1*((u[i][(j+1)][k][1]-(2.0*u[i][j][k][1]))+u[i][(j-1)][k][1])))+(const__yycon2*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][1]*vp1)-(u[i][(j-1)][k][1]*vm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dy3ty1*((u_1-(2.0*u[i][j][k][2]))+u_5)))+((const__yycon2*const__con43)*((vp1-(2.0*vijk))+vm1)))-(const__ty2*(((u_1*vp1)-(u_5*vm1))+((((u_7-square_0)-u_3)+square_1)*const__c2))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dy4ty1*((u[i][(j+1)][k][3]-(2.0*u[i][j][k][3]))+u[i][(j-1)][k][3])))+(const__yycon2*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][3]*vp1)-(u[i][(j-1)][k][3]*vm1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dy5ty1*((u_7-(2.0*u[i][j][k][4]))+u_3)))+(const__yycon3*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(const__yycon4*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(const__yycon5*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[i][(j-1)][k]))))-(const__ty2*((((const__c1*u_7)-(const__c2*square_0))*vp1)-(((const__c1*u_3)-(const__c2*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel4(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	j=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#9#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	j=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#10#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][(j-1)][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#11#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (j=3; j<(grid_points_2-3); j ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
				}
			}
		}
	}
	j=(grid_points_2-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#12#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))));
			}
		}
	}
	j=(grid_points_2-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#13#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel5(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_0;
	double u_2;
	double u_3;
	double u_4;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, wijk, wm1, wp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_4=u[i][j][(k-1)][4];
				u_3=u[i][j][(k-1)][3];
				u_2=u[i][j][(k+1)][4];
				u_0=u[i][j][(k+1)][3];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dz1tz1*((u[i][j][(k+1)][0]-(2.0*u[i][j][k][0]))+u[i][j][(k-1)][0])))-(const__tz2*(u_0-u_3)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dz2tz1*((u[i][j][(k+1)][1]-(2.0*u[i][j][k][1]))+u[i][j][(k-1)][1])))+(const__zzcon2*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][1]*wp1)-(u[i][j][(k-1)][1]*wm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dz3tz1*((u[i][j][(k+1)][2]-(2.0*u[i][j][k][2]))+u[i][j][(k-1)][2])))+(const__zzcon2*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][2]*wp1)-(u[i][j][(k-1)][2]*wm1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dz4tz1*((u_0-(2.0*u[i][j][k][3]))+u_3)))+((const__zzcon2*const__con43)*((wp1-(2.0*wijk))+wm1)))-(const__tz2*(((u_0*wp1)-(u_3*wm1))+((((u_2-square_0)-u_4)+square_1)*const__c2))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dz5tz1*((u_2-(2.0*u[i][j][k][4]))+u_4)))+(const__zzcon3*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(const__zzcon4*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(const__zzcon5*(((u_2*rho_i[i][j][(k+1)])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_4*rho_i[i][j][(k-1)]))))-(const__tz2*((((const__c1*u_2)-(const__c2*square_0))*wp1)-(((const__c1*u_4)-(const__c2*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_kernel6(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#15#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	k=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#16#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][j][(k-1)][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#17#0 
		for (k=3; k<(grid_points_0-3); k ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
				}
			}
		}
	}
	k=(grid_points_0-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#18#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))));
			}
		}
	}
	k=(grid_points_0-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#19#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_kernel7(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(dt, grid_points, rhs) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#20#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]*const__dt);
				}
			}
		}
	}
}

static void compute_rhs(void )
{
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon5, ( & zzcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon4, ( & zzcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon3, ( & zzcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon2, ( & zzcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5tz1, ( & dz5tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4tz1, ( & dz4tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3tz1, ( & dz3tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2tz1, ( & dz2tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1tz1, ( & dz1tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon5, ( & yycon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon4, ( & yycon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon3, ( & yycon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon2, ( & yycon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5ty1, ( & dy5ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4ty1, ( & dy4ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3ty1, ( & dy3ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2ty1, ( & dy2ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1ty1, ( & dy1ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon5, ( & xxcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon4, ( & xxcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon3, ( & xxcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon2, ( & xxcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5tx1, ( & dx5tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4tx1, ( & dx4tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3tx1, ( & dx3tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2tx1, ( & dx2tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1tx1, ( & dx1tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuNumBlocks=((int)ceil((((float)MAX(5, grid_points[2]))/1024.0F)));
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
#pragma omp parallel shared(forcing, grid_points, qs, rho_i, rhs, square, u, us, vs, ws) private(i, j, k, m, rho_inv)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) nocudafree(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) nog2cmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(forcing, grid_points, u) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2], u[i][j][k][1], u[i][j][k][2], u[i][j][k][3]) 
	compute_rhs_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) nocudafree(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) nog2cmemtr(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs) 
#pragma cuda gpurun registerRO(square[(i+1)][j][k], square[(i-1)][j][k], u[(i+1)][j][k][1], u[(i+1)][j][k][4], u[(i-1)][j][k][1], u[(i-1)][j][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, tx2, xxcon2, xxcon3, xxcon4, xxcon5) 
	compute_rhs_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) nocudafree(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) nog2cmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs) 
#pragma cuda gpurun registerRO(square[i][(j+1)][k], square[i][(j-1)][k], u[i][(j+1)][k][2], u[i][(j+1)][k][4], u[i][(j-1)][k][2], u[i][(j-1)][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, ty2, yycon2, yycon3, yycon4, yycon5) 
	compute_rhs_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock5(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nocudafree(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nog2cmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs) 
#pragma cuda gpurun registerRO(square[i][j][(k+1)], square[i][j][(k-1)], u[i][j][(k+1)][3], u[i][j][(k+1)][4], u[i][j][(k-1)][3], u[i][j][(k-1)][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, tz2, zzcon2, zzcon3, zzcon4, zzcon5) 
	compute_rhs_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
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
#pragma omp parallel shared(dt, grid_points, rhs) private(i, j, k, m)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, q, rhs, ue) nocudafree(buf, cuf, dt, grid_points, q, rhs, u, ue) nog2cmemtr(buf, cuf, dt, grid_points, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, rhs) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs) 
#pragma cuda gpurun constant(dt) 
	compute_rhs_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

__global__ void compute_rhs_clnd2_kernel0(double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute the reciprocal of density, and the kinetic energy, 
	   c     and the speed of sound.
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	double u_0;
	double u_1;
	double u_2;
	int i;
	int j;
	int k;
	int m;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=_gtid;
#pragma omp for shared(grid_points, qs, rho_i, square, u, us, vs, ws) private(i, rho_inv) nowait
	if (k<grid_points_0)
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<grid_points_2; j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<grid_points_1; i ++ )
			{
				u_2=u[i][j][k][1];
				u_1=u[i][j][k][2];
				u_0=u[i][j][k][3];
				rho_inv=(1.0/u[i][j][k][0]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_2*rho_inv);
				vs[i][j][k]=(u_1*rho_inv);
				ws[i][j][k]=(u_0*rho_inv);
				square[i][j][k]=((0.5*(((u_2*u_2)+(u_1*u_1))+(u_0*u_0)))*rho_inv);
				qs[i][j][k]=(square[i][j][k]*rho_inv);
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	m=_gtid;
#pragma omp for shared(forcing, grid_points, rhs) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#1#0 
		for (k=0; k<grid_points_0; k ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (j=0; j<grid_points_2; j ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (i=0; i<grid_points_1; i ++ )
				{
					rhs[i][j][k][m]=forcing[i][j][k][m];
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel1(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_2;
	double u_3;
	double u_4;
	double u_6;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, uijk, um1, up1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_6=u[(i+1)][j][k][1];
				u_4=u[(i-1)][j][k][1];
				u_3=u[(i-1)][j][k][4];
				u_2=u[(i+1)][j][k][4];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dx1tx1*((u[(i+1)][j][k][0]-(2.0*u[i][j][k][0]))+u[(i-1)][j][k][0])))-(const__tx2*(u_6-u_4)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dx2tx1*((u_6-(2.0*u[i][j][k][1]))+u_4)))+((const__xxcon2*const__con43)*((up1-(2.0*uijk))+um1)))-(const__tx2*(((u_6*up1)-(u_4*um1))+((((u_2-square_1)-u_3)+square_0)*const__c2))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dx3tx1*((u[(i+1)][j][k][2]-(2.0*u[i][j][k][2]))+u[(i-1)][j][k][2])))+(const__xxcon2*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][2]*up1)-(u[(i-1)][j][k][2]*um1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dx4tx1*((u[(i+1)][j][k][3]-(2.0*u[i][j][k][3]))+u[(i-1)][j][k][3])))+(const__xxcon2*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][3]*up1)-(u[(i-1)][j][k][3]*um1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dx5tx1*((u_2-(2.0*u[i][j][k][4]))+u_3)))+(const__xxcon3*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(const__xxcon4*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(const__xxcon5*(((u_2*rho_i[(i+1)][j][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[(i-1)][j][k]))))-(const__tx2*((((const__c1*u_2)-(const__c2*square_1))*up1)-(((const__c1*u_3)-(const__c2*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel2(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	i=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#3#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	i=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#4#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[(i-1)][j][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#5#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (i=3; i<(grid_points_1-3); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
				}
			}
		}
	}
	i=(grid_points_1-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#6#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))));
			}
		}
	}
	i=(grid_points_1-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#7#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel3(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_1;
	double u_3;
	double u_5;
	double u_7;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, vijk, vm1, vp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_7=u[i][(j+1)][k][4];
				u_5=u[i][(j-1)][k][2];
				u_3=u[i][(j-1)][k][4];
				u_1=u[i][(j+1)][k][2];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dy1ty1*((u[i][(j+1)][k][0]-(2.0*u[i][j][k][0]))+u[i][(j-1)][k][0])))-(const__ty2*(u_1-u_5)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dy2ty1*((u[i][(j+1)][k][1]-(2.0*u[i][j][k][1]))+u[i][(j-1)][k][1])))+(const__yycon2*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][1]*vp1)-(u[i][(j-1)][k][1]*vm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dy3ty1*((u_1-(2.0*u[i][j][k][2]))+u_5)))+((const__yycon2*const__con43)*((vp1-(2.0*vijk))+vm1)))-(const__ty2*(((u_1*vp1)-(u_5*vm1))+((((u_7-square_0)-u_3)+square_1)*const__c2))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dy4ty1*((u[i][(j+1)][k][3]-(2.0*u[i][j][k][3]))+u[i][(j-1)][k][3])))+(const__yycon2*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][3]*vp1)-(u[i][(j-1)][k][3]*vm1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dy5ty1*((u_7-(2.0*u[i][j][k][4]))+u_3)))+(const__yycon3*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(const__yycon4*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(const__yycon5*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[i][(j-1)][k]))))-(const__ty2*((((const__c1*u_7)-(const__c2*square_0))*vp1)-(((const__c1*u_3)-(const__c2*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel4(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	j=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#9#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	j=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#10#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][(j-1)][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#11#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (j=3; j<(grid_points_2-3); j ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
				}
			}
		}
	}
	j=(grid_points_2-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#12#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))));
			}
		}
	}
	j=(grid_points_2-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#13#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel5(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_0;
	double u_2;
	double u_3;
	double u_4;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, wijk, wm1, wp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_4=u[i][j][(k-1)][4];
				u_3=u[i][j][(k-1)][3];
				u_2=u[i][j][(k+1)][4];
				u_0=u[i][j][(k+1)][3];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dz1tz1*((u[i][j][(k+1)][0]-(2.0*u[i][j][k][0]))+u[i][j][(k-1)][0])))-(const__tz2*(u_0-u_3)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dz2tz1*((u[i][j][(k+1)][1]-(2.0*u[i][j][k][1]))+u[i][j][(k-1)][1])))+(const__zzcon2*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][1]*wp1)-(u[i][j][(k-1)][1]*wm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dz3tz1*((u[i][j][(k+1)][2]-(2.0*u[i][j][k][2]))+u[i][j][(k-1)][2])))+(const__zzcon2*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][2]*wp1)-(u[i][j][(k-1)][2]*wm1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dz4tz1*((u_0-(2.0*u[i][j][k][3]))+u_3)))+((const__zzcon2*const__con43)*((wp1-(2.0*wijk))+wm1)))-(const__tz2*(((u_0*wp1)-(u_3*wm1))+((((u_2-square_0)-u_4)+square_1)*const__c2))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dz5tz1*((u_2-(2.0*u[i][j][k][4]))+u_4)))+(const__zzcon3*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(const__zzcon4*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(const__zzcon5*(((u_2*rho_i[i][j][(k+1)])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_4*rho_i[i][j][(k-1)]))))-(const__tz2*((((const__c1*u_2)-(const__c2*square_0))*wp1)-(((const__c1*u_4)-(const__c2*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel6(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#15#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	k=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#16#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][j][(k-1)][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#17#0 
		for (k=3; k<(grid_points_0-3); k ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
				}
			}
		}
	}
	k=(grid_points_0-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#18#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))));
			}
		}
	}
	k=(grid_points_0-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#19#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd2_kernel7(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(dt, grid_points, rhs) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#20#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]*const__dt);
				}
			}
		}
	}
}

static void compute_rhs_clnd2(void )
{
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon5, ( & zzcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon4, ( & zzcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon3, ( & zzcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon2, ( & zzcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5tz1, ( & dz5tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4tz1, ( & dz4tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3tz1, ( & dz3tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2tz1, ( & dz2tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1tz1, ( & dz1tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon5, ( & yycon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon4, ( & yycon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon3, ( & yycon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon2, ( & yycon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5ty1, ( & dy5ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4ty1, ( & dy4ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3ty1, ( & dy3ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2ty1, ( & dy2ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1ty1, ( & dy1ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon5, ( & xxcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon4, ( & xxcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon3, ( & xxcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon2, ( & xxcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5tx1, ( & dx5tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4tx1, ( & dx4tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3tx1, ( & dx3tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2tx1, ( & dx2tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1tx1, ( & dx1tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuNumBlocks=((int)ceil((((float)MAX(5, grid_points[2]))/1024.0F)));
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
#pragma omp parallel shared(forcing, grid_points, qs, rho_i, rhs, square, u, us, vs, ws) private(i, j, k, m, rho_inv)
#pragma cuda gpurun noc2gmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) noshared(Pface) nog2cmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) nocudafree(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(forcing, grid_points, qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2], u[i][j][k][1], u[i][j][k][2], u[i][j][k][3]) 
	compute_rhs_clnd2_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) nocudafree(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(square[(i+1)][j][k], square[(i-1)][j][k], u[(i+1)][j][k][1], u[(i+1)][j][k][4], u[(i-1)][j][k][1], u[(i-1)][j][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, tx2, xxcon2, xxcon3, xxcon4, xxcon5) 
	compute_rhs_clnd2_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd2_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) nocudafree(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(square[i][(j+1)][k], square[i][(j-1)][k], u[i][(j+1)][k][2], u[i][(j+1)][k][4], u[i][(j-1)][k][2], u[i][(j-1)][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, ty2, yycon2, yycon3, yycon4, yycon5) 
	compute_rhs_clnd2_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd2_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock5(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nocudafree(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(square[i][j][(k+1)], square[i][j][(k-1)], u[i][j][(k+1)][3], u[i][j][(k+1)][4], u[i][j][(k-1)][3], u[i][j][(k-1)][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, tz2, zzcon2, zzcon3, zzcon4, zzcon5) 
	compute_rhs_clnd2_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs_clnd2) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd2_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
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
#pragma omp parallel shared(dt, grid_points, rhs) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dt, grid_points, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, dt, grid_points, q, ue) nocudafree(buf, cuf, dt, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dt, grid_points, rhs) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs_clnd2) 
#pragma cuda gpurun constant(dt) 
	compute_rhs_clnd2_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	gpuBytes=(((((((162/2)*2)+1)*(((162/2)*2)+1))*(((162/2)*2)+1))*5)*sizeof (double));
	CUDA_SAFE_CALL(cudaMemcpy(rhs, gpu__rhs, gpuBytes, cudaMemcpyDeviceToHost));
	return ;
}

__global__ void compute_rhs_clnd1_kernel0(double forcing[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)], int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute the reciprocal of density, and the kinetic energy, 
	   c     and the speed of sound.
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	double u_0;
	double u_1;
	double u_2;
	int i;
	int j;
	int k;
	int m;
	double rho_inv;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=_gtid;
#pragma omp for shared(grid_points, qs, rho_i, square, u, us, vs, ws) private(i, rho_inv) nowait
	if (k<grid_points_0)
	{
#pragma loop name compute_rhs#0#0 
		for (j=0; j<grid_points_2; j ++ )
		{
#pragma loop name compute_rhs#0#0#0 
			for (i=0; i<grid_points_1; i ++ )
			{
				u_2=u[i][j][k][1];
				u_1=u[i][j][k][2];
				u_0=u[i][j][k][3];
				rho_inv=(1.0/u[i][j][k][0]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u_2*rho_inv);
				vs[i][j][k]=(u_1*rho_inv);
				ws[i][j][k]=(u_0*rho_inv);
				square[i][j][k]=((0.5*(((u_2*u_2)+(u_1*u_1))+(u_0*u_0)))*rho_inv);
				qs[i][j][k]=(square[i][j][k]*rho_inv);
			}
		}
	}
	/*
	   --------------------------------------------------------------------
	   c copy the exact forcing term to the right hand side;  because 
	   c this forcing term is known, we can store it on the whole grid
	   c including the boundary                   
	   c-------------------------------------------------------------------
	 */
	m=_gtid;
#pragma omp for shared(forcing, grid_points, rhs) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#1#0 
		for (k=0; k<grid_points_0; k ++ )
		{
#pragma loop name compute_rhs#1#0#0 
			for (j=0; j<grid_points_2; j ++ )
			{
#pragma loop name compute_rhs#1#0#0#0 
				for (i=0; i<grid_points_1; i ++ )
				{
					rhs[i][j][k][m]=forcing[i][j][k][m];
				}
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel1(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute xi-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_2;
	double u_3;
	double u_4;
	double u_6;
	int i;
	int j;
	int k;
	double uijk;
	double um1;
	double up1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, uijk, um1, up1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#2#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_6=u[(i+1)][j][k][1];
				u_4=u[(i-1)][j][k][1];
				u_3=u[(i-1)][j][k][4];
				u_2=u[(i+1)][j][k][4];
				square_1=square[(i+1)][j][k];
				square_0=square[(i-1)][j][k];
				uijk=us[i][j][k];
				up1=us[(i+1)][j][k];
				um1=us[(i-1)][j][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dx1tx1*((u[(i+1)][j][k][0]-(2.0*u[i][j][k][0]))+u[(i-1)][j][k][0])))-(const__tx2*(u_6-u_4)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dx2tx1*((u_6-(2.0*u[i][j][k][1]))+u_4)))+((const__xxcon2*const__con43)*((up1-(2.0*uijk))+um1)))-(const__tx2*(((u_6*up1)-(u_4*um1))+((((u_2-square_1)-u_3)+square_0)*const__c2))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dx3tx1*((u[(i+1)][j][k][2]-(2.0*u[i][j][k][2]))+u[(i-1)][j][k][2])))+(const__xxcon2*((vs[(i+1)][j][k]-(2.0*vs[i][j][k]))+vs[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][2]*up1)-(u[(i-1)][j][k][2]*um1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dx4tx1*((u[(i+1)][j][k][3]-(2.0*u[i][j][k][3]))+u[(i-1)][j][k][3])))+(const__xxcon2*((ws[(i+1)][j][k]-(2.0*ws[i][j][k]))+ws[(i-1)][j][k])))-(const__tx2*((u[(i+1)][j][k][3]*up1)-(u[(i-1)][j][k][3]*um1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dx5tx1*((u_2-(2.0*u[i][j][k][4]))+u_3)))+(const__xxcon3*((qs[(i+1)][j][k]-(2.0*qs[i][j][k]))+qs[(i-1)][j][k])))+(const__xxcon4*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(const__xxcon5*(((u_2*rho_i[(i+1)][j][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[(i-1)][j][k]))))-(const__tx2*((((const__c1*u_2)-(const__c2*square_1))*up1)-(((const__c1*u_3)-(const__c2*square_0))*um1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel2(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order xi-direction dissipation               
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	i=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#3#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#3#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	i=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#4#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#4#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[(i-1)][j][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#5#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#5#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#5#0#0#0 
				for (i=3; i<(grid_points_1-3); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))+u[(i+2)][j][k][m])));
				}
			}
		}
	}
	i=(grid_points_1-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#6#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#6#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[(i+1)][j][k][m]))));
			}
		}
	}
	i=(grid_points_1-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#7#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#7#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[(i-2)][j][k][m]-(4.0*u[(i-1)][j][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel3(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute eta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_1;
	double u_3;
	double u_5;
	double u_7;
	int i;
	int j;
	int k;
	double vijk;
	double vm1;
	double vp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, vijk, vm1, vp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#8#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#8#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_7=u[i][(j+1)][k][4];
				u_5=u[i][(j-1)][k][2];
				u_3=u[i][(j-1)][k][4];
				u_1=u[i][(j+1)][k][2];
				square_1=square[i][(j-1)][k];
				square_0=square[i][(j+1)][k];
				vijk=vs[i][j][k];
				vp1=vs[i][(j+1)][k];
				vm1=vs[i][(j-1)][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dy1ty1*((u[i][(j+1)][k][0]-(2.0*u[i][j][k][0]))+u[i][(j-1)][k][0])))-(const__ty2*(u_1-u_5)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dy2ty1*((u[i][(j+1)][k][1]-(2.0*u[i][j][k][1]))+u[i][(j-1)][k][1])))+(const__yycon2*((us[i][(j+1)][k]-(2.0*us[i][j][k]))+us[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][1]*vp1)-(u[i][(j-1)][k][1]*vm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dy3ty1*((u_1-(2.0*u[i][j][k][2]))+u_5)))+((const__yycon2*const__con43)*((vp1-(2.0*vijk))+vm1)))-(const__ty2*(((u_1*vp1)-(u_5*vm1))+((((u_7-square_0)-u_3)+square_1)*const__c2))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dy4ty1*((u[i][(j+1)][k][3]-(2.0*u[i][j][k][3]))+u[i][(j-1)][k][3])))+(const__yycon2*((ws[i][(j+1)][k]-(2.0*ws[i][j][k]))+ws[i][(j-1)][k])))-(const__ty2*((u[i][(j+1)][k][3]*vp1)-(u[i][(j-1)][k][3]*vm1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dy5ty1*((u_7-(2.0*u[i][j][k][4]))+u_3)))+(const__yycon3*((qs[i][(j+1)][k]-(2.0*qs[i][j][k]))+qs[i][(j-1)][k])))+(const__yycon4*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(const__yycon5*(((u_7*rho_i[i][(j+1)][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_3*rho_i[i][(j-1)][k]))))-(const__ty2*((((const__c1*u_7)-(const__c2*square_0))*vp1)-(((const__c1*u_3)-(const__c2*square_1))*vm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel4(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order eta-direction dissipation         
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	j=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#9#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#9#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	j=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#10#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#10#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][(j-1)][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#11#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#11#0#0 
			for (j=3; j<(grid_points_2-3); j ++ )
			{
#pragma loop name compute_rhs#11#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))+u[i][(j+2)][k][m])));
				}
			}
		}
	}
	j=(grid_points_2-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#12#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#12#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][(j+1)][k][m]))));
			}
		}
	}
	j=(grid_points_2-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#13#0 
		for (k=1; k<(grid_points_0-1); k ++ )
		{
#pragma loop name compute_rhs#13#0#0 
			for (i=1; i<(grid_points_1-1); i ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][(j-2)][k][m]-(4.0*u[i][(j-1)][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel5(int * grid_points, double qs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rho_i[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double square[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5], double us[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double vs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)], double ws[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)])
{
	/*
	   --------------------------------------------------------------------
	   c     compute zeta-direction fluxes 
	   c-------------------------------------------------------------------
	 */
	double square_0;
	double square_1;
	double u_0;
	double u_2;
	double u_3;
	double u_4;
	int i;
	int j;
	int k;
	double wijk;
	double wm1;
	double wp1;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	k=(_gtid+1);
#pragma omp for shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, wijk, wm1, wp1)
	if (k<(grid_points[2]-1))
	{
#pragma loop name compute_rhs#14#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name compute_rhs#14#0#0 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				u_4=u[i][j][(k-1)][4];
				u_3=u[i][j][(k-1)][3];
				u_2=u[i][j][(k+1)][4];
				u_0=u[i][j][(k+1)][3];
				square_1=square[i][j][(k-1)];
				square_0=square[i][j][(k+1)];
				wijk=ws[i][j][k];
				wp1=ws[i][j][(k+1)];
				wm1=ws[i][j][(k-1)];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(const__dz1tz1*((u[i][j][(k+1)][0]-(2.0*u[i][j][k][0]))+u[i][j][(k-1)][0])))-(const__tz2*(u_0-u_3)));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(const__dz2tz1*((u[i][j][(k+1)][1]-(2.0*u[i][j][k][1]))+u[i][j][(k-1)][1])))+(const__zzcon2*((us[i][j][(k+1)]-(2.0*us[i][j][k]))+us[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][1]*wp1)-(u[i][j][(k-1)][1]*wm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(const__dz3tz1*((u[i][j][(k+1)][2]-(2.0*u[i][j][k][2]))+u[i][j][(k-1)][2])))+(const__zzcon2*((vs[i][j][(k+1)]-(2.0*vs[i][j][k]))+vs[i][j][(k-1)])))-(const__tz2*((u[i][j][(k+1)][2]*wp1)-(u[i][j][(k-1)][2]*wm1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(const__dz4tz1*((u_0-(2.0*u[i][j][k][3]))+u_3)))+((const__zzcon2*const__con43)*((wp1-(2.0*wijk))+wm1)))-(const__tz2*(((u_0*wp1)-(u_3*wm1))+((((u_2-square_0)-u_4)+square_1)*const__c2))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(const__dz5tz1*((u_2-(2.0*u[i][j][k][4]))+u_4)))+(const__zzcon3*((qs[i][j][(k+1)]-(2.0*qs[i][j][k]))+qs[i][j][(k-1)])))+(const__zzcon4*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(const__zzcon5*(((u_2*rho_i[i][j][(k+1)])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u_4*rho_i[i][j][(k-1)]))))-(const__tz2*((((const__c1*u_2)-(const__c2*square_0))*wp1)-(((const__c1*u_4)-(const__c2*square_1))*wm1))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel6(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5], double u[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     add fourth order zeta-direction dissipation                
	   c-------------------------------------------------------------------
	 */
	int grid_points_0;
	int grid_points_1;
	int grid_points_2;
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	grid_points_2=grid_points[1];
	grid_points_1=grid_points[0];
	grid_points_0=grid_points[2];
	k=1;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#15#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#15#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	k=2;
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#16#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#16#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((( - 4.0)*u[i][j][(k-1)][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
			}
		}
	}
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#17#0 
		for (k=3; k<(grid_points_0-3); k ++ )
		{
#pragma loop name compute_rhs#17#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
#pragma loop name compute_rhs#17#0#0#0 
				for (i=1; i<(grid_points_1-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))+u[i][j][(k+2)][m])));
				}
			}
		}
	}
	k=(grid_points_0-3);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i) nowait
	if (m<5)
	{
#pragma loop name compute_rhs#18#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#18#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*(((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][(k+1)][m]))));
			}
		}
	}
	k=(grid_points_0-2);
	m=_gtid;
#pragma omp for shared(dssp, grid_points, rhs, u) private(i)
	if (m<5)
	{
#pragma loop name compute_rhs#19#0 
		for (i=1; i<(grid_points_1-1); i ++ )
		{
#pragma loop name compute_rhs#19#0#0 
			for (j=1; j<(grid_points_2-1); j ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(const__dssp*((u[i][j][(k-2)][m]-(4.0*u[i][j][(k-1)][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
}

__global__ void compute_rhs_clnd1_kernel7(int * grid_points, double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(dt, grid_points, rhs) private(j)
	if (m<5)
	{
#pragma loop name compute_rhs#20#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name compute_rhs#20#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
#pragma loop name compute_rhs#20#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]*const__dt);
				}
			}
		}
	}
}

static void compute_rhs_clnd1(void )
{
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dt, ( & dt), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon5, ( & zzcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon4, ( & zzcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon3, ( & zzcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__zzcon2, ( & zzcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tz2, ( & tz2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz5tz1, ( & dz5tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz4tz1, ( & dz4tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz3tz1, ( & dz3tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz2tz1, ( & dz2tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dz1tz1, ( & dz1tz1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon5, ( & yycon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon4, ( & yycon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon3, ( & yycon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__yycon2, ( & yycon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__ty2, ( & ty2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy5ty1, ( & dy5ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy4ty1, ( & dy4ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy3ty1, ( & dy3ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy2ty1, ( & dy2ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dy1ty1, ( & dy1ty1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dssp, ( & dssp), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon5, ( & xxcon5), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon4, ( & xxcon4), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon3, ( & xxcon3), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__xxcon2, ( & xxcon2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__tx2, ( & tx2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx5tx1, ( & dx5tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx4tx1, ( & dx4tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx3tx1, ( & dx3tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx2tx1, ( & dx2tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__dx1tx1, ( & dx1tx1), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__con43, ( & con43), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c2, ( & c2), gpuBytes));
	gpuBytes=sizeof (double);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(const__c1, ( & c1), gpuBytes));
	gpuNumBlocks=((int)ceil((((float)MAX(5, grid_points[2]))/1024.0F)));
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
#pragma omp parallel shared(forcing, grid_points, qs, rho_i, rhs, square, u, us, vs, ws) private(i, j, k, m, rho_inv)
#pragma cuda gpurun noc2gmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) noshared(Pface) nog2cmemtr(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) nocudafree(buf, cuf, forcing, grid_points, q, qs, rho_i, rhs, square, u, ue, us, vs, ws) 
#pragma cuda gpurun nocudamalloc(forcing, grid_points, qs, rho_i, rhs, square, u, us, vs, ws) 
#pragma cuda ainfo kernelid(0) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2], u[i][j][k][1], u[i][j][k][2], u[i][j][k][3]) 
	compute_rhs_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][(5+1)])gpu__forcing), gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) private(i, j, k, uijk, um1, up1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) nocudafree(buf, c1, c2, con43, cuf, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, q, qs, rho_i, rhs, square, tx2, u, ue, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, grid_points, qs, rho_i, rhs, square, tx2, u, us, vs, ws, xxcon2, xxcon3, xxcon4, xxcon5) 
#pragma cuda ainfo kernelid(1) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(square[(i+1)][j][k], square[(i-1)][j][k], u[(i+1)][j][k][1], u[(i+1)][j][k][4], u[(i-1)][j][k][1], u[(i-1)][j][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, tx2, xxcon2, xxcon3, xxcon4, xxcon5) 
	compute_rhs_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(2) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock3(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid3(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) private(i, j, k, vijk, vm1, vp1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) nocudafree(buf, c1, c2, con43, cuf, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, q, qs, rho_i, rhs, square, ty2, u, ue, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, grid_points, qs, rho_i, rhs, square, ty2, u, us, vs, ws, yycon2, yycon3, yycon4, yycon5) 
#pragma cuda ainfo kernelid(3) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(square[i][(j+1)][k], square[i][(j-1)][k], u[i][(j+1)][k][2], u[i][(j+1)][k][4], u[i][(j-1)][k][2], u[i][(j-1)][k][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, ty2, yycon2, yycon3, yycon4, yycon5) 
	compute_rhs_clnd1_kernel3<<<dimGrid3, dimBlock3, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(4) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd1_kernel4<<<dimGrid4, dimBlock4, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
	dim3 dimBlock5(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[2]))/1024.0F)));
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
	dim3 dimGrid5(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) private(i, j, k, wijk, wm1, wp1)
#pragma cuda gpurun noc2gmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) noshared(Pface) nog2cmemtr(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) nocudafree(buf, c1, c2, con43, cuf, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, q, qs, rho_i, rhs, square, tz2, u, ue, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda gpurun nocudamalloc(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, grid_points, qs, rho_i, rhs, square, tz2, u, us, vs, ws, zzcon2, zzcon3, zzcon4, zzcon5) 
#pragma cuda ainfo kernelid(5) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(square[i][j][(k+1)], square[i][j][(k-1)], u[i][j][(k+1)][3], u[i][j][(k+1)][4], u[i][j][(k-1)][3], u[i][j][(k-1)][4]) 
#pragma cuda gpurun constant(c1, c2, con43, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, tz2, zzcon2, zzcon3, zzcon4, zzcon5) 
	compute_rhs_clnd1_kernel5<<<dimGrid5, dimBlock5, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__qs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__rho_i), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__square), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__us), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__vs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)])gpu__ws));
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
#pragma omp parallel shared(dssp, grid_points, rhs, u) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) noshared(Pface) nog2cmemtr(buf, cuf, dssp, grid_points, q, rhs, u, ue) nocudafree(buf, cuf, dssp, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dssp, grid_points, rhs, u) 
#pragma cuda ainfo kernelid(6) procname(compute_rhs_clnd1) 
#pragma cuda gpurun registerRO(grid_points[0], grid_points[1], grid_points[2]) 
#pragma cuda gpurun constant(dssp) 
	compute_rhs_clnd1_kernel6<<<dimGrid6, dimBlock6, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs), ((double (*)[((((162+1)/2)*2)+1)][((((162+1)/2)*2)+1)][5])gpu__u));
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
#pragma omp parallel shared(dt, grid_points, rhs) private(i, j, k, m)
#pragma cuda gpurun noc2gmemtr(buf, cuf, dt, grid_points, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, dt, grid_points, q, rhs, ue) nocudafree(buf, cuf, dt, grid_points, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(dt, grid_points, rhs) 
#pragma cuda ainfo kernelid(7) procname(compute_rhs_clnd1) 
#pragma cuda gpurun constant(dt) 
	compute_rhs_clnd1_kernel7<<<dimGrid7, dimBlock7, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
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
	ce[0][1]=0.0;
	ce[0][2]=0.0;
	ce[0][3]=4.0;
	ce[0][4]=5.0;
	ce[0][5]=3.0;
	ce[0][6]=0.5;
	ce[0][7]=0.02;
	ce[0][8]=0.01;
	ce[0][9]=0.03;
	ce[0][10]=0.5;
	ce[0][11]=0.4;
	ce[0][12]=0.3;
	ce[1][0]=1.0;
	ce[1][1]=0.0;
	ce[1][2]=0.0;
	ce[1][3]=0.0;
	ce[1][4]=1.0;
	ce[1][5]=2.0;
	ce[1][6]=3.0;
	ce[1][7]=0.01;
	ce[1][8]=0.03;
	ce[1][9]=0.02;
	ce[1][10]=0.4;
	ce[1][11]=0.3;
	ce[1][12]=0.5;
	ce[2][0]=2.0;
	ce[2][1]=2.0;
	ce[2][2]=0.0;
	ce[2][3]=0.0;
	ce[2][4]=0.0;
	ce[2][5]=2.0;
	ce[2][6]=3.0;
	ce[2][7]=0.04;
	ce[2][8]=0.03;
	ce[2][9]=0.05;
	ce[2][10]=0.3;
	ce[2][11]=0.5;
	ce[2][12]=0.4;
	ce[3][0]=2.0;
	ce[3][1]=2.0;
	ce[3][2]=0.0;
	ce[3][3]=0.0;
	ce[3][4]=0.0;
	ce[3][5]=2.0;
	ce[3][6]=3.0;
	ce[3][7]=0.03;
	ce[3][8]=0.05;
	ce[3][9]=0.04;
	ce[3][10]=0.2;
	ce[3][11]=0.1;
	ce[3][12]=0.3;
	ce[4][0]=5.0;
	ce[4][1]=4.0;
	ce[4][2]=3.0;
	ce[4][3]=2.0;
	ce[4][4]=0.1;
	ce[4][5]=0.4;
	ce[4][6]=0.3;
	ce[4][7]=0.05;
	ce[4][8]=0.04;
	ce[4][9]=0.03;
	ce[4][10]=0.1;
	ce[4][11]=0.3;
	ce[4][12]=0.2;
	c1=1.4;
	c2=0.4;
	c3=0.1;
	c4=1.0;
	c5=1.4;
	dnxm1=(1.0/((double)(grid_points[0]-1)));
	dnym1=(1.0/((double)(grid_points[1]-1)));
	dnzm1=(1.0/((double)(grid_points[2]-1)));
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
static void verify(int no_time_steps, char * cclass, int * verified)
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c  verification routine                         
	   c-------------------------------------------------------------------
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
	   c-------------------------------------------------------------------
	 */
	epsilon=1.0E-8;
	/*
	   --------------------------------------------------------------------
	   c   compute the error norm and the residual norm, and exit if not printing
	   c-------------------------------------------------------------------
	 */
	error_norm(xce);
	compute_rhs_clnd2();
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
	   c    reference data for 12X12X12 grids after 100 time steps, with DT = 1.0d-02
	   c-------------------------------------------------------------------
	 */
	if (((((grid_points[0]==12)&&(grid_points[1]==12))&&(grid_points[2]==12))&&(no_time_steps==60)))
	{
		( * cclass)='S';
		dtref=0.01;
		/*
		   --------------------------------------------------------------------
		   c  Reference values of RMS-norms of residual.
		   c-------------------------------------------------------------------
		 */
		xcrref[0]=0.17034283709541312;
		xcrref[1]=0.012975252070034096;
		xcrref[2]=0.032527926989486054;
		xcrref[3]=0.0264364212751668;
		xcrref[4]=0.1921178413174443;
		/*
		   --------------------------------------------------------------------
		   c  Reference values of RMS-norms of solution error.
		   c-------------------------------------------------------------------
		 */
		xceref[0]=4.997691334581158E-4;
		xceref[1]=4.519566678296193E-5;
		xceref[2]=7.397376517292135E-5;
		xceref[3]=7.382123863243973E-5;
		xceref[4]=8.926963098749145E-4;
		/*
		   --------------------------------------------------------------------
		   c    reference data for 24X24X24 grids after 200 time steps, with DT = 0.8d-3
		   c-------------------------------------------------------------------
		 */
	}
	else
	{
		if (((((grid_points[0]==24)&&(grid_points[1]==24))&&(grid_points[2]==24))&&(no_time_steps==200)))
		{
			( * cclass)='W';
			dtref=8.0E-4;
			/*
			   --------------------------------------------------------------------
			   c  Reference values of RMS-norms of residual.
			   c-------------------------------------------------------------------
			 */
			xcrref[0]=112.5590409344;
			xcrref[1]=11.80007595731;
			xcrref[2]=27.10329767846;
			xcrref[3]=24.69174937669;
			xcrref[4]=263.8427874317;
			/*
			   --------------------------------------------------------------------
			   c  Reference values of RMS-norms of solution error.
			   c-------------------------------------------------------------------
			 */
			xceref[0]=4.419655736008;
			xceref[1]=0.4638531260002;
			xceref[2]=1.011551749967;
			xceref[3]=0.9235878729944;
			xceref[4]=10.18045837718;
			/*
			   --------------------------------------------------------------------
			   c    reference data for 64X64X64 grids after 200 time steps, with DT = 0.8d-3
			   c-------------------------------------------------------------------
			 */
		}
		else
		{
			if (((((grid_points[0]==64)&&(grid_points[1]==64))&&(grid_points[2]==64))&&(no_time_steps==200)))
			{
				( * cclass)='A';
				dtref=8.0E-4;
				/*
				   --------------------------------------------------------------------
				   c  Reference values of RMS-norms of residual.
				   c-------------------------------------------------------------------
				 */
				xcrref[0]=108.06346714637264;
				xcrref[1]=11.319730901220813;
				xcrref[2]=25.974354511582465;
				xcrref[3]=23.66562254467891;
				xcrref[4]=252.78963211748345;
				/*
				   --------------------------------------------------------------------
				   c  Reference values of RMS-norms of solution error.
				   c-------------------------------------------------------------------
				 */
				xceref[0]=4.2348416040525025;
				xceref[1]=0.443902824969957;
				xceref[2]=0.9669248013634565;
				xceref[3]=0.8830206303976548;
				xceref[4]=9.737990177082928;
				/*
				   --------------------------------------------------------------------
				   c    reference data for 102X102X102 grids after 200 time steps,
				   c    with DT = 3.0d-04
				   c-------------------------------------------------------------------
				 */
			}
			else
			{
				if (((((grid_points[0]==102)&&(grid_points[1]==102))&&(grid_points[2]==102))&&(no_time_steps==200)))
				{
					( * cclass)='B';
					dtref=3.0E-4;
					/*
					   --------------------------------------------------------------------
					   c  Reference values of RMS-norms of residual.
					   c-------------------------------------------------------------------
					 */
					xcrref[0]=1423.3597229287254;
					xcrref[1]=99.33052259015024;
					xcrref[2]=356.46025644535285;
					xcrref[3]=324.8544795908409;
					xcrref[4]=3270.7541254659363;
					/*
					   --------------------------------------------------------------------
					   c  Reference values of RMS-norms of solution error.
					   c-------------------------------------------------------------------
					 */
					xceref[0]=52.96984714093686;
					xceref[1]=4.463289611567067;
					xceref[2]=13.122573342210174;
					xceref[3]=12.006925323559145;
					xceref[4]=124.59576151035986;
					/*
					   --------------------------------------------------------------------
					   c    reference data for 162X162X162 grids after 200 time steps,
					   c    with DT = 1.0d-04
					   c-------------------------------------------------------------------
					 */
				}
				else
				{
					if (((((grid_points[0]==162)&&(grid_points[1]==162))&&(grid_points[2]==162))&&(no_time_steps==200)))
					{
						( * cclass)='C';
						dtref=1.0E-4;
						/*
						   --------------------------------------------------------------------
						   c  Reference values of RMS-norms of residual.
						   c-------------------------------------------------------------------
						 */
						xcrref[0]=6239.8116551764615;
						xcrref[1]=507.93239190423964;
						xcrref[2]=1542.3530093013596;
						xcrref[3]=1330.238792929119;
						xcrref[4]=11604.087428436455;
						/*
						   --------------------------------------------------------------------
						   c  Reference values of RMS-norms of solution error.
						   c-------------------------------------------------------------------
						 */
						xceref[0]=164.62008369091265;
						xceref[1]=11.497107903824313;
						xceref[2]=41.20744620746151;
						xceref[3]=37.08765105969417;
						xceref[4]=362.11053051841265;
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
	   c-------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c    Compute the difference of solution values and the known reference values.
	   c-------------------------------------------------------------------
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
	   c-------------------------------------------------------------------
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
		if ((( * verified)==1))
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
static void x_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     
	   c     Performs line solves in X direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix, 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c     
	   c-------------------------------------------------------------------
	 */
	lhsx();
	x_solve_cell();
	x_backsubstitute();
	return ;
}

static void x_solve_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     
	   c     Performs line solves in X direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix, 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c     
	   c-------------------------------------------------------------------
	 */
	lhsx_clnd1();
	x_solve_cell_clnd1();
	x_backsubstitute_clnd1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void x_backsubstitute_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_backsubstitute#0#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name x_backsubstitute#0#0#0#0 
			for (m=0; m<5; m ++ )
			{
#pragma loop name x_backsubstitute#0#0#0#0#0 
				for (n=0; n<5; n ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[(i+1)][j][k][n]));
				}
			}
		}
	}
}

static void x_backsubstitute(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell, then generate U(isize)=rhs[isize)
	   c     else assume U(isize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(istart) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int i;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
#pragma loop name x_backsubstitute#0 
	for (i=(grid_points[0]-2); i>=0; i -- )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_backsubstitute) 
		x_backsubstitute_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	return ;
}

__global__ void x_backsubstitute_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (m<5)
	{
#pragma loop name x_backsubstitute#0#0#0 
		for (n=0; n<5; n ++ )
		{
#pragma loop name x_backsubstitute#0#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
#pragma loop name x_backsubstitute#0#0#0#0#0 
				for (j=1; j<(grid_points[1]-1); j ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[(i+1)][j][k][n]));
				}
			}
		}
	}
}

static void x_backsubstitute_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell, then generate U(isize)=rhs[isize)
	   c     else assume U(isize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(istart) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int i;
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
#pragma loop name x_backsubstitute#0 
	for (i=(grid_points[0]-2); i>=0; i -- )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_backsubstitute_clnd1) 
		x_backsubstitute_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void x_solve_cell_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int isize;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	isize=(grid_points[0]-1);
	/*
	   --------------------------------------------------------------------
	   c     outer most do loops - sweeping in i direction
	   c-------------------------------------------------------------------
	 */
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(0,j,k) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[0][j][k][1], lhs[0][j][k][2], rhs[0][j][k]);
		}
	}
}

__global__ void x_solve_cell_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#1#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(i) = rhs(i) - Arhs(i-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[(i-1)][j][k], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(i) = B(i) - C(i-1)A(i)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[(i-1)][j][k][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(1,j,k) by b_inverse(1,j,k) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void x_solve_cell_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int isize;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(isize) = rhs(isize) - Arhs(isize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[isize][j][k][0], rhs[(isize-1)][j][k], rhs[isize][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(isize) = B(isize) - C(isize-1)A(isize)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[isize][j][k][0], lhs[(isize-1)][j][k][2], lhs[isize][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs() by b_inverse() and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][j][k][1], rhs[i][j][k]);
		}
	}
}

static void x_solve_cell(void )
{
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(IMAX) and rhs'(IMAX) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int i;
	int isize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(isize, j, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_solve_cell) 
	x_solve_cell_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name x_solve_cell#1 
	for (i=1; i<isize; i ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(x_solve_cell) 
		x_solve_cell_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, isize, j, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(x_solve_cell) 
	x_solve_cell_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

__global__ void x_solve_cell_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int isize;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	isize=(grid_points[0]-1);
	/*
	   --------------------------------------------------------------------
	   c     outer most do loops - sweeping in i direction
	   c-------------------------------------------------------------------
	 */
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(0,j,k) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[0][j][k][1], lhs[0][j][k][2], rhs[0][j][k]);
		}
	}
}

__global__ void x_solve_cell_clnd1_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#1#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(i) = rhs(i) - Arhs(i-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[(i-1)][j][k], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(i) = B(i) - C(i-1)A(i)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[(i-1)][j][k][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(1,j,k) by b_inverse(1,j,k) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void x_solve_cell_clnd1_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int isize;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	j=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(j)
	if (j<(grid_points[1]-1))
	{
#pragma loop name x_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(isize) = rhs(isize) - Arhs(isize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[isize][j][k][0], rhs[(isize-1)][j][k], rhs[isize][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(isize) = B(isize) - C(isize-1)A(isize)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[isize][j][k][0], lhs[(isize-1)][j][k][2], lhs[isize][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs() by b_inverse() and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][j][k][1], rhs[i][j][k]);
		}
	}
}

static void x_solve_cell_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(IMAX) and rhs'(IMAX) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int i;
	int isize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(isize, j, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(x_solve_cell_clnd1) 
	x_solve_cell_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name x_solve_cell#1 
	for (i=1; i<isize; i ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(x_solve_cell_clnd1) 
		x_solve_cell_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[1]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, isize, j, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(x_solve_cell_clnd1) 
	x_solve_cell_clnd1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__device__ static void dev_matvec_sub(double ablock[5][5], double avec[5], double bvec[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     subtracts bvec=bvec - ablockavec
	   c-------------------------------------------------------------------
	 */
	int i;
#pragma loop name matvec_sub#0 
	for (i=0; i<5; i ++ )
	{
		/*
		   --------------------------------------------------------------------
		   c            rhs(i,ic,jc,kc,ccell) = rhs(i,ic,jc,kc,ccell) 
		   c     $           - lhs[i,1,ablock,ia,ja,ka,acell)
		   c-------------------------------------------------------------------
		 */
		bvec[i]=(((((bvec[i]-(ablock[i][0]*avec[0]))-(ablock[i][1]*avec[1]))-(ablock[i][2]*avec[2]))-(ablock[i][3]*avec[3]))-(ablock[i][4]*avec[4]));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__device__ static void dev_matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
	   c-------------------------------------------------------------------
	 */
	int j;
#pragma loop name matmul_sub#0 
	for (j=0; j<5; j ++ )
	{
		cblock[0][j]=(((((cblock[0][j]-(ablock[0][0]*bblock[0][j]))-(ablock[0][1]*bblock[1][j]))-(ablock[0][2]*bblock[2][j]))-(ablock[0][3]*bblock[3][j]))-(ablock[0][4]*bblock[4][j]));
		cblock[1][j]=(((((cblock[1][j]-(ablock[1][0]*bblock[0][j]))-(ablock[1][1]*bblock[1][j]))-(ablock[1][2]*bblock[2][j]))-(ablock[1][3]*bblock[3][j]))-(ablock[1][4]*bblock[4][j]));
		cblock[2][j]=(((((cblock[2][j]-(ablock[2][0]*bblock[0][j]))-(ablock[2][1]*bblock[1][j]))-(ablock[2][2]*bblock[2][j]))-(ablock[2][3]*bblock[3][j]))-(ablock[2][4]*bblock[4][j]));
		cblock[3][j]=(((((cblock[3][j]-(ablock[3][0]*bblock[0][j]))-(ablock[3][1]*bblock[1][j]))-(ablock[3][2]*bblock[2][j]))-(ablock[3][3]*bblock[3][j]))-(ablock[3][4]*bblock[4][j]));
		cblock[4][j]=(((((cblock[4][j]-(ablock[4][0]*bblock[0][j]))-(ablock[4][1]*bblock[1][j]))-(ablock[4][2]*bblock[2][j]))-(ablock[4][3]*bblock[3][j]))-(ablock[4][4]*bblock[4][j]));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__device__ static void dev_binvcrhs(double lhs[5][5], double c[5][5], double r[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	double pivot;
	double coeff;
	/*
	   --------------------------------------------------------------------
	   c     
	   c-------------------------------------------------------------------
	 */
	pivot=(1.0/lhs[0][0]);
	lhs[0][1]=(lhs[0][1]*pivot);
	lhs[0][2]=(lhs[0][2]*pivot);
	lhs[0][3]=(lhs[0][3]*pivot);
	lhs[0][4]=(lhs[0][4]*pivot);
	c[0][0]=(c[0][0]*pivot);
	c[0][1]=(c[0][1]*pivot);
	c[0][2]=(c[0][2]*pivot);
	c[0][3]=(c[0][3]*pivot);
	c[0][4]=(c[0][4]*pivot);
	r[0]=(r[0]*pivot);
	coeff=lhs[1][0];
	lhs[1][1]=(lhs[1][1]-(coeff*lhs[0][1]));
	lhs[1][2]=(lhs[1][2]-(coeff*lhs[0][2]));
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[0][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[0][4]));
	c[1][0]=(c[1][0]-(coeff*c[0][0]));
	c[1][1]=(c[1][1]-(coeff*c[0][1]));
	c[1][2]=(c[1][2]-(coeff*c[0][2]));
	c[1][3]=(c[1][3]-(coeff*c[0][3]));
	c[1][4]=(c[1][4]-(coeff*c[0][4]));
	r[1]=(r[1]-(coeff*r[0]));
	coeff=lhs[2][0];
	lhs[2][1]=(lhs[2][1]-(coeff*lhs[0][1]));
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[0][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[0][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[0][4]));
	c[2][0]=(c[2][0]-(coeff*c[0][0]));
	c[2][1]=(c[2][1]-(coeff*c[0][1]));
	c[2][2]=(c[2][2]-(coeff*c[0][2]));
	c[2][3]=(c[2][3]-(coeff*c[0][3]));
	c[2][4]=(c[2][4]-(coeff*c[0][4]));
	r[2]=(r[2]-(coeff*r[0]));
	coeff=lhs[3][0];
	lhs[3][1]=(lhs[3][1]-(coeff*lhs[0][1]));
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[0][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[0][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[0][4]));
	c[3][0]=(c[3][0]-(coeff*c[0][0]));
	c[3][1]=(c[3][1]-(coeff*c[0][1]));
	c[3][2]=(c[3][2]-(coeff*c[0][2]));
	c[3][3]=(c[3][3]-(coeff*c[0][3]));
	c[3][4]=(c[3][4]-(coeff*c[0][4]));
	r[3]=(r[3]-(coeff*r[0]));
	coeff=lhs[4][0];
	lhs[4][1]=(lhs[4][1]-(coeff*lhs[0][1]));
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[0][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[0][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[0][4]));
	c[4][0]=(c[4][0]-(coeff*c[0][0]));
	c[4][1]=(c[4][1]-(coeff*c[0][1]));
	c[4][2]=(c[4][2]-(coeff*c[0][2]));
	c[4][3]=(c[4][3]-(coeff*c[0][3]));
	c[4][4]=(c[4][4]-(coeff*c[0][4]));
	r[4]=(r[4]-(coeff*r[0]));
	pivot=(1.0/lhs[1][1]);
	lhs[1][2]=(lhs[1][2]*pivot);
	lhs[1][3]=(lhs[1][3]*pivot);
	lhs[1][4]=(lhs[1][4]*pivot);
	c[1][0]=(c[1][0]*pivot);
	c[1][1]=(c[1][1]*pivot);
	c[1][2]=(c[1][2]*pivot);
	c[1][3]=(c[1][3]*pivot);
	c[1][4]=(c[1][4]*pivot);
	r[1]=(r[1]*pivot);
	coeff=lhs[0][1];
	lhs[0][2]=(lhs[0][2]-(coeff*lhs[1][2]));
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[1][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[1][4]));
	c[0][0]=(c[0][0]-(coeff*c[1][0]));
	c[0][1]=(c[0][1]-(coeff*c[1][1]));
	c[0][2]=(c[0][2]-(coeff*c[1][2]));
	c[0][3]=(c[0][3]-(coeff*c[1][3]));
	c[0][4]=(c[0][4]-(coeff*c[1][4]));
	r[0]=(r[0]-(coeff*r[1]));
	coeff=lhs[2][1];
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[1][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[1][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[1][4]));
	c[2][0]=(c[2][0]-(coeff*c[1][0]));
	c[2][1]=(c[2][1]-(coeff*c[1][1]));
	c[2][2]=(c[2][2]-(coeff*c[1][2]));
	c[2][3]=(c[2][3]-(coeff*c[1][3]));
	c[2][4]=(c[2][4]-(coeff*c[1][4]));
	r[2]=(r[2]-(coeff*r[1]));
	coeff=lhs[3][1];
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[1][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[1][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[1][4]));
	c[3][0]=(c[3][0]-(coeff*c[1][0]));
	c[3][1]=(c[3][1]-(coeff*c[1][1]));
	c[3][2]=(c[3][2]-(coeff*c[1][2]));
	c[3][3]=(c[3][3]-(coeff*c[1][3]));
	c[3][4]=(c[3][4]-(coeff*c[1][4]));
	r[3]=(r[3]-(coeff*r[1]));
	coeff=lhs[4][1];
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[1][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[1][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[1][4]));
	c[4][0]=(c[4][0]-(coeff*c[1][0]));
	c[4][1]=(c[4][1]-(coeff*c[1][1]));
	c[4][2]=(c[4][2]-(coeff*c[1][2]));
	c[4][3]=(c[4][3]-(coeff*c[1][3]));
	c[4][4]=(c[4][4]-(coeff*c[1][4]));
	r[4]=(r[4]-(coeff*r[1]));
	pivot=(1.0/lhs[2][2]);
	lhs[2][3]=(lhs[2][3]*pivot);
	lhs[2][4]=(lhs[2][4]*pivot);
	c[2][0]=(c[2][0]*pivot);
	c[2][1]=(c[2][1]*pivot);
	c[2][2]=(c[2][2]*pivot);
	c[2][3]=(c[2][3]*pivot);
	c[2][4]=(c[2][4]*pivot);
	r[2]=(r[2]*pivot);
	coeff=lhs[0][2];
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[2][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[2][4]));
	c[0][0]=(c[0][0]-(coeff*c[2][0]));
	c[0][1]=(c[0][1]-(coeff*c[2][1]));
	c[0][2]=(c[0][2]-(coeff*c[2][2]));
	c[0][3]=(c[0][3]-(coeff*c[2][3]));
	c[0][4]=(c[0][4]-(coeff*c[2][4]));
	r[0]=(r[0]-(coeff*r[2]));
	coeff=lhs[1][2];
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[2][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[2][4]));
	c[1][0]=(c[1][0]-(coeff*c[2][0]));
	c[1][1]=(c[1][1]-(coeff*c[2][1]));
	c[1][2]=(c[1][2]-(coeff*c[2][2]));
	c[1][3]=(c[1][3]-(coeff*c[2][3]));
	c[1][4]=(c[1][4]-(coeff*c[2][4]));
	r[1]=(r[1]-(coeff*r[2]));
	coeff=lhs[3][2];
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[2][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[2][4]));
	c[3][0]=(c[3][0]-(coeff*c[2][0]));
	c[3][1]=(c[3][1]-(coeff*c[2][1]));
	c[3][2]=(c[3][2]-(coeff*c[2][2]));
	c[3][3]=(c[3][3]-(coeff*c[2][3]));
	c[3][4]=(c[3][4]-(coeff*c[2][4]));
	r[3]=(r[3]-(coeff*r[2]));
	coeff=lhs[4][2];
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[2][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[2][4]));
	c[4][0]=(c[4][0]-(coeff*c[2][0]));
	c[4][1]=(c[4][1]-(coeff*c[2][1]));
	c[4][2]=(c[4][2]-(coeff*c[2][2]));
	c[4][3]=(c[4][3]-(coeff*c[2][3]));
	c[4][4]=(c[4][4]-(coeff*c[2][4]));
	r[4]=(r[4]-(coeff*r[2]));
	pivot=(1.0/lhs[3][3]);
	lhs[3][4]=(lhs[3][4]*pivot);
	c[3][0]=(c[3][0]*pivot);
	c[3][1]=(c[3][1]*pivot);
	c[3][2]=(c[3][2]*pivot);
	c[3][3]=(c[3][3]*pivot);
	c[3][4]=(c[3][4]*pivot);
	r[3]=(r[3]*pivot);
	coeff=lhs[0][3];
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[3][4]));
	c[0][0]=(c[0][0]-(coeff*c[3][0]));
	c[0][1]=(c[0][1]-(coeff*c[3][1]));
	c[0][2]=(c[0][2]-(coeff*c[3][2]));
	c[0][3]=(c[0][3]-(coeff*c[3][3]));
	c[0][4]=(c[0][4]-(coeff*c[3][4]));
	r[0]=(r[0]-(coeff*r[3]));
	coeff=lhs[1][3];
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[3][4]));
	c[1][0]=(c[1][0]-(coeff*c[3][0]));
	c[1][1]=(c[1][1]-(coeff*c[3][1]));
	c[1][2]=(c[1][2]-(coeff*c[3][2]));
	c[1][3]=(c[1][3]-(coeff*c[3][3]));
	c[1][4]=(c[1][4]-(coeff*c[3][4]));
	r[1]=(r[1]-(coeff*r[3]));
	coeff=lhs[2][3];
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[3][4]));
	c[2][0]=(c[2][0]-(coeff*c[3][0]));
	c[2][1]=(c[2][1]-(coeff*c[3][1]));
	c[2][2]=(c[2][2]-(coeff*c[3][2]));
	c[2][3]=(c[2][3]-(coeff*c[3][3]));
	c[2][4]=(c[2][4]-(coeff*c[3][4]));
	r[2]=(r[2]-(coeff*r[3]));
	coeff=lhs[4][3];
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[3][4]));
	c[4][0]=(c[4][0]-(coeff*c[3][0]));
	c[4][1]=(c[4][1]-(coeff*c[3][1]));
	c[4][2]=(c[4][2]-(coeff*c[3][2]));
	c[4][3]=(c[4][3]-(coeff*c[3][3]));
	c[4][4]=(c[4][4]-(coeff*c[3][4]));
	r[4]=(r[4]-(coeff*r[3]));
	pivot=(1.0/lhs[4][4]);
	c[4][0]=(c[4][0]*pivot);
	c[4][1]=(c[4][1]*pivot);
	c[4][2]=(c[4][2]*pivot);
	c[4][3]=(c[4][3]*pivot);
	c[4][4]=(c[4][4]*pivot);
	r[4]=(r[4]*pivot);
	coeff=lhs[0][4];
	c[0][0]=(c[0][0]-(coeff*c[4][0]));
	c[0][1]=(c[0][1]-(coeff*c[4][1]));
	c[0][2]=(c[0][2]-(coeff*c[4][2]));
	c[0][3]=(c[0][3]-(coeff*c[4][3]));
	c[0][4]=(c[0][4]-(coeff*c[4][4]));
	r[0]=(r[0]-(coeff*r[4]));
	coeff=lhs[1][4];
	c[1][0]=(c[1][0]-(coeff*c[4][0]));
	c[1][1]=(c[1][1]-(coeff*c[4][1]));
	c[1][2]=(c[1][2]-(coeff*c[4][2]));
	c[1][3]=(c[1][3]-(coeff*c[4][3]));
	c[1][4]=(c[1][4]-(coeff*c[4][4]));
	r[1]=(r[1]-(coeff*r[4]));
	coeff=lhs[2][4];
	c[2][0]=(c[2][0]-(coeff*c[4][0]));
	c[2][1]=(c[2][1]-(coeff*c[4][1]));
	c[2][2]=(c[2][2]-(coeff*c[4][2]));
	c[2][3]=(c[2][3]-(coeff*c[4][3]));
	c[2][4]=(c[2][4]-(coeff*c[4][4]));
	r[2]=(r[2]-(coeff*r[4]));
	coeff=lhs[3][4];
	c[3][0]=(c[3][0]-(coeff*c[4][0]));
	c[3][1]=(c[3][1]-(coeff*c[4][1]));
	c[3][2]=(c[3][2]-(coeff*c[4][2]));
	c[3][3]=(c[3][3]-(coeff*c[4][3]));
	c[3][4]=(c[3][4]-(coeff*c[4][4]));
	r[3]=(r[3]-(coeff*r[4]));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__device__ static void dev_binvrhs(double lhs[5][5], double r[5])
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	double pivot;
	double coeff;
	/*
	   --------------------------------------------------------------------
	   c     
	   c-------------------------------------------------------------------
	 */
	pivot=(1.0/lhs[0][0]);
	lhs[0][1]=(lhs[0][1]*pivot);
	lhs[0][2]=(lhs[0][2]*pivot);
	lhs[0][3]=(lhs[0][3]*pivot);
	lhs[0][4]=(lhs[0][4]*pivot);
	r[0]=(r[0]*pivot);
	coeff=lhs[1][0];
	lhs[1][1]=(lhs[1][1]-(coeff*lhs[0][1]));
	lhs[1][2]=(lhs[1][2]-(coeff*lhs[0][2]));
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[0][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[0][4]));
	r[1]=(r[1]-(coeff*r[0]));
	coeff=lhs[2][0];
	lhs[2][1]=(lhs[2][1]-(coeff*lhs[0][1]));
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[0][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[0][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[0][4]));
	r[2]=(r[2]-(coeff*r[0]));
	coeff=lhs[3][0];
	lhs[3][1]=(lhs[3][1]-(coeff*lhs[0][1]));
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[0][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[0][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[0][4]));
	r[3]=(r[3]-(coeff*r[0]));
	coeff=lhs[4][0];
	lhs[4][1]=(lhs[4][1]-(coeff*lhs[0][1]));
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[0][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[0][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[0][4]));
	r[4]=(r[4]-(coeff*r[0]));
	pivot=(1.0/lhs[1][1]);
	lhs[1][2]=(lhs[1][2]*pivot);
	lhs[1][3]=(lhs[1][3]*pivot);
	lhs[1][4]=(lhs[1][4]*pivot);
	r[1]=(r[1]*pivot);
	coeff=lhs[0][1];
	lhs[0][2]=(lhs[0][2]-(coeff*lhs[1][2]));
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[1][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[1][4]));
	r[0]=(r[0]-(coeff*r[1]));
	coeff=lhs[2][1];
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[1][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[1][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[1][4]));
	r[2]=(r[2]-(coeff*r[1]));
	coeff=lhs[3][1];
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[1][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[1][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[1][4]));
	r[3]=(r[3]-(coeff*r[1]));
	coeff=lhs[4][1];
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[1][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[1][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[1][4]));
	r[4]=(r[4]-(coeff*r[1]));
	pivot=(1.0/lhs[2][2]);
	lhs[2][3]=(lhs[2][3]*pivot);
	lhs[2][4]=(lhs[2][4]*pivot);
	r[2]=(r[2]*pivot);
	coeff=lhs[0][2];
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[2][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[2][4]));
	r[0]=(r[0]-(coeff*r[2]));
	coeff=lhs[1][2];
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[2][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[2][4]));
	r[1]=(r[1]-(coeff*r[2]));
	coeff=lhs[3][2];
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[2][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[2][4]));
	r[3]=(r[3]-(coeff*r[2]));
	coeff=lhs[4][2];
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[2][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[2][4]));
	r[4]=(r[4]-(coeff*r[2]));
	pivot=(1.0/lhs[3][3]);
	lhs[3][4]=(lhs[3][4]*pivot);
	r[3]=(r[3]*pivot);
	coeff=lhs[0][3];
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[3][4]));
	r[0]=(r[0]-(coeff*r[3]));
	coeff=lhs[1][3];
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[3][4]));
	r[1]=(r[1]-(coeff*r[3]));
	coeff=lhs[2][3];
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[3][4]));
	r[2]=(r[2]-(coeff*r[3]));
	coeff=lhs[4][3];
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[3][4]));
	r[4]=(r[4]-(coeff*r[3]));
	pivot=(1.0/lhs[4][4]);
	r[4]=(r[4]*pivot);
	coeff=lhs[0][4];
	r[0]=(r[0]-(coeff*r[4]));
	coeff=lhs[1][4];
	r[1]=(r[1]-(coeff*r[4]));
	coeff=lhs[2][4];
	r[2]=(r[2]-(coeff*r[4]));
	coeff=lhs[3][4];
	r[3]=(r[3]-(coeff*r[4]));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void y_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     Performs line solves in Y direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix][ 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c-------------------------------------------------------------------
	 */
	lhsy();
	y_solve_cell();
	y_backsubstitute();
	return ;
}

static void y_solve_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     Performs line solves in Y direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix][ 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c-------------------------------------------------------------------
	 */
	lhsy_clnd1();
	y_solve_cell_clnd1();
	y_backsubstitute_clnd1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void y_backsubstitute_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_backsubstitute#0#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
#pragma loop name y_backsubstitute#0#0#0#0 
			for (m=0; m<5; m ++ )
			{
#pragma loop name y_backsubstitute#0#0#0#0#0 
				for (n=0; n<5; n ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][(j+1)][k][n]));
				}
			}
		}
	}
}

static void y_backsubstitute(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell][ then generate U(jsize)=rhs(jsize)
	   c     else assume U(jsize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(jstart) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int j;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma loop name y_backsubstitute#0 
	for (j=(grid_points[1]-2); j>=0; j -- )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_backsubstitute) 
		y_backsubstitute_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	return ;
}

__global__ void y_backsubstitute_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (m<5)
	{
#pragma loop name y_backsubstitute#0#0#0 
		for (n=0; n<5; n ++ )
		{
#pragma loop name y_backsubstitute#0#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
#pragma loop name y_backsubstitute#0#0#0#0#0 
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][(j+1)][k][n]));
				}
			}
		}
	}
}

static void y_backsubstitute_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell][ then generate U(jsize)=rhs(jsize)
	   c     else assume U(jsize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(jstart) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int j;
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
#pragma loop name y_backsubstitute#0 
	for (j=(grid_points[1]-2); j>=0; j -- )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_backsubstitute_clnd1) 
		y_backsubstitute_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void y_solve_cell_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int jsize;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	jsize=(grid_points[1]-1);
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,0,k) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][0][k][1], lhs[i][0][k][2], rhs[i][0][k]);
		}
	}
}

__global__ void y_solve_cell_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#1#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     subtract Alhs_vector(j-1) from lhs_vector(j)
			   c     
			   c     rhs(j) = rhs(j) - A*rhs(j-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[i][(j-1)][k], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(j) = B(j) - C(j-1)A(j)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[i][(j-1)][k][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(i,1,k) by b_inverse(i,1,k) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void y_solve_cell_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int jsize;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(jsize) = rhs(jsize) - Arhs(jsize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][jsize][k][0], rhs[i][(jsize-1)][k], rhs[i][jsize][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(jsize) = B(jsize) - C(jsize-1)A(jsize)
			   c     call matmul_sub(aa,i,jsize,k,c,
			   c     $              cc,i,jsize-1,k,c,BB,i,jsize,k)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][jsize][k][0], lhs[i][(jsize-1)][k][2], lhs[i][jsize][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][jsize][k][1], rhs[i][jsize][k]);
		}
	}
}

static void y_solve_cell(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(JMAX) and rhs'(JMAX) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int j;
	int jsize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, jsize, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_solve_cell) 
	y_solve_cell_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name y_solve_cell#1 
	for (j=1; j<jsize; j ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(y_solve_cell) 
		y_solve_cell_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, jsize, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(y_solve_cell) 
	y_solve_cell_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

__global__ void y_solve_cell_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int jsize;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	jsize=(grid_points[1]-1);
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,0,k) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][0][k][1], lhs[i][0][k][2], rhs[i][0][k]);
		}
	}
}

__global__ void y_solve_cell_clnd1_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#1#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     subtract Alhs_vector(j-1) from lhs_vector(j)
			   c     
			   c     rhs(j) = rhs(j) - A*rhs(j-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[i][(j-1)][k], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(j) = B(j) - C(j-1)A(j)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[i][(j-1)][k][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(i,1,k) by b_inverse(i,1,k) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void y_solve_cell_clnd1_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int jsize;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name y_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(jsize) = rhs(jsize) - Arhs(jsize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][jsize][k][0], rhs[i][(jsize-1)][k], rhs[i][jsize][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(jsize) = B(jsize) - C(jsize-1)A(jsize)
			   c     call matmul_sub(aa,i,jsize,k,c,
			   c     $              cc,i,jsize-1,k,c,BB,i,jsize,k)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][jsize][k][0], lhs[i][(jsize-1)][k][2], lhs[i][jsize][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][jsize][k][1], rhs[i][jsize][k]);
		}
	}
}

static void y_solve_cell_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(JMAX) and rhs'(JMAX) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	int j;
	int jsize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, jsize, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(y_solve_cell_clnd1) 
	y_solve_cell_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name y_solve_cell#1 
	for (j=1; j<jsize; j ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(y_solve_cell_clnd1) 
		y_solve_cell_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, jsize, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(y_solve_cell_clnd1) 
	y_solve_cell_clnd1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
static void z_solve(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     Performs line solves in Z direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix, 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c-------------------------------------------------------------------
	 */
	lhsz();
	z_solve_cell();
	z_backsubstitute();
	return ;
}

static void z_solve_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     Performs line solves in Z direction by first factoring
	   c     the block-tridiagonal matrix into an upper triangular matrix, 
	   c     and then performing back substitution to solve for the unknow
	   c     vectors of each line.  
	   c     
	   c     Make sure we treat elements zero to cell_size in the direction
	   c     of the sweep.
	   c-------------------------------------------------------------------
	 */
	lhsz_clnd1();
	z_solve_cell_clnd1();
	z_backsubstitute_clnd1();
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void z_backsubstitute_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_backsubstitute#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
#pragma loop name z_backsubstitute#0#0#0 
			for (k=(grid_points[2]-2); k>=0; k -- )
			{
#pragma loop name z_backsubstitute#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
#pragma loop name z_backsubstitute#0#0#0#0#0 
					for (n=0; n<5; n ++ )
					{
						rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][j][(k+1)][n]));
					}
				}
			}
		}
	}
}

static void z_backsubstitute(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell, then generate U(ksize)=rhs(ksize)
	   c     else assume U(ksize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(kstart) will be sent to next cell
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_backsubstitute) 
	z_backsubstitute_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

__global__ void z_backsubstitute_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int m;
	int n;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	m=_gtid;
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (m<5)
	{
#pragma loop name z_backsubstitute#0#0 
		for (n=0; n<5; n ++ )
		{
#pragma loop name z_backsubstitute#0#0#0 
			for (k=(grid_points[2]-2); k>=0; k -- )
			{
#pragma loop name z_backsubstitute#0#0#0#0 
				for (j=1; j<(grid_points[1]-1); j ++ )
				{
#pragma loop name z_backsubstitute#0#0#0#0#0 
					for (i=1; i<(grid_points[0]-1); i ++ )
					{
						rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][j][(k+1)][n]));
					}
				}
			}
		}
	}
}

static void z_backsubstitute_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     back solve: if last cell, then generate U(ksize)=rhs(ksize)
	   c     else assume U(ksize) is loaded in un pack backsub_info
	   c     so just use it
	   c     after call u(kstart) will be sent to next cell
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k, m, n)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_backsubstitute_clnd1) 
	z_backsubstitute_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

/*
   --------------------------------------------------------------------
   --------------------------------------------------------------------
 */
__global__ void z_solve_cell_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int ksize;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	ksize=(grid_points[2]-1);
	/*
	   --------------------------------------------------------------------
	   c     outer most do loops - sweeping in i direction
	   c-------------------------------------------------------------------
	 */
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,0) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][0][1], lhs[i][j][0][2], rhs[i][j][0]);
		}
	}
}

__global__ void z_solve_cell_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#1#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     subtract Alhs_vector(k-1) from lhs_vector(k)
			   c     
			   c     rhs(k) = rhs(k) - A*rhs(k-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[i][j][(k-1)], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(k) = B(k) - C(k-1)A(k)
			   c     call matmul_sub(aa,i,j,k,c,cc,i,j,k-1,c,BB,i,j,k)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[i][j][(k-1)][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(i,j,1) by b_inverse(i,j,1) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void z_solve_cell_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Now finish up special cases for last cell
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int ksize;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(ksize) = rhs(ksize) - Arhs(ksize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][ksize][0], rhs[i][j][(ksize-1)], rhs[i][j][ksize]);
			/*
			   --------------------------------------------------------------------
			   c     B(ksize) = B(ksize) - C(ksize-1)A(ksize)
			   c     call matmul_sub(aa,i,j,ksize,c,
			   c     $              cc,i,j,ksize-1,c,BB,i,j,ksize)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][ksize][0], lhs[i][j][(ksize-1)][2], lhs[i][j][ksize][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][j][ksize][1], rhs[i][j][ksize]);
		}
	}
}

static void z_solve_cell(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	   c-------------------------------------------------------------------
	 */
	int k;
	int ksize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, ksize)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_solve_cell) 
	z_solve_cell_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name z_solve_cell#1 
	for (k=1; k<ksize; k ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(z_solve_cell) 
		z_solve_cell_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, ksize)
#pragma cuda gpurun noshared(Pface) noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(z_solve_cell) 
	z_solve_cell_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

__global__ void z_solve_cell_clnd1_kernel0(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int ksize;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	ksize=(grid_points[2]-1);
	/*
	   --------------------------------------------------------------------
	   c     outer most do loops - sweeping in i direction
	   c-------------------------------------------------------------------
	 */
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,0) by b_inverse and copy back to c
			   c     multiply rhs(0) by b_inverse(0) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][0][1], lhs[i][j][0][2], rhs[i][j][0]);
		}
	}
}

__global__ void z_solve_cell_clnd1_kernel1(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	int i;
	int j;
	int k;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#1#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     subtract Alhs_vector(k-1) from lhs_vector(k)
			   c     
			   c     rhs(k) = rhs(k) - A*rhs(k-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][k][0], rhs[i][j][(k-1)], rhs[i][j][k]);
			/*
			   --------------------------------------------------------------------
			   c     B(k) = B(k) - C(k-1)A(k)
			   c     call matmul_sub(aa,i,j,k,c,cc,i,j,k-1,c,BB,i,j,k)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][k][0], lhs[i][j][(k-1)][2], lhs[i][j][k][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply c(i,j,k) by b_inverse and copy back to c
			   c     multiply rhs(i,j,1) by b_inverse(i,j,1) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
		}
	}
}

__global__ void z_solve_cell_clnd1_kernel2(int * grid_points, double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5], double rhs[(((162/2)*2)+1)][(((162/2)*2)+1)][(((162/2)*2)+1)][5])
{
	/*
	   --------------------------------------------------------------------
	   c     Now finish up special cases for last cell
	   c-------------------------------------------------------------------
	 */
	int i;
	int j;
	int ksize;
	int _bid = (blockIdx.x+(blockIdx.y*gridDim.x));
	int _gtid = (threadIdx.x+(_bid*blockDim.x));
	i=(_gtid+1);
#pragma omp for shared(grid_points, lhs, rhs) private(i)
	if (i<(grid_points[0]-1))
	{
#pragma loop name z_solve_cell#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			   --------------------------------------------------------------------
			   c     rhs(ksize) = rhs(ksize) - Arhs(ksize-1)
			   c-------------------------------------------------------------------
			 */
			dev_matvec_sub(lhs[i][j][ksize][0], rhs[i][j][(ksize-1)], rhs[i][j][ksize]);
			/*
			   --------------------------------------------------------------------
			   c     B(ksize) = B(ksize) - C(ksize-1)A(ksize)
			   c     call matmul_sub(aa,i,j,ksize,c,
			   c     $              cc,i,j,ksize-1,c,BB,i,j,ksize)
			   c-------------------------------------------------------------------
			 */
			dev_matmul_sub(lhs[i][j][ksize][0], lhs[i][j][(ksize-1)][2], lhs[i][j][ksize][1]);
			/*
			   --------------------------------------------------------------------
			   c     multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
			   c-------------------------------------------------------------------
			 */
			dev_binvrhs(lhs[i][j][ksize][1], rhs[i][j][ksize]);
		}
	}
}

static void z_solve_cell_clnd1(void )
{
	/*
	   --------------------------------------------------------------------
	   --------------------------------------------------------------------
	 */
	/*
	   --------------------------------------------------------------------
	   c     performs guaussian elimination on this cell.
	   c     
	   c     assumes that unpacking routines for non-first cells 
	   c     preload C' and rhs' from previous cell.
	   c     
	   c     assumed send happens outside this routine, but that
	   c     c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	   c-------------------------------------------------------------------
	 */
	int k;
	int ksize;
	dim3 dimBlock0(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, ksize)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(0) procname(z_solve_cell_clnd1) 
	z_solve_cell_clnd1_kernel0<<<dimGrid0, dimBlock0, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	/*
	   --------------------------------------------------------------------
	   c     begin inner most do loop
	   c     do all the elements of the cell unless last 
	   c-------------------------------------------------------------------
	 */
	dim3 dimBlock1(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid1(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma loop name z_solve_cell#1 
	for (k=1; k<ksize; k ++ )
	{
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, k)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(1) procname(z_solve_cell_clnd1) 
		z_solve_cell_clnd1_kernel1<<<dimGrid1, dimBlock1, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	}
	dim3 dimBlock2(gpuNumThreads, 1, 1);
	gpuNumBlocks=((int)ceil((((float)(-2+grid_points[0]))/1024.0F)));
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
	dim3 dimGrid2(gpuNumBlocks1, gpuNumBlocks2, 1);
	gpuNumBlocks=(gpuNumBlocks1*gpuNumBlocks2);
	totalNumThreads=(gpuNumBlocks*gpuNumThreads);
#pragma omp parallel shared(grid_points, lhs, rhs) private(i, j, ksize)
#pragma cuda gpurun noc2gmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) noshared(Pface) nog2cmemtr(buf, cuf, grid_points, lhs, q, rhs, ue) nocudafree(buf, cuf, grid_points, lhs, q, rhs, u, ue) 
#pragma cuda gpurun nocudamalloc(grid_points, lhs, rhs) 
#pragma cuda ainfo kernelid(2) procname(z_solve_cell_clnd1) 
	z_solve_cell_clnd1_kernel2<<<dimGrid2, dimBlock2, 0, 0>>>(gpu__grid_points, ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][3][5][5])gpu__lhs), ((double (*)[(((162/2)*2)+1)][(((162/2)*2)+1)][5])gpu__rhs));
	return ;
}

