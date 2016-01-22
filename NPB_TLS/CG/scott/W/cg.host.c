//-------------------------------------------------------------------------------
//Host code 
//Generated at : Thu Oct 25 14:33:26 2012
//Compiler options: 
//      Software Cache  true
//      Local Memory    true
//      DefaultParallelDepth    3
//      UserDefParallelDepth    false
//      EnableLoopInterchange   true
//      Generating debug/profiling code false
//      EnableMLFeatureCollection       false
//      Array Linearization     false
//      GPU TLs true
//      Strict TLS Checking     true
//      Check TLS Conflict at the end of function       true
//      Use OCL TLS     false
//-------------------------------------------------------------------------------

#include "npb-C.h"
#include "npbparams.h"
#include "sys/time.h"
#include "ocldef.h"

static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static int colidx[637001];
static int rowstr[7002];
static int iv[14002];
static int arow[637001];
static int acol[637001];
static double v[7002];
static double aelt[637001];
static double a[637001];
static double x[7003];
static double z[7003];
static double p[7003];
static double q[7003];
static double r[7003];
static double w[7003];
static double amult;
static double tran;
static double rnorm;
static void conj_grad();
static void makea(int n, int nz, double a[], ocl_buffer * __ocl_buffer_a,
		  int colidx[], ocl_buffer * __ocl_buffer_colidx, int rowstr[],
		  ocl_buffer * __ocl_buffer_rowstr, int nonzer, int firstrow,
		  int lastrow, int firstcol, int lastcol, double rcond,
		  int arow[], ocl_buffer * __ocl_buffer_arow, int acol[],
		  ocl_buffer * __ocl_buffer_acol, double aelt[],
		  ocl_buffer * __ocl_buffer_aelt, double v[],
		  ocl_buffer * __ocl_buffer_v, int iv[],
		  ocl_buffer * __ocl_buffer_iv, double shift);
static void sparse(double a[], ocl_buffer * __ocl_buffer_a, int colidx[],
		   ocl_buffer * __ocl_buffer_colidx, int rowstr[],
		   ocl_buffer * __ocl_buffer_rowstr, int n, int arow[],
		   ocl_buffer * __ocl_buffer_arow, int acol[],
		   ocl_buffer * __ocl_buffer_acol, double aelt[],
		   ocl_buffer * __ocl_buffer_aelt, int firstrow, int lastrow,
		   double x[], ocl_buffer * __ocl_buffer_x, boolean mark[],
		   ocl_buffer * __ocl_buffer_mark, int nzloc[],
		   ocl_buffer * __ocl_buffer_nzloc, int nnza);
static void sprnvc(int n, int nz, double v[], ocl_buffer * __ocl_buffer_v,
		   int iv[], ocl_buffer * __ocl_buffer_iv, int nzloc[],
		   ocl_buffer * __ocl_buffer_nzloc, int mark[],
		   ocl_buffer * __ocl_buffer_mark);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], ocl_buffer * __ocl_buffer_v, int iv[],
		   ocl_buffer * __ocl_buffer_iv, int *nzv,
		   ocl_buffer * __ocl_buffer_nzv, int i, double val);
double randlc(double *x, ocl_buffer * __ocl_buffer_x, double a)
{
	{
		double t1, t2, t3, t4, a1, a2, x1, x2, z;
		t1 = (0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5) * a;
		a1 = (int)t1;
		a2 = a -
		    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0) * a1;
		t1 = (0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5) * (*x);
		x1 = (int)t1;
		x2 = (*x) -
		    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0) * x1;
		t1 = a1 * x2 + a2 * x1;
		t2 = (int)((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			    0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			    0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5) * t1);
		z = t1 -
		    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0) * t2;
		t3 = (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		      2.0 * 2.0 * 2.0 * 2.0 * 2.0) * z + a2 * x2;
		t4 = (int)(((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			     0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			     0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5) * (0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5 *
									 0.5)) *
			   t3);
		(*x) =
		    t3 -
		    ((2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		      2.0 * 2.0 * 2.0 * 2.0 * 2.0) * (2.0 * 2.0 * 2.0 * 2.0 *
						      2.0 * 2.0 * 2.0 * 2.0 *
						      2.0 * 2.0 * 2.0 * 2.0 *
						      2.0 * 2.0 * 2.0 * 2.0 *
						      2.0 * 2.0 * 2.0 * 2.0 *
						      2.0 * 2.0 * 2.0)) * t4;
		return (((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			  0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			  0.5 * 0.5 * 0.5 * 0.5 * 0.5) * (0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5 * 0.5 *
							  0.5 * 0.5)) * (*x));
	}
}

void vranlc(int n, double *x_seed, ocl_buffer * __ocl_buffer_x_seed, double a,
	    double y[], ocl_buffer * __ocl_buffer_y)
{
	{
		int i;
		double x, t1, t2, t3, t4, a1, a2, x1, x2, z;
		t1 = (0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
		      0.5 * 0.5 * 0.5 * 0.5 * 0.5) * a;
		a1 = (int)t1;
		a2 = a -
		    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
		     2.0 * 2.0 * 2.0) * a1;
		x = *x_seed;
		for (i = 1; i <= n; i++) {
			t1 = (0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5) * x;
			x1 = (int)t1;
			x2 = x -
			    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0) * x1;
			t1 = a1 * x2 + a2 * x1;
			t2 = (int)((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				    0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				    0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				    0.5 * 0.5) * t1);
			z = t1 -
			    (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			     2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0) * t2;
			t3 = (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0) * z +
			    a2 * x2;
			t4 = (int)(((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				     0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				     0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
				     0.5 * 0.5) * (0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
						   0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
						   0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
						   0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
						   0.5 * 0.5 * 0.5)) * t3);
			x = t3 -
			    ((2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 *
			      2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0) * (2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0 *
									  2.0))
			    * t4;
			y[i] =
			    ((0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 *
			      0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5) * (0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5 *
									  0.5))
			    * x;
		}
		*x_seed = x;
	}
}

int main(int argc, char **argv, ocl_buffer * __ocl_buffer_argv)
{
	{
		init_ocl_runtime();
		int i, j, k, it;
		int nthreads = 1;
		double zeta;
		double norm_temp11;
		double norm_temp12;
		double t, mflops;
		char class;
		boolean verified;
		double zeta_verify_value, epsilon;
		struct timeval t1, t2;
		firstrow = 1;
		lastrow = 7000;
		firstcol = 1;
		lastcol = 7000;
		if (7000 == 1400 && 8 == 7 && 15 == 15 && 12.0 == 10.0) {
			class = 'S';
			zeta_verify_value = 8.5971775078648;
		} else if (7000 == 7000 && 8 == 8 && 15 == 15 && 12.0 == 12.0) {
			class = 'W';
			zeta_verify_value = 10.362595087124;
		} else if (7000 == 14000 && 8 == 11 && 15 == 15 && 12.0 == 20.0) {
			class = 'A';
			zeta_verify_value = 17.130235054029;
		} else if (7000 == 75000 && 8 == 13 && 15 == 75 && 12.0 == 60.0) {
			class = 'B';
			zeta_verify_value = 22.712745482631;
		} else if (7000 == 150000 && 8 == 15 && 15 == 75
			   && 12.0 == 110.0) {
			class = 'C';
			zeta_verify_value = 28.973605592845;
		} else {
			class = 'U';
		}
		printf
		    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - CG Benchmark\n");
		printf(" Size: %10d\n", 7000);
		printf(" Iterations: %5d\n", 15);
		naa = 7000;
		nzz = 7000 * (8 + 1) * (8 + 1) + 7000 * (8 + 2);
		tran = 314159265.0;
		amult = 1220703125.0;
		zeta = randlc(&tran, NULL, amult);
		makea(naa, nzz, a, __ocl_buffer_a, colidx, __ocl_buffer_colidx,
		      rowstr, __ocl_buffer_rowstr, 8, firstrow, lastrow,
		      firstcol, lastcol, 1.0e-1, arow, NULL, acol, NULL, aelt,
		      NULL, v, NULL, iv, NULL, 12.0);
		{
			//--------------------------------------------------------------
			//Loop defined at line 313 of cg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] =
				    (lastrow - firstrow + 1) - (1) + 1;
				_ocl_gws[1] = (rowstr[j + 1]) - (rowstr[j]);

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_main_0, 0,
						      __ocl_buffer_colidx);
				oclSetKernelArg(__ocl_main_0, 1, sizeof(int),
						&firstcol);
				oclSetKernelArgBuffer(__ocl_main_0, 2,
						      __ocl_buffer_rowstr);
				int __ocl_j_bound = lastrow - firstrow + 1;
				oclSetKernelArg(__ocl_main_0, 3, sizeof(int),
						&__ocl_j_bound);
				int __ocl_k_bound = rowstr[j + 1];
				oclSetKernelArg(__ocl_main_0, 4, sizeof(int),
						&__ocl_k_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_colidx);
				oclDevWrites(__ocl_buffer_rowstr);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_main_0, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
				oclSync();
#endif
			}

			//--------------------------------------------------------------
			//Loop defined at line 323 of cg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (7000 + 1) - (1) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_main_1, 0,
						      __ocl_buffer_x);
				int __ocl_i_bound = 7000 + 1;
				oclSetKernelArg(__ocl_main_1, 1, sizeof(int),
						&__ocl_i_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_x);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_main_1, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
				oclSync();
#endif
			}

			zeta = 0.0;
			for (it = 1; it <= 1; it++) {
				conj_grad();
				{
					norm_temp11 = 0.0;
					norm_temp12 = 0.0;
				}
				//--------------------------------------------------------------
				//Loop defined at line 355 of cg.c
				//--------------------------------------------------------------
				{
					//------------------------------------------
					//Reduction step 1
					//------------------------------------------
					size_t _ocl_gws[1];
					_ocl_gws[0] =
					    (lastcol - firstcol + 1) - (1) + 1;

					oclGetWorkSize(1, _ocl_gws, NULL);
					size_t __ocl_act_buf_size =
					    (_ocl_gws[0]);
					REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: norm_temp11
					CREATE_REDUCTION_STEP1_BUFFER
					    (__ocl_buffer_norm_temp11_main_2_size,
					     __ocl_buf_size,
					     __ocl_buffer_norm_temp11_main_2,
					     double);
//Prepare buffer for the reduction variable: norm_temp12
					CREATE_REDUCTION_STEP1_BUFFER
					    (__ocl_buffer_norm_temp12_main_2_size,
					     __ocl_buf_size,
					     __ocl_buffer_norm_temp12_main_2,
					     double);

					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
					if (__ocl_buf_size > __ocl_act_buf_size) {
						oclSetKernelArgBuffer
						    (__ocl_main_2_reduction_step0,
						     0,
						     __ocl_buffer_norm_temp11_main_2);
						oclSetKernelArgBuffer
						    (__ocl_main_2_reduction_step0,
						     1,
						     __ocl_buffer_norm_temp12_main_2);
						unsigned int __ocl_buffer_offset
						    =
						    __ocl_buf_size -
						    __ocl_act_buf_size;
						oclSetKernelArg
						    (__ocl_main_2_reduction_step0,
						     2, sizeof(unsigned int),
						     &__ocl_act_buf_size);
						oclSetKernelArg
						    (__ocl_main_2_reduction_step0,
						     3, sizeof(unsigned int),
						     &__ocl_buffer_offset);

						size_t __offset_work_size[1] =
						    { __ocl_buffer_offset };
						oclRunKernel
						    (__ocl_main_2_reduction_step0,
						     1, __offset_work_size);
					}

					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step1, 0,
					     __ocl_buffer_x);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step1, 1,
					     __ocl_buffer_z);
					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
					int __ocl_j_bound =
					    lastcol - firstcol + 1;
					oclSetKernelArg
					    (__ocl_main_2_reduction_step1, 2,
					     sizeof(int), &__ocl_j_bound);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step1, 3,
					     __ocl_buffer_norm_temp11_main_2);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step1, 4,
					     __ocl_buffer_norm_temp12_main_2);
					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//Write set (BEGIN) 
					//------------------------------------------
					//------------------------------------------
					//Write set (END) 
					//------------------------------------------
					//------------------------------------------
					//Read only buffers (BEGIN) 
					//------------------------------------------
					oclDevReads(__ocl_buffer_x);
					oclDevReads(__ocl_buffer_z);
					//------------------------------------------
					//Read only buffers (END) 
					//------------------------------------------

					oclRunKernel
					    (__ocl_main_2_reduction_step1, 1,
					     _ocl_gws);

//Reduction Step 2
					unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
					CREATE_REDUCTION_STEP2_BUFFER
					    (__ocl_output_norm_temp11_main_2_size,
					     __ocl_num_block, 16,
					     __ocl_output_buffer_norm_temp11_main_2,
					     __ocl_output_norm_temp11_main_2,
					     double);
					CREATE_REDUCTION_STEP2_BUFFER
					    (__ocl_output_norm_temp12_main_2_size,
					     __ocl_num_block, 16,
					     __ocl_output_buffer_norm_temp12_main_2,
					     __ocl_output_norm_temp12_main_2,
					     double);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step2, 0,
					     __ocl_buffer_norm_temp11_main_2);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step2, 1,
					     __ocl_output_buffer_norm_temp11_main_2);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step2, 2,
					     __ocl_buffer_norm_temp12_main_2);
					oclSetKernelArgBuffer
					    (__ocl_main_2_reduction_step2, 3,
					     __ocl_output_buffer_norm_temp12_main_2);

					oclDevWrites
					    (__ocl_output_buffer_norm_temp11_main_2);
					oclDevWrites
					    (__ocl_output_buffer_norm_temp12_main_2);

					size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
					size_t __ocl_localThreads[] =
					    { GROUP_SIZE };

					oclRunKernelL
					    (__ocl_main_2_reduction_step2, 1,
					     __ocl_globalThreads,
					     __ocl_localThreads);

//Do the final reduction part on the CPU
					oclHostReads
					    (__ocl_output_buffer_norm_temp11_main_2);
					oclHostReads
					    (__ocl_output_buffer_norm_temp12_main_2);
					oclSync();

					for (unsigned __ocl_i = 0;
					     __ocl_i < __ocl_num_block;
					     __ocl_i++) {
						norm_temp11 =
						    norm_temp11 +
						    __ocl_output_norm_temp11_main_2
						    [__ocl_i];
						norm_temp12 =
						    norm_temp12 +
						    __ocl_output_norm_temp12_main_2
						    [__ocl_i];
					}

				}

				norm_temp12 = 1.0 / sqrt(norm_temp12);
				//--------------------------------------------------------------
				//Loop defined at line 366 of cg.c
				//--------------------------------------------------------------
				{
					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
					size_t _ocl_gws[1];
					_ocl_gws[0] =
					    (lastcol - firstcol + 1) - (1) + 1;

					oclGetWorkSize(1, _ocl_gws, NULL);
					oclSetKernelArgBuffer(__ocl_main_3, 0,
							      __ocl_buffer_x);
					oclSetKernelArg(__ocl_main_3, 1,
							sizeof(double),
							&norm_temp12);
					oclSetKernelArgBuffer(__ocl_main_3, 2,
							      __ocl_buffer_z);
					int __ocl_j_bound =
					    lastcol - firstcol + 1;
					oclSetKernelArg(__ocl_main_3, 3,
							sizeof(int),
							&__ocl_j_bound);
					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//Write set (BEGIN) 
					//------------------------------------------
					oclDevWrites(__ocl_buffer_x);
					//------------------------------------------
					//Write set (END) 
					//------------------------------------------
					//------------------------------------------
					//Read only variables (BEGIN) 
					//------------------------------------------
					oclDevReads(__ocl_buffer_z);
					//------------------------------------------
					//Read only variables (END) 
					//------------------------------------------

					oclRunKernel(__ocl_main_3, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
					oclSync();
#endif
				}

			}
			//--------------------------------------------------------------
			//Loop defined at line 376 of cg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (7000 + 1) - (1) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_main_4, 0,
						      __ocl_buffer_x);
				int __ocl_i_bound = 7000 + 1;
				oclSetKernelArg(__ocl_main_4, 1, sizeof(int),
						&__ocl_i_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_x);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_main_4, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
				oclSync();
#endif
			}

			zeta = 0.0;
		}
		timer_clear(1);
		timer_start(1);
		gettimeofday(&t1, ((void *)0));
		{
			for (it = 1; it <= 15; it++) {
				conj_grad(colidx, rowstr, x, z, a, p, q, r, w,
					  &rnorm);
				{
					norm_temp11 = 0.0;
					norm_temp12 = 0.0;
				}
				//--------------------------------------------------------------
				//Loop defined at line 416 of cg.c
				//--------------------------------------------------------------
				{
					//------------------------------------------
					//Reduction step 1
					//------------------------------------------
					size_t _ocl_gws[1];
					_ocl_gws[0] =
					    (lastcol - firstcol + 1) - (1) + 1;

					oclGetWorkSize(1, _ocl_gws, NULL);
					size_t __ocl_act_buf_size =
					    (_ocl_gws[0]);
					REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: norm_temp11
					CREATE_REDUCTION_STEP1_BUFFER
					    (__ocl_buffer_norm_temp11_main_5_size,
					     __ocl_buf_size,
					     __ocl_buffer_norm_temp11_main_5,
					     double);
//Prepare buffer for the reduction variable: norm_temp12
					CREATE_REDUCTION_STEP1_BUFFER
					    (__ocl_buffer_norm_temp12_main_5_size,
					     __ocl_buf_size,
					     __ocl_buffer_norm_temp12_main_5,
					     double);

					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
					if (__ocl_buf_size > __ocl_act_buf_size) {
						oclSetKernelArgBuffer
						    (__ocl_main_5_reduction_step0,
						     0,
						     __ocl_buffer_norm_temp11_main_5);
						oclSetKernelArgBuffer
						    (__ocl_main_5_reduction_step0,
						     1,
						     __ocl_buffer_norm_temp12_main_5);
						unsigned int __ocl_buffer_offset
						    =
						    __ocl_buf_size -
						    __ocl_act_buf_size;
						oclSetKernelArg
						    (__ocl_main_5_reduction_step0,
						     2, sizeof(unsigned int),
						     &__ocl_act_buf_size);
						oclSetKernelArg
						    (__ocl_main_5_reduction_step0,
						     3, sizeof(unsigned int),
						     &__ocl_buffer_offset);

						size_t __offset_work_size[1] =
						    { __ocl_buffer_offset };
						oclRunKernel
						    (__ocl_main_5_reduction_step0,
						     1, __offset_work_size);
					}

					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step1, 0,
					     __ocl_buffer_x);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step1, 1,
					     __ocl_buffer_z);
					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
					int __ocl_j_bound =
					    lastcol - firstcol + 1;
					oclSetKernelArg
					    (__ocl_main_5_reduction_step1, 2,
					     sizeof(int), &__ocl_j_bound);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step1, 3,
					     __ocl_buffer_norm_temp11_main_5);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step1, 4,
					     __ocl_buffer_norm_temp12_main_5);
					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//Write set (BEGIN) 
					//------------------------------------------
					//------------------------------------------
					//Write set (END) 
					//------------------------------------------
					//------------------------------------------
					//Read only buffers (BEGIN) 
					//------------------------------------------
					oclDevReads(__ocl_buffer_x);
					oclDevReads(__ocl_buffer_z);
					//------------------------------------------
					//Read only buffers (END) 
					//------------------------------------------

					oclRunKernel
					    (__ocl_main_5_reduction_step1, 1,
					     _ocl_gws);

//Reduction Step 2
					unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
					CREATE_REDUCTION_STEP2_BUFFER
					    (__ocl_output_norm_temp11_main_5_size,
					     __ocl_num_block, 16,
					     __ocl_output_buffer_norm_temp11_main_5,
					     __ocl_output_norm_temp11_main_5,
					     double);
					CREATE_REDUCTION_STEP2_BUFFER
					    (__ocl_output_norm_temp12_main_5_size,
					     __ocl_num_block, 16,
					     __ocl_output_buffer_norm_temp12_main_5,
					     __ocl_output_norm_temp12_main_5,
					     double);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step2, 0,
					     __ocl_buffer_norm_temp11_main_5);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step2, 1,
					     __ocl_output_buffer_norm_temp11_main_5);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step2, 2,
					     __ocl_buffer_norm_temp12_main_5);
					oclSetKernelArgBuffer
					    (__ocl_main_5_reduction_step2, 3,
					     __ocl_output_buffer_norm_temp12_main_5);

					oclDevWrites
					    (__ocl_output_buffer_norm_temp11_main_5);
					oclDevWrites
					    (__ocl_output_buffer_norm_temp12_main_5);

					size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
					size_t __ocl_localThreads[] =
					    { GROUP_SIZE };

					oclRunKernelL
					    (__ocl_main_5_reduction_step2, 1,
					     __ocl_globalThreads,
					     __ocl_localThreads);

//Do the final reduction part on the CPU
					oclHostReads
					    (__ocl_output_buffer_norm_temp11_main_5);
					oclHostReads
					    (__ocl_output_buffer_norm_temp12_main_5);
					oclSync();

					for (unsigned __ocl_i = 0;
					     __ocl_i < __ocl_num_block;
					     __ocl_i++) {
						norm_temp11 =
						    norm_temp11 +
						    __ocl_output_norm_temp11_main_5
						    [__ocl_i];
						norm_temp12 =
						    norm_temp12 +
						    __ocl_output_norm_temp12_main_5
						    [__ocl_i];
					}

				}

				{
					norm_temp12 = 1.0 / sqrt(norm_temp12);
					zeta = 12.0 + 1.0 / norm_temp11;
				}
				{
					if (it == 1) {
						printf
						    ("   iteration           ||r||                 zeta\n");
					}
					printf("    %5d       %20.14e%20.13e\n",
					       it, rnorm, zeta);
				}
				//--------------------------------------------------------------
				//Loop defined at line 440 of cg.c
				//--------------------------------------------------------------
				{
					//------------------------------------------
					//OpenCL kernel arguments (BEGIN) 
					//------------------------------------------
					size_t _ocl_gws[1];
					_ocl_gws[0] =
					    (lastcol - firstcol + 1) - (1) + 1;

					oclGetWorkSize(1, _ocl_gws, NULL);
					oclSetKernelArgBuffer(__ocl_main_6, 0,
							      __ocl_buffer_x);
					oclSetKernelArg(__ocl_main_6, 1,
							sizeof(double),
							&norm_temp12);
					oclSetKernelArgBuffer(__ocl_main_6, 2,
							      __ocl_buffer_z);
					int __ocl_j_bound =
					    lastcol - firstcol + 1;
					oclSetKernelArg(__ocl_main_6, 3,
							sizeof(int),
							&__ocl_j_bound);
					//------------------------------------------
					//OpenCL kernel arguments (END) 
					//------------------------------------------

					//------------------------------------------
					//Write set (BEGIN) 
					//------------------------------------------
					oclDevWrites(__ocl_buffer_x);
					//------------------------------------------
					//Write set (END) 
					//------------------------------------------
					//------------------------------------------
					//Read only variables (BEGIN) 
					//------------------------------------------
					oclDevReads(__ocl_buffer_z);
					//------------------------------------------
					//Read only variables (END) 
					//------------------------------------------

					oclRunKernel(__ocl_main_6, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
					oclSync();
#endif
				}

			}
		}
		gettimeofday(&t2, ((void *)0));
		timer_stop(1);
		t = timer_read(1);
		t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) * 1E-06;
		printf(" Benchmark completed\n");
		epsilon = 1.0e-10;
		if (class != 'U') {
			if (fabs(zeta - zeta_verify_value) <= epsilon) {
				verified = 1;
				printf(" VERIFICATION SUCCESSFUL\n");
				printf(" Zeta is    %20.12e\n", zeta);
				printf(" Error is   %20.12e\n",
				       zeta - zeta_verify_value);
			} else {
				verified = 0;
				printf(" VERIFICATION FAILED\n");
				printf(" Zeta                %20.12e\n", zeta);
				printf(" The correct zeta is %20.12e\n",
				       zeta_verify_value);
			}
		} else {
			verified = 0;
			printf(" Problem size unknown\n");
			printf(" NO VERIFICATION PERFORMED\n");
		}
		if (t != 0.0) {
			mflops =
			    (2.0 * 15 * 7000) * (3.0 + (8 * (8 + 1)) +
						 25.0 * (5.0 + (8 * (8 + 1))) +
						 3.0) / t / 1000000.0;
		} else {
			mflops = 0.0;
		}
		c_print_results("CG", class, 7000, 0, 0, 15, nthreads, t,
				mflops, "          floating point", verified,
				"2.3", "25 Oct 2012", "gcc -fopenmp",
				"gcc -fopenmp", "-lm", "-I../common", "-O3 ",
				"(none)", "randdp");
	}
}

static void conj_grad()
{
	static double d, sum, rho, rho0, alpha, beta;
	int i, j, k;
	int cgit, cgitmax = 25;
	rho = 0.0;
	//--------------------------------------------------------------
	//Loop defined at line 518 of cg.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (naa + 1) - (1) + 1;

		oclGetWorkSize(1, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 0, __ocl_buffer_q);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 1, __ocl_buffer_z);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 2, __ocl_buffer_r);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 3, __ocl_buffer_x);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 4, __ocl_buffer_p);
		oclSetKernelArgBuffer(__ocl_conj_grad_0, 5, __ocl_buffer_w);
		int __ocl_j_bound = naa + 1;
		oclSetKernelArg(__ocl_conj_grad_0, 6, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_q);
		oclDevWrites(__ocl_buffer_z);
		oclDevWrites(__ocl_buffer_r);
		oclDevWrites(__ocl_buffer_p);
		oclDevWrites(__ocl_buffer_w);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_x);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_conj_grad_0, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 531 of cg.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//Reduction step 1
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

		oclGetWorkSize(1, _ocl_gws, NULL);
		size_t __ocl_act_buf_size = (_ocl_gws[0]);
		REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: rho
		CREATE_REDUCTION_STEP1_BUFFER(__ocl_buffer_rho_conj_grad_1_size,
					      __ocl_buf_size,
					      __ocl_buffer_rho_conj_grad_1,
					      double);

		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
		if (__ocl_buf_size > __ocl_act_buf_size) {
			oclSetKernelArgBuffer(__ocl_conj_grad_1_reduction_step0,
					      0, __ocl_buffer_rho_conj_grad_1);
			unsigned int __ocl_buffer_offset =
			    __ocl_buf_size - __ocl_act_buf_size;
			oclSetKernelArg(__ocl_conj_grad_1_reduction_step0, 1,
					sizeof(unsigned int),
					&__ocl_act_buf_size);
			oclSetKernelArg(__ocl_conj_grad_1_reduction_step0, 2,
					sizeof(unsigned int),
					&__ocl_buffer_offset);

			size_t __offset_work_size[1] = { __ocl_buffer_offset };
			oclRunKernel(__ocl_conj_grad_1_reduction_step0, 1,
				     __offset_work_size);
		}

		oclSetKernelArgBuffer(__ocl_conj_grad_1_reduction_step1, 0,
				      __ocl_buffer_x);
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		int __ocl_j_bound = lastcol - firstcol + 1;
		oclSetKernelArg(__ocl_conj_grad_1_reduction_step1, 1,
				sizeof(int), &__ocl_j_bound);
		oclSetKernelArgBuffer(__ocl_conj_grad_1_reduction_step1, 2,
				      __ocl_buffer_rho_conj_grad_1);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only buffers (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_x);
		//------------------------------------------
		//Read only buffers (END) 
		//------------------------------------------

		oclRunKernel(__ocl_conj_grad_1_reduction_step1, 1, _ocl_gws);

//Reduction Step 2
		unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
		CREATE_REDUCTION_STEP2_BUFFER(__ocl_output_rho_conj_grad_1_size,
					      __ocl_num_block, 16,
					      __ocl_output_buffer_rho_conj_grad_1,
					      __ocl_output_rho_conj_grad_1,
					      double);
		oclSetKernelArgBuffer(__ocl_conj_grad_1_reduction_step2, 0,
				      __ocl_buffer_rho_conj_grad_1);
		oclSetKernelArgBuffer(__ocl_conj_grad_1_reduction_step2, 1,
				      __ocl_output_buffer_rho_conj_grad_1);

		oclDevWrites(__ocl_output_buffer_rho_conj_grad_1);

		size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
		size_t __ocl_localThreads[] = { GROUP_SIZE };

		oclRunKernelL(__ocl_conj_grad_1_reduction_step2, 1,
			      __ocl_globalThreads, __ocl_localThreads);

//Do the final reduction part on the CPU
		oclHostReads(__ocl_output_buffer_rho_conj_grad_1);
		oclSync();

		for (unsigned __ocl_i = 0; __ocl_i < __ocl_num_block; __ocl_i++) {
			rho = rho + __ocl_output_rho_conj_grad_1[__ocl_i];
		}

	}

	for (cgit = 1; cgit <= cgitmax; cgit++) {
		{
			rho0 = rho;
			d = 0.0;
			rho = 0.0;
		}
		//--------------------------------------------------------------
		//Loop defined at line 563 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastrow - firstrow + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_conj_grad_2, 0,
					      __ocl_buffer_rowstr);
			oclSetKernelArgBuffer(__ocl_conj_grad_2, 1,
					      __ocl_buffer_a);
			oclSetKernelArgBuffer(__ocl_conj_grad_2, 2,
					      __ocl_buffer_p);
			oclSetKernelArgBuffer(__ocl_conj_grad_2, 3,
					      __ocl_buffer_colidx);
			oclSetKernelArgBuffer(__ocl_conj_grad_2, 4,
					      __ocl_buffer_w);
			int __ocl_j_bound = lastrow - firstrow + 1;
			oclSetKernelArg(__ocl_conj_grad_2, 5, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_w);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_rowstr);
			oclDevReads(__ocl_buffer_a);
			oclDevReads(__ocl_buffer_p);
			oclDevReads(__ocl_buffer_colidx);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_2, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
			//---------------------------------------
			// GPU TLS Checking (BEGIN)
			//---------------------------------------
			{
#ifdef __RUN_CHECKING_KERNEL__
//Checking w
				{
					size_t __ocl_gws_w = (7003);
					oclSetKernelArg(__ocl_tls_1D_checking,
							0, sizeof(unsigned),
							&__ocl_gws_w);
					oclSetKernelArgBuffer
					    (__ocl_tls_1D_checking, 1,
					     rd_oclb_w);
					oclSetKernelArgBuffer
					    (__ocl_tls_1D_checking, 2,
					     wr_oclb_w);
					oclSetKernelArgBuffer
					    (__ocl_tls_1D_checking, 3,
					     __oclb_gpu_tls_conflict_flag);
					oclDevWrites(wr_oclb_w);
					oclDevWrites(rd_oclb_w);
					oclDevWrites
					    (__oclb_gpu_tls_conflict_flag);
					oclRunKernel(__ocl_tls_1D_checking, 1,
						     &__ocl_gws_w);
				}
#endif

				oclHostReads(__oclb_gpu_tls_conflict_flag);
				oclSync();
#ifdef __DUMP_TLS_CONFLICT__
				if (gpu_tls_conflict_flag) {
					fprintf(stderr, "conflict detected.\n");
				}
#endif
			}
			//---------------------------------------
			// GPU TLS Checking (END)
			//---------------------------------------
		}

		//--------------------------------------------------------------
		//Loop defined at line 572 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_conj_grad_3, 0,
					      __ocl_buffer_q);
			oclSetKernelArgBuffer(__ocl_conj_grad_3, 1,
					      __ocl_buffer_w);
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_3, 2, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_q);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_w);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_3, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		//--------------------------------------------------------------
		//Loop defined at line 580 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_conj_grad_4, 0,
					      __ocl_buffer_w);
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_4, 1, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_w);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_4, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		//--------------------------------------------------------------
		//Loop defined at line 588 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//Reduction step 1
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			size_t __ocl_act_buf_size = (_ocl_gws[0]);
			REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: d
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_d_conj_grad_5_size, __ocl_buf_size,
			     __ocl_buffer_d_conj_grad_5, double);

			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
			if (__ocl_buf_size > __ocl_act_buf_size) {
				oclSetKernelArgBuffer
				    (__ocl_conj_grad_5_reduction_step0, 0,
				     __ocl_buffer_d_conj_grad_5);
				unsigned int __ocl_buffer_offset =
				    __ocl_buf_size - __ocl_act_buf_size;
				oclSetKernelArg
				    (__ocl_conj_grad_5_reduction_step0, 1,
				     sizeof(unsigned int), &__ocl_act_buf_size);
				oclSetKernelArg
				    (__ocl_conj_grad_5_reduction_step0, 2,
				     sizeof(unsigned int),
				     &__ocl_buffer_offset);

				size_t __offset_work_size[1] =
				    { __ocl_buffer_offset };
				oclRunKernel(__ocl_conj_grad_5_reduction_step0,
					     1, __offset_work_size);
			}

			oclSetKernelArgBuffer(__ocl_conj_grad_5_reduction_step1,
					      0, __ocl_buffer_p);
			oclSetKernelArgBuffer(__ocl_conj_grad_5_reduction_step1,
					      1, __ocl_buffer_q);
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_5_reduction_step1, 2,
					sizeof(int), &__ocl_j_bound);
			oclSetKernelArgBuffer(__ocl_conj_grad_5_reduction_step1,
					      3, __ocl_buffer_d_conj_grad_5);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only buffers (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_p);
			oclDevReads(__ocl_buffer_q);
			//------------------------------------------
			//Read only buffers (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_5_reduction_step1, 1,
				     _ocl_gws);

//Reduction Step 2
			unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_d_conj_grad_5_size, __ocl_num_block,
			     16, __ocl_output_buffer_d_conj_grad_5,
			     __ocl_output_d_conj_grad_5, double);
			oclSetKernelArgBuffer(__ocl_conj_grad_5_reduction_step2,
					      0, __ocl_buffer_d_conj_grad_5);
			oclSetKernelArgBuffer(__ocl_conj_grad_5_reduction_step2,
					      1,
					      __ocl_output_buffer_d_conj_grad_5);

			oclDevWrites(__ocl_output_buffer_d_conj_grad_5);

			size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
			size_t __ocl_localThreads[] = { GROUP_SIZE };

			oclRunKernelL(__ocl_conj_grad_5_reduction_step2, 1,
				      __ocl_globalThreads, __ocl_localThreads);

//Do the final reduction part on the CPU
			oclHostReads(__ocl_output_buffer_d_conj_grad_5);
			oclSync();

			for (unsigned __ocl_i = 0; __ocl_i < __ocl_num_block;
			     __ocl_i++) {
				d = d + __ocl_output_d_conj_grad_5[__ocl_i];
			}

		}

		alpha = rho0 / d;
		//--------------------------------------------------------------
		//Loop defined at line 608 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_conj_grad_6, 0,
					      __ocl_buffer_z);
			oclSetKernelArg(__ocl_conj_grad_6, 1, sizeof(double),
					&alpha);
			oclSetKernelArgBuffer(__ocl_conj_grad_6, 2,
					      __ocl_buffer_p);
			oclSetKernelArgBuffer(__ocl_conj_grad_6, 3,
					      __ocl_buffer_r);
			oclSetKernelArgBuffer(__ocl_conj_grad_6, 4,
					      __ocl_buffer_q);
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_6, 5, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_z);
			oclDevWrites(__ocl_buffer_r);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_p);
			oclDevReads(__ocl_buffer_q);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_6, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		//--------------------------------------------------------------
		//Loop defined at line 618 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//Reduction step 1
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			size_t __ocl_act_buf_size = (_ocl_gws[0]);
			REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: rho
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_rho_conj_grad_7_size, __ocl_buf_size,
			     __ocl_buffer_rho_conj_grad_7, double);

			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
			if (__ocl_buf_size > __ocl_act_buf_size) {
				oclSetKernelArgBuffer
				    (__ocl_conj_grad_7_reduction_step0, 0,
				     __ocl_buffer_rho_conj_grad_7);
				unsigned int __ocl_buffer_offset =
				    __ocl_buf_size - __ocl_act_buf_size;
				oclSetKernelArg
				    (__ocl_conj_grad_7_reduction_step0, 1,
				     sizeof(unsigned int), &__ocl_act_buf_size);
				oclSetKernelArg
				    (__ocl_conj_grad_7_reduction_step0, 2,
				     sizeof(unsigned int),
				     &__ocl_buffer_offset);

				size_t __offset_work_size[1] =
				    { __ocl_buffer_offset };
				oclRunKernel(__ocl_conj_grad_7_reduction_step0,
					     1, __offset_work_size);
			}

			oclSetKernelArgBuffer(__ocl_conj_grad_7_reduction_step1,
					      0, __ocl_buffer_r);
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_7_reduction_step1, 1,
					sizeof(int), &__ocl_j_bound);
			oclSetKernelArgBuffer(__ocl_conj_grad_7_reduction_step1,
					      2, __ocl_buffer_rho_conj_grad_7);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only buffers (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_r);
			//------------------------------------------
			//Read only buffers (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_7_reduction_step1, 1,
				     _ocl_gws);

//Reduction Step 2
			unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_rho_conj_grad_7_size, __ocl_num_block,
			     16, __ocl_output_buffer_rho_conj_grad_7,
			     __ocl_output_rho_conj_grad_7, double);
			oclSetKernelArgBuffer(__ocl_conj_grad_7_reduction_step2,
					      0, __ocl_buffer_rho_conj_grad_7);
			oclSetKernelArgBuffer(__ocl_conj_grad_7_reduction_step2,
					      1,
					      __ocl_output_buffer_rho_conj_grad_7);

			oclDevWrites(__ocl_output_buffer_rho_conj_grad_7);

			size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
			size_t __ocl_localThreads[] = { GROUP_SIZE };

			oclRunKernelL(__ocl_conj_grad_7_reduction_step2, 1,
				      __ocl_globalThreads, __ocl_localThreads);

//Do the final reduction part on the CPU
			oclHostReads(__ocl_output_buffer_rho_conj_grad_7);
			oclSync();

			for (unsigned __ocl_i = 0; __ocl_i < __ocl_num_block;
			     __ocl_i++) {
				rho =
				    rho + __ocl_output_rho_conj_grad_7[__ocl_i];
			}

		}

		beta = rho / rho0;
		//--------------------------------------------------------------
		//Loop defined at line 632 of cg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_conj_grad_8, 0,
					      __ocl_buffer_p);
			oclSetKernelArgBuffer(__ocl_conj_grad_8, 1,
					      __ocl_buffer_r);
			oclSetKernelArg(__ocl_conj_grad_8, 2, sizeof(double),
					&beta);
			int __ocl_j_bound = lastcol - firstcol + 1;
			oclSetKernelArg(__ocl_conj_grad_8, 3, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_p);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_r);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_conj_grad_8, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

	}
	sum = 0.0;
	//--------------------------------------------------------------
	//Loop defined at line 646 of cg.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (lastrow - firstrow + 1) - (1) + 1;

		oclGetWorkSize(1, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_conj_grad_9, 0,
				      __ocl_buffer_rowstr);
		oclSetKernelArgBuffer(__ocl_conj_grad_9, 1, __ocl_buffer_a);
		oclSetKernelArgBuffer(__ocl_conj_grad_9, 2, __ocl_buffer_z);
		oclSetKernelArgBuffer(__ocl_conj_grad_9, 3,
				      __ocl_buffer_colidx);
		oclSetKernelArgBuffer(__ocl_conj_grad_9, 4, __ocl_buffer_w);
		int __ocl_j_bound = lastrow - firstrow + 1;
		oclSetKernelArg(__ocl_conj_grad_9, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_w);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rowstr);
		oclDevReads(__ocl_buffer_a);
		oclDevReads(__ocl_buffer_z);
		oclDevReads(__ocl_buffer_colidx);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_conj_grad_9, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking w
			{
				size_t __ocl_gws_w = (7003);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_w);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_w);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_w);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_w);
				oclDevWrites(rd_oclb_w);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_w);
			}
#endif

			oclHostReads(__oclb_gpu_tls_conflict_flag);
			oclSync();
#ifdef __DUMP_TLS_CONFLICT__
			if (gpu_tls_conflict_flag) {
				fprintf(stderr, "conflict detected.\n");
			}
#endif
		}
		//---------------------------------------
		// GPU TLS Checking (END)
		//---------------------------------------
	}

	//--------------------------------------------------------------
	//Loop defined at line 655 of cg.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

		oclGetWorkSize(1, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_conj_grad_10, 0, __ocl_buffer_r);
		oclSetKernelArgBuffer(__ocl_conj_grad_10, 1, __ocl_buffer_w);
		int __ocl_j_bound = lastcol - firstcol + 1;
		oclSetKernelArg(__ocl_conj_grad_10, 2, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_r);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_w);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_conj_grad_10, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 663 of cg.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//Reduction step 1
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (lastcol - firstcol + 1) - (1) + 1;

		oclGetWorkSize(1, _ocl_gws, NULL);
		size_t __ocl_act_buf_size = (_ocl_gws[0]);
		REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: sum
		CREATE_REDUCTION_STEP1_BUFFER
		    (__ocl_buffer_sum_conj_grad_11_size, __ocl_buf_size,
		     __ocl_buffer_sum_conj_grad_11, double);

		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
		if (__ocl_buf_size > __ocl_act_buf_size) {
			oclSetKernelArgBuffer
			    (__ocl_conj_grad_11_reduction_step0, 0,
			     __ocl_buffer_sum_conj_grad_11);
			unsigned int __ocl_buffer_offset =
			    __ocl_buf_size - __ocl_act_buf_size;
			oclSetKernelArg(__ocl_conj_grad_11_reduction_step0, 1,
					sizeof(unsigned int),
					&__ocl_act_buf_size);
			oclSetKernelArg(__ocl_conj_grad_11_reduction_step0, 2,
					sizeof(unsigned int),
					&__ocl_buffer_offset);

			size_t __offset_work_size[1] = { __ocl_buffer_offset };
			oclRunKernel(__ocl_conj_grad_11_reduction_step0, 1,
				     __offset_work_size);
		}

		oclSetKernelArgBuffer(__ocl_conj_grad_11_reduction_step1, 0,
				      __ocl_buffer_x);
		oclSetKernelArgBuffer(__ocl_conj_grad_11_reduction_step1, 1,
				      __ocl_buffer_r);
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		int __ocl_j_bound = lastcol - firstcol + 1;
		oclSetKernelArg(__ocl_conj_grad_11_reduction_step1, 2,
				sizeof(int), &__ocl_j_bound);
		oclSetKernelArgBuffer(__ocl_conj_grad_11_reduction_step1, 3,
				      __ocl_buffer_sum_conj_grad_11);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only buffers (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_x);
		oclDevReads(__ocl_buffer_r);
		//------------------------------------------
		//Read only buffers (END) 
		//------------------------------------------

		oclRunKernel(__ocl_conj_grad_11_reduction_step1, 1, _ocl_gws);

//Reduction Step 2
		unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
		CREATE_REDUCTION_STEP2_BUFFER
		    (__ocl_output_sum_conj_grad_11_size, __ocl_num_block, 16,
		     __ocl_output_buffer_sum_conj_grad_11,
		     __ocl_output_sum_conj_grad_11, double);
		oclSetKernelArgBuffer(__ocl_conj_grad_11_reduction_step2, 0,
				      __ocl_buffer_sum_conj_grad_11);
		oclSetKernelArgBuffer(__ocl_conj_grad_11_reduction_step2, 1,
				      __ocl_output_buffer_sum_conj_grad_11);

		oclDevWrites(__ocl_output_buffer_sum_conj_grad_11);

		size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
		size_t __ocl_localThreads[] = { GROUP_SIZE };

		oclRunKernelL(__ocl_conj_grad_11_reduction_step2, 1,
			      __ocl_globalThreads, __ocl_localThreads);

//Do the final reduction part on the CPU
		oclHostReads(__ocl_output_buffer_sum_conj_grad_11);
		oclSync();

		for (unsigned __ocl_i = 0; __ocl_i < __ocl_num_block; __ocl_i++) {
			sum = sum + __ocl_output_sum_conj_grad_11[__ocl_i];
		}

	}

	{
		(rnorm) = sqrt(sum);
	}
}

static void makea(int n, int nz, double a[], ocl_buffer * __ocl_buffer_a,
		  int colidx[], ocl_buffer * __ocl_buffer_colidx, int rowstr[],
		  ocl_buffer * __ocl_buffer_rowstr, int nonzer, int firstrow,
		  int lastrow, int firstcol, int lastcol, double rcond,
		  int arow[], ocl_buffer * __ocl_buffer_arow, int acol[],
		  ocl_buffer * __ocl_buffer_acol, double aelt[],
		  ocl_buffer * __ocl_buffer_aelt, double v[],
		  ocl_buffer * __ocl_buffer_v, int iv[],
		  ocl_buffer * __ocl_buffer_iv, double shift)
{
	{
		int i, nnza, iouter, ivelt, ivelt1, irow, nzv;
		double size, ratio, scale;
		int jcol;
		size = 1.0;
		ratio = pow(rcond, (1.0 / (double)n));
		nnza = 0;
		for (i = 1; i <= n; i++) {
			colidx[n + i] = 0;
		}
		for (iouter = 1; iouter <= n; iouter++) {
			nzv = nonzer;
			sprnvc(n, nzv, v, __ocl_buffer_v, iv, __ocl_buffer_iv,
			       &(colidx[0]), __ocl_buffer_colidx, &(colidx[n]),
			       __ocl_buffer_colidx);
			vecset(n, v, __ocl_buffer_v, iv, __ocl_buffer_iv, &nzv,
			       NULL, iouter, 0.5);
			for (ivelt = 1; ivelt <= nzv; ivelt++) {
				jcol = iv[ivelt];
				if (jcol >= firstcol && jcol <= lastcol) {
					scale = size * v[ivelt];
					for (ivelt1 = 1; ivelt1 <= nzv;
					     ivelt1++) {
						irow = iv[ivelt1];
						if (irow >= firstrow
						    && irow <= lastrow) {
							nnza = nnza + 1;
							if (nnza > nz) {
								printf
								    ("Space for matrix elements exceeded in makea\n");
								printf
								    ("nnza, nzmax = %d, %d\n",
								     nnza, nz);
								printf
								    ("iouter = %d\n",
								     iouter);
								exit(1);
							}
							acol[nnza] = jcol;
							arow[nnza] = irow;
							aelt[nnza] =
							    v[ivelt1] * scale;
						}
					}
				}
			}
			size = size * ratio;
		}
		for (i = firstrow; i <= lastrow; i++) {
			if (i >= firstcol && i <= lastcol) {
				iouter = n + i;
				nnza = nnza + 1;
				if (nnza > nz) {
					printf
					    ("Space for matrix elements exceeded in makea\n");
					printf("nnza, nzmax = %d, %d\n", nnza,
					       nz);
					printf("iouter = %d\n", iouter);
					exit(1);
				}
				acol[nnza] = i;
				arow[nnza] = i;
				aelt[nnza] = rcond - shift;
			}
		}
		sparse(a, __ocl_buffer_a, colidx, __ocl_buffer_colidx, rowstr,
		       __ocl_buffer_rowstr, n, arow, __ocl_buffer_arow, acol,
		       __ocl_buffer_acol, aelt, __ocl_buffer_aelt, firstrow,
		       lastrow, v, __ocl_buffer_v, &(iv[0]), __ocl_buffer_iv,
		       &(iv[n]), __ocl_buffer_iv, nnza);
	}
}

static void sparse(double a[], ocl_buffer * __ocl_buffer_a, int colidx[],
		   ocl_buffer * __ocl_buffer_colidx, int rowstr[],
		   ocl_buffer * __ocl_buffer_rowstr, int n, int arow[],
		   ocl_buffer * __ocl_buffer_arow, int acol[],
		   ocl_buffer * __ocl_buffer_acol, double aelt[],
		   ocl_buffer * __ocl_buffer_aelt, int firstrow, int lastrow,
		   double x[], ocl_buffer * __ocl_buffer_x, boolean mark[],
		   ocl_buffer * __ocl_buffer_mark, int nzloc[],
		   ocl_buffer * __ocl_buffer_nzloc, int nnza)
{
	{
		int nrows;
		int i, j, jajp1, nza, k, nzrow;
		double xi;
		nrows = lastrow - firstrow + 1;
		for (j = 1; j <= n; j++) {
			rowstr[j] = 0;
			mark[j] = 0;
		}
		rowstr[n + 1] = 0;
		for (nza = 1; nza <= nnza; nza++) {
			j = (arow[nza] - firstrow + 1) + 1;
			rowstr[j] = rowstr[j] + 1;
		}
		rowstr[1] = 1;
		for (j = 2; j <= nrows + 1; j++) {
			rowstr[j] = rowstr[j] + rowstr[j - 1];
		}
		for (nza = 1; nza <= nnza; nza++) {
			j = arow[nza] - firstrow + 1;
			k = rowstr[j];
			a[k] = aelt[nza];
			colidx[k] = acol[nza];
			rowstr[j] = rowstr[j] + 1;
		}
		for (j = nrows; j >= 1; j--) {
			rowstr[j + 1] = rowstr[j];
		}
		rowstr[1] = 1;
		nza = 0;
		for (i = 1; i <= n; i++) {
			x[i] = 0.0;
			mark[i] = 0;
		}
		jajp1 = rowstr[1];
		for (j = 1; j <= nrows; j++) {
			nzrow = 0;
			for (k = jajp1; k < rowstr[j + 1]; k++) {
				i = colidx[k];
				x[i] = x[i] + a[k];
				if (mark[i] == 0 && x[i] != 0.0) {
					mark[i] = 1;
					nzrow = nzrow + 1;
					nzloc[nzrow] = i;
				}
			}
			for (k = 1; k <= nzrow; k++) {
				i = nzloc[k];
				mark[i] = 0;
				xi = x[i];
				x[i] = 0.0;
				if (xi != 0.0) {
					nza = nza + 1;
					a[nza] = xi;
					colidx[nza] = i;
				}
			}
			jajp1 = rowstr[j + 1];
			rowstr[j + 1] = nza + rowstr[1];
		}
	}
}

static void sprnvc(int n, int nz, double v[], ocl_buffer * __ocl_buffer_v,
		   int iv[], ocl_buffer * __ocl_buffer_iv, int nzloc[],
		   ocl_buffer * __ocl_buffer_nzloc, int mark[],
		   ocl_buffer * __ocl_buffer_mark)
{
	{
		int nn1;
		int nzrow, nzv, ii, i;
		double vecelt, vecloc;
		nzv = 0;
		nzrow = 0;
		nn1 = 1;
		do {
			nn1 = 2 * nn1;
		} while (nn1 < n);
		while (nzv < nz) {
			vecelt = randlc(&tran, NULL, amult);
			vecloc = randlc(&tran, NULL, amult);
			i = icnvrt(vecloc, nn1) + 1;
			if (i > n)
				continue;
			if (mark[i] == 0) {
				mark[i] = 1;
				nzrow = nzrow + 1;
				nzloc[nzrow] = i;
				nzv = nzv + 1;
				v[nzv] = vecelt;
				iv[nzv] = i;
			}
		}
		for (ii = 1; ii <= nzrow; ii++) {
			i = nzloc[ii];
			mark[i] = 0;
		}
	}
}

static int icnvrt(double x, int ipwr2)
{
	return ((int)(ipwr2 * x));
}

static void vecset(int n, double v[], ocl_buffer * __ocl_buffer_v, int iv[],
		   ocl_buffer * __ocl_buffer_iv, int *nzv,
		   ocl_buffer * __ocl_buffer_nzv, int i, double val)
{
	{
		int k;
		boolean set;
		set = 0;
		for (k = 1; k <= *nzv; k++) {
			if (iv[k] == i) {
				v[k] = val;
				set = 1;
			}
		}
		if (set == 0) {
			*nzv = *nzv + 1;
			v[*nzv] = val;
			iv[*nzv] = i;
		}
	}
}

//---------------------------------------------------------------------------
//OCL related routines (BEGIN)
//---------------------------------------------------------------------------

static void init_ocl_runtime()
{
	int err;

	if (unlikely(err = oclInit("AMD", 0))) {
		fprintf(stderr, "Failed to init ocl runtime:%d.\n", err);
		exit(err);
	}

	__ocl_program = oclBuildProgram("cg.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_main_0 = oclCreateKernel(__ocl_program, "main_0");
	DYN_PROGRAM_CHECK(__ocl_main_0);
	__ocl_main_1 = oclCreateKernel(__ocl_program, "main_1");
	DYN_PROGRAM_CHECK(__ocl_main_1);
	__ocl_main_2_reduction_step0 =
	    oclCreateKernel(__ocl_program, "main_2_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_main_2_reduction_step0);
	__ocl_main_2_reduction_step1 =
	    oclCreateKernel(__ocl_program, "main_2_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_main_2_reduction_step1);
	__ocl_main_2_reduction_step2 =
	    oclCreateKernel(__ocl_program, "main_2_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_main_2_reduction_step2);
	__ocl_main_3 = oclCreateKernel(__ocl_program, "main_3");
	DYN_PROGRAM_CHECK(__ocl_main_3);
	__ocl_main_4 = oclCreateKernel(__ocl_program, "main_4");
	DYN_PROGRAM_CHECK(__ocl_main_4);
	__ocl_main_5_reduction_step0 =
	    oclCreateKernel(__ocl_program, "main_5_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_main_5_reduction_step0);
	__ocl_main_5_reduction_step1 =
	    oclCreateKernel(__ocl_program, "main_5_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_main_5_reduction_step1);
	__ocl_main_5_reduction_step2 =
	    oclCreateKernel(__ocl_program, "main_5_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_main_5_reduction_step2);
	__ocl_main_6 = oclCreateKernel(__ocl_program, "main_6");
	DYN_PROGRAM_CHECK(__ocl_main_6);
	__ocl_conj_grad_0 = oclCreateKernel(__ocl_program, "conj_grad_0");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_0);
	__ocl_conj_grad_1_reduction_step0 =
	    oclCreateKernel(__ocl_program, "conj_grad_1_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_1_reduction_step0);
	__ocl_conj_grad_1_reduction_step1 =
	    oclCreateKernel(__ocl_program, "conj_grad_1_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_1_reduction_step1);
	__ocl_conj_grad_1_reduction_step2 =
	    oclCreateKernel(__ocl_program, "conj_grad_1_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_1_reduction_step2);
	__ocl_conj_grad_2 = oclCreateKernel(__ocl_program, "conj_grad_2");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_2);
	__ocl_conj_grad_3 = oclCreateKernel(__ocl_program, "conj_grad_3");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_3);
	__ocl_conj_grad_4 = oclCreateKernel(__ocl_program, "conj_grad_4");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_4);
	__ocl_conj_grad_5_reduction_step0 =
	    oclCreateKernel(__ocl_program, "conj_grad_5_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_5_reduction_step0);
	__ocl_conj_grad_5_reduction_step1 =
	    oclCreateKernel(__ocl_program, "conj_grad_5_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_5_reduction_step1);
	__ocl_conj_grad_5_reduction_step2 =
	    oclCreateKernel(__ocl_program, "conj_grad_5_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_5_reduction_step2);
	__ocl_conj_grad_6 = oclCreateKernel(__ocl_program, "conj_grad_6");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_6);
	__ocl_conj_grad_7_reduction_step0 =
	    oclCreateKernel(__ocl_program, "conj_grad_7_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_7_reduction_step0);
	__ocl_conj_grad_7_reduction_step1 =
	    oclCreateKernel(__ocl_program, "conj_grad_7_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_7_reduction_step1);
	__ocl_conj_grad_7_reduction_step2 =
	    oclCreateKernel(__ocl_program, "conj_grad_7_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_7_reduction_step2);
	__ocl_conj_grad_8 = oclCreateKernel(__ocl_program, "conj_grad_8");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_8);
	__ocl_conj_grad_9 = oclCreateKernel(__ocl_program, "conj_grad_9");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_9);
	__ocl_conj_grad_10 = oclCreateKernel(__ocl_program, "conj_grad_10");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_10);
	__ocl_conj_grad_11_reduction_step0 =
	    oclCreateKernel(__ocl_program, "conj_grad_11_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_11_reduction_step0);
	__ocl_conj_grad_11_reduction_step1 =
	    oclCreateKernel(__ocl_program, "conj_grad_11_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_11_reduction_step1);
	__ocl_conj_grad_11_reduction_step2 =
	    oclCreateKernel(__ocl_program, "conj_grad_11_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_conj_grad_11_reduction_step2);
	__ocl_tls_1D_checking =
	    oclCreateKernel(__ocl_program, "TLS_Checking_1D");
	DYN_PROGRAM_CHECK(__ocl_tls_1D_checking);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_colidx = oclCreateBuffer(colidx, (637001) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_colidx, -1);
	__ocl_buffer_rowstr = oclCreateBuffer(rowstr, (7002) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_rowstr, -1);
	__ocl_buffer_x = oclCreateBuffer(x, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_x, -1);
	__ocl_buffer_z = oclCreateBuffer(z, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_z, -1);
	__ocl_buffer_q = oclCreateBuffer(q, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_q, -1);
	__ocl_buffer_r = oclCreateBuffer(r, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_r, -1);
	__ocl_buffer_p = oclCreateBuffer(p, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_p, -1);
	__ocl_buffer_w = oclCreateBuffer(w, (7003) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_w, -1);
	__ocl_buffer_a = oclCreateBuffer(a, (637001) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_a, -1);

	//------------------------------------------
	// GPU TLS wr/rd buffers (BEGIN)
	//------------------------------------------
	rd_oclb_w = oclCreateBuffer(rd_log_w, (7003) * sizeof(int));
	wr_oclb_w = oclCreateBuffer(wr_log_w, (7003) * sizeof(int));
	oclHostWrites(rd_oclb_w);
	oclHostWrites(wr_oclb_w);
	DYN_BUFFER_CHECK(rd_oclb_w, -1);
	DYN_BUFFER_CHECK(wr_oclb_w, -1);
	__oclb_gpu_tls_conflict_flag =
	    oclCreateBuffer(&gpu_tls_conflict_flag, 1 * sizeof(int));

	//------------------------------------------
	// GPU TLS wr/rd buffers (END)
	//------------------------------------------
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_colidx);
	oclHostWrites(__ocl_buffer_rowstr);
	oclHostWrites(__ocl_buffer_x);
	oclHostWrites(__ocl_buffer_z);
	oclHostWrites(__ocl_buffer_q);
	oclHostWrites(__ocl_buffer_r);
	oclHostWrites(__ocl_buffer_p);
	oclHostWrites(__ocl_buffer_w);
	oclHostWrites(__ocl_buffer_a);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_colidx);
	oclReleaseBuffer(__ocl_buffer_rowstr);
	oclReleaseBuffer(__ocl_buffer_x);
	oclReleaseBuffer(__ocl_buffer_z);
	oclReleaseBuffer(__ocl_buffer_q);
	oclReleaseBuffer(__ocl_buffer_r);
	oclReleaseBuffer(__ocl_buffer_p);
	oclReleaseBuffer(__ocl_buffer_w);
	oclReleaseBuffer(__ocl_buffer_a);
	if (__ocl_buffer_norm_temp11_main_2_size > 0) {
		oclReleaseBuffer(__ocl_buffer_norm_temp11_main_2);
		__ocl_buffer_norm_temp11_main_2_size = 0;
	}
	if (__ocl_output_norm_temp11_main_2_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_norm_temp11_main_2);
		free(__ocl_output_norm_temp11_main_2);
		__ocl_output_norm_temp11_main_2_size = 0;
	}
	if (__ocl_buffer_norm_temp12_main_2_size > 0) {
		oclReleaseBuffer(__ocl_buffer_norm_temp12_main_2);
		__ocl_buffer_norm_temp12_main_2_size = 0;
	}
	if (__ocl_output_norm_temp12_main_2_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_norm_temp12_main_2);
		free(__ocl_output_norm_temp12_main_2);
		__ocl_output_norm_temp12_main_2_size = 0;
	}
	if (__ocl_buffer_norm_temp11_main_5_size > 0) {
		oclReleaseBuffer(__ocl_buffer_norm_temp11_main_5);
		__ocl_buffer_norm_temp11_main_5_size = 0;
	}
	if (__ocl_output_norm_temp11_main_5_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_norm_temp11_main_5);
		free(__ocl_output_norm_temp11_main_5);
		__ocl_output_norm_temp11_main_5_size = 0;
	}
	if (__ocl_buffer_norm_temp12_main_5_size > 0) {
		oclReleaseBuffer(__ocl_buffer_norm_temp12_main_5);
		__ocl_buffer_norm_temp12_main_5_size = 0;
	}
	if (__ocl_output_norm_temp12_main_5_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_norm_temp12_main_5);
		free(__ocl_output_norm_temp12_main_5);
		__ocl_output_norm_temp12_main_5_size = 0;
	}
	if (__ocl_buffer_rho_conj_grad_1_size > 0) {
		oclReleaseBuffer(__ocl_buffer_rho_conj_grad_1);
		__ocl_buffer_rho_conj_grad_1_size = 0;
	}
	if (__ocl_output_rho_conj_grad_1_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_rho_conj_grad_1);
		free(__ocl_output_rho_conj_grad_1);
		__ocl_output_rho_conj_grad_1_size = 0;
	}
	if (__ocl_buffer_d_conj_grad_5_size > 0) {
		oclReleaseBuffer(__ocl_buffer_d_conj_grad_5);
		__ocl_buffer_d_conj_grad_5_size = 0;
	}
	if (__ocl_output_d_conj_grad_5_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_d_conj_grad_5);
		free(__ocl_output_d_conj_grad_5);
		__ocl_output_d_conj_grad_5_size = 0;
	}
	if (__ocl_buffer_rho_conj_grad_7_size > 0) {
		oclReleaseBuffer(__ocl_buffer_rho_conj_grad_7);
		__ocl_buffer_rho_conj_grad_7_size = 0;
	}
	if (__ocl_output_rho_conj_grad_7_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_rho_conj_grad_7);
		free(__ocl_output_rho_conj_grad_7);
		__ocl_output_rho_conj_grad_7_size = 0;
	}
	if (__ocl_buffer_sum_conj_grad_11_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum_conj_grad_11);
		__ocl_buffer_sum_conj_grad_11_size = 0;
	}
	if (__ocl_output_sum_conj_grad_11_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum_conj_grad_11);
		free(__ocl_output_sum_conj_grad_11);
		__ocl_output_sum_conj_grad_11_size = 0;
	}
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_colidx);
	oclHostWrites(__ocl_buffer_rowstr);
	oclHostWrites(__ocl_buffer_x);
	oclHostWrites(__ocl_buffer_z);
	oclHostWrites(__ocl_buffer_q);
	oclHostWrites(__ocl_buffer_r);
	oclHostWrites(__ocl_buffer_p);
	oclHostWrites(__ocl_buffer_w);
	oclHostWrites(__ocl_buffer_a);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

void ocl_gputls_checking()
{
	oclHostReads(__oclb_gpu_tls_conflict_flag);
	oclSync();
	if (gpu_tls_conflict_flag) {
		fprintf(stderr, "Found conflict.\n");
	} else {
		fprintf(stdout, "No conflict.\n");
	}
}

#ifdef PROFILING
static void dump_profiling()
{
	FILE *prof = fopen("profiling-cg", "w");
	float kernel = 0.0f, buffer = 0.0f;

	kernel += oclDumpKernelProfiling(__ocl_main_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_2_reduction_step0, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_2_reduction_step1, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_2_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_3, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_4, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_5_reduction_step0, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_5_reduction_step1, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_5_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_main_6, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_0, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_1_reduction_step0, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_1_reduction_step1, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_1_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_2, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_3, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_4, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_5_reduction_step0, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_5_reduction_step1, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_5_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_6, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_7_reduction_step0, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_7_reduction_step1, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_7_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_8, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_9, prof);
	kernel += oclDumpKernelProfiling(__ocl_conj_grad_10, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_11_reduction_step0, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_11_reduction_step1, prof);
	kernel +=
	    oclDumpKernelProfiling(__ocl_conj_grad_11_reduction_step2, prof);

	buffer += oclDumpBufferProfiling(__ocl_buffer_colidx, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_rowstr, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_x, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_z, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_q, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_r, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_p, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_w, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_a, prof);
	PROFILE_LOCALVAR_OCL_BUFFERS(buffer, prof);

	fprintf(stderr, "-- kernel: %.3fms\n", kernel);
	fprintf(stderr, "-- buffer: %.3fms\n", buffer);
	fclose(prof);
}
#endif

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
