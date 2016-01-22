//-------------------------------------------------------------------------------
//Host code 
//Generated at : Mon Aug  6 14:03:01 2012
//Compiler options: 
//      Software Cache  true
//      Local Memory    true
//      DefaultParallelDepth    3
//      UserDefParallelDepth    false
//      EnableLoopInterchange   true
//      Generating debug/profiling code false
//      EnableMLFeatureCollection       false
//      Array Linearization     false
//      GPU TLs false
//      Strict TLS Checking     true
//      Check TLS Conflict at the end of function       true
//-------------------------------------------------------------------------------

#include "npb-C.h"
#include "npbparams.h"
#include "sys/time.h"
#include "ocldef.h"

static double x[2048];
static double q[10];
double myrandlc(double *x, ocl_buffer * __ocl_buffer_x, double a)
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

void myvranlc(int n, double *x_seed, ocl_buffer * __ocl_buffer_x_seed, double a,
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
		double Mops, t1, t2, t3, t4, x1, x2, sx, sy, an, tt, gc;
		double dum[3] = { 1.0, 1.0, 1.0 };
		DECLARE_LOCALVAR_OCL_BUFFER(dum, double, (3));
		int np, ierr, node, no_nodes, i, ik, kk, l, k, nit, ierrcode,
		    no_large_nodes, np_add, k_offset, j;
		int nthreads = 1;
		boolean verified;
		char size[14];
		DECLARE_LOCALVAR_OCL_BUFFER(size, char, (14));
		struct timeval tt1, tt2;
		printf
		    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - EP Benchmark\n");
		sprintf(size, "%12.0f", pow(2.0, 32 + 1));
		for (j = 13; j >= 1; j--) {
			if (size[j] == '.')
				size[j] = ' ';
		}
		printf(" Number of random numbers generated: %13s\n", size);
		verified = 0;
		np = (1 << (32 - 10));

		myvranlc(0, &(dum[0]), __ocl_buffer_dum, dum[1], &(dum[2]),
			 __ocl_buffer_dum);
		dum[0] = myrandlc(&(dum[1]), __ocl_buffer_dum, dum[2]);
		for (i = 0; i < 2 * (1 << 10); i++)
			x[i] = -1.0e99;
		Mops = log(sqrt(fabs((((1.0) > (1.0)) ? (1.0) : (1.0)))));
		timer_clear(1);
		timer_clear(2);
		timer_clear(3);
		myvranlc(0, &t1, NULL, 1220703125.0, x, __ocl_buffer_x);
		t1 = 1220703125.0;
		for (i = 1; i <= 10 + 1; i++) {
			t2 = myrandlc(&t1, NULL, t1);
		}
		an = t1;
		tt = 271828183.0;
		gc = 0.0;
		sx = 0.0;
		sy = 0.0;
		for (i = 0; i <= 10 - 1; i++) {
			q[i] = 0.0;
		}
		k_offset = -1;
		timer_start(1);
		gettimeofday(&tt1, ((void *)0));
		{
			double t1, t2, t3, t4, x1, x2;
			int kk, i, ik;
			//--------------------------------------------------------------
			//Loop defined at line 288 of ep.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//Reduction step 1
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (np) - (1) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				size_t __ocl_act_buf_size = (_ocl_gws[0]);
				REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: sx
				CREATE_REDUCTION_STEP1_BUFFER
				    (__ocl_buffer_sx_main_0_size,
				     __ocl_buf_size, __ocl_buffer_sx_main_0,
				     double);
//Prepare buffer for the reduction variable: sy
				CREATE_REDUCTION_STEP1_BUFFER
				    (__ocl_buffer_sy_main_0_size,
				     __ocl_buf_size, __ocl_buffer_sy_main_0,
				     double);

				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
				if (__ocl_buf_size > __ocl_act_buf_size) {
					oclSetKernelArgBuffer
					    (__ocl_main_0_reduction_step0, 0,
					     __ocl_buffer_sx_main_0);
					oclSetKernelArgBuffer
					    (__ocl_main_0_reduction_step0, 1,
					     __ocl_buffer_sy_main_0);
					unsigned int __ocl_buffer_offset =
					    __ocl_buf_size - __ocl_act_buf_size;
					oclSetKernelArg
					    (__ocl_main_0_reduction_step0, 2,
					     sizeof(unsigned int),
					     &__ocl_act_buf_size);
					oclSetKernelArg
					    (__ocl_main_0_reduction_step0, 3,
					     sizeof(unsigned int),
					     &__ocl_buffer_offset);

					size_t __offset_work_size[1] =
					    { __ocl_buffer_offset };
					oclRunKernel
					    (__ocl_main_0_reduction_step0, 1,
					     __offset_work_size);
				}

				oclSetKernelArg(__ocl_main_0_reduction_step1, 0,
						sizeof(int), &k_offset);
				oclSetKernelArg(__ocl_main_0_reduction_step1, 1,
						sizeof(double), &an);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step1, 2,
				     __ocl_buffer_x);
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				int __ocl_k_bound = np;
				oclSetKernelArg(__ocl_main_0_reduction_step1, 3,
						sizeof(int), &__ocl_k_bound);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step1, 4,
				     __ocl_buffer_sx_main_0);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step1, 5,
				     __ocl_buffer_sy_main_0);
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

				oclRunKernel(__ocl_main_0_reduction_step1, 1,
					     _ocl_gws);

//Reduction Step 2
				unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
				CREATE_REDUCTION_STEP2_BUFFER
				    (__ocl_output_sx_main_0_size,
				     __ocl_num_block, 16,
				     __ocl_output_buffer_sx_main_0,
				     __ocl_output_sx_main_0, double);
				CREATE_REDUCTION_STEP2_BUFFER
				    (__ocl_output_sy_main_0_size,
				     __ocl_num_block, 16,
				     __ocl_output_buffer_sy_main_0,
				     __ocl_output_sy_main_0, double);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step2, 0,
				     __ocl_buffer_sx_main_0);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step2, 1,
				     __ocl_output_buffer_sx_main_0);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step2, 2,
				     __ocl_buffer_sy_main_0);
				oclSetKernelArgBuffer
				    (__ocl_main_0_reduction_step2, 3,
				     __ocl_output_buffer_sy_main_0);

				oclDevWrites(__ocl_output_buffer_sx_main_0);
				oclDevWrites(__ocl_output_buffer_sy_main_0);

				size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
				size_t __ocl_localThreads[] = { GROUP_SIZE };

				oclRunKernelL(__ocl_main_0_reduction_step2, 1,
					      __ocl_globalThreads,
					      __ocl_localThreads);

//Do the final reduction part on the CPU
				oclHostReads(__ocl_output_buffer_sx_main_0);
				oclHostReads(__ocl_output_buffer_sy_main_0);
				oclSync();

				for (unsigned __ocl_i = 0;
				     __ocl_i < __ocl_num_block; __ocl_i++) {
					sx = sx +
					    __ocl_output_sx_main_0[__ocl_i];
					sy = sy +
					    __ocl_output_sy_main_0[__ocl_i];
				}

			}

		}
		sync_ocl_buffers();
		gettimeofday(&tt2, ((void *)0));
		timer_stop(1);
		double tm = timer_read(1);
		nit = 0;
		if (32 == 24) {
			if ((fabs((sx - (-3.247834652034740e3)) / sx) <= 1.0e-8)
			    && (fabs((sy - (-6.958407078382297e3)) / sy) <=
				1.0e-8)) {
				verified = 1;
			}
		} else if (32 == 25) {
			if ((fabs((sx - (-2.863319731645753e3)) / sx) <= 1.0e-8)
			    && (fabs((sy - (-6.320053679109499e3)) / sy) <=
				1.0e-8)) {
				verified = 1;
			}
		} else if (32 == 28) {
			if ((fabs((sx - (-4.295875165629892e3)) / sx) <= 1.0e-8)
			    && (fabs((sy - (-1.580732573678431e4)) / sy) <=
				1.0e-8)) {
				verified = 1;
			}
		} else if (32 == 30) {
			if ((fabs((sx - (4.033815542441498e4)) / sx) <= 1.0e-8)
			    && (fabs((sy - (-2.660669192809235e4)) / sy) <=
				1.0e-8)) {
				verified = 1;
			}
		} else if (32 == 32) {
			if ((fabs((sx - (4.764367927995374e4)) / sx) <= 1.0e-8)
			    && (fabs((sy - (-8.084072988043731e4)) / sy) <=
				1.0e-8)) {
				verified = 1;
			}
		}
		Mops = pow(2.0, 32 + 1) / tm / 1000000.0;
		printf
		    ("EP Benchmark Results: \nCPU Time = %10.4f\nN = 2^%5d\nNo. Gaussian Pairs = %15.0f\nSums = %25.15e %25.15e\nCounts:\n",
		     tm, 32, gc, sx, sy);
		c_print_results("EP", 'C', 32 + 1, 0, 0, nit, nthreads, tm,
				Mops, "Random numbers generated", verified,
				"2.3", "06 Aug 2012", "gcc", "gcc", "(none)",
				"-I../common", "-std=c99 -O3 -fopenmp",
				"-lm -fopenmp", "randdp");
		if (verified == 1) {
			printf(" Verification Successful\n");
		} else {
			printf(" Verification Failed\n");
		}
		release_ocl_buffers();
		if (0 == 1) {
			printf("Total time:     %f", timer_read(1));
			printf("Gaussian pairs: %f", timer_read(2));
			printf("Random numbers: %f", timer_read(3));
		}
	}
}

//---------------------------------------------------------------------------
//OCL related routines (BEGIN)
//---------------------------------------------------------------------------

static void init_ocl_runtime()
{
	int err;

	if (unlikely(err = oclInit("NVIDIA", 1))) {
		fprintf(stderr, "Failed to init ocl runtime:%d.\n", err);
		exit(err);
	}

	__ocl_program = oclBuildProgram("ep.C.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_main_0_reduction_step0 =
	    oclCreateKernel(__ocl_program, "main_0_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_main_0_reduction_step0);
	__ocl_main_0_reduction_step1 =
	    oclCreateKernel(__ocl_program, "main_0_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_main_0_reduction_step1);
	__ocl_main_0_reduction_step2 =
	    oclCreateKernel(__ocl_program, "main_0_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_main_0_reduction_step2);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_x = oclCreateBuffer(x, (2048) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_x, -1);
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_x);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_x);
	if (__ocl_buffer_sx_main_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sx_main_0);
		__ocl_buffer_sx_main_0_size = 0;
	}
	if (__ocl_output_sx_main_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sx_main_0);
		free(__ocl_output_sx_main_0);
		__ocl_output_sx_main_0_size = 0;
	}
	if (__ocl_buffer_sy_main_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sy_main_0);
		__ocl_buffer_sy_main_0_size = 0;
	}
	if (__ocl_output_sy_main_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sy_main_0);
		free(__ocl_output_sy_main_0);
		__ocl_output_sy_main_0_size = 0;
	}
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_x);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
