//-------------------------------------------------------------------------------
//Host code 
//Generated at : Thu Aug  9 10:00:42 2012
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
#include "header.h"
#include "ocldef.h"

static void add();
static void adi();
static void error_norm(double rms[5], ocl_buffer * __ocl_buffer_rms);
static void rhs_norm(double rms[5], ocl_buffer * __ocl_buffer_rms);
static void exact_rhs();
static void exact_solution(double xi, double eta, double zeta, double dtemp[5],
			   ocl_buffer * __ocl_buffer_dtemp);
static void initialize();
static void lhsinit();
static void lhsx();
static void lhsy();
static void lhsz();
static void compute_rhs();
static void set_constants();
static void verify(int no_time_steps, char *class,
		   ocl_buffer * __ocl_buffer_class, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified);
static void x_solve();
static void x_backsubstitute();
static void x_solve_cell();
static void matvec_sub(double ablock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_ablock, int ablock_0,
		       int ablock_1, int ablock_2, int ablock_3,
		       double avec[5][65][65][65],
		       ocl_buffer * __ocl_buffer_avec, int avec_0, int avec_1,
		       int avec_2, double bvec[5][65][65][65],
		       ocl_buffer * __ocl_buffer_bvec, int bvec_0, int bvec_1,
		       int bvec_2);
static void matmul_sub(double ablock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_ablock, int ablock_0,
		       int ablock_1, int ablock_2, int ablock_3,
		       double bblock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_bblock, int bblock_0,
		       int bblock_1, int bblock_2, int bblock_3,
		       double cblock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_cblock, int cblock_0,
		       int cblock_1, int cblock_2, int cblock_3);
static void binvcrhs(double lhs[3][5][5][65][65][65],
		     ocl_buffer * __ocl_buffer_lhs, int lhs_0, int lhs_1,
		     int lhs_2, int lhs_3, double c[3][5][5][65][65][65],
		     ocl_buffer * __ocl_buffer_c, int c_0, int c_1, int c_2,
		     int c_3, double r[5][65][65][65],
		     ocl_buffer * __ocl_buffer_r, int r_0, int r_1, int r_2);
static void binvrhs(double lhs[3][5][5][65][65][65],
		    ocl_buffer * __ocl_buffer_lhs, int lhs_0, int lhs_1,
		    int lhs_2, int lhs_3, double r[5][65][65][65],
		    ocl_buffer * __ocl_buffer_r, int r_0, int r_1, int r_2);
static void y_solve();
static void y_backsubstitute();
static void y_solve_cell();
static void z_solve();
static void z_backsubstitute();
static void z_solve_cell();
int main(int argc, char **argv, ocl_buffer * __ocl_buffer_argv)
{
	{
		int niter, step, n3;
		int nthreads = 1;
		double navg, mflops;
		double tmax;
		boolean verified;
		char class;
		FILE *fp;
		init_ocl_runtime();
		printf
		    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - BT Benchmark\n\n");
		fp = fopen("inputbt.data", "r");
		if (fp != ((void *)0)) {
			printf(" Reading from input file inputbt.data");
			fscanf(fp, "%d", &niter);
			while (fgetc(fp) != '\n') ;
			fscanf(fp, "%lg", &dt);
			while (fgetc(fp) != '\n') ;
			fscanf(fp, "%d%d%d", &grid_points[0], &grid_points[1],
			       &grid_points[2]);
			fclose(fp);
		} else {
			printf
			    (" No input file inputbt.data. Using compiled defaults\n");
			niter = 200;
			dt = 0.0008;
			grid_points[0] = 64;
			grid_points[1] = 64;
			grid_points[2] = 64;
		}
		printf(" Size: %3dx%3dx%3d\n", grid_points[0], grid_points[1],
		       grid_points[2]);
		printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);
		if (grid_points[0] > 64 || grid_points[1] > 64
		    || grid_points[2] > 64) {
			printf(" %dx%dx%d\n", grid_points[0], grid_points[1],
			       grid_points[2]);
			printf
			    (" Problem size too big for compiled array sizes\n");
			exit(1);
		}
		set_constants();
		{
		}
		{
			initialize();
			lhsinit();
			exact_rhs();
			adi();
			initialize();
		}
		flush_ocl_buffers();
		timer_clear(1);
		timer_start(1);
		{
			for (step = 1; step <= niter; step++) {
				if (step % 20 == 0 || step == 1) {
					printf(" Time step %4d\n", step);
				}
				adi();
			}
		}
		flush_ocl_buffers();
		timer_stop(1);
		tmax = timer_read(1);
		verify(niter, &class, NULL, &verified, NULL);
		n3 = grid_points[0] * grid_points[1] * grid_points[2];
		navg = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0;
		if (tmax != 0.0) {
			mflops =
			    1.0e-6 * (double)niter *(3478.8 * (double)n3 -
						     17655.7 * ((navg) *
								(navg)) +
						     28023.7 * navg) / tmax;
		} else {
			mflops = 0.0;
		}
		release_ocl_buffers();
		c_print_results("BT", class, grid_points[0], grid_points[1],
				grid_points[2], niter, nthreads, tmax, mflops,
				"          floating point", verified, "2.3",
				"09 Aug 2012", "gcc", "gcc", "(none)",
				"-I../common", "-std=c99 -O3 -fopenmp",
				"-lm -fopenmp", "(none)");
	}
}

static void add()
{
	int i, j, k, m;
	//--------------------------------------------------------------
	//Loop defined at line 225 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_add_0, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_add_0, 1, __ocl_buffer_rhs);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_add_0, 2, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_add_0, 3, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_add_0, 4, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_add_0, 3, _ocl_gws);
	}

}

static void adi()
{
	compute_rhs();
	x_solve();
	y_solve();
	z_solve();
	add();
}

static void error_norm(double rms[5], ocl_buffer * __ocl_buffer_rms)
{
	{
		int i, j, k, m, d;
		double xi, eta, zeta, u_exact[5], add;
		DECLARE_LOCALVAR_OCL_BUFFER(u_exact, double, (5));
		for (m = 0; m < 5; m++) {
			rms[m] = 0.0;
		}
		for (i = 0; i < grid_points[0]; i++) {
			xi = (double)i *dnxm1;
			for (j = 0; j < grid_points[1]; j++) {
				eta = (double)j *dnym1;
				for (k = 0; k < grid_points[2]; k++) {
					zeta = (double)k *dnzm1;

					exact_solution(xi, eta, zeta, u_exact,
						       __ocl_buffer_u_exact);
					for (m = 0; m < 5; m++) {
						add =
						    u[m][i][j][k] - u_exact[m];
						rms[m] = rms[m] + add * add;
					}
				}
			}
		}
		for (m = 0; m < 5; m++) {
			for (d = 0; d <= 2; d++) {
				rms[m] = rms[m] / (double)(grid_points[d] - 2);
			}
			rms[m] = sqrt(rms[m]);
		}
	}
}

static void rhs_norm(double rms[5], ocl_buffer * __ocl_buffer_rms)
{
	{
		int i, j, k, d, m;
		double add;
		for (m = 0; m < 5; m++) {
			rms[m] = 0.0;
		}
		for (i = 1; i < grid_points[0] - 1; i++) {
			for (j = 1; j < grid_points[1] - 1; j++) {
				for (k = 1; k < grid_points[2] - 1; k++) {
					for (m = 0; m < 5; m++) {
						add = rhs[m][i][j][k];
						rms[m] = rms[m] + add * add;
					}
				}
			}
		}
		for (m = 0; m < 5; m++) {
			for (d = 0; d <= 2; d++) {
				rms[m] = rms[m] / (double)(grid_points[d] - 2);
			}
			rms[m] = sqrt(rms[m]);
		}
	}
}

static void exact_rhs()
{
	double dtemp[5], xi, eta, zeta, dtpp;
	DECLARE_LOCALVAR_OCL_BUFFER(dtemp, double, (5));
	int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;
	//--------------------------------------------------------------
	//Loop defined at line 348 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_exact_rhs_0, 0,
				      __ocl_buffer_forcing);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_exact_rhs_0, 1, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_exact_rhs_0, 2, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_exact_rhs_0, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_forcing);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_exact_rhs_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 364 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_exact_rhs_1, 0, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_exact_rhs_1, 1, sizeof(double), &dnzm1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_1, 2,
				      __ocl_buffer_grid_points);
		oclSetKernelArg(__ocl_exact_rhs_1, 3, sizeof(double), &dnxm1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_1, 4,
				      __ocl_buffer_forcing);
		oclSetKernelArg(__ocl_exact_rhs_1, 5, sizeof(double), &tx2);
		oclSetKernelArg(__ocl_exact_rhs_1, 6, sizeof(double), &dx1tx1);
		oclSetKernelArg(__ocl_exact_rhs_1, 7, sizeof(double), &c2);
		oclSetKernelArg(__ocl_exact_rhs_1, 8, sizeof(double), &xxcon1);
		oclSetKernelArg(__ocl_exact_rhs_1, 9, sizeof(double), &dx2tx1);
		oclSetKernelArg(__ocl_exact_rhs_1, 10, sizeof(double), &xxcon2);
		oclSetKernelArg(__ocl_exact_rhs_1, 11, sizeof(double), &dx3tx1);
		oclSetKernelArg(__ocl_exact_rhs_1, 12, sizeof(double), &dx4tx1);
		oclSetKernelArg(__ocl_exact_rhs_1, 13, sizeof(double), &c1);
		oclSetKernelArg(__ocl_exact_rhs_1, 14, sizeof(double), &xxcon3);
		oclSetKernelArg(__ocl_exact_rhs_1, 15, sizeof(double), &xxcon4);
		oclSetKernelArg(__ocl_exact_rhs_1, 16, sizeof(double), &xxcon5);
		oclSetKernelArg(__ocl_exact_rhs_1, 17, sizeof(double), &dx5tx1);
		oclSetKernelArg(__ocl_exact_rhs_1, 18, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_exact_rhs_1, 19, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_exact_rhs_1, 20, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_exact_rhs_1, 21, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_forcing);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_grid_points);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_exact_rhs_1, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 464 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_exact_rhs_2, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_exact_rhs_2, 1, sizeof(double), &dnzm1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_2, 2,
				      __ocl_buffer_grid_points);
		oclSetKernelArg(__ocl_exact_rhs_2, 3, sizeof(double), &dnym1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_2, 4,
				      __ocl_buffer_forcing);
		oclSetKernelArg(__ocl_exact_rhs_2, 5, sizeof(double), &ty2);
		oclSetKernelArg(__ocl_exact_rhs_2, 6, sizeof(double), &dy1ty1);
		oclSetKernelArg(__ocl_exact_rhs_2, 7, sizeof(double), &yycon2);
		oclSetKernelArg(__ocl_exact_rhs_2, 8, sizeof(double), &dy2ty1);
		oclSetKernelArg(__ocl_exact_rhs_2, 9, sizeof(double), &c2);
		oclSetKernelArg(__ocl_exact_rhs_2, 10, sizeof(double), &yycon1);
		oclSetKernelArg(__ocl_exact_rhs_2, 11, sizeof(double), &dy3ty1);
		oclSetKernelArg(__ocl_exact_rhs_2, 12, sizeof(double), &dy4ty1);
		oclSetKernelArg(__ocl_exact_rhs_2, 13, sizeof(double), &c1);
		oclSetKernelArg(__ocl_exact_rhs_2, 14, sizeof(double), &yycon3);
		oclSetKernelArg(__ocl_exact_rhs_2, 15, sizeof(double), &yycon4);
		oclSetKernelArg(__ocl_exact_rhs_2, 16, sizeof(double), &yycon5);
		oclSetKernelArg(__ocl_exact_rhs_2, 17, sizeof(double), &dy5ty1);
		oclSetKernelArg(__ocl_exact_rhs_2, 18, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_exact_rhs_2, 19, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_exact_rhs_2, 20, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_exact_rhs_2, 21, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_forcing);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_grid_points);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_exact_rhs_2, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 566 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_exact_rhs_3, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_exact_rhs_3, 1, sizeof(double), &dnym1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_3, 2,
				      __ocl_buffer_grid_points);
		oclSetKernelArg(__ocl_exact_rhs_3, 3, sizeof(double), &dnzm1);
		oclSetKernelArgBuffer(__ocl_exact_rhs_3, 4,
				      __ocl_buffer_forcing);
		oclSetKernelArg(__ocl_exact_rhs_3, 5, sizeof(double), &tz2);
		oclSetKernelArg(__ocl_exact_rhs_3, 6, sizeof(double), &dz1tz1);
		oclSetKernelArg(__ocl_exact_rhs_3, 7, sizeof(double), &zzcon2);
		oclSetKernelArg(__ocl_exact_rhs_3, 8, sizeof(double), &dz2tz1);
		oclSetKernelArg(__ocl_exact_rhs_3, 9, sizeof(double), &dz3tz1);
		oclSetKernelArg(__ocl_exact_rhs_3, 10, sizeof(double), &c2);
		oclSetKernelArg(__ocl_exact_rhs_3, 11, sizeof(double), &zzcon1);
		oclSetKernelArg(__ocl_exact_rhs_3, 12, sizeof(double), &dz4tz1);
		oclSetKernelArg(__ocl_exact_rhs_3, 13, sizeof(double), &c1);
		oclSetKernelArg(__ocl_exact_rhs_3, 14, sizeof(double), &zzcon3);
		oclSetKernelArg(__ocl_exact_rhs_3, 15, sizeof(double), &zzcon4);
		oclSetKernelArg(__ocl_exact_rhs_3, 16, sizeof(double), &zzcon5);
		oclSetKernelArg(__ocl_exact_rhs_3, 17, sizeof(double), &dz5tz1);
		oclSetKernelArg(__ocl_exact_rhs_3, 18, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_exact_rhs_3, 19, __ocl_buffer_ce);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_exact_rhs_3, 20, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_exact_rhs_3, 21, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_forcing);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_grid_points);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_exact_rhs_3, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 666 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_exact_rhs_4, 0,
				      __ocl_buffer_forcing);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_exact_rhs_4, 1, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_exact_rhs_4, 2, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_exact_rhs_4, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_forcing);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_exact_rhs_4, 3, _ocl_gws);
	}

}

static void exact_solution(double xi, double eta, double zeta, double dtemp[5],
			   ocl_buffer * __ocl_buffer_dtemp)
{
	{
		int m;
		for (m = 0; m < 5; m++) {
			dtemp[m] =
			    ce[m][0] + xi * (ce[m][1] +
					     xi * (ce[m][4] +
						   xi * (ce[m][7] +
							 xi * ce[m][10]))) +
			    eta * (ce[m][2] +
				   eta * (ce[m][5] +
					  eta * (ce[m][8] + eta * ce[m][11]))) +
			    zeta * (ce[m][3] +
				    zeta * (ce[m][6] +
					    zeta * (ce[m][9] +
						    zeta * ce[m][12])));
		}
	}
}

static void initialize()
{
	int i, j, k, m, ix, iy, iz;
	double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];
	DECLARE_LOCALVAR_OCL_BUFFER(Pface, double, (2 * 3 * 5));
	DECLARE_LOCALVAR_OCL_BUFFER(temp, double, (5));
	//--------------------------------------------------------------
	//Loop defined at line 728 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (64) - (0);
		_ocl_gws[1] = (64) - (0);
		_ocl_gws[2] = (64) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_initialize_0, 0, __ocl_buffer_u);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 745 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_1, 0, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_initialize_1, 1, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_initialize_1, 2, sizeof(double), &dnzm1);
		oclSetKernelArgBuffer(__ocl_initialize_1, 3, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_initialize_1, 4, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_initialize_1, 5, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_initialize_1, 6, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_initialize_1, 7, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_1, 3, _ocl_gws);
	}

	i = 0;
	xi = 0.0;
	//--------------------------------------------------------------
	//Loop defined at line 795 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_2, 0, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_initialize_2, 1, sizeof(double), &dnzm1);
		oclSetKernelArg(__ocl_initialize_2, 2, sizeof(double), &xi);
		oclSetKernelArgBuffer(__ocl_initialize_2, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_2, 4, sizeof(int), &i);
		oclSetKernelArgBuffer(__ocl_initialize_2, 5, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_initialize_2, 6, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_initialize_2, 7, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_2, 2, _ocl_gws);
	}

	i = grid_points[0] - 1;
	xi = 1.0;
	//--------------------------------------------------------------
	//Loop defined at line 815 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_3, 0, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_initialize_3, 1, sizeof(double), &dnzm1);
		oclSetKernelArg(__ocl_initialize_3, 2, sizeof(double), &xi);
		oclSetKernelArgBuffer(__ocl_initialize_3, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_3, 4, sizeof(int), &i);
		oclSetKernelArgBuffer(__ocl_initialize_3, 5, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_initialize_3, 6, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_initialize_3, 7, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_3, 2, _ocl_gws);
	}

	j = 0;
	eta = 0.0;
	//--------------------------------------------------------------
	//Loop defined at line 834 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[0]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_4, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_initialize_4, 1, sizeof(double), &dnzm1);
		oclSetKernelArg(__ocl_initialize_4, 2, sizeof(double), &eta);
		oclSetKernelArgBuffer(__ocl_initialize_4, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_4, 4, sizeof(int), &j);
		oclSetKernelArgBuffer(__ocl_initialize_4, 5, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_initialize_4, 6, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_initialize_4, 7, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_4, 2, _ocl_gws);
	}

	j = grid_points[1] - 1;
	eta = 1.0;
	//--------------------------------------------------------------
	//Loop defined at line 853 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[0]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_5, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_initialize_5, 1, sizeof(double), &dnzm1);
		oclSetKernelArg(__ocl_initialize_5, 2, sizeof(double), &eta);
		oclSetKernelArgBuffer(__ocl_initialize_5, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_5, 4, sizeof(int), &j);
		oclSetKernelArgBuffer(__ocl_initialize_5, 5, __ocl_buffer_ce);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_initialize_5, 6, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_initialize_5, 7, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_5, 2, _ocl_gws);
	}

	k = 0;
	zeta = 0.0;
	//--------------------------------------------------------------
	//Loop defined at line 872 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1]) - (0);
		_ocl_gws[1] = (grid_points[0]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_6, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_initialize_6, 1, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_initialize_6, 2, sizeof(double), &zeta);
		oclSetKernelArgBuffer(__ocl_initialize_6, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_6, 4, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_initialize_6, 5, __ocl_buffer_ce);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_initialize_6, 6, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_initialize_6, 7, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_6, 2, _ocl_gws);
	}

	k = grid_points[2] - 1;
	zeta = 1.0;
	//--------------------------------------------------------------
	//Loop defined at line 891 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1]) - (0);
		_ocl_gws[1] = (grid_points[0]) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_initialize_7, 0, sizeof(double), &dnxm1);
		oclSetKernelArg(__ocl_initialize_7, 1, sizeof(double), &dnym1);
		oclSetKernelArg(__ocl_initialize_7, 2, sizeof(double), &zeta);
		oclSetKernelArgBuffer(__ocl_initialize_7, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_initialize_7, 4, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_initialize_7, 5, __ocl_buffer_ce);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_initialize_7, 6, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_initialize_7, 7, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_u);
		oclDevWrites(__ocl_buffer_ce);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_initialize_7, 2, _ocl_gws);
	}

}

static void lhsinit()
{
	int i, j, k, m, n;
	//--------------------------------------------------------------
	//Loop defined at line 919 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsinit_0, 0, __ocl_buffer_lhs);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_lhsinit_0, 1, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_lhsinit_0, 2, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_lhsinit_0, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsinit_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 939 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsinit_1, 0, __ocl_buffer_lhs);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_lhsinit_1, 1, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_lhsinit_1, 2, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_lhsinit_1, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsinit_1, 3, _ocl_gws);
	}

}

static void lhsx()
{
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 970 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 1, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 2, __ocl_buffer_fjac);
		oclSetKernelArg(__ocl_lhsx_0, 3, sizeof(double), &c2);
		oclSetKernelArg(__ocl_lhsx_0, 4, sizeof(double), &c1);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 5, __ocl_buffer_njac);
		oclSetKernelArg(__ocl_lhsx_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsx_0, 7, sizeof(double), &c3c4);
		oclSetKernelArg(__ocl_lhsx_0, 8, sizeof(double), &c1345);
		oclSetKernelArg(__ocl_lhsx_0, 9, sizeof(double), &dt);
		oclSetKernelArg(__ocl_lhsx_0, 10, sizeof(double), &tx1);
		oclSetKernelArg(__ocl_lhsx_0, 11, sizeof(double), &tx2);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 12, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_0, 13, sizeof(double), &dx1);
		oclSetKernelArg(__ocl_lhsx_0, 14, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_lhsx_0, 15, sizeof(double), &dx3);
		oclSetKernelArg(__ocl_lhsx_0, 16, sizeof(double), &dx4);
		oclSetKernelArg(__ocl_lhsx_0, 17, sizeof(double), &dx5);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_lhsx_0, 18, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_lhsx_0, 19, sizeof(int), &__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_fjac);
		oclDevWrites(__ocl_buffer_njac);
		oclDevWrites(__ocl_buffer_lhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_grid_points);
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsx_0, 2, _ocl_gws);
	}

}

static void lhsy()
{
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 1256 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 1, __ocl_buffer_fjac);
		oclSetKernelArg(__ocl_lhsy_0, 2, sizeof(double), &c2);
		oclSetKernelArg(__ocl_lhsy_0, 3, sizeof(double), &c1);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 4, __ocl_buffer_njac);
		oclSetKernelArg(__ocl_lhsy_0, 5, sizeof(double), &c3c4);
		oclSetKernelArg(__ocl_lhsy_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsy_0, 7, sizeof(double), &c1345);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_lhsy_0, 8, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_lhsy_0, 9, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_lhsy_0, 10, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_fjac);
		oclDevWrites(__ocl_buffer_njac);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsy_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1360 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_lhsy_1, 0, sizeof(double), &dt);
		oclSetKernelArg(__ocl_lhsy_1, 1, sizeof(double), &ty1);
		oclSetKernelArg(__ocl_lhsy_1, 2, sizeof(double), &ty2);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 3, __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 4, __ocl_buffer_fjac);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 5, __ocl_buffer_njac);
		oclSetKernelArg(__ocl_lhsy_1, 6, sizeof(double), &dy1);
		oclSetKernelArg(__ocl_lhsy_1, 7, sizeof(double), &dy2);
		oclSetKernelArg(__ocl_lhsy_1, 8, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_lhsy_1, 9, sizeof(double), &dy4);
		oclSetKernelArg(__ocl_lhsy_1, 10, sizeof(double), &dy5);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_lhsy_1, 11, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_lhsy_1, 12, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_lhsy_1, 13, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_fjac);
		oclDevReads(__ocl_buffer_njac);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsy_1, 3, _ocl_gws);
	}

}

static void lhsz()
{
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 1554 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 1, __ocl_buffer_fjac);
		oclSetKernelArg(__ocl_lhsz_0, 2, sizeof(double), &c2);
		oclSetKernelArg(__ocl_lhsz_0, 3, sizeof(double), &c1);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 4, __ocl_buffer_njac);
		oclSetKernelArg(__ocl_lhsz_0, 5, sizeof(double), &c3c4);
		oclSetKernelArg(__ocl_lhsz_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsz_0, 7, sizeof(double), &c3);
		oclSetKernelArg(__ocl_lhsz_0, 8, sizeof(double), &c4);
		oclSetKernelArg(__ocl_lhsz_0, 9, sizeof(double), &c1345);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_lhsz_0, 10, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_lhsz_0, 11, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_lhsz_0, 12, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_fjac);
		oclDevWrites(__ocl_buffer_njac);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsz_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1658 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_lhsz_1, 0, sizeof(double), &dt);
		oclSetKernelArg(__ocl_lhsz_1, 1, sizeof(double), &tz1);
		oclSetKernelArg(__ocl_lhsz_1, 2, sizeof(double), &tz2);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 3, __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 4, __ocl_buffer_fjac);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 5, __ocl_buffer_njac);
		oclSetKernelArg(__ocl_lhsz_1, 6, sizeof(double), &dz1);
		oclSetKernelArg(__ocl_lhsz_1, 7, sizeof(double), &dz2);
		oclSetKernelArg(__ocl_lhsz_1, 8, sizeof(double), &dz3);
		oclSetKernelArg(__ocl_lhsz_1, 9, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_lhsz_1, 10, sizeof(double), &dz5);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_lhsz_1, 11, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_lhsz_1, 12, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_lhsz_1, 13, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_fjac);
		oclDevReads(__ocl_buffer_njac);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsz_1, 3, _ocl_gws);
	}

}

static void compute_rhs()
{
	int i, j, k, m;
	double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
	//--------------------------------------------------------------
	//Loop defined at line 1845 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 1,
				      __ocl_buffer_rho_i);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 2, __ocl_buffer_us);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 3, __ocl_buffer_vs);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 4, __ocl_buffer_ws);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 5,
				      __ocl_buffer_square);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 6, __ocl_buffer_qs);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_compute_rhs_0, 7, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_compute_rhs_0, 8, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_compute_rhs_0, 9, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rho_i);
		oclDevWrites(__ocl_buffer_us);
		oclDevWrites(__ocl_buffer_vs);
		oclDevWrites(__ocl_buffer_ws);
		oclDevWrites(__ocl_buffer_square);
		oclDevWrites(__ocl_buffer_qs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1870 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2]) - (0);
		_ocl_gws[1] = (grid_points[1]) - (0);
		_ocl_gws[2] = (grid_points[0]) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_1, 0, __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_compute_rhs_1, 1,
				      __ocl_buffer_forcing);
		int __ocl_k_bound = grid_points[2];
		oclSetKernelArg(__ocl_compute_rhs_1, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1];
		oclSetKernelArg(__ocl_compute_rhs_1, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0];
		oclSetKernelArg(__ocl_compute_rhs_1, 4, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_forcing);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_1, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1886 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 0, __ocl_buffer_us);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 1, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_2, 2, sizeof(double),
				&dx1tx1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_compute_rhs_2, 4, sizeof(double), &tx2);
		oclSetKernelArg(__ocl_compute_rhs_2, 5, sizeof(double),
				&dx2tx1);
		oclSetKernelArg(__ocl_compute_rhs_2, 6, sizeof(double),
				&xxcon2);
		oclSetKernelArg(__ocl_compute_rhs_2, 7, sizeof(double), &con43);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 8,
				      __ocl_buffer_square);
		oclSetKernelArg(__ocl_compute_rhs_2, 9, sizeof(double), &c2);
		oclSetKernelArg(__ocl_compute_rhs_2, 10, sizeof(double),
				&dx3tx1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 11, __ocl_buffer_vs);
		oclSetKernelArg(__ocl_compute_rhs_2, 12, sizeof(double),
				&dx4tx1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 13, __ocl_buffer_ws);
		oclSetKernelArg(__ocl_compute_rhs_2, 14, sizeof(double),
				&dx5tx1);
		oclSetKernelArg(__ocl_compute_rhs_2, 15, sizeof(double),
				&xxcon3);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 16, __ocl_buffer_qs);
		oclSetKernelArg(__ocl_compute_rhs_2, 17, sizeof(double),
				&xxcon4);
		oclSetKernelArg(__ocl_compute_rhs_2, 18, sizeof(double),
				&xxcon5);
		oclSetKernelArgBuffer(__ocl_compute_rhs_2, 19,
				      __ocl_buffer_rho_i);
		oclSetKernelArg(__ocl_compute_rhs_2, 20, sizeof(double), &c1);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_2, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_2, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_2, 23, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_us);
		oclDevReads(__ocl_buffer_u);
		oclDevReads(__ocl_buffer_square);
		oclDevReads(__ocl_buffer_vs);
		oclDevReads(__ocl_buffer_ws);
		oclDevReads(__ocl_buffer_qs);
		oclDevReads(__ocl_buffer_rho_i);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_2, 3, _ocl_gws);
	}

	i = 1;
	//--------------------------------------------------------------
	//Loop defined at line 1949 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_3, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_3, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_3, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_3, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_3, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_3, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_3, 3, _ocl_gws);
	}

	i = 2;
	//--------------------------------------------------------------
	//Loop defined at line 1963 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_4, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_4, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_4, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_4, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_4, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_4, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_4, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1976 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 3) - (3);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_5, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_5, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_5, 2, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_5, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_5, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 3;
		oclSetKernelArg(__ocl_compute_rhs_5, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_5, 3, _ocl_gws);
	}

	i = grid_points[0] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 1993 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_6, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_6, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_6, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_6, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_6, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_6, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_6, 3, _ocl_gws);
	}

	i = grid_points[0] - 2;
	//--------------------------------------------------------------
	//Loop defined at line 2007 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_7, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_7, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_7, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_7, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_7, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_7, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_7, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2023 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 0, __ocl_buffer_vs);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 1, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_8, 2, sizeof(double),
				&dy1ty1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_compute_rhs_8, 4, sizeof(double), &ty2);
		oclSetKernelArg(__ocl_compute_rhs_8, 5, sizeof(double),
				&dy2ty1);
		oclSetKernelArg(__ocl_compute_rhs_8, 6, sizeof(double),
				&yycon2);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 7, __ocl_buffer_us);
		oclSetKernelArg(__ocl_compute_rhs_8, 8, sizeof(double),
				&dy3ty1);
		oclSetKernelArg(__ocl_compute_rhs_8, 9, sizeof(double), &con43);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 10,
				      __ocl_buffer_square);
		oclSetKernelArg(__ocl_compute_rhs_8, 11, sizeof(double), &c2);
		oclSetKernelArg(__ocl_compute_rhs_8, 12, sizeof(double),
				&dy4ty1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 13, __ocl_buffer_ws);
		oclSetKernelArg(__ocl_compute_rhs_8, 14, sizeof(double),
				&dy5ty1);
		oclSetKernelArg(__ocl_compute_rhs_8, 15, sizeof(double),
				&yycon3);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 16, __ocl_buffer_qs);
		oclSetKernelArg(__ocl_compute_rhs_8, 17, sizeof(double),
				&yycon4);
		oclSetKernelArg(__ocl_compute_rhs_8, 18, sizeof(double),
				&yycon5);
		oclSetKernelArgBuffer(__ocl_compute_rhs_8, 19,
				      __ocl_buffer_rho_i);
		oclSetKernelArg(__ocl_compute_rhs_8, 20, sizeof(double), &c1);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_8, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_8, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_8, 23, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_vs);
		oclDevReads(__ocl_buffer_u);
		oclDevReads(__ocl_buffer_us);
		oclDevReads(__ocl_buffer_square);
		oclDevReads(__ocl_buffer_ws);
		oclDevReads(__ocl_buffer_qs);
		oclDevReads(__ocl_buffer_rho_i);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_8, 3, _ocl_gws);
	}

	j = 1;
	//--------------------------------------------------------------
	//Loop defined at line 2081 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_9, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_9, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_9, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_9, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_9, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_9, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_9, 3, _ocl_gws);
	}

	j = 2;
	//--------------------------------------------------------------
	//Loop defined at line 2095 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_10, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_10, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_10, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_10, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_10, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_10, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_10, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2108 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 3) - (3);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_11, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_11, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_11, 2, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_11, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 3;
		oclSetKernelArg(__ocl_compute_rhs_11, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_11, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_11, 3, _ocl_gws);
	}

	j = grid_points[1] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 2125 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_12, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_12, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_12, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_12, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_12, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_12, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_12, 3, _ocl_gws);
	}

	j = grid_points[1] - 2;
	//--------------------------------------------------------------
	//Loop defined at line 2139 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_13, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_13, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_13, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_13, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_13, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_13, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_13, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2155 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 0, __ocl_buffer_ws);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 1,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_14, 2, sizeof(double),
				&dz1tz1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 3, __ocl_buffer_u);
		oclSetKernelArg(__ocl_compute_rhs_14, 4, sizeof(double), &tz2);
		oclSetKernelArg(__ocl_compute_rhs_14, 5, sizeof(double),
				&dz2tz1);
		oclSetKernelArg(__ocl_compute_rhs_14, 6, sizeof(double),
				&zzcon2);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 7, __ocl_buffer_us);
		oclSetKernelArg(__ocl_compute_rhs_14, 8, sizeof(double),
				&dz3tz1);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 9, __ocl_buffer_vs);
		oclSetKernelArg(__ocl_compute_rhs_14, 10, sizeof(double),
				&dz4tz1);
		oclSetKernelArg(__ocl_compute_rhs_14, 11, sizeof(double),
				&con43);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 12,
				      __ocl_buffer_square);
		oclSetKernelArg(__ocl_compute_rhs_14, 13, sizeof(double), &c2);
		oclSetKernelArg(__ocl_compute_rhs_14, 14, sizeof(double),
				&dz5tz1);
		oclSetKernelArg(__ocl_compute_rhs_14, 15, sizeof(double),
				&zzcon3);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 16,
				      __ocl_buffer_qs);
		oclSetKernelArg(__ocl_compute_rhs_14, 17, sizeof(double),
				&zzcon4);
		oclSetKernelArg(__ocl_compute_rhs_14, 18, sizeof(double),
				&zzcon5);
		oclSetKernelArgBuffer(__ocl_compute_rhs_14, 19,
				      __ocl_buffer_rho_i);
		oclSetKernelArg(__ocl_compute_rhs_14, 20, sizeof(double), &c1);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_14, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_14, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_14, 23, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_ws);
		oclDevReads(__ocl_buffer_u);
		oclDevReads(__ocl_buffer_us);
		oclDevReads(__ocl_buffer_vs);
		oclDevReads(__ocl_buffer_square);
		oclDevReads(__ocl_buffer_qs);
		oclDevReads(__ocl_buffer_rho_i);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_14, 3, _ocl_gws);
	}

	oclSwapDimensions (__ocl_buffer_rhs, 5 * (64 / 2 * 2 + 1), 64 / 2 * 2 + 1, 1,
			  64 / 2 * 2 + 1, sizeof(double));
	oclSwapDimensions (__ocl_buffer_u, 5 * (64 / 2 * 2 + 1), 64 / 2 * 2 + 1, 1,
			  64 / 2 * 2 + 1, sizeof(double));
	k = 1;
	//--------------------------------------------------------------
	//Loop defined at line 2220 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_15, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_15, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_15, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_15, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_15, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_15, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_15, 3, _ocl_gws);
	}

	k = 2;
	//--------------------------------------------------------------
	//Loop defined at line 2240 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_16, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_16, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_16, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_16, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_16, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_16, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_16, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2259 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[2] - 3) - (3);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_17, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_17, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_17, 2, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_17, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_k_bound = grid_points[2] - 3;
		oclSetKernelArg(__ocl_compute_rhs_17, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_17, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_17, 3, _ocl_gws);
	}

	k = grid_points[2] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 2283 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_18, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_18, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_18, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_18, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_18, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_18, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_18, 3, _ocl_gws);
	}

	k = grid_points[2] - 2;
	//--------------------------------------------------------------
	//Loop defined at line 2303 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_19, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_19, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_19, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_19, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_19, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_19, 5, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_u);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_19, 3, _ocl_gws);
	}

	oclSwapDimensions (__ocl_buffer_rhs, 5 * (64 / 2 * 2 + 1), 64 / 2 * 2 + 1, 1,
			  64 / 2 * 2 + 1, sizeof(double));
	oclSwapDimensions (__ocl_buffer_u, 5 * (64 / 2 * 2 + 1), 64 / 2 * 2 + 1, 1,
			  64 / 2 * 2 + 1, sizeof(double));
	//--------------------------------------------------------------
	//Loop defined at line 2327 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);
		_ocl_gws[2] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_20, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_20, 1, sizeof(double), &dt);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_20, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_20, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_20, 4, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_compute_rhs_20, 3, _ocl_gws);
	}

}

static void set_constants()
{
	ce[0][0] = 2.0;
	ce[0][1] = 0.0;
	ce[0][2] = 0.0;
	ce[0][3] = 4.0;
	ce[0][4] = 5.0;
	ce[0][5] = 3.0;
	ce[0][6] = 0.5;
	ce[0][7] = 0.02;
	ce[0][8] = 0.01;
	ce[0][9] = 0.03;
	ce[0][10] = 0.5;
	ce[0][11] = 0.4;
	ce[0][12] = 0.3;
	ce[1][0] = 1.0;
	ce[1][1] = 0.0;
	ce[1][2] = 0.0;
	ce[1][3] = 0.0;
	ce[1][4] = 1.0;
	ce[1][5] = 2.0;
	ce[1][6] = 3.0;
	ce[1][7] = 0.01;
	ce[1][8] = 0.03;
	ce[1][9] = 0.02;
	ce[1][10] = 0.4;
	ce[1][11] = 0.3;
	ce[1][12] = 0.5;
	ce[2][0] = 2.0;
	ce[2][1] = 2.0;
	ce[2][2] = 0.0;
	ce[2][3] = 0.0;
	ce[2][4] = 0.0;
	ce[2][5] = 2.0;
	ce[2][6] = 3.0;
	ce[2][7] = 0.04;
	ce[2][8] = 0.03;
	ce[2][9] = 0.05;
	ce[2][10] = 0.3;
	ce[2][11] = 0.5;
	ce[2][12] = 0.4;
	ce[3][0] = 2.0;
	ce[3][1] = 2.0;
	ce[3][2] = 0.0;
	ce[3][3] = 0.0;
	ce[3][4] = 0.0;
	ce[3][5] = 2.0;
	ce[3][6] = 3.0;
	ce[3][7] = 0.03;
	ce[3][8] = 0.05;
	ce[3][9] = 0.04;
	ce[3][10] = 0.2;
	ce[3][11] = 0.1;
	ce[3][12] = 0.3;
	ce[4][0] = 5.0;
	ce[4][1] = 4.0;
	ce[4][2] = 3.0;
	ce[4][3] = 2.0;
	ce[4][4] = 0.1;
	ce[4][5] = 0.4;
	ce[4][6] = 0.3;
	ce[4][7] = 0.05;
	ce[4][8] = 0.04;
	ce[4][9] = 0.03;
	ce[4][10] = 0.1;
	ce[4][11] = 0.3;
	ce[4][12] = 0.2;
	c1 = 1.4;
	c2 = 0.4;
	c3 = 0.1;
	c4 = 1.0;
	c5 = 1.4;
	dnxm1 = 1.0 / (double)(grid_points[0] - 1);
	dnym1 = 1.0 / (double)(grid_points[1] - 1);
	dnzm1 = 1.0 / (double)(grid_points[2] - 1);
	c1c2 = c1 * c2;
	c1c5 = c1 * c5;
	c3c4 = c3 * c4;
	c1345 = c1c5 * c3c4;
	conz1 = (1.0 - c1c5);
	tx1 = 1.0 / (dnxm1 * dnxm1);
	tx2 = 1.0 / (2.0 * dnxm1);
	tx3 = 1.0 / dnxm1;
	ty1 = 1.0 / (dnym1 * dnym1);
	ty2 = 1.0 / (2.0 * dnym1);
	ty3 = 1.0 / dnym1;
	tz1 = 1.0 / (dnzm1 * dnzm1);
	tz2 = 1.0 / (2.0 * dnzm1);
	tz3 = 1.0 / dnzm1;
	dx1 = 0.75;
	dx2 = 0.75;
	dx3 = 0.75;
	dx4 = 0.75;
	dx5 = 0.75;
	dy1 = 0.75;
	dy2 = 0.75;
	dy3 = 0.75;
	dy4 = 0.75;
	dy5 = 0.75;
	dz1 = 1.0;
	dz2 = 1.0;
	dz3 = 1.0;
	dz4 = 1.0;
	dz5 = 1.0;
	dxmax = (((dx3) > (dx4)) ? (dx3) : (dx4));
	dymax = (((dy2) > (dy4)) ? (dy2) : (dy4));
	dzmax = (((dz2) > (dz3)) ? (dz2) : (dz3));
	dssp =
	    0.25 *
	    (((dx1) >
	      ((((dy1) > (dz1)) ? (dy1) : (dz1)))) ? (dx1) : ((((dy1) >
								(dz1)) ? (dy1)
							       : (dz1))));
	c4dssp = 4.0 * dssp;
	c5dssp = 5.0 * dssp;
	dttx1 = dt * tx1;
	dttx2 = dt * tx2;
	dtty1 = dt * ty1;
	dtty2 = dt * ty2;
	dttz1 = dt * tz1;
	dttz2 = dt * tz2;
	c2dttx1 = 2.0 * dttx1;
	c2dtty1 = 2.0 * dtty1;
	c2dttz1 = 2.0 * dttz1;
	dtdssp = dt * dssp;
	comz1 = dtdssp;
	comz4 = 4.0 * dtdssp;
	comz5 = 5.0 * dtdssp;
	comz6 = 6.0 * dtdssp;
	c3c4tx3 = c3c4 * tx3;
	c3c4ty3 = c3c4 * ty3;
	c3c4tz3 = c3c4 * tz3;
	dx1tx1 = dx1 * tx1;
	dx2tx1 = dx2 * tx1;
	dx3tx1 = dx3 * tx1;
	dx4tx1 = dx4 * tx1;
	dx5tx1 = dx5 * tx1;
	dy1ty1 = dy1 * ty1;
	dy2ty1 = dy2 * ty1;
	dy3ty1 = dy3 * ty1;
	dy4ty1 = dy4 * ty1;
	dy5ty1 = dy5 * ty1;
	dz1tz1 = dz1 * tz1;
	dz2tz1 = dz2 * tz1;
	dz3tz1 = dz3 * tz1;
	dz4tz1 = dz4 * tz1;
	dz5tz1 = dz5 * tz1;
	c2iv = 2.5;
	con43 = 4.0 / 3.0;
	con16 = 1.0 / 6.0;
	xxcon1 = c3c4tx3 * con43 * tx3;
	xxcon2 = c3c4tx3 * tx3;
	xxcon3 = c3c4tx3 * conz1 * tx3;
	xxcon4 = c3c4tx3 * con16 * tx3;
	xxcon5 = c3c4tx3 * c1c5 * tx3;
	yycon1 = c3c4ty3 * con43 * ty3;
	yycon2 = c3c4ty3 * ty3;
	yycon3 = c3c4ty3 * conz1 * ty3;
	yycon4 = c3c4ty3 * con16 * ty3;
	yycon5 = c3c4ty3 * c1c5 * ty3;
	zzcon1 = c3c4tz3 * con43 * tz3;
	zzcon2 = c3c4tz3 * tz3;
	zzcon3 = c3c4tz3 * conz1 * tz3;
	zzcon4 = c3c4tz3 * con16 * tz3;
	zzcon5 = c3c4tz3 * c1c5 * tz3;
}

static void verify(int no_time_steps, char *class,
		   ocl_buffer * __ocl_buffer_class, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified)
{
	{
		double xcrref[5], xceref[5], xcrdif[5], xcedif[5], epsilon,
		    xce[5], xcr[5], dtref;
		DECLARE_LOCALVAR_OCL_BUFFER(xcrref, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xceref, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xcrdif, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xcedif, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xce, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xcr, double, (5));
		int m;
		epsilon = 1.0e-08;

		error_norm(xce, __ocl_buffer_xce);
		compute_rhs();
		sync_ocl_buffers();

		rhs_norm(xcr, __ocl_buffer_xcr);
		for (m = 0; m < 5; m++) {
			xcr[m] = xcr[m] / dt;
		}
		*class = 'U';
		*verified = 1;
		for (m = 0; m < 5; m++) {
			xcrref[m] = 1.0;
			xceref[m] = 1.0;
		}
		if (grid_points[0] == 12 && grid_points[1] == 12
		    && grid_points[2] == 12 && no_time_steps == 60) {
			*class = 'S';
			dtref = 1.0e-2;
			xcrref[0] = 1.7034283709541311e-01;
			xcrref[1] = 1.2975252070034097e-02;
			xcrref[2] = 3.2527926989486055e-02;
			xcrref[3] = 2.6436421275166801e-02;
			xcrref[4] = 1.9211784131744430e-01;
			xceref[0] = 4.9976913345811579e-04;
			xceref[1] = 4.5195666782961927e-05;
			xceref[2] = 7.3973765172921357e-05;
			xceref[3] = 7.3821238632439731e-05;
			xceref[4] = 8.9269630987491446e-04;
		} else if (grid_points[0] == 24 && grid_points[1] == 24
			   && grid_points[2] == 24 && no_time_steps == 200) {
			*class = 'W';
			dtref = 0.8e-3;
			xcrref[0] = 0.1125590409344e+03;
			xcrref[1] = 0.1180007595731e+02;
			xcrref[2] = 0.2710329767846e+02;
			xcrref[3] = 0.2469174937669e+02;
			xcrref[4] = 0.2638427874317e+03;
			xceref[0] = 0.4419655736008e+01;
			xceref[1] = 0.4638531260002e+00;
			xceref[2] = 0.1011551749967e+01;
			xceref[3] = 0.9235878729944e+00;
			xceref[4] = 0.1018045837718e+02;
		} else if (grid_points[0] == 64 && grid_points[1] == 64
			   && grid_points[2] == 64 && no_time_steps == 200) {
			*class = 'A';
			dtref = 0.8e-3;
			xcrref[0] = 1.0806346714637264e+02;
			xcrref[1] = 1.1319730901220813e+01;
			xcrref[2] = 2.5974354511582465e+01;
			xcrref[3] = 2.3665622544678910e+01;
			xcrref[4] = 2.5278963211748344e+02;
			xceref[0] = 4.2348416040525025e+00;
			xceref[1] = 4.4390282496995698e-01;
			xceref[2] = 9.6692480136345650e-01;
			xceref[3] = 8.8302063039765474e-01;
			xceref[4] = 9.7379901770829278e+00;
		} else if (grid_points[0] == 102 && grid_points[1] == 102
			   && grid_points[2] == 102 && no_time_steps == 200) {
			*class = 'B';
			dtref = 3.0e-4;
			xcrref[0] = 1.4233597229287254e+03;
			xcrref[1] = 9.9330522590150238e+01;
			xcrref[2] = 3.5646025644535285e+02;
			xcrref[3] = 3.2485447959084092e+02;
			xcrref[4] = 3.2707541254659363e+03;
			xceref[0] = 5.2969847140936856e+01;
			xceref[1] = 4.4632896115670668e+00;
			xceref[2] = 1.3122573342210174e+01;
			xceref[3] = 1.2006925323559144e+01;
			xceref[4] = 1.2459576151035986e+02;
		} else if (grid_points[0] == 162 && grid_points[1] == 162
			   && grid_points[2] == 162 && no_time_steps == 200) {
			*class = 'C';
			dtref = 1.0e-4;
			xcrref[0] = 0.62398116551764615e+04;
			xcrref[1] = 0.50793239190423964e+03;
			xcrref[2] = 0.15423530093013596e+04;
			xcrref[3] = 0.13302387929291190e+04;
			xcrref[4] = 0.11604087428436455e+05;
			xceref[0] = 0.16462008369091265e+03;
			xceref[1] = 0.11497107903824313e+02;
			xceref[2] = 0.41207446207461508e+02;
			xceref[3] = 0.37087651059694167e+02;
			xceref[4] = 0.36211053051841265e+03;
		} else {
			*verified = 0;
		}
		for (m = 0; m < 5; m++) {
			xcrdif[m] = fabs((xcr[m] - xcrref[m]) / xcrref[m]);
			xcedif[m] = fabs((xce[m] - xceref[m]) / xceref[m]);
		}
		if (*class != 'U') {
			printf(" Verification being performed for class %1c\n",
			       *class);
			printf(" accuracy setting for epsilon = %20.13e\n",
			       epsilon);
			if (fabs(dt - dtref) > epsilon) {
				*verified = 0;
				*class = 'U';
				printf
				    (" DT does not match the reference value of %15.8e\n",
				     dtref);
			}
		} else {
			printf(" Unknown class\n");
		}
		if (*class != 'U') {
			printf(" Comparison of RMS-norms of residual\n");
		} else {
			printf(" RMS-norms of residual\n");
		}
		for (m = 0; m < 5; m++) {
			if (*class == 'U') {
				printf("          %2d%20.13e\n", m, xcr[m]);
			} else if (xcrdif[m] > epsilon
				   || xcrdif[m] != xcrdif[m]) {
				*verified = 0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n",
				       m, xcr[m], xcrref[m], xcrdif[m]);
			} else {
				printf("          %2d%20.13e%20.13e%20.13e\n",
				       m, xcr[m], xcrref[m], xcrdif[m]);
			}
		}
		if (*class != 'U') {
			printf(" Comparison of RMS-norms of solution error\n");
		} else {
			printf(" RMS-norms of solution error\n");
		}
		for (m = 0; m < 5; m++) {
			if (*class == 'U') {
				printf("          %2d%20.13e\n", m, xce[m]);
			} else if (xcedif[m] > epsilon
				   || xcedif[m] != xcedif[m]) {
				*verified = 0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n",
				       m, xce[m], xceref[m], xcedif[m]);
			} else {
				printf("          %2d%20.13e%20.13e%20.13e\n",
				       m, xce[m], xceref[m], xcedif[m]);
			}
		}
		if (*class == 'U') {
			printf(" No reference values provided\n");
			printf(" No verification performed\n");
		} else if (*verified == 1) {
			printf(" Verification Successful\n");
		} else {
			printf(" Verification failed\n");
		}
	}
}

static void x_solve()
{
	lhsx();
	x_solve_cell();
	x_backsubstitute();
}

static void x_backsubstitute()
{
	int i, j, k, m, n;
	for (i = grid_points[0] - 2; i >= 0; i--) {
		//--------------------------------------------------------------
		//Loop defined at line 2853 of bt.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (grid_points[2] - 1) - (1);
			_ocl_gws[1] = (grid_points[1] - 1) - (1);
			_ocl_gws[2] = (5) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_x_backsubstitute_0, 0,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_x_backsubstitute_0, 1,
					sizeof(int), &i);
			oclSetKernelArgBuffer(__ocl_x_backsubstitute_0, 2,
					      __ocl_buffer_lhs);
			int __ocl_k_bound = grid_points[2] - 1;
			oclSetKernelArg(__ocl_x_backsubstitute_0, 3,
					sizeof(int), &__ocl_k_bound);
			int __ocl_j_bound = grid_points[1] - 1;
			oclSetKernelArg(__ocl_x_backsubstitute_0, 4,
					sizeof(int), &__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_rhs);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_lhs);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_x_backsubstitute_0, 3, _ocl_gws);
		}

	}
}

static void x_solve_cell()
{
	int i, j, k, isize;
	isize = grid_points[0] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 2891 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_x_solve_cell_0, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_x_solve_cell_0, 1,
				      __ocl_buffer_rhs);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_x_solve_cell_0, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_x_solve_cell_0, 3, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_x_solve_cell_0, 2, _ocl_gws);
	}

	for (i = 1; i < isize; i++) {
		//--------------------------------------------------------------
		//Loop defined at line 2912 of bt.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 1) - (1);
			_ocl_gws[1] = (grid_points[1] - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_x_solve_cell_1, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_x_solve_cell_1, 1, sizeof(int),
					&i);
			oclSetKernelArgBuffer(__ocl_x_solve_cell_1, 2,
					      __ocl_buffer_rhs);
			int __ocl_k_bound = grid_points[2] - 1;
			oclSetKernelArg(__ocl_x_solve_cell_1, 3, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1] - 1;
			oclSetKernelArg(__ocl_x_solve_cell_1, 4, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_lhs);
			oclDevWrites(__ocl_buffer_rhs);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_x_solve_cell_1, 2, _ocl_gws);
		}

	}
	//--------------------------------------------------------------
	//Loop defined at line 2945 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[1] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_x_solve_cell_2, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_x_solve_cell_2, 1, sizeof(int), &isize);
		oclSetKernelArgBuffer(__ocl_x_solve_cell_2, 2,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_x_solve_cell_2, 3, sizeof(int), &i);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_x_solve_cell_2, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_x_solve_cell_2, 5, sizeof(int),
				&__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_x_solve_cell_2, 2, _ocl_gws);
	}

}

static void matvec_sub(double ablock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_ablock, int ablock_0,
		       int ablock_1, int ablock_2, int ablock_3,
		       double avec[5][65][65][65],
		       ocl_buffer * __ocl_buffer_avec, int avec_0, int avec_1,
		       int avec_2, double bvec[5][65][65][65],
		       ocl_buffer * __ocl_buffer_bvec, int bvec_0, int bvec_1,
		       int bvec_2)
{
	{
		int i;
		for (i = 0; i < 5; i++) {
			bvec[i][bvec_0][bvec_1][bvec_2] =
			    bvec[i][bvec_0][bvec_1][bvec_2] -
			    ablock[ablock_3][i][0][ablock_0][ablock_1][ablock_2]
			    * avec[0][avec_0][avec_1][avec_2] -
			    ablock[ablock_3][i][1][ablock_0][ablock_1][ablock_2]
			    * avec[1][avec_0][avec_1][avec_2] -
			    ablock[ablock_3][i][2][ablock_0][ablock_1][ablock_2]
			    * avec[2][avec_0][avec_1][avec_2] -
			    ablock[ablock_3][i][3][ablock_0][ablock_1][ablock_2]
			    * avec[3][avec_0][avec_1][avec_2] -
			    ablock[ablock_3][i][4][ablock_0][ablock_1][ablock_2]
			    * avec[4][avec_0][avec_1][avec_2];
		}
	}
}

static void matmul_sub(double ablock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_ablock, int ablock_0,
		       int ablock_1, int ablock_2, int ablock_3,
		       double bblock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_bblock, int bblock_0,
		       int bblock_1, int bblock_2, int bblock_3,
		       double cblock[3][5][5][65][65][65],
		       ocl_buffer * __ocl_buffer_cblock, int cblock_0,
		       int cblock_1, int cblock_2, int cblock_3)
{
	{
		int j;
		for (j = 0; j < 5; j++) {
			cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] =
			    cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2]
			    -
			    ablock[ablock_3][0][0][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][0][1][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][0][2][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][0][3][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][0][4][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][4][j][bblock_0][bblock_1]
			    [bblock_2];
			cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] =
			    cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2]
			    -
			    ablock[ablock_3][1][0][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][1][1][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][1][2][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][1][3][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][1][4][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][4][j][bblock_0][bblock_1]
			    [bblock_2];
			cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] =
			    cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2]
			    -
			    ablock[ablock_3][2][0][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][2][1][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][2][2][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][2][3][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][2][4][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][4][j][bblock_0][bblock_1]
			    [bblock_2];
			cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] =
			    cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2]
			    -
			    ablock[ablock_3][3][0][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][3][1][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][3][2][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][3][3][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][3][4][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][4][j][bblock_0][bblock_1]
			    [bblock_2];
			cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] =
			    cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2]
			    -
			    ablock[ablock_3][4][0][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][4][1][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][4][2][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][4][3][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			    -
			    ablock[ablock_3][4][4][ablock_0][ablock_1][ablock_2]
			    *
			    bblock[bblock_3][4][j][bblock_0][bblock_1]
			    [bblock_2];
		}
	}
}

static void binvcrhs(double lhs[3][5][5][65][65][65],
		     ocl_buffer * __ocl_buffer_lhs, int lhs_0, int lhs_1,
		     int lhs_2, int lhs_3, double c[3][5][5][65][65][65],
		     ocl_buffer * __ocl_buffer_c, int c_0, int c_1, int c_2,
		     int c_3, double r[5][65][65][65],
		     ocl_buffer * __ocl_buffer_r, int r_0, int r_1, int r_2)
{
	{
		double pivot, coeff;
		pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
		c[c_3][0][0][c_0][c_1][c_2] =
		    c[c_3][0][0][c_0][c_1][c_2] * pivot;
		c[c_3][0][1][c_0][c_1][c_2] =
		    c[c_3][0][1][c_0][c_1][c_2] * pivot;
		c[c_3][0][2][c_0][c_1][c_2] =
		    c[c_3][0][2][c_0][c_1][c_2] * pivot;
		c[c_3][0][3][c_0][c_1][c_2] =
		    c[c_3][0][3][c_0][c_1][c_2] * pivot;
		c[c_3][0][4][c_0][c_1][c_2] =
		    c[c_3][0][4][c_0][c_1][c_2] * pivot;
		r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		c[c_3][1][0][c_0][c_1][c_2] =
		    c[c_3][1][0][c_0][c_1][c_2] -
		    coeff * c[c_3][0][0][c_0][c_1][c_2];
		c[c_3][1][1][c_0][c_1][c_2] =
		    c[c_3][1][1][c_0][c_1][c_2] -
		    coeff * c[c_3][0][1][c_0][c_1][c_2];
		c[c_3][1][2][c_0][c_1][c_2] =
		    c[c_3][1][2][c_0][c_1][c_2] -
		    coeff * c[c_3][0][2][c_0][c_1][c_2];
		c[c_3][1][3][c_0][c_1][c_2] =
		    c[c_3][1][3][c_0][c_1][c_2] -
		    coeff * c[c_3][0][3][c_0][c_1][c_2];
		c[c_3][1][4][c_0][c_1][c_2] =
		    c[c_3][1][4][c_0][c_1][c_2] -
		    coeff * c[c_3][0][4][c_0][c_1][c_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		c[c_3][2][0][c_0][c_1][c_2] =
		    c[c_3][2][0][c_0][c_1][c_2] -
		    coeff * c[c_3][0][0][c_0][c_1][c_2];
		c[c_3][2][1][c_0][c_1][c_2] =
		    c[c_3][2][1][c_0][c_1][c_2] -
		    coeff * c[c_3][0][1][c_0][c_1][c_2];
		c[c_3][2][2][c_0][c_1][c_2] =
		    c[c_3][2][2][c_0][c_1][c_2] -
		    coeff * c[c_3][0][2][c_0][c_1][c_2];
		c[c_3][2][3][c_0][c_1][c_2] =
		    c[c_3][2][3][c_0][c_1][c_2] -
		    coeff * c[c_3][0][3][c_0][c_1][c_2];
		c[c_3][2][4][c_0][c_1][c_2] =
		    c[c_3][2][4][c_0][c_1][c_2] -
		    coeff * c[c_3][0][4][c_0][c_1][c_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		c[c_3][3][0][c_0][c_1][c_2] =
		    c[c_3][3][0][c_0][c_1][c_2] -
		    coeff * c[c_3][0][0][c_0][c_1][c_2];
		c[c_3][3][1][c_0][c_1][c_2] =
		    c[c_3][3][1][c_0][c_1][c_2] -
		    coeff * c[c_3][0][1][c_0][c_1][c_2];
		c[c_3][3][2][c_0][c_1][c_2] =
		    c[c_3][3][2][c_0][c_1][c_2] -
		    coeff * c[c_3][0][2][c_0][c_1][c_2];
		c[c_3][3][3][c_0][c_1][c_2] =
		    c[c_3][3][3][c_0][c_1][c_2] -
		    coeff * c[c_3][0][3][c_0][c_1][c_2];
		c[c_3][3][4][c_0][c_1][c_2] =
		    c[c_3][3][4][c_0][c_1][c_2] -
		    coeff * c[c_3][0][4][c_0][c_1][c_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		c[c_3][4][0][c_0][c_1][c_2] =
		    c[c_3][4][0][c_0][c_1][c_2] -
		    coeff * c[c_3][0][0][c_0][c_1][c_2];
		c[c_3][4][1][c_0][c_1][c_2] =
		    c[c_3][4][1][c_0][c_1][c_2] -
		    coeff * c[c_3][0][1][c_0][c_1][c_2];
		c[c_3][4][2][c_0][c_1][c_2] =
		    c[c_3][4][2][c_0][c_1][c_2] -
		    coeff * c[c_3][0][2][c_0][c_1][c_2];
		c[c_3][4][3][c_0][c_1][c_2] =
		    c[c_3][4][3][c_0][c_1][c_2] -
		    coeff * c[c_3][0][3][c_0][c_1][c_2];
		c[c_3][4][4][c_0][c_1][c_2] =
		    c[c_3][4][4][c_0][c_1][c_2] -
		    coeff * c[c_3][0][4][c_0][c_1][c_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
		c[c_3][1][0][c_0][c_1][c_2] =
		    c[c_3][1][0][c_0][c_1][c_2] * pivot;
		c[c_3][1][1][c_0][c_1][c_2] =
		    c[c_3][1][1][c_0][c_1][c_2] * pivot;
		c[c_3][1][2][c_0][c_1][c_2] =
		    c[c_3][1][2][c_0][c_1][c_2] * pivot;
		c[c_3][1][3][c_0][c_1][c_2] =
		    c[c_3][1][3][c_0][c_1][c_2] * pivot;
		c[c_3][1][4][c_0][c_1][c_2] =
		    c[c_3][1][4][c_0][c_1][c_2] * pivot;
		r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		c[c_3][0][0][c_0][c_1][c_2] =
		    c[c_3][0][0][c_0][c_1][c_2] -
		    coeff * c[c_3][1][0][c_0][c_1][c_2];
		c[c_3][0][1][c_0][c_1][c_2] =
		    c[c_3][0][1][c_0][c_1][c_2] -
		    coeff * c[c_3][1][1][c_0][c_1][c_2];
		c[c_3][0][2][c_0][c_1][c_2] =
		    c[c_3][0][2][c_0][c_1][c_2] -
		    coeff * c[c_3][1][2][c_0][c_1][c_2];
		c[c_3][0][3][c_0][c_1][c_2] =
		    c[c_3][0][3][c_0][c_1][c_2] -
		    coeff * c[c_3][1][3][c_0][c_1][c_2];
		c[c_3][0][4][c_0][c_1][c_2] =
		    c[c_3][0][4][c_0][c_1][c_2] -
		    coeff * c[c_3][1][4][c_0][c_1][c_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		c[c_3][2][0][c_0][c_1][c_2] =
		    c[c_3][2][0][c_0][c_1][c_2] -
		    coeff * c[c_3][1][0][c_0][c_1][c_2];
		c[c_3][2][1][c_0][c_1][c_2] =
		    c[c_3][2][1][c_0][c_1][c_2] -
		    coeff * c[c_3][1][1][c_0][c_1][c_2];
		c[c_3][2][2][c_0][c_1][c_2] =
		    c[c_3][2][2][c_0][c_1][c_2] -
		    coeff * c[c_3][1][2][c_0][c_1][c_2];
		c[c_3][2][3][c_0][c_1][c_2] =
		    c[c_3][2][3][c_0][c_1][c_2] -
		    coeff * c[c_3][1][3][c_0][c_1][c_2];
		c[c_3][2][4][c_0][c_1][c_2] =
		    c[c_3][2][4][c_0][c_1][c_2] -
		    coeff * c[c_3][1][4][c_0][c_1][c_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		c[c_3][3][0][c_0][c_1][c_2] =
		    c[c_3][3][0][c_0][c_1][c_2] -
		    coeff * c[c_3][1][0][c_0][c_1][c_2];
		c[c_3][3][1][c_0][c_1][c_2] =
		    c[c_3][3][1][c_0][c_1][c_2] -
		    coeff * c[c_3][1][1][c_0][c_1][c_2];
		c[c_3][3][2][c_0][c_1][c_2] =
		    c[c_3][3][2][c_0][c_1][c_2] -
		    coeff * c[c_3][1][2][c_0][c_1][c_2];
		c[c_3][3][3][c_0][c_1][c_2] =
		    c[c_3][3][3][c_0][c_1][c_2] -
		    coeff * c[c_3][1][3][c_0][c_1][c_2];
		c[c_3][3][4][c_0][c_1][c_2] =
		    c[c_3][3][4][c_0][c_1][c_2] -
		    coeff * c[c_3][1][4][c_0][c_1][c_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		c[c_3][4][0][c_0][c_1][c_2] =
		    c[c_3][4][0][c_0][c_1][c_2] -
		    coeff * c[c_3][1][0][c_0][c_1][c_2];
		c[c_3][4][1][c_0][c_1][c_2] =
		    c[c_3][4][1][c_0][c_1][c_2] -
		    coeff * c[c_3][1][1][c_0][c_1][c_2];
		c[c_3][4][2][c_0][c_1][c_2] =
		    c[c_3][4][2][c_0][c_1][c_2] -
		    coeff * c[c_3][1][2][c_0][c_1][c_2];
		c[c_3][4][3][c_0][c_1][c_2] =
		    c[c_3][4][3][c_0][c_1][c_2] -
		    coeff * c[c_3][1][3][c_0][c_1][c_2];
		c[c_3][4][4][c_0][c_1][c_2] =
		    c[c_3][4][4][c_0][c_1][c_2] -
		    coeff * c[c_3][1][4][c_0][c_1][c_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
		c[c_3][2][0][c_0][c_1][c_2] =
		    c[c_3][2][0][c_0][c_1][c_2] * pivot;
		c[c_3][2][1][c_0][c_1][c_2] =
		    c[c_3][2][1][c_0][c_1][c_2] * pivot;
		c[c_3][2][2][c_0][c_1][c_2] =
		    c[c_3][2][2][c_0][c_1][c_2] * pivot;
		c[c_3][2][3][c_0][c_1][c_2] =
		    c[c_3][2][3][c_0][c_1][c_2] * pivot;
		c[c_3][2][4][c_0][c_1][c_2] =
		    c[c_3][2][4][c_0][c_1][c_2] * pivot;
		r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		c[c_3][0][0][c_0][c_1][c_2] =
		    c[c_3][0][0][c_0][c_1][c_2] -
		    coeff * c[c_3][2][0][c_0][c_1][c_2];
		c[c_3][0][1][c_0][c_1][c_2] =
		    c[c_3][0][1][c_0][c_1][c_2] -
		    coeff * c[c_3][2][1][c_0][c_1][c_2];
		c[c_3][0][2][c_0][c_1][c_2] =
		    c[c_3][0][2][c_0][c_1][c_2] -
		    coeff * c[c_3][2][2][c_0][c_1][c_2];
		c[c_3][0][3][c_0][c_1][c_2] =
		    c[c_3][0][3][c_0][c_1][c_2] -
		    coeff * c[c_3][2][3][c_0][c_1][c_2];
		c[c_3][0][4][c_0][c_1][c_2] =
		    c[c_3][0][4][c_0][c_1][c_2] -
		    coeff * c[c_3][2][4][c_0][c_1][c_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		c[c_3][1][0][c_0][c_1][c_2] =
		    c[c_3][1][0][c_0][c_1][c_2] -
		    coeff * c[c_3][2][0][c_0][c_1][c_2];
		c[c_3][1][1][c_0][c_1][c_2] =
		    c[c_3][1][1][c_0][c_1][c_2] -
		    coeff * c[c_3][2][1][c_0][c_1][c_2];
		c[c_3][1][2][c_0][c_1][c_2] =
		    c[c_3][1][2][c_0][c_1][c_2] -
		    coeff * c[c_3][2][2][c_0][c_1][c_2];
		c[c_3][1][3][c_0][c_1][c_2] =
		    c[c_3][1][3][c_0][c_1][c_2] -
		    coeff * c[c_3][2][3][c_0][c_1][c_2];
		c[c_3][1][4][c_0][c_1][c_2] =
		    c[c_3][1][4][c_0][c_1][c_2] -
		    coeff * c[c_3][2][4][c_0][c_1][c_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		c[c_3][3][0][c_0][c_1][c_2] =
		    c[c_3][3][0][c_0][c_1][c_2] -
		    coeff * c[c_3][2][0][c_0][c_1][c_2];
		c[c_3][3][1][c_0][c_1][c_2] =
		    c[c_3][3][1][c_0][c_1][c_2] -
		    coeff * c[c_3][2][1][c_0][c_1][c_2];
		c[c_3][3][2][c_0][c_1][c_2] =
		    c[c_3][3][2][c_0][c_1][c_2] -
		    coeff * c[c_3][2][2][c_0][c_1][c_2];
		c[c_3][3][3][c_0][c_1][c_2] =
		    c[c_3][3][3][c_0][c_1][c_2] -
		    coeff * c[c_3][2][3][c_0][c_1][c_2];
		c[c_3][3][4][c_0][c_1][c_2] =
		    c[c_3][3][4][c_0][c_1][c_2] -
		    coeff * c[c_3][2][4][c_0][c_1][c_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		c[c_3][4][0][c_0][c_1][c_2] =
		    c[c_3][4][0][c_0][c_1][c_2] -
		    coeff * c[c_3][2][0][c_0][c_1][c_2];
		c[c_3][4][1][c_0][c_1][c_2] =
		    c[c_3][4][1][c_0][c_1][c_2] -
		    coeff * c[c_3][2][1][c_0][c_1][c_2];
		c[c_3][4][2][c_0][c_1][c_2] =
		    c[c_3][4][2][c_0][c_1][c_2] -
		    coeff * c[c_3][2][2][c_0][c_1][c_2];
		c[c_3][4][3][c_0][c_1][c_2] =
		    c[c_3][4][3][c_0][c_1][c_2] -
		    coeff * c[c_3][2][3][c_0][c_1][c_2];
		c[c_3][4][4][c_0][c_1][c_2] =
		    c[c_3][4][4][c_0][c_1][c_2] -
		    coeff * c[c_3][2][4][c_0][c_1][c_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
		c[c_3][3][0][c_0][c_1][c_2] =
		    c[c_3][3][0][c_0][c_1][c_2] * pivot;
		c[c_3][3][1][c_0][c_1][c_2] =
		    c[c_3][3][1][c_0][c_1][c_2] * pivot;
		c[c_3][3][2][c_0][c_1][c_2] =
		    c[c_3][3][2][c_0][c_1][c_2] * pivot;
		c[c_3][3][3][c_0][c_1][c_2] =
		    c[c_3][3][3][c_0][c_1][c_2] * pivot;
		c[c_3][3][4][c_0][c_1][c_2] =
		    c[c_3][3][4][c_0][c_1][c_2] * pivot;
		r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		c[c_3][0][0][c_0][c_1][c_2] =
		    c[c_3][0][0][c_0][c_1][c_2] -
		    coeff * c[c_3][3][0][c_0][c_1][c_2];
		c[c_3][0][1][c_0][c_1][c_2] =
		    c[c_3][0][1][c_0][c_1][c_2] -
		    coeff * c[c_3][3][1][c_0][c_1][c_2];
		c[c_3][0][2][c_0][c_1][c_2] =
		    c[c_3][0][2][c_0][c_1][c_2] -
		    coeff * c[c_3][3][2][c_0][c_1][c_2];
		c[c_3][0][3][c_0][c_1][c_2] =
		    c[c_3][0][3][c_0][c_1][c_2] -
		    coeff * c[c_3][3][3][c_0][c_1][c_2];
		c[c_3][0][4][c_0][c_1][c_2] =
		    c[c_3][0][4][c_0][c_1][c_2] -
		    coeff * c[c_3][3][4][c_0][c_1][c_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		c[c_3][1][0][c_0][c_1][c_2] =
		    c[c_3][1][0][c_0][c_1][c_2] -
		    coeff * c[c_3][3][0][c_0][c_1][c_2];
		c[c_3][1][1][c_0][c_1][c_2] =
		    c[c_3][1][1][c_0][c_1][c_2] -
		    coeff * c[c_3][3][1][c_0][c_1][c_2];
		c[c_3][1][2][c_0][c_1][c_2] =
		    c[c_3][1][2][c_0][c_1][c_2] -
		    coeff * c[c_3][3][2][c_0][c_1][c_2];
		c[c_3][1][3][c_0][c_1][c_2] =
		    c[c_3][1][3][c_0][c_1][c_2] -
		    coeff * c[c_3][3][3][c_0][c_1][c_2];
		c[c_3][1][4][c_0][c_1][c_2] =
		    c[c_3][1][4][c_0][c_1][c_2] -
		    coeff * c[c_3][3][4][c_0][c_1][c_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		c[c_3][2][0][c_0][c_1][c_2] =
		    c[c_3][2][0][c_0][c_1][c_2] -
		    coeff * c[c_3][3][0][c_0][c_1][c_2];
		c[c_3][2][1][c_0][c_1][c_2] =
		    c[c_3][2][1][c_0][c_1][c_2] -
		    coeff * c[c_3][3][1][c_0][c_1][c_2];
		c[c_3][2][2][c_0][c_1][c_2] =
		    c[c_3][2][2][c_0][c_1][c_2] -
		    coeff * c[c_3][3][2][c_0][c_1][c_2];
		c[c_3][2][3][c_0][c_1][c_2] =
		    c[c_3][2][3][c_0][c_1][c_2] -
		    coeff * c[c_3][3][3][c_0][c_1][c_2];
		c[c_3][2][4][c_0][c_1][c_2] =
		    c[c_3][2][4][c_0][c_1][c_2] -
		    coeff * c[c_3][3][4][c_0][c_1][c_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		c[c_3][4][0][c_0][c_1][c_2] =
		    c[c_3][4][0][c_0][c_1][c_2] -
		    coeff * c[c_3][3][0][c_0][c_1][c_2];
		c[c_3][4][1][c_0][c_1][c_2] =
		    c[c_3][4][1][c_0][c_1][c_2] -
		    coeff * c[c_3][3][1][c_0][c_1][c_2];
		c[c_3][4][2][c_0][c_1][c_2] =
		    c[c_3][4][2][c_0][c_1][c_2] -
		    coeff * c[c_3][3][2][c_0][c_1][c_2];
		c[c_3][4][3][c_0][c_1][c_2] =
		    c[c_3][4][3][c_0][c_1][c_2] -
		    coeff * c[c_3][3][3][c_0][c_1][c_2];
		c[c_3][4][4][c_0][c_1][c_2] =
		    c[c_3][4][4][c_0][c_1][c_2] -
		    coeff * c[c_3][3][4][c_0][c_1][c_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
		c[c_3][4][0][c_0][c_1][c_2] =
		    c[c_3][4][0][c_0][c_1][c_2] * pivot;
		c[c_3][4][1][c_0][c_1][c_2] =
		    c[c_3][4][1][c_0][c_1][c_2] * pivot;
		c[c_3][4][2][c_0][c_1][c_2] =
		    c[c_3][4][2][c_0][c_1][c_2] * pivot;
		c[c_3][4][3][c_0][c_1][c_2] =
		    c[c_3][4][3][c_0][c_1][c_2] * pivot;
		c[c_3][4][4][c_0][c_1][c_2] =
		    c[c_3][4][4][c_0][c_1][c_2] * pivot;
		r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		c[c_3][0][0][c_0][c_1][c_2] =
		    c[c_3][0][0][c_0][c_1][c_2] -
		    coeff * c[c_3][4][0][c_0][c_1][c_2];
		c[c_3][0][1][c_0][c_1][c_2] =
		    c[c_3][0][1][c_0][c_1][c_2] -
		    coeff * c[c_3][4][1][c_0][c_1][c_2];
		c[c_3][0][2][c_0][c_1][c_2] =
		    c[c_3][0][2][c_0][c_1][c_2] -
		    coeff * c[c_3][4][2][c_0][c_1][c_2];
		c[c_3][0][3][c_0][c_1][c_2] =
		    c[c_3][0][3][c_0][c_1][c_2] -
		    coeff * c[c_3][4][3][c_0][c_1][c_2];
		c[c_3][0][4][c_0][c_1][c_2] =
		    c[c_3][0][4][c_0][c_1][c_2] -
		    coeff * c[c_3][4][4][c_0][c_1][c_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		c[c_3][1][0][c_0][c_1][c_2] =
		    c[c_3][1][0][c_0][c_1][c_2] -
		    coeff * c[c_3][4][0][c_0][c_1][c_2];
		c[c_3][1][1][c_0][c_1][c_2] =
		    c[c_3][1][1][c_0][c_1][c_2] -
		    coeff * c[c_3][4][1][c_0][c_1][c_2];
		c[c_3][1][2][c_0][c_1][c_2] =
		    c[c_3][1][2][c_0][c_1][c_2] -
		    coeff * c[c_3][4][2][c_0][c_1][c_2];
		c[c_3][1][3][c_0][c_1][c_2] =
		    c[c_3][1][3][c_0][c_1][c_2] -
		    coeff * c[c_3][4][3][c_0][c_1][c_2];
		c[c_3][1][4][c_0][c_1][c_2] =
		    c[c_3][1][4][c_0][c_1][c_2] -
		    coeff * c[c_3][4][4][c_0][c_1][c_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		c[c_3][2][0][c_0][c_1][c_2] =
		    c[c_3][2][0][c_0][c_1][c_2] -
		    coeff * c[c_3][4][0][c_0][c_1][c_2];
		c[c_3][2][1][c_0][c_1][c_2] =
		    c[c_3][2][1][c_0][c_1][c_2] -
		    coeff * c[c_3][4][1][c_0][c_1][c_2];
		c[c_3][2][2][c_0][c_1][c_2] =
		    c[c_3][2][2][c_0][c_1][c_2] -
		    coeff * c[c_3][4][2][c_0][c_1][c_2];
		c[c_3][2][3][c_0][c_1][c_2] =
		    c[c_3][2][3][c_0][c_1][c_2] -
		    coeff * c[c_3][4][3][c_0][c_1][c_2];
		c[c_3][2][4][c_0][c_1][c_2] =
		    c[c_3][2][4][c_0][c_1][c_2] -
		    coeff * c[c_3][4][4][c_0][c_1][c_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		c[c_3][3][0][c_0][c_1][c_2] =
		    c[c_3][3][0][c_0][c_1][c_2] -
		    coeff * c[c_3][4][0][c_0][c_1][c_2];
		c[c_3][3][1][c_0][c_1][c_2] =
		    c[c_3][3][1][c_0][c_1][c_2] -
		    coeff * c[c_3][4][1][c_0][c_1][c_2];
		c[c_3][3][2][c_0][c_1][c_2] =
		    c[c_3][3][2][c_0][c_1][c_2] -
		    coeff * c[c_3][4][2][c_0][c_1][c_2];
		c[c_3][3][3][c_0][c_1][c_2] =
		    c[c_3][3][3][c_0][c_1][c_2] -
		    coeff * c[c_3][4][3][c_0][c_1][c_2];
		c[c_3][3][4][c_0][c_1][c_2] =
		    c[c_3][3][4][c_0][c_1][c_2] -
		    coeff * c[c_3][4][4][c_0][c_1][c_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	}
}

static void binvrhs(double lhs[3][5][5][65][65][65],
		    ocl_buffer * __ocl_buffer_lhs, int lhs_0, int lhs_1,
		    int lhs_2, int lhs_3, double r[5][65][65][65],
		    ocl_buffer * __ocl_buffer_r, int r_0, int r_1, int r_2)
{
	{
		double pivot, coeff;
		pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
		r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
		r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
		r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
		r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
		lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
		    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
		    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		r[4][r_0][r_1][r_2] =
		    r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
		pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
		r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
		coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
		r[0][r_0][r_1][r_2] =
		    r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
		r[1][r_0][r_1][r_2] =
		    r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
		r[2][r_0][r_1][r_2] =
		    r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
		coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
		r[3][r_0][r_1][r_2] =
		    r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	}
}

static void y_solve()
{
	lhsy();
	y_solve_cell();
	y_backsubstitute();
}

static void y_backsubstitute()
{
	int i, j, k, m, n;
	for (j = grid_points[1] - 2; j >= 0; j--) {
		//--------------------------------------------------------------
		//Loop defined at line 3497 of bt.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (grid_points[2] - 1) - (1);
			_ocl_gws[1] = (grid_points[0] - 1) - (1);
			_ocl_gws[2] = (5) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_y_backsubstitute_0, 0,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_y_backsubstitute_0, 1,
					sizeof(int), &j);
			oclSetKernelArgBuffer(__ocl_y_backsubstitute_0, 2,
					      __ocl_buffer_lhs);
			int __ocl_k_bound = grid_points[2] - 1;
			oclSetKernelArg(__ocl_y_backsubstitute_0, 3,
					sizeof(int), &__ocl_k_bound);
			int __ocl_i_bound = grid_points[0] - 1;
			oclSetKernelArg(__ocl_y_backsubstitute_0, 4,
					sizeof(int), &__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_rhs);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_lhs);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_y_backsubstitute_0, 3, _ocl_gws);
		}

	}
}

static void y_solve_cell()
{
	int i, j, k, jsize;
	jsize = grid_points[1] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 3535 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_y_solve_cell_0, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_y_solve_cell_0, 1,
				      __ocl_buffer_rhs);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_y_solve_cell_0, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_y_solve_cell_0, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_y_solve_cell_0, 2, _ocl_gws);
	}

	for (j = 1; j < jsize; j++) {
		//--------------------------------------------------------------
		//Loop defined at line 3556 of bt.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 1) - (1);
			_ocl_gws[1] = (grid_points[0] - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_y_solve_cell_1, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_y_solve_cell_1, 1, sizeof(int),
					&j);
			oclSetKernelArgBuffer(__ocl_y_solve_cell_1, 2,
					      __ocl_buffer_rhs);
			int __ocl_k_bound = grid_points[2] - 1;
			oclSetKernelArg(__ocl_y_solve_cell_1, 3, sizeof(int),
					&__ocl_k_bound);
			int __ocl_i_bound = grid_points[0] - 1;
			oclSetKernelArg(__ocl_y_solve_cell_1, 4, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_lhs);
			oclDevWrites(__ocl_buffer_rhs);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_y_solve_cell_1, 2, _ocl_gws);
		}

	}
	//--------------------------------------------------------------
	//Loop defined at line 3589 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_y_solve_cell_2, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_y_solve_cell_2, 1, sizeof(int), &jsize);
		oclSetKernelArgBuffer(__ocl_y_solve_cell_2, 2,
				      __ocl_buffer_rhs);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_y_solve_cell_2, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_y_solve_cell_2, 4, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_y_solve_cell_2, 2, _ocl_gws);
	}

}

static void z_solve()
{
	lhsz();
	z_solve_cell();
	z_backsubstitute();
}

static void z_backsubstitute()
{
	int i, j, k, m, n;
	//--------------------------------------------------------------
	//Loop defined at line 3673 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_backsubstitute_0, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_z_backsubstitute_0, 1,
				      __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_z_backsubstitute_0, 2,
				      __ocl_buffer_lhs);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_z_backsubstitute_0, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_z_backsubstitute_0, 4, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_grid_points);
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_backsubstitute_0, 2, _ocl_gws);
	}

}

static void z_solve_cell()
{
	int i, j, k, ksize;
	ksize = grid_points[2] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 3720 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_cell_0, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_z_solve_cell_0, 1,
				      __ocl_buffer_rhs);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_z_solve_cell_0, 2, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_z_solve_cell_0, 3, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_cell_0, 2, _ocl_gws);
	}

	for (k = 1; k < ksize; k++) {
		//--------------------------------------------------------------
		//Loop defined at line 3748 of bt.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[1] - 1) - (1);
			_ocl_gws[1] = (grid_points[0] - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_z_solve_cell_1, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_z_solve_cell_1, 1, sizeof(int),
					&k);
			oclSetKernelArgBuffer(__ocl_z_solve_cell_1, 2,
					      __ocl_buffer_rhs);
			int __ocl_j_bound = grid_points[1] - 1;
			oclSetKernelArg(__ocl_z_solve_cell_1, 3, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = grid_points[0] - 1;
			oclSetKernelArg(__ocl_z_solve_cell_1, 4, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_lhs);
			oclDevWrites(__ocl_buffer_rhs);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_z_solve_cell_1, 2, _ocl_gws);
		}

	}
	//--------------------------------------------------------------
	//Loop defined at line 3802 of bt.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 1) - (1);
		_ocl_gws[1] = (grid_points[0] - 1) - (1);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_cell_2, 0,
				      __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_cell_2, 1, sizeof(int), &ksize);
		oclSetKernelArgBuffer(__ocl_z_solve_cell_2, 2,
				      __ocl_buffer_rhs);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_z_solve_cell_2, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_z_solve_cell_2, 4, sizeof(int),
				&__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_lhs);
		oclDevWrites(__ocl_buffer_rhs);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_cell_2, 2, _ocl_gws);
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

	__ocl_program = oclBuildProgram("bt.A.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_add_0 = oclCreateKernel(__ocl_program, "add_0");
	DYN_PROGRAM_CHECK(__ocl_add_0);
	__ocl_exact_rhs_0 = oclCreateKernel(__ocl_program, "exact_rhs_0");
	DYN_PROGRAM_CHECK(__ocl_exact_rhs_0);
	__ocl_exact_rhs_1 = oclCreateKernel(__ocl_program, "exact_rhs_1");
	DYN_PROGRAM_CHECK(__ocl_exact_rhs_1);
	__ocl_exact_rhs_2 = oclCreateKernel(__ocl_program, "exact_rhs_2");
	DYN_PROGRAM_CHECK(__ocl_exact_rhs_2);
	__ocl_exact_rhs_3 = oclCreateKernel(__ocl_program, "exact_rhs_3");
	DYN_PROGRAM_CHECK(__ocl_exact_rhs_3);
	__ocl_exact_rhs_4 = oclCreateKernel(__ocl_program, "exact_rhs_4");
	DYN_PROGRAM_CHECK(__ocl_exact_rhs_4);
	__ocl_initialize_0 = oclCreateKernel(__ocl_program, "initialize_0");
	DYN_PROGRAM_CHECK(__ocl_initialize_0);
	__ocl_initialize_1 = oclCreateKernel(__ocl_program, "initialize_1");
	DYN_PROGRAM_CHECK(__ocl_initialize_1);
	__ocl_initialize_2 = oclCreateKernel(__ocl_program, "initialize_2");
	DYN_PROGRAM_CHECK(__ocl_initialize_2);
	__ocl_initialize_3 = oclCreateKernel(__ocl_program, "initialize_3");
	DYN_PROGRAM_CHECK(__ocl_initialize_3);
	__ocl_initialize_4 = oclCreateKernel(__ocl_program, "initialize_4");
	DYN_PROGRAM_CHECK(__ocl_initialize_4);
	__ocl_initialize_5 = oclCreateKernel(__ocl_program, "initialize_5");
	DYN_PROGRAM_CHECK(__ocl_initialize_5);
	__ocl_initialize_6 = oclCreateKernel(__ocl_program, "initialize_6");
	DYN_PROGRAM_CHECK(__ocl_initialize_6);
	__ocl_initialize_7 = oclCreateKernel(__ocl_program, "initialize_7");
	DYN_PROGRAM_CHECK(__ocl_initialize_7);
	__ocl_lhsinit_0 = oclCreateKernel(__ocl_program, "lhsinit_0");
	DYN_PROGRAM_CHECK(__ocl_lhsinit_0);
	__ocl_lhsinit_1 = oclCreateKernel(__ocl_program, "lhsinit_1");
	DYN_PROGRAM_CHECK(__ocl_lhsinit_1);
	__ocl_lhsx_0 = oclCreateKernel(__ocl_program, "lhsx_0");
	DYN_PROGRAM_CHECK(__ocl_lhsx_0);
	__ocl_lhsy_0 = oclCreateKernel(__ocl_program, "lhsy_0");
	DYN_PROGRAM_CHECK(__ocl_lhsy_0);
	__ocl_lhsy_1 = oclCreateKernel(__ocl_program, "lhsy_1");
	DYN_PROGRAM_CHECK(__ocl_lhsy_1);
	__ocl_lhsz_0 = oclCreateKernel(__ocl_program, "lhsz_0");
	DYN_PROGRAM_CHECK(__ocl_lhsz_0);
	__ocl_lhsz_1 = oclCreateKernel(__ocl_program, "lhsz_1");
	DYN_PROGRAM_CHECK(__ocl_lhsz_1);
	__ocl_compute_rhs_0 = oclCreateKernel(__ocl_program, "compute_rhs_0");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_0);
	__ocl_compute_rhs_1 = oclCreateKernel(__ocl_program, "compute_rhs_1");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_1);
	__ocl_compute_rhs_2 = oclCreateKernel(__ocl_program, "compute_rhs_2");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_2);
	__ocl_compute_rhs_3 = oclCreateKernel(__ocl_program, "compute_rhs_3");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_3);
	__ocl_compute_rhs_4 = oclCreateKernel(__ocl_program, "compute_rhs_4");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_4);
	__ocl_compute_rhs_5 = oclCreateKernel(__ocl_program, "compute_rhs_5");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_5);
	__ocl_compute_rhs_6 = oclCreateKernel(__ocl_program, "compute_rhs_6");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_6);
	__ocl_compute_rhs_7 = oclCreateKernel(__ocl_program, "compute_rhs_7");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_7);
	__ocl_compute_rhs_8 = oclCreateKernel(__ocl_program, "compute_rhs_8");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_8);
	__ocl_compute_rhs_9 = oclCreateKernel(__ocl_program, "compute_rhs_9");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_9);
	__ocl_compute_rhs_10 = oclCreateKernel(__ocl_program, "compute_rhs_10");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_10);
	__ocl_compute_rhs_11 = oclCreateKernel(__ocl_program, "compute_rhs_11");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_11);
	__ocl_compute_rhs_12 = oclCreateKernel(__ocl_program, "compute_rhs_12");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_12);
	__ocl_compute_rhs_13 = oclCreateKernel(__ocl_program, "compute_rhs_13");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_13);
	__ocl_compute_rhs_14 = oclCreateKernel(__ocl_program, "compute_rhs_14");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_14);
	__ocl_compute_rhs_15 = oclCreateKernel(__ocl_program, "compute_rhs_15");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_15);
	__ocl_compute_rhs_16 = oclCreateKernel(__ocl_program, "compute_rhs_16");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_16);
	__ocl_compute_rhs_17 = oclCreateKernel(__ocl_program, "compute_rhs_17");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_17);
	__ocl_compute_rhs_18 = oclCreateKernel(__ocl_program, "compute_rhs_18");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_18);
	__ocl_compute_rhs_19 = oclCreateKernel(__ocl_program, "compute_rhs_19");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_19);
	__ocl_compute_rhs_20 = oclCreateKernel(__ocl_program, "compute_rhs_20");
	DYN_PROGRAM_CHECK(__ocl_compute_rhs_20);
	__ocl_x_backsubstitute_0 =
	    oclCreateKernel(__ocl_program, "x_backsubstitute_0");
	DYN_PROGRAM_CHECK(__ocl_x_backsubstitute_0);
	__ocl_x_solve_cell_0 = oclCreateKernel(__ocl_program, "x_solve_cell_0");
	DYN_PROGRAM_CHECK(__ocl_x_solve_cell_0);
	__ocl_x_solve_cell_1 = oclCreateKernel(__ocl_program, "x_solve_cell_1");
	DYN_PROGRAM_CHECK(__ocl_x_solve_cell_1);
	__ocl_x_solve_cell_2 = oclCreateKernel(__ocl_program, "x_solve_cell_2");
	DYN_PROGRAM_CHECK(__ocl_x_solve_cell_2);
	__ocl_y_backsubstitute_0 =
	    oclCreateKernel(__ocl_program, "y_backsubstitute_0");
	DYN_PROGRAM_CHECK(__ocl_y_backsubstitute_0);
	__ocl_y_solve_cell_0 = oclCreateKernel(__ocl_program, "y_solve_cell_0");
	DYN_PROGRAM_CHECK(__ocl_y_solve_cell_0);
	__ocl_y_solve_cell_1 = oclCreateKernel(__ocl_program, "y_solve_cell_1");
	DYN_PROGRAM_CHECK(__ocl_y_solve_cell_1);
	__ocl_y_solve_cell_2 = oclCreateKernel(__ocl_program, "y_solve_cell_2");
	DYN_PROGRAM_CHECK(__ocl_y_solve_cell_2);
	__ocl_z_backsubstitute_0 =
	    oclCreateKernel(__ocl_program, "z_backsubstitute_0");
	DYN_PROGRAM_CHECK(__ocl_z_backsubstitute_0);
	__ocl_z_solve_cell_0 = oclCreateKernel(__ocl_program, "z_solve_cell_0");
	DYN_PROGRAM_CHECK(__ocl_z_solve_cell_0);
	__ocl_z_solve_cell_1 = oclCreateKernel(__ocl_program, "z_solve_cell_1");
	DYN_PROGRAM_CHECK(__ocl_z_solve_cell_1);
	__ocl_z_solve_cell_2 = oclCreateKernel(__ocl_program, "z_solve_cell_2");
	DYN_PROGRAM_CHECK(__ocl_z_solve_cell_2);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_u =
	    oclCreateBuffer(u, (5 * 65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_u, -1);
	__ocl_buffer_rhs =
	    oclCreateBuffer(rhs, (5 * 65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rhs, -1);
	__ocl_buffer_forcing =
	    oclCreateBuffer(forcing, (6 * 65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_forcing, -1);
	__ocl_buffer_grid_points =
	    oclCreateBuffer(grid_points, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_grid_points, -1);
	__ocl_buffer_ue = oclCreateBuffer(ue, (5 * 64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ue, -1);
	__ocl_buffer_buf = oclCreateBuffer(buf, (5 * 64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_buf, -1);
	__ocl_buffer_cuf = oclCreateBuffer(cuf, (64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_cuf, -1);
	__ocl_buffer_q = oclCreateBuffer(q, (64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_q, -1);
	__ocl_buffer_ce = oclCreateBuffer(ce, (5 * 13) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ce, -1);
	__ocl_buffer_lhs =
	    oclCreateBuffer(lhs, (3 * 5 * 5 * 65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_lhs, -1);
	__ocl_buffer_fjac =
	    oclCreateBuffer(fjac, (5 * 5 * 65 * 65 * 64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_fjac, -1);
	__ocl_buffer_njac =
	    oclCreateBuffer(njac, (5 * 5 * 65 * 65 * 64) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_njac, -1);
	__ocl_buffer_rho_i =
	    oclCreateBuffer(rho_i, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rho_i, -1);
	__ocl_buffer_us = oclCreateBuffer(us, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_us, -1);
	__ocl_buffer_vs = oclCreateBuffer(vs, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_vs, -1);
	__ocl_buffer_ws = oclCreateBuffer(ws, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ws, -1);
	__ocl_buffer_square =
	    oclCreateBuffer(square, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_square, -1);
	__ocl_buffer_qs = oclCreateBuffer(qs, (65 * 65 * 65) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_qs, -1);
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_rhs);
	oclHostWrites(__ocl_buffer_forcing);
	oclHostWrites(__ocl_buffer_grid_points);
	oclHostWrites(__ocl_buffer_ue);
	oclHostWrites(__ocl_buffer_buf);
	oclHostWrites(__ocl_buffer_cuf);
	oclHostWrites(__ocl_buffer_q);
	oclHostWrites(__ocl_buffer_ce);
	oclHostWrites(__ocl_buffer_lhs);
	oclHostWrites(__ocl_buffer_fjac);
	oclHostWrites(__ocl_buffer_njac);
	oclHostWrites(__ocl_buffer_rho_i);
	oclHostWrites(__ocl_buffer_us);
	oclHostWrites(__ocl_buffer_vs);
	oclHostWrites(__ocl_buffer_ws);
	oclHostWrites(__ocl_buffer_square);
	oclHostWrites(__ocl_buffer_qs);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_u);
	oclReleaseBuffer(__ocl_buffer_rhs);
	oclReleaseBuffer(__ocl_buffer_forcing);
	oclReleaseBuffer(__ocl_buffer_grid_points);
	oclReleaseBuffer(__ocl_buffer_ue);
	oclReleaseBuffer(__ocl_buffer_buf);
	oclReleaseBuffer(__ocl_buffer_cuf);
	oclReleaseBuffer(__ocl_buffer_q);
	oclReleaseBuffer(__ocl_buffer_ce);
	oclReleaseBuffer(__ocl_buffer_lhs);
	oclReleaseBuffer(__ocl_buffer_fjac);
	oclReleaseBuffer(__ocl_buffer_njac);
	oclReleaseBuffer(__ocl_buffer_rho_i);
	oclReleaseBuffer(__ocl_buffer_us);
	oclReleaseBuffer(__ocl_buffer_vs);
	oclReleaseBuffer(__ocl_buffer_ws);
	oclReleaseBuffer(__ocl_buffer_square);
	oclReleaseBuffer(__ocl_buffer_qs);
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_rhs);
	oclHostWrites(__ocl_buffer_forcing);
	oclHostWrites(__ocl_buffer_grid_points);
	oclHostWrites(__ocl_buffer_ue);
	oclHostWrites(__ocl_buffer_buf);
	oclHostWrites(__ocl_buffer_cuf);
	oclHostWrites(__ocl_buffer_q);
	oclHostWrites(__ocl_buffer_ce);
	oclHostWrites(__ocl_buffer_lhs);
	oclHostWrites(__ocl_buffer_fjac);
	oclHostWrites(__ocl_buffer_njac);
	oclHostWrites(__ocl_buffer_rho_i);
	oclHostWrites(__ocl_buffer_us);
	oclHostWrites(__ocl_buffer_vs);
	oclHostWrites(__ocl_buffer_ws);
	oclHostWrites(__ocl_buffer_square);
	oclHostWrites(__ocl_buffer_qs);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
