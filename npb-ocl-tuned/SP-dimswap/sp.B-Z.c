//-------------------------------------------------------------------------------
//Host code 
//Generated at : Tue Aug  7 13:28:43 2012
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
static void ninvr();
static void pinvr();
static void compute_rhs();
static void set_constants();
static void txinvr();
static void tzetar();
static void verify(int no_time_steps, char *class,
		   ocl_buffer * __ocl_buffer_class, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified);
static void x_solve();
static void y_solve();
static void z_solve();
int main(int argc, char **argv, ocl_buffer * __ocl_buffer_argv)
{
	{
		int niter, step;
		double mflops, tmax;
		int nthreads = 1;
		boolean verified;
		char class;
		FILE *fp;
		printf
		    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - SP Benchmark\n\n");
		fp = fopen("inputsp.data", "r");
		if (fp != ((void *)0)) {
			printf(" Reading from input file inputsp.data\n");
			fscanf(fp, "%d", &niter);
			while (fgetc(fp) != '\n') ;
			fscanf(fp, "%lf", &dt);
			while (fgetc(fp) != '\n') ;
			fscanf(fp, "%d%d%d", &grid_points[0], &grid_points[1],
			       &grid_points[2]);
			fclose(fp);
		} else {
			printf
			    (" No input file inputsp.data. Using compiled defaults");
			niter = 400;
			dt = 0.001;
			grid_points[0] = 102;
			grid_points[1] = 102;
			grid_points[2] = 102;
		}
		printf(" Size: %3dx%3dx%3d\n", grid_points[0], grid_points[1],
		       grid_points[2]);
		printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);
		if ((grid_points[0] > 102) || (grid_points[1] > 102)
		    || (grid_points[2] > 102)) {
			printf("%d, %d, %d\n", grid_points[0], grid_points[1],
			       grid_points[2]);
			printf
			    (" Problem size too big for compiled array sizes\n");
			exit(1);
		}
		set_constants();
		init_ocl_runtime();
		initialize();
		lhsinit();
		exact_rhs();
		{
			adi();
			sync_ocl_buffers();
		}
		initialize();
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
		sync_ocl_buffers();
		timer_stop(1);
		tmax = timer_read(1);
		verify(niter, &class, NULL, &verified, NULL);
		if (tmax != 0) {
			mflops =
			    (881.174 * pow((double)102, 3.0) -
			     4683.91 * (((double)102) * ((double)102)) +
			     11484.5 * (double)102 -
			     19272.4) * (double)niter / (tmax * 1000000.0);
		} else {
			mflops = 0.0;
		}
		release_ocl_buffers();
		c_print_results("SP", class, grid_points[0], grid_points[1],
				grid_points[2], niter, nthreads, tmax, mflops,
				"          floating point", verified, "2.3",
				"07 Aug 2012", "gcc", "gcc", "(none)",
				"-I../common", "-std=c99 -O3 -fopenmp",
				"-lm -fopenmp", "(none)");
#ifdef PROFILE_SWAP
        printf ("t_swap: %fms\n", t_swap);
        printf ("t_z: %fms\n", t_z);
        printf ("t_rhs: %fms\n", t_rhs);
#endif
	}
}

static void add()
{
	int i, j, k, m;
	//--------------------------------------------------------------
	//Loop defined at line 203 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_add_0, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_add_0, 1, __ocl_buffer_rhs);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_add_0, 2, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_add_0, 3, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	txinvr();
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
		for (i = 0; i <= grid_points[0] - 1; i++) {
			xi = (double)i *dnxm1;
			for (j = 0; j <= grid_points[1] - 1; j++) {
				eta = (double)j *dnym1;
				for (k = 0; k <= grid_points[2] - 1; k++) {
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
			for (d = 0; d < 3; d++) {
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
		for (i = 0; i <= grid_points[0] - 2; i++) {
			for (j = 0; j <= grid_points[1] - 2; j++) {
				for (k = 0; k <= grid_points[2] - 2; k++) {
					for (m = 0; m < 5; m++) {
						add = rhs[m][i][j][k];
						rms[m] = rms[m] + add * add;
					}
				}
			}
		}
		for (m = 0; m < 5; m++) {
			for (d = 0; d < 3; d++) {
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
	for (m = 0; m < 5; m++) {
		for (i = 0; i <= grid_points[0] - 1; i++) {
			for (j = 0; j <= grid_points[1] - 1; j++) {
				for (k = 0; k <= grid_points[2] - 1; k++) {
					forcing[m][i][j][k] = 0.0;
				}
			}
		}
	}
	for (k = 1; k <= grid_points[2] - 2; k++) {
		zeta = (double)k *dnzm1;
		for (j = 1; j <= grid_points[1] - 2; j++) {
			eta = (double)j *dnym1;
			for (i = 0; i <= grid_points[0] - 1; i++) {
				xi = (double)i *dnxm1;

				exact_solution(xi, eta, zeta, dtemp,
					       __ocl_buffer_dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][i] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m < 5; m++) {
					buf[m][i] = dtpp * dtemp[m];
				}
				cuf[i] = buf[1][i] * buf[1][i];
				buf[0][i] =
				    cuf[i] + buf[2][i] * buf[2][i] +
				    buf[3][i] * buf[3][i];
				q[i] =
				    0.5 * (buf[1][i] * ue[1][i] +
					   buf[2][i] * ue[2][i] +
					   buf[3][i] * ue[3][i]);
			}
			for (i = 1; i <= grid_points[0] - 2; i++) {
				im1 = i - 1;
				ip1 = i + 1;
				forcing[0][i][j][k] =
				    forcing[0][i][j][k] - tx2 * (ue[1][ip1] -
								 ue[1][im1]) +
				    dx1tx1 * (ue[0][ip1] - 2.0 * ue[0][i] +
					      ue[0][im1]);
				forcing[1][i][j][k] =
				    forcing[1][i][j][k] -
				    tx2 *
				    ((ue[1][ip1] * buf[1][ip1] +
				      c2 * (ue[4][ip1] - q[ip1])) -
				     (ue[1][im1] * buf[1][im1] +
				      c2 * (ue[4][im1] - q[im1]))) +
				    xxcon1 * (buf[1][ip1] - 2.0 * buf[1][i] +
					      buf[1][im1]) +
				    dx2tx1 * (ue[1][ip1] - 2.0 * ue[1][i] +
					      ue[1][im1]);
				forcing[2][i][j][k] =
				    forcing[2][i][j][k] -
				    tx2 * (ue[2][ip1] * buf[1][ip1] -
					   ue[2][im1] * buf[1][im1]) +
				    xxcon2 * (buf[2][ip1] - 2.0 * buf[2][i] +
					      buf[2][im1]) +
				    dx3tx1 * (ue[2][ip1] - 2.0 * ue[2][i] +
					      ue[2][im1]);
				forcing[3][i][j][k] =
				    forcing[3][i][j][k] -
				    tx2 * (ue[3][ip1] * buf[1][ip1] -
					   ue[3][im1] * buf[1][im1]) +
				    xxcon2 * (buf[3][ip1] - 2.0 * buf[3][i] +
					      buf[3][im1]) +
				    dx4tx1 * (ue[3][ip1] - 2.0 * ue[3][i] +
					      ue[3][im1]);
				forcing[4][i][j][k] =
				    forcing[4][i][j][k] -
				    tx2 * (buf[1][ip1] *
					   (c1 * ue[4][ip1] - c2 * q[ip1]) -
					   buf[1][im1] * (c1 * ue[4][im1] -
							  c2 * q[im1])) +
				    0.5 * xxcon3 * (buf[0][ip1] -
						    2.0 * buf[0][i] +
						    buf[0][im1]) +
				    xxcon4 * (cuf[ip1] - 2.0 * cuf[i] +
					      cuf[im1]) +
				    xxcon5 * (buf[4][ip1] - 2.0 * buf[4][i] +
					      buf[4][im1]) +
				    dx5tx1 * (ue[4][ip1] - 2.0 * ue[4][i] +
					      ue[4][im1]);
			}
			for (m = 0; m < 5; m++) {
				i = 1;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (5.0 * ue[m][i] -
					    4.0 * ue[m][i + 1] + ue[m][i + 2]);
				i = 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (-4.0 * ue[m][i - 1] +
					    6.0 * ue[m][i] - 4.0 * ue[m][i +
									 1] +
					    ue[m][i + 2]);
			}
			for (m = 0; m < 5; m++) {
				for (i = 3; i <= grid_points[0] - 4; i++) {
					forcing[m][i][j][k] =
					    forcing[m][i][j][k] -
					    dssp * (ue[m][i - 2] -
						    4.0 * ue[m][i - 1] +
						    6.0 * ue[m][i] -
						    4.0 * ue[m][i + 1] +
						    ue[m][i + 2]);
				}
			}
			for (m = 0; m < 5; m++) {
				i = grid_points[0] - 3;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][i - 2] -
								  4.0 *
								  ue[m][i - 1] +
								  6.0 *
								  ue[m][i] -
								  4.0 *
								  ue[m][i + 1]);
				i = grid_points[0] - 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][i - 2] -
								  4.0 *
								  ue[m][i - 1] +
								  5.0 *
								  ue[m][i]);
			}
		}
	}
	for (k = 1; k <= grid_points[2] - 2; k++) {
		zeta = (double)k *dnzm1;
		for (i = 1; i <= grid_points[0] - 2; i++) {
			xi = (double)i *dnxm1;
			for (j = 0; j <= grid_points[1] - 1; j++) {
				eta = (double)j *dnym1;
				exact_solution(xi, eta, zeta, dtemp,
					       __ocl_buffer_dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][j] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m < 5; m++) {
					buf[m][j] = dtpp * dtemp[m];
				}
				cuf[j] = buf[2][j] * buf[2][j];
				buf[0][j] =
				    cuf[j] + buf[1][j] * buf[1][j] +
				    buf[3][j] * buf[3][j];
				q[j] =
				    0.5 * (buf[1][j] * ue[1][j] +
					   buf[2][j] * ue[2][j] +
					   buf[3][j] * ue[3][j]);
			}
			for (j = 1; j <= grid_points[1] - 2; j++) {
				jm1 = j - 1;
				jp1 = j + 1;
				forcing[0][i][j][k] =
				    forcing[0][i][j][k] - ty2 * (ue[2][jp1] -
								 ue[2][jm1]) +
				    dy1ty1 * (ue[0][jp1] - 2.0 * ue[0][j] +
					      ue[0][jm1]);
				forcing[1][i][j][k] =
				    forcing[1][i][j][k] -
				    ty2 * (ue[1][jp1] * buf[2][jp1] -
					   ue[1][jm1] * buf[2][jm1]) +
				    yycon2 * (buf[1][jp1] - 2.0 * buf[1][j] +
					      buf[1][jm1]) +
				    dy2ty1 * (ue[1][jp1] - 2.0 * ue[1][j] +
					      ue[1][jm1]);
				forcing[2][i][j][k] =
				    forcing[2][i][j][k] -
				    ty2 *
				    ((ue[2][jp1] * buf[2][jp1] +
				      c2 * (ue[4][jp1] - q[jp1])) -
				     (ue[2][jm1] * buf[2][jm1] +
				      c2 * (ue[4][jm1] - q[jm1]))) +
				    yycon1 * (buf[2][jp1] - 2.0 * buf[2][j] +
					      buf[2][jm1]) +
				    dy3ty1 * (ue[2][jp1] - 2.0 * ue[2][j] +
					      ue[2][jm1]);
				forcing[3][i][j][k] =
				    forcing[3][i][j][k] -
				    ty2 * (ue[3][jp1] * buf[2][jp1] -
					   ue[3][jm1] * buf[2][jm1]) +
				    yycon2 * (buf[3][jp1] - 2.0 * buf[3][j] +
					      buf[3][jm1]) +
				    dy4ty1 * (ue[3][jp1] - 2.0 * ue[3][j] +
					      ue[3][jm1]);
				forcing[4][i][j][k] =
				    forcing[4][i][j][k] -
				    ty2 * (buf[2][jp1] *
					   (c1 * ue[4][jp1] - c2 * q[jp1]) -
					   buf[2][jm1] * (c1 * ue[4][jm1] -
							  c2 * q[jm1])) +
				    0.5 * yycon3 * (buf[0][jp1] -
						    2.0 * buf[0][j] +
						    buf[0][jm1]) +
				    yycon4 * (cuf[jp1] - 2.0 * cuf[j] +
					      cuf[jm1]) +
				    yycon5 * (buf[4][jp1] - 2.0 * buf[4][j] +
					      buf[4][jm1]) +
				    dy5ty1 * (ue[4][jp1] - 2.0 * ue[4][j] +
					      ue[4][jm1]);
			}
			for (m = 0; m < 5; m++) {
				j = 1;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (5.0 * ue[m][j] -
					    4.0 * ue[m][j + 1] + ue[m][j + 2]);
				j = 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (-4.0 * ue[m][j - 1] +
					    6.0 * ue[m][j] - 4.0 * ue[m][j +
									 1] +
					    ue[m][j + 2]);
			}
			for (m = 0; m < 5; m++) {
				for (j = 3; j <= grid_points[1] - 4; j++) {
					forcing[m][i][j][k] =
					    forcing[m][i][j][k] -
					    dssp * (ue[m][j - 2] -
						    4.0 * ue[m][j - 1] +
						    6.0 * ue[m][j] -
						    4.0 * ue[m][j + 1] +
						    ue[m][j + 2]);
				}
			}
			for (m = 0; m < 5; m++) {
				j = grid_points[1] - 3;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][j - 2] -
								  4.0 *
								  ue[m][j - 1] +
								  6.0 *
								  ue[m][j] -
								  4.0 *
								  ue[m][j + 1]);
				j = grid_points[1] - 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][j - 2] -
								  4.0 *
								  ue[m][j - 1] +
								  5.0 *
								  ue[m][j]);
			}
		}
	}
	for (j = 1; j <= grid_points[1] - 2; j++) {
		eta = (double)j *dnym1;
		for (i = 1; i <= grid_points[0] - 2; i++) {
			xi = (double)i *dnxm1;
			for (k = 0; k <= grid_points[2] - 1; k++) {
				zeta = (double)k *dnzm1;
				exact_solution(xi, eta, zeta, dtemp,
					       __ocl_buffer_dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][k] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m < 5; m++) {
					buf[m][k] = dtpp * dtemp[m];
				}
				cuf[k] = buf[3][k] * buf[3][k];
				buf[0][k] =
				    cuf[k] + buf[1][k] * buf[1][k] +
				    buf[2][k] * buf[2][k];
				q[k] =
				    0.5 * (buf[1][k] * ue[1][k] +
					   buf[2][k] * ue[2][k] +
					   buf[3][k] * ue[3][k]);
			}
			for (k = 1; k <= grid_points[2] - 2; k++) {
				km1 = k - 1;
				kp1 = k + 1;
				forcing[0][i][j][k] =
				    forcing[0][i][j][k] - tz2 * (ue[3][kp1] -
								 ue[3][km1]) +
				    dz1tz1 * (ue[0][kp1] - 2.0 * ue[0][k] +
					      ue[0][km1]);
				forcing[1][i][j][k] =
				    forcing[1][i][j][k] -
				    tz2 * (ue[1][kp1] * buf[3][kp1] -
					   ue[1][km1] * buf[3][km1]) +
				    zzcon2 * (buf[1][kp1] - 2.0 * buf[1][k] +
					      buf[1][km1]) +
				    dz2tz1 * (ue[1][kp1] - 2.0 * ue[1][k] +
					      ue[1][km1]);
				forcing[2][i][j][k] =
				    forcing[2][i][j][k] -
				    tz2 * (ue[2][kp1] * buf[3][kp1] -
					   ue[2][km1] * buf[3][km1]) +
				    zzcon2 * (buf[2][kp1] - 2.0 * buf[2][k] +
					      buf[2][km1]) +
				    dz3tz1 * (ue[2][kp1] - 2.0 * ue[2][k] +
					      ue[2][km1]);
				forcing[3][i][j][k] =
				    forcing[3][i][j][k] -
				    tz2 *
				    ((ue[3][kp1] * buf[3][kp1] +
				      c2 * (ue[4][kp1] - q[kp1])) -
				     (ue[3][km1] * buf[3][km1] +
				      c2 * (ue[4][km1] - q[km1]))) +
				    zzcon1 * (buf[3][kp1] - 2.0 * buf[3][k] +
					      buf[3][km1]) +
				    dz4tz1 * (ue[3][kp1] - 2.0 * ue[3][k] +
					      ue[3][km1]);
				forcing[4][i][j][k] =
				    forcing[4][i][j][k] -
				    tz2 * (buf[3][kp1] *
					   (c1 * ue[4][kp1] - c2 * q[kp1]) -
					   buf[3][km1] * (c1 * ue[4][km1] -
							  c2 * q[km1])) +
				    0.5 * zzcon3 * (buf[0][kp1] -
						    2.0 * buf[0][k] +
						    buf[0][km1]) +
				    zzcon4 * (cuf[kp1] - 2.0 * cuf[k] +
					      cuf[km1]) +
				    zzcon5 * (buf[4][kp1] - 2.0 * buf[4][k] +
					      buf[4][km1]) +
				    dz5tz1 * (ue[4][kp1] - 2.0 * ue[4][k] +
					      ue[4][km1]);
			}
			for (m = 0; m < 5; m++) {
				k = 1;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (5.0 * ue[m][k] -
					    4.0 * ue[m][k + 1] + ue[m][k + 2]);
				k = 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] -
				    dssp * (-4.0 * ue[m][k - 1] +
					    6.0 * ue[m][k] - 4.0 * ue[m][k +
									 1] +
					    ue[m][k + 2]);
			}
			for (m = 0; m < 5; m++) {
				for (k = 3; k <= grid_points[2] - 4; k++) {
					forcing[m][i][j][k] =
					    forcing[m][i][j][k] -
					    dssp * (ue[m][k - 2] -
						    4.0 * ue[m][k - 1] +
						    6.0 * ue[m][k] -
						    4.0 * ue[m][k + 1] +
						    ue[m][k + 2]);
				}
			}
			for (m = 0; m < 5; m++) {
				k = grid_points[2] - 3;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][k - 2] -
								  4.0 *
								  ue[m][k - 1] +
								  6.0 *
								  ue[m][k] -
								  4.0 *
								  ue[m][k + 1]);
				k = grid_points[2] - 2;
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][k - 2] -
								  4.0 *
								  ue[m][k - 1] +
								  5.0 *
								  ue[m][k]);
			}
		}
	}
	for (m = 0; m < 5; m++) {
		for (i = 1; i <= grid_points[0] - 2; i++) {
			for (j = 1; j <= grid_points[1] - 2; j++) {
				for (k = 1; k <= grid_points[2] - 2; k++) {
					forcing[m][i][j][k] =
					    -1.0 * forcing[m][i][j][k];
				}
			}
		}
	}
}

static void exact_solution(double xi, double eta, double zeta, double dtemp[5],
			   ocl_buffer * __ocl_buffer_dtemp)
{
	{
		int m;
		for (m = 0; m < 5; m++) {
			dtemp[m] =
			    ce[0][m] + xi * (ce[1][m] +
					     xi * (ce[4][m] +
						   xi * (ce[7][m] +
							 xi * ce[10][m]))) +
			    eta * (ce[2][m] +
				   eta * (ce[5][m] +
					  eta * (ce[8][m] + eta * ce[11][m]))) +
			    zeta * (ce[3][m] +
				    zeta * (ce[6][m] +
					    zeta * (ce[9][m] +
						    zeta * ce[12][m])));
		}
	}
}

static void initialize()
{
	int i, j, k, m, ix, iy, iz;
	double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];
	DECLARE_LOCALVAR_OCL_BUFFER(Pface, double, (2 * 3 * 5));
	DECLARE_LOCALVAR_OCL_BUFFER(temp, double, (5));
	for (i = 0; i <= 102 - 1; i++) {
		for (j = 0; j <= 102 - 1; j++) {
			for (k = 0; k <= 102 - 1; k++) {
				u[0][i][j][k] = 1.0;
				u[1][i][j][k] = 0.0;
				u[2][i][j][k] = 0.0;
				u[3][i][j][k] = 0.0;
				u[4][i][j][k] = 1.0;
			}
		}
	}
	for (i = 0; i <= grid_points[0] - 1; i++) {
		xi = (double)i *dnxm1;
		for (j = 0; j <= grid_points[1] - 1; j++) {
			eta = (double)j *dnym1;
			for (k = 0; k <= grid_points[2] - 1; k++) {
				zeta = (double)k *dnzm1;
				for (ix = 0; ix < 2; ix++) {

					exact_solution((double)ix, eta, zeta,
						       &Pface[ix][0][0],
						       __ocl_buffer_Pface);
				}
				for (iy = 0; iy < 2; iy++) {
					exact_solution(xi, (double)iy, zeta,
						       &Pface[iy][1][0],
						       __ocl_buffer_Pface);
				}
				for (iz = 0; iz < 2; iz++) {
					exact_solution(xi, eta, (double)iz,
						       &Pface[iz][2][0],
						       __ocl_buffer_Pface);
				}
				for (m = 0; m < 5; m++) {
					Pxi =
					    xi * Pface[1][0][m] + (1.0 -
								   xi) *
					    Pface[0][0][m];
					Peta =
					    eta * Pface[1][1][m] + (1.0 -
								    eta) *
					    Pface[0][1][m];
					Pzeta =
					    zeta * Pface[1][2][m] + (1.0 -
								     zeta) *
					    Pface[0][2][m];
					u[m][i][j][k] =
					    Pxi + Peta + Pzeta - Pxi * Peta -
					    Pxi * Pzeta - Peta * Pzeta +
					    Pxi * Peta * Pzeta;
				}
			}
		}
	}
	xi = 0.0;
	i = 0;
	for (j = 0; j < grid_points[1]; j++) {
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[2]; k++) {
			zeta = (double)k *dnzm1;

			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	xi = 1.0;
	i = grid_points[0] - 1;
	for (j = 0; j < grid_points[1]; j++) {
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[2]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	eta = 0.0;
	j = 0;
	for (i = 0; i < grid_points[0]; i++) {
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[2]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	eta = 1.0;
	j = grid_points[1] - 1;
	for (i = 0; i < grid_points[0]; i++) {
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[2]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	zeta = 0.0;
	k = 0;
	for (i = 0; i < grid_points[0]; i++) {
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[1]; j++) {
			eta = (double)j *dnym1;
			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	zeta = 1.0;
	k = grid_points[2] - 1;
	for (i = 0; i < grid_points[0]; i++) {
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[1]; j++) {
			eta = (double)j *dnym1;
			exact_solution(xi, eta, zeta, temp, __ocl_buffer_temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
}

static void lhsinit()
{
	int i, j, k, n;
	for (n = 0; n < 15; n++) {
		//--------------------------------------------------------------
		//Loop defined at line 870 of sp.c
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
			oclSetKernelArgBuffer(__ocl_lhsinit_0, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_lhsinit_0, 1, sizeof(int), &n);
			int __ocl_k_bound = grid_points[2];
			oclSetKernelArg(__ocl_lhsinit_0, 2, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1];
			oclSetKernelArg(__ocl_lhsinit_0, 3, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = grid_points[0];
			oclSetKernelArg(__ocl_lhsinit_0, 4, sizeof(int),
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

	}
	for (n = 0; n < 3; n++) {
		//--------------------------------------------------------------
		//Loop defined at line 887 of sp.c
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
			oclSetKernelArgBuffer(__ocl_lhsinit_1, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_lhsinit_1, 1, sizeof(int), &n);
			int __ocl_k_bound = grid_points[2];
			oclSetKernelArg(__ocl_lhsinit_1, 2, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1];
			oclSetKernelArg(__ocl_lhsinit_1, 3, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = grid_points[0];
			oclSetKernelArg(__ocl_lhsinit_1, 4, sizeof(int),
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
}

static void lhsx()
{
	double ru1;
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 916 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 1) - (0) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_lhsx_0, 0, sizeof(double), &c3c4);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 1, __ocl_buffer_rho_i);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 2, __ocl_buffer_cv);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 3, __ocl_buffer_us);
		oclSetKernelArgBuffer(__ocl_lhsx_0, 4, __ocl_buffer_rhon);
		oclSetKernelArg(__ocl_lhsx_0, 5, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_lhsx_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsx_0, 7, sizeof(double), &dx5);
		oclSetKernelArg(__ocl_lhsx_0, 8, sizeof(double), &c1c5);
		oclSetKernelArg(__ocl_lhsx_0, 9, sizeof(double), &dxmax);
		oclSetKernelArg(__ocl_lhsx_0, 10, sizeof(double), &dx1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_0, 11, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_0, 12, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_lhsx_0, 13, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_cv);
		oclDevWrites(__ocl_buffer_rhon);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rho_i);
		oclDevReads(__ocl_buffer_us);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsx_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 931 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_1, 1, sizeof(double), &dttx2);
		oclSetKernelArgBuffer(__ocl_lhsx_1, 2, __ocl_buffer_cv);
		oclSetKernelArg(__ocl_lhsx_1, 3, sizeof(double), &dttx1);
		oclSetKernelArgBuffer(__ocl_lhsx_1, 4, __ocl_buffer_rhon);
		oclSetKernelArg(__ocl_lhsx_1, 5, sizeof(double), &c2dttx1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_1, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_1, 7, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsx_1, 8, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_cv);
		oclDevReads(__ocl_buffer_rhon);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsx_1, 3, _ocl_gws);
	}

	i = 1;
	//--------------------------------------------------------------
	//Loop defined at line 952 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_2, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_2, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_lhsx_2, 2, sizeof(double), &comz5);
		oclSetKernelArg(__ocl_lhsx_2, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsx_2, 4, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsx_2, 5, sizeof(double), &comz6);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_2, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_2, 7, sizeof(int), &__ocl_j_bound);
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

		oclRunKernel(__ocl_lhsx_2, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 968 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 4) - (3) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_3, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_3, 1, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsx_3, 2, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsx_3, 3, sizeof(double), &comz6);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_3, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_3, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 4;
		oclSetKernelArg(__ocl_lhsx_3, 6, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsx_3, 3, _ocl_gws);
	}

	i = grid_points[0] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 985 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_4, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_4, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_lhsx_4, 2, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsx_4, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsx_4, 4, sizeof(double), &comz6);
		oclSetKernelArg(__ocl_lhsx_4, 5, sizeof(double), &comz5);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_4, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_4, 7, sizeof(int), &__ocl_j_bound);
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

		oclRunKernel(__ocl_lhsx_4, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1006 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsx_5, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsx_5, 1, sizeof(double), &dttx2);
		oclSetKernelArgBuffer(__ocl_lhsx_5, 2, __ocl_buffer_speed);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsx_5, 3, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsx_5, 4, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsx_5, 5, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_speed);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsx_5, 3, _ocl_gws);
	}

}

static void lhsy()
{
	double ru1;
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 1048 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 1) - (0) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_lhsy_0, 0, sizeof(double), &c3c4);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 1, __ocl_buffer_rho_i);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 2, __ocl_buffer_cv);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 3, __ocl_buffer_vs);
		oclSetKernelArgBuffer(__ocl_lhsy_0, 4, __ocl_buffer_rhoq);
		oclSetKernelArg(__ocl_lhsy_0, 5, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_lhsy_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsy_0, 7, sizeof(double), &dy5);
		oclSetKernelArg(__ocl_lhsy_0, 8, sizeof(double), &c1c5);
		oclSetKernelArg(__ocl_lhsy_0, 9, sizeof(double), &dymax);
		oclSetKernelArg(__ocl_lhsy_0, 10, sizeof(double), &dy1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_0, 11, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_lhsy_0, 12, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_0, 13, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_cv);
		oclDevWrites(__ocl_buffer_rhoq);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rho_i);
		oclDevReads(__ocl_buffer_vs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsy_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1063 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsy_1, 1, sizeof(double), &dtty2);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 2, __ocl_buffer_cv);
		oclSetKernelArg(__ocl_lhsy_1, 3, sizeof(double), &dtty1);
		oclSetKernelArgBuffer(__ocl_lhsy_1, 4, __ocl_buffer_rhoq);
		oclSetKernelArg(__ocl_lhsy_1, 5, sizeof(double), &c2dtty1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_1, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsy_1, 7, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_1, 8, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_cv);
		oclDevReads(__ocl_buffer_rhoq);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsy_1, 3, _ocl_gws);
	}

	j = 1;
	//--------------------------------------------------------------
	//Loop defined at line 1084 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_2, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsy_2, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_lhsy_2, 2, sizeof(double), &comz5);
		oclSetKernelArg(__ocl_lhsy_2, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsy_2, 4, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsy_2, 5, sizeof(double), &comz6);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_2, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_2, 7, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsy_2, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1101 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 4) - (3) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_3, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsy_3, 1, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsy_3, 2, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsy_3, 3, sizeof(double), &comz6);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_3, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 4;
		oclSetKernelArg(__ocl_lhsy_3, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_3, 6, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsy_3, 3, _ocl_gws);
	}

	j = grid_points[1] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 1117 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_4, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsy_4, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_lhsy_4, 2, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsy_4, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsy_4, 4, sizeof(double), &comz6);
		oclSetKernelArg(__ocl_lhsy_4, 5, sizeof(double), &comz5);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_4, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_4, 7, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsy_4, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1137 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsy_5, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsy_5, 1, sizeof(double), &dtty2);
		oclSetKernelArgBuffer(__ocl_lhsy_5, 2, __ocl_buffer_speed);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsy_5, 3, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsy_5, 4, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsy_5, 5, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_speed);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsy_5, 3, _ocl_gws);
	}

}

static void lhsz()
{
	double ru1;
	int i, j, k;
	//--------------------------------------------------------------
	//Loop defined at line 1179 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (0) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_lhsz_0, 0, sizeof(double), &c3c4);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 1, __ocl_buffer_rho_i);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 2, __ocl_buffer_cv);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 3, __ocl_buffer_ws);
		oclSetKernelArgBuffer(__ocl_lhsz_0, 4, __ocl_buffer_rhos);
		oclSetKernelArg(__ocl_lhsz_0, 5, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_lhsz_0, 6, sizeof(double), &con43);
		oclSetKernelArg(__ocl_lhsz_0, 7, sizeof(double), &dz5);
		oclSetKernelArg(__ocl_lhsz_0, 8, sizeof(double), &c1c5);
		oclSetKernelArg(__ocl_lhsz_0, 9, sizeof(double), &dzmax);
		oclSetKernelArg(__ocl_lhsz_0, 10, sizeof(double), &dz1);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_lhsz_0, 11, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_0, 12, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_0, 13, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_cv);
		oclDevWrites(__ocl_buffer_rhos);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rho_i);
		oclDevReads(__ocl_buffer_ws);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsz_0, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1194 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsz_1, 1, sizeof(double), &dttz2);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 2, __ocl_buffer_cv);
		oclSetKernelArg(__ocl_lhsz_1, 3, sizeof(double), &dttz1);
		oclSetKernelArgBuffer(__ocl_lhsz_1, 4, __ocl_buffer_rhos);
		oclSetKernelArg(__ocl_lhsz_1, 5, sizeof(double), &c2dttz1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsz_1, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_1, 7, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_1, 8, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_cv);
		oclDevReads(__ocl_buffer_rhos);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsz_1, 3, _ocl_gws);
	}

	k = 1;
	//--------------------------------------------------------------
	//Loop defined at line 1215 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_2, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsz_2, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_lhsz_2, 2, sizeof(double), &comz5);
		oclSetKernelArg(__ocl_lhsz_2, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsz_2, 4, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsz_2, 5, sizeof(double), &comz6);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_2, 6, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_2, 7, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsz_2, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1232 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 4) - (3) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_3, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsz_3, 1, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsz_3, 2, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsz_3, 3, sizeof(double), &comz6);
		int __ocl_k_bound = grid_points[2] - 4;
		oclSetKernelArg(__ocl_lhsz_3, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_3, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_3, 6, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsz_3, 3, _ocl_gws);
	}

	k = grid_points[2] - 3;
	//--------------------------------------------------------------
	//Loop defined at line 1249 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_4, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsz_4, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_lhsz_4, 2, sizeof(double), &comz1);
		oclSetKernelArg(__ocl_lhsz_4, 3, sizeof(double), &comz4);
		oclSetKernelArg(__ocl_lhsz_4, 4, sizeof(double), &comz6);
		oclSetKernelArg(__ocl_lhsz_4, 5, sizeof(double), &comz5);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_4, 6, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_4, 7, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_lhsz_4, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 1269 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_lhsz_5, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_lhsz_5, 1, sizeof(double), &dttz2);
		oclSetKernelArgBuffer(__ocl_lhsz_5, 2, __ocl_buffer_speed);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_lhsz_5, 3, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_lhsz_5, 4, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_lhsz_5, 5, sizeof(int), &__ocl_i_bound);
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
		oclDevReads(__ocl_buffer_speed);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_lhsz_5, 3, _ocl_gws);
	}

}

static void ninvr()
{
	int i, j, k;
	double r1, r2, r3, r4, r5, t1, t2;
	//--------------------------------------------------------------
	//Loop defined at line 1309 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_ninvr_0, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_ninvr_0, 1, sizeof(double), &bt);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_ninvr_0, 2, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_ninvr_0, 3, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_ninvr_0, 4, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_ninvr_0, 3, _ocl_gws);
	}

}

static void pinvr()
{
	int i, j, k;
	double r1, r2, r3, r4, r5, t1, t2;
	//--------------------------------------------------------------
	//Loop defined at line 1351 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_pinvr_0, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_pinvr_0, 1, sizeof(double), &bt);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_pinvr_0, 2, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_pinvr_0, 3, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_pinvr_0, 4, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_pinvr_0, 3, _ocl_gws);
	}

}

static void compute_rhs()
{
	int i, j, k, m;
	double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
	//--------------------------------------------------------------
	//Loop defined at line 1395 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (0) + 1;
		_ocl_gws[1] = (grid_points[1] - 1) - (0) + 1;
		_ocl_gws[2] = (grid_points[0] - 1) - (0) + 1;

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
		oclSetKernelArg(__ocl_compute_rhs_0, 7, sizeof(double), &c1c2);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 8,
				      __ocl_buffer_speed);
		oclSetKernelArgBuffer(__ocl_compute_rhs_0, 9,
				      __ocl_buffer_ainv);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_0, 10, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_0, 11, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
		oclSetKernelArg(__ocl_compute_rhs_0, 12, sizeof(int),
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
		oclDevWrites(__ocl_buffer_speed);
		oclDevWrites(__ocl_buffer_ainv);
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
	//Loop defined at line 1427 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 1) - (0) + 1;
		_ocl_gws[1] = (grid_points[1] - 1) - (0) + 1;
		_ocl_gws[2] = (grid_points[0] - 1) - (0) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_1, 0, __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_compute_rhs_1, 1,
				      __ocl_buffer_forcing);
		int __ocl_k_bound = grid_points[2] - 1;
		oclSetKernelArg(__ocl_compute_rhs_1, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 1;
		oclSetKernelArg(__ocl_compute_rhs_1, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 1;
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
	//Loop defined at line 1444 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

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
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_2, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_2, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1507 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_3, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_3, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_3, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_3, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_3, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
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
	//Loop defined at line 1521 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_4, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_4, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_4, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_4, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_4, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
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
	//Loop defined at line 1534 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 3 * 1 - 1) - (3 * 1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_5, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_5, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_5, 2, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_5, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_5, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 3 * 1 - 1;
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
	//Loop defined at line 1551 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_6, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_6, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_6, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_6, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_6, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
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
	//Loop defined at line 1565 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_7, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_7, 1, sizeof(int), &i);
		oclSetKernelArg(__ocl_compute_rhs_7, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_7, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_7, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
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
	//Loop defined at line 1583 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

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
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_8, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_8, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1642 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_9, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_9, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_9, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_9, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_9, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1656 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_10, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_10, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_10, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_10, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_10, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1669 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 3 * 1 - 1) - (3 * 1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_11, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_11, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_11, 2, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_11, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 3 * 1 - 1;
		oclSetKernelArg(__ocl_compute_rhs_11, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1686 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_12, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_12, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_12, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_12, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_12, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1700 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_13, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_13, 1, sizeof(int), &j);
		oclSetKernelArg(__ocl_compute_rhs_13, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_13, 3, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_13, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1718 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

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
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_14, 21, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_14, 22, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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

#ifdef PROFILE_SWAP
    oclSync();
    ocl_timer ti;
    startTimer (&ti);
#endif
	k = 1;
	//--------------------------------------------------------------
	//Loop defined at line 1783 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_15, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_15, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_15, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_15, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_15, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1803 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_16, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_16, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_16, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_16, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_16, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1822 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 3 * 1 - 1) - (3 * 1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_17, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_17, 1, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_17, 2, __ocl_buffer_u);
		int __ocl_k_bound = grid_points[2] - 3 * 1 - 1;
		oclSetKernelArg(__ocl_compute_rhs_17, 3, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_17, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1846 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_18, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_18, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_18, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_18, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_18, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	//Loop defined at line 1866 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_19, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_19, 1, sizeof(int), &k);
		oclSetKernelArg(__ocl_compute_rhs_19, 2, sizeof(double), &dssp);
		oclSetKernelArgBuffer(__ocl_compute_rhs_19, 3, __ocl_buffer_u);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_19, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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

#ifdef PROFILE_SWAP
    oclSync();
    stopTimer (&ti);
    t_rhs += elapsedTime (&ti);
#endif
	//--------------------------------------------------------------
	//Loop defined at line 1890 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_compute_rhs_20, 0,
				      __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_compute_rhs_20, 1, sizeof(double), &dt);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_compute_rhs_20, 2, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_compute_rhs_20, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
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
	ce[1][0] = 0.0;
	ce[2][0] = 0.0;
	ce[3][0] = 4.0;
	ce[4][0] = 5.0;
	ce[5][0] = 3.0;
	ce[6][0] = 0.5;
	ce[7][0] = 0.02;
	ce[8][0] = 0.01;
	ce[9][0] = 0.03;
	ce[10][0] = 0.5;
	ce[11][0] = 0.4;
	ce[12][0] = 0.3;
	ce[0][1] = 1.0;
	ce[1][1] = 0.0;
	ce[2][1] = 0.0;
	ce[3][1] = 0.0;
	ce[4][1] = 1.0;
	ce[5][1] = 2.0;
	ce[6][1] = 3.0;
	ce[7][1] = 0.01;
	ce[8][1] = 0.03;
	ce[9][1] = 0.02;
	ce[10][1] = 0.4;
	ce[11][1] = 0.3;
	ce[12][1] = 0.5;
	ce[0][2] = 2.0;
	ce[1][2] = 2.0;
	ce[2][2] = 0.0;
	ce[3][2] = 0.0;
	ce[4][2] = 0.0;
	ce[5][2] = 2.0;
	ce[6][2] = 3.0;
	ce[7][2] = 0.04;
	ce[8][2] = 0.03;
	ce[9][2] = 0.05;
	ce[10][2] = 0.3;
	ce[11][2] = 0.5;
	ce[12][2] = 0.4;
	ce[0][3] = 2.0;
	ce[1][3] = 2.0;
	ce[2][3] = 0.0;
	ce[3][3] = 0.0;
	ce[4][3] = 0.0;
	ce[5][3] = 2.0;
	ce[6][3] = 3.0;
	ce[7][3] = 0.03;
	ce[8][3] = 0.05;
	ce[9][3] = 0.04;
	ce[10][3] = 0.2;
	ce[11][3] = 0.1;
	ce[12][3] = 0.3;
	ce[0][4] = 5.0;
	ce[1][4] = 4.0;
	ce[2][4] = 3.0;
	ce[3][4] = 2.0;
	ce[4][4] = 0.1;
	ce[5][4] = 0.4;
	ce[6][4] = 0.3;
	ce[7][4] = 0.05;
	ce[8][4] = 0.04;
	ce[9][4] = 0.03;
	ce[10][4] = 0.1;
	ce[11][4] = 0.3;
	ce[12][4] = 0.2;
	c1 = 1.4;
	c2 = 0.4;
	c3 = 0.1;
	c4 = 1.0;
	c5 = 1.4;
	bt = sqrt(0.5);
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

static void txinvr()
{
	int i, j, k;
	double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;
	//--------------------------------------------------------------
	//Loop defined at line 2119 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 0, __ocl_buffer_rho_i);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 1, __ocl_buffer_us);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 2, __ocl_buffer_vs);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 3, __ocl_buffer_ws);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 4, __ocl_buffer_speed);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 5, __ocl_buffer_ainv);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 6, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_txinvr_0, 7, sizeof(double), &c2);
		oclSetKernelArgBuffer(__ocl_txinvr_0, 8, __ocl_buffer_qs);
		oclSetKernelArg(__ocl_txinvr_0, 9, sizeof(double), &bt);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_txinvr_0, 10, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_txinvr_0, 11, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_txinvr_0, 12, sizeof(int),
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
		oclDevReads(__ocl_buffer_rho_i);
		oclDevReads(__ocl_buffer_us);
		oclDevReads(__ocl_buffer_vs);
		oclDevReads(__ocl_buffer_ws);
		oclDevReads(__ocl_buffer_speed);
		oclDevReads(__ocl_buffer_ainv);
		oclDevReads(__ocl_buffer_qs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_txinvr_0, 3, _ocl_gws);
	}

}

static void tzetar()
{
	int i, j, k;
	double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5, btuz,
	    acinv, ac2u, uzik1;
	//--------------------------------------------------------------
	//Loop defined at line 2170 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 0, __ocl_buffer_us);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 1, __ocl_buffer_vs);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 2, __ocl_buffer_ws);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 3, __ocl_buffer_speed);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 4, __ocl_buffer_ainv);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 5, __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 6, __ocl_buffer_u);
		oclSetKernelArg(__ocl_tzetar_0, 7, sizeof(double), &bt);
		oclSetKernelArgBuffer(__ocl_tzetar_0, 8, __ocl_buffer_qs);
		oclSetKernelArg(__ocl_tzetar_0, 9, sizeof(double), &c2iv);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_tzetar_0, 10, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_tzetar_0, 11, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_tzetar_0, 12, sizeof(int),
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
		oclDevReads(__ocl_buffer_vs);
		oclDevReads(__ocl_buffer_ws);
		oclDevReads(__ocl_buffer_speed);
		oclDevReads(__ocl_buffer_ainv);
		oclDevReads(__ocl_buffer_u);
		oclDevReads(__ocl_buffer_qs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_tzetar_0, 3, _ocl_gws);
	}

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
		    && grid_points[2] == 12 && no_time_steps == 100) {
			*class = 'S';
			dtref = 1.5e-2;
			xcrref[0] = 2.7470315451339479e-02;
			xcrref[1] = 1.0360746705285417e-02;
			xcrref[2] = 1.6235745065095532e-02;
			xcrref[3] = 1.5840557224455615e-02;
			xcrref[4] = 3.4849040609362460e-02;
			xceref[0] = 2.7289258557377227e-05;
			xceref[1] = 1.0364446640837285e-05;
			xceref[2] = 1.6154798287166471e-05;
			xceref[3] = 1.5750704994480102e-05;
			xceref[4] = 3.4177666183390531e-05;
		} else if (grid_points[0] == 36 && grid_points[1] == 36
			   && grid_points[2] == 36 && no_time_steps == 400) {
			*class = 'W';
			dtref = 1.5e-3;
			xcrref[0] = 0.1893253733584e-02;
			xcrref[1] = 0.1717075447775e-03;
			xcrref[2] = 0.2778153350936e-03;
			xcrref[3] = 0.2887475409984e-03;
			xcrref[4] = 0.3143611161242e-02;
			xceref[0] = 0.7542088599534e-04;
			xceref[1] = 0.6512852253086e-05;
			xceref[2] = 0.1049092285688e-04;
			xceref[3] = 0.1128838671535e-04;
			xceref[4] = 0.1212845639773e-03;
		} else if (grid_points[0] == 64 && grid_points[1] == 64
			   && grid_points[2] == 64 && no_time_steps == 400) {
			*class = 'A';
			dtref = 1.5e-3;
			xcrref[0] = 2.4799822399300195;
			xcrref[1] = 1.1276337964368832;
			xcrref[2] = 1.5028977888770491;
			xcrref[3] = 1.4217816211695179;
			xcrref[4] = 2.1292113035138280;
			xceref[0] = 1.0900140297820550e-04;
			xceref[1] = 3.7343951769282091e-05;
			xceref[2] = 5.0092785406541633e-05;
			xceref[3] = 4.7671093939528255e-05;
			xceref[4] = 1.3621613399213001e-04;
		} else if (grid_points[0] == 102 && grid_points[1] == 102
			   && grid_points[2] == 102 && no_time_steps == 400) {
			*class = 'B';
			dtref = 1.0e-3;
			xcrref[0] = 0.6903293579998e+02;
			xcrref[1] = 0.3095134488084e+02;
			xcrref[2] = 0.4103336647017e+02;
			xcrref[3] = 0.3864769009604e+02;
			xcrref[4] = 0.5643482272596e+02;
			xceref[0] = 0.9810006190188e-02;
			xceref[1] = 0.1022827905670e-02;
			xceref[2] = 0.1720597911692e-02;
			xceref[3] = 0.1694479428231e-02;
			xceref[4] = 0.1847456263981e-01;
		} else if (grid_points[0] == 162 && grid_points[1] == 162
			   && grid_points[2] == 162 && no_time_steps == 400) {
			*class = 'C';
			dtref = 0.67e-3;
			xcrref[0] = 0.5881691581829e+03;
			xcrref[1] = 0.2454417603569e+03;
			xcrref[2] = 0.3293829191851e+03;
			xcrref[3] = 0.3081924971891e+03;
			xcrref[4] = 0.4597223799176e+03;
			xceref[0] = 0.2598120500183e+00;
			xceref[1] = 0.2590888922315e-01;
			xceref[2] = 0.5132886416320e-01;
			xceref[3] = 0.4806073419454e-01;
			xceref[4] = 0.5483377491301e+00;
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
		} else if (*verified) {
			printf(" Verification Successful\n");
		} else {
			printf(" Verification failed\n");
		}
	}
}

static void x_solve()
{
	int i, j, k, n, i1, i2, m;
	double fac1, fac2;
	lhsx();
	n = 0;
	for (i = 0; i <= grid_points[0] - 3; i++) {
		i1 = i + 1;
		i2 = i + 2;
		//--------------------------------------------------------------
		//Loop defined at line 2517 of sp.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
			_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_x_solve_0, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_x_solve_0, 1, sizeof(int), &n);
			oclSetKernelArg(__ocl_x_solve_0, 2, sizeof(int), &i);
			oclSetKernelArgBuffer(__ocl_x_solve_0, 3,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_x_solve_0, 4, sizeof(int), &i1);
			oclSetKernelArg(__ocl_x_solve_0, 5, sizeof(int), &i2);
			int __ocl_k_bound = grid_points[2] - 2;
			oclSetKernelArg(__ocl_x_solve_0, 6, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1] - 2;
			oclSetKernelArg(__ocl_x_solve_0, 7, sizeof(int),
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

			oclRunKernel(__ocl_x_solve_0, 2, _ocl_gws);
		}

	}
	i = grid_points[0] - 2;
	i1 = grid_points[0] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 2557 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_x_solve_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_x_solve_1, 1, sizeof(int), &n);
		oclSetKernelArg(__ocl_x_solve_1, 2, sizeof(int), &i);
		oclSetKernelArgBuffer(__ocl_x_solve_1, 3, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_x_solve_1, 4, sizeof(int), &i1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_x_solve_1, 5, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_x_solve_1, 6, sizeof(int),
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

		oclRunKernel(__ocl_x_solve_1, 2, _ocl_gws);
	}

	for (m = 3; m < 5; m++) {
		n = (m - 3 + 1) * 5;
		for (i = 0; i <= grid_points[0] - 3; i++) {
			i1 = i + 1;
			i2 = i + 2;
			//--------------------------------------------------------------
			//Loop defined at line 2597 of sp.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
				_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_x_solve_2, 0,
						      __ocl_buffer_lhs);
				oclSetKernelArg(__ocl_x_solve_2, 1, sizeof(int),
						&n);
				oclSetKernelArg(__ocl_x_solve_2, 2, sizeof(int),
						&i);
				oclSetKernelArgBuffer(__ocl_x_solve_2, 3,
						      __ocl_buffer_rhs);
				oclSetKernelArg(__ocl_x_solve_2, 4, sizeof(int),
						&m);
				oclSetKernelArg(__ocl_x_solve_2, 5, sizeof(int),
						&i1);
				oclSetKernelArg(__ocl_x_solve_2, 6, sizeof(int),
						&i2);
				int __ocl_k_bound = grid_points[2] - 2;
				oclSetKernelArg(__ocl_x_solve_2, 7, sizeof(int),
						&__ocl_k_bound);
				int __ocl_j_bound = grid_points[1] - 2;
				oclSetKernelArg(__ocl_x_solve_2, 8, sizeof(int),
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

				oclRunKernel(__ocl_x_solve_2, 2, _ocl_gws);
			}

		}
		i = grid_points[0] - 2;
		i1 = grid_points[0] - 1;
		//--------------------------------------------------------------
		//Loop defined at line 2628 of sp.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
			_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_x_solve_3, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_x_solve_3, 1, sizeof(int), &n);
			oclSetKernelArg(__ocl_x_solve_3, 2, sizeof(int), &i);
			oclSetKernelArgBuffer(__ocl_x_solve_3, 3,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_x_solve_3, 4, sizeof(int), &m);
			oclSetKernelArg(__ocl_x_solve_3, 5, sizeof(int), &i1);
			int __ocl_k_bound = grid_points[2] - 2;
			oclSetKernelArg(__ocl_x_solve_3, 6, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1] - 2;
			oclSetKernelArg(__ocl_x_solve_3, 7, sizeof(int),
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

			oclRunKernel(__ocl_x_solve_3, 2, _ocl_gws);
		}

	}
	i = grid_points[0] - 2;
	i1 = grid_points[0] - 1;
	n = 0;
	//--------------------------------------------------------------
	//Loop defined at line 2660 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (3) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_x_solve_4, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_x_solve_4, 1, sizeof(int), &i);
		oclSetKernelArgBuffer(__ocl_x_solve_4, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_x_solve_4, 3, sizeof(int), &n);
		oclSetKernelArg(__ocl_x_solve_4, 4, sizeof(int), &i1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_x_solve_4, 5, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_x_solve_4, 6, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_x_solve_4, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2672 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (3);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_x_solve_5, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_x_solve_5, 1, sizeof(int), &i);
		oclSetKernelArgBuffer(__ocl_x_solve_5, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_x_solve_5, 3, sizeof(int), &i1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_x_solve_5, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_x_solve_5, 5, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_x_solve_5, 3, _ocl_gws);
	}

	n = 0;
	for (i = grid_points[0] - 3; i >= 0; i--) {
		i1 = i + 1;
		i2 = i + 2;
		//--------------------------------------------------------------
		//Loop defined at line 2693 of sp.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
			_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;
			_ocl_gws[2] = (3) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_x_solve_6, 0,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_x_solve_6, 1, sizeof(int), &i);
			oclSetKernelArgBuffer(__ocl_x_solve_6, 2,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_x_solve_6, 3, sizeof(int), &n);
			oclSetKernelArg(__ocl_x_solve_6, 4, sizeof(int), &i1);
			oclSetKernelArg(__ocl_x_solve_6, 5, sizeof(int), &i2);
			int __ocl_k_bound = grid_points[2] - 2;
			oclSetKernelArg(__ocl_x_solve_6, 6, sizeof(int),
					&__ocl_k_bound);
			int __ocl_j_bound = grid_points[1] - 2;
			oclSetKernelArg(__ocl_x_solve_6, 7, sizeof(int),
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
			oclDevReads(__ocl_buffer_lhs);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_x_solve_6, 3, _ocl_gws);
		}

	}
	for (m = 3; m < 5; m++) {
		n = (m - 3 + 1) * 5;
		for (i = grid_points[0] - 3; i >= 0; i--) {
			i1 = i + 1;
			i2 = i + 2;
			//--------------------------------------------------------------
			//Loop defined at line 2715 of sp.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
				_ocl_gws[1] = (grid_points[1] - 2) - (1) + 1;

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_x_solve_7, 0,
						      __ocl_buffer_rhs);
				oclSetKernelArg(__ocl_x_solve_7, 1, sizeof(int),
						&m);
				oclSetKernelArg(__ocl_x_solve_7, 2, sizeof(int),
						&i);
				oclSetKernelArgBuffer(__ocl_x_solve_7, 3,
						      __ocl_buffer_lhs);
				oclSetKernelArg(__ocl_x_solve_7, 4, sizeof(int),
						&n);
				oclSetKernelArg(__ocl_x_solve_7, 5, sizeof(int),
						&i1);
				oclSetKernelArg(__ocl_x_solve_7, 6, sizeof(int),
						&i2);
				int __ocl_k_bound = grid_points[2] - 2;
				oclSetKernelArg(__ocl_x_solve_7, 7, sizeof(int),
						&__ocl_k_bound);
				int __ocl_j_bound = grid_points[1] - 2;
				oclSetKernelArg(__ocl_x_solve_7, 8, sizeof(int),
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
				oclDevReads(__ocl_buffer_lhs);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_x_solve_7, 2, _ocl_gws);
			}

		}
	}
	ninvr();
}

static void y_solve()
{
	int i, j, k, n, j1, j2, m;
	double fac1, fac2;
	lhsy();
	n = 0;
	for (j = 0; j <= grid_points[1] - 3; j++) {
		j1 = j + 1;
		j2 = j + 2;
		//--------------------------------------------------------------
		//Loop defined at line 2762 of sp.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
			_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_y_solve_0, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_y_solve_0, 1, sizeof(int), &n);
			oclSetKernelArg(__ocl_y_solve_0, 2, sizeof(int), &j);
			oclSetKernelArgBuffer(__ocl_y_solve_0, 3,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_y_solve_0, 4, sizeof(int), &j1);
			oclSetKernelArg(__ocl_y_solve_0, 5, sizeof(int), &j2);
			int __ocl_k_bound = grid_points[2] - 2;
			oclSetKernelArg(__ocl_y_solve_0, 6, sizeof(int),
					&__ocl_k_bound);
			int __ocl_i_bound = grid_points[0] - 2;
			oclSetKernelArg(__ocl_y_solve_0, 7, sizeof(int),
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

			oclRunKernel(__ocl_y_solve_0, 2, _ocl_gws);
		}

	}
	j = grid_points[1] - 2;
	j1 = grid_points[1] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 2802 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_y_solve_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_y_solve_1, 1, sizeof(int), &n);
		oclSetKernelArg(__ocl_y_solve_1, 2, sizeof(int), &j);
		oclSetKernelArgBuffer(__ocl_y_solve_1, 3, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_y_solve_1, 4, sizeof(int), &j1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_y_solve_1, 5, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_y_solve_1, 6, sizeof(int),
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

		oclRunKernel(__ocl_y_solve_1, 2, _ocl_gws);
	}

	for (m = 3; m < 5; m++) {
		n = (m - 3 + 1) * 5;
		for (j = 0; j <= grid_points[1] - 3; j++) {
			j1 = j + 1;
			j2 = j + 2;
			//--------------------------------------------------------------
			//Loop defined at line 2841 of sp.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
				_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_y_solve_2, 0,
						      __ocl_buffer_lhs);
				oclSetKernelArg(__ocl_y_solve_2, 1, sizeof(int),
						&n);
				oclSetKernelArg(__ocl_y_solve_2, 2, sizeof(int),
						&j);
				oclSetKernelArgBuffer(__ocl_y_solve_2, 3,
						      __ocl_buffer_rhs);
				oclSetKernelArg(__ocl_y_solve_2, 4, sizeof(int),
						&m);
				oclSetKernelArg(__ocl_y_solve_2, 5, sizeof(int),
						&j1);
				oclSetKernelArg(__ocl_y_solve_2, 6, sizeof(int),
						&j2);
				int __ocl_k_bound = grid_points[2] - 2;
				oclSetKernelArg(__ocl_y_solve_2, 7, sizeof(int),
						&__ocl_k_bound);
				int __ocl_i_bound = grid_points[0] - 2;
				oclSetKernelArg(__ocl_y_solve_2, 8, sizeof(int),
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

				oclRunKernel(__ocl_y_solve_2, 2, _ocl_gws);
			}

		}
		j = grid_points[1] - 2;
		j1 = grid_points[1] - 1;
		//--------------------------------------------------------------
		//Loop defined at line 2872 of sp.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
			_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_y_solve_3, 0,
					      __ocl_buffer_lhs);
			oclSetKernelArg(__ocl_y_solve_3, 1, sizeof(int), &n);
			oclSetKernelArg(__ocl_y_solve_3, 2, sizeof(int), &j);
			oclSetKernelArgBuffer(__ocl_y_solve_3, 3,
					      __ocl_buffer_rhs);
			oclSetKernelArg(__ocl_y_solve_3, 4, sizeof(int), &m);
			oclSetKernelArg(__ocl_y_solve_3, 5, sizeof(int), &j1);
			int __ocl_k_bound = grid_points[2] - 2;
			oclSetKernelArg(__ocl_y_solve_3, 6, sizeof(int),
					&__ocl_k_bound);
			int __ocl_i_bound = grid_points[0] - 2;
			oclSetKernelArg(__ocl_y_solve_3, 7, sizeof(int),
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

			oclRunKernel(__ocl_y_solve_3, 2, _ocl_gws);
		}

	}
	j = grid_points[1] - 2;
	j1 = grid_points[1] - 1;
	n = 0;
	//--------------------------------------------------------------
	//Loop defined at line 2903 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (3) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_y_solve_4, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_y_solve_4, 1, sizeof(int), &j);
		oclSetKernelArgBuffer(__ocl_y_solve_4, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_y_solve_4, 3, sizeof(int), &n);
		oclSetKernelArg(__ocl_y_solve_4, 4, sizeof(int), &j1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_y_solve_4, 5, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_y_solve_4, 6, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_y_solve_4, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 2915 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (3);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_y_solve_5, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_y_solve_5, 1, sizeof(int), &j);
		oclSetKernelArgBuffer(__ocl_y_solve_5, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_y_solve_5, 3, sizeof(int), &j1);
		int __ocl_k_bound = grid_points[2] - 2;
		oclSetKernelArg(__ocl_y_solve_5, 4, sizeof(int),
				&__ocl_k_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_y_solve_5, 5, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_y_solve_5, 3, _ocl_gws);
	}

	n = 0;
	for (m = 0; m < 3; m++) {
		for (j = grid_points[1] - 3; j >= 0; j--) {
			j1 = j + 1;
			j2 = j + 2;
			//--------------------------------------------------------------
			//Loop defined at line 2936 of sp.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
				_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_y_solve_6, 0,
						      __ocl_buffer_rhs);
				oclSetKernelArg(__ocl_y_solve_6, 1, sizeof(int),
						&m);
				oclSetKernelArg(__ocl_y_solve_6, 2, sizeof(int),
						&j);
				oclSetKernelArgBuffer(__ocl_y_solve_6, 3,
						      __ocl_buffer_lhs);
				oclSetKernelArg(__ocl_y_solve_6, 4, sizeof(int),
						&n);
				oclSetKernelArg(__ocl_y_solve_6, 5, sizeof(int),
						&j1);
				oclSetKernelArg(__ocl_y_solve_6, 6, sizeof(int),
						&j2);
				int __ocl_k_bound = grid_points[2] - 2;
				oclSetKernelArg(__ocl_y_solve_6, 7, sizeof(int),
						&__ocl_k_bound);
				int __ocl_i_bound = grid_points[0] - 2;
				oclSetKernelArg(__ocl_y_solve_6, 8, sizeof(int),
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
				oclDevReads(__ocl_buffer_lhs);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_y_solve_6, 2, _ocl_gws);
			}

		}
	}
	for (m = 3; m < 5; m++) {
		for (j = grid_points[1] - 3; j >= 0; j--) {
			n = (m - 3 + 1) * 5;
			j1 = j + 1;
			j2 = j1 + 1;
			//--------------------------------------------------------------
			//Loop defined at line 2958 of sp.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (grid_points[2] - 2) - (1) + 1;
				_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_y_solve_7, 0,
						      __ocl_buffer_rhs);
				oclSetKernelArg(__ocl_y_solve_7, 1, sizeof(int),
						&m);
				oclSetKernelArg(__ocl_y_solve_7, 2, sizeof(int),
						&j);
				oclSetKernelArgBuffer(__ocl_y_solve_7, 3,
						      __ocl_buffer_lhs);
				oclSetKernelArg(__ocl_y_solve_7, 4, sizeof(int),
						&n);
				oclSetKernelArg(__ocl_y_solve_7, 5, sizeof(int),
						&j1);
				oclSetKernelArg(__ocl_y_solve_7, 6, sizeof(int),
						&j2);
				int __ocl_k_bound = grid_points[2] - 2;
				oclSetKernelArg(__ocl_y_solve_7, 7, sizeof(int),
						&__ocl_k_bound);
				int __ocl_i_bound = grid_points[0] - 2;
				oclSetKernelArg(__ocl_y_solve_7, 8, sizeof(int),
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
				oclDevReads(__ocl_buffer_lhs);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_y_solve_7, 2, _ocl_gws);
			}

		}
	}
	pinvr();
}

static void z_solve()
{
	int i, j, k, n, k1, k2, m;
	double fac1, fac2;
	lhsz();
	n = 0;
#ifdef PROFILE_SWAP
    oclSync();
    ocl_timer ti;
    startTimer (&ti);
#endif
	oclSwapDimensions (__ocl_buffer_lhs, 15 * (102 / 2 * 2 + 1), 102 / 2 * 2 + 1, 1,
			  102 / 2 * 2 + 1, sizeof(double));
	oclSwapDimensions (__ocl_buffer_rhs, 5 * (102 / 2 * 2 + 1), 102 / 2 * 2 + 1, 1,
			  102 / 2 * 2 + 1, sizeof(double));
#ifdef PROFILE_SWAP
    oclSync();
    stopTimer (&ti);
    t_swap += elapsedTime (&ti);
    startTimer (&ti);
#endif
	//--------------------------------------------------------------
	//Loop defined at line 3006 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_0, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_z_solve_0, 1, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_0, 2, sizeof(int), &n);
		oclSetKernelArgBuffer(__ocl_z_solve_0, 3, __ocl_buffer_rhs);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_0, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_0, 5, sizeof(int),
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
		oclDevReads(__ocl_buffer_grid_points);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_0, 2, _ocl_gws);
	}

	k = grid_points[2] - 2;
	k1 = grid_points[2] - 1;
	//--------------------------------------------------------------
	//Loop defined at line 3073 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_1, 0, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_1, 1, sizeof(int), &n);
		oclSetKernelArg(__ocl_z_solve_1, 2, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_z_solve_1, 3, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_z_solve_1, 4, sizeof(int), &k1);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_1, 5, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_1, 6, sizeof(int),
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

		oclRunKernel(__ocl_z_solve_1, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 3131 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (3);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_2, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_z_solve_2, 1, __ocl_buffer_lhs);
		oclSetKernelArgBuffer(__ocl_z_solve_2, 2, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_z_solve_2, 3, sizeof(double), &fac2);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_2, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_2, 5, sizeof(int),
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
		oclDevReads(__ocl_buffer_grid_points);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_2, 3, _ocl_gws);
	}

	k = grid_points[2] - 2;
	k1 = grid_points[2] - 1;
	n = 0;
	//--------------------------------------------------------------
	//Loop defined at line 3236 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (3) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_3, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_z_solve_3, 1, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_z_solve_3, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_3, 3, sizeof(int), &n);
		oclSetKernelArg(__ocl_z_solve_3, 4, sizeof(int), &k1);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_3, 5, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_3, 6, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_3, 3, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 3253 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;
		_ocl_gws[2] = (5) - (3);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_4, 0, __ocl_buffer_rhs);
		oclSetKernelArg(__ocl_z_solve_4, 1, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_z_solve_4, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_4, 3, sizeof(int), &k1);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_4, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_4, 5, sizeof(int),
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
		oclDevReads(__ocl_buffer_lhs);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_z_solve_4, 3, _ocl_gws);
	}

	n = 0;
	//--------------------------------------------------------------
	//Loop defined at line 3280 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_5, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_z_solve_5, 1, __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_z_solve_5, 2, __ocl_buffer_lhs);
		oclSetKernelArg(__ocl_z_solve_5, 3, sizeof(int), &n);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_5, 4, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_5, 5, sizeof(int),
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

		oclRunKernel(__ocl_z_solve_5, 2, _ocl_gws);
	}

	//--------------------------------------------------------------
	//Loop defined at line 3307 of sp.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (grid_points[1] - 2) - (1) + 1;
		_ocl_gws[1] = (grid_points[0] - 2) - (1) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_z_solve_6, 0,
				      __ocl_buffer_grid_points);
		oclSetKernelArgBuffer(__ocl_z_solve_6, 1, __ocl_buffer_rhs);
		oclSetKernelArgBuffer(__ocl_z_solve_6, 2, __ocl_buffer_lhs);
		int __ocl_j_bound = grid_points[1] - 2;
		oclSetKernelArg(__ocl_z_solve_6, 3, sizeof(int),
				&__ocl_j_bound);
		int __ocl_i_bound = grid_points[0] - 2;
		oclSetKernelArg(__ocl_z_solve_6, 4, sizeof(int),
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

		oclRunKernel(__ocl_z_solve_6, 2, _ocl_gws);
	}

#ifdef PROFILE_SWAP
    oclSync();
    stopTimer (&ti);
    t_z += elapsedTime (&ti);
    startTimer (&ti);
#endif
	oclSwapDimensions (__ocl_buffer_lhs, 15 * (102 / 2 * 2 + 1), 102 / 2 * 2 + 1, 1,
			  102 / 2 * 2 + 1, sizeof(double));
	oclSwapDimensions (__ocl_buffer_rhs, 5 * (102 / 2 * 2 + 1), 102 / 2 * 2 + 1, 1,
			  102 / 2 * 2 + 1, sizeof(double));
#ifdef PROFILE_SWAP
    oclSync();
    stopTimer (&ti);
    t_swap += elapsedTime (&ti);
#endif

	tzetar();
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

	__ocl_program = oclBuildProgram("sp.B.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_add_0 = oclCreateKernel(__ocl_program, "add_0");
	DYN_PROGRAM_CHECK(__ocl_add_0);
	__ocl_lhsinit_0 = oclCreateKernel(__ocl_program, "lhsinit_0");
	DYN_PROGRAM_CHECK(__ocl_lhsinit_0);
	__ocl_lhsinit_1 = oclCreateKernel(__ocl_program, "lhsinit_1");
	DYN_PROGRAM_CHECK(__ocl_lhsinit_1);
	__ocl_lhsx_0 = oclCreateKernel(__ocl_program, "lhsx_0");
	DYN_PROGRAM_CHECK(__ocl_lhsx_0);
	__ocl_lhsx_1 = oclCreateKernel(__ocl_program, "lhsx_1");
	DYN_PROGRAM_CHECK(__ocl_lhsx_1);
	__ocl_lhsx_2 = oclCreateKernel(__ocl_program, "lhsx_2");
	DYN_PROGRAM_CHECK(__ocl_lhsx_2);
	__ocl_lhsx_3 = oclCreateKernel(__ocl_program, "lhsx_3");
	DYN_PROGRAM_CHECK(__ocl_lhsx_3);
	__ocl_lhsx_4 = oclCreateKernel(__ocl_program, "lhsx_4");
	DYN_PROGRAM_CHECK(__ocl_lhsx_4);
	__ocl_lhsx_5 = oclCreateKernel(__ocl_program, "lhsx_5");
	DYN_PROGRAM_CHECK(__ocl_lhsx_5);
	__ocl_lhsy_0 = oclCreateKernel(__ocl_program, "lhsy_0");
	DYN_PROGRAM_CHECK(__ocl_lhsy_0);
	__ocl_lhsy_1 = oclCreateKernel(__ocl_program, "lhsy_1");
	DYN_PROGRAM_CHECK(__ocl_lhsy_1);
	__ocl_lhsy_2 = oclCreateKernel(__ocl_program, "lhsy_2");
	DYN_PROGRAM_CHECK(__ocl_lhsy_2);
	__ocl_lhsy_3 = oclCreateKernel(__ocl_program, "lhsy_3");
	DYN_PROGRAM_CHECK(__ocl_lhsy_3);
	__ocl_lhsy_4 = oclCreateKernel(__ocl_program, "lhsy_4");
	DYN_PROGRAM_CHECK(__ocl_lhsy_4);
	__ocl_lhsy_5 = oclCreateKernel(__ocl_program, "lhsy_5");
	DYN_PROGRAM_CHECK(__ocl_lhsy_5);
	__ocl_lhsz_0 = oclCreateKernel(__ocl_program, "lhsz_0");
	DYN_PROGRAM_CHECK(__ocl_lhsz_0);
	__ocl_lhsz_1 = oclCreateKernel(__ocl_program, "lhsz_1");
	DYN_PROGRAM_CHECK(__ocl_lhsz_1);
	__ocl_lhsz_2 = oclCreateKernel(__ocl_program, "lhsz_2");
	DYN_PROGRAM_CHECK(__ocl_lhsz_2);
	__ocl_lhsz_3 = oclCreateKernel(__ocl_program, "lhsz_3");
	DYN_PROGRAM_CHECK(__ocl_lhsz_3);
	__ocl_lhsz_4 = oclCreateKernel(__ocl_program, "lhsz_4");
	DYN_PROGRAM_CHECK(__ocl_lhsz_4);
	__ocl_lhsz_5 = oclCreateKernel(__ocl_program, "lhsz_5");
	DYN_PROGRAM_CHECK(__ocl_lhsz_5);
	__ocl_ninvr_0 = oclCreateKernel(__ocl_program, "ninvr_0");
	DYN_PROGRAM_CHECK(__ocl_ninvr_0);
	__ocl_pinvr_0 = oclCreateKernel(__ocl_program, "pinvr_0");
	DYN_PROGRAM_CHECK(__ocl_pinvr_0);
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
	__ocl_txinvr_0 = oclCreateKernel(__ocl_program, "txinvr_0");
	DYN_PROGRAM_CHECK(__ocl_txinvr_0);
	__ocl_tzetar_0 = oclCreateKernel(__ocl_program, "tzetar_0");
	DYN_PROGRAM_CHECK(__ocl_tzetar_0);
	__ocl_x_solve_0 = oclCreateKernel(__ocl_program, "x_solve_0");
	DYN_PROGRAM_CHECK(__ocl_x_solve_0);
	__ocl_x_solve_1 = oclCreateKernel(__ocl_program, "x_solve_1");
	DYN_PROGRAM_CHECK(__ocl_x_solve_1);
	__ocl_x_solve_2 = oclCreateKernel(__ocl_program, "x_solve_2");
	DYN_PROGRAM_CHECK(__ocl_x_solve_2);
	__ocl_x_solve_3 = oclCreateKernel(__ocl_program, "x_solve_3");
	DYN_PROGRAM_CHECK(__ocl_x_solve_3);
	__ocl_x_solve_4 = oclCreateKernel(__ocl_program, "x_solve_4");
	DYN_PROGRAM_CHECK(__ocl_x_solve_4);
	__ocl_x_solve_5 = oclCreateKernel(__ocl_program, "x_solve_5");
	DYN_PROGRAM_CHECK(__ocl_x_solve_5);
	__ocl_x_solve_6 = oclCreateKernel(__ocl_program, "x_solve_6");
	DYN_PROGRAM_CHECK(__ocl_x_solve_6);
	__ocl_x_solve_7 = oclCreateKernel(__ocl_program, "x_solve_7");
	DYN_PROGRAM_CHECK(__ocl_x_solve_7);
	__ocl_y_solve_0 = oclCreateKernel(__ocl_program, "y_solve_0");
	DYN_PROGRAM_CHECK(__ocl_y_solve_0);
	__ocl_y_solve_1 = oclCreateKernel(__ocl_program, "y_solve_1");
	DYN_PROGRAM_CHECK(__ocl_y_solve_1);
	__ocl_y_solve_2 = oclCreateKernel(__ocl_program, "y_solve_2");
	DYN_PROGRAM_CHECK(__ocl_y_solve_2);
	__ocl_y_solve_3 = oclCreateKernel(__ocl_program, "y_solve_3");
	DYN_PROGRAM_CHECK(__ocl_y_solve_3);
	__ocl_y_solve_4 = oclCreateKernel(__ocl_program, "y_solve_4");
	DYN_PROGRAM_CHECK(__ocl_y_solve_4);
	__ocl_y_solve_5 = oclCreateKernel(__ocl_program, "y_solve_5");
	DYN_PROGRAM_CHECK(__ocl_y_solve_5);
	__ocl_y_solve_6 = oclCreateKernel(__ocl_program, "y_solve_6");
	DYN_PROGRAM_CHECK(__ocl_y_solve_6);
	__ocl_y_solve_7 = oclCreateKernel(__ocl_program, "y_solve_7");
	DYN_PROGRAM_CHECK(__ocl_y_solve_7);
	__ocl_z_solve_0 = oclCreateKernel(__ocl_program, "z_solve_0");
	DYN_PROGRAM_CHECK(__ocl_z_solve_0);
	__ocl_z_solve_1 = oclCreateKernel(__ocl_program, "z_solve_1");
	DYN_PROGRAM_CHECK(__ocl_z_solve_1);
	__ocl_z_solve_2 = oclCreateKernel(__ocl_program, "z_solve_2");
	DYN_PROGRAM_CHECK(__ocl_z_solve_2);
	__ocl_z_solve_3 = oclCreateKernel(__ocl_program, "z_solve_3");
	DYN_PROGRAM_CHECK(__ocl_z_solve_3);
	__ocl_z_solve_4 = oclCreateKernel(__ocl_program, "z_solve_4");
	DYN_PROGRAM_CHECK(__ocl_z_solve_4);
	__ocl_z_solve_5 = oclCreateKernel(__ocl_program, "z_solve_5");
	DYN_PROGRAM_CHECK(__ocl_z_solve_5);
	__ocl_z_solve_6 = oclCreateKernel(__ocl_program, "z_solve_6");
	DYN_PROGRAM_CHECK(__ocl_z_solve_6);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_u =
	    oclCreateBuffer(u, (5 * 103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_u, -1);
	__ocl_buffer_rhs =
	    oclCreateBuffer(rhs, (5 * 103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rhs, -1);
	__ocl_buffer_lhs =
	    oclCreateBuffer(lhs, (15 * 103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_lhs, -1);
	__ocl_buffer_rho_i =
	    oclCreateBuffer(rho_i, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rho_i, -1);
	__ocl_buffer_cv =
	    oclCreateBuffer(cv, (102 * 102 * 102) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_cv, -1);
	__ocl_buffer_us =
	    oclCreateBuffer(us, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_us, -1);
	__ocl_buffer_rhon =
	    oclCreateBuffer(rhon, (102 * 102 * 102) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rhon, -1);
	__ocl_buffer_speed =
	    oclCreateBuffer(speed, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_speed, -1);
	__ocl_buffer_vs =
	    oclCreateBuffer(vs, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_vs, -1);
	__ocl_buffer_rhoq =
	    oclCreateBuffer(rhoq, (102 * 102 * 102) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rhoq, -1);
	__ocl_buffer_ws =
	    oclCreateBuffer(ws, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ws, -1);
	__ocl_buffer_rhos =
	    oclCreateBuffer(rhos, (102 * 102 * 102) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rhos, -1);
	__ocl_buffer_square =
	    oclCreateBuffer(square, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_square, -1);
	__ocl_buffer_qs =
	    oclCreateBuffer(qs, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_qs, -1);
	__ocl_buffer_ainv =
	    oclCreateBuffer(ainv, (103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ainv, -1);
	__ocl_buffer_forcing =
	    oclCreateBuffer(forcing, (5 * 103 * 103 * 103) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_forcing, -1);
	__ocl_buffer_grid_points =
	    oclCreateBuffer(grid_points, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_grid_points, -1);
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_rhs);
	oclHostWrites(__ocl_buffer_lhs);
	oclHostWrites(__ocl_buffer_rho_i);
	oclHostWrites(__ocl_buffer_cv);
	oclHostWrites(__ocl_buffer_us);
	oclHostWrites(__ocl_buffer_rhon);
	oclHostWrites(__ocl_buffer_speed);
	oclHostWrites(__ocl_buffer_vs);
	oclHostWrites(__ocl_buffer_rhoq);
	oclHostWrites(__ocl_buffer_ws);
	oclHostWrites(__ocl_buffer_rhos);
	oclHostWrites(__ocl_buffer_square);
	oclHostWrites(__ocl_buffer_qs);
	oclHostWrites(__ocl_buffer_ainv);
	oclHostWrites(__ocl_buffer_forcing);
	oclHostWrites(__ocl_buffer_grid_points);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_u);
	oclReleaseBuffer(__ocl_buffer_rhs);
	oclReleaseBuffer(__ocl_buffer_lhs);
	oclReleaseBuffer(__ocl_buffer_rho_i);
	oclReleaseBuffer(__ocl_buffer_cv);
	oclReleaseBuffer(__ocl_buffer_us);
	oclReleaseBuffer(__ocl_buffer_rhon);
	oclReleaseBuffer(__ocl_buffer_speed);
	oclReleaseBuffer(__ocl_buffer_vs);
	oclReleaseBuffer(__ocl_buffer_rhoq);
	oclReleaseBuffer(__ocl_buffer_ws);
	oclReleaseBuffer(__ocl_buffer_rhos);
	oclReleaseBuffer(__ocl_buffer_square);
	oclReleaseBuffer(__ocl_buffer_qs);
	oclReleaseBuffer(__ocl_buffer_ainv);
	oclReleaseBuffer(__ocl_buffer_forcing);
	oclReleaseBuffer(__ocl_buffer_grid_points);
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_rhs);
	oclHostWrites(__ocl_buffer_lhs);
	oclHostWrites(__ocl_buffer_rho_i);
	oclHostWrites(__ocl_buffer_cv);
	oclHostWrites(__ocl_buffer_us);
	oclHostWrites(__ocl_buffer_rhon);
	oclHostWrites(__ocl_buffer_speed);
	oclHostWrites(__ocl_buffer_vs);
	oclHostWrites(__ocl_buffer_rhoq);
	oclHostWrites(__ocl_buffer_ws);
	oclHostWrites(__ocl_buffer_rhos);
	oclHostWrites(__ocl_buffer_square);
	oclHostWrites(__ocl_buffer_qs);
	oclHostWrites(__ocl_buffer_ainv);
	oclHostWrites(__ocl_buffer_forcing);
	oclHostWrites(__ocl_buffer_grid_points);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
