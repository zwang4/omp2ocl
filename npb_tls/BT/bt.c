//#include "ticktock.h"
/*--------------------------------------------------------------------

  NAS Parallel Benchmarks 2.3 OpenMP C versions - BT

  This benchmark is an OpenMP C version of the NPB BT code.

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

Authors: R. Van der Wijngaart
T. Harris
M. Yarrow

OpenMP C version: S. Satoh

--------------------------------------------------------------------*/

#include "npb-C.h"

/* global variables */
#include "header.h"

/* function declarations */
static void add(void);
static void adi(void);
static void error_norm(double rms[5]);
static void rhs_norm(double rms[5]);
static void exact_rhs(void);
static void exact_solution(double xi, double eta, double zeta,
		double dtemp[5]);
static void initialize(void);
static void lhsinit(void);
static void lhsx(void);
static void lhsy(void);
static void lhsz(void);
static void compute_rhs(void);
static void set_constants(void);
static void verify(int no_time_steps, char *class, boolean *verified);
static void x_solve(void);
static void x_backsubstitute(void);
static void x_solve_cell(void);
static void matvec_sub(double ablock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int ablock_0, int ablock_1, int ablock_2, int ablock_3, double avec[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int avec_0, int avec_1, int avec_2, double bvec[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int bvec_0, int bvec_1, int bvec_2);
static void matmul_sub(double ablock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int ablock_0, int ablock_1, int ablock_2, int ablock_3,
		double bblock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int bblock_0, int bblock_1, int bblock_2, int bblock_3,
		double cblock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int cblock_0, int cblock_1, int cblock_2, int cblock_3);
static void binvcrhs(double lhs[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int lhs_0, int lhs_1, int lhs_2, int lhs_3, double c[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int c_0, int c_1, int c_2, int c_3, double r[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int r_0, int r_1, int r_2);
static void binvrhs(double lhs[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int lhs_0, int lhs_1, int lhs_2, int lhs_3, double r[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int r_0, int r_1, int r_2);
static void y_solve(void);
static void y_backsubstitute(void);
static void y_solve_cell(void);
static void z_solve(void);
static void z_backsubstitute(void);
static void z_solve_cell(void);

/*--------------------------------------------------------------------
  program BT
  c-------------------------------------------------------------------*/
int main(int argc, char **argv) {

	int niter, step, n3;
	int nthreads = 1;
	double navg, mflops;

	double tmax;
	boolean verified;
	char class;
	FILE *fp;

	/*--------------------------------------------------------------------
	  c      Root node reads input file (if it exists) else takes
	  c      defaults from parameters
	  c-------------------------------------------------------------------*/

	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
			" - BT Benchmark\n\n");

	fp = fopen("inputbt.data", "r");
	if (fp != NULL) {
		printf(" Reading from input file inputbt.data");
		fscanf(fp, "%d", &niter);
		while (fgetc(fp) != '\n');
		fscanf(fp, "%lg", &dt);
		while (fgetc(fp) != '\n');
		fscanf(fp, "%d%d%d",
				&grid_points[0],  &grid_points[1],  &grid_points[2]);
		fclose(fp);
	} else {
		printf(" No input file inputbt.data. Using compiled defaults\n");

		niter = NITER_DEFAULT;
		dt    = DT_DEFAULT;
		grid_points[0] = PROBLEM_SIZE;
		grid_points[1] = PROBLEM_SIZE;
		grid_points[2] = PROBLEM_SIZE;
	}

	printf(" Size: %3dx%3dx%3d\n",
			grid_points[0], grid_points[1], grid_points[2]);
	printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);

	if (grid_points[0] > IMAX ||
			grid_points[1] > JMAX ||
			grid_points[2] > KMAX) {
		printf(" %dx%dx%d\n", grid_points[0], grid_points[1], grid_points[2]);
		printf(" Problem size too big for compiled array sizes\n");
		exit(1);
	}

	set_constants();

#pragma omp2ocl init
	{
		initialize();

		lhsinit();

		exact_rhs();

		/*--------------------------------------------------------------------
		  c      do one time step to touch all code, and reinitialize
		  c-------------------------------------------------------------------*/
		adi();

		initialize();
	} /* end parallel */

	timer_clear(1);
	timer_start(1);

	for (step = 1; step <= niter; step++) {

		if (step%20 == 0 || step == 1) {
			//////#pragma omp master	
			printf(" Time step %4d\n", step);
		}

		adi();
	}

#pragma omp2ocl sync

	timer_stop(1);
	tmax = timer_read(1);

	verify(niter, &class, &verified);

	n3 = grid_points[0]*grid_points[1]*grid_points[2];
	navg = (grid_points[0]+grid_points[1]+grid_points[2])/3.0;
	if ( tmax != 0.0 ) {
		mflops = 1.0e-6*(double)niter*
			(3478.8*(double)n3-17655.7*pow2(navg)+28023.7*navg) / tmax;
	} else {
		mflops = 0.0;
	}
	c_print_results("BT", class, grid_points[0], 
			grid_points[1], grid_points[2], niter, nthreads,
			tmax, mflops, "          floating point", 
			verified, NPBVERSION,COMPILETIME, CS1, CS2, CS3, CS4, CS5, 
			CS6, "(none)");
#pragma omp2ocl term
}

/*--------------------------------------------------------------------
  c-------------------------------------------------------------------*/

static void add(void) {

	/*--------------------------------------------------------------------
	  c     addition of update to the vector u
	  c-------------------------------------------------------------------*/

	int i, j, k, m;

#pragma omp parallel for schedule(static,1)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < 5; m++) {
					u[m][i][j][k] = u[m][i][j][k] + rhs[m][i][j][k];
				}
			}
		}
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void adi(void) {
	compute_rhs();

	x_solve();

	y_solve();

	z_solve();

	add();
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void error_norm(double rms[5]) {

	/*--------------------------------------------------------------------
	  c     this function computes the norm of the difference between the
	  c     computed solution and the exact solution
	  c-------------------------------------------------------------------*/

	int i, j, k, m, d;
	double xi, eta, zeta, u_exact[5], add;

	for (m = 0; m < 5; m++) {
		rms[m] = 0.0;
	}

	for (i = 0; i < grid_points[0]; i++) {
		xi = (double)i * dnxm1;
		for (j = 0; j < grid_points[1]; j++) {
			eta = (double)j * dnym1;
			for (k = 0; k < grid_points[2]; k++) {
				zeta = (double)k * dnzm1;
				exact_solution(xi, eta, zeta, u_exact);

				for (m = 0; m < 5; m++) {
					add = u[m][i][j][k] - u_exact[m];
					rms[m] = rms[m] + add*add;
				}
			}
		}
	}

	for (m = 0; m < 5; m++) {
		for (d = 0; d <= 2; d++) {
			rms[m] = rms[m] / (double)(grid_points[d]-2);
		}
		rms[m] = sqrt(rms[m]);
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void rhs_norm(double rms[5]) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	int i, j, k, d, m;
	double add;

	for (m = 0; m < 5; m++) {
		rms[m] = 0.0;
	}

	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < 5; m++) {
					add = rhs[m][i][j][k];
					rms[m] = rms[m] + add*add;
				}
			}
		}
	}

	for (m = 0; m < 5; m++) {
		for (d = 0; d <= 2; d++) {
			rms[m] = rms[m] / (double)(grid_points[d]-2);
		}
		rms[m] = sqrt(rms[m]);
	}
}


/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void exact_rhs(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     compute the right hand side based on exact solution
	  c-------------------------------------------------------------------*/

	double dtemp[5], xi, eta, zeta, dtpp;
	int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;

	/*--------------------------------------------------------------------
	  c     initialize                                  
	  c-------------------------------------------------------------------*/
	//trace_start("exact_rhs", 1);  
#pragma omp parallel for schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				for (m = 0; m < 5; m++) {
					forcing[m][i][j][k] = 0.0;
				}
			}
		}
	}
	//trace_stop("exact_rhs", 1);  

	/*--------------------------------------------------------------------
	  c     xi-direction flux differences                      
	  c-------------------------------------------------------------------*/
	//trace_start("exact_rhs", 2);  
#pragma omp parallel for private(dtemp, xi, eta, zeta, dtpp, im1, jp1, jm1, km1, kp1) schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			eta = (double)j * dnym1;
			zeta = (double)k * dnzm1;

			for (i = 0; i < grid_points[0]; i++) {
				xi = (double)i * dnxm1;

				exact_solution(xi, eta, zeta, dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][i] = dtemp[m];
				}

				dtpp = 1.0 / dtemp[0];

				for (m = 1; m <= 4; m++) {
					buf[m][i] = dtpp * dtemp[m];
				}

				cuf[i]   = buf[1][i] * buf[1][i];
				buf[0][i] = cuf[i] + buf[2][i] * buf[2][i] + 
					buf[3][i] * buf[3][i];
				q[i] = 0.5*(buf[1][i]*ue[1][i] + buf[2][i]*ue[2][i] +
						buf[3][i]*ue[3][i]);
			}

			for (i = 1; i < grid_points[0]-1; i++) {
				im1 = i-1;
				ip1 = i+1;

				forcing[0][i][j][k] = forcing[0][i][j][k] -
					tx2*(ue[1][ip1]-ue[1][im1])+
					dx1tx1*(ue[0][ip1]-2.0*ue[0][i]+ue[0][im1]);

				forcing[1][i][j][k] = forcing[1][i][j][k] -
					tx2 * ((ue[1][ip1]*buf[1][ip1]+c2*(ue[4][ip1]-q[ip1]))-
							(ue[1][im1]*buf[1][im1]+c2*(ue[4][im1]-q[im1])))+
					xxcon1*(buf[1][ip1]-2.0*buf[1][i]+buf[1][im1])+
					dx2tx1*( ue[1][ip1]-2.0* ue[1][i]+ ue[1][im1]);

				forcing[2][i][j][k] = forcing[2][i][j][k] -
					tx2 * (ue[2][ip1]*buf[1][ip1]-ue[2][im1]*buf[1][im1])+
					xxcon2*(buf[2][ip1]-2.0*buf[2][i]+buf[2][im1])+
					dx3tx1*( ue[2][ip1]-2.0* ue[2][i]+ ue[2][im1]);

				forcing[3][i][j][k] = forcing[3][i][j][k] -
					tx2*(ue[3][ip1]*buf[1][ip1]-ue[3][im1]*buf[1][im1])+
					xxcon2*(buf[3][ip1]-2.0*buf[3][i]+buf[3][im1])+
					dx4tx1*( ue[3][ip1]-2.0* ue[3][i]+ ue[3][im1]);

				forcing[4][i][j][k] = forcing[4][i][j][k] -
					tx2*(buf[1][ip1]*(c1*ue[4][ip1]-c2*q[ip1])-
							buf[1][im1]*(c1*ue[4][im1]-c2*q[im1]))+
					0.5*xxcon3*(buf[0][ip1]-2.0*buf[0][i]+buf[0][im1])+
					xxcon4*(cuf[ip1]-2.0*cuf[i]+cuf[im1])+
					xxcon5*(buf[4][ip1]-2.0*buf[4][i]+buf[4][im1])+
					dx5tx1*( ue[4][ip1]-2.0* ue[4][i]+ ue[4][im1]);
			}

			/*--------------------------------------------------------------------
			  c     Fourth-order dissipation                         
			  c-------------------------------------------------------------------*/

			for (m = 0; m < 5; m++) {
				i = 1;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(5.0*ue[m][i] - 4.0*ue[m][i+1] +ue[m][i+2]);
				i = 2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(-4.0*ue[m][i-1] + 6.0*ue[m][i] -
					 4.0*ue[m][i+1] +     ue[m][i+2]);
			}

			for (m = 0; m < 5; m++) {
				for (i = 1*3; i <= grid_points[0]-3*1-1; i++) {
					forcing[m][i][j][k] = forcing[m][i][j][k] - dssp*
						(ue[m][i-2] - 4.0*ue[m][i-1] +
						 6.0*ue[m][i] - 4.0*ue[m][i+1] + ue[m][i+2]);
				}
			}

			for (m = 0; m < 5; m++) {
				i = grid_points[0]-3;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][i-2] - 4.0*ue[m][i-1] +
					 6.0*ue[m][i] - 4.0*ue[m][i+1]);
				i = grid_points[0]-2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][i-2] - 4.0*ue[m][i-1] + 5.0*ue[m][i]);
			}

		}
	}
	//trace_stop("exact_rhs", 2);  

	/*--------------------------------------------------------------------
	  c     eta-direction flux differences             
	  ------------------------------------------------------------------*/
	//trace_start("exact_rhs", 3);  
#pragma omp parallel for private(dtemp, xi, eta, zeta, dtpp, ip1, im1, jp1, jm1, km1, kp1) schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {

		for (k = 1; k < grid_points[2]-1; k++) {
			xi = (double)i * dnxm1;
			zeta = (double)k * dnzm1;

			for (j = 0; j < grid_points[1]; j++) {
				eta = (double)j * dnym1;

				exact_solution(xi, eta, zeta, dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][j] = dtemp[m];
				}

				dtpp = 1.0/dtemp[0];

				for (m = 1; m <= 4; m++) {
					buf[m][j] = dtpp * dtemp[m];
				}

				cuf[j]   = buf[2][j] * buf[2][j];
				buf[0][j] = cuf[j] + buf[1][j] * buf[1][j] + 
					buf[3][j] * buf[3][j];
				q[j] = 0.5*(buf[1][j]*ue[1][j] + buf[2][j]*ue[2][j] +
						buf[3][j]*ue[3][j]);
			}

			for (j = 1; j < grid_points[1]-1; j++) {
				jm1 = j-1;
				jp1 = j+1;

				forcing[0][i][j][k] = forcing[0][i][j][k] -
					ty2*( ue[2][jp1]-ue[2][jm1] )+
					dy1ty1*(ue[0][jp1]-2.0*ue[0][j]+ue[0][jm1]);

				forcing[1][i][j][k] = forcing[1][i][j][k] -
					ty2*(ue[1][jp1]*buf[2][jp1]-ue[1][jm1]*buf[2][jm1])+
					yycon2*(buf[1][jp1]-2.0*buf[1][j]+buf[1][jm1])+
					dy2ty1*( ue[1][jp1]-2.0* ue[1][j]+ ue[1][jm1]);

				forcing[2][i][j][k] = forcing[2][i][j][k] -
					ty2*((ue[2][jp1]*buf[2][jp1]+c2*(ue[4][jp1]-q[jp1]))-
							(ue[2][jm1]*buf[2][jm1]+c2*(ue[4][jm1]-q[jm1])))+
					yycon1*(buf[2][jp1]-2.0*buf[2][j]+buf[2][jm1])+
					dy3ty1*( ue[2][jp1]-2.0*ue[2][j] +ue[2][jm1]);

				forcing[3][i][j][k] = forcing[3][i][j][k] -
					ty2*(ue[3][jp1]*buf[2][jp1]-ue[3][jm1]*buf[2][jm1])+
					yycon2*(buf[3][jp1]-2.0*buf[3][j]+buf[3][jm1])+
					dy4ty1*( ue[3][jp1]-2.0*ue[3][j]+ ue[3][jm1]);

				forcing[4][i][j][k] = forcing[4][i][j][k] -
					ty2*(buf[2][jp1]*(c1*ue[4][jp1]-c2*q[jp1])-
							buf[2][jm1]*(c1*ue[4][jm1]-c2*q[jm1]))+
					0.5*yycon3*(buf[0][jp1]-2.0*buf[0][j]+
							buf[0][jm1])+
					yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+
					yycon5*(buf[4][jp1]-2.0*buf[4][j]+buf[4][jm1])+
					dy5ty1*(ue[4][jp1]-2.0*ue[4][j]+ue[4][jm1]);
			}

			/*--------------------------------------------------------------------
			  c     Fourth-order dissipation                      
			  c-------------------------------------------------------------------*/
			for (m = 0; m < 5; m++) {
				j = 1;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(5.0*ue[m][j] - 4.0*ue[m][j+1] +ue[m][j+2]);
				j = 2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(-4.0*ue[m][j-1] + 6.0*ue[m][j] -
					 4.0*ue[m][j+1] +       ue[m][j+2]);
			}

			for (m = 0; m < 5; m++) {
				for (j = 1*3; j <= grid_points[1]-3*1-1; j++) {
					forcing[m][i][j][k] = forcing[m][i][j][k] - dssp*
						(ue[m][j-2] - 4.0*ue[m][j-1] +
						 6.0*ue[m][j] - 4.0*ue[m][j+1] + ue[m][j+2]);
				}
			}

			for (m = 0; m < 5; m++) {
				j = grid_points[1]-3;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][j-2] - 4.0*ue[m][j-1] +
					 6.0*ue[m][j] - 4.0*ue[m][j+1]);
				j = grid_points[1]-2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][j-2] - 4.0*ue[m][j-1] + 5.0*ue[m][j]);
			}

		}
	}
	//trace_stop("exact_rhs", 3);  


	/*--------------------------------------------------------------------
	  c     zeta-direction flux differences                      
	  c-------------------------------------------------------------------*/
	//trace_start("exact_rhs", 4);  
#pragma omp parallel for private(dtemp, xi, eta, zeta, dtpp, ip1, im1, jp1, jm1, km1, kp1) schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			xi = (double)i * dnxm1;
			eta = (double)j * dnym1;

			for (k = 0; k < grid_points[2]; k++) {
				zeta = (double)k * dnzm1;

				exact_solution(xi, eta, zeta, dtemp);
				for (m = 0; m < 5; m++) {
					ue[m][k] = dtemp[m];
				}

				dtpp = 1.0/dtemp[0];

				for (m = 1; m <= 4; m++) {
					buf[m][k] = dtpp * dtemp[m];
				}

				cuf[k]   = buf[3][k] * buf[3][k];
				buf[0][k] = cuf[k] + buf[1][k] * buf[1][k] + 
					buf[2][k] * buf[2][k];
				q[k] = 0.5*(buf[1][k]*ue[1][k] + buf[2][k]*ue[2][k] +
						buf[3][k]*ue[3][k]);
			}

			for (k = 1; k < grid_points[2]-1; k++) {
				km1 = k-1;
				kp1 = k+1;

				forcing[0][i][j][k] = forcing[0][i][j][k] -
					tz2*( ue[3][kp1]-ue[3][km1] )+
					dz1tz1*(ue[0][kp1]-2.0*ue[0][k]+ue[0][km1]);

				forcing[1][i][j][k] = forcing[1][i][j][k] -
					tz2 * (ue[1][kp1]*buf[3][kp1]-ue[1][km1]*buf[3][km1])+
					zzcon2*(buf[1][kp1]-2.0*buf[1][k]+buf[1][km1])+
					dz2tz1*( ue[1][kp1]-2.0* ue[1][k]+ ue[1][km1]);

				forcing[2][i][j][k] = forcing[2][i][j][k] -
					tz2 * (ue[2][kp1]*buf[3][kp1]-ue[2][km1]*buf[3][km1])+
					zzcon2*(buf[2][kp1]-2.0*buf[2][k]+buf[2][km1])+
					dz3tz1*(ue[2][kp1]-2.0*ue[2][k]+ue[2][km1]);

				forcing[3][i][j][k] = forcing[3][i][j][k] -
					tz2 * ((ue[3][kp1]*buf[3][kp1]+c2*(ue[4][kp1]-q[kp1]))-
							(ue[3][km1]*buf[3][km1]+c2*(ue[4][km1]-q[km1])))+
					zzcon1*(buf[3][kp1]-2.0*buf[3][k]+buf[3][km1])+
					dz4tz1*( ue[3][kp1]-2.0*ue[3][k] +ue[3][km1]);

				forcing[4][i][j][k] = forcing[4][i][j][k] -
					tz2 * (buf[3][kp1]*(c1*ue[4][kp1]-c2*q[kp1])-
							buf[3][km1]*(c1*ue[4][km1]-c2*q[km1]))+
					0.5*zzcon3*(buf[0][kp1]-2.0*buf[0][k]
							+buf[0][km1])+
					zzcon4*(cuf[kp1]-2.0*cuf[k]+cuf[km1])+
					zzcon5*(buf[4][kp1]-2.0*buf[4][k]+buf[4][km1])+
					dz5tz1*( ue[4][kp1]-2.0*ue[4][k]+ ue[4][km1]);
			}

			/*--------------------------------------------------------------------
			  c     Fourth-order dissipation                        
			  c-------------------------------------------------------------------*/
			for (m = 0; m < 5; m++) {
				k = 1;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(5.0*ue[m][k] - 4.0*ue[m][k+1] +ue[m][k+2]);
				k = 2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(-4.0*ue[m][k-1] + 6.0*ue[m][k] -
					 4.0*ue[m][k+1] +       ue[m][k+2]);
			}

			for (m = 0; m < 5; m++) {
				for (k = 1*3; k <= grid_points[2]-3*1-1; k++) {
					forcing[m][i][j][k] = forcing[m][i][j][k] - dssp*
						(ue[m][k-2] - 4.0*ue[m][k-1] +
						 6.0*ue[m][k] - 4.0*ue[m][k+1] + ue[m][k+2]);
				}
			}

			for (m = 0; m < 5; m++) {
				k = grid_points[2]-3;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][k-2] - 4.0*ue[m][k-1] +
					 6.0*ue[m][k] - 4.0*ue[m][k+1]);
				k = grid_points[2]-2;
				forcing[m][i][j][k] = forcing[m][i][j][k] - dssp *
					(ue[m][k-2] - 4.0*ue[m][k-1] + 5.0*ue[m][k]);
			}

		}
	}
	//trace_stop("exact_rhs", 4);  

	/*--------------------------------------------------------------------
	  c     now change the sign of the forcing function, 
	  c-------------------------------------------------------------------*/
	//trace_start("exact_rhs", 5);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < 5; m++) {
					forcing[m][i][j][k] = -1.0 * forcing[m][i][j][k];
				}
			}
		}
	}
	//trace_stop("exact_rhs", 5);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void exact_solution(double xi, double eta, double zeta,
		double dtemp[5]) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     this function returns the exact solution at point xi, eta, zeta  
	  c-------------------------------------------------------------------*/

	int m;

	for (m = 0; m < 5; m++) {
		dtemp[m] =  ce[m][0] +
			xi*(ce[m][1] + xi*(ce[m][4] + xi*(ce[m][7]
							+ xi*ce[m][10]))) +
			eta*(ce[m][2] + eta*(ce[m][5] + eta*(ce[m][8]
							+ eta*ce[m][11])))+
			zeta*(ce[m][3] + zeta*(ce[m][6] + zeta*(ce[m][9] + 
							zeta*ce[m][12])));
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void initialize(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     This subroutine initializes the field variable u using 
	  c     tri-linear transfinite interpolation of the boundary values     
	  c-------------------------------------------------------------------*/

	int i, j, k, m, ix, iy, iz;
	double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];

	/*--------------------------------------------------------------------
	  c  Later (in compute_rhs) we compute 1/u for every element. A few of 
	  c  the corner elements are not used, but it convenient (and faster) 
	  c  to compute the whole thing with a simple loop. Make sure those 
	  c  values are nonzero by initializing the whole thing here. 
	  c-------------------------------------------------------------------*/
	//trace_start("initialize", 1);  
#pragma omp parallel for schedule(static)
	for (i = 0; i < IMAX; i++) {
		for (j = 0; j < IMAX; j++) {
			for (k = 0; k < IMAX; k++) {
				for (m = 0; m < 5; m++) {
					u[m][i][j][k] = 1.0;
				}
			}
		}
	}
	//trace_stop("initialize", 1);  

	/*--------------------------------------------------------------------
	  c     first store the "interpolated" values everywhere on the grid    
	  c-------------------------------------------------------------------*/

	//trace_start("initialize", 2);  
#pragma omp parallel for private(ix, iy, iz, xi, eta, zeta, Pface, Pxi, Peta, Pzeta, temp) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				eta = (double)j * dnym1;
				xi = (double)i * dnxm1;
				zeta = (double)k * dnzm1;

				for (ix = 0; ix < 2; ix++) {
					exact_solution((double)ix, eta, zeta, 
							&(Pface[ix][0][0]));
				}

				for (iy = 0; iy < 2; iy++) {
					exact_solution(xi, (double)iy , zeta, 
							&Pface[iy][1][0]);
				}

				for (iz = 0; iz < 2; iz++) {
					exact_solution(xi, eta, (double)iz,   
							&Pface[iz][2][0]);
				}

				for (m = 0; m < 5; m++) {
					Pxi   = xi   * Pface[1][0][m] + 
						(1.0-xi)   * Pface[0][0][m];
					Peta  = eta  * Pface[1][1][m] + 
						(1.0-eta)  * Pface[0][1][m];
					Pzeta = zeta * Pface[1][2][m] + 
						(1.0-zeta) * Pface[0][2][m];

					u[m][i][j][k] = Pxi + Peta + Pzeta - 
						Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + 
						Pxi*Peta*Pzeta;
				}
			}
		}
	}
	//trace_stop("initialize", 2);  

	/*--------------------------------------------------------------------
	  c     now store the exact values on the boundaries        
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     west face                                                  
	  c-------------------------------------------------------------------*/
	i = 0;
	xi = 0.0;
	//trace_start("initialize", 3);  
#pragma omp parallel for private(eta, zeta, temp) schedule(static)
	for (j = 0; j < grid_points[1]; j++) {
		for (k = 0; k < grid_points[2]; k++) {
			eta = (double)j * dnym1;
			zeta = (double)k * dnzm1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 3);  

	/*--------------------------------------------------------------------
	  c     east face                                                      
	  c-------------------------------------------------------------------*/

	i = grid_points[0]-1;
	xi = 1.0;
	//trace_start("initialize", 4);  
#pragma omp parallel for private(eta, zeta, temp) schedule(static)
	for (j = 0; j < grid_points[1]; j++) {
		for (k = 0; k < grid_points[2]; k++) {
			eta = (double)j * dnym1;
			zeta = (double)k * dnzm1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 4);  

	/*--------------------------------------------------------------------
	  c     south face                                                 
	  c-------------------------------------------------------------------*/
	j = 0;
	eta = 0.0;
	//trace_start("initialize", 5);  
#pragma omp parallel for private(xi, zeta, temp) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (k = 0; k < grid_points[2]; k++) {
			xi = (double)i * dnxm1;
			zeta = (double)k * dnzm1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 5);  

	/*--------------------------------------------------------------------
	  c     north face                                    
	  c-------------------------------------------------------------------*/
	j = grid_points[1]-1;
	eta = 1.0;
	//trace_start("initialize", 6);  
#pragma omp parallel for private(xi, zeta, temp) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (k = 0; k < grid_points[2]; k++) {
			xi = (double)i * dnxm1;
			zeta = (double)k * dnzm1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 6);  

	/*--------------------------------------------------------------------
	  c     bottom face                                       
	  c-------------------------------------------------------------------*/
	k = 0;
	zeta = 0.0;
	//trace_start("initialize", 7);  
#pragma omp parallel for  private(xi, eta, temp) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			xi = (double)i *dnxm1;
			eta = (double)j * dnym1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 7);  

	/*--------------------------------------------------------------------
	  c     top face     
	  c-------------------------------------------------------------------*/
	k = grid_points[2]-1;
	zeta = 1.0;
	//trace_start("initialize", 8);  
#pragma omp parallel for private(xi, eta, temp) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			xi = (double)i * dnxm1;
			eta = (double)j * dnym1;
			exact_solution(xi, eta, zeta, temp);
			for (m = 0; m < 5; m++) {
				u[m][i][j][k] = temp[m];
			}
		}
	}
	//trace_stop("initialize", 8);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void lhsinit(void) {

	int i, j, k, m, n;

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     zero the whole left hand side for starters
	  -------------------------------------------------------------------*/
	//trace_start("lhsinit", 1);  
#pragma omp parallel for schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				for (m = 0; m < 5; m++) {
					for (n = 0; n < 5; n++) {
						lhs[0][m][n][i][j][k] = 0.0;
						lhs[1][m][n][i][j][k] = 0.0;
						lhs[2][m][n][i][j][k] = 0.0;
					}
				}
			}
		}
	}
	//trace_stop("lhsinit", 1);  

	/*--------------------------------------------------------------------
	  c     next, set all diagonal values to 1. This is overkill, but convenient
	  c-------------------------------------------------------------------*/
	//trace_start("lhsinit", 2);  
#pragma omp parallel for schedule(static) 
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				for (m = 0; m < 5; m++) {
					lhs[1][m][m][i][j][k] = 1.0;
				}
			}
		}
	}
	//trace_stop("lhsinit", 2);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void lhsx(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     This function computes the left hand side in the xi-direction
	  c-------------------------------------------------------------------*/

	int i, j, k;

	/*--------------------------------------------------------------------
	  c     determine a (labeled f) and n jacobians
	  c-------------------------------------------------------------------*/
	//trace_start("lhsx", 1);  
#pragma omp parallel for private(tmp1, tmp2, tmp3) schedule(dynamic)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (i = 0; i < grid_points[0]; i++) {

				tmp1 = 1.0 / u[0][i][j][k];
				tmp2 = tmp1 * tmp1;
				tmp3 = tmp1 * tmp2;
				/*--------------------------------------------------------------------
				  c     
				  c-------------------------------------------------------------------*/
				fjac[0][0][ i][ j][ k] = 0.0;
				fjac[0][1][ i][ j][ k] = 1.0;
				fjac[0][2][ i][ j][ k] = 0.0;
				fjac[0][3][ i][ j][ k] = 0.0;
				fjac[0][4][ i][ j][ k] = 0.0;

				fjac[1][0][ i][ j][ k] = -(u[1][i][j][k] * tmp2 * 
						u[1][i][j][k])
					+ c2 * 0.50 * (u[1][i][j][k] * u[1][i][j][k]
							+ u[2][i][j][k] * u[2][i][j][k]
							+ u[3][i][j][k] * u[3][i][j][k] ) * tmp2;
				fjac[1][1][i][j][k] = ( 2.0 - c2 )
					* ( u[1][i][j][k] / u[0][i][j][k] );
				fjac[1][2][i][j][k] = - c2 * ( u[2][i][j][k] * tmp1 );
				fjac[1][3][i][j][k] = - c2 * ( u[3][i][j][k] * tmp1 );
				fjac[1][4][i][j][k] = c2;

				fjac[2][0][i][j][k] = - ( u[1][i][j][k]*u[2][i][j][k] ) * tmp2;
				fjac[2][1][i][j][k] = u[2][i][j][k] * tmp1;
				fjac[2][2][i][j][k] = u[1][i][j][k] * tmp1;
				fjac[2][3][i][j][k] = 0.0;
				fjac[2][4][i][j][k] = 0.0;

				fjac[3][0][i][j][k] = - ( u[1][i][j][k]*u[3][i][j][k] ) * tmp2;
				fjac[3][1][i][j][k] = u[3][i][j][k] * tmp1;
				fjac[3][2][i][j][k] = 0.0;
				fjac[3][3][i][j][k] = u[1][i][j][k] * tmp1;
				fjac[3][4][i][j][k] = 0.0;

				fjac[4][0][i][j][k] = ( c2 * ( u[1][i][j][k] * u[1][i][j][k]
							+ u[2][i][j][k] * u[2][i][j][k]
							+ u[3][i][j][k] * u[3][i][j][k] ) * tmp2
						- c1 * ( u[4][i][j][k] * tmp1 ) )
					* ( u[1][i][j][k] * tmp1 );
				fjac[4][1][i][j][k] = c1 *  u[4][i][j][k] * tmp1 
					- 0.50 * c2
					* (  3.0*u[1][i][j][k]*u[1][i][j][k]
							+ u[2][i][j][k]*u[2][i][j][k]
							+ u[3][i][j][k]*u[3][i][j][k] ) * tmp2;
				fjac[4][2][i][j][k] = - c2 * ( u[2][i][j][k]*u[1][i][j][k] )
					* tmp2;
				fjac[4][3][i][j][k] = - c2 * ( u[3][i][j][k]*u[1][i][j][k] )
					* tmp2;
				fjac[4][4][i][j][k] = c1 * ( u[1][i][j][k] * tmp1 );

				njac[0][0][i][j][k] = 0.0;
				njac[0][1][i][j][k] = 0.0;
				njac[0][2][i][j][k] = 0.0;
				njac[0][3][i][j][k] = 0.0;
				njac[0][4][i][j][k] = 0.0;

				njac[1][0][i][j][k] = - con43 * c3c4 * tmp2 * u[1][i][j][k];
				njac[1][1][i][j][k] =   con43 * c3c4 * tmp1;
				njac[1][2][i][j][k] =   0.0;
				njac[1][3][i][j][k] =   0.0;
				njac[1][4][i][j][k] =   0.0;

				njac[2][0][i][j][k] = - c3c4 * tmp2 * u[2][i][j][k];
				njac[2][1][i][j][k] =   0.0;
				njac[2][2][i][j][k] =   c3c4 * tmp1;
				njac[2][3][i][j][k] =   0.0;
				njac[2][4][i][j][k] =   0.0;

				njac[3][0][i][j][k] = - c3c4 * tmp2 * u[3][i][j][k];
				njac[3][1][i][j][k] =   0.0;
				njac[3][2][i][j][k] =   0.0;
				njac[3][3][i][j][k] =   c3c4 * tmp1;
				njac[3][4][i][j][k] =   0.0;

				njac[4][0][i][j][k] = - ( con43 * c3c4
						- c1345 ) * tmp3 * (pow2(u[1][i][j][k]))
					- ( c3c4 - c1345 ) * tmp3 * (pow2(u[2][i][j][k]))
					- ( c3c4 - c1345 ) * tmp3 * (pow2(u[3][i][j][k]))
					- c1345 * tmp2 * u[4][i][j][k];

				njac[4][1][i][j][k] = ( con43 * c3c4
						- c1345 ) * tmp2 * u[1][i][j][k];
				njac[4][2][i][j][k] = ( c3c4 - c1345 ) * tmp2 * u[2][i][j][k];
				njac[4][3][i][j][k] = ( c3c4 - c1345 ) * tmp2 * u[3][i][j][k];
				njac[4][4][i][j][k] = ( c1345 ) * tmp1;

			}
			/*--------------------------------------------------------------------
			  c     now jacobians set, so form left hand side in x direction
			  c-------------------------------------------------------------------*/
			for (i = 1; i < grid_points[0]-1; i++) {

				tmp1 = dt * tx1;
				tmp2 = dt * tx2;

				lhs[AA][0][0][i][j][k] = - tmp2 * fjac[0][0][i-1][j][k]
					- tmp1 * njac[0][0][i-1][j][k]
					- tmp1 * dx1;
				lhs[AA][0][1][i][j][k] = - tmp2 * fjac[0][1][i-1][j][k]
					- tmp1 * njac[0][1][i-1][j][k];
				lhs[AA][0][2][i][j][k] = - tmp2 * fjac[0][2][i-1][j][k]
					- tmp1 * njac[0][2][i-1][j][k];
				lhs[AA][0][3][i][j][k] = - tmp2 * fjac[0][3][i-1][j][k]
					- tmp1 * njac[0][3][i-1][j][k];
				lhs[AA][0][4][i][j][k] = - tmp2 * fjac[0][4][i-1][j][k]
					- tmp1 * njac[0][4][i-1][j][k];

				lhs[AA][1][0][i][j][k] = - tmp2 * fjac[1][0][i-1][j][k]
					- tmp1 * njac[1][0][i-1][j][k];
				lhs[AA][1][1][i][j][k] = - tmp2 * fjac[1][1][i-1][j][k]
					- tmp1 * njac[1][1][i-1][j][k]
					- tmp1 * dx2;
				lhs[AA][1][2][i][j][k] = - tmp2 * fjac[1][2][i-1][j][k]
					- tmp1 * njac[1][2][i-1][j][k];
				lhs[AA][1][3][i][j][k] = - tmp2 * fjac[1][3][i-1][j][k]
					- tmp1 * njac[1][3][i-1][j][k];
				lhs[AA][1][4][i][j][k] = - tmp2 * fjac[1][4][i-1][j][k]
					- tmp1 * njac[1][4][i-1][j][k];

				lhs[AA][2][0][i][j][k] = - tmp2 * fjac[2][0][i-1][j][k]
					- tmp1 * njac[2][0][i-1][j][k];
				lhs[AA][2][1][i][j][k] = - tmp2 * fjac[2][1][i-1][j][k]
					- tmp1 * njac[2][1][i-1][j][k];
				lhs[AA][2][2][i][j][k] = - tmp2 * fjac[2][2][i-1][j][k]
					- tmp1 * njac[2][2][i-1][j][k]
					- tmp1 * dx3;
				lhs[AA][2][3][i][j][k] = - tmp2 * fjac[2][3][i-1][j][k]
					- tmp1 * njac[2][3][i-1][j][k];
				lhs[AA][2][4][i][j][k] = - tmp2 * fjac[2][4][i-1][j][k]
					- tmp1 * njac[2][4][i-1][j][k];

				lhs[AA][3][0][i][j][k] = - tmp2 * fjac[3][0][i-1][j][k]
					- tmp1 * njac[3][0][i-1][j][k];
				lhs[AA][3][1][i][j][k] = - tmp2 * fjac[3][1][i-1][j][k]
					- tmp1 * njac[3][1][i-1][j][k];
				lhs[AA][3][2][i][j][k] = - tmp2 * fjac[3][2][i-1][j][k]
					- tmp1 * njac[3][2][i-1][j][k];
				lhs[AA][3][3][i][j][k] = - tmp2 * fjac[3][3][i-1][j][k]
					- tmp1 * njac[3][3][i-1][j][k]
					- tmp1 * dx4;
				lhs[AA][3][4][i][j][k] = - tmp2 * fjac[3][4][i-1][j][k]
					- tmp1 * njac[3][4][i-1][j][k];

				lhs[AA][4][0][i][j][k] = - tmp2 * fjac[4][0][i-1][j][k]
					- tmp1 * njac[4][0][i-1][j][k];
				lhs[AA][4][1][i][j][k] = - tmp2 * fjac[4][1][i-1][j][k]
					- tmp1 * njac[4][1][i-1][j][k];
				lhs[AA][4][2][i][j][k] = - tmp2 * fjac[4][2][i-1][j][k]
					- tmp1 * njac[4][2][i-1][j][k];
				lhs[AA][4][3][i][j][k] = - tmp2 * fjac[4][3][i-1][j][k]
					- tmp1 * njac[4][3][i-1][j][k];
				lhs[AA][4][4][i][j][k] = - tmp2 * fjac[4][4][i-1][j][k]
					- tmp1 * njac[4][4][i-1][j][k]
					- tmp1 * dx5;

				lhs[BB][0][0][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[0][0][i][j][k]
					+ tmp1 * 2.0 * dx1;
				lhs[BB][0][1][i][j][k] = tmp1 * 2.0 * njac[0][1][i][j][k];
				lhs[BB][0][2][i][j][k] = tmp1 * 2.0 * njac[0][2][i][j][k];
				lhs[BB][0][3][i][j][k] = tmp1 * 2.0 * njac[0][3][i][j][k];
				lhs[BB][0][4][i][j][k] = tmp1 * 2.0 * njac[0][4][i][j][k];

				lhs[BB][1][0][i][j][k] = tmp1 * 2.0 * njac[1][0][i][j][k];
				lhs[BB][1][1][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[1][1][i][j][k]
					+ tmp1 * 2.0 * dx2;
				lhs[BB][1][2][i][j][k] = tmp1 * 2.0 * njac[1][2][i][j][k];
				lhs[BB][1][3][i][j][k] = tmp1 * 2.0 * njac[1][3][i][j][k];
				lhs[BB][1][4][i][j][k] = tmp1 * 2.0 * njac[1][4][i][j][k];

				lhs[BB][2][0][i][j][k] = tmp1 * 2.0 * njac[2][0][i][j][k];
				lhs[BB][2][1][i][j][k] = tmp1 * 2.0 * njac[2][1][i][j][k];
				lhs[BB][2][2][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[2][2][i][j][k]
					+ tmp1 * 2.0 * dx3;
				lhs[BB][2][3][i][j][k] = tmp1 * 2.0 * njac[2][3][i][j][k];
				lhs[BB][2][4][i][j][k] = tmp1 * 2.0 * njac[2][4][i][j][k];

				lhs[BB][3][0][i][j][k] = tmp1 * 2.0 * njac[3][0][i][j][k];
				lhs[BB][3][1][i][j][k] = tmp1 * 2.0 * njac[3][1][i][j][k];
				lhs[BB][3][2][i][j][k] = tmp1 * 2.0 * njac[3][2][i][j][k];
				lhs[BB][3][3][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[3][3][i][j][k]
					+ tmp1 * 2.0 * dx4;
				lhs[BB][3][4][i][j][k] = tmp1 * 2.0 * njac[3][4][i][j][k];

				lhs[BB][4][0][i][j][k] = tmp1 * 2.0 * njac[4][0][i][j][k];
				lhs[BB][4][1][i][j][k] = tmp1 * 2.0 * njac[4][1][i][j][k];
				lhs[BB][4][2][i][j][k] = tmp1 * 2.0 * njac[4][2][i][j][k];
				lhs[BB][4][3][i][j][k] = tmp1 * 2.0 * njac[4][3][i][j][k];
				lhs[BB][4][4][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[4][4][i][j][k]
					+ tmp1 * 2.0 * dx5;

				lhs[CC][0][0][i][j][k] =  tmp2 * fjac[0][0][i+1][j][k]
					- tmp1 * njac[0][0][i+1][j][k]
					- tmp1 * dx1;
				lhs[CC][0][1][i][j][k] =  tmp2 * fjac[0][1][i+1][j][k]
					- tmp1 * njac[0][1][i+1][j][k];
				lhs[CC][0][2][i][j][k] =  tmp2 * fjac[0][2][i+1][j][k]
					- tmp1 * njac[0][2][i+1][j][k];
				lhs[CC][0][3][i][j][k] =  tmp2 * fjac[0][3][i+1][j][k]
					- tmp1 * njac[0][3][i+1][j][k];
				lhs[CC][0][4][i][j][k] =  tmp2 * fjac[0][4][i+1][j][k]
					- tmp1 * njac[0][4][i+1][j][k];

				lhs[CC][1][0][i][j][k] =  tmp2 * fjac[1][0][i+1][j][k]
					- tmp1 * njac[1][0][i+1][j][k];
				lhs[CC][1][1][i][j][k] =  tmp2 * fjac[1][1][i+1][j][k]
					- tmp1 * njac[1][1][i+1][j][k]
					- tmp1 * dx2;
				lhs[CC][1][2][i][j][k] =  tmp2 * fjac[1][2][i+1][j][k]
					- tmp1 * njac[1][2][i+1][j][k];
				lhs[CC][1][3][i][j][k] =  tmp2 * fjac[1][3][i+1][j][k]
					- tmp1 * njac[1][3][i+1][j][k];
				lhs[CC][1][4][i][j][k] =  tmp2 * fjac[1][4][i+1][j][k]
					- tmp1 * njac[1][4][i+1][j][k];

				lhs[CC][2][0][i][j][k] =  tmp2 * fjac[2][0][i+1][j][k]
					- tmp1 * njac[2][0][i+1][j][k];
				lhs[CC][2][1][i][j][k] =  tmp2 * fjac[2][1][i+1][j][k]
					- tmp1 * njac[2][1][i+1][j][k];
				lhs[CC][2][2][i][j][k] =  tmp2 * fjac[2][2][i+1][j][k]
					- tmp1 * njac[2][2][i+1][j][k]
					- tmp1 * dx3;
				lhs[CC][2][3][i][j][k] =  tmp2 * fjac[2][3][i+1][j][k]
					- tmp1 * njac[2][3][i+1][j][k];
				lhs[CC][2][4][i][j][k] =  tmp2 * fjac[2][4][i+1][j][k]
					- tmp1 * njac[2][4][i+1][j][k];

				lhs[CC][3][0][i][j][k] =  tmp2 * fjac[3][0][i+1][j][k]
					- tmp1 * njac[3][0][i+1][j][k];
				lhs[CC][3][1][i][j][k] =  tmp2 * fjac[3][1][i+1][j][k]
					- tmp1 * njac[3][1][i+1][j][k];
				lhs[CC][3][2][i][j][k] =  tmp2 * fjac[3][2][i+1][j][k]
					- tmp1 * njac[3][2][i+1][j][k];
				lhs[CC][3][3][i][j][k] =  tmp2 * fjac[3][3][i+1][j][k]
					- tmp1 * njac[3][3][i+1][j][k]
					- tmp1 * dx4;
				lhs[CC][3][4][i][j][k] =  tmp2 * fjac[3][4][i+1][j][k]
					- tmp1 * njac[3][4][i+1][j][k];

				lhs[CC][4][0][i][j][k] =  tmp2 * fjac[4][0][i+1][j][k]
					- tmp1 * njac[4][0][i+1][j][k];
				lhs[CC][4][1][i][j][k] =  tmp2 * fjac[4][1][i+1][j][k]
					- tmp1 * njac[4][1][i+1][j][k];
				lhs[CC][4][2][i][j][k] =  tmp2 * fjac[4][2][i+1][j][k]
					- tmp1 * njac[4][2][i+1][j][k];
				lhs[CC][4][3][i][j][k] =  tmp2 * fjac[4][3][i+1][j][k]
					- tmp1 * njac[4][3][i+1][j][k];
				lhs[CC][4][4][i][j][k] =  tmp2 * fjac[4][4][i+1][j][k]
					- tmp1 * njac[4][4][i+1][j][k]
					- tmp1 * dx5;

			}
		}
	}
	//trace_stop("lhsx", 1);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void lhsy(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     This function computes the left hand side for the three y-factors   
	  c-------------------------------------------------------------------*/

	int i, j, k;

	/*--------------------------------------------------------------------
	  c     Compute the indices for storing the tri-diagonal matrix;
	  c     determine a (labeled f) and n jacobians for cell c
	  c-------------------------------------------------------------------*/
	//trace_start("lhsy", 1);  
#pragma omp parallel for private(tmp1, tmp2, tmp3) schedule(dynamic)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {

				tmp1 = 1.0 / u[0][i][j][k];
				tmp2 = tmp1 * tmp1;
				tmp3 = tmp1 * tmp2;

				fjac[0][0][ i][ j][ k] = 0.0;
				fjac[0][1][ i][ j][ k] = 0.0;
				fjac[0][2][ i][ j][ k] = 1.0;
				fjac[0][3][ i][ j][ k] = 0.0;
				fjac[0][4][ i][ j][ k] = 0.0;

				fjac[1][0][i][j][k] = - ( u[1][i][j][k]*u[2][i][j][k] )
					* tmp2;
				fjac[1][1][i][j][k] = u[2][i][j][k] * tmp1;
				fjac[1][2][i][j][k] = u[1][i][j][k] * tmp1;
				fjac[1][3][i][j][k] = 0.0;
				fjac[1][4][i][j][k] = 0.0;

				fjac[2][0][i][j][k] = - ( u[2][i][j][k]*u[2][i][j][k]*tmp2)
					+ 0.50 * c2 * ( (  u[1][i][j][k] * u[1][i][j][k]
								+ u[2][i][j][k] * u[2][i][j][k]
								+ u[3][i][j][k] * u[3][i][j][k] )
							* tmp2 );
				fjac[2][1][i][j][k] = - c2 *  u[1][i][j][k] * tmp1;
				fjac[2][2][i][j][k] = ( 2.0 - c2 )
					*  u[2][i][j][k] * tmp1;
				fjac[2][3][i][j][k] = - c2 * u[3][i][j][k] * tmp1;
				fjac[2][4][i][j][k] = c2;

				fjac[3][0][i][j][k] = - ( u[2][i][j][k]*u[3][i][j][k] )
					* tmp2;
				fjac[3][1][i][j][k] = 0.0;
				fjac[3][2][i][j][k] = u[3][i][j][k] * tmp1;
				fjac[3][3][i][j][k] = u[2][i][j][k] * tmp1;
				fjac[3][4][i][j][k] = 0.0;

				fjac[4][0][i][j][k] = ( c2 * (  u[1][i][j][k] * u[1][i][j][k]
							+ u[2][i][j][k] * u[2][i][j][k]
							+ u[3][i][j][k] * u[3][i][j][k] )
						* tmp2
						- c1 * u[4][i][j][k] * tmp1 ) 
					* u[2][i][j][k] * tmp1;
				fjac[4][1][i][j][k] = - c2 * u[1][i][j][k]*u[2][i][j][k] 
					* tmp2;
				fjac[4][2][i][j][k] = c1 * u[4][i][j][k] * tmp1 
					- 0.50 * c2 
					* ( (  u[1][i][j][k]*u[1][i][j][k]
								+ 3.0 * u[2][i][j][k]*u[2][i][j][k]
								+ u[3][i][j][k]*u[3][i][j][k] )
							* tmp2 );
				fjac[4][3][i][j][k] = - c2 * ( u[2][i][j][k]*u[3][i][j][k] )
					* tmp2;
				fjac[4][4][i][j][k] = c1 * u[2][i][j][k] * tmp1; 

				njac[0][0][i][j][k] = 0.0;
				njac[0][1][i][j][k] = 0.0;
				njac[0][2][i][j][k] = 0.0;
				njac[0][3][i][j][k] = 0.0;
				njac[0][4][i][j][k] = 0.0;

				njac[1][0][i][j][k] = - c3c4 * tmp2 * u[1][i][j][k];
				njac[1][1][i][j][k] =   c3c4 * tmp1;
				njac[1][2][i][j][k] =   0.0;
				njac[1][3][i][j][k] =   0.0;
				njac[1][4][i][j][k] =   0.0;

				njac[2][0][i][j][k] = - con43 * c3c4 * tmp2 * u[2][i][j][k];
				njac[2][1][i][j][k] =   0.0;
				njac[2][2][i][j][k] =   con43 * c3c4 * tmp1;
				njac[2][3][i][j][k] =   0.0;
				njac[2][4][i][j][k] =   0.0;

				njac[3][0][i][j][k] = - c3c4 * tmp2 * u[3][i][j][k];
				njac[3][1][i][j][k] =   0.0;
				njac[3][2][i][j][k] =   0.0;
				njac[3][3][i][j][k] =   c3c4 * tmp1;
				njac[3][4][i][j][k] =   0.0;

				njac[4][0][i][j][k] = - (  c3c4
						- c1345 ) * tmp3 * (pow2(u[1][i][j][k]))
					- ( con43 * c3c4
							- c1345 ) * tmp3 * (pow2(u[2][i][j][k]))
					- ( c3c4 - c1345 ) * tmp3 * (pow2(u[3][i][j][k]))
					- c1345 * tmp2 * u[4][i][j][k];

				njac[4][1][i][j][k] = (  c3c4 - c1345 ) * tmp2 * u[1][i][j][k];
				njac[4][2][i][j][k] = ( con43 * c3c4
						- c1345 ) * tmp2 * u[2][i][j][k];
				njac[4][3][i][j][k] = ( c3c4 - c1345 ) * tmp2 * u[3][i][j][k];
				njac[4][4][i][j][k] = ( c1345 ) * tmp1;

			}
		}
	}
	//trace_stop("lhsy", 1);  

	/*--------------------------------------------------------------------
	  c     now joacobians set, so form left hand side in y direction
	  c-------------------------------------------------------------------*/
	//trace_start("lhsy", 2);  
#pragma omp parallel for private(tmp1, tmp2) schedule(dynamic)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {

				tmp1 = dt * ty1;
				tmp2 = dt * ty2;

				lhs[AA][0][0][i][j][k] = - tmp2 * fjac[0][0][i][j-1][k]
					- tmp1 * njac[0][0][i][j-1][k]
					- tmp1 * dy1;
				lhs[AA][0][1][i][j][k] = - tmp2 * fjac[0][1][i][j-1][k]
					- tmp1 * njac[0][1][i][j-1][k];
				lhs[AA][0][2][i][j][k] = - tmp2 * fjac[0][2][i][j-1][k]
					- tmp1 * njac[0][2][i][j-1][k];
				lhs[AA][0][3][i][j][k] = - tmp2 * fjac[0][3][i][j-1][k]
					- tmp1 * njac[0][3][i][j-1][k];
				lhs[AA][0][4][i][j][k] = - tmp2 * fjac[0][4][i][j-1][k]
					- tmp1 * njac[0][4][i][j-1][k];

				lhs[AA][1][0][i][j][k] = - tmp2 * fjac[1][0][i][j-1][k]
					- tmp1 * njac[1][0][i][j-1][k];
				lhs[AA][1][1][i][j][k] = - tmp2 * fjac[1][1][i][j-1][k]
					- tmp1 * njac[1][1][i][j-1][k]
					- tmp1 * dy2;
				lhs[AA][1][2][i][j][k] = - tmp2 * fjac[1][2][i][j-1][k]
					- tmp1 * njac[1][2][i][j-1][k];
				lhs[AA][1][3][i][j][k] = - tmp2 * fjac[1][3][i][j-1][k]
					- tmp1 * njac[1][3][i][j-1][k];
				lhs[AA][1][4][i][j][k] = - tmp2 * fjac[1][4][i][j-1][k]
					- tmp1 * njac[1][4][i][j-1][k];

				lhs[AA][2][0][i][j][k] = - tmp2 * fjac[2][0][i][j-1][k]
					- tmp1 * njac[2][0][i][j-1][k];
				lhs[AA][2][1][i][j][k] = - tmp2 * fjac[2][1][i][j-1][k]
					- tmp1 * njac[2][1][i][j-1][k];
				lhs[AA][2][2][i][j][k] = - tmp2 * fjac[2][2][i][j-1][k]
					- tmp1 * njac[2][2][i][j-1][k]
					- tmp1 * dy3;
				lhs[AA][2][3][i][j][k] = - tmp2 * fjac[2][3][i][j-1][k]
					- tmp1 * njac[2][3][i][j-1][k];
				lhs[AA][2][4][i][j][k] = - tmp2 * fjac[2][4][i][j-1][k]
					- tmp1 * njac[2][4][i][j-1][k];

				lhs[AA][3][0][i][j][k] = - tmp2 * fjac[3][0][i][j-1][k]
					- tmp1 * njac[3][0][i][j-1][k];
				lhs[AA][3][1][i][j][k] = - tmp2 * fjac[3][1][i][j-1][k]
					- tmp1 * njac[3][1][i][j-1][k];
				lhs[AA][3][2][i][j][k] = - tmp2 * fjac[3][2][i][j-1][k]
					- tmp1 * njac[3][2][i][j-1][k];
				lhs[AA][3][3][i][j][k] = - tmp2 * fjac[3][3][i][j-1][k]
					- tmp1 * njac[3][3][i][j-1][k]
					- tmp1 * dy4;
				lhs[AA][3][4][i][j][k] = - tmp2 * fjac[3][4][i][j-1][k]
					- tmp1 * njac[3][4][i][j-1][k];

				lhs[AA][4][0][i][j][k] = - tmp2 * fjac[4][0][i][j-1][k]
					- tmp1 * njac[4][0][i][j-1][k];
				lhs[AA][4][1][i][j][k] = - tmp2 * fjac[4][1][i][j-1][k]
					- tmp1 * njac[4][1][i][j-1][k];
				lhs[AA][4][2][i][j][k] = - tmp2 * fjac[4][2][i][j-1][k]
					- tmp1 * njac[4][2][i][j-1][k];
				lhs[AA][4][3][i][j][k] = - tmp2 * fjac[4][3][i][j-1][k]
					- tmp1 * njac[4][3][i][j-1][k];
				lhs[AA][4][4][i][j][k] = - tmp2 * fjac[4][4][i][j-1][k]
					- tmp1 * njac[4][4][i][j-1][k]
					- tmp1 * dy5;

				lhs[BB][0][0][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[0][0][i][j][k]
					+ tmp1 * 2.0 * dy1;
				lhs[BB][0][1][i][j][k] = tmp1 * 2.0 * njac[0][1][i][j][k];
				lhs[BB][0][2][i][j][k] = tmp1 * 2.0 * njac[0][2][i][j][k];
				lhs[BB][0][3][i][j][k] = tmp1 * 2.0 * njac[0][3][i][j][k];
				lhs[BB][0][4][i][j][k] = tmp1 * 2.0 * njac[0][4][i][j][k];

				lhs[BB][1][0][i][j][k] = tmp1 * 2.0 * njac[1][0][i][j][k];
				lhs[BB][1][1][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[1][1][i][j][k]
					+ tmp1 * 2.0 * dy2;
				lhs[BB][1][2][i][j][k] = tmp1 * 2.0 * njac[1][2][i][j][k];
				lhs[BB][1][3][i][j][k] = tmp1 * 2.0 * njac[1][3][i][j][k];
				lhs[BB][1][4][i][j][k] = tmp1 * 2.0 * njac[1][4][i][j][k];

				lhs[BB][2][0][i][j][k] = tmp1 * 2.0 * njac[2][0][i][j][k];
				lhs[BB][2][1][i][j][k] = tmp1 * 2.0 * njac[2][1][i][j][k];
				lhs[BB][2][2][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[2][2][i][j][k]
					+ tmp1 * 2.0 * dy3;
				lhs[BB][2][3][i][j][k] = tmp1 * 2.0 * njac[2][3][i][j][k];
				lhs[BB][2][4][i][j][k] = tmp1 * 2.0 * njac[2][4][i][j][k];

				lhs[BB][3][0][i][j][k] = tmp1 * 2.0 * njac[3][0][i][j][k];
				lhs[BB][3][1][i][j][k] = tmp1 * 2.0 * njac[3][1][i][j][k];
				lhs[BB][3][2][i][j][k] = tmp1 * 2.0 * njac[3][2][i][j][k];
				lhs[BB][3][3][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[3][3][i][j][k]
					+ tmp1 * 2.0 * dy4;
				lhs[BB][3][4][i][j][k] = tmp1 * 2.0 * njac[3][4][i][j][k];

				lhs[BB][4][0][i][j][k] = tmp1 * 2.0 * njac[4][0][i][j][k];
				lhs[BB][4][1][i][j][k] = tmp1 * 2.0 * njac[4][1][i][j][k];
				lhs[BB][4][2][i][j][k] = tmp1 * 2.0 * njac[4][2][i][j][k];
				lhs[BB][4][3][i][j][k] = tmp1 * 2.0 * njac[4][3][i][j][k];
				lhs[BB][4][4][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[4][4][i][j][k] 
					+ tmp1 * 2.0 * dy5;

				lhs[CC][0][0][i][j][k] =  tmp2 * fjac[0][0][i][j+1][k]
					- tmp1 * njac[0][0][i][j+1][k]
					- tmp1 * dy1;
				lhs[CC][0][1][i][j][k] =  tmp2 * fjac[0][1][i][j+1][k]
					- tmp1 * njac[0][1][i][j+1][k];
				lhs[CC][0][2][i][j][k] =  tmp2 * fjac[0][2][i][j+1][k]
					- tmp1 * njac[0][2][i][j+1][k];
				lhs[CC][0][3][i][j][k] =  tmp2 * fjac[0][3][i][j+1][k]
					- tmp1 * njac[0][3][i][j+1][k];
				lhs[CC][0][4][i][j][k] =  tmp2 * fjac[0][4][i][j+1][k]
					- tmp1 * njac[0][4][i][j+1][k];

				lhs[CC][1][0][i][j][k] =  tmp2 * fjac[1][0][i][j+1][k]
					- tmp1 * njac[1][0][i][j+1][k];
				lhs[CC][1][1][i][j][k] =  tmp2 * fjac[1][1][i][j+1][k]
					- tmp1 * njac[1][1][i][j+1][k]
					- tmp1 * dy2;
				lhs[CC][1][2][i][j][k] =  tmp2 * fjac[1][2][i][j+1][k]
					- tmp1 * njac[1][2][i][j+1][k];
				lhs[CC][1][3][i][j][k] =  tmp2 * fjac[1][3][i][j+1][k]
					- tmp1 * njac[1][3][i][j+1][k];
				lhs[CC][1][4][i][j][k] =  tmp2 * fjac[1][4][i][j+1][k]
					- tmp1 * njac[1][4][i][j+1][k];

				lhs[CC][2][0][i][j][k] =  tmp2 * fjac[2][0][i][j+1][k]
					- tmp1 * njac[2][0][i][j+1][k];
				lhs[CC][2][1][i][j][k] =  tmp2 * fjac[2][1][i][j+1][k]
					- tmp1 * njac[2][1][i][j+1][k];
				lhs[CC][2][2][i][j][k] =  tmp2 * fjac[2][2][i][j+1][k]
					- tmp1 * njac[2][2][i][j+1][k]
					- tmp1 * dy3;
				lhs[CC][2][3][i][j][k] =  tmp2 * fjac[2][3][i][j+1][k]
					- tmp1 * njac[2][3][i][j+1][k];
				lhs[CC][2][4][i][j][k] =  tmp2 * fjac[2][4][i][j+1][k]
					- tmp1 * njac[2][4][i][j+1][k];

				lhs[CC][3][0][i][j][k] =  tmp2 * fjac[3][0][i][j+1][k]
					- tmp1 * njac[3][0][i][j+1][k];
				lhs[CC][3][1][i][j][k] =  tmp2 * fjac[3][1][i][j+1][k]
					- tmp1 * njac[3][1][i][j+1][k];
				lhs[CC][3][2][i][j][k] =  tmp2 * fjac[3][2][i][j+1][k]
					- tmp1 * njac[3][2][i][j+1][k];
				lhs[CC][3][3][i][j][k] =  tmp2 * fjac[3][3][i][j+1][k]
					- tmp1 * njac[3][3][i][j+1][k]
					- tmp1 * dy4;
				lhs[CC][3][4][i][j][k] =  tmp2 * fjac[3][4][i][j+1][k]
					- tmp1 * njac[3][4][i][j+1][k];

				lhs[CC][4][0][i][j][k] =  tmp2 * fjac[4][0][i][j+1][k]
					- tmp1 * njac[4][0][i][j+1][k];
				lhs[CC][4][1][i][j][k] =  tmp2 * fjac[4][1][i][j+1][k]
					- tmp1 * njac[4][1][i][j+1][k];
				lhs[CC][4][2][i][j][k] =  tmp2 * fjac[4][2][i][j+1][k]
					- tmp1 * njac[4][2][i][j+1][k];
				lhs[CC][4][3][i][j][k] =  tmp2 * fjac[4][3][i][j+1][k]
					- tmp1 * njac[4][3][i][j+1][k];
				lhs[CC][4][4][i][j][k] =  tmp2 * fjac[4][4][i][j+1][k]
					- tmp1 * njac[4][4][i][j+1][k]
					- tmp1 * dy5;

			}
		}
	}
	//trace_stop("lhsy", 2);  
}


/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void lhsz(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     This function computes the left hand side for the three z-factors   
	  c-------------------------------------------------------------------*/

	int i, j, k;

	/*--------------------------------------------------------------------
	  c     Compute the indices for storing the block-diagonal matrix;
	  c     determine c (labeled f) and s jacobians
	  c---------------------------------------------------------------------*/
	//trace_start("lhsz", 1);  
#pragma omp parallel for private(tmp1, tmp2, tmp3) schedule(dynamic)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 0; k < grid_points[2]; k++) {

				tmp1 = 1.0 / u[0][i][j][k];
				tmp2 = tmp1 * tmp1;
				tmp3 = tmp1 * tmp2;

				fjac[0][0][i][j][k] = 0.0;
				fjac[0][1][i][j][k] = 0.0;
				fjac[0][2][i][j][k] = 0.0;
				fjac[0][3][i][j][k] = 1.0;
				fjac[0][4][i][j][k] = 0.0;

				fjac[1][0][i][j][k] = - ( u[1][i][j][k]*u[3][i][j][k] ) 
					* tmp2;
				fjac[1][1][i][j][k] = u[3][i][j][k] * tmp1;
				fjac[1][2][i][j][k] = 0.0;
				fjac[1][3][i][j][k] = u[1][i][j][k] * tmp1;
				fjac[1][4][i][j][k] = 0.0;

				fjac[2][0][i][j][k] = - ( u[2][i][j][k]*u[3][i][j][k] )
					* tmp2;
				fjac[2][1][i][j][k] = 0.0;
				fjac[2][2][i][j][k] = u[3][i][j][k] * tmp1;
				fjac[2][3][i][j][k] = u[2][i][j][k] * tmp1;
				fjac[2][4][i][j][k] = 0.0;

				fjac[3][0][i][j][k] = - (u[3][i][j][k]*u[3][i][j][k] * tmp2 ) 
					+ 0.50 * c2 * ( (  u[1][i][j][k] * u[1][i][j][k]
								+ u[2][i][j][k] * u[2][i][j][k]
								+ u[3][i][j][k] * u[3][i][j][k] ) * tmp2 );
				fjac[3][1][i][j][k] = - c2 *  u[1][i][j][k] * tmp1;
				fjac[3][2][i][j][k] = - c2 *  u[2][i][j][k] * tmp1;
				fjac[3][3][i][j][k] = ( 2.0 - c2 )
					*  u[3][i][j][k] * tmp1;
				fjac[3][4][i][j][k] = c2;

				fjac[4][0][i][j][k] = ( c2 * (  u[1][i][j][k] * u[1][i][j][k]
							+ u[2][i][j][k] * u[2][i][j][k]
							+ u[3][i][j][k] * u[3][i][j][k] )
						* tmp2
						- c1 * ( u[4][i][j][k] * tmp1 ) )
					* ( u[3][i][j][k] * tmp1 );
				fjac[4][1][i][j][k] = - c2 * ( u[1][i][j][k]*u[3][i][j][k] )
					* tmp2;
				fjac[4][2][i][j][k] = - c2 * ( u[2][i][j][k]*u[3][i][j][k] )
					* tmp2;
				fjac[4][3][i][j][k] = c1 * ( u[4][i][j][k] * tmp1 )
					- 0.50 * c2
					* ( (  u[1][i][j][k]*u[1][i][j][k]
								+ u[2][i][j][k]*u[2][i][j][k]
								+ 3.0*u[3][i][j][k]*u[3][i][j][k] )
							* tmp2 );
				fjac[4][4][i][j][k] = c1 * u[3][i][j][k] * tmp1;

				njac[0][0][i][j][k] = 0.0;
				njac[0][1][i][j][k] = 0.0;
				njac[0][2][i][j][k] = 0.0;
				njac[0][3][i][j][k] = 0.0;
				njac[0][4][i][j][k] = 0.0;

				njac[1][0][i][j][k] = - c3c4 * tmp2 * u[1][i][j][k];
				njac[1][1][i][j][k] =   c3c4 * tmp1;
				njac[1][2][i][j][k] =   0.0;
				njac[1][3][i][j][k] =   0.0;
				njac[1][4][i][j][k] =   0.0;

				njac[2][0][i][j][k] = - c3c4 * tmp2 * u[2][i][j][k];
				njac[2][1][i][j][k] =   0.0;
				njac[2][2][i][j][k] =   c3c4 * tmp1;
				njac[2][3][i][j][k] =   0.0;
				njac[2][4][i][j][k] =   0.0;

				njac[3][0][i][j][k] = - con43 * c3c4 * tmp2 * u[3][i][j][k];
				njac[3][1][i][j][k] =   0.0;
				njac[3][2][i][j][k] =   0.0;
				njac[3][3][i][j][k] =   con43 * c3 * c4 * tmp1;
				njac[3][4][i][j][k] =   0.0;

				njac[4][0][i][j][k] = - (  c3c4
						- c1345 ) * tmp3 * (pow2(u[1][i][j][k]))
					- ( c3c4 - c1345 ) * tmp3 * (pow2(u[2][i][j][k]))
					- ( con43 * c3c4
							- c1345 ) * tmp3 * (pow2(u[3][i][j][k]))
					- c1345 * tmp2 * u[4][i][j][k];

				njac[4][1][i][j][k] = (  c3c4 - c1345 ) * tmp2 * u[1][i][j][k];
				njac[4][2][i][j][k] = (  c3c4 - c1345 ) * tmp2 * u[2][i][j][k];
				njac[4][3][i][j][k] = ( con43 * c3c4
						- c1345 ) * tmp2 * u[3][i][j][k];
				njac[4][4][i][j][k] = ( c1345 )* tmp1;

			}
		}
	}
	//trace_stop("lhsz", 1);  

	/*--------------------------------------------------------------------
	  c     now jacobians set, so form left hand side in z direction
	  c-------------------------------------------------------------------*/

	//trace_start("lhsz", 2);  
#pragma omp parallel for private(tmp1, tmp2, tmp3) schedule(dynamic)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {

				tmp1 = dt * tz1;
				tmp2 = dt * tz2;

				lhs[AA][0][0][i][j][k] = - tmp2 * fjac[0][0][i][j][k-1]
					- tmp1 * njac[0][0][i][j][k-1]
					- tmp1 * dz1;
				lhs[AA][0][1][i][j][k] = - tmp2 * fjac[0][1][i][j][k-1]
					- tmp1 * njac[0][1][i][j][k-1];
				lhs[AA][0][2][i][j][k] = - tmp2 * fjac[0][2][i][j][k-1]
					- tmp1 * njac[0][2][i][j][k-1];
				lhs[AA][0][3][i][j][k] = - tmp2 * fjac[0][3][i][j][k-1]
					- tmp1 * njac[0][3][i][j][k-1];
				lhs[AA][0][4][i][j][k] = - tmp2 * fjac[0][4][i][j][k-1]
					- tmp1 * njac[0][4][i][j][k-1];

				lhs[AA][1][0][i][j][k] = - tmp2 * fjac[1][0][i][j][k-1]
					- tmp1 * njac[1][0][i][j][k-1];
				lhs[AA][1][1][i][j][k] = - tmp2 * fjac[1][1][i][j][k-1]
					- tmp1 * njac[1][1][i][j][k-1]
					- tmp1 * dz2;
				lhs[AA][1][2][i][j][k] = - tmp2 * fjac[1][2][i][j][k-1]
					- tmp1 * njac[1][2][i][j][k-1];
				lhs[AA][1][3][i][j][k] = - tmp2 * fjac[1][3][i][j][k-1]
					- tmp1 * njac[1][3][i][j][k-1];
				lhs[AA][1][4][i][j][k] = - tmp2 * fjac[1][4][i][j][k-1]
					- tmp1 * njac[1][4][i][j][k-1];

				lhs[AA][2][0][i][j][k] = - tmp2 * fjac[2][0][i][j][k-1]
					- tmp1 * njac[2][0][i][j][k-1];
				lhs[AA][2][1][i][j][k] = - tmp2 * fjac[2][1][i][j][k-1]
					- tmp1 * njac[2][1][i][j][k-1];
				lhs[AA][2][2][i][j][k] = - tmp2 * fjac[2][2][i][j][k-1]
					- tmp1 * njac[2][2][i][j][k-1]
					- tmp1 * dz3;
				lhs[AA][2][3][i][j][k] = - tmp2 * fjac[2][3][i][j][k-1]
					- tmp1 * njac[2][3][i][j][k-1];
				lhs[AA][2][4][i][j][k] = - tmp2 * fjac[2][4][i][j][k-1]
					- tmp1 * njac[2][4][i][j][k-1];

				lhs[AA][3][0][i][j][k] = - tmp2 * fjac[3][0][i][j][k-1]
					- tmp1 * njac[3][0][i][j][k-1];
				lhs[AA][3][1][i][j][k] = - tmp2 * fjac[3][1][i][j][k-1]
					- tmp1 * njac[3][1][i][j][k-1];
				lhs[AA][3][2][i][j][k] = - tmp2 * fjac[3][2][i][j][k-1]
					- tmp1 * njac[3][2][i][j][k-1];
				lhs[AA][3][3][i][j][k] = - tmp2 * fjac[3][3][i][j][k-1]
					- tmp1 * njac[3][3][i][j][k-1]
					- tmp1 * dz4;
				lhs[AA][3][4][i][j][k] = - tmp2 * fjac[3][4][i][j][k-1]
					- tmp1 * njac[3][4][i][j][k-1];

				lhs[AA][4][0][i][j][k] = - tmp2 * fjac[4][0][i][j][k-1]
					- tmp1 * njac[4][0][i][j][k-1];
				lhs[AA][4][1][i][j][k] = - tmp2 * fjac[4][1][i][j][k-1]
					- tmp1 * njac[4][1][i][j][k-1];
				lhs[AA][4][2][i][j][k] = - tmp2 * fjac[4][2][i][j][k-1]
					- tmp1 * njac[4][2][i][j][k-1];
				lhs[AA][4][3][i][j][k] = - tmp2 * fjac[4][3][i][j][k-1]
					- tmp1 * njac[4][3][i][j][k-1];
				lhs[AA][4][4][i][j][k] = - tmp2 * fjac[4][4][i][j][k-1]
					- tmp1 * njac[4][4][i][j][k-1]
					- tmp1 * dz5;

				lhs[BB][0][0][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[0][0][i][j][k]
					+ tmp1 * 2.0 * dz1;
				lhs[BB][0][1][i][j][k] = tmp1 * 2.0 * njac[0][1][i][j][k];
				lhs[BB][0][2][i][j][k] = tmp1 * 2.0 * njac[0][2][i][j][k];
				lhs[BB][0][3][i][j][k] = tmp1 * 2.0 * njac[0][3][i][j][k];
				lhs[BB][0][4][i][j][k] = tmp1 * 2.0 * njac[0][4][i][j][k];

				lhs[BB][1][0][i][j][k] = tmp1 * 2.0 * njac[1][0][i][j][k];
				lhs[BB][1][1][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[1][1][i][j][k]
					+ tmp1 * 2.0 * dz2;
				lhs[BB][1][2][i][j][k] = tmp1 * 2.0 * njac[1][2][i][j][k];
				lhs[BB][1][3][i][j][k] = tmp1 * 2.0 * njac[1][3][i][j][k];
				lhs[BB][1][4][i][j][k] = tmp1 * 2.0 * njac[1][4][i][j][k];

				lhs[BB][2][0][i][j][k] = tmp1 * 2.0 * njac[2][0][i][j][k];
				lhs[BB][2][1][i][j][k] = tmp1 * 2.0 * njac[2][1][i][j][k];
				lhs[BB][2][2][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[2][2][i][j][k]
					+ tmp1 * 2.0 * dz3;
				lhs[BB][2][3][i][j][k] = tmp1 * 2.0 * njac[2][3][i][j][k];
				lhs[BB][2][4][i][j][k] = tmp1 * 2.0 * njac[2][4][i][j][k];

				lhs[BB][3][0][i][j][k] = tmp1 * 2.0 * njac[3][0][i][j][k];
				lhs[BB][3][1][i][j][k] = tmp1 * 2.0 * njac[3][1][i][j][k];
				lhs[BB][3][2][i][j][k] = tmp1 * 2.0 * njac[3][2][i][j][k];
				lhs[BB][3][3][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[3][3][i][j][k]
					+ tmp1 * 2.0 * dz4;
				lhs[BB][3][4][i][j][k] = tmp1 * 2.0 * njac[3][4][i][j][k];

				lhs[BB][4][0][i][j][k] = tmp1 * 2.0 * njac[4][0][i][j][k];
				lhs[BB][4][1][i][j][k] = tmp1 * 2.0 * njac[4][1][i][j][k];
				lhs[BB][4][2][i][j][k] = tmp1 * 2.0 * njac[4][2][i][j][k];
				lhs[BB][4][3][i][j][k] = tmp1 * 2.0 * njac[4][3][i][j][k];
				lhs[BB][4][4][i][j][k] = 1.0
					+ tmp1 * 2.0 * njac[4][4][i][j][k]
					+ tmp1 * 2.0 * dz5;

				lhs[CC][0][0][i][j][k] =  tmp2 * fjac[0][0][i][j][k+1]
					- tmp1 * njac[0][0][i][j][k+1]
					- tmp1 * dz1;
				lhs[CC][0][1][i][j][k] =  tmp2 * fjac[0][1][i][j][k+1]
					- tmp1 * njac[0][1][i][j][k+1];
				lhs[CC][0][2][i][j][k] =  tmp2 * fjac[0][2][i][j][k+1]
					- tmp1 * njac[0][2][i][j][k+1];
				lhs[CC][0][3][i][j][k] =  tmp2 * fjac[0][3][i][j][k+1]
					- tmp1 * njac[0][3][i][j][k+1];
				lhs[CC][0][4][i][j][k] =  tmp2 * fjac[0][4][i][j][k+1]
					- tmp1 * njac[0][4][i][j][k+1];

				lhs[CC][1][0][i][j][k] =  tmp2 * fjac[1][0][i][j][k+1]
					- tmp1 * njac[1][0][i][j][k+1];
				lhs[CC][1][1][i][j][k] =  tmp2 * fjac[1][1][i][j][k+1]
					- tmp1 * njac[1][1][i][j][k+1]
					- tmp1 * dz2;
				lhs[CC][1][2][i][j][k] =  tmp2 * fjac[1][2][i][j][k+1]
					- tmp1 * njac[1][2][i][j][k+1];
				lhs[CC][1][3][i][j][k] =  tmp2 * fjac[1][3][i][j][k+1]
					- tmp1 * njac[1][3][i][j][k+1];
				lhs[CC][1][4][i][j][k] =  tmp2 * fjac[1][4][i][j][k+1]
					- tmp1 * njac[1][4][i][j][k+1];

				lhs[CC][2][0][i][j][k] =  tmp2 * fjac[2][0][i][j][k+1]
					- tmp1 * njac[2][0][i][j][k+1];
				lhs[CC][2][1][i][j][k] =  tmp2 * fjac[2][1][i][j][k+1]
					- tmp1 * njac[2][1][i][j][k+1];
				lhs[CC][2][2][i][j][k] =  tmp2 * fjac[2][2][i][j][k+1]
					- tmp1 * njac[2][2][i][j][k+1]
					- tmp1 * dz3;
				lhs[CC][2][3][i][j][k] =  tmp2 * fjac[2][3][i][j][k+1]
					- tmp1 * njac[2][3][i][j][k+1];
				lhs[CC][2][4][i][j][k] =  tmp2 * fjac[2][4][i][j][k+1]
					- tmp1 * njac[2][4][i][j][k+1];

				lhs[CC][3][0][i][j][k] =  tmp2 * fjac[3][0][i][j][k+1]
					- tmp1 * njac[3][0][i][j][k+1];
				lhs[CC][3][1][i][j][k] =  tmp2 * fjac[3][1][i][j][k+1]
					- tmp1 * njac[3][1][i][j][k+1];
				lhs[CC][3][2][i][j][k] =  tmp2 * fjac[3][2][i][j][k+1]
					- tmp1 * njac[3][2][i][j][k+1];
				lhs[CC][3][3][i][j][k] =  tmp2 * fjac[3][3][i][j][k+1]
					- tmp1 * njac[3][3][i][j][k+1]
					- tmp1 * dz4;
				lhs[CC][3][4][i][j][k] =  tmp2 * fjac[3][4][i][j][k+1]
					- tmp1 * njac[3][4][i][j][k+1];

				lhs[CC][4][0][i][j][k] =  tmp2 * fjac[4][0][i][j][k+1]
					- tmp1 * njac[4][0][i][j][k+1];
				lhs[CC][4][1][i][j][k] =  tmp2 * fjac[4][1][i][j][k+1]
					- tmp1 * njac[4][1][i][j][k+1];
				lhs[CC][4][2][i][j][k] =  tmp2 * fjac[4][2][i][j][k+1]
					- tmp1 * njac[4][2][i][j][k+1];
				lhs[CC][4][3][i][j][k] =  tmp2 * fjac[4][3][i][j][k+1]
					- tmp1 * njac[4][3][i][j][k+1];
				lhs[CC][4][4][i][j][k] =  tmp2 * fjac[4][4][i][j][k+1]
					- tmp1 * njac[4][4][i][j][k+1]
					- tmp1 * dz5;

			}
		}
	}
	//trace_start("lhsz", 2);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void compute_rhs(void) {

	int i, j, k, m;
	double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;

	/*--------------------------------------------------------------------
	  c     compute the reciprocal of density, and the kinetic energy, 
	  c     and the speed of sound.
	  c-------------------------------------------------------------------*/
	//trace_start("compute_rhs", 1);  
#pragma omp parallel for  private(rho_inv) schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				rho_inv = 1.0/u[0][i][j][k];
				rho_i[i][j][k] = rho_inv;
				us[i][j][k] = u[1][i][j][k] * rho_inv;
				vs[i][j][k] = u[2][i][j][k] * rho_inv;
				ws[i][j][k] = u[3][i][j][k] * rho_inv;
				square[i][j][k] = 0.5 * (u[1][i][j][k]*u[1][i][j][k] + 
						u[2][i][j][k]*u[2][i][j][k] +
						u[3][i][j][k]*u[3][i][j][k] ) * rho_inv;
				qs[i][j][k] = square[i][j][k] * rho_inv;
			}
		}
	}
	//trace_stop("compute_rhs", 1);  

	/*--------------------------------------------------------------------
	  c copy the exact forcing term to the right hand side;  because 
	  c this forcing term is known, we can store it on the whole grid
	  c including the boundary                   
	  c-------------------------------------------------------------------*/

	//trace_start("compute_rhs", 2);  
#pragma omp parallel for schedule(static)
	for (i = 0; i < grid_points[0]; i++) {
		for (j = 0; j < grid_points[1]; j++) {
			for (k = 0; k < grid_points[2]; k++) {
				for (m = 0; m < 5; m++) {
					rhs[m][i][j][k] = forcing[m][i][j][k];
				}
			}
		}
	}
	//trace_stop("compute_rhs", 2);  

	/*--------------------------------------------------------------------
	  c     compute xi-direction fluxes 
	  c-------------------------------------------------------------------*/
	//trace_start("compute_rhs", 3);  
#pragma omp parallel for private(uijk, up1, um1) schedule(static,1)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				uijk = us[i][j][k];
				up1  = us[i+1][j][k];
				um1  = us[i-1][j][k];

				rhs[0][i][j][k] = rhs[0][i][j][k] + dx1tx1 * 
					(u[0][i+1][j][k] - 2.0*u[0][i][j][k] + 
					 u[0][i-1][j][k]) -
					tx2 * (u[1][i+1][j][k] - u[1][i-1][j][k]);

				rhs[1][i][j][k] = rhs[1][i][j][k] + dx2tx1 * 
					(u[1][i+1][j][k] - 2.0*u[1][i][j][k] + 
					 u[1][i-1][j][k]) +
					xxcon2*con43 * (up1 - 2.0*uijk + um1) -
					tx2 * (u[1][i+1][j][k]*up1 - 
							u[1][i-1][j][k]*um1 +
							(u[4][i+1][j][k]- square[i+1][j][k]-
							 u[4][i-1][j][k]+ square[i-1][j][k])*
							c2);

				rhs[2][i][j][k] = rhs[2][i][j][k] + dx3tx1 * 
					(u[2][i+1][j][k] - 2.0*u[2][i][j][k] +
					 u[2][i-1][j][k]) +
					xxcon2 * (vs[i+1][j][k] - 2.0*vs[i][j][k] +
							vs[i-1][j][k]) -
					tx2 * (u[2][i+1][j][k]*up1 - 
							u[2][i-1][j][k]*um1);

				rhs[3][i][j][k] = rhs[3][i][j][k] + dx4tx1 * 
					(u[3][i+1][j][k] - 2.0*u[3][i][j][k] +
					 u[3][i-1][j][k]) +
					xxcon2 * (ws[i+1][j][k] - 2.0*ws[i][j][k] +
							ws[i-1][j][k]) -
					tx2 * (u[3][i+1][j][k]*up1 - 
							u[3][i-1][j][k]*um1);

				rhs[4][i][j][k] = rhs[4][i][j][k] + dx5tx1 * 
					(u[4][i+1][j][k] - 2.0*u[4][i][j][k] +
					 u[4][i-1][j][k]) +
					xxcon3 * (qs[i+1][j][k] - 2.0*qs[i][j][k] +
							qs[i-1][j][k]) +
					xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
							um1*um1) +
					xxcon5 * (u[4][i+1][j][k]*rho_i[i+1][j][k] - 
							2.0*u[4][i][j][k]*rho_i[i][j][k] +
							u[4][i-1][j][k]*rho_i[i-1][j][k]) -
					tx2 * ( (c1*u[4][i+1][j][k] - 
								c2*square[i+1][j][k])*up1 -
							(c1*u[4][i-1][j][k] - 
							 c2*square[i-1][j][k])*um1 );
			}
		}
	}
	//trace_stop("compute_rhs", 3);  

	/*--------------------------------------------------------------------
	  c     add fourth order xi-direction dissipation               
	  c-------------------------------------------------------------------*/
	i = 1;
	//trace_start("compute_rhs", 4);  
#pragma omp parallel for schedule(static,1)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k]- dssp * 
					( 5.0*u[m][i][j][k] - 4.0*u[m][i+1][j][k] +
					  u[m][i+2][j][k]);
			}
		}
	}
	//trace_stop("compute_rhs", 4);  

	i = 2;
	//trace_start("compute_rhs", 5);  
#pragma omp parallel for schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
					(-4.0*u[m][i-1][j][k] + 6.0*u[m][i][j][k] -
					 4.0*u[m][i+1][j][k] + u[m][i+2][j][k]);
			}
		}
	}
	//trace_start("compute_rhs", 5);  

	//trace_start("compute_rhs", 6);  
#pragma omp parallel for schedule(static)
	for (i = 3; i < grid_points[0]-3; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < 5; m++) {
					rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
						(  u[m][i-2][j][k] - 4.0*u[m][i-1][j][k] + 
						   6.0*u[m][i][j][k] - 4.0*u[m][i+1][j][k] + 
						   u[m][i+2][j][k] );
				}
			}
		}
	}
	//trace_stop("compute_rhs", 6);  

	i = grid_points[0]-3;
	//trace_start("compute_rhs", 7);  
#pragma omp parallel for schedule(static) 
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i-2][j][k] - 4.0*u[m][i-1][j][k] + 
					  6.0*u[m][i][j][k] - 4.0*u[m][i+1][j][k] );
			}
		}
	}
	//trace_stop("compute_rhs", 7);  

	i = grid_points[0]-2;
	//trace_start("compute_rhs", 8);  
#pragma omp parallel for schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i-2][j][k] - 4.*u[m][i-1][j][k] +
					  5.0*u[m][i][j][k] );
			}
		}
	}
	//trace_stop("compute_rhs", 8);  

	/*--------------------------------------------------------------------
	  c     compute eta-direction fluxes 
	  c-------------------------------------------------------------------*/
	//trace_start("compute_rhs", 9);  
#pragma omp parallel for private(vijk, vp1, vm1) schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				vijk = vs[i][j][k];
				vp1  = vs[i][j+1][k];
				vm1  = vs[i][j-1][k];
				rhs[0][i][j][k] = rhs[0][i][j][k] + dy1ty1 * 
					(u[0][i][j+1][k] - 2.0*u[0][i][j][k] + 
					 u[0][i][j-1][k]) -
					ty2 * (u[2][i][j+1][k] - u[2][i][j-1][k]);
				rhs[1][i][j][k] = rhs[1][i][j][k] + dy2ty1 * 
					(u[1][i][j+1][k] - 2.0*u[1][i][j][k] + 
					 u[1][i][j-1][k]) +
					yycon2 * (us[i][j+1][k] - 2.0*us[i][j][k] + 
							us[i][j-1][k]) -
					ty2 * (u[1][i][j+1][k]*vp1 - 
							u[1][i][j-1][k]*vm1);
				rhs[2][i][j][k] = rhs[2][i][j][k] + dy3ty1 * 
					(u[2][i][j+1][k] - 2.0*u[2][i][j][k] + 
					 u[2][i][j-1][k]) +
					yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
					ty2 * (u[2][i][j+1][k]*vp1 - 
							u[2][i][j-1][k]*vm1 +
							(u[4][i][j+1][k] - square[i][j+1][k] - 
							 u[4][i][j-1][k] + square[i][j-1][k])
							*c2);
				rhs[3][i][j][k] = rhs[3][i][j][k] + dy4ty1 * 
					(u[3][i][j+1][k] - 2.0*u[3][i][j][k] + 
					 u[3][i][j-1][k]) +
					yycon2 * (ws[i][j+1][k] - 2.0*ws[i][j][k] + 
							ws[i][j-1][k]) -
					ty2 * (u[3][i][j+1][k]*vp1 - 
							u[3][i][j-1][k]*vm1);
				rhs[4][i][j][k] = rhs[4][i][j][k] + dy5ty1 * 
					(u[4][i][j+1][k] - 2.0*u[4][i][j][k] + 
					 u[4][i][j-1][k]) +
					yycon3 * (qs[i][j+1][k] - 2.0*qs[i][j][k] + 
							qs[i][j-1][k]) +
					yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
							vm1*vm1) +
					yycon5 * (u[4][i][j+1][k]*rho_i[i][j+1][k] - 
							2.0*u[4][i][j][k]*rho_i[i][j][k] +
							u[4][i][j-1][k]*rho_i[i][j-1][k]) -
					ty2 * ((c1*u[4][i][j+1][k] - 
								c2*square[i][j+1][k]) * vp1 -
							(c1*u[4][i][j-1][k] - 
							 c2*square[i][j-1][k]) * vm1);
			}
		}
	}
	//trace_stop("compute_rhs", 9);  

	/*--------------------------------------------------------------------
	  c     add fourth order eta-direction dissipation         
	  c-------------------------------------------------------------------*/
	j = 1;
	//trace_start("compute_rhs", 10);  
#pragma omp parallel for schedule(static,1)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k]- dssp * 
					( 5.0*u[m][i][j][k] - 4.0*u[m][i][j+1][k] +
					  u[m][i][j+2][k]);
			}
		}
	}
	//trace_stop("compute_rhs", 10);  

	j = 2;
	//trace_start("compute_rhs", 11);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
					(-4.0*u[m][i][j-1][k] + 6.0*u[m][i][j][k] -
					 4.0*u[m][i][j+1][k] + u[m][i][j+2][k]);
			}
		}
	}
	//trace_stop("compute_rhs", 11);  

	//trace_start("compute_rhs", 12);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 3; j < grid_points[1]-3; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < 5; m++) {
					rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
						(  u[m][i][j-2][k] - 4.0*u[m][i][j-1][k] + 
						   6.0*u[m][i][j][k] - 4.0*u[m][i][j+1][k] + 
						   u[m][i][j+2][k] );
				}
			}
		}
	}
	//trace_stop("compute_rhs", 12);  

	j = grid_points[1]-3;
	//trace_start("compute_rhs", 13);  
#pragma omp parallel for schedule(static,1)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i][j-2][k] - 4.0*u[m][i][j-1][k] + 
					  6.0*u[m][i][j][k] - 4.0*u[m][i][j+1][k] );
			}
		}
	}
	//trace_stop("compute_rhs", 13);  

	j = grid_points[1]-2;
	//trace_start("compute_rhs", 14);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i][j-2][k] - 4.*u[m][i][j-1][k] +
					  5.*u[m][i][j][k] );
			}
		}
	}
	//trace_stop("compute_rhs", 14);  

	/*--------------------------------------------------------------------
	  c     compute zeta-direction fluxes 
	  c--t-----------------------------------------------------------------*/
	//trace_start("compute_rhs", 15);  
#pragma omp parallel for private(wijk, wp1, wm1) schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				wijk = ws[i][j][k];
				wp1  = ws[i][j][k+1];
				wm1  = ws[i][j][k-1];

				rhs[0][i][j][k] = rhs[0][i][j][k] + dz1tz1 * 
					(u[0][i][j][k+1] - 2.0*u[0][i][j][k] + 
					 u[0][i][j][k-1]) -
					tz2 * (u[3][i][j][k+1] - u[3][i][j][k-1]);
				rhs[1][i][j][k] = rhs[1][i][j][k] + dz2tz1 * 
					(u[1][i][j][k+1] - 2.0*u[1][i][j][k] + 
					 u[1][i][j][k-1]) +
					zzcon2 * (us[i][j][k+1] - 2.0*us[i][j][k] + 
							us[i][j][k-1]) -
					tz2 * (u[1][i][j][k+1]*wp1 - 
							u[1][i][j][k-1]*wm1);
				rhs[2][i][j][k] = rhs[2][i][j][k] + dz3tz1 * 
					(u[2][i][j][k+1] - 2.0*u[2][i][j][k] + 
					 u[2][i][j][k-1]) +
					zzcon2 * (vs[i][j][k+1] - 2.0*vs[i][j][k] + 
							vs[i][j][k-1]) -
					tz2 * (u[2][i][j][k+1]*wp1 - 
							u[2][i][j][k-1]*wm1);
				rhs[3][i][j][k] = rhs[3][i][j][k] + dz4tz1 * 
					(u[3][i][j][k+1] - 2.0*u[3][i][j][k] + 
					 u[3][i][j][k-1]) +
					zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
					tz2 * (u[3][i][j][k+1]*wp1 - 
							u[3][i][j][k-1]*wm1 +
							(u[4][i][j][k+1] - square[i][j][k+1] - 
							 u[4][i][j][k-1] + square[i][j][k-1])
							*c2);
				rhs[4][i][j][k] = rhs[4][i][j][k] + dz5tz1 * 
					(u[4][i][j][k+1] - 2.0*u[4][i][j][k] + 
					 u[4][i][j][k-1]) +
					zzcon3 * (qs[i][j][k+1] - 2.0*qs[i][j][k] + 
							qs[i][j][k-1]) +
					zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
							wm1*wm1) +
					zzcon5 * (u[4][i][j][k+1]*rho_i[i][j][k+1] - 
							2.0*u[4][i][j][k]*rho_i[i][j][k] +
							u[4][i][j][k-1]*rho_i[i][j][k-1]) -
					tz2 * ( (c1*u[4][i][j][k+1] - 
								c2*square[i][j][k+1])*wp1 -
							(c1*u[4][i][j][k-1] - 
							 c2*square[i][j][k-1])*wm1);
			}
		}
	}
	//trace_stop("compute_rhs", 15);  

	/*--------------------------------------------------------------------
	  c     add fourth order zeta-direction dissipation                
	  c-------------------------------------------------------------------*/
	k = 1;
	//trace_start("compute_rhs", 16);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k]- dssp * 
					( 5.0*u[m][i][j][k] - 4.0*u[m][i][j][k+1] +
					  u[m][i][j][k+2]);
			}
		}
	}
	//trace_stop("compute_rhs", 16);  

	k = 2;
	//trace_start("compute_rhs", 17);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
					(-4.0*u[m][i][j][k-1] + 6.0*u[m][i][j][k] -
					 4.0*u[m][i][j][k+1] + u[m][i][j][k+2]);
			}
		}
	}
	//trace_stop("compute_rhs", 17);  

	//trace_start("compute_rhs", 18);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 3; k < grid_points[2]-3; k++) {
				for (m = 0; m < 5; m++) {
					rhs[m][i][j][k] = rhs[m][i][j][k] - dssp * 
						(  u[m][i][j][k-2] - 4.0*u[m][i][j][k-1] + 
						   6.0*u[m][i][j][k] - 4.0*u[m][i][j][k+1] + 
						   u[m][i][j][k+2] );
				}
			}
		}
	}
	//trace_stop("compute_rhs", 18);  

	k = grid_points[2]-3;
	//trace_start("compute_rhs", 19);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i][j][k-2] - 4.0*u[m][i][j][k-1] + 
					  6.0*u[m][i][j][k] - 4.0*u[m][i][j][k+1] );
			}
		}
	}
	//trace_stop("compute_rhs", 19);  

	k = grid_points[2]-2;
	//trace_start("compute_rhs", 20);  
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (m = 0; m < 5; m++) {
				rhs[m][i][j][k] = rhs[m][i][j][k] - dssp *
					( u[m][i][j][k-2] - 4.0*u[m][i][j][k-1] +
					  5.0*u[m][i][j][k] );
			}
		}
	}
	//trace_stop("compute_rhs", 20);  

	//trace_start("compute_rhs", 21);  
#pragma omp parallel for schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {
			for (m = 0; m < 5; m++) {
				for (i = 1; i < grid_points[0]-1; i++) {
					rhs[m][i][j][k] = rhs[m][i][j][k] * dt;
				}
			}
		}
	}
	//trace_stop("compute_rhs", 21);  
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void set_constants(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	ce[0][0]  = 2.0;
	ce[0][1]  = 0.0;
	ce[0][2]  = 0.0;
	ce[0][3]  = 4.0;
	ce[0][4]  = 5.0;
	ce[0][5]  = 3.0;
	ce[0][6]  = 0.5;
	ce[0][7]  = 0.02;
	ce[0][8]  = 0.01;
	ce[0][9]  = 0.03;
	ce[0][10] = 0.5;
	ce[0][11] = 0.4;
	ce[0][12] = 0.3;

	ce[1][0]  = 1.0;
	ce[1][1]  = 0.0;
	ce[1][2]  = 0.0;
	ce[1][3]  = 0.0;
	ce[1][4]  = 1.0;
	ce[1][5]  = 2.0;
	ce[1][6]  = 3.0;
	ce[1][7]  = 0.01;
	ce[1][8]  = 0.03;
	ce[1][9]  = 0.02;
	ce[1][10] = 0.4;
	ce[1][11] = 0.3;
	ce[1][12] = 0.5;

	ce[2][0]  = 2.0;
	ce[2][1]  = 2.0;
	ce[2][2]  = 0.0;
	ce[2][3]  = 0.0;
	ce[2][4]  = 0.0;
	ce[2][5]  = 2.0;
	ce[2][6]  = 3.0;
	ce[2][7]  = 0.04;
	ce[2][8]  = 0.03;
	ce[2][9]  = 0.05;
	ce[2][10] = 0.3;
	ce[2][11] = 0.5;
	ce[2][12] = 0.4;

	ce[3][0]  = 2.0;
	ce[3][1]  = 2.0;
	ce[3][2]  = 0.0;
	ce[3][3]  = 0.0;
	ce[3][4]  = 0.0;
	ce[3][5]  = 2.0;
	ce[3][6]  = 3.0;
	ce[3][7]  = 0.03;
	ce[3][8]  = 0.05;
	ce[3][9]  = 0.04;
	ce[3][10] = 0.2;
	ce[3][11] = 0.1;
	ce[3][12] = 0.3;

	ce[4][0]  = 5.0;
	ce[4][1]  = 4.0;
	ce[4][2]  = 3.0;
	ce[4][3]  = 2.0;
	ce[4][4]  = 0.1;
	ce[4][5]  = 0.4;
	ce[4][6]  = 0.3;
	ce[4][7]  = 0.05;
	ce[4][8]  = 0.04;
	ce[4][9]  = 0.03;
	ce[4][10] = 0.1;
	ce[4][11] = 0.3;
	ce[4][12] = 0.2;

	c1 = 1.4;
	c2 = 0.4;
	c3 = 0.1;
	c4 = 1.0;
	c5 = 1.4;

	dnxm1 = 1.0 / (double)(grid_points[0]-1);
	dnym1 = 1.0 / (double)(grid_points[1]-1);
	dnzm1 = 1.0 / (double)(grid_points[2]-1);

	c1c2 = c1 * c2;
	c1c5 = c1 * c5;
	c3c4 = c3 * c4;
	c1345 = c1c5 * c3c4;

	conz1 = (1.0-c1c5);

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

	dxmax = max(dx3, dx4);
	dymax = max(dy2, dy4);
	dzmax = max(dz2, dz3);

	dssp = 0.25 * max(dx1, max(dy1, dz1) );

	c4dssp = 4.0 * dssp;
	c5dssp = 5.0 * dssp;

	dttx1 = dt*tx1;
	dttx2 = dt*tx2;
	dtty1 = dt*ty1;
	dtty2 = dt*ty2;
	dttz1 = dt*tz1;
	dttz2 = dt*tz2;

	c2dttx1 = 2.0*dttx1;
	c2dtty1 = 2.0*dtty1;
	c2dttz1 = 2.0*dttz1;

	dtdssp = dt*dssp;

	comz1  = dtdssp;
	comz4  = 4.0*dtdssp;
	comz5  = 5.0*dtdssp;
	comz6  = 6.0*dtdssp;

	c3c4tx3 = c3c4*tx3;
	c3c4ty3 = c3c4*ty3;
	c3c4tz3 = c3c4*tz3;

	dx1tx1 = dx1*tx1;
	dx2tx1 = dx2*tx1;
	dx3tx1 = dx3*tx1;
	dx4tx1 = dx4*tx1;
	dx5tx1 = dx5*tx1;

	dy1ty1 = dy1*ty1;
	dy2ty1 = dy2*ty1;
	dy3ty1 = dy3*ty1;
	dy4ty1 = dy4*ty1;
	dy5ty1 = dy5*ty1;

	dz1tz1 = dz1*tz1;
	dz2tz1 = dz2*tz1;
	dz3tz1 = dz3*tz1;
	dz4tz1 = dz4*tz1;
	dz5tz1 = dz5*tz1;

	c2iv  = 2.5;
	con43 = 4.0/3.0;
	con16 = 1.0/6.0;

	xxcon1 = c3c4tx3*con43*tx3;
	xxcon2 = c3c4tx3*tx3;
	xxcon3 = c3c4tx3*conz1*tx3;
	xxcon4 = c3c4tx3*con16*tx3;
	xxcon5 = c3c4tx3*c1c5*tx3;

	yycon1 = c3c4ty3*con43*ty3;
	yycon2 = c3c4ty3*ty3;
	yycon3 = c3c4ty3*conz1*ty3;
	yycon4 = c3c4ty3*con16*ty3;
	yycon5 = c3c4ty3*c1c5*ty3;

	zzcon1 = c3c4tz3*con43*tz3;
	zzcon2 = c3c4tz3*tz3;
	zzcon3 = c3c4tz3*conz1*tz3;
	zzcon4 = c3c4tz3*con16*tz3;
	zzcon5 = c3c4tz3*c1c5*tz3;
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void verify(int no_time_steps, char *class, boolean *verified) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c  verification routine                         
	  c-------------------------------------------------------------------*/

	double xcrref[5],xceref[5],xcrdif[5],xcedif[5], 
	       epsilon, xce[5], xcr[5], dtref;
	int m;

	/*--------------------------------------------------------------------
	  c   tolerance level
	  c-------------------------------------------------------------------*/
	epsilon = 1.0e-08;


	/*--------------------------------------------------------------------
	  c   compute the error norm and the residual norm, and exit if not printing
	  c-------------------------------------------------------------------*/
	error_norm(xce);
	compute_rhs();

	sync_ocl_buffers();

	rhs_norm(xcr);

	for (m = 0; m < 5; m++) {
		xcr[m] = xcr[m] / dt;
	}

	*class = 'U';
	*verified = TRUE;

	for (m = 0; m < 5; m++) {
		xcrref[m] = 1.0;
		xceref[m] = 1.0;
	}

	/*--------------------------------------------------------------------
	  c    reference data for 12X12X12 grids after 100 time steps, with DT = 1.0d-02
	  c-------------------------------------------------------------------*/
	if (grid_points[0] == 12 &&
			grid_points[1] == 12 &&
			grid_points[2] == 12 &&
			no_time_steps == 60) {

		*class = 'S';
		dtref = 1.0e-2;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of residual.
		  c-------------------------------------------------------------------*/
		xcrref[0] = 1.7034283709541311e-01;
		xcrref[1] = 1.2975252070034097e-02;
		xcrref[2] = 3.2527926989486055e-02;
		xcrref[3] = 2.6436421275166801e-02;
		xcrref[4] = 1.9211784131744430e-01;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of solution error.
		  c-------------------------------------------------------------------*/
		xceref[0] = 4.9976913345811579e-04;
		xceref[1] = 4.5195666782961927e-05;
		xceref[2] = 7.3973765172921357e-05;
		xceref[3] = 7.3821238632439731e-05;
		xceref[4] = 8.9269630987491446e-04;

		/*--------------------------------------------------------------------
		  c    reference data for 24X24X24 grids after 200 time steps, with DT = 0.8d-3
		  c-------------------------------------------------------------------*/
	} else if (grid_points[0] == 24 &&
			grid_points[1] == 24 &&
			grid_points[2] == 24 &&
			no_time_steps == 200) {

		*class = 'W';
		dtref = 0.8e-3;
		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of residual.
		  c-------------------------------------------------------------------*/
		xcrref[0] = 0.1125590409344e+03;
		xcrref[1] = 0.1180007595731e+02;
		xcrref[2] = 0.2710329767846e+02;
		xcrref[3] = 0.2469174937669e+02;
		xcrref[4] = 0.2638427874317e+03;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of solution error.
		  c-------------------------------------------------------------------*/
		xceref[0] = 0.4419655736008e+01;
		xceref[1] = 0.4638531260002e+00;
		xceref[2] = 0.1011551749967e+01;
		xceref[3] = 0.9235878729944e+00;
		xceref[4] = 0.1018045837718e+02;


		/*--------------------------------------------------------------------
		  c    reference data for 64X64X64 grids after 200 time steps, with DT = 0.8d-3
		  c-------------------------------------------------------------------*/
	} else if (grid_points[0] == 64 &&
			grid_points[1] == 64 &&
			grid_points[2] == 64 &&
			no_time_steps == 200) {

		*class = 'A';
		dtref = 0.8e-3;
		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of residual.
		  c-------------------------------------------------------------------*/
		xcrref[0] = 1.0806346714637264e+02;
		xcrref[1] = 1.1319730901220813e+01;
		xcrref[2] = 2.5974354511582465e+01;
		xcrref[3] = 2.3665622544678910e+01;
		xcrref[4] = 2.5278963211748344e+02;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of solution error.
		  c-------------------------------------------------------------------*/
		xceref[0] = 4.2348416040525025e+00;
		xceref[1] = 4.4390282496995698e-01;
		xceref[2] = 9.6692480136345650e-01;
		xceref[3] = 8.8302063039765474e-01;
		xceref[4] = 9.7379901770829278e+00;

		/*--------------------------------------------------------------------
		  c    reference data for 102X102X102 grids after 200 time steps,
		  c    with DT = 3.0d-04
		  c-------------------------------------------------------------------*/
	} else if (grid_points[0] == 102 &&
			grid_points[1] == 102 &&
			grid_points[2] == 102 &&
			no_time_steps == 200) {

		*class = 'B';
		dtref = 3.0e-4;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of residual.
		  c-------------------------------------------------------------------*/
		xcrref[0] = 1.4233597229287254e+03;
		xcrref[1] = 9.9330522590150238e+01;
		xcrref[2] = 3.5646025644535285e+02;
		xcrref[3] = 3.2485447959084092e+02;
		xcrref[4] = 3.2707541254659363e+03;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of solution error.
		  c-------------------------------------------------------------------*/
		xceref[0] = 5.2969847140936856e+01;
		xceref[1] = 4.4632896115670668e+00;
		xceref[2] = 1.3122573342210174e+01;
		xceref[3] = 1.2006925323559144e+01;
		xceref[4] = 1.2459576151035986e+02;

		/*--------------------------------------------------------------------
		  c    reference data for 162X162X162 grids after 200 time steps,
		  c    with DT = 1.0d-04
		  c-------------------------------------------------------------------*/
	} else if (grid_points[0] == 162 &&
			grid_points[1] == 162 &&
			grid_points[2] == 162 &&
			no_time_steps == 200) {

		*class = 'C';
		dtref = 1.0e-4;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of residual.
		  c-------------------------------------------------------------------*/
		xcrref[0] = 0.62398116551764615e+04;
		xcrref[1] = 0.50793239190423964e+03;
		xcrref[2] = 0.15423530093013596e+04;
		xcrref[3] = 0.13302387929291190e+04;
		xcrref[4] = 0.11604087428436455e+05;

		/*--------------------------------------------------------------------
		  c  Reference values of RMS-norms of solution error.
		  c-------------------------------------------------------------------*/
		xceref[0] = 0.16462008369091265e+03;
		xceref[1] = 0.11497107903824313e+02;
		xceref[2] = 0.41207446207461508e+02;
		xceref[3] = 0.37087651059694167e+02;
		xceref[4] = 0.36211053051841265e+03;

	} else {
		*verified = FALSE;
	}

	/*--------------------------------------------------------------------
	  c    verification test for residuals if gridsize is either 12X12X12 or 
	  c    64X64X64 or 102X102X102 or 162X162X162
	  c-------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c    Compute the difference of solution values and the known reference values.
	  c-------------------------------------------------------------------*/
	for (m = 0; m < 5; m++) {

		xcrdif[m] = fabs((xcr[m]-xcrref[m])/xcrref[m]);
		xcedif[m] = fabs((xce[m]-xceref[m])/xceref[m]);

	}

	/*--------------------------------------------------------------------
	  c    Output the comparison of computed results to known cases.
	  c-------------------------------------------------------------------*/

	if (*class != 'U') {
		printf(" Verification being performed for class %1c\n", *class);
		printf(" accuracy setting for epsilon = %20.13e\n", epsilon);
		if (fabs(dt-dtref) > epsilon) {
			*verified = FALSE;
			*class = 'U';
			printf(" DT does not match the reference value of %15.8e\n", dtref);
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
		} else if (xcrdif[m] > epsilon) {
			*verified = FALSE;
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
		} else if (xcedif[m] > epsilon) {
			*verified = FALSE;
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
	} else if (*verified == TRUE) {
		printf(" Verification Successful\n");
	} else {
		printf(" Verification failed\n");
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void x_solve(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     
	  c     Performs line solves in X direction by first factoring
	  c     the block-tridiagonal matrix into an upper triangular matrix, 
	  c     and then performing back substitution to solve for the unknow
	  c     vectors of each line.  
	  c     
	  c     Make sure we treat elements zero to cell_size in the direction
	  c     of the sweep.
	  c     
	  c-------------------------------------------------------------------*/

	lhsx();
	x_solve_cell();
	x_backsubstitute();
}


/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void x_backsubstitute(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     back solve: if last cell, then generate U(isize)=rhs[isize)
	  c     else assume U(isize) is loaded in un pack backsub_info
	  c     so just use it
	  c     after call u(istart) will be sent to next cell
	  c-------------------------------------------------------------------*/

	int i, j, k, m, n;

	for (i = grid_points[0]-2; i >= 0; i--) {
#pragma omp parallel for schedule(static) parallel_depth(3)
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < BLOCK_SIZE; m++) {
					for (n = 0; n < BLOCK_SIZE; n++) {
						rhs[m][i][j][k] = rhs[m][i][j][k]
							- lhs[CC][m][n][i][j][k]*rhs[n][i+1][j][k];
					}
				}
			}
		}
		//trace_stop("x_backsubstitute", 1);
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void x_solve_cell(void) {

	/*--------------------------------------------------------------------
	  c     performs guaussian elimination on this cell.
	  c     
	  c     assumes that unpacking routines for non-first cells 
	  c     preload C' and rhs' from previous cell.
	  c     
	  c     assumed send happens outside this routine, but that
	  c     c'(IMAX) and rhs'(IMAX) will be sent to next cell
	  c-------------------------------------------------------------------*/

	int i,j,k,isize;

	isize = grid_points[0]-1;

	/*--------------------------------------------------------------------
	  c     outer most do loops - sweeping in i direction
	  c-------------------------------------------------------------------*/
	//trace_start("x_solve_cell", 1);
#pragma omp parallel for schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {

			/*--------------------------------------------------------------------
			  c     multiply c(0,j,k) by b_inverse and copy back to c
			  c     multiply rhs(0) by b_inverse(0) and copy to rhs
			  c-------------------------------------------------------------------*/
			binvcrhs( lhs,0,j,k,BB,
					lhs,0,j,k,CC,
					rhs,0,j,k );
		}
	}
	//trace_stop("x_solve_cell", 1);

	/*--------------------------------------------------------------------
	  c     begin inner most do loop
	  c     do all the elements of the cell unless last 
	  c-------------------------------------------------------------------*/
	for (i = 1; i < isize; i++) {
		//trace_start("x_solve_cell", 2);
#pragma omp parallel for schedule(static)
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = 1; k < grid_points[2]-1; k++) {

				/*--------------------------------------------------------------------
				  c     rhs(i) = rhs(i) - A*rhs(i-1)
				  c-------------------------------------------------------------------*/
				matvec_sub(lhs,i,j,k,AA,
						rhs,i-1,j,k, rhs,i,j,k);

				/*--------------------------------------------------------------------
				  c     B(i) = B(i) - C(i-1)*A(i)
				  c-------------------------------------------------------------------*/
				matmul_sub(lhs,i,j,k,AA,
						lhs,i-1,j,k,CC,
						lhs,i,j,k,BB);


				/*--------------------------------------------------------------------
				  c     multiply c(i,j,k) by b_inverse and copy back to c
				  c     multiply rhs(1,j,k) by b_inverse(1,j,k) and copy to rhs
				  c-------------------------------------------------------------------*/
				binvcrhs( lhs,i,j,k,BB,
						lhs,i,j,k,CC,
						rhs,i,j,k );

			}
		}
		//trace_stop("x_solve_cell", 2);
	}

	//trace_start("x_solve_cell", 3);
#pragma omp parallel for schedule(static)
	for (j = 1; j < grid_points[1]-1; j++) {
		for (k = 1; k < grid_points[2]-1; k++) {

			/*--------------------------------------------------------------------
			  c     rhs(isize) = rhs(isize) - A*rhs(isize-1)
			  c-------------------------------------------------------------------*/
			matvec_sub(lhs,isize,j,k,AA,
					rhs,isize-1,j,k, rhs,isize,j,k);

			/*--------------------------------------------------------------------
			  c     B(isize) = B(isize) - C(isize-1)*A(isize)
			  c-------------------------------------------------------------------*/
			matmul_sub(lhs,isize,j,k,AA,
					lhs,isize-1,j,k,CC,
					lhs,isize,j,k,BB);

			/*--------------------------------------------------------------------
			  c     multiply rhs() by b_inverse() and copy to rhs
			  c-------------------------------------------------------------------*/
			binvrhs( lhs,i,j,k,BB,
					rhs,i,j,k );

		}
	}
	//trace_stop("x_solve_cell", 3);
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void matvec_sub(double ablock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int ablock_0, int ablock_1, int ablock_2, int ablock_3, double avec[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int avec_0, int avec_1, int avec_2, double bvec[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int bvec_0, int bvec_1, int bvec_2) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     subtracts bvec=bvec - ablock*avec
	  c-------------------------------------------------------------------*/

	int i;

	for (i = 0; i < 5; i++) {
		/*--------------------------------------------------------------------
		  c            rhs(i,ic,jc,kc,ccell) = rhs(i,ic,jc,kc,ccell) 
		  c     $           - lhs[i,1,ablock,ia,ja,ka,acell)*
		  c-------------------------------------------------------------------*/
		bvec[i][bvec_0][bvec_1][bvec_2] = bvec[i][bvec_0][bvec_1][bvec_2] - ablock[ablock_3][i][0][ablock_0][ablock_1][ablock_2]*avec[0][avec_0][avec_1][avec_2]
			- ablock[ablock_3][i][1][ablock_0][ablock_1][ablock_2]*avec[1][avec_0][avec_1][avec_2]
			- ablock[ablock_3][i][2][ablock_0][ablock_1][ablock_2]*avec[2][avec_0][avec_1][avec_2]
			- ablock[ablock_3][i][3][ablock_0][ablock_1][ablock_2]*avec[3][avec_0][avec_1][avec_2]
			- ablock[ablock_3][i][4][ablock_0][ablock_1][ablock_2]*avec[4][avec_0][avec_1][avec_2];
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void matmul_sub(double ablock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int ablock_0, int ablock_1, int ablock_2, int ablock_3,
		double bblock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int bblock_0, int bblock_1, int bblock_2, int bblock_3,
		double cblock[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int cblock_0, int cblock_1, int cblock_2, int cblock_3) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
	  c-------------------------------------------------------------------*/

	int j;

	for (j = 0; j < 5; j++) {
		cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] = cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] - ablock[ablock_3][0][0][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][0][1][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][0][2][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][0][3][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][0][4][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] = cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] - ablock[ablock_3][1][0][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][1][1][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][1][2][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][1][3][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][1][4][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] = cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] - ablock[ablock_3][2][0][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][2][1][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][2][2][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][2][3][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][2][4][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] = cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] - ablock[ablock_3][3][0][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][3][1][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][3][2][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][3][3][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][3][4][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] = cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] - ablock[ablock_3][4][0][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][4][1][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][4][2][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][4][3][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2]
			- ablock[ablock_3][4][4][ablock_0][ablock_1][ablock_2]*bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void binvcrhs(double lhs[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int lhs_0, int lhs_1, int lhs_2, int lhs_3, double c[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int c_0, int c_1, int c_2, int c_3, double r[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int r_0, int r_1, int r_2) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	double pivot, coeff;

	/*--------------------------------------------------------------------
	  c     
	  c-------------------------------------------------------------------*/

	pivot = 1.00/lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]*pivot;
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2]*pivot;
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2]*pivot;
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2]*pivot;
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2]*pivot;
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2]*pivot;
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] - coeff*c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] - coeff*c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] - coeff*c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] - coeff*c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] - coeff*c[c_3][0][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] - coeff*c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] - coeff*c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] - coeff*c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] - coeff*c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] - coeff*c[c_3][0][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] - coeff*c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] - coeff*c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] - coeff*c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] - coeff*c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] - coeff*c[c_3][0][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] - coeff*c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] - coeff*c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] - coeff*c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] - coeff*c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] - coeff*c[c_3][0][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]*pivot;
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2]*pivot;
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2]*pivot;
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2]*pivot;
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2]*pivot;
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2]*pivot;
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] - coeff*c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] - coeff*c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] - coeff*c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] - coeff*c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] - coeff*c[c_3][1][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] - coeff*c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] - coeff*c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] - coeff*c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] - coeff*c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] - coeff*c[c_3][1][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] - coeff*c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] - coeff*c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] - coeff*c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] - coeff*c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] - coeff*c[c_3][1][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] - coeff*c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] - coeff*c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] - coeff*c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] - coeff*c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] - coeff*c[c_3][1][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]*pivot;
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2]*pivot;
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2]*pivot;
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2]*pivot;
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2]*pivot;
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2]*pivot;
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] - coeff*c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] - coeff*c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] - coeff*c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] - coeff*c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] - coeff*c[c_3][2][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] - coeff*c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] - coeff*c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] - coeff*c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] - coeff*c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] - coeff*c[c_3][2][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] - coeff*c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] - coeff*c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] - coeff*c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] - coeff*c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] - coeff*c[c_3][2][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] - coeff*c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] - coeff*c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] - coeff*c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] - coeff*c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] - coeff*c[c_3][2][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]*pivot;
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2]*pivot;
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2]*pivot;
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2]*pivot;
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2]*pivot;
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2]*pivot;
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] - coeff*c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] - coeff*c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] - coeff*c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] - coeff*c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] - coeff*c[c_3][3][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] - coeff*c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] - coeff*c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] - coeff*c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] - coeff*c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] - coeff*c[c_3][3][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] - coeff*c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] - coeff*c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] - coeff*c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] - coeff*c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] - coeff*c[c_3][3][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] - coeff*c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] - coeff*c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] - coeff*c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] - coeff*c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] - coeff*c[c_3][3][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2]*pivot;
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2]*pivot;
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2]*pivot;
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2]*pivot;
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2]*pivot;
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] - coeff*c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] - coeff*c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] - coeff*c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] - coeff*c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] - coeff*c[c_3][4][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] - coeff*c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] - coeff*c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] - coeff*c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] - coeff*c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] - coeff*c[c_3][4][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] - coeff*c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] - coeff*c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] - coeff*c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] - coeff*c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] - coeff*c[c_3][4][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] - coeff*c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] - coeff*c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] - coeff*c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] - coeff*c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] - coeff*c[c_3][4][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void binvrhs(double lhs[3][5][5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int lhs_0, int lhs_1, int lhs_2, int lhs_3, double r[5][IMAX/2*2+1][JMAX/2*2+1][KMAX/2*2+1], int r_0, int r_1, int r_2) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	double pivot, coeff;

	/*--------------------------------------------------------------------
	  c     
	  c-------------------------------------------------------------------*/

	pivot = 1.00/lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]*pivot;
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[0][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]*pivot;
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[1][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2]*pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]*pivot;
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[2][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2]*pivot;
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];

	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2]= lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] - coeff*lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]   - coeff*r[3][r_0][r_1][r_2];


	pivot = 1.00/lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2]   = r[4][r_0][r_1][r_2]  *pivot;

	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2]   = r[0][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2]   = r[1][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2]   = r[2][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2]   = r[3][r_0][r_1][r_2]   - coeff*r[4][r_0][r_1][r_2];

}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void y_solve(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     Performs line solves in Y direction by first factoring
	  c     the block-tridiagonal matrix into an upper triangular matrix][ 
	  c     and then performing back substitution to solve for the unknow
	  c     vectors of each line.  
	  c     
	  c     Make sure we treat elements zero to cell_size in the direction
	  c     of the sweep.
	  c-------------------------------------------------------------------*/

	lhsy();
	y_solve_cell();
	y_backsubstitute();
}


/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void y_backsubstitute(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     back solve: if last cell][ then generate U(jsize)=rhs(jsize)
	  c     else assume U(jsize) is loaded in un pack backsub_info
	  c     so just use it
	  c     after call u(jstart) will be sent to next cell
	  c-------------------------------------------------------------------*/

	int i, j, k, m, n;

	for (j = grid_points[1]-2; j >= 0; j--) {
#pragma omp parallel for schedule(static) parallel_depth(3)
		for (i = 1; i < grid_points[0]-1; i++) {
			for (k = 1; k < grid_points[2]-1; k++) {
				for (m = 0; m < BLOCK_SIZE; m++) {
					for (n = 0; n < BLOCK_SIZE; n++) {
						rhs[m][i][j][k] = rhs[m][i][j][k] 
							- lhs[CC][m][n][i][j][k]*rhs[n][i][j+1][k];
					}
				}
			}
		}
		//trace_stop("y_backsubstitute", 1);
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void y_solve_cell(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     performs guaussian elimination on this cell.
	  c     
	  c     assumes that unpacking routines for non-first cells 
	  c     preload C' and rhs' from previous cell.
	  c     
	  c     assumed send happens outside this routine, but that
	  c     c'(JMAX) and rhs'(JMAX) will be sent to next cell
	  c-------------------------------------------------------------------*/

	int i, j, k, jsize;

	jsize = grid_points[1]-1;

	//trace_start("y_solve_cell", 1);
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {

			/*--------------------------------------------------------------------
			  c     multiply c(i,0,k) by b_inverse and copy back to c
			  c     multiply rhs(0) by b_inverse(0) and copy to rhs
			  c-------------------------------------------------------------------*/
			binvcrhs( lhs,i,0,k,BB,
					lhs,i,0,k,CC,
					rhs,i,0,k );
		}
	}
	//trace_stop("y_solve_cell", 1);

	/*--------------------------------------------------------------------
	  c     begin inner most do loop
	  c     do all the elements of the cell unless last 
	  c-------------------------------------------------------------------*/
	for (j = 1; j < jsize; j++) {
		//trace_start("y_solve_cell", 2);
#pragma omp parallel for schedule(static)
		for (i = 1; i < grid_points[0]-1; i++) {
			for (k = 1; k < grid_points[2]-1; k++) {

				/*--------------------------------------------------------------------
				  c     subtract A*lhs_vector(j-1) from lhs_vector(j)
				  c     
				  c     rhs(j) = rhs(j) - A*rhs(j-1)
				  c-------------------------------------------------------------------*/
				matvec_sub(lhs,i,j,k,AA,
						rhs,i,j-1,k, rhs,i,j,k);

				/*--------------------------------------------------------------------
				  c     B(j) = B(j) - C(j-1)*A(j)
				  c-------------------------------------------------------------------*/
				matmul_sub(lhs,i,j,k,AA,
						lhs,i,j-1,k,CC,
						lhs,i,j,k,BB);

				/*--------------------------------------------------------------------
				  c     multiply c(i,j,k) by b_inverse and copy back to c
				  c     multiply rhs(i,1,k) by b_inverse(i,1,k) and copy to rhs
				  c-------------------------------------------------------------------*/
				binvcrhs( lhs,i,j,k,BB,
						lhs,i,j,k,CC,
						rhs,i,j,k );

			}
		}
		//trace_stop("y_solve_cell", 2);
	}

	//trace_start("y_solve_cell", 3);
#pragma omp parallel for schedule(static,1)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (k = 1; k < grid_points[2]-1; k++) {

			/*--------------------------------------------------------------------
			  c     rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
			  c-------------------------------------------------------------------*/
			matvec_sub(lhs,i,jsize,k,AA,
					rhs,i,jsize-1,k, rhs,i,jsize,k);

			/*--------------------------------------------------------------------
			  c     B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
			  c     call matmul_sub(aa,i,jsize,k,c,
			  c     $              cc,i,jsize-1,k,c,BB,i,jsize,k)
			  c-------------------------------------------------------------------*/
			matmul_sub(lhs,i,jsize,k,AA,
					lhs,i,jsize-1,k,CC,
					lhs,i,jsize,k,BB);

			/*--------------------------------------------------------------------
			  c     multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
			  c-------------------------------------------------------------------*/
			binvrhs( lhs,i,jsize,k,BB,
					rhs,i,jsize,k );

		}
	}
	//trace_stop("y_solve_cell", 3);
}


/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void z_solve(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     Performs line solves in Z direction by first factoring
	  c     the block-tridiagonal matrix into an upper triangular matrix, 
	  c     and then performing back substitution to solve for the unknow
	  c     vectors of each line.  
	  c     
	  c     Make sure we treat elements zero to cell_size in the direction
	  c     of the sweep.
	  c-------------------------------------------------------------------*/

	lhsz();
	z_solve_cell();
	z_backsubstitute();
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void z_backsubstitute(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     back solve: if last cell, then generate U(ksize)=rhs(ksize)
	  c     else assume U(ksize) is loaded in un pack backsub_info
	  c     so just use it
	  c     after call u(kstart) will be sent to next cell
	  c-------------------------------------------------------------------*/

	int i, j, k, m, n;

#pragma omp parallel for schedule(static) parallel_depth(2)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {
			for (k = grid_points[2]-2; k >= 0; k--) {
				for (m = 0; m < BLOCK_SIZE; m++) {
					for (n = 0; n < BLOCK_SIZE; n++) {
						rhs[m][i][j][k] = rhs[m][i][j][k] 
							- lhs[CC][m][n][i][j][k]*rhs[n][i][j][k+1];
					}
				}
			}
		}
	}
}

/*--------------------------------------------------------------------
  --------------------------------------------------------------------*/

static void z_solve_cell(void) {

	/*--------------------------------------------------------------------
	  --------------------------------------------------------------------*/

	/*--------------------------------------------------------------------
	  c     performs guaussian elimination on this cell.
	  c     
	  c     assumes that unpacking routines for non-first cells 
	  c     preload C' and rhs' from previous cell.
	  c     
	  c     assumed send happens outside this routine, but that
	  c     c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	  c-------------------------------------------------------------------*/

	int i,j,k,ksize;

	ksize = grid_points[2]-1;

	/*--------------------------------------------------------------------
	  c     outer most do loops - sweeping in i direction
	  c-------------------------------------------------------------------*/
	//trace_start("z_solve_cell", 1);
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {

			/*--------------------------------------------------------------------
			  c     multiply c(i,j,0) by b_inverse and copy back to c
			  c     multiply rhs(0) by b_inverse(0) and copy to rhs
			  c-------------------------------------------------------------------*/
			binvcrhs( lhs,i,j,0,BB,
					lhs,i,j,0,CC,
					rhs,i,j,0 );

		}
	}
	//trace_stop("z_solve_cell", 1);

	/*--------------------------------------------------------------------
	  c     begin inner most do loop
	  c     do all the elements of the cell unless last 
	  c-------------------------------------------------------------------*/
	for (k = 1; k < ksize; k++) {
		//trace_start("z_solve_cell", 2);
#pragma omp parallel for schedule(static)
		for (i = 1; i < grid_points[0]-1; i++) {
			for (j = 1; j < grid_points[1]-1; j++) {

				/*--------------------------------------------------------------------
				  c     subtract A*lhs_vector(k-1) from lhs_vector(k)
				  c     
				  c     rhs(k) = rhs(k) - A*rhs(k-1)
				  c-------------------------------------------------------------------*/
				matvec_sub(lhs,i,j,k,AA,
						rhs,i,j,k-1, rhs,i,j,k);

				/*--------------------------------------------------------------------
				  c     B(k) = B(k) - C(k-1)*A(k)
				  c     call matmul_sub(aa,i,j,k,c,cc,i,j,k-1,c,BB,i,j,k)
				  c-------------------------------------------------------------------*/
				matmul_sub(lhs,i,j,k,AA,
						lhs,i,j,k-1,CC,
						lhs,i,j,k,BB);

				/*--------------------------------------------------------------------
				  c     multiply c(i,j,k) by b_inverse and copy back to c
				  c     multiply rhs(i,j,1) by b_inverse(i,j,1) and copy to rhs
				  c-------------------------------------------------------------------*/
				binvcrhs( lhs,i,j,k,BB,
						lhs,i,j,k,CC,
						rhs,i,j,k );

			}
		}
		//trace_stop("z_solve_cell", 2);
	}

	/*--------------------------------------------------------------------
	  c     Now finish up special cases for last cell
	  c-------------------------------------------------------------------*/
	//trace_start("z_solve_cell", 3);
#pragma omp parallel for schedule(static)
	for (i = 1; i < grid_points[0]-1; i++) {
		for (j = 1; j < grid_points[1]-1; j++) {

			/*--------------------------------------------------------------------
			  c     rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
			  c-------------------------------------------------------------------*/
			matvec_sub(lhs,i,j,ksize,AA,
					rhs,i,j,ksize-1, rhs,i,j,ksize);

			/*--------------------------------------------------------------------
			  c     B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
			  c     call matmul_sub(aa,i,j,ksize,c,
			  c     $              cc,i,j,ksize-1,c,BB,i,j,ksize)
			  c-------------------------------------------------------------------*/
			matmul_sub(lhs,i,j,ksize,AA,
					lhs,i,j,ksize-1,CC,
					lhs,i,j,ksize,BB);

			/*--------------------------------------------------------------------
			  c     multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
			  c-------------------------------------------------------------------*/
			binvrhs( lhs,i,j,ksize,BB,
					rhs,i,j,ksize );

		}
	}
	//trace_stop("z_solve_cell", 3);
}
