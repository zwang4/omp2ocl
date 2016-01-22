//-------------------------------------------------------------------------------
//Host code 
//Generated at : Mon Aug  6 14:07:34 2012
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
#include "globals.h"
#include "ocldef.h"

static int is1;
static int is2;
static int is3;
static int ie1;
static int ie2;
static int ie3;
static double z1[1037];
static double z2[1037];
static double z3[1037];
static double u1[1037];
static double u2[1037];
static double r1[1037];
static double r2[1037];
static double xx1[1037];
static double yy1[1037];
static int zran3_j1[10][2];
static int zran3_j2[10][2];
static int zran3_j3[10][2];
void my_vranlc(int n, double *x_seed, ocl_buffer * __ocl_buffer_x_seed,
	       double a, double *y_data, ocl_buffer * __ocl_buffer_y_data,
	       unsigned int *y_idx, ocl_buffer * __ocl_buffer_y_idx, int i2,
	       int i3)
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
			y_data[y_idx[y_idx[i] + i2] + i3] =
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

static void setup(int *n1, ocl_buffer * __ocl_buffer_n1, int *n2,
		  ocl_buffer * __ocl_buffer_n2, int *n3,
		  ocl_buffer * __ocl_buffer_n3, int lt);
static void mg3P(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		 unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		 double *v_data, ocl_buffer * __ocl_buffer_v_data,
		 unsigned int *v_idx, ocl_buffer * __ocl_buffer_v_idx,
		 double *r_data, ocl_buffer * __ocl_buffer_r_data,
		 unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		 double a[4], ocl_buffer * __ocl_buffer_a, double c[4],
		 ocl_buffer * __ocl_buffer_c, int n1, int n2, int n3, int k);
static void psinv(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, double *u_data,
		  ocl_buffer * __ocl_buffer_u_data, unsigned int *u_idx,
		  ocl_buffer * __ocl_buffer_u_idx, unsigned int u_offset,
		  int n1, int n2, int n3, double c[4],
		  ocl_buffer * __ocl_buffer_c, int k);
static void resid(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		  unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		  unsigned int u_offset, double *v_data,
		  ocl_buffer * __ocl_buffer_v_data, unsigned int *v_idx,
		  ocl_buffer * __ocl_buffer_v_idx, unsigned int v_offset,
		  double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, int n1, int n2, int n3, double a[4],
		  ocl_buffer * __ocl_buffer_a, int k);
static void rprj3(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, int m1k, int m2k, int m3k,
		  double *s_data, ocl_buffer * __ocl_buffer_s_data,
		  unsigned int *s_idx, ocl_buffer * __ocl_buffer_s_idx,
		  unsigned int s_offset, int m1j, int m2j, int m3j, int k);
static void interp(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		   unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		   unsigned int z_offset, int mm1, int mm2, int mm3,
		   double *u_data, ocl_buffer * __ocl_buffer_u_data,
		   unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		   unsigned int u_offset, int n1, int n2, int n3, int k);
static void norm2u3(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		    unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		    unsigned int r_offset, int n1, int n2, int n3, double *rnm2,
		    ocl_buffer * __ocl_buffer_rnm2, double *rnmu,
		    ocl_buffer * __ocl_buffer_rnmu, int nx, int ny, int nz);
static void rep_nrm(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		    unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		    unsigned int u_offset, int n1, int n2, int n3, char *title,
		    ocl_buffer * __ocl_buffer_title, int kk);
static void comm3(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		  unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		  unsigned int u_offset, int n1, int n2, int n3, int kk);
static void zran3(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		  unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx, int n1,
		  int n2, int n3, int nx, int ny, int k);
static void showall(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		    unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		    unsigned int z_offset, int n1, int n2, int n3);
static double power(double a, int n);
static void bubble(double ten[1037][2], ocl_buffer * __ocl_buffer_ten,
		   int j1[1037][2], ocl_buffer * __ocl_buffer_j1,
		   int j2[1037][2], ocl_buffer * __ocl_buffer_j2,
		   int j3[1037][2], ocl_buffer * __ocl_buffer_j3, int m,
		   int ind);
static void zero3(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		  unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		  unsigned int z_offset, int n1, int n2, int n3);
int main(int argc, char *argv[], ocl_buffer * __ocl_buffer_argv)
{
	{
		int k, it;
		double t, tinit, mflops;
		int nthreads = 1;
		init_ocl_runtime();
		double *u_data = ((void *)0);
		double *v_data = ((void *)0);
		double *r_data = ((void *)0);
		unsigned int *u_idx = ((void *)0);
		unsigned int *v_idx = ((void *)0);
		unsigned int *r_idx = ((void *)0);
		unsigned int u_data_len = 0;
		unsigned int u_idx_len = 0;
		unsigned int v_data_len = 0;
		unsigned int v_idx_len = 0;
		unsigned int r_data_len = 0;
		unsigned int r_idx_len = 0;
		double a[4], c[4];
		DECLARE_LOCALVAR_OCL_BUFFER(a, double, (4));
		DECLARE_LOCALVAR_OCL_BUFFER(c, double, (4));
		double rnm2, rnmu;
		double epsilon = 1.0e-8;
		int n1, n2, n3, nit;
		double verify_value;
		boolean verified;
		int i, j, l;
		timer_clear(1);
		timer_clear(2);
		timer_start(2);
		printf
		    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - MG Benchmark\n\n");
		printf(" No input file. Using compiled defaults\n");
		lt = 9;
		nit = 20;
		nx[lt] = 512;
		ny[lt] = 512;
		nz[lt] = 512;
		for (i = 0; i <= 7; i++) {
			debug_vec[i] = 0;
		}
		if ((nx[lt] != ny[lt]) || (nx[lt] != nz[lt])) {
			Class = 'U';
		} else if (nx[lt] == 32 && nit == 4) {
			Class = 'S';
		} else if (nx[lt] == 64 && nit == 40) {
			Class = 'W';
		} else if (nx[lt] == 256 && nit == 20) {
			Class = 'B';
		} else if (nx[lt] == 512 && nit == 20) {
			Class = 'C';
		} else if (nx[lt] == 256 && nit == 4) {
			Class = 'A';
		} else {
			Class = 'U';
		}
		a[0] = -8.0 / 3.0;
		a[1] = 0.0;
		a[2] = 1.0 / 6.0;
		a[3] = 1.0 / 12.0;
		if (Class == 'A' || Class == 'S' || Class == 'W') {
			c[0] = -3.0 / 8.0;
			c[1] = 1.0 / 32.0;
			c[2] = -1.0 / 64.0;
			c[3] = 0.0;
		} else {
			c[0] = -3.0 / 17.0;
			c[1] = 1.0 / 33.0;
			c[2] = -1.0 / 61.0;
			c[3] = 0.0;
		}
		lb = 1;
		setup(&n1, NULL, &n2, NULL, &n3, NULL, lt);
		u_idx_len = lt + 1;
		u_idx =
		    (unsigned int *)malloc(u_idx_len * sizeof(unsigned int));
		for (l = lt; l >= 1; l--) {
			u_idx[l] = u_idx_len;
			u_idx_len += m1[l];
			u_idx =
			    (unsigned int *)realloc(u_idx,
						    u_idx_len *
						    sizeof(unsigned int));
			for (i = 0; i < m1[l]; i++) {
				u_idx[u_idx[l] + i] = u_idx_len;
				u_idx_len += m2[l];
				u_idx =
				    (unsigned int *)realloc(u_idx,
							    u_idx_len *
							    sizeof(unsigned
								   int));
				for (j = 0; j < m2[l]; j++) {
					u_idx[u_idx[u_idx[l] + i] + j] =
					    u_data_len;
					u_data_len += m3[l];
					if (!u_data)
						u_data =
						    (double *)malloc(u_data_len
								     *
								     sizeof
								     (double));
					else
						u_data =
						    (double *)realloc(u_data,
								      u_data_len
								      *
								      sizeof
								      (double));
				}
			}
		}
		v_idx_len = m1[lt];
		v_idx =
		    (unsigned int *)malloc(v_idx_len * sizeof(unsigned int));
		for (i = 0; i < m1[lt]; i++) {
			v_idx[i] = v_idx_len;
			v_idx_len += m2[lt];
			v_idx =
			    (unsigned int *)realloc(v_idx,
						    v_idx_len *
						    sizeof(unsigned int));
			for (j = 0; j < m2[lt]; j++) {
				v_idx[v_idx[i] + j] = v_data_len;
				v_data_len += m3[lt];
				if (!v_data)
					v_data =
					    (double *)malloc(v_data_len *
							     sizeof(double));
				else
					v_data =
					    (double *)realloc(v_data,
							      v_data_len *
							      sizeof(double));
			}
		}
		r_idx_len = lt + 1;
		r_idx =
		    (unsigned int *)malloc(r_idx_len * sizeof(unsigned int));
		for (l = lt; l >= 1; l--) {
			r_idx[l] = r_idx_len;
			r_idx_len += m1[l];
			r_idx =
			    (unsigned int *)realloc(r_idx,
						    r_idx_len *
						    sizeof(unsigned int));
			for (i = 0; i < m1[l]; i++) {
				r_idx[r_idx[l] + i] = r_idx_len;
				r_idx_len += m2[l];
				r_idx =
				    (unsigned int *)realloc(r_idx,
							    r_idx_len *
							    sizeof(unsigned
								   int));
				for (j = 0; j < m2[l]; j++) {
					r_idx[r_idx[r_idx[l] + i] + j] =
					    r_data_len;
					r_data_len += m3[l];
					if (!r_data)
						r_data =
						    (double *)malloc(r_data_len
								     *
								     sizeof
								     (double));
					else
						r_data =
						    (double *)realloc(r_data,
								      r_data_len
								      *
								      sizeof
								      (double));
				}
			}
		}
		DECLARE_LOCALVAR_OCL_BUFFER(u_data, double, (u_data_len));
		DECLARE_LOCALVAR_OCL_BUFFER(u_idx, unsigned, (u_idx_len));

		zero3(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], n1, n2, n3);
		DECLARE_LOCALVAR_OCL_BUFFER(v_data, double, (v_data_len));
		DECLARE_LOCALVAR_OCL_BUFFER(v_idx, unsigned, (v_idx_len));

		zran3(v_data, __ocl_buffer_v_data, v_idx, __ocl_buffer_v_idx,
		      n1, n2, n3, nx[lt], ny[lt], lt);
		flush_ocl_buffers();
		norm2u3(v_data, __ocl_buffer_v_data, v_idx, __ocl_buffer_v_idx,
			0, n1, n2, n3, &rnm2, NULL, &rnmu, NULL, nx[lt], ny[lt],
			nz[lt]);
		{
			printf(" Size: %3dx%3dx%3d (class %1c)\n", nx[lt],
			       ny[lt], nz[lt], Class);
			printf(" Iterations: %3d\n", nit);
		}
		DECLARE_LOCALVAR_OCL_BUFFER(r_data, double, (r_data_len));
		DECLARE_LOCALVAR_OCL_BUFFER(r_idx, unsigned, (r_idx_len));

		resid(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], v_data, __ocl_buffer_v_data, v_idx,
		      __ocl_buffer_v_idx, 0, r_data, __ocl_buffer_r_data, r_idx,
		      __ocl_buffer_r_idx, r_idx[lt], n1, n2, n3, a,
		      __ocl_buffer_a, lt);
		norm2u3(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
			r_idx[lt], n1, n2, n3, &rnm2, NULL, &rnmu, NULL, nx[lt],
			ny[lt], nz[lt]);

		mg3P(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		     v_data, __ocl_buffer_v_data, v_idx, __ocl_buffer_v_idx,
		     r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx, a,
		     __ocl_buffer_a, c, __ocl_buffer_c, n1, n2, n3, lt);
		resid(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], v_data, __ocl_buffer_v_data, v_idx,
		      __ocl_buffer_v_idx, 0, r_data, __ocl_buffer_r_data, r_idx,
		      __ocl_buffer_r_idx, r_idx[lt], n1, n2, n3, a,
		      __ocl_buffer_a, lt);
		setup(&n1, NULL, &n2, NULL, &n3, NULL, lt);
		zero3(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], n1, n2, n3);
		zran3(v_data, __ocl_buffer_v_data, v_idx, __ocl_buffer_v_idx,
		      n1, n2, n3, nx[lt], ny[lt], lt);
		flush_ocl_buffers();
		timer_stop(2);
		timer_start(1);
		resid(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], v_data, __ocl_buffer_v_data, v_idx,
		      __ocl_buffer_v_idx, 0, r_data, __ocl_buffer_r_data, r_idx,
		      __ocl_buffer_r_idx, r_idx[lt], n1, n2, n3, a,
		      __ocl_buffer_a, lt);
		norm2u3(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
			r_idx[lt], n1, n2, n3, &rnm2, NULL, &rnmu, NULL, nx[lt],
			ny[lt], nz[lt]);
		for (it = 1; it <= nit; it++) {
			mg3P(u_data, __ocl_buffer_u_data, u_idx,
			     __ocl_buffer_u_idx, v_data, __ocl_buffer_v_data,
			     v_idx, __ocl_buffer_v_idx, r_data,
			     __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx, a,
			     __ocl_buffer_a, c, __ocl_buffer_c, n1, n2, n3, lt);
			resid(u_data, __ocl_buffer_u_data, u_idx,
			      __ocl_buffer_u_idx, u_idx[lt], v_data,
			      __ocl_buffer_v_data, v_idx, __ocl_buffer_v_idx, 0,
			      r_data, __ocl_buffer_r_data, r_idx,
			      __ocl_buffer_r_idx, r_idx[lt], n1, n2, n3, a,
			      __ocl_buffer_a, lt);
		}
		norm2u3(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
			r_idx[lt], n1, n2, n3, &rnm2, NULL, &rnmu, NULL, nx[lt],
			ny[lt], nz[lt]);
		sync_ocl_buffers();
		timer_stop(1);
		t = timer_read(1);
		tinit = timer_read(2);
		verified = 0;
		verify_value = 0.0;
		printf(" Initialization time: %15.3f seconds\n", tinit);
		printf(" Benchmark completed\n");
		if (Class != 'U') {
			if (Class == 'S') {
				verify_value = 0.530770700573e-04;
			} else if (Class == 'W') {
				verify_value = 0.250391406439e-17;
			} else if (Class == 'A') {
				verify_value = 0.2433365309e-5;
			} else if (Class == 'B') {
				verify_value = 0.180056440132e-5;
			} else if (Class == 'C') {
				verify_value = 0.570674826298e-06;
			}
			if (fabs(rnm2 - verify_value) <= epsilon) {
				verified = 1;
				printf(" VERIFICATION SUCCESSFUL\n");
				printf(" L2 Norm is %20.12e\n", rnm2);
				printf(" Error is   %20.12e\n",
				       rnm2 - verify_value);
			} else {
				verified = 0;
				printf(" VERIFICATION FAILED\n");
				printf(" L2 Norm is             %20.12e\n",
				       rnm2);
				printf(" The correct L2 Norm is %20.12e\n",
				       verify_value);
			}
		} else {
			verified = 0;
			printf(" Problem size unknown\n");
			printf(" NO VERIFICATION PERFORMED\n");
		}
		if (t != 0.0) {
			int nn = nx[lt] * ny[lt] * nz[lt];
			mflops = 58. * nit * nn * 1.0e-6 / t;
		} else {
			mflops = 0.0;
		}
		release_ocl_buffers();
		c_print_results("MG", Class, nx[lt], ny[lt], nz[lt], nit,
				nthreads, t, mflops, "          floating point",
				verified, "2.3", "06 Aug 2012", "gcc", "gcc",
				"(none)", "-I../common",
				"-std=c99 -O3 -fopenmp", "-lm -fopenmp",
				"randdp");
	}
}

static void setup(int *n1, ocl_buffer * __ocl_buffer_n1, int *n2,
		  ocl_buffer * __ocl_buffer_n2, int *n3,
		  ocl_buffer * __ocl_buffer_n3, int lt)
{
	{
		int k;
		for (k = lt - 1; k >= 1; k--) {
			nx[k] = nx[k + 1] / 2;
			ny[k] = ny[k + 1] / 2;
			nz[k] = nz[k + 1] / 2;
		}
		for (k = 1; k <= lt; k++) {
			m1[k] = nx[k] + 2;
			m2[k] = nz[k] + 2;
			m3[k] = ny[k] + 2;
		}
		is1 = 1;
		ie1 = nx[lt];
		*n1 = nx[lt] + 2;
		is2 = 1;
		ie2 = ny[lt];
		*n2 = ny[lt] + 2;
		is3 = 1;
		ie3 = nz[lt];
		*n3 = nz[lt] + 2;
		if (debug_vec[1] >= 1) {
			printf(" in setup, \n");
			printf
			    ("  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
			printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n", lt,
			       nx[lt], ny[lt], nz[lt], *n1, *n2, *n3, is1, is2,
			       is3, ie1, ie2, ie3);
		}
	}
}

static void mg3P(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		 unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		 double *v_data, ocl_buffer * __ocl_buffer_v_data,
		 unsigned int *v_idx, ocl_buffer * __ocl_buffer_v_idx,
		 double *r_data, ocl_buffer * __ocl_buffer_r_data,
		 unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		 double a[4], ocl_buffer * __ocl_buffer_a, double c[4],
		 ocl_buffer * __ocl_buffer_c, int n1, int n2, int n3, int k)
{
	{
		int j;
		for (k = lt; k >= lb + 1; k--) {
			j = k - 1;
			rprj3(r_data, __ocl_buffer_r_data, r_idx,
			      __ocl_buffer_r_idx, r_idx[k], m1[k], m2[k], m3[k],
			      r_data, __ocl_buffer_r_data, r_idx,
			      __ocl_buffer_r_idx, r_idx[j], m1[j], m2[j], m3[j],
			      k);
		}
		k = lb;
		zero3(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[k], m1[k], m2[k], m3[k]);
		psinv(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
		      r_idx[k], u_data, __ocl_buffer_u_data, u_idx,
		      __ocl_buffer_u_idx, u_idx[k], m1[k], m2[k], m3[k], c,
		      __ocl_buffer_c, k);
		for (k = lb + 1; k <= lt - 1; k++) {
			j = k - 1;
			zero3(u_data, __ocl_buffer_u_data, u_idx,
			      __ocl_buffer_u_idx, u_idx[k], m1[k], m2[k],
			      m3[k]);
			interp(u_data, __ocl_buffer_u_data, u_idx,
			       __ocl_buffer_u_idx, u_idx[j], m1[j], m2[j],
			       m3[j], u_data, __ocl_buffer_u_data, u_idx,
			       __ocl_buffer_u_idx, u_idx[k], m1[k], m2[k],
			       m3[k], k);
			resid(u_data, __ocl_buffer_u_data, u_idx,
			      __ocl_buffer_u_idx, u_idx[k], r_data,
			      __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
			      r_idx[k], r_data, __ocl_buffer_r_data, r_idx,
			      __ocl_buffer_r_idx, r_idx[k], m1[k], m2[k], m3[k],
			      a, __ocl_buffer_a, k);
			psinv(r_data, __ocl_buffer_r_data, r_idx,
			      __ocl_buffer_r_idx, r_idx[k], u_data,
			      __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
			      u_idx[k], m1[k], m2[k], m3[k], c, __ocl_buffer_c,
			      k);
		}
		j = lt - 1;
		k = lt;
		interp(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		       u_idx[j], m1[j], m2[j], m3[j], u_data,
		       __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		       u_idx[lt], n1, n2, n3, k);
		resid(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_idx[lt], v_data, __ocl_buffer_v_data, v_idx,
		      __ocl_buffer_v_idx, 0, r_data, __ocl_buffer_r_data, r_idx,
		      __ocl_buffer_r_idx, r_idx[lt], n1, n2, n3, a,
		      __ocl_buffer_a, k);
		psinv(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
		      r_idx[lt], u_data, __ocl_buffer_u_data, u_idx,
		      __ocl_buffer_u_idx, u_idx[lt], n1, n2, n3, c,
		      __ocl_buffer_c, k);
	}
}

static void psinv(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, double *u_data,
		  ocl_buffer * __ocl_buffer_u_data, unsigned int *u_idx,
		  ocl_buffer * __ocl_buffer_u_idx, unsigned int u_offset,
		  int n1, int n2, int n3, double c[4],
		  ocl_buffer * __ocl_buffer_c, int k)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_data_psinv,__ocl_p_r_data_psinv, r_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_idx_psinv,__ocl_p_r_idx_psinv, r_idx, (), unsigned);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_data_psinv,__ocl_p_u_data_psinv, u_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_idx_psinv,__ocl_p_u_idx_psinv, u_idx, (), unsigned);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_c_psinv, __ocl_p_c_psinv, c, (4),
				 double);
	{
		int i3, i2, i1;
		//--------------------------------------------------------------
		//Loop defined at line 643 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (n3 - 1) - (1);
			_ocl_gws[1] = (n2 - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_psinv_0, 0, sizeof(int), &i1);
			oclSetKernelArg(__ocl_psinv_0, 1, sizeof(int), &n1);
			oclSetKernelArgBuffer(__ocl_psinv_0, 2,
					      __ocl_buffer_r_data);
			oclSetKernelArgBuffer(__ocl_psinv_0, 3,
					      __ocl_buffer_r_idx);
			oclSetKernelArg(__ocl_psinv_0, 4, sizeof(unsigned),
					&r_offset);
			oclSetKernelArgBuffer(__ocl_psinv_0, 5,
					      __ocl_buffer_u_data);
			oclSetKernelArgBuffer(__ocl_psinv_0, 6,
					      __ocl_buffer_u_idx);
			oclSetKernelArg(__ocl_psinv_0, 7, sizeof(unsigned),
					&u_offset);
			oclSetKernelArgBuffer(__ocl_psinv_0, 8, __ocl_buffer_c);
			int __ocl_i3_bound = n3 - 1;
			oclSetKernelArg(__ocl_psinv_0, 9, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i2_bound = n2 - 1;
			oclSetKernelArg(__ocl_psinv_0, 10, sizeof(int),
					&__ocl_i2_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_u_data);
			oclDevWrites(__ocl_buffer_u_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_r_data);
			oclDevReads(__ocl_buffer_r_idx);
			oclDevReads(__ocl_buffer_c);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_psinv_0, 2, _ocl_gws);
		}

		comm3(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
		      u_offset, n1, n2, n3, k);
		if (debug_vec[0] >= 1) {
			rep_nrm(u_data, __ocl_buffer_u_data, u_idx,
				__ocl_buffer_u_idx, u_offset, n1, n2, n3,
				"   psinv", NULL, k);
		}
		if (debug_vec[3] >= k) {
			showall(u_data, __ocl_buffer_u_data, u_idx,
				__ocl_buffer_u_idx, u_offset, n1, n2, n3);
		}
	}
}

static void resid(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		  unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		  unsigned int u_offset, double *v_data,
		  ocl_buffer * __ocl_buffer_v_data, unsigned int *v_idx,
		  ocl_buffer * __ocl_buffer_v_idx, unsigned int v_offset,
		  double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, int n1, int n2, int n3, double a[4],
		  ocl_buffer * __ocl_buffer_a, int k)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_data_resid,__ocl_p_u_data_resid, u_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_idx_resid,__ocl_p_u_idx_resid, u_idx, (), unsigned);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_data_resid,__ocl_p_r_data_resid, r_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_idx_resid,__ocl_p_r_idx_resid, r_idx, (), unsigned);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_v_data_resid,__ocl_p_v_data_resid, v_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_v_idx_resid,__ocl_p_v_idx_resid, v_idx, (), unsigned);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_a_resid, __ocl_p_a_resid, a, (4),
				 double);
	{
		int i3, i2, i1;
		//--------------------------------------------------------------
		//Loop defined at line 716 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (n3 - 1) - (1);
			_ocl_gws[1] = (n2 - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_resid_0, 0, sizeof(int), &i1);
			oclSetKernelArg(__ocl_resid_0, 1, sizeof(int), &n1);
			oclSetKernelArgBuffer(__ocl_resid_0, 2,
					      __ocl_buffer_u_data);
			oclSetKernelArgBuffer(__ocl_resid_0, 3,
					      __ocl_buffer_u_idx);
			oclSetKernelArg(__ocl_resid_0, 4, sizeof(unsigned),
					&u_offset);
			oclSetKernelArgBuffer(__ocl_resid_0, 5,
					      __ocl_buffer_r_data);
			oclSetKernelArgBuffer(__ocl_resid_0, 6,
					      __ocl_buffer_r_idx);
			oclSetKernelArg(__ocl_resid_0, 7, sizeof(unsigned),
					&r_offset);
			oclSetKernelArgBuffer(__ocl_resid_0, 8,
					      __ocl_buffer_v_data);
			oclSetKernelArgBuffer(__ocl_resid_0, 9,
					      __ocl_buffer_v_idx);
			oclSetKernelArg(__ocl_resid_0, 10, sizeof(unsigned),
					&v_offset);
			oclSetKernelArgBuffer(__ocl_resid_0, 11,
					      __ocl_buffer_a);
			int __ocl_i3_bound = n3 - 1;
			oclSetKernelArg(__ocl_resid_0, 12, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i2_bound = n2 - 1;
			oclSetKernelArg(__ocl_resid_0, 13, sizeof(int),
					&__ocl_i2_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_r_data);
			oclDevWrites(__ocl_buffer_r_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_u_data);
			oclDevReads(__ocl_buffer_u_idx);
			oclDevReads(__ocl_buffer_v_data);
			oclDevReads(__ocl_buffer_v_idx);
			oclDevReads(__ocl_buffer_a);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_resid_0, 2, _ocl_gws);
		}

		comm3(r_data, __ocl_buffer_r_data, r_idx, __ocl_buffer_r_idx,
		      r_offset, n1, n2, n3, k);
		if (debug_vec[0] >= 1) {
			rep_nrm(r_data, __ocl_buffer_r_data, r_idx,
				__ocl_buffer_r_idx, r_offset, n1, n2, n3,
				"   resid", NULL, k);
		}
		if (debug_vec[2] >= k) {
			showall(r_data, __ocl_buffer_r_data, r_idx,
				__ocl_buffer_r_idx, r_offset, n1, n2, n3);
		}
	}
}

static void rprj3(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		  unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		  unsigned int r_offset, int m1k, int m2k, int m3k,
		  double *s_data, ocl_buffer * __ocl_buffer_s_data,
		  unsigned int *s_idx, ocl_buffer * __ocl_buffer_s_idx,
		  unsigned int s_offset, int m1j, int m2j, int m3j, int k)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_data_rprj3,__ocl_p_r_data_rprj3, r_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_r_idx_rprj3,__ocl_p_r_idx_rprj3, r_idx, (), unsigned);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_s_data_rprj3,__ocl_p_s_data_rprj3, s_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_s_idx_rprj3,__ocl_p_s_idx_rprj3, s_idx, (), unsigned);
	{
		int j3, j2, j1, i3, i2, i1, d1, d2, d3;
		double x2, y2;
		if (m1k == 3) {
			d1 = 2;
		} else {
			d1 = 1;
		}
		if (m2k == 3) {
			d2 = 2;
		} else {
			d2 = 1;
		}
		if (m3k == 3) {
			d3 = 2;
		} else {
			d3 = 1;
		}
		//--------------------------------------------------------------
		//Loop defined at line 804 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (m3j - 1) - (1);

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_rprj3_0, 0, sizeof(int), &d3);
			oclSetKernelArg(__ocl_rprj3_0, 1, sizeof(int), &j2);
			oclSetKernelArg(__ocl_rprj3_0, 2, sizeof(int), &m2j);
			oclSetKernelArg(__ocl_rprj3_0, 3, sizeof(int), &d2);
			oclSetKernelArg(__ocl_rprj3_0, 4, sizeof(int), &j1);
			oclSetKernelArg(__ocl_rprj3_0, 5, sizeof(int), &m1j);
			oclSetKernelArg(__ocl_rprj3_0, 6, sizeof(int), &d1);
			oclSetKernelArgBuffer(__ocl_rprj3_0, 7,
					      __ocl_buffer_r_data);
			oclSetKernelArgBuffer(__ocl_rprj3_0, 8,
					      __ocl_buffer_r_idx);
			oclSetKernelArg(__ocl_rprj3_0, 9, sizeof(unsigned),
					&r_offset);
			oclSetKernelArgBuffer(__ocl_rprj3_0, 10,
					      __ocl_buffer_s_data);
			oclSetKernelArgBuffer(__ocl_rprj3_0, 11,
					      __ocl_buffer_s_idx);
			oclSetKernelArg(__ocl_rprj3_0, 12, sizeof(unsigned),
					&s_offset);
			int __ocl_j3_bound = m3j - 1;
			oclSetKernelArg(__ocl_rprj3_0, 13, sizeof(int),
					&__ocl_j3_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_s_data);
			oclDevWrites(__ocl_buffer_s_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_r_data);
			oclDevReads(__ocl_buffer_r_idx);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_rprj3_0, 1, _ocl_gws);
		}

		comm3(s_data, __ocl_buffer_s_data, s_idx, __ocl_buffer_s_idx,
		      s_offset, m1j, m2j, m3j, k - 1);
		if (debug_vec[0] >= 1) {
			rep_nrm(s_data, __ocl_buffer_s_data, s_idx,
				__ocl_buffer_s_idx, s_offset, m1j, m2j, m3j,
				"   rprj3", NULL, k - 1);
		}
		if (debug_vec[4] >= k) {
			showall(s_data, __ocl_buffer_s_data, s_idx,
				__ocl_buffer_s_idx, s_offset, m1j, m2j, m3j);
		}
	}
}

static void interp(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		   unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		   unsigned int z_offset, int mm1, int mm2, int mm3,
		   double *u_data, ocl_buffer * __ocl_buffer_u_data,
		   unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		   unsigned int u_offset, int n1, int n2, int n3, int k)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_data_interp,__ocl_p_z_data_interp, z_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_idx_interp,__ocl_p_z_idx_interp, z_idx, (), unsigned);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_data_interp,__ocl_p_u_data_interp, u_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_idx_interp,__ocl_p_u_idx_interp, u_idx, (), unsigned);
	{
		int i3, i2, i1, d1, d2, d3, t1, t2, t3;
		if (n1 != 3 && n2 != 3 && n3 != 3) {
			//--------------------------------------------------------------
			//Loop defined at line 893 of mg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[2];
				_ocl_gws[0] = (mm3 - 1) - (0);
				_ocl_gws[1] = (mm2 - 1) - (0);

				oclGetWorkSize(2, _ocl_gws, NULL);
				oclSetKernelArg(__ocl_interp_0, 0, sizeof(int),
						&i1);
				oclSetKernelArg(__ocl_interp_0, 1, sizeof(int),
						&mm1);
				oclSetKernelArgBuffer(__ocl_interp_0, 2,
						      __ocl_buffer_z_data);
				oclSetKernelArgBuffer(__ocl_interp_0, 3,
						      __ocl_buffer_z_idx);
				oclSetKernelArg(__ocl_interp_0, 4,
						sizeof(unsigned), &z_offset);
				oclSetKernelArgBuffer(__ocl_interp_0, 5,
						      __ocl_buffer_u_data);
				oclSetKernelArgBuffer(__ocl_interp_0, 6,
						      __ocl_buffer_u_idx);
				oclSetKernelArg(__ocl_interp_0, 7,
						sizeof(unsigned), &u_offset);
				int __ocl_i3_bound = mm3 - 1;
				oclSetKernelArg(__ocl_interp_0, 8, sizeof(int),
						&__ocl_i3_bound);
				int __ocl_i2_bound = mm2 - 1;
				oclSetKernelArg(__ocl_interp_0, 9, sizeof(int),
						&__ocl_i2_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_u_data);
				oclDevWrites(__ocl_buffer_u_idx);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				oclDevReads(__ocl_buffer_z_data);
				oclDevReads(__ocl_buffer_z_idx);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_interp_0, 2, _ocl_gws);
			}

		} else {
			if (n1 == 3) {
				d1 = 2;
				t1 = 1;
			} else {
				d1 = 1;
				t1 = 0;
			}
			if (n2 == 3) {
				d2 = 2;
				t2 = 1;
			} else {
				d2 = 1;
				t2 = 0;
			}
			if (n3 == 3) {
				d3 = 2;
				t3 = 1;
			} else {
				d3 = 1;
				t3 = 0;
			}
			//--------------------------------------------------------------
			//Loop defined at line 971 of mg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (mm3 - 1) - (d3) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				oclSetKernelArg(__ocl_interp_1, 0, sizeof(int),
						&i2);
				oclSetKernelArg(__ocl_interp_1, 1, sizeof(int),
						&d2);
				oclSetKernelArg(__ocl_interp_1, 2, sizeof(int),
						&mm2);
				oclSetKernelArg(__ocl_interp_1, 3, sizeof(int),
						&i1);
				oclSetKernelArg(__ocl_interp_1, 4, sizeof(int),
						&d1);
				oclSetKernelArg(__ocl_interp_1, 5, sizeof(int),
						&mm1);
				oclSetKernelArgBuffer(__ocl_interp_1, 6,
						      __ocl_buffer_u_data);
				oclSetKernelArgBuffer(__ocl_interp_1, 7,
						      __ocl_buffer_u_idx);
				oclSetKernelArg(__ocl_interp_1, 8,
						sizeof(unsigned), &u_offset);
				oclSetKernelArg(__ocl_interp_1, 9, sizeof(int),
						&d3);
				oclSetKernelArgBuffer(__ocl_interp_1, 10,
						      __ocl_buffer_z_data);
				oclSetKernelArgBuffer(__ocl_interp_1, 11,
						      __ocl_buffer_z_idx);
				oclSetKernelArg(__ocl_interp_1, 12,
						sizeof(unsigned), &z_offset);
				oclSetKernelArg(__ocl_interp_1, 13, sizeof(int),
						&t1);
				oclSetKernelArg(__ocl_interp_1, 14, sizeof(int),
						&t2);
				int __ocl_i3_bound = mm3 - 1;
				oclSetKernelArg(__ocl_interp_1, 15, sizeof(int),
						&__ocl_i3_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_u_data);
				oclDevWrites(__ocl_buffer_u_idx);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				oclDevReads(__ocl_buffer_z_data);
				oclDevReads(__ocl_buffer_z_idx);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_interp_1, 1, _ocl_gws);
			}

			//--------------------------------------------------------------
			//Loop defined at line 1012 of mg.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (mm3 - 1) - (1) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				oclSetKernelArg(__ocl_interp_2, 0, sizeof(int),
						&i2);
				oclSetKernelArg(__ocl_interp_2, 1, sizeof(int),
						&d2);
				oclSetKernelArg(__ocl_interp_2, 2, sizeof(int),
						&mm2);
				oclSetKernelArg(__ocl_interp_2, 3, sizeof(int),
						&i1);
				oclSetKernelArg(__ocl_interp_2, 4, sizeof(int),
						&d1);
				oclSetKernelArg(__ocl_interp_2, 5, sizeof(int),
						&mm1);
				oclSetKernelArgBuffer(__ocl_interp_2, 6,
						      __ocl_buffer_u_data);
				oclSetKernelArgBuffer(__ocl_interp_2, 7,
						      __ocl_buffer_u_idx);
				oclSetKernelArg(__ocl_interp_2, 8,
						sizeof(unsigned), &u_offset);
				oclSetKernelArg(__ocl_interp_2, 9, sizeof(int),
						&t3);
				oclSetKernelArgBuffer(__ocl_interp_2, 10,
						      __ocl_buffer_z_data);
				oclSetKernelArgBuffer(__ocl_interp_2, 11,
						      __ocl_buffer_z_idx);
				oclSetKernelArg(__ocl_interp_2, 12,
						sizeof(unsigned), &z_offset);
				oclSetKernelArg(__ocl_interp_2, 13, sizeof(int),
						&t1);
				oclSetKernelArg(__ocl_interp_2, 14, sizeof(int),
						&t2);
				int __ocl_i3_bound = mm3 - 1;
				oclSetKernelArg(__ocl_interp_2, 15, sizeof(int),
						&__ocl_i3_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_u_data);
				oclDevWrites(__ocl_buffer_u_idx);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				oclDevReads(__ocl_buffer_z_data);
				oclDevReads(__ocl_buffer_z_idx);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_interp_2, 1, _ocl_gws);
			}

		}
		{
			if (debug_vec[0] >= 1) {
				rep_nrm(z_data, __ocl_buffer_z_data, z_idx,
					__ocl_buffer_z_idx, z_offset, mm1, mm2,
					mm3, "z: inter", NULL, k - 1);
				rep_nrm(u_data, __ocl_buffer_u_data, u_idx,
					__ocl_buffer_u_idx, u_offset, n1, n2,
					n3, "u: inter", NULL, k);
			}
			if (debug_vec[5] >= k) {
				showall(z_data, __ocl_buffer_z_data, z_idx,
					__ocl_buffer_z_idx, z_offset, mm1, mm2,
					mm3);
				showall(u_data, __ocl_buffer_u_data, u_idx,
					__ocl_buffer_u_idx, u_offset, n1, n2,
					n3);
			}
		}
	}
}

static void norm2u3(double *r_data, ocl_buffer * __ocl_buffer_r_data,
		    unsigned int *r_idx, ocl_buffer * __ocl_buffer_r_idx,
		    unsigned int r_offset, int n1, int n2, int n3, double *rnm2,
		    ocl_buffer * __ocl_buffer_rnm2, double *rnmu,
		    ocl_buffer * __ocl_buffer_rnmu, int nx, int ny, int nz)
{
	{
		static double s = 0.0;
		double tmp;
		int i3, i2, i1, n;
		double p_s = 0.0, p_a = 0.0;
		n = nx * ny * nz;
		oclHostReads(__ocl_buffer_r_data);
		oclSync();
		for (i3 = 1; i3 < n3 - 1; i3++) {
			for (i2 = 1; i2 < n2 - 1; i2++) {
				for (i1 = 1; i1 < n1 - 1; i1++) {
					p_s =
					    p_s +
					    r_data[r_idx
						   [r_idx[r_offset + i1] + i2] +
						   i3] *
					    r_data[r_idx
						   [r_idx[r_offset + i1] + i2] +
						   i3];
					tmp =
					    fabs(r_data
						 [r_idx
						  [r_idx[r_offset + i1] + i2] +
						  i3]);
					if (tmp > p_a)
						p_a = tmp;
				}
			}
		}
		{
			s += p_s;
			if (p_a > *rnmu)
				*rnmu = p_a;
		}
		{
			*rnm2 = sqrt(s / (double)n);
			s = 0.0;
		}
	}
}

static void rep_nrm(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		    unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		    unsigned int u_offset, int n1, int n2, int n3, char *title,
		    ocl_buffer * __ocl_buffer_title, int kk)
{
	{
		double rnm2, rnmu;
		norm2u3(u_data, __ocl_buffer_u_data, u_idx, __ocl_buffer_u_idx,
			u_offset, n1, n2, n3, &rnm2, NULL, &rnmu, NULL, nx[kk],
			ny[kk], nz[kk]);
		printf(" Level%2d in %8s: norms =%21.14e%21.14e\n", kk, title,
		       rnm2, rnmu);
	}
}

static void comm3(double *u_data, ocl_buffer * __ocl_buffer_u_data,
		  unsigned int *u_idx, ocl_buffer * __ocl_buffer_u_idx,
		  unsigned int u_offset, int n1, int n2, int n3, int kk)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_data_comm3,__ocl_p_u_data_comm3, u_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u_idx_comm3,__ocl_p_u_idx_comm3, u_idx, (), unsigned);
	{
		int i1, i2, i3;
		//--------------------------------------------------------------
		//Loop defined at line 1168 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (n3 - 1) - (1);
			_ocl_gws[1] = (n2 - 1) - (1);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_comm3_0, 0,
					      __ocl_buffer_u_data);
			oclSetKernelArgBuffer(__ocl_comm3_0, 1,
					      __ocl_buffer_u_idx);
			oclSetKernelArg(__ocl_comm3_0, 2, sizeof(unsigned),
					&u_offset);
			oclSetKernelArg(__ocl_comm3_0, 3, sizeof(int), &n1);
			int __ocl_i3_bound = n3 - 1;
			oclSetKernelArg(__ocl_comm3_0, 4, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i2_bound = n2 - 1;
			oclSetKernelArg(__ocl_comm3_0, 5, sizeof(int),
					&__ocl_i2_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_u_data);
			oclDevWrites(__ocl_buffer_u_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_comm3_0, 2, _ocl_gws);
		}

		//--------------------------------------------------------------
		//Loop defined at line 1181 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (n3 - 1) - (1);
			_ocl_gws[1] = (n1) - (0);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_comm3_1, 0,
					      __ocl_buffer_u_data);
			oclSetKernelArgBuffer(__ocl_comm3_1, 1,
					      __ocl_buffer_u_idx);
			oclSetKernelArg(__ocl_comm3_1, 2, sizeof(unsigned),
					&u_offset);
			oclSetKernelArg(__ocl_comm3_1, 3, sizeof(int), &n2);
			int __ocl_i3_bound = n3 - 1;
			oclSetKernelArg(__ocl_comm3_1, 4, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i1_bound = n1;
			oclSetKernelArg(__ocl_comm3_1, 5, sizeof(int),
					&__ocl_i1_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_u_data);
			oclDevWrites(__ocl_buffer_u_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_comm3_1, 2, _ocl_gws);
		}

		//--------------------------------------------------------------
		//Loop defined at line 1194 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (n2) - (0);
			_ocl_gws[1] = (n1) - (0);

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_comm3_2, 0,
					      __ocl_buffer_u_data);
			oclSetKernelArgBuffer(__ocl_comm3_2, 1,
					      __ocl_buffer_u_idx);
			oclSetKernelArg(__ocl_comm3_2, 2, sizeof(unsigned),
					&u_offset);
			oclSetKernelArg(__ocl_comm3_2, 3, sizeof(int), &n3);
			int __ocl_i2_bound = n2;
			oclSetKernelArg(__ocl_comm3_2, 4, sizeof(int),
					&__ocl_i2_bound);
			int __ocl_i1_bound = n1;
			oclSetKernelArg(__ocl_comm3_2, 5, sizeof(int),
					&__ocl_i1_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_u_data);
			oclDevWrites(__ocl_buffer_u_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_comm3_2, 2, _ocl_gws);
		}

	}
}

static void zran3(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		  unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx, int n1,
		  int n2, int n3, int nx, int ny, int k)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_data_zran3,__ocl_p_z_data_zran3, z_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_idx_zran3,__ocl_p_z_idx_zran3, z_idx, (), unsigned);
	{
		int i0, m0, m1;
		int i1, i2, i3, d1, e1, e2, e3;
		double xx, x0, x1, a1, a2, ai;
		double ten[10][2], best;
		DECLARE_LOCALVAR_OCL_BUFFER(ten, double, (10 * 2));
		int i;
		int jg[4][10][2];
		DECLARE_LOCALVAR_OCL_BUFFER(jg, int, (4 * 10 * 2));
		double rdummy;
		a1 = power(pow(5.0, 13), nx);
		a2 = power(pow(5.0, 13), nx * ny);
		{
			zero3(z_data, __ocl_buffer_z_data, z_idx,
			      __ocl_buffer_z_idx, 0, n1, n2, n3);
		}
		oclHostWrites(__ocl_buffer_z_data);
		oclHostWrites(__ocl_buffer_z_idx);
		oclSync();
		i = is1 - 1 + nx * (is2 - 1 + ny * (is3 - 1));
		ai = power(pow(5.0, 13), i);
		d1 = ie1 - is1 + 1;
		e1 = ie1 - is1 + 2;
		e2 = ie2 - is2 + 2;
		e3 = ie3 - is3 + 2;
		x0 = 314159265.e0;
		rdummy = randlc(&x0, ai);
		for (i3 = 1; i3 < e3; i3++) {
			x1 = x0;
			for (i2 = 1; i2 < e2; i2++) {
				xx = x1;
				my_vranlc(d1, &xx, NULL, pow(5.0, 13), z_data,
					  __ocl_buffer_z_data, z_idx,
					  __ocl_buffer_z_idx, i2, i3);
				rdummy = randlc(&x1, a1);
			}
			rdummy = randlc(&x0, a2);
		}
		for (i = 0; i < 10; i++) {
			ten[i][1] = 0.0;
			zran3_j1[i][1] = 0;
			zran3_j2[i][1] = 0;
			zran3_j3[i][1] = 0;
			ten[i][0] = 1.0;
			zran3_j1[i][0] = 0;
			zran3_j2[i][0] = 0;
			zran3_j3[i][0] = 0;
		}
		for (i3 = 1; i3 < n3 - 1; i3++) {
			for (i2 = 1; i2 < n2 - 1; i2++) {
				for (i1 = 1; i1 < n1 - 1; i1++) {
					if (z_data[z_idx[z_idx[i1] + i2] + i3] >
					    ten[0][1]) {
						ten[0][1] =
						    z_data[z_idx[z_idx[i1] + i2]
							   + i3];
						zran3_j1[0][1] = i1;
						zran3_j2[0][1] = i2;
						zran3_j3[0][1] = i3;

						bubble(ten, __ocl_buffer_ten,
						       zran3_j1,
						       __ocl_buffer_zran3_j1,
						       zran3_j2,
						       __ocl_buffer_zran3_j2,
						       zran3_j3,
						       __ocl_buffer_zran3_j3,
						       10, 1);
					}
					if (z_data[z_idx[z_idx[i1] + i2] + i3] <
					    ten[0][0]) {
						ten[0][0] =
						    z_data[z_idx[z_idx[i1] + i2]
							   + i3];
						zran3_j1[0][0] = i1;
						zran3_j2[0][0] = i2;
						zran3_j3[0][0] = i3;
						bubble(ten, __ocl_buffer_ten,
						       zran3_j1,
						       __ocl_buffer_zran3_j1,
						       zran3_j2,
						       __ocl_buffer_zran3_j2,
						       zran3_j3,
						       __ocl_buffer_zran3_j3,
						       10, 0);
					}
				}
			}
		}
		i1 = 10 - 1;
		i0 = 10 - 1;
		for (i = 10 - 1; i >= 0; i--) {
			best =
			    z_data[z_idx
				   [z_idx[zran3_j1[i1][1]] + zran3_j2[i1][1]] +
				   zran3_j3[i1][1]];
			if (best ==
			    z_data[z_idx
				   [z_idx[zran3_j1[i1][1]] + zran3_j2[i1][1]] +
				   zran3_j3[i1][1]]) {
				jg[0][i][1] = 0;
				jg[1][i][1] = is1 - 1 + zran3_j1[i1][1];
				jg[2][i][1] = is2 - 1 + zran3_j2[i1][1];
				jg[3][i][1] = is3 - 1 + zran3_j3[i1][1];
				i1 = i1 - 1;
			} else {
				jg[0][i][1] = 0;
				jg[1][i][1] = 0;
				jg[2][i][1] = 0;
				jg[3][i][1] = 0;
			}
			ten[i][1] = best;
			best =
			    z_data[z_idx
				   [z_idx[zran3_j1[i0][0]] + zran3_j2[i0][0]] +
				   zran3_j3[i0][0]];
			if (best ==
			    z_data[z_idx
				   [z_idx[zran3_j1[i0][0]] + zran3_j2[i0][0]] +
				   zran3_j3[i0][0]]) {
				jg[0][i][0] = 0;
				jg[1][i][0] = is1 - 1 + zran3_j1[i0][0];
				jg[2][i][0] = is2 - 1 + zran3_j2[i0][0];
				jg[3][i][0] = is3 - 1 + zran3_j3[i0][0];
				i0 = i0 - 1;
			} else {
				jg[0][i][0] = 0;
				jg[1][i][0] = 0;
				jg[2][i][0] = 0;
				jg[3][i][0] = 0;
			}
			ten[i][0] = best;
		}
		m1 = i1 + 1;
		m0 = i0 + 1;
		//--------------------------------------------------------------
		//Loop defined at line 1387 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (n3) - (0);
			_ocl_gws[1] = (n2) - (0);
			_ocl_gws[2] = (n1) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_zran3_0, 0,
					      __ocl_buffer_z_data);
			oclSetKernelArgBuffer(__ocl_zran3_0, 1,
					      __ocl_buffer_z_idx);
			int __ocl_i3_bound = n3;
			oclSetKernelArg(__ocl_zran3_0, 2, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i2_bound = n2;
			oclSetKernelArg(__ocl_zran3_0, 3, sizeof(int),
					&__ocl_i2_bound);
			int __ocl_i1_bound = n1;
			oclSetKernelArg(__ocl_zran3_0, 4, sizeof(int),
					&__ocl_i1_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_z_data);
			oclDevWrites(__ocl_buffer_z_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_zran3_0, 3, _ocl_gws);
		}

		//--------------------------------------------------------------
		//Loop defined at line 1397 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (10 - 1) - (m0) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_zran3_1, 0,
					      __ocl_buffer_z_data);
			oclSetKernelArgBuffer(__ocl_zran3_1, 1,
					      __ocl_buffer_z_idx);
			oclSetKernelArgBuffer(__ocl_zran3_1, 2,
					      __ocl_buffer_zran3_j1);
			oclSetKernelArgBuffer(__ocl_zran3_1, 3,
					      __ocl_buffer_zran3_j2);
			oclSetKernelArgBuffer(__ocl_zran3_1, 4,
					      __ocl_buffer_zran3_j3);
			oclSetKernelArg(__ocl_zran3_1, 5, sizeof(int), &m0);
			int __ocl_i_bound = 10 - 1;
			oclSetKernelArg(__ocl_zran3_1, 6, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_z_data);
			oclDevWrites(__ocl_buffer_z_idx);
			oclDevWrites(__ocl_buffer_zran3_j1);
			oclDevWrites(__ocl_buffer_zran3_j2);
			oclDevWrites(__ocl_buffer_zran3_j3);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_zran3_1, 1, _ocl_gws);
		}

		//--------------------------------------------------------------
		//Loop defined at line 1404 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[1];
			_ocl_gws[0] = (10 - 1) - (m1) + 1;

			oclGetWorkSize(1, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_zran3_2, 0,
					      __ocl_buffer_z_data);
			oclSetKernelArgBuffer(__ocl_zran3_2, 1,
					      __ocl_buffer_z_idx);
			oclSetKernelArgBuffer(__ocl_zran3_2, 2,
					      __ocl_buffer_zran3_j1);
			oclSetKernelArgBuffer(__ocl_zran3_2, 3,
					      __ocl_buffer_zran3_j2);
			oclSetKernelArgBuffer(__ocl_zran3_2, 4,
					      __ocl_buffer_zran3_j3);
			oclSetKernelArg(__ocl_zran3_2, 5, sizeof(int), &m1);
			int __ocl_i_bound = 10 - 1;
			oclSetKernelArg(__ocl_zran3_2, 6, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_z_data);
			oclDevWrites(__ocl_buffer_z_idx);
			oclDevWrites(__ocl_buffer_zran3_j1);
			oclDevWrites(__ocl_buffer_zran3_j2);
			oclDevWrites(__ocl_buffer_zran3_j3);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_zran3_2, 1, _ocl_gws);
		}

		comm3(z_data, __ocl_buffer_z_data, z_idx, __ocl_buffer_z_idx, 0,
		      n1, n2, n3, k);
	}
}

static void showall(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		    unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		    unsigned int z_offset, int n1, int n2, int n3)
{
	{
		int i1, i2, i3;
		int m1, m2, m3;
		m1 = (((n1) < (18)) ? (n1) : (18));
		m2 = (((n2) < (14)) ? (n2) : (14));
		m3 = (((n3) < (18)) ? (n3) : (18));
		printf("\n");
		for (i3 = 0; i3 < m3; i3++) {
			for (i1 = 0; i1 < m1; i1++) {
				for (i2 = 0; i2 < m2; i2++) {
					printf("%6.3f",
					       z_data[z_idx
						      [z_idx[z_offset + i1] +
						       i2] + i3]);
				}
				printf("\n");
			}
			printf(" - - - - - - - \n");
		}
		printf("\n");
	}
}

static double power(double a, int n)
{
	double aj;
	int nj;
	double rdummy;
	double power;
	power = 1.0;
	nj = n;
	aj = a;
	while (nj != 0) {
		if ((nj % 2) == 1)
			rdummy = randlc(&power, aj);
		rdummy = randlc(&aj, aj);
		nj = nj / 2;
	}
	return (power);
}

static void bubble(double ten[1037][2], ocl_buffer * __ocl_buffer_ten,
		   int j1[1037][2], ocl_buffer * __ocl_buffer_j1,
		   int j2[1037][2], ocl_buffer * __ocl_buffer_j2,
		   int j3[1037][2], ocl_buffer * __ocl_buffer_j3, int m,
		   int ind)
{
	{
		double temp;
		int i, j_temp;
		if (ind == 1) {
			for (i = 0; i < m - 1; i++) {
				if (ten[i][ind] > ten[i + 1][ind]) {
					temp = ten[i + 1][ind];
					ten[i + 1][ind] = ten[i][ind];
					ten[i][ind] = temp;
					j_temp = j1[i + 1][ind];
					j1[i + 1][ind] = j1[i][ind];
					j1[i][ind] = j_temp;
					j_temp = j2[i + 1][ind];
					j2[i + 1][ind] = j2[i][ind];
					j2[i][ind] = j_temp;
					j_temp = j3[i + 1][ind];
					j3[i + 1][ind] = j3[i][ind];
					j3[i][ind] = j_temp;
				} else {
					return;
				}
			}
		} else {
			for (i = 0; i < m - 1; i++) {
				if (ten[i][ind] < ten[i + 1][ind]) {
					temp = ten[i + 1][ind];
					ten[i + 1][ind] = ten[i][ind];
					ten[i][ind] = temp;
					j_temp = j1[i + 1][ind];
					j1[i + 1][ind] = j1[i][ind];
					j1[i][ind] = j_temp;
					j_temp = j2[i + 1][ind];
					j2[i + 1][ind] = j2[i][ind];
					j2[i][ind] = j_temp;
					j_temp = j3[i + 1][ind];
					j3[i + 1][ind] = j3[i][ind];
					j3[i][ind] = j_temp;
				} else {
					return;
				}
			}
		}
	}
}

static void zero3(double *z_data, ocl_buffer * __ocl_buffer_z_data,
		  unsigned int *z_idx, ocl_buffer * __ocl_buffer_z_idx,
		  unsigned int z_offset, int n1, int n2, int n3)
{
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_data_zero3,__ocl_p_z_data_zero3, z_data, (), double);
//CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_z_idx_zero3,__ocl_p_z_idx_zero3, z_idx, (), unsigned);
	{
		int i1, i2, i3;
		//--------------------------------------------------------------
		//Loop defined at line 1552 of mg.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (n3) - (0);
			_ocl_gws[1] = (n2) - (0);
			_ocl_gws[2] = (n1) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_zero3_0, 0,
					      __ocl_buffer_z_data);
			oclSetKernelArgBuffer(__ocl_zero3_0, 1,
					      __ocl_buffer_z_idx);
			oclSetKernelArg(__ocl_zero3_0, 2, sizeof(unsigned),
					&z_offset);
			int __ocl_i3_bound = n3;
			oclSetKernelArg(__ocl_zero3_0, 3, sizeof(int),
					&__ocl_i3_bound);
			int __ocl_i2_bound = n2;
			oclSetKernelArg(__ocl_zero3_0, 4, sizeof(int),
					&__ocl_i2_bound);
			int __ocl_i1_bound = n1;
			oclSetKernelArg(__ocl_zero3_0, 5, sizeof(int),
					&__ocl_i1_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_z_data);
			oclDevWrites(__ocl_buffer_z_idx);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_zero3_0, 3, _ocl_gws);
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

	__ocl_program = oclBuildProgram("mg.C.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_psinv_0 = oclCreateKernel(__ocl_program, "psinv_0");
	DYN_PROGRAM_CHECK(__ocl_psinv_0);
	__ocl_resid_0 = oclCreateKernel(__ocl_program, "resid_0");
	DYN_PROGRAM_CHECK(__ocl_resid_0);
	__ocl_rprj3_0 = oclCreateKernel(__ocl_program, "rprj3_0");
	DYN_PROGRAM_CHECK(__ocl_rprj3_0);
	__ocl_interp_0 = oclCreateKernel(__ocl_program, "interp_0");
	DYN_PROGRAM_CHECK(__ocl_interp_0);
	__ocl_interp_1 = oclCreateKernel(__ocl_program, "interp_1");
	DYN_PROGRAM_CHECK(__ocl_interp_1);
	__ocl_interp_2 = oclCreateKernel(__ocl_program, "interp_2");
	DYN_PROGRAM_CHECK(__ocl_interp_2);
	__ocl_comm3_0 = oclCreateKernel(__ocl_program, "comm3_0");
	DYN_PROGRAM_CHECK(__ocl_comm3_0);
	__ocl_comm3_1 = oclCreateKernel(__ocl_program, "comm3_1");
	DYN_PROGRAM_CHECK(__ocl_comm3_1);
	__ocl_comm3_2 = oclCreateKernel(__ocl_program, "comm3_2");
	DYN_PROGRAM_CHECK(__ocl_comm3_2);
	__ocl_zran3_0 = oclCreateKernel(__ocl_program, "zran3_0");
	DYN_PROGRAM_CHECK(__ocl_zran3_0);
	__ocl_zran3_1 = oclCreateKernel(__ocl_program, "zran3_1");
	DYN_PROGRAM_CHECK(__ocl_zran3_1);
	__ocl_zran3_2 = oclCreateKernel(__ocl_program, "zran3_2");
	DYN_PROGRAM_CHECK(__ocl_zran3_2);
	__ocl_zero3_0 = oclCreateKernel(__ocl_program, "zero3_0");
	DYN_PROGRAM_CHECK(__ocl_zero3_0);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_z1 = oclCreateBuffer(z1, (1037) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_z1, -1);
	__ocl_buffer_z2 = oclCreateBuffer(z2, (1037) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_z2, -1);
	__ocl_buffer_z3 = oclCreateBuffer(z3, (1037) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_z3, -1);
	__ocl_buffer_zran3_j1 =
	    oclCreateBuffer(zran3_j1, (10 * 2) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_zran3_j1, -1);
	__ocl_buffer_zran3_j2 =
	    oclCreateBuffer(zran3_j2, (10 * 2) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_zran3_j2, -1);
	__ocl_buffer_zran3_j3 =
	    oclCreateBuffer(zran3_j3, (10 * 2) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_zran3_j3, -1);
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_z1);
	oclHostWrites(__ocl_buffer_z2);
	oclHostWrites(__ocl_buffer_z3);
	oclHostWrites(__ocl_buffer_zran3_j1);
	oclHostWrites(__ocl_buffer_zran3_j2);
	oclHostWrites(__ocl_buffer_zran3_j3);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_z1);
	oclReleaseBuffer(__ocl_buffer_z2);
	oclReleaseBuffer(__ocl_buffer_z3);
	oclReleaseBuffer(__ocl_buffer_zran3_j1);
	oclReleaseBuffer(__ocl_buffer_zran3_j2);
	oclReleaseBuffer(__ocl_buffer_zran3_j3);
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_z1);
	oclHostWrites(__ocl_buffer_z2);
	oclHostWrites(__ocl_buffer_z3);
	oclHostWrites(__ocl_buffer_zran3_j1);
	oclHostWrites(__ocl_buffer_zran3_j2);
	oclHostWrites(__ocl_buffer_zran3_j3);
	if (__ocl_buffer_r_data_psinv) {
		oclHostWrites(__ocl_buffer_r_data_psinv);
	}
	if (__ocl_buffer_r_idx_psinv) {
		oclHostWrites(__ocl_buffer_r_idx_psinv);
	}
	if (__ocl_buffer_u_data_psinv) {
		oclHostWrites(__ocl_buffer_u_data_psinv);
	}
	if (__ocl_buffer_u_idx_psinv) {
		oclHostWrites(__ocl_buffer_u_idx_psinv);
	}
	if (__ocl_buffer_c_psinv) {
		oclHostWrites(__ocl_buffer_c_psinv);
	}
	if (__ocl_buffer_u_data_resid) {
		oclHostWrites(__ocl_buffer_u_data_resid);
	}
	if (__ocl_buffer_u_idx_resid) {
		oclHostWrites(__ocl_buffer_u_idx_resid);
	}
	if (__ocl_buffer_r_data_resid) {
		oclHostWrites(__ocl_buffer_r_data_resid);
	}
	if (__ocl_buffer_r_idx_resid) {
		oclHostWrites(__ocl_buffer_r_idx_resid);
	}
	if (__ocl_buffer_v_data_resid) {
		oclHostWrites(__ocl_buffer_v_data_resid);
	}
	if (__ocl_buffer_v_idx_resid) {
		oclHostWrites(__ocl_buffer_v_idx_resid);
	}
	if (__ocl_buffer_a_resid) {
		oclHostWrites(__ocl_buffer_a_resid);
	}
	if (__ocl_buffer_r_data_rprj3) {
		oclHostWrites(__ocl_buffer_r_data_rprj3);
	}
	if (__ocl_buffer_r_idx_rprj3) {
		oclHostWrites(__ocl_buffer_r_idx_rprj3);
	}
	if (__ocl_buffer_s_data_rprj3) {
		oclHostWrites(__ocl_buffer_s_data_rprj3);
	}
	if (__ocl_buffer_s_idx_rprj3) {
		oclHostWrites(__ocl_buffer_s_idx_rprj3);
	}
	if (__ocl_buffer_z_data_interp) {
		oclHostWrites(__ocl_buffer_z_data_interp);
	}
	if (__ocl_buffer_z_idx_interp) {
		oclHostWrites(__ocl_buffer_z_idx_interp);
	}
	if (__ocl_buffer_u_data_interp) {
		oclHostWrites(__ocl_buffer_u_data_interp);
	}
	if (__ocl_buffer_u_idx_interp) {
		oclHostWrites(__ocl_buffer_u_idx_interp);
	}
	if (__ocl_buffer_u_data_comm3) {
		oclHostWrites(__ocl_buffer_u_data_comm3);
	}
	if (__ocl_buffer_u_idx_comm3) {
		oclHostWrites(__ocl_buffer_u_idx_comm3);
	}
	if (__ocl_buffer_z_data_zran3) {
		oclHostWrites(__ocl_buffer_z_data_zran3);
	}
	if (__ocl_buffer_z_idx_zran3) {
		oclHostWrites(__ocl_buffer_z_idx_zran3);
	}
	if (__ocl_buffer_z_data_zero3) {
		oclHostWrites(__ocl_buffer_z_data_zero3);
	}
	if (__ocl_buffer_z_idx_zero3) {
		oclHostWrites(__ocl_buffer_z_idx_zero3);
	}
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
