//-------------------------------------------------------------------------------
//Host code 
//Generated at : Wed Aug  8 16:02:15 2012
//Compiler options: 
//      Software Cache  true
//      Local Memory    true
//      DefaultParallelDepth    3
//      UserDefParallelDepth    false
//      EnableLoopInterchange   true
//      Generating debug/profiling code false
//      EnableMLFeatureCollection       false
//      Array Linearization     true
//      GPU TLs false
//      Strict TLS Checking     true
//      Check TLS Conflict at the end of function       true
//-------------------------------------------------------------------------------

#include "npb-C.h"
#include "global.h"
#include "ocldef.h"

static double yy0_real[256][18];
static double yy0_imag[256][18];
static double yy1_real[256][18];
static double yy1_imag[256][18];
static void evolve(double u0_real[128][256][256],
		   ocl_buffer * __ocl_buffer_u0_real,
		   double u0_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_u0_imag,
		   double u1_real[128][256][256],
		   ocl_buffer * __ocl_buffer_u1_real,
		   double u1_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_u1_imag, int t,
		   int indexmap[128][256][256],
		   ocl_buffer * __ocl_buffer_indexmap, int d[3],
		   ocl_buffer * __ocl_buffer_d);
static void compute_initial_conditions(double u0_real[128][256][256],
				       ocl_buffer * __ocl_buffer_u0_real,
				       double u0_imag[128][256][256],
				       ocl_buffer * __ocl_buffer_u0_imag,
				       int d[3], ocl_buffer * __ocl_buffer_d);
static void ipow46(double a, int exponent, double *result,
		   ocl_buffer * __ocl_buffer_result);
static void setup();
static void compute_indexmap(int indexmap[128][256][256],
			     ocl_buffer * __ocl_buffer_indexmap, int d[3],
			     ocl_buffer * __ocl_buffer_d);
static void print_timers();
static void fft(int dir, double x1_real[128][256][256],
		ocl_buffer * __ocl_buffer_x1_real,
		ocl_buffer * rd__ocl_buffer_x1_real,
		ocl_buffer * wr__ocl_buffer_x1_real,
		double x1_imag[128][256][256],
		ocl_buffer * __ocl_buffer_x1_imag,
		ocl_buffer * rd__ocl_buffer_x1_imag,
		ocl_buffer * wr__ocl_buffer_x1_imag,
		double x2_real[128][256][256],
		ocl_buffer * __ocl_buffer_x2_real,
		ocl_buffer * rd__ocl_buffer_x2_real,
		ocl_buffer * wr__ocl_buffer_x2_real,
		double x2_imag[128][256][256],
		ocl_buffer * __ocl_buffer_x2_imag,
		ocl_buffer * rd__ocl_buffer_x2_imag,
		ocl_buffer * wr__ocl_buffer_x2_imag
		);
static void cffts1(int is, int d[3], 
		   ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   ocl_buffer * rd__ocl_buffer_x_real,
		   ocl_buffer * wr__ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   ocl_buffer * rd__ocl_buffer_x_imag,
		   ocl_buffer * wr__ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   ocl_buffer * rd__ocl_buffer_xout_real,
		   ocl_buffer * wr__ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag,
		   ocl_buffer * rd__ocl_buffer_xout_imag,
		   ocl_buffer * wr__ocl_buffer_xout_imag
		   );
static void cffts2(int is, int d[3], ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag);
static void cffts3(int is, int d[3], ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag);
static void fft_init(int n);
static void cfftz(int is, int m, int n, double x_real[256][18],
		  ocl_buffer * __ocl_buffer_x_real, double x_imag[256][18],
		  ocl_buffer * __ocl_buffer_x_imag, double y_real[256][18],
		  ocl_buffer * __ocl_buffer_y_real, double y_imag[256][18],
		  ocl_buffer * __ocl_buffer_y_imag);
static void fftz2(int is, int l, int m, int n, int ny, int ny1,
		  double u_real[256], ocl_buffer * __ocl_buffer_u_real,
		  double u_imag[256], ocl_buffer * __ocl_buffer_u_imag,
		  double x_real[256][18], ocl_buffer * __ocl_buffer_x_real,
		  double x_imag[256][18], ocl_buffer * __ocl_buffer_x_imag,
		  double y_real[256][18], ocl_buffer * __ocl_buffer_y_real,
		  double y_imag[256][18], ocl_buffer * __ocl_buffer_y_imag);
static int ilog2(int n);
static void checksum(int i, double u1_real[128][256][256],
		     ocl_buffer * __ocl_buffer_u1_real,
		     double u1_imag[128][256][256],
		     ocl_buffer * __ocl_buffer_u1_imag, int d[3],
		     ocl_buffer * __ocl_buffer_d);
static void verify(int d1, int d2, int d3, int nt, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified, char *class,
		   ocl_buffer * __ocl_buffer_class);
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

int main(int argc, char **argv)
{
	{
		int i, ierr;
		init_ocl_runtime();
		static double u0_real[128][256][256], u0_imag[128][256][256];
		DECLARE_LOCALVAR_OCL_BUFFER(u0_real, double, (128 * 256 * 256));
		DECLARE_LOCALVAR_OCL_BUFFER(u0_imag, double, (128 * 256 * 256));
		static double pad1_real[3], pad1_imag[3];
		DECLARE_LOCALVAR_OCL_BUFFER(pad1_real, double, (3));
		DECLARE_LOCALVAR_OCL_BUFFER(pad1_imag, double, (3));
		static double u1_real[128][256][256], u1_imag[128][256][256];
		DECLARE_LOCALVAR_OCL_BUFFER(u1_real, double, (128 * 256 * 256));
		DECLARE_LOCALVAR_OCL_BUFFER(u1_imag, double, (128 * 256 * 256));
		static double pad2_real[3], pad2_imag[3];
		DECLARE_LOCALVAR_OCL_BUFFER(pad2_real, double, (3));
		DECLARE_LOCALVAR_OCL_BUFFER(pad2_imag, double, (3));
		static double u2_real[128][256][256], u2_imag[128][256][256];
		DECLARE_LOCALVAR_OCL_BUFFER(u2_real, double, (128 * 256 * 256));
		DECLARE_LOCALVAR_OCL_BUFFER(u2_imag, double, (128 * 256 * 256));
		static double pad3_real[3], pad3_imag[3];
		DECLARE_LOCALVAR_OCL_BUFFER(pad3_real, double, (3));
		DECLARE_LOCALVAR_OCL_BUFFER(pad3_imag, double, (3));
		static int indexmap[128][256][256];
		DECLARE_LOCALVAR_OCL_BUFFER(indexmap, int, (128 * 256 * 256));
		int iter;
		int nthreads = 1;
		double total_time, mflops;
		boolean verified;
		char class;
		for (i = 0; i < 7; i++) {
			timer_clear(i);
		}
		setup();
		{

			compute_indexmap(indexmap, __ocl_buffer_indexmap,
					 dims[2], NULL);
			{

				compute_initial_conditions(u1_real,
							   __ocl_buffer_u1_real,
							   u1_imag,
							   __ocl_buffer_u1_imag,
							   dims[0], NULL);
				fft_init(dims[0][0]);
			}

			fft(1, u1_real, 
				__ocl_buffer_u1_real, 
				rd__ocl_buffer_u1_real, 
				wr__ocl_buffer_u1_real, 
				u1_imag,
			    __ocl_buffer_u1_imag, 
			    rd__ocl_buffer_u1_imag, 
			    wr__ocl_buffer_u1_imag, 
			    u0_real, 
			    __ocl_buffer_u0_real, 
			    rd__ocl_buffer_u0_real,
			    wr__ocl_buffer_u0_imag,
			    u0_imag, __ocl_buffer_u0_imag,
			    rd__ocl_buffer_u0_imag,
			    wr__ocl_buffer_u0_imag
			    );
		}
		for (i = 0; i < 7; i++) {
			timer_clear(i);
		}
		oclHostWrites(__ocl_buffer_u1_real);
		oclHostWrites(__ocl_buffer_u1_imag);
		flush_ocl_buffers();
		timer_start(0);
		if (0 == 1)
			timer_start(1);
		{
			compute_indexmap(indexmap, __ocl_buffer_indexmap,
					 dims[2], NULL);
			{
				compute_initial_conditions(u1_real,
							   __ocl_buffer_u1_real,
							   u1_imag,
							   __ocl_buffer_u1_imag,
							   dims[0], NULL);
				fft_init(dims[0][0]);
			}
			if (0 == 1) {
				timer_stop(1);
			}
			if (0 == 1) {
				timer_start(2);
			}
			fft(1, u1_real, 
					__ocl_buffer_u1_real, 
					rd__ocl_buffer_u1_real, 
					wr__ocl_buffer_u1_real, 
					u1_imag,
			    __ocl_buffer_u1_imag, 
			    rd__ocl_buffer_u1_imag, 
			    wr__ocl_buffer_u1_imag, 
			    u0_real, 
			    __ocl_buffer_u0_real,
			    rd__ocl_buffer_u0_real,
			    wr__ocl_buffer_u0_real,
			    u0_imag, __ocl_buffer_u0_imag,
			    rd__ocl_buffer_u0_imag,
			    wr__ocl_buffer_u0_imag
			    );
			if (0 == 1) {
				timer_stop(2);
			}
			for (iter = 1; iter <= niter; iter++) {
				if (0 == 1) {
					timer_start(3);
				}
				evolve(u0_real, __ocl_buffer_u0_real, u0_imag,
				       __ocl_buffer_u0_imag, u1_real,
				       __ocl_buffer_u1_real, u1_imag,
				       __ocl_buffer_u1_imag, iter, indexmap,
				       __ocl_buffer_indexmap, dims[0], NULL);
				if (0 == 1) {
					timer_stop(3);
				}
				if (0 == 1) {
					timer_start(2);
				}

				fft(-1, u1_real, 
				    __ocl_buffer_u1_real, 
				    rd__ocl_buffer_u1_real, 
				    wr__ocl_buffer_u1_real, 
				    u1_imag,
				    __ocl_buffer_u1_imag, 
				    rd__ocl_buffer_u1_imag, 
				    wr__ocl_buffer_u1_imag, 
				    u2_real,
				    __ocl_buffer_u2_real, 
				    rd__ocl_buffer_u2_real, 
				    wr__ocl_buffer_u2_real, 
				    u2_imag,
				    __ocl_buffer_u2_imag,
				    rd__ocl_buffer_u2_imag,
				    wr__ocl_buffer_u2_imag
				    );
				if (0 == 1) {
					timer_stop(2);
				}
				if (0 == 1) {
					timer_start(4);
				}
				checksum(iter, u2_real, __ocl_buffer_u2_real,
					 u2_imag, __ocl_buffer_u2_imag, dims[0],
					 NULL);
				if (0 == 1) {
					timer_stop(4);
				}
			}
			verify(256, 256, 128, niter, &verified, NULL, &class,
			       NULL);
		}
		sync_ocl_buffers();
		timer_stop(0);
		total_time = timer_read(0);
		if (total_time != 0.0) {
			mflops =
			    1.0e-6 * (double)(8388608) * (14.8157 +
							  7.19641 *
							  log((double)(8388608))
							  + (5.23518 +
							     7.21113 *
							     log((double)
								 (8388608))) *
							  niter) / total_time;
		} else {
			mflops = 0.0;
		}
		c_print_results("FT", class, 256, 256, 128, niter, nthreads,
				total_time, mflops, "          floating point",
				verified, "2.3", "08 Aug 2012", "gcc", "gcc",
				"(none)", "-I../common",
				"-std=c99 -O3 -fopenmp", "-lm -fopenmp",
				"randdp");
		if (0 == 1)
			print_timers();
	}
}

static void evolve(double u0_real[128][256][256],
		   ocl_buffer * __ocl_buffer_u0_real,
		   double u0_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_u0_imag,
		   double u1_real[128][256][256],
		   ocl_buffer * __ocl_buffer_u1_real,
		   double u1_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_u1_imag, int t,
		   int indexmap[128][256][256],
		   ocl_buffer * __ocl_buffer_indexmap, int d[3],
		   ocl_buffer * __ocl_buffer_d)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u1_real_evolve,
				 __ocl_p_u1_real_evolve, u1_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u0_real_evolve,
				 __ocl_p_u0_real_evolve, u0_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_indexmap_evolve,
				 __ocl_p_indexmap_evolve, indexmap,
				 (128 * 256 * 256), int);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u1_imag_evolve,
				 __ocl_p_u1_imag_evolve, u1_imag,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u0_imag_evolve,
				 __ocl_p_u0_imag_evolve, u0_imag,
				 (128 * 256 * 256), double);
	{
		int i, j, k;
		//--------------------------------------------------------------
		//Loop defined at line 395 of ft.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (d[0]) - (0);
			_ocl_gws[1] = (d[1]) - (0);
			_ocl_gws[2] = (d[2]) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_evolve_0, 0,
					      __ocl_buffer_u1_real);
			oclSetKernelArgBuffer(__ocl_evolve_0, 1,
					      __ocl_buffer_u0_real);
			oclSetKernelArgBuffer(__ocl_evolve_0, 2,
					      __ocl_buffer_ex);
			oclSetKernelArg(__ocl_evolve_0, 3, sizeof(int), &t);
			oclSetKernelArgBuffer(__ocl_evolve_0, 4,
					      __ocl_buffer_indexmap);
			oclSetKernelArgBuffer(__ocl_evolve_0, 5,
					      __ocl_buffer_u1_imag);
			oclSetKernelArgBuffer(__ocl_evolve_0, 6,
					      __ocl_buffer_u0_imag);
			int __ocl_i_bound = d[0];
			oclSetKernelArg(__ocl_evolve_0, 7, sizeof(int),
					&__ocl_i_bound);
			int __ocl_j_bound = d[1];
			oclSetKernelArg(__ocl_evolve_0, 8, sizeof(int),
					&__ocl_j_bound);
			int __ocl_k_bound = d[2];
			oclSetKernelArg(__ocl_evolve_0, 9, sizeof(int),
					&__ocl_k_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_u1_real);
			oclDevWrites(__ocl_buffer_u1_imag);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_u0_real);
			oclDevReads(__ocl_buffer_ex);
			oclDevReads(__ocl_buffer_indexmap);
			oclDevReads(__ocl_buffer_u0_imag);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_evolve_0, 3, _ocl_gws);
		}

	}
}

static void compute_initial_conditions(double u0_real[128][256][256],
				       ocl_buffer * __ocl_buffer_u0_real,
				       double u0_imag[128][256][256],
				       ocl_buffer * __ocl_buffer_u0_imag,
				       int d[3], ocl_buffer * __ocl_buffer_d)
{
	{
		int k;
		double x0, start, an, dummy;
		static double tmp[131073];
		DECLARE_LOCALVAR_OCL_BUFFER(tmp, double, (131073));
		int i, j, t;
		start = 314159265.0;
		ipow46(1220703125.0,
		       (zstart[0] - 1) * 2 * 256 * 256 + (ystart[0] -
							  1) * 2 * 256, &an,
		       NULL);
		dummy = myrandlc(&start, NULL, an);
		ipow46(1220703125.0, 2 * 256 * 256, &an, NULL);
		for (k = 0; k < dims[0][2]; k++) {
			x0 = start;

			myvranlc(2 * 256 * dims[0][1], &x0, NULL, 1220703125.0,
				 tmp, __ocl_buffer_tmp);
			t = 1;
			for (j = 0; j < dims[0][1]; j++)
				for (i = 0; i < 256; i++) {
					u0_real[k][j][i] = tmp[t++];
					u0_imag[k][j][i] = tmp[t++];
				}
			if (k != dims[0][2])
				dummy = myrandlc(&start, NULL, an);
		}
	}
}

static void ipow46(double a, int exponent, double *result,
		   ocl_buffer * __ocl_buffer_result)
{
	{
		double dummy, q, r;
		int n, n2;
		*result = 1;
		if (exponent == 0)
			return;
		q = a;
		r = 1;
		n = exponent;
		while (n > 1) {
			n2 = n / 2;
			if (n2 * 2 == n) {
				dummy = myrandlc(&q, NULL, q);
				n = n2;
			} else {
				dummy = myrandlc(&r, NULL, q);
				n = n - 1;
			}
		}
		dummy = myrandlc(&r, NULL, q);
		*result = r;
	}
}

static void setup()
{
	int ierr, i, j, fstatus;
	printf
	    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - FT Benchmark\n\n");
	niter = 6;
	printf(" Size                : %3dx%3dx%3d\n", 256, 256, 128);
	printf(" Iterations          :     %7d\n", niter);
	for (i = 0; i < 3; i++) {
		dims[i][0] = 256;
		dims[i][1] = 256;
		dims[i][2] = 128;
	}
	for (i = 0; i < 3; i++) {
		xstart[i] = 1;
		xend[i] = 256;
		ystart[i] = 1;
		yend[i] = 256;
		zstart[i] = 1;
		zend[i] = 128;
	}
	fftblock = 16;
	fftblockpad = 18;
	if (fftblock != 16)
		fftblockpad = fftblock + 3;
}

static void compute_indexmap(int indexmap[128][256][256],
			     ocl_buffer * __ocl_buffer_indexmap, int d[3],
			     ocl_buffer * __ocl_buffer_d)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_indexmap_compute_indexmap,
				 __ocl_p_indexmap_compute_indexmap, indexmap,
				 (128 * 256 * 256), int);
	{
		int i, j, k, ii, ii2, jj, ij2, kk;
		double ap;
		int xstart_i, ystart_i, zstart_i;
		xstart_i = xstart[2];
		ystart_i = ystart[2];
		zstart_i = zstart[2];
		//--------------------------------------------------------------
		//Loop defined at line 579 of ft.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (dims[2][0]) - (0);
			_ocl_gws[1] = (dims[2][1]) - (0);
			_ocl_gws[2] = (dims[2][2]) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_compute_indexmap_0, 0,
					sizeof(int), &xstart_i);
			oclSetKernelArg(__ocl_compute_indexmap_0, 1,
					sizeof(int), &ystart_i);
			oclSetKernelArg(__ocl_compute_indexmap_0, 2,
					sizeof(int), &zstart_i);
			oclSetKernelArgBuffer(__ocl_compute_indexmap_0, 3,
					      __ocl_buffer_indexmap);
			int __ocl_i_bound = dims[2][0];
			oclSetKernelArg(__ocl_compute_indexmap_0, 4,
					sizeof(int), &__ocl_i_bound);
			int __ocl_j_bound = dims[2][1];
			oclSetKernelArg(__ocl_compute_indexmap_0, 5,
					sizeof(int), &__ocl_j_bound);
			int __ocl_k_bound = dims[2][2];
			oclSetKernelArg(__ocl_compute_indexmap_0, 6,
					sizeof(int), &__ocl_k_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_indexmap);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_compute_indexmap_0, 3, _ocl_gws);
		}

		{
			ap = -4.0 * 1.0e-6 * 3.141592653589793238 *
			    3.141592653589793238;
			ex[0] = 1.0;
			ex[1] = exp(ap);
			for (i = 2;
			     i <=
			     (6 *
			      (256 * 256 / 4 + 256 * 256 / 4 + 128 * 128 / 4));
			     i++) {
				ex[i] = ex[i - 1] * ex[1];
			}
		}
	}
}

static void print_timers()
{
	int i;
	char *tstrings[7] =
	    { "          total ", "          setup ", "            fft ",
    "         evolve ", "       checksum ", "         fftlow ",
    "        fftcopy " };
	DECLARE_LOCALVAR_OCL_BUFFER(tstrings, char, (7));
	for (i = 0; i < 7; i++) {
		if (timer_read(i) != 0.0) {
			printf("timer %2d(%16s( :%10.6f\n", i, tstrings[i],
			       timer_read(i));
		}
	}
}

static void fft(int dir, double x1_real[128][256][256],
		ocl_buffer * __ocl_buffer_x1_real,
		ocl_buffer * rd__ocl_buffer_x1_real,
		ocl_buffer * wr__ocl_buffer_x1_real,
		double x1_imag[128][256][256],
		ocl_buffer * __ocl_buffer_x1_imag,
		ocl_buffer * rd__ocl_buffer_x1_imag,
		ocl_buffer * wr__ocl_buffer_x1_imag,
		double x2_real[128][256][256],
		ocl_buffer * __ocl_buffer_x2_real,
		ocl_buffer * rd__ocl_buffer_x2_real,
		ocl_buffer * wr__ocl_buffer_x2_real,
		double x2_imag[128][256][256],
		ocl_buffer * __ocl_buffer_x2_imag,
		ocl_buffer * rd__ocl_buffer_x2_imag,
		ocl_buffer * wr__ocl_buffer_x2_imag)
{
	{
		if (dir == 1) {
			cffts1(1, dims[0], NULL, x1_real, 
				__ocl_buffer_x1_real,
				rd__ocl_buffer_x1_real,
				wr__ocl_buffer_x1_real,
			       x1_imag, 
			       __ocl_buffer_x1_imag, 
			       rd__ocl_buffer_x1_imag, 
			       wr__ocl_buffer_x1_imag, 
			       x1_real,
			       __ocl_buffer_x1_real, 
			       rd__ocl_buffer_x1_real, 
			       wr__ocl_buffer_x1_real, 
			       x1_imag,
			       __ocl_buffer_x1_imag,
			       rd__ocl_buffer_x1_imag,
			       wr__ocl_buffer_x1_imag
			       );
			cffts2(1, dims[1], NULL, x1_real, __ocl_buffer_x1_real,
			       x1_imag, __ocl_buffer_x1_imag, x1_real,
			       __ocl_buffer_x1_real, x1_imag,
			       __ocl_buffer_x1_imag);
			cffts3(1, dims[2], NULL, x1_real, __ocl_buffer_x1_real,
			       x1_imag, __ocl_buffer_x1_imag, x2_real,
			       __ocl_buffer_x2_real, x2_imag,
			       __ocl_buffer_x2_imag);
		} else {
			cffts3(-1, dims[2], NULL, x1_real, __ocl_buffer_x1_real,
			       x1_imag, __ocl_buffer_x1_imag, x1_real,
			       __ocl_buffer_x1_real, x1_imag,
			       __ocl_buffer_x1_imag);
			cffts2(-1, dims[1], NULL, x1_real, __ocl_buffer_x1_real,
			       x1_imag, __ocl_buffer_x1_imag, x1_real,
			       __ocl_buffer_x1_real, x1_imag,
			       __ocl_buffer_x1_imag);
			cffts1(-1, dims[0], NULL, x1_real, 
					__ocl_buffer_x1_real,
					rd__ocl_buffer_x1_real,
					wr__ocl_buffer_x1_real,
			       x1_imag, 
			       __ocl_buffer_x1_imag, 
			       rd__ocl_buffer_x1_imag, 
			       wr__ocl_buffer_x1_imag, 
			       x2_real,
			       __ocl_buffer_x2_real, 
			       rd__ocl_buffer_x2_real, 
			       wr__ocl_buffer_x2_real, 
			       x2_imag,
			       __ocl_buffer_x2_imag,
			       rd__ocl_buffer_x2_imag,
			       wr__ocl_buffer_x2_imag
			       );
		}
	}
}

static void cffts1(int is, int d[3], 
		  ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   ocl_buffer * rd__ocl_buffer_x_real,
		   ocl_buffer * wr__ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   ocl_buffer * rd__ocl_buffer_x_imag,
		   ocl_buffer * wr__ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   ocl_buffer * rd__ocl_buffer_xout_real,
		   ocl_buffer * wr__ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag,
		   ocl_buffer * rd__ocl_buffer_xout_imag,
		   ocl_buffer * wr__ocl_buffer_xout_imag
		   )
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_d_cffts1, __ocl_p_d_cffts1, d,
				 (3), int);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_real_cffts1,
				 __ocl_p_x_real_cffts1, x_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_imag_cffts1,
				 __ocl_p_x_imag_cffts1, x_imag,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_real_cffts1,
				 __ocl_p_xout_real_cffts1, xout_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_imag_cffts1,
				 __ocl_p_xout_imag_cffts1, xout_imag,
				 (128 * 256 * 256), double);
	{
		int logd[3];
		DECLARE_LOCALVAR_OCL_BUFFER(logd, int, (3));
		int i, j, k, jj;
		for (i = 0; i < 3; i++) {
			logd[i] = ilog2(d[i]);
		}
		int logd_0 = logd[0];
		//--------------------------------------------------------------
		//Loop defined at line 690 of ft.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (d[1] - fftblock) - (0) + 1;
			OCL_NEAREST_MULTD(_ocl_gws[0], (size_t) fftblock);
			_ocl_gws[1] = (d[2]) - (0);
			

			oclGetWorkSize(2, _ocl_gws, NULL);
			size_t _ocl_thread_num = (_ocl_gws[0] * _ocl_gws[1]);
			oclSetKernelArg(__ocl_cffts1_0, 0, sizeof(int),
					&fftblock);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 1,
					      __ocl_buffer_d);
			
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_real_cffts1_0,
						  __ocl_buffer_yy0_real_cffts1_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 2, __ocl_buffer_yy0_real_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 3, rd__ocl_buffer_yy0_real_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 4, wr__ocl_buffer_yy0_real_cffts1_0);
			
			oclSetKernelArgBuffer(__ocl_cffts1_0, 5, __ocl_buffer_x_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 6, rd__ocl_buffer_x_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 7, wr__ocl_buffer_x_real);
			
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_imag_cffts1_0,
						  __ocl_buffer_yy0_imag_cffts1_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 8, __ocl_buffer_yy0_imag_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 9, rd__ocl_buffer_yy0_imag_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 10, wr__ocl_buffer_yy0_imag_cffts1_0);
			
			oclSetKernelArgBuffer(__ocl_cffts1_0, 11, __ocl_buffer_x_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 12, rd__ocl_buffer_x_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 13, wr__ocl_buffer_x_imag);
			
			oclSetKernelArg(__ocl_cffts1_0, 14, sizeof(int), &is);
			oclSetKernelArg(__ocl_cffts1_0, 15, sizeof(int), &logd_0);

			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_real_cffts1_0,
						  __ocl_buffer_yy1_real_cffts1_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 16, __ocl_buffer_yy1_real_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 17, rd__ocl_buffer_yy1_real_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 18, wr__ocl_buffer_yy1_real_cffts1_0);

			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_imag_cffts1_0,
						  __ocl_buffer_yy1_imag_cffts1_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 19, __ocl_buffer_yy1_imag_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 20, rd__ocl_buffer_yy1_imag_cffts1_0);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 21, wr__ocl_buffer_yy1_imag_cffts1_0);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 22, __ocl_buffer_xout_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 23, rd__ocl_buffer_xout_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 24, wr__ocl_buffer_xout_real);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 25, __ocl_buffer_xout_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 26, rd__ocl_buffer_xout_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 27, wr__ocl_buffer_xout_imag);

			oclSetKernelArgBuffer(__ocl_cffts1_0, 28, __ocl_buffer_u_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 29, rd_oclb_u_real);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 30, wr_oclb_u_real);
			
			oclSetKernelArgBuffer(__ocl_cffts1_0, 31, __ocl_buffer_u_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 32, rd_oclb_u_imag);
			oclSetKernelArgBuffer(__ocl_cffts1_0, 33, wr_oclb_u_imag);
			
			oclSetKernelArg(__ocl_cffts1_0, 34, sizeof(int),
					&fftblockpad);
			int __ocl_jj_inc = fftblock;
			oclSetKernelArg(__ocl_cffts1_0, 35, sizeof(int),
					&__ocl_jj_inc);
			int __ocl_jj_bound = d[1] - fftblock;
			oclSetKernelArg(__ocl_cffts1_0, 36, sizeof(int),
					&__ocl_jj_bound);
			int __ocl_k_bound = d[2];
			oclSetKernelArg(__ocl_cffts1_0, 37, sizeof(int),
					&__ocl_k_bound);
			
			oclSetKernelArgBuffer(__ocl_cffts1_0, 38, __oclb_gpu_tls_conflict_flag);
			
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_d);
			oclDevWrites(__ocl_buffer_xout_real);
			oclDevWrites(__ocl_buffer_xout_imag);
			oclDevWrites(__ocl_buffer_u_real);
			oclDevWrites(__ocl_buffer_u_imag);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_x_real);
			oclDevReads(__ocl_buffer_x_imag);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_cffts1_0, 2, _ocl_gws);
#ifdef OCL_RELEASE_GTP_BUFFERS_IMMEDIATE
			oclSync();
			
			//oclReleaseBuffer(wr__ocl_buffer_yy0_imag_cffts1_0);
			
			//oclReleaseBuffer(rd__ocl_buffer_x_real);
			//oclReleaseBuffer(wr__ocl_buffer_x_real);

			//oclReleaseBuffer(rd__ocl_buffer_yy1_real_cffts1_0);
			//oclReleaseBuffer(wr__ocl_buffer_yy1_real_cffts1_0);

			if (__ocl_th_yy0_real_cffts1_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_real_cffts1_0);
				free(__ocl_th_yy0_real_cffts1_0);
				__ocl_th_yy0_real_cffts1_0 = NULL;
				__ocl_buffer_yy0_real_cffts1_0 = NULL;
				__ocl_th_yy0_real_cffts1_0_length = 0;
			}
			if (__ocl_th_yy0_imag_cffts1_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_imag_cffts1_0);
				//free(__ocl_th_yy0_imag_cffts1_0);
				__ocl_th_yy0_imag_cffts1_0 = NULL;
				__ocl_buffer_yy0_imag_cffts1_0 = NULL;
				__ocl_th_yy0_imag_cffts1_0_length = 0;
			}
			if (__ocl_th_yy1_real_cffts1_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_real_cffts1_0);
				free(__ocl_th_yy1_real_cffts1_0);
				__ocl_th_yy1_real_cffts1_0 = NULL;
				__ocl_buffer_yy1_real_cffts1_0 = NULL;
				__ocl_th_yy1_real_cffts1_0_length = 0;
			}
			if (__ocl_th_yy1_imag_cffts1_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_imag_cffts1_0);
				free(__ocl_th_yy1_imag_cffts1_0);
				__ocl_th_yy1_imag_cffts1_0 = NULL;
				__ocl_buffer_yy1_imag_cffts1_0 = NULL;
				__ocl_th_yy1_imag_cffts1_0_length = 0;
			}
#endif
		}

	}
}

static void cffts2(int is, int d[3], ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_d_cffts2, __ocl_p_d_cffts2, d,
				 (3), int);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_real_cffts2,
				 __ocl_p_x_real_cffts2, x_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_imag_cffts2,
				 __ocl_p_x_imag_cffts2, x_imag,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_real_cffts2,
				 __ocl_p_xout_real_cffts2, xout_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_imag_cffts2,
				 __ocl_p_xout_imag_cffts2, xout_imag,
				 (128 * 256 * 256), double);
	{
		int logd[3];
		DECLARE_LOCALVAR_OCL_BUFFER(logd, int, (3));
		int i, j, k, ii;
		for (i = 0; i < 3; i++) {
			logd[i] = ilog2(d[i]);
		}
		int logd_1 = logd[1];
		//--------------------------------------------------------------
		//Loop defined at line 738 of ft.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (d[0] - fftblock) - (0) + 1;
			OCL_NEAREST_MULTD(_ocl_gws[0], (size_t) fftblock);
			_ocl_gws[1] = (d[2]) - (0);

			oclGetWorkSize(2, _ocl_gws, NULL);
			size_t _ocl_thread_num = (_ocl_gws[0] * _ocl_gws[1]);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 0,
					      __ocl_buffer_d);
			oclSetKernelArg(__ocl_cffts2_0, 1, sizeof(int),
					&fftblock);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_real_cffts2_0,
						  __ocl_buffer_yy0_real_cffts2_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 2,
					      __ocl_buffer_yy0_real_cffts2_0);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 3,
					      __ocl_buffer_x_real);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_imag_cffts2_0,
						  __ocl_buffer_yy0_imag_cffts2_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 4,
					      __ocl_buffer_yy0_imag_cffts2_0);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 5,
					      __ocl_buffer_x_imag);
			oclSetKernelArg(__ocl_cffts2_0, 6, sizeof(int), &is);
			oclSetKernelArg(__ocl_cffts2_0, 7, sizeof(int),
					&logd_1);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_real_cffts2_0,
						  __ocl_buffer_yy1_real_cffts2_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 8,
					      __ocl_buffer_yy1_real_cffts2_0);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_imag_cffts2_0,
						  __ocl_buffer_yy1_imag_cffts2_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 9,
					      __ocl_buffer_yy1_imag_cffts2_0);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 10,
					      __ocl_buffer_xout_real);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 11,
					      __ocl_buffer_xout_imag);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 12,
					      __ocl_buffer_u_real);
			oclSetKernelArgBuffer(__ocl_cffts2_0, 13,
					      __ocl_buffer_u_imag);
			oclSetKernelArg(__ocl_cffts2_0, 14, sizeof(int),
					&fftblockpad);
			int __ocl_ii_inc = fftblock;
			oclSetKernelArg(__ocl_cffts2_0, 15, sizeof(int),
					&__ocl_ii_inc);
			int __ocl_ii_bound = d[0] - fftblock;
			oclSetKernelArg(__ocl_cffts2_0, 16, sizeof(int),
					&__ocl_ii_bound);
			int __ocl_k_bound = d[2];
			oclSetKernelArg(__ocl_cffts2_0, 17, sizeof(int),
					&__ocl_k_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_d);
			oclDevWrites(__ocl_buffer_xout_real);
			oclDevWrites(__ocl_buffer_xout_imag);
			oclDevWrites(__ocl_buffer_u_real);
			oclDevWrites(__ocl_buffer_u_imag);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_x_real);
			oclDevReads(__ocl_buffer_x_imag);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_cffts2_0, 2, _ocl_gws);
#ifdef OCL_RELEASE_GTP_BUFFERS_IMMEDIATE
			oclSync();
			if (__ocl_th_yy0_real_cffts2_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_real_cffts2_0);
				free(__ocl_th_yy0_real_cffts2_0);
				__ocl_th_yy0_real_cffts2_0 = NULL;
				__ocl_buffer_yy0_real_cffts2_0 = NULL;
				__ocl_th_yy0_real_cffts2_0_length = 0;
			}
			if (__ocl_th_yy0_imag_cffts2_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_imag_cffts2_0);
				free(__ocl_th_yy0_imag_cffts2_0);
				__ocl_th_yy0_imag_cffts2_0 = NULL;
				__ocl_buffer_yy0_imag_cffts2_0 = NULL;
				__ocl_th_yy0_imag_cffts2_0_length = 0;
			}
			if (__ocl_th_yy1_real_cffts2_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_real_cffts2_0);
				free(__ocl_th_yy1_real_cffts2_0);
				__ocl_th_yy1_real_cffts2_0 = NULL;
				__ocl_buffer_yy1_real_cffts2_0 = NULL;
				__ocl_th_yy1_real_cffts2_0_length = 0;
			}
			if (__ocl_th_yy1_imag_cffts2_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_imag_cffts2_0);
				free(__ocl_th_yy1_imag_cffts2_0);
				__ocl_th_yy1_imag_cffts2_0 = NULL;
				__ocl_buffer_yy1_imag_cffts2_0 = NULL;
				__ocl_th_yy1_imag_cffts2_0_length = 0;
			}
#endif
		}

	}
}

static void cffts3(int is, int d[3], ocl_buffer * __ocl_buffer_d,
		   double x_real[128][256][256],
		   ocl_buffer * __ocl_buffer_x_real,
		   double x_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_x_imag,
		   double xout_real[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_real,
		   double xout_imag[128][256][256],
		   ocl_buffer * __ocl_buffer_xout_imag)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_d_cffts3, __ocl_p_d_cffts3, d,
				 (3), int);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_real_cffts3,
				 __ocl_p_x_real_cffts3, x_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_x_imag_cffts3,
				 __ocl_p_x_imag_cffts3, x_imag,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_real_cffts3,
				 __ocl_p_xout_real_cffts3, xout_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_xout_imag_cffts3,
				 __ocl_p_xout_imag_cffts3, xout_imag,
				 (128 * 256 * 256), double);
	{
		int logd[3];
		DECLARE_LOCALVAR_OCL_BUFFER(logd, int, (3));
		int i, j, k, ii;
		for (i = 0; i < 3; i++) {
			logd[i] = ilog2(d[i]);
		}
		int logd_2 = logd[2];
		//--------------------------------------------------------------
		//Loop defined at line 785 of ft.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (d[0] - fftblock) - (0) + 1;
			OCL_NEAREST_MULTD(_ocl_gws[0], (size_t) fftblock);
			_ocl_gws[1] = (d[1]) - (0);

			oclGetWorkSize(2, _ocl_gws, NULL);
			size_t _ocl_thread_num = (_ocl_gws[0] * _ocl_gws[1]);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 0,
					      __ocl_buffer_d);
			oclSetKernelArg(__ocl_cffts3_0, 1, sizeof(int),
					&fftblock);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_real_cffts3_0,
						  __ocl_buffer_yy0_real_cffts3_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 2,
					      __ocl_buffer_yy0_real_cffts3_0);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 3,
					      __ocl_buffer_x_real);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy0_imag_cffts3_0,
						  __ocl_buffer_yy0_imag_cffts3_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 4,
					      __ocl_buffer_yy0_imag_cffts3_0);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 5,
					      __ocl_buffer_x_imag);
			oclSetKernelArg(__ocl_cffts3_0, 6, sizeof(int), &is);
			oclSetKernelArg(__ocl_cffts3_0, 7, sizeof(int),
					&logd_2);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_real_cffts3_0,
						  __ocl_buffer_yy1_real_cffts3_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 8,
					      __ocl_buffer_yy1_real_cffts3_0);
			CREATE_THREAD_PRIVATE_BUF(__ocl_th_yy1_imag_cffts3_0,
						  __ocl_buffer_yy1_imag_cffts3_0,
						  double,
						  (4608 * _ocl_thread_num),
						  DEFAULT_ALIGN_SIZE);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 9,
					      __ocl_buffer_yy1_imag_cffts3_0);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 10,
					      __ocl_buffer_xout_real);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 11,
					      __ocl_buffer_xout_imag);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 12,
					      __ocl_buffer_u_real);
			oclSetKernelArgBuffer(__ocl_cffts3_0, 13,
					      __ocl_buffer_u_imag);
			oclSetKernelArg(__ocl_cffts3_0, 14, sizeof(int),
					&fftblockpad);
			int __ocl_ii_inc = fftblock;
			oclSetKernelArg(__ocl_cffts3_0, 15, sizeof(int),
					&__ocl_ii_inc);
			int __ocl_ii_bound = d[0] - fftblock;
			oclSetKernelArg(__ocl_cffts3_0, 16, sizeof(int),
					&__ocl_ii_bound);
			int __ocl_j_bound = d[1];
			oclSetKernelArg(__ocl_cffts3_0, 17, sizeof(int),
					&__ocl_j_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_d);
			oclDevWrites(__ocl_buffer_xout_real);
			oclDevWrites(__ocl_buffer_xout_imag);
			oclDevWrites(__ocl_buffer_u_real);
			oclDevWrites(__ocl_buffer_u_imag);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_x_real);
			oclDevReads(__ocl_buffer_x_imag);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_cffts3_0, 2, _ocl_gws);
#ifdef OCL_RELEASE_GTP_BUFFERS_IMMEDIATE
			oclSync();
			if (__ocl_th_yy0_real_cffts3_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_real_cffts3_0);
				free(__ocl_th_yy0_real_cffts3_0);
				__ocl_th_yy0_real_cffts3_0 = NULL;
				__ocl_buffer_yy0_real_cffts3_0 = NULL;
				__ocl_th_yy0_real_cffts3_0_length = 0;
			}
			if (__ocl_th_yy0_imag_cffts3_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy0_imag_cffts3_0);
				free(__ocl_th_yy0_imag_cffts3_0);
				__ocl_th_yy0_imag_cffts3_0 = NULL;
				__ocl_buffer_yy0_imag_cffts3_0 = NULL;
				__ocl_th_yy0_imag_cffts3_0_length = 0;
			}
			if (__ocl_th_yy1_real_cffts3_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_real_cffts3_0);
				free(__ocl_th_yy1_real_cffts3_0);
				__ocl_th_yy1_real_cffts3_0 = NULL;
				__ocl_buffer_yy1_real_cffts3_0 = NULL;
				__ocl_th_yy1_real_cffts3_0_length = 0;
			}
			if (__ocl_th_yy1_imag_cffts3_0_length) {
				oclReleaseBuffer
				    (__ocl_buffer_yy1_imag_cffts3_0);
				free(__ocl_th_yy1_imag_cffts3_0);
				__ocl_th_yy1_imag_cffts3_0 = NULL;
				__ocl_buffer_yy1_imag_cffts3_0 = NULL;
				__ocl_th_yy1_imag_cffts3_0_length = 0;
			}
#endif
		}

	}
}

static void fft_init(int n)
{
	int m, nu, ku, i, j, ln;
	double t, ti;
	nu = n;
	m = ilog2(n);
	u_real[0] = (double)m;
	u_imag[0] = 0.0;
	ku = 1;
	ln = 1;
	for (j = 1; j <= m; j++) {
		t = 3.141592653589793238 / ln;
		for (i = 0; i <= ln - 1; i++) {
			ti = i * t;
			u_real[i + ku] = cos(ti);
			u_imag[i + ku] = sin(ti);
		}
		ku = ku + ln;
		ln = 2 * ln;
	}
}

static void cfftz(int is, int m, int n, double x_real[256][18],
		  ocl_buffer * __ocl_buffer_x_real, double x_imag[256][18],
		  ocl_buffer * __ocl_buffer_x_imag, double y_real[256][18],
		  ocl_buffer * __ocl_buffer_y_real, double y_imag[256][18],
		  ocl_buffer * __ocl_buffer_y_imag)
{
	{
		int i, j, l, mx;
		mx = (int)(u_real[0]);
		for (l = 1; l <= m; l += 2) {
			fftz2(is, l, m, n, fftblock, fftblockpad, u_real,
			      __ocl_buffer_u_real, u_imag, __ocl_buffer_u_imag,
			      x_real, __ocl_buffer_x_real, x_imag,
			      __ocl_buffer_x_imag, y_real, __ocl_buffer_y_real,
			      y_imag, __ocl_buffer_y_imag);
			if (l == m)
				break;
			fftz2(is, l + 1, m, n, fftblock, fftblockpad, u_real,
			      __ocl_buffer_u_real, u_imag, __ocl_buffer_u_imag,
			      y_real, __ocl_buffer_y_real, y_imag,
			      __ocl_buffer_y_imag, x_real, __ocl_buffer_x_real,
			      x_imag, __ocl_buffer_x_imag);
		}
		if (m % 2 == 1) {
			for (j = 0; j < n; j++) {
				for (i = 0; i < fftblock; i++) {
					x_real[j][i] = y_real[j][i];
					x_imag[j][i] = y_imag[j][i];
				}
			}
		}
	}
}

static void fftz2(int is, int l, int m, int n, int ny, int ny1,
		  double u_real[256], ocl_buffer * __ocl_buffer_u_real,
		  double u_imag[256], ocl_buffer * __ocl_buffer_u_imag,
		  double x_real[256][18], ocl_buffer * __ocl_buffer_x_real,
		  double x_imag[256][18], ocl_buffer * __ocl_buffer_x_imag,
		  double y_real[256][18], ocl_buffer * __ocl_buffer_y_real,
		  double y_imag[256][18], ocl_buffer * __ocl_buffer_y_imag)
{
	{
		int k, n1, li, lj, lk, ku, i, j, i11, i12, i21, i22;
		double u1_real, x11_real, x21_real;
		double u1_imag, x11_imag, x21_imag;
		n1 = n / 2;
		if (l - 1 == 0) {
			lk = 1;
		} else {
			lk = 2 << ((l - 1) - 1);
		}
		if (m - l == 0) {
			li = 1;
		} else {
			li = 2 << ((m - l) - 1);
		}
		lj = 2 * lk;
		ku = li;
		for (i = 0; i < li; i++) {
			i11 = i * lk;
			i12 = i11 + n1;
			i21 = i * lj;
			i22 = i21 + lk;
			if (is >= 1) {
				u1_real = u_real[ku + i];
				u1_imag = u_imag[ku + i];
			} else {
				u1_real = u_real[ku + i];
				u1_imag = -u_imag[ku + i];
			}
			for (k = 0; k < lk; k++) {
				for (j = 0; j < ny; j++) {
					double x11real, x11imag;
					double x21real, x21imag;
					x11real = x_real[i11 + k][j];
					x11imag = x_imag[i11 + k][j];
					x21real = x_real[i12 + k][j];
					x21imag = x_imag[i12 + k][j];
					y_real[i21 + k][j] = x11real + x21real;
					y_imag[i21 + k][j] = x11imag + x21imag;
					y_real[i22 + k][j] =
					    u1_real * (x11real - x21real) -
					    u1_imag * (x11imag - x21imag);
					y_imag[i22 + k][j] =
					    u1_real * (x11imag - x21imag) +
					    u1_imag * (x11real - x21real);
				}
			}
		}
	}
}

static int ilog2(int n)
{
	int nn, lg;
	if (n == 1) {
		return 0;
	}
	lg = 1;
	nn = 2;
	while (nn < n) {
		nn = nn << 1;
		lg++;
	}
	return lg;
}

static void checksum(int i, double u1_real[128][256][256],
		     ocl_buffer * __ocl_buffer_u1_real,
		     double u1_imag[128][256][256],
		     ocl_buffer * __ocl_buffer_u1_imag, int d[3],
		     ocl_buffer * __ocl_buffer_d)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u1_real_checksum,
				 __ocl_p_u1_real_checksum, u1_real,
				 (128 * 256 * 256), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_u1_imag_checksum,
				 __ocl_p_u1_imag_checksum, u1_imag,
				 (128 * 256 * 256), double);
	{
		int j, q, r, s, ierr;
		double _chk_real, _allchk_real;
		double _chk_imag, _allchk_imag;
		_chk_real = 0.0;
		_chk_imag = 0.0;
		{
			double chk_real = _chk_real;
			double chk_imag = _chk_imag;
			//--------------------------------------------------------------
			//Loop defined at line 1019 of ft.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//Reduction step 1
				//------------------------------------------
				size_t _ocl_gws[1];
				_ocl_gws[0] = (1024) - (1) + 1;

				oclGetWorkSize(1, _ocl_gws, NULL);
				size_t __ocl_act_buf_size = (_ocl_gws[0]);
				REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: chk_real
				CREATE_REDUCTION_STEP1_BUFFER
				    (__ocl_buffer_chk_real_checksum_0_size,
				     __ocl_buf_size,
				     __ocl_buffer_chk_real_checksum_0, double);
//Prepare buffer for the reduction variable: chk_imag
				CREATE_REDUCTION_STEP1_BUFFER
				    (__ocl_buffer_chk_imag_checksum_0_size,
				     __ocl_buf_size,
				     __ocl_buffer_chk_imag_checksum_0, double);

				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
				if (__ocl_buf_size > __ocl_act_buf_size) {
					oclSetKernelArgBuffer
					    (__ocl_checksum_0_reduction_step0,
					     0,
					     __ocl_buffer_chk_real_checksum_0);
					oclSetKernelArgBuffer
					    (__ocl_checksum_0_reduction_step0,
					     1,
					     __ocl_buffer_chk_imag_checksum_0);
					unsigned int __ocl_buffer_offset =
					    __ocl_buf_size - __ocl_act_buf_size;
					oclSetKernelArg
					    (__ocl_checksum_0_reduction_step0,
					     2, sizeof(unsigned int),
					     &__ocl_act_buf_size);
					oclSetKernelArg
					    (__ocl_checksum_0_reduction_step0,
					     3, sizeof(unsigned int),
					     &__ocl_buffer_offset);

					size_t __offset_work_size[1] =
					    { __ocl_buffer_offset };
					oclRunKernel
					    (__ocl_checksum_0_reduction_step0,
					     1, __offset_work_size);
				}

				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 0,
				     __ocl_buffer_xstart);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 1,
				     __ocl_buffer_xend);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 2,
				     __ocl_buffer_ystart);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 3,
				     __ocl_buffer_yend);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 4,
				     __ocl_buffer_zstart);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 5,
				     __ocl_buffer_zend);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 6,
				     __ocl_buffer_u1_real);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 7,
				     __ocl_buffer_u1_imag);
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 8,
				     __ocl_buffer_chk_real_checksum_0);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step1, 9,
				     __ocl_buffer_chk_imag_checksum_0);
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
				oclDevReads(__ocl_buffer_xstart);
				oclDevReads(__ocl_buffer_xend);
				oclDevReads(__ocl_buffer_ystart);
				oclDevReads(__ocl_buffer_yend);
				oclDevReads(__ocl_buffer_zstart);
				oclDevReads(__ocl_buffer_zend);
				oclDevReads(__ocl_buffer_u1_real);
				oclDevReads(__ocl_buffer_u1_imag);
				//------------------------------------------
				//Read only buffers (END) 
				//------------------------------------------

				oclRunKernel(__ocl_checksum_0_reduction_step1,
					     1, _ocl_gws);

//Reduction Step 2
				unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
				CREATE_REDUCTION_STEP2_BUFFER
				    (__ocl_output_chk_real_checksum_0_size,
				     __ocl_num_block, 16,
				     __ocl_output_buffer_chk_real_checksum_0,
				     __ocl_output_chk_real_checksum_0, double);
				CREATE_REDUCTION_STEP2_BUFFER
				    (__ocl_output_chk_imag_checksum_0_size,
				     __ocl_num_block, 16,
				     __ocl_output_buffer_chk_imag_checksum_0,
				     __ocl_output_chk_imag_checksum_0, double);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step2, 0,
				     __ocl_buffer_chk_real_checksum_0);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step2, 1,
				     __ocl_output_buffer_chk_real_checksum_0);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step2, 2,
				     __ocl_buffer_chk_imag_checksum_0);
				oclSetKernelArgBuffer
				    (__ocl_checksum_0_reduction_step2, 3,
				     __ocl_output_buffer_chk_imag_checksum_0);

				oclDevWrites
				    (__ocl_output_buffer_chk_real_checksum_0);
				oclDevWrites
				    (__ocl_output_buffer_chk_imag_checksum_0);

				size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
				size_t __ocl_localThreads[] = { GROUP_SIZE };

				oclRunKernelL(__ocl_checksum_0_reduction_step2,
					      1, __ocl_globalThreads,
					      __ocl_localThreads);

//Do the final reduction part on the CPU
				oclHostReads
				    (__ocl_output_buffer_chk_real_checksum_0);
				oclHostReads
				    (__ocl_output_buffer_chk_imag_checksum_0);
				oclSync();

				for (unsigned __ocl_i = 0;
				     __ocl_i < __ocl_num_block; __ocl_i++) {
					chk_real =
					    chk_real +
					    __ocl_output_chk_real_checksum_0
					    [__ocl_i];
					chk_imag =
					    chk_imag +
					    __ocl_output_chk_imag_checksum_0
					    [__ocl_i];
				}

			}

			_chk_real = chk_real;
			_chk_imag = chk_imag;
		}
		{
			sums_real[i] += _chk_real;
			sums_imag[i] += _chk_imag;
		}
		{
			sums_real[i] = sums_real[i] / (double)(8388608);
			sums_imag[i] = sums_imag[i] / (double)(8388608);
			printf("T = %5d     Checksum = %22.12e %22.12e\n", i,
			       sums_real[i], sums_imag[i]);
		}
	}
}

static void verify(int d1, int d2, int d3, int nt, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified, char *class,
		   ocl_buffer * __ocl_buffer_class)
{
	{
		int ierr, size, i;
		double err, epsilon;
		double vdata_real_s[7] =
		    { 0.0, 5.546087004964e+02, 5.546385409189e+02,
       5.546148406171e+02, 5.545423607415e+02, 5.544255039624e+02,
       5.542683411902e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_real_s, double, (7));
		double vdata_imag_s[7] =
		    { 0.0, 4.845363331978e+02, 4.865304269511e+02,
       4.883910722336e+02, 4.901273169046e+02, 4.917475857993e+02,
       4.932597244941e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_imag_s, double, (7));
		double vdata_real_w[7] =
		    { 0.0, 5.673612178944e+02, 5.631436885271e+02,
       5.594024089970e+02, 5.560698047020e+02, 5.530898991250e+02,
       5.504159734538e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_real_w, double, (7));
		double vdata_imag_w[7] =
		    { 0.0, 5.293246849175e+02, 5.282149986629e+02,
       5.270996558037e+02, 5.260027904925e+02, 5.249400845633e+02,
       5.239212247086e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_imag_w, double, (7));
		double vdata_real_a[7] =
		    { 0.0, 5.046735008193e+02, 5.059412319734e+02,
       5.069376896287e+02, 5.077892868474e+02, 5.085233095391e+02,
       5.091487099959e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_real_a, double, (7));
		double vdata_imag_a[7] =
		    { 0.0, 5.114047905510e+02, 5.098809666433e+02,
       5.098144042213e+02, 5.101336130759e+02, 5.104914655194e+02,
       5.107917842803e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_imag_a, double, (7));
		double vdata_real_b[21] =
		    { 0.0, 5.177643571579e+02, 5.154521291263e+02,
      5.146409228649e+02, 5.142378756213e+02, 5.139626667737e+02,
      5.137423460082e+02, 5.135547056878e+02, 5.133910925466e+02,
      5.132470705390e+02, 5.131197729984e+02, 5.130070319283e+02,
      5.129070537032e+02, 5.128182883502e+02, 5.127393733383e+02,
      5.126691062020e+02, 5.126064276004e+02, 5.125504076570e+02,
      5.125002331720e+02, 5.124551951846e+02, 5.124146770029e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_real_b, double, (21));
		double vdata_imag_b[21] =
		    { 0.0, 5.077803458597e+02, 5.088249431599e+02,
      5.096208912659e+02, 5.101023387619e+02, 5.103976610617e+02,
      5.105948019802e+02, 5.107404165783e+02, 5.108576573661e+02,
      5.109577278523e+02, 5.110460304483e+02, 5.111252433800e+02,
      5.111968077718e+02, 5.112616233064e+02, 5.113203605551e+02,
      5.113735928093e+02, 5.114218460548e+02, 5.114656139760e+02,
      5.115053595966e+02, 5.115415130407e+02, 5.115744692211e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_imag_b, double, (21));
		double vdata_real_c[21] =
		    { 0.0, 5.195078707457e+02, 5.155422171134e+02,
      5.144678022222e+02, 5.140150594328e+02, 5.137550426810e+02,
      5.135811056728e+02, 5.134569343165e+02, 5.133651975661e+02,
      5.132955192805e+02, 5.132410471738e+02, 5.131971141679e+02,
      5.131605205716e+02, 5.131290734194e+02, 5.131012720314e+02,
      5.130760908195e+02, 5.130528295923e+02, 5.130310107773e+02,
      5.130103090133e+02, 5.129905029333e+02, 5.129714421109e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_real_c, double, (21));
		double vdata_imag_c[21] =
		    { 0.0, 5.149019699238e+02, 5.127578201997e+02,
      5.122251847514e+02, 5.121090289018e+02, 5.121143685824e+02,
      5.121496764568e+02, 5.121870921893e+02, 5.122193250322e+02,
      5.122454735794e+02, 5.122663649603e+02, 5.122830879827e+02,
      5.122965869718e+02, 5.123075927445e+02, 5.123166486553e+02,
      5.123241541685e+02, 5.123304037599e+02, 5.123356167976e+02,
      5.123399592211e+02, 5.123435588985e+02, 5.123465164008e+02 };
		DECLARE_LOCALVAR_OCL_BUFFER(vdata_imag_c, double, (21));
		epsilon = 1.0e-12;
		*verified = 1;
		*class = 'U';
		if (d1 == 64 && d2 == 64 && d3 == 64 && nt == 6) {
			*class = 'S';
			for (i = 1; i <= nt; i++) {
				err =
				    ((sums_real[i]) -
				     vdata_real_s[i]) / vdata_real_s[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
				err =
				    ((sums_imag[i]) -
				     vdata_imag_s[i]) / vdata_imag_s[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
			}
		} else if (d1 == 128 && d2 == 128 && d3 == 32 && nt == 6) {
			*class = 'W';
			for (i = 1; i <= nt; i++) {
				err =
				    ((sums_real[i]) -
				     vdata_real_w[i]) / vdata_real_w[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
				err =
				    ((sums_imag[i]) -
				     vdata_imag_w[i]) / vdata_imag_w[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
			}
		} else if (d1 == 256 && d2 == 256 && d3 == 128 && nt == 6) {
			*class = 'A';
			for (i = 1; i <= nt; i++) {
				err =
				    (sums_real[i] -
				     vdata_real_a[i]) / vdata_real_a[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
				err =
				    ((sums_imag[i]) -
				     vdata_imag_a[i]) / vdata_imag_a[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
			}
		} else if (d1 == 512 && d2 == 256 && d3 == 256 && nt == 20) {
			*class = 'B';
			for (i = 1; i <= nt; i++) {
				err =
				    ((sums_real[i]) -
				     vdata_real_b[i]) / vdata_real_b[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
				err =
				    ((sums_imag[i]) -
				     vdata_imag_b[i]) / vdata_imag_b[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
			}
		} else if (d1 == 512 && d2 == 512 && d3 == 512 && nt == 20) {
			*class = 'C';
			for (i = 1; i <= nt; i++) {
				err =
				    ((sums_real[i]) -
				     vdata_real_c[i]) / vdata_real_c[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
				err =
				    ((sums_imag[i]) -
				     vdata_imag_c[i]) / vdata_imag_c[i];
				if (fabs(err) > epsilon) {
					*verified = 0;
					break;
				}
			}
		}
		if (*class != 'U') {
			printf("Result verification successful\n");
		} else {
			printf("Result verification failed\n");
		}
		printf("class = %1c\n", *class);
	}
}

//---------------------------------------------------------------------------
//OCL related routines (BEGIN)
//---------------------------------------------------------------------------

static void init_ocl_runtime()
{
	int err;

	if (unlikely(err = oclInit("NVIDIA", 0))) {
		fprintf(stderr, "Failed to init ocl runtime:%d.\n", err);
		exit(err);
	}

	__ocl_program = oclBuildProgram("ft.A.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_evolve_0 = oclCreateKernel(__ocl_program, "evolve_0");
	DYN_PROGRAM_CHECK(__ocl_evolve_0);
	__ocl_compute_indexmap_0 =
	    oclCreateKernel(__ocl_program, "compute_indexmap_0");
	DYN_PROGRAM_CHECK(__ocl_compute_indexmap_0);
	__ocl_cffts1_0 = oclCreateKernel(__ocl_program, "cffts1_0");
	DYN_PROGRAM_CHECK(__ocl_cffts1_0);
	__ocl_cffts2_0 = oclCreateKernel(__ocl_program, "cffts2_0");
	DYN_PROGRAM_CHECK(__ocl_cffts2_0);
	__ocl_cffts3_0 = oclCreateKernel(__ocl_program, "cffts3_0");
	DYN_PROGRAM_CHECK(__ocl_cffts3_0);
	__ocl_checksum_0_reduction_step0 =
	    oclCreateKernel(__ocl_program, "checksum_0_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_checksum_0_reduction_step0);
	__ocl_checksum_0_reduction_step1 =
	    oclCreateKernel(__ocl_program, "checksum_0_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_checksum_0_reduction_step1);
	__ocl_checksum_0_reduction_step2 =
	    oclCreateKernel(__ocl_program, "checksum_0_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_checksum_0_reduction_step2);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_ex = oclCreateBuffer(ex, (221185) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ex, -1);
	__ocl_buffer_yy0_real =
	    oclCreateBuffer(yy0_real, (256 * 18) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_yy0_real, -1);
	__ocl_buffer_yy0_imag =
	    oclCreateBuffer(yy0_imag, (256 * 18) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_yy0_imag, -1);
	__ocl_buffer_yy1_real =
	    oclCreateBuffer(yy1_real, (256 * 18) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_yy1_real, -1);
	__ocl_buffer_yy1_imag =
	    oclCreateBuffer(yy1_imag, (256 * 18) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_yy1_imag, -1);
	
	__ocl_buffer_u_real = oclCreateBuffer(u_real, (256) * sizeof(double));
	rd_oclb_u_real = oclCreateBuffer(rd_log_u_real, (256) * sizeof(double));
	wr_oclb_u_real = oclCreateBuffer(wr_log_u_real, (256) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_u_real, -1);

	__ocl_buffer_u_imag = oclCreateBuffer(u_imag, (256) * sizeof(double));
	rd_oclb_u_imag = oclCreateBuffer(rd_log_u_imag, (256) * sizeof(double));
	wr_oclb_u_imag = oclCreateBuffer(wr_log_u_imag, (256) * sizeof(double));

	DYN_BUFFER_CHECK(__ocl_buffer_u_imag, -1);
	
	__ocl_buffer_xstart = oclCreateBuffer(xstart, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_xstart, -1);
	__ocl_buffer_xend = oclCreateBuffer(xend, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_xend, -1);
	__ocl_buffer_ystart = oclCreateBuffer(ystart, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_ystart, -1);
	__ocl_buffer_yend = oclCreateBuffer(yend, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_yend, -1);
	__ocl_buffer_zstart = oclCreateBuffer(zstart, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_zstart, -1);
	__ocl_buffer_zend = oclCreateBuffer(zend, (3) * sizeof(int));
	DYN_BUFFER_CHECK(__ocl_buffer_zend, -1);
	__oclb_gpu_tls_conflict_flag =
	    oclCreateBuffer(&gpu_tls_conflict_flag, 1 * sizeof(int));
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_ex);
	oclHostWrites(__ocl_buffer_yy0_real);
	oclHostWrites(__ocl_buffer_yy0_imag);
	oclHostWrites(__ocl_buffer_yy1_real);
	oclHostWrites(__ocl_buffer_yy1_imag);
	oclHostWrites(__ocl_buffer_u_real);
	oclHostWrites(__ocl_buffer_u_imag);
	oclHostWrites(__ocl_buffer_xstart);
	oclHostWrites(__ocl_buffer_xend);
	oclHostWrites(__ocl_buffer_ystart);
	oclHostWrites(__ocl_buffer_yend);
	oclHostWrites(__ocl_buffer_zstart);
	oclHostWrites(__ocl_buffer_zend);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_ex);
	oclReleaseBuffer(__ocl_buffer_yy0_real);
	oclReleaseBuffer(__ocl_buffer_yy0_imag);
	oclReleaseBuffer(__ocl_buffer_yy1_real);
	oclReleaseBuffer(__ocl_buffer_yy1_imag);
	oclReleaseBuffer(__ocl_buffer_u_real);
	oclReleaseBuffer(__ocl_buffer_u_imag);
	oclReleaseBuffer(__ocl_buffer_xstart);
	oclReleaseBuffer(__ocl_buffer_xend);
	oclReleaseBuffer(__ocl_buffer_ystart);
	oclReleaseBuffer(__ocl_buffer_yend);
	oclReleaseBuffer(__ocl_buffer_zstart);
	oclReleaseBuffer(__ocl_buffer_zend);
	if (__ocl_th_yy0_real_cffts1_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_real_cffts1_0);
		free(__ocl_th_yy0_real_cffts1_0);
		__ocl_th_yy0_real_cffts1_0_length = 0;
	}
	if (__ocl_th_yy0_imag_cffts1_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_imag_cffts1_0);
		free(__ocl_th_yy0_imag_cffts1_0);
		__ocl_th_yy0_imag_cffts1_0_length = 0;
	}
	if (__ocl_th_yy1_real_cffts1_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_real_cffts1_0);
		free(__ocl_th_yy1_real_cffts1_0);
		__ocl_th_yy1_real_cffts1_0_length = 0;
	}
	if (__ocl_th_yy1_imag_cffts1_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_imag_cffts1_0);
		free(__ocl_th_yy1_imag_cffts1_0);
		__ocl_th_yy1_imag_cffts1_0_length = 0;
	}
	if (__ocl_th_yy0_real_cffts2_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_real_cffts2_0);
		free(__ocl_th_yy0_real_cffts2_0);
		__ocl_th_yy0_real_cffts2_0_length = 0;
	}
	if (__ocl_th_yy0_imag_cffts2_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_imag_cffts2_0);
		free(__ocl_th_yy0_imag_cffts2_0);
		__ocl_th_yy0_imag_cffts2_0_length = 0;
	}
	if (__ocl_th_yy1_real_cffts2_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_real_cffts2_0);
		free(__ocl_th_yy1_real_cffts2_0);
		__ocl_th_yy1_real_cffts2_0_length = 0;
	}
	if (__ocl_th_yy1_imag_cffts2_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_imag_cffts2_0);
		free(__ocl_th_yy1_imag_cffts2_0);
		__ocl_th_yy1_imag_cffts2_0_length = 0;
	}
	if (__ocl_th_yy0_real_cffts3_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_real_cffts3_0);
		free(__ocl_th_yy0_real_cffts3_0);
		__ocl_th_yy0_real_cffts3_0_length = 0;
	}
	if (__ocl_th_yy0_imag_cffts3_0) {
		oclReleaseBuffer(__ocl_buffer_yy0_imag_cffts3_0);
		free(__ocl_th_yy0_imag_cffts3_0);
		__ocl_th_yy0_imag_cffts3_0_length = 0;
	}
	if (__ocl_th_yy1_real_cffts3_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_real_cffts3_0);
		free(__ocl_th_yy1_real_cffts3_0);
		__ocl_th_yy1_real_cffts3_0_length = 0;
	}
	if (__ocl_th_yy1_imag_cffts3_0) {
		oclReleaseBuffer(__ocl_buffer_yy1_imag_cffts3_0);
		free(__ocl_th_yy1_imag_cffts3_0);
		__ocl_th_yy1_imag_cffts3_0_length = 0;
	}
	if (__ocl_buffer_chk_real_checksum_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_chk_real_checksum_0);
		__ocl_buffer_chk_real_checksum_0_size = 0;
	}
	if (__ocl_output_chk_real_checksum_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_chk_real_checksum_0);
		free(__ocl_output_chk_real_checksum_0);
		__ocl_output_chk_real_checksum_0_size = 0;
	}
	if (__ocl_buffer_chk_imag_checksum_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_chk_imag_checksum_0);
		__ocl_buffer_chk_imag_checksum_0_size = 0;
	}
	if (__ocl_output_chk_imag_checksum_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_chk_imag_checksum_0);
		free(__ocl_output_chk_imag_checksum_0);
		__ocl_output_chk_imag_checksum_0_size = 0;
	}
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_ex);
	oclHostWrites(__ocl_buffer_yy0_real);
	oclHostWrites(__ocl_buffer_yy0_imag);
	oclHostWrites(__ocl_buffer_yy1_real);
	oclHostWrites(__ocl_buffer_yy1_imag);
	oclHostWrites(__ocl_buffer_u_real);
	oclHostWrites(__ocl_buffer_u_imag);
	oclHostWrites(__ocl_buffer_xstart);
	oclHostWrites(__ocl_buffer_xend);
	oclHostWrites(__ocl_buffer_ystart);
	oclHostWrites(__ocl_buffer_yend);
	oclHostWrites(__ocl_buffer_zstart);
	oclHostWrites(__ocl_buffer_zend);
	if (__ocl_buffer_u1_real_evolve) {
		oclHostWrites(__ocl_buffer_u1_real_evolve);
	}
	if (__ocl_buffer_u0_real_evolve) {
		oclHostWrites(__ocl_buffer_u0_real_evolve);
	}
	if (__ocl_buffer_indexmap_evolve) {
		oclHostWrites(__ocl_buffer_indexmap_evolve);
	}
	if (__ocl_buffer_u1_imag_evolve) {
		oclHostWrites(__ocl_buffer_u1_imag_evolve);
	}
	if (__ocl_buffer_u0_imag_evolve) {
		oclHostWrites(__ocl_buffer_u0_imag_evolve);
	}
	if (__ocl_buffer_indexmap_compute_indexmap) {
		oclHostWrites(__ocl_buffer_indexmap_compute_indexmap);
	}
	if (__ocl_buffer_d_cffts1) {
		oclHostWrites(__ocl_buffer_d_cffts1);
	}
	if (__ocl_buffer_x_real_cffts1) {
		oclHostWrites(__ocl_buffer_x_real_cffts1);
	}
	if (__ocl_buffer_x_imag_cffts1) {
		oclHostWrites(__ocl_buffer_x_imag_cffts1);
	}
	if (__ocl_buffer_xout_real_cffts1) {
		oclHostWrites(__ocl_buffer_xout_real_cffts1);
	}
	if (__ocl_buffer_xout_imag_cffts1) {
		oclHostWrites(__ocl_buffer_xout_imag_cffts1);
	}
	if (__ocl_buffer_d_cffts2) {
		oclHostWrites(__ocl_buffer_d_cffts2);
	}
	if (__ocl_buffer_x_real_cffts2) {
		oclHostWrites(__ocl_buffer_x_real_cffts2);
	}
	if (__ocl_buffer_x_imag_cffts2) {
		oclHostWrites(__ocl_buffer_x_imag_cffts2);
	}
	if (__ocl_buffer_xout_real_cffts2) {
		oclHostWrites(__ocl_buffer_xout_real_cffts2);
	}
	if (__ocl_buffer_xout_imag_cffts2) {
		oclHostWrites(__ocl_buffer_xout_imag_cffts2);
	}
	if (__ocl_buffer_d_cffts3) {
		oclHostWrites(__ocl_buffer_d_cffts3);
	}
	if (__ocl_buffer_x_real_cffts3) {
		oclHostWrites(__ocl_buffer_x_real_cffts3);
	}
	if (__ocl_buffer_x_imag_cffts3) {
		oclHostWrites(__ocl_buffer_x_imag_cffts3);
	}
	if (__ocl_buffer_xout_real_cffts3) {
		oclHostWrites(__ocl_buffer_xout_real_cffts3);
	}
	if (__ocl_buffer_xout_imag_cffts3) {
		oclHostWrites(__ocl_buffer_xout_imag_cffts3);
	}
	if (__ocl_buffer_u1_real_checksum) {
		oclHostWrites(__ocl_buffer_u1_real_checksum);
	}
	if (__ocl_buffer_u1_imag_checksum) {
		oclHostWrites(__ocl_buffer_u1_imag_checksum);
	}
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
