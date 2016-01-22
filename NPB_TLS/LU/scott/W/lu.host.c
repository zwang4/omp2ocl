//-------------------------------------------------------------------------------
//Host code 
//Generated at : Thu Oct 25 14:32:45 2012
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
#include "applu.h"
#include "ocldef.h"

static void blts(int nx, int ny, int nz, int k, double omega,
		 double v[33][33][33][5], ocl_buffer * __ocl_buffer_v,
		 double ldz[33][33][5][5], ocl_buffer * __ocl_buffer_ldz,
		 double ldy[33][33][5][5], ocl_buffer * __ocl_buffer_ldy,
		 double ldx[33][33][5][5], ocl_buffer * __ocl_buffer_ldx,
		 double d[33][33][5][5], ocl_buffer * __ocl_buffer_d, int ist,
		 int iend, int jst, int jend, int nx0, int ny0);
static void buts(int nx, int ny, int nz, int k, double omega,
		 double v[33][33][33][5], ocl_buffer * __ocl_buffer_v,
		 double tv[33][33][5], ocl_buffer * __ocl_buffer_tv,
		 double d[33][33][5][5], ocl_buffer * __ocl_buffer_d,
		 double udx[33][33][5][5], ocl_buffer * __ocl_buffer_udx,
		 double udy[33][33][5][5], ocl_buffer * __ocl_buffer_udy,
		 double udz[33][33][5][5], ocl_buffer * __ocl_buffer_udz,
		 int ist, int iend, int jst, int jend, int nx0, int ny0);
static void domain();
static void erhs();
static void error();
static void exact(int i, int j, int k, double u000ijk[5],
		  ocl_buffer * __ocl_buffer_u000ijk);
static void jacld(int k);
static void jacu(int k);
static void l2norm(int nx0, int ny0, int nz0, int ist, int iend, int jst,
		   int jend, double v[33][33][33][5],
		   ocl_buffer * __ocl_buffer_v, double sum[5],
		   ocl_buffer * __ocl_buffer_sum);
static void pintgr();
static void read_input();
static void rhs();
static void setbv();
static void setcoeff();
static void setiv();
static void ssor();
static void verify(double xcr[5], ocl_buffer * __ocl_buffer_xcr, double xce[5],
		   ocl_buffer * __ocl_buffer_xce, double xci, char *class,
		   ocl_buffer * __ocl_buffer_class, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified);
int main(int argc, char **argv, ocl_buffer * __ocl_buffer_argv)
{
	{
		init_ocl_runtime();
		char class;
		boolean verified;
		double mflops;
		int nthreads = 1;
		read_input();
		domain();
		setcoeff();
		sync_ocl_buffers();
		{
			setbv();
			setiv();
			erhs();
		}
		sync_ocl_buffers();
		ssor();
		error();
		pintgr();
		verify(rsdnm, NULL, errnm, NULL, frc, &class, NULL, &verified,
		       NULL);
		mflops =
		    (double)itmax *(1984.77 * (double)nx0 * (double)ny0 *
				    (double)nz0 -
				    10923.3 *
				    (((double)(nx0 + ny0 + nz0) / 3.0) *
				     ((double)(nx0 + ny0 + nz0) / 3.0)) +
				    27770.9 * (double)(nx0 + ny0 + nz0) / 3.0 -
				    144010.0) / (maxtime * 1000000.0);
		c_print_results("LU", class, nx0, ny0, nz0, itmax, nthreads,
				maxtime, mflops, "          floating point",
				verified, "2.3", "25 Oct 2012", "gcc -fopenmp",
				"gcc -fopenmp", "-lm", "-I../common", "-O3 ",
				"(none)", "(none)");
	}
}

static void blts(int nx, int ny, int nz, int k, double omega,
		 double v[33][33][33][5], ocl_buffer * __ocl_buffer_v,
		 double ldz[33][33][5][5], ocl_buffer * __ocl_buffer_ldz,
		 double ldy[33][33][5][5], ocl_buffer * __ocl_buffer_ldy,
		 double ldx[33][33][5][5], ocl_buffer * __ocl_buffer_ldx,
		 double d[33][33][5][5], ocl_buffer * __ocl_buffer_d, int ist,
		 int iend, int jst, int jend, int nx0, int ny0)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_v_blts, __ocl_p_v_blts, v,
				 (33 * 33 * 33 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_ldz_blts, __ocl_p_ldz_blts, ldz,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_ldy_blts, __ocl_p_ldy_blts, ldy,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_ldx_blts, __ocl_p_ldx_blts, ldx,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_d_blts, __ocl_p_d_blts, d,
				 (33 * 33 * 5 * 5), double);
	{
		int i, j, m;
		double tmp, tmp1;
		double tmat[5][5];
		DECLARE_LOCALVAR_OCL_BUFFER(tmat, double, (5 * 5));
		//--------------------------------------------------------------
		//Loop defined at line 214 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (5) - (0);
			_ocl_gws[1] = (jend) - (jst) + 1;
			_ocl_gws[2] = (iend) - (ist) + 1;

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_blts_0, 0, __ocl_buffer_v);
			oclSetKernelArg(__ocl_blts_0, 1, sizeof(int), &k);
			oclSetKernelArg(__ocl_blts_0, 2, sizeof(double),
					&omega);
			oclSetKernelArgBuffer(__ocl_blts_0, 3,
					      __ocl_buffer_ldz);
			oclSetKernelArg(__ocl_blts_0, 4, sizeof(int), &jst);
			oclSetKernelArg(__ocl_blts_0, 5, sizeof(int), &ist);
			int __ocl_j_bound = jend;
			oclSetKernelArg(__ocl_blts_0, 6, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = iend;
			oclSetKernelArg(__ocl_blts_0, 7, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_v);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_ldz);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_blts_0, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		//--------------------------------------------------------------
		//Loop defined at line 228 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (jend) - (jst) + 1;
			_ocl_gws[1] = (iend) - (ist) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_blts_1, 0, sizeof(int), &m);
			oclSetKernelArgBuffer(__ocl_blts_1, 1, __ocl_buffer_v);
			oclSetKernelArg(__ocl_blts_1, 2, sizeof(int), &k);
			oclSetKernelArg(__ocl_blts_1, 3, sizeof(double),
					&omega);
			oclSetKernelArgBuffer(__ocl_blts_1, 4,
					      __ocl_buffer_ldy);
			oclSetKernelArgBuffer(__ocl_blts_1, 5,
					      __ocl_buffer_ldx);
			oclSetKernelArgBuffer(__ocl_blts_1, 6, __ocl_buffer_d);
			oclSetKernelArg(__ocl_blts_1, 7, sizeof(int), &jst);
			oclSetKernelArg(__ocl_blts_1, 8, sizeof(int), &ist);
			int __ocl_j_bound = jend;
			oclSetKernelArg(__ocl_blts_1, 9, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = iend;
			oclSetKernelArg(__ocl_blts_1, 10, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_v);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_ldy);
			oclDevReads(__ocl_buffer_ldx);
			oclDevReads(__ocl_buffer_d);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_blts_1, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

	}
}

static void buts(int nx, int ny, int nz, int k, double omega,
		 double v[33][33][33][5], ocl_buffer * __ocl_buffer_v,
		 double tv[33][33][5], ocl_buffer * __ocl_buffer_tv,
		 double d[33][33][5][5], ocl_buffer * __ocl_buffer_d,
		 double udx[33][33][5][5], ocl_buffer * __ocl_buffer_udx,
		 double udy[33][33][5][5], ocl_buffer * __ocl_buffer_udy,
		 double udz[33][33][5][5], ocl_buffer * __ocl_buffer_udz,
		 int ist, int iend, int jst, int jend, int nx0, int ny0)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_tv_buts, __ocl_p_tv_buts, tv,
				 (33 * 33 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_udz_buts, __ocl_p_udz_buts, udz,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_v_buts, __ocl_p_v_buts, v,
				 (33 * 33 * 33 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_udy_buts, __ocl_p_udy_buts, udy,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_udx_buts, __ocl_p_udx_buts, udx,
				 (33 * 33 * 5 * 5), double);
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_d_buts, __ocl_p_d_buts, d,
				 (33 * 33 * 5 * 5), double);
	{
		int i, j, m;
		double tmp, tmp1;
		double tmat[5][5];
		DECLARE_LOCALVAR_OCL_BUFFER(tmat, double, (5 * 5));
		//--------------------------------------------------------------
		//Loop defined at line 454 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (5) - (0);
			_ocl_gws[1] = (jend) - (jst) + 1;
			_ocl_gws[2] = (iend) - (ist) + 1;

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_buts_0, 0, __ocl_buffer_tv);
			oclSetKernelArg(__ocl_buts_0, 1, sizeof(double),
					&omega);
			oclSetKernelArgBuffer(__ocl_buts_0, 2,
					      __ocl_buffer_udz);
			oclSetKernelArgBuffer(__ocl_buts_0, 3, __ocl_buffer_v);
			oclSetKernelArg(__ocl_buts_0, 4, sizeof(int), &k);
			oclSetKernelArg(__ocl_buts_0, 5, sizeof(int), &jst);
			oclSetKernelArg(__ocl_buts_0, 6, sizeof(int), &ist);
			int __ocl_j_bound = jend;
			oclSetKernelArg(__ocl_buts_0, 7, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = iend;
			oclSetKernelArg(__ocl_buts_0, 8, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_tv);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_udz);
			oclDevReads(__ocl_buffer_v);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_buts_0, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		//--------------------------------------------------------------
		//Loop defined at line 470 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[2];
			_ocl_gws[0] = (jend) - (jst) + 1;
			_ocl_gws[1] = (iend) - (ist) + 1;

			oclGetWorkSize(2, _ocl_gws, NULL);
			oclSetKernelArg(__ocl_buts_1, 0, sizeof(int), &m);
			oclSetKernelArgBuffer(__ocl_buts_1, 1, __ocl_buffer_tv);
			oclSetKernelArg(__ocl_buts_1, 2, sizeof(double),
					&omega);
			oclSetKernelArgBuffer(__ocl_buts_1, 3,
					      __ocl_buffer_udy);
			oclSetKernelArgBuffer(__ocl_buts_1, 4, __ocl_buffer_v);
			oclSetKernelArg(__ocl_buts_1, 5, sizeof(int), &k);
			oclSetKernelArgBuffer(__ocl_buts_1, 6,
					      __ocl_buffer_udx);
			oclSetKernelArgBuffer(__ocl_buts_1, 7, __ocl_buffer_d);
			oclSetKernelArg(__ocl_buts_1, 8, sizeof(int), &jst);
			oclSetKernelArg(__ocl_buts_1, 9, sizeof(int), &ist);
			int __ocl_j_bound = jend;
			oclSetKernelArg(__ocl_buts_1, 10, sizeof(int),
					&__ocl_j_bound);
			int __ocl_i_bound = iend;
			oclSetKernelArg(__ocl_buts_1, 11, sizeof(int),
					&__ocl_i_bound);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_tv);
			oclDevWrites(__ocl_buffer_v);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			oclDevReads(__ocl_buffer_udy);
			oclDevReads(__ocl_buffer_udx);
			oclDevReads(__ocl_buffer_d);
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_buts_1, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

	}
}

static void domain()
{
	nx = nx0;
	ny = ny0;
	nz = nz0;
	if (nx < 4 || ny < 4 || nz < 4) {
		printf
		    ("     SUBDOMAIN SIZE IS TOO SMALL - \n     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n     SO THAT NX, NY AND NZ ARE GREATER THAN OR EQUAL\n     TO 4 THEY ARE CURRENTLY%3d%3d%3d\n",
		     nx, ny, nz);
		exit(1);
	}
	if (nx > 33 || ny > 33 || nz > 33) {
		printf
		    ("     SUBDOMAIN SIZE IS TOO LARGE - \n     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n     SO THAT NX, NY AND NZ ARE LESS THAN OR EQUAL TO \n     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY.  THEY ARE\n     CURRENTLY%4d%4d%4d\n",
		     nx, ny, nz);
		exit(1);
	}
	ist = 1;
	iend = nx - 2;
	jst = 1;
	jend = ny - 2;
}

static void erhs()
{
	int i, j, k, m;
	int iglob, jglob;
	int L1, L2;
	int ist1, iend1;
	int jst1, jend1;
	double dsspm;
	double xi, eta, zeta;
	double q;
	double u21, u31, u41;
	double tmp;
	double u21i, u31i, u41i, u51i;
	double u21j, u31j, u41j, u51j;
	double u21k, u31k, u41k, u51k;
	double u21im1, u31im1, u41im1, u51im1;
	double u21jm1, u31jm1, u41jm1, u51jm1;
	double u21km1, u31km1, u41km1, u51km1;
	dsspm = dssp;
	//--------------------------------------------------------------
	//Loop defined at line 741 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz) - (0);
		_ocl_gws[1] = (ny) - (0);
		_ocl_gws[2] = (nx) - (0);

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_erhs_0, 0, __ocl_buffer_frct);
		int __ocl_k_bound = nz;
		oclSetKernelArg(__ocl_erhs_0, 1, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = ny;
		oclSetKernelArg(__ocl_erhs_0, 2, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = nx;
		oclSetKernelArg(__ocl_erhs_0, 3, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_frct);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_0, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 752 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (nx) - (0);

		oclGetWorkSize(1, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_erhs_1, 0, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_erhs_1, 1, sizeof(int), &ny);
		oclSetKernelArg(__ocl_erhs_1, 2, sizeof(int), &ny0);
		oclSetKernelArg(__ocl_erhs_1, 3, sizeof(int), &nz);
		oclSetKernelArgBuffer(__ocl_erhs_1, 4, __ocl_buffer_rsd);
		oclSetKernelArgBuffer(__ocl_erhs_1, 5, __ocl_buffer_ce);
		int __ocl_i_bound = nx;
		oclSetKernelArg(__ocl_erhs_1, 6, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rsd);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_ce);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_1, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	L1 = 0;
	L2 = nx - 1;
	//--------------------------------------------------------------
	//Loop defined at line 787 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz - 1) - (1);
		_ocl_gws[1] = (jend) - (jst) + 1;
		_ocl_gws[2] = (L2) - (L1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_erhs_2, 0, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_erhs_2, 1, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_erhs_2, 2, sizeof(int), &jst);
		oclSetKernelArg(__ocl_erhs_2, 3, sizeof(int), &L1);
		int __ocl_k_bound = nz - 1;
		oclSetKernelArg(__ocl_erhs_2, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_erhs_2, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = L2;
		oclSetKernelArg(__ocl_erhs_2, 6, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rsd);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_2, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 806 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (jend) - (jst) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_erhs_3, 0, sizeof(int), &ist);
		oclSetKernelArg(__ocl_erhs_3, 1, sizeof(int), &iend);
		oclSetKernelArgBuffer(__ocl_erhs_3, 2, __ocl_buffer_frct);
		oclSetKernelArg(__ocl_erhs_3, 3, sizeof(double), &tx2);
		oclSetKernelArgBuffer(__ocl_erhs_3, 4, __ocl_buffer_flux);
		oclSetKernelArg(__ocl_erhs_3, 5, sizeof(int), &L2);
		oclSetKernelArgBuffer(__ocl_erhs_3, 6, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_erhs_3, 7, sizeof(double), &tx3);
		oclSetKernelArg(__ocl_erhs_3, 8, sizeof(double), &dx1);
		oclSetKernelArg(__ocl_erhs_3, 9, sizeof(double), &tx1);
		oclSetKernelArg(__ocl_erhs_3, 10, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_erhs_3, 11, sizeof(double), &dx3);
		oclSetKernelArg(__ocl_erhs_3, 12, sizeof(double), &dx4);
		oclSetKernelArg(__ocl_erhs_3, 13, sizeof(double), &dx5);
		oclSetKernelArg(__ocl_erhs_3, 14, sizeof(double), &dsspm);
		oclSetKernelArg(__ocl_erhs_3, 15, sizeof(int), &nx);
		oclSetKernelArg(__ocl_erhs_3, 16, sizeof(int), &jst);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_erhs_3, 17, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_erhs_3, 18, sizeof(int), &__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_frct);
		oclDevWrites(__ocl_buffer_flux);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rsd);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_3, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	L1 = 0;
	L2 = ny - 1;
	//--------------------------------------------------------------
	//Loop defined at line 918 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (L2) - (L1) + 1;
		_ocl_gws[2] = (iend) - (ist) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_erhs_4, 0, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_erhs_4, 1, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_erhs_4, 2, sizeof(int), &L1);
		oclSetKernelArg(__ocl_erhs_4, 3, sizeof(int), &ist);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_erhs_4, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = L2;
		oclSetKernelArg(__ocl_erhs_4, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_erhs_4, 6, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rsd);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_4, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 937 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_erhs_5, 0, sizeof(int), &jst);
		oclSetKernelArg(__ocl_erhs_5, 1, sizeof(int), &jend);
		oclSetKernelArgBuffer(__ocl_erhs_5, 2, __ocl_buffer_frct);
		oclSetKernelArg(__ocl_erhs_5, 3, sizeof(double), &ty2);
		oclSetKernelArgBuffer(__ocl_erhs_5, 4, __ocl_buffer_flux);
		oclSetKernelArg(__ocl_erhs_5, 5, sizeof(int), &L2);
		oclSetKernelArgBuffer(__ocl_erhs_5, 6, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_erhs_5, 7, sizeof(double), &ty3);
		oclSetKernelArg(__ocl_erhs_5, 8, sizeof(double), &dy1);
		oclSetKernelArg(__ocl_erhs_5, 9, sizeof(double), &ty1);
		oclSetKernelArg(__ocl_erhs_5, 10, sizeof(double), &dy2);
		oclSetKernelArg(__ocl_erhs_5, 11, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_erhs_5, 12, sizeof(double), &dy4);
		oclSetKernelArg(__ocl_erhs_5, 13, sizeof(double), &dy5);
		oclSetKernelArg(__ocl_erhs_5, 14, sizeof(double), &dsspm);
		oclSetKernelArg(__ocl_erhs_5, 15, sizeof(int), &ny);
		oclSetKernelArg(__ocl_erhs_5, 16, sizeof(int), &ist);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_erhs_5, 17, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_erhs_5, 18, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_frct);
		oclDevWrites(__ocl_buffer_flux);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rsd);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_5, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 1047 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (jend) - (jst) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_erhs_6, 0, sizeof(int), &nz);
		oclSetKernelArgBuffer(__ocl_erhs_6, 1, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_erhs_6, 2, __ocl_buffer_rsd);
		oclSetKernelArgBuffer(__ocl_erhs_6, 3, __ocl_buffer_frct);
		oclSetKernelArg(__ocl_erhs_6, 4, sizeof(double), &tz2);
		oclSetKernelArg(__ocl_erhs_6, 5, sizeof(double), &tz3);
		oclSetKernelArg(__ocl_erhs_6, 6, sizeof(double), &dz1);
		oclSetKernelArg(__ocl_erhs_6, 7, sizeof(double), &tz1);
		oclSetKernelArg(__ocl_erhs_6, 8, sizeof(double), &dz2);
		oclSetKernelArg(__ocl_erhs_6, 9, sizeof(double), &dz3);
		oclSetKernelArg(__ocl_erhs_6, 10, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_erhs_6, 11, sizeof(double), &dz5);
		oclSetKernelArg(__ocl_erhs_6, 12, sizeof(double), &dsspm);
		oclSetKernelArg(__ocl_erhs_6, 13, sizeof(int), &jst);
		oclSetKernelArg(__ocl_erhs_6, 14, sizeof(int), &ist);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_erhs_6, 15, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_erhs_6, 16, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
		oclDevWrites(__ocl_buffer_frct);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_rsd);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_erhs_6, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

}

static void error()
{
	int i, j, k, m;
	int iglob, jglob;
	double tmp;
	double u000ijk[5];
	DECLARE_LOCALVAR_OCL_BUFFER(u000ijk, double, (5));
	for (m = 0; m < 5; m++) {
		errnm[m] = 0.0;
	}
	for (i = ist; i <= iend; i++) {
		iglob = i;
		for (j = jst; j <= jend; j++) {
			jglob = j;
			for (k = 1; k <= nz - 2; k++) {

				exact(iglob, jglob, k, u000ijk,
				      __ocl_buffer_u000ijk);
				for (m = 0; m < 5; m++) {
					tmp = (u000ijk[m] - u[i][j][k][m]);
					errnm[m] = errnm[m] + tmp * tmp;
				}
			}
		}
	}
	for (m = 0; m < 5; m++) {
		errnm[m] = sqrt(errnm[m] / ((nx0 - 2) * (ny0 - 2) * (nz0 - 2)));
	}
}

static void exact(int i, int j, int k, double u000ijk[5],
		  ocl_buffer * __ocl_buffer_u000ijk)
{
	{
		int m;
		double xi, eta, zeta;
		xi = ((double)i) / (nx0 - 1);
		eta = ((double)j) / (ny0 - 1);
		zeta = ((double)k) / (nz - 1);
		for (m = 0; m < 5; m++) {
			u000ijk[m] =
			    ce[m][0] + ce[m][1] * xi + ce[m][2] * eta +
			    ce[m][3] * zeta + ce[m][4] * xi * xi +
			    ce[m][5] * eta * eta + ce[m][6] * zeta * zeta +
			    ce[m][7] * xi * xi * xi +
			    ce[m][8] * eta * eta * eta +
			    ce[m][9] * zeta * zeta * zeta +
			    ce[m][10] * xi * xi * xi * xi +
			    ce[m][11] * eta * eta * eta * eta +
			    ce[m][12] * zeta * zeta * zeta * zeta;
		}
	}
}

static void jacld(int k)
{
	int i, j;
	double r43;
	double c1345;
	double c34;
	double tmp1, tmp2, tmp3;
	r43 = (4.0 / 3.0);
	c1345 = 1.40e+00 * 1.00e-01 * 1.00e+00 * 1.40e+00;
	c34 = 1.00e-01 * 1.00e+00;
	//--------------------------------------------------------------
	//Loop defined at line 1269 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (jend) - (jst) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_jacld_0, 0, __ocl_buffer_u);
		oclSetKernelArg(__ocl_jacld_0, 1, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_jacld_0, 2, __ocl_buffer_d);
		oclSetKernelArg(__ocl_jacld_0, 3, sizeof(double), &dt);
		oclSetKernelArg(__ocl_jacld_0, 4, sizeof(double), &tx1);
		oclSetKernelArg(__ocl_jacld_0, 5, sizeof(double), &dx1);
		oclSetKernelArg(__ocl_jacld_0, 6, sizeof(double), &ty1);
		oclSetKernelArg(__ocl_jacld_0, 7, sizeof(double), &dy1);
		oclSetKernelArg(__ocl_jacld_0, 8, sizeof(double), &tz1);
		oclSetKernelArg(__ocl_jacld_0, 9, sizeof(double), &dz1);
		oclSetKernelArg(__ocl_jacld_0, 10, sizeof(double), &r43);
		oclSetKernelArg(__ocl_jacld_0, 11, sizeof(double), &c34);
		oclSetKernelArg(__ocl_jacld_0, 12, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_jacld_0, 13, sizeof(double), &dy2);
		oclSetKernelArg(__ocl_jacld_0, 14, sizeof(double), &dz2);
		oclSetKernelArg(__ocl_jacld_0, 15, sizeof(double), &dx3);
		oclSetKernelArg(__ocl_jacld_0, 16, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_jacld_0, 17, sizeof(double), &dz3);
		oclSetKernelArg(__ocl_jacld_0, 18, sizeof(double), &dx4);
		oclSetKernelArg(__ocl_jacld_0, 19, sizeof(double), &dy4);
		oclSetKernelArg(__ocl_jacld_0, 20, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_jacld_0, 21, sizeof(double), &c1345);
		oclSetKernelArg(__ocl_jacld_0, 22, sizeof(double), &dx5);
		oclSetKernelArg(__ocl_jacld_0, 23, sizeof(double), &dy5);
		oclSetKernelArg(__ocl_jacld_0, 24, sizeof(double), &dz5);
		oclSetKernelArgBuffer(__ocl_jacld_0, 25, __ocl_buffer_a);
		oclSetKernelArg(__ocl_jacld_0, 26, sizeof(double), &tz2);
		oclSetKernelArgBuffer(__ocl_jacld_0, 27, __ocl_buffer_b);
		oclSetKernelArg(__ocl_jacld_0, 28, sizeof(double), &ty2);
		oclSetKernelArgBuffer(__ocl_jacld_0, 29, __ocl_buffer_c);
		oclSetKernelArg(__ocl_jacld_0, 30, sizeof(double), &tx2);
		oclSetKernelArg(__ocl_jacld_0, 31, sizeof(int), &jst);
		oclSetKernelArg(__ocl_jacld_0, 32, sizeof(int), &ist);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_jacld_0, 33, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_jacld_0, 34, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_d);
		oclDevWrites(__ocl_buffer_a);
		oclDevWrites(__ocl_buffer_b);
		oclDevWrites(__ocl_buffer_c);
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

		oclRunKernel(__ocl_jacld_0, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

}

static void jacu(int k)
{
	int i, j;
	double r43;
	double c1345;
	double c34;
	double tmp1, tmp2, tmp3;
	r43 = (4.0 / 3.0);
	c1345 = 1.40e+00 * 1.00e-01 * 1.00e+00 * 1.40e+00;
	c34 = 1.00e-01 * 1.00e+00;
	//--------------------------------------------------------------
	//Loop defined at line 1643 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (jend) - (jst) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_jacu_0, 0, __ocl_buffer_u);
		oclSetKernelArg(__ocl_jacu_0, 1, sizeof(int), &k);
		oclSetKernelArgBuffer(__ocl_jacu_0, 2, __ocl_buffer_d);
		oclSetKernelArg(__ocl_jacu_0, 3, sizeof(double), &dt);
		oclSetKernelArg(__ocl_jacu_0, 4, sizeof(double), &tx1);
		oclSetKernelArg(__ocl_jacu_0, 5, sizeof(double), &dx1);
		oclSetKernelArg(__ocl_jacu_0, 6, sizeof(double), &ty1);
		oclSetKernelArg(__ocl_jacu_0, 7, sizeof(double), &dy1);
		oclSetKernelArg(__ocl_jacu_0, 8, sizeof(double), &tz1);
		oclSetKernelArg(__ocl_jacu_0, 9, sizeof(double), &dz1);
		oclSetKernelArg(__ocl_jacu_0, 10, sizeof(double), &r43);
		oclSetKernelArg(__ocl_jacu_0, 11, sizeof(double), &c34);
		oclSetKernelArg(__ocl_jacu_0, 12, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_jacu_0, 13, sizeof(double), &dy2);
		oclSetKernelArg(__ocl_jacu_0, 14, sizeof(double), &dz2);
		oclSetKernelArg(__ocl_jacu_0, 15, sizeof(double), &dx3);
		oclSetKernelArg(__ocl_jacu_0, 16, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_jacu_0, 17, sizeof(double), &dz3);
		oclSetKernelArg(__ocl_jacu_0, 18, sizeof(double), &dx4);
		oclSetKernelArg(__ocl_jacu_0, 19, sizeof(double), &dy4);
		oclSetKernelArg(__ocl_jacu_0, 20, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_jacu_0, 21, sizeof(double), &c1345);
		oclSetKernelArg(__ocl_jacu_0, 22, sizeof(double), &dx5);
		oclSetKernelArg(__ocl_jacu_0, 23, sizeof(double), &dy5);
		oclSetKernelArg(__ocl_jacu_0, 24, sizeof(double), &dz5);
		oclSetKernelArgBuffer(__ocl_jacu_0, 25, __ocl_buffer_a);
		oclSetKernelArg(__ocl_jacu_0, 26, sizeof(double), &tx2);
		oclSetKernelArgBuffer(__ocl_jacu_0, 27, __ocl_buffer_b);
		oclSetKernelArg(__ocl_jacu_0, 28, sizeof(double), &ty2);
		oclSetKernelArgBuffer(__ocl_jacu_0, 29, __ocl_buffer_c);
		oclSetKernelArg(__ocl_jacu_0, 30, sizeof(double), &tz2);
		oclSetKernelArg(__ocl_jacu_0, 31, sizeof(int), &jst);
		oclSetKernelArg(__ocl_jacu_0, 32, sizeof(int), &ist);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_jacu_0, 33, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_jacu_0, 34, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_d);
		oclDevWrites(__ocl_buffer_a);
		oclDevWrites(__ocl_buffer_b);
		oclDevWrites(__ocl_buffer_c);
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

		oclRunKernel(__ocl_jacu_0, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

}

static void l2norm(int nx0, int ny0, int nz0, int ist, int iend, int jst,
		   int jend, double v[33][33][33][5],
		   ocl_buffer * __ocl_buffer_v, double sum[5],
		   ocl_buffer * __ocl_buffer_sum)
{
	CREATE_FUNC_LEVEL_BUFFER(__ocl_buffer_v_l2norm, __ocl_p_v_l2norm, v,
				 (33 * 33 * 33 * 5), double);
	{
		int i, j, k, m;
		double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 =
		    0.0;
		for (m = 0; m < 5; m++) {
			sum[m] = 0.0;
		}
		//--------------------------------------------------------------
		//Loop defined at line 2020 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//Reduction step 1
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (nz0 - 2) - (1) + 1;
			_ocl_gws[1] = (jend) - (jst) + 1;
			_ocl_gws[2] = (iend) - (ist) + 1;

			oclGetWorkSize(3, _ocl_gws, NULL);
			size_t __ocl_act_buf_size =
			    (_ocl_gws[0] * _ocl_gws[1] * _ocl_gws[2]);
			REDUCTION_STEP1_MULT_NDRANGE();
//Prepare buffer for the reduction variable: sum0
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_sum0_l2norm_0_size, __ocl_buf_size,
			     __ocl_buffer_sum0_l2norm_0, double);
//Prepare buffer for the reduction variable: sum1
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_sum1_l2norm_0_size, __ocl_buf_size,
			     __ocl_buffer_sum1_l2norm_0, double);
//Prepare buffer for the reduction variable: sum2
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_sum2_l2norm_0_size, __ocl_buf_size,
			     __ocl_buffer_sum2_l2norm_0, double);
//Prepare buffer for the reduction variable: sum3
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_sum3_l2norm_0_size, __ocl_buf_size,
			     __ocl_buffer_sum3_l2norm_0, double);
//Prepare buffer for the reduction variable: sum4
			CREATE_REDUCTION_STEP1_BUFFER
			    (__ocl_buffer_sum4_l2norm_0_size, __ocl_buf_size,
			     __ocl_buffer_sum4_l2norm_0, double);

			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
//init the round-up buffer spaces so that I can apply vectorisation on the second step
			if (__ocl_buf_size > __ocl_act_buf_size) {
				oclSetKernelArgBuffer
				    (__ocl_l2norm_0_reduction_step0, 0,
				     __ocl_buffer_sum0_l2norm_0);
				oclSetKernelArgBuffer
				    (__ocl_l2norm_0_reduction_step0, 1,
				     __ocl_buffer_sum1_l2norm_0);
				oclSetKernelArgBuffer
				    (__ocl_l2norm_0_reduction_step0, 2,
				     __ocl_buffer_sum2_l2norm_0);
				oclSetKernelArgBuffer
				    (__ocl_l2norm_0_reduction_step0, 3,
				     __ocl_buffer_sum3_l2norm_0);
				oclSetKernelArgBuffer
				    (__ocl_l2norm_0_reduction_step0, 4,
				     __ocl_buffer_sum4_l2norm_0);
				unsigned int __ocl_buffer_offset =
				    __ocl_buf_size - __ocl_act_buf_size;
				oclSetKernelArg(__ocl_l2norm_0_reduction_step0,
						5, sizeof(unsigned int),
						&__ocl_act_buf_size);
				oclSetKernelArg(__ocl_l2norm_0_reduction_step0,
						6, sizeof(unsigned int),
						&__ocl_buffer_offset);

				size_t __offset_work_size[1] =
				    { __ocl_buffer_offset };
				oclRunKernel(__ocl_l2norm_0_reduction_step0, 1,
					     __offset_work_size);
			}

			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1, 0,
					      __ocl_buffer_v);
			oclSetKernelArg(__ocl_l2norm_0_reduction_step1, 1,
					sizeof(int), &jst);
			oclSetKernelArg(__ocl_l2norm_0_reduction_step1, 2,
					sizeof(int), &ist);
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			int __ocl_k_bound = nz0 - 2;
			oclSetKernelArg(__ocl_l2norm_0_reduction_step1, 3,
					sizeof(int), &__ocl_k_bound);
			int __ocl_j_bound = jend;
			oclSetKernelArg(__ocl_l2norm_0_reduction_step1, 4,
					sizeof(int), &__ocl_j_bound);
			int __ocl_i_bound = iend;
			oclSetKernelArg(__ocl_l2norm_0_reduction_step1, 5,
					sizeof(int), &__ocl_i_bound);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1, 6,
					      __ocl_buffer_sum0_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1, 7,
					      __ocl_buffer_sum1_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1, 8,
					      __ocl_buffer_sum2_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1, 9,
					      __ocl_buffer_sum3_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step1,
					      10, __ocl_buffer_sum4_l2norm_0);
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
			oclDevReads(__ocl_buffer_v);
			//------------------------------------------
			//Read only buffers (END) 
			//------------------------------------------

			oclRunKernel(__ocl_l2norm_0_reduction_step1, 3,
				     _ocl_gws);

//Reduction Step 2
			unsigned __ocl_num_block = __ocl_buf_size / (GROUP_SIZE * 4);	/*Vectorisation by a factor of 4 */
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_sum0_l2norm_0_size, __ocl_num_block,
			     16, __ocl_output_buffer_sum0_l2norm_0,
			     __ocl_output_sum0_l2norm_0, double);
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_sum1_l2norm_0_size, __ocl_num_block,
			     16, __ocl_output_buffer_sum1_l2norm_0,
			     __ocl_output_sum1_l2norm_0, double);
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_sum2_l2norm_0_size, __ocl_num_block,
			     16, __ocl_output_buffer_sum2_l2norm_0,
			     __ocl_output_sum2_l2norm_0, double);
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_sum3_l2norm_0_size, __ocl_num_block,
			     16, __ocl_output_buffer_sum3_l2norm_0,
			     __ocl_output_sum3_l2norm_0, double);
			CREATE_REDUCTION_STEP2_BUFFER
			    (__ocl_output_sum4_l2norm_0_size, __ocl_num_block,
			     16, __ocl_output_buffer_sum4_l2norm_0,
			     __ocl_output_sum4_l2norm_0, double);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 0,
					      __ocl_buffer_sum0_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 1,
					      __ocl_output_buffer_sum0_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 2,
					      __ocl_buffer_sum1_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 3,
					      __ocl_output_buffer_sum1_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 4,
					      __ocl_buffer_sum2_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 5,
					      __ocl_output_buffer_sum2_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 6,
					      __ocl_buffer_sum3_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 7,
					      __ocl_output_buffer_sum3_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 8,
					      __ocl_buffer_sum4_l2norm_0);
			oclSetKernelArgBuffer(__ocl_l2norm_0_reduction_step2, 9,
					      __ocl_output_buffer_sum4_l2norm_0);

			oclDevWrites(__ocl_output_buffer_sum0_l2norm_0);
			oclDevWrites(__ocl_output_buffer_sum1_l2norm_0);
			oclDevWrites(__ocl_output_buffer_sum2_l2norm_0);
			oclDevWrites(__ocl_output_buffer_sum3_l2norm_0);
			oclDevWrites(__ocl_output_buffer_sum4_l2norm_0);

			size_t __ocl_globalThreads[] = { __ocl_buf_size / 4 };	/* Each work item performs 4 reductions */
			size_t __ocl_localThreads[] = { GROUP_SIZE };

			oclRunKernelL(__ocl_l2norm_0_reduction_step2, 1,
				      __ocl_globalThreads, __ocl_localThreads);

//Do the final reduction part on the CPU
			oclHostReads(__ocl_output_buffer_sum0_l2norm_0);
			oclHostReads(__ocl_output_buffer_sum1_l2norm_0);
			oclHostReads(__ocl_output_buffer_sum2_l2norm_0);
			oclHostReads(__ocl_output_buffer_sum3_l2norm_0);
			oclHostReads(__ocl_output_buffer_sum4_l2norm_0);
			oclSync();

			for (unsigned __ocl_i = 0; __ocl_i < __ocl_num_block;
			     __ocl_i++) {
				sum0 =
				    sum0 + __ocl_output_sum0_l2norm_0[__ocl_i];
				sum1 =
				    sum1 + __ocl_output_sum1_l2norm_0[__ocl_i];
				sum2 =
				    sum2 + __ocl_output_sum2_l2norm_0[__ocl_i];
				sum3 =
				    sum3 + __ocl_output_sum3_l2norm_0[__ocl_i];
				sum4 =
				    sum4 + __ocl_output_sum4_l2norm_0[__ocl_i];
			}

		}

		{
			sum[0] += sum0;
			sum[1] += sum1;
			sum[2] += sum2;
			sum[3] += sum3;
			sum[4] += sum4;
		}
		for (m = 0; m < 5; m++) {
			sum[m] =
			    sqrt(sum[m] / ((nx0 - 2) * (ny0 - 2) * (nz0 - 2)));
		}
	}
}

static void pintgr()
{
	int i, j, k;
	int ibeg, ifin, ifin1;
	int jbeg, jfin, jfin1;
	int iglob, iglob1, iglob2;
	int jglob, jglob1, jglob2;
	double phi1[35][35];
	DECLARE_LOCALVAR_OCL_BUFFER(phi1, double, (35 * 35));
	double phi2[35][35];
	DECLARE_LOCALVAR_OCL_BUFFER(phi2, double, (35 * 35));
	double frc1, frc2, frc3;
	ibeg = nx;
	ifin = 0;
	iglob1 = -1;
	iglob2 = nx - 1;
	if (iglob1 >= ii1 && iglob2 < ii2 + nx)
		ibeg = 0;
	if (iglob1 >= ii1 - nx && iglob2 <= ii2)
		ifin = nx;
	if (ii1 >= iglob1 && ii1 <= iglob2)
		ibeg = ii1;
	if (ii2 >= iglob1 && ii2 <= iglob2)
		ifin = ii2;
	jbeg = ny;
	jfin = -1;
	jglob1 = 0;
	jglob2 = ny - 1;
	if (jglob1 >= ji1 && jglob2 < ji2 + ny)
		jbeg = 0;
	if (jglob1 > ji1 - ny && jglob2 <= ji2)
		jfin = ny;
	if (ji1 >= jglob1 && ji1 <= jglob2)
		jbeg = ji1;
	if (ji2 >= jglob1 && ji2 <= jglob2)
		jfin = ji2;
	ifin1 = ifin;
	jfin1 = jfin;
	if (ifin1 == ii2)
		ifin1 = ifin - 1;
	if (jfin1 == ji2)
		jfin1 = jfin - 1;
	for (i = 0; i <= 33 + 1; i++) {
		for (k = 0; k <= 33 + 1; k++) {
			phi1[i][k] = 0.0;
			phi2[i][k] = 0.0;
		}
	}
	for (i = ibeg; i <= ifin; i++) {
		iglob = i;
		for (j = jbeg; j <= jfin; j++) {
			jglob = j;
			k = ki1;
			phi1[i][j] =
			    0.40e+00 * (u[i][j][k][4] -
					0.50 *
					(((u[i][j][k][1]) * (u[i][j][k][1])) +
					 ((u[i][j][k][2]) * (u[i][j][k][2])) +
					 ((u[i][j][k][3]) * (u[i][j][k][3]))) /
					u[i][j][k][0]);
			k = ki2;
			phi2[i][j] =
			    0.40e+00 * (u[i][j][k][4] -
					0.50 *
					(((u[i][j][k][1]) * (u[i][j][k][1])) +
					 ((u[i][j][k][2]) * (u[i][j][k][2])) +
					 ((u[i][j][k][3]) * (u[i][j][k][3]))) /
					u[i][j][k][0]);
		}
	}
	frc1 = 0.0;
	for (i = ibeg; i <= ifin1; i++) {
		for (j = jbeg; j <= jfin1; j++) {
			frc1 =
			    frc1 + (phi1[i][j] + phi1[i + 1][j] +
				    phi1[i][j + 1] + phi1[i + 1][j + 1] +
				    phi2[i][j] + phi2[i + 1][j] + phi2[i][j +
									  1] +
				    phi2[i + 1][j + 1]);
		}
	}
	frc1 = dxi * deta * frc1;
	for (i = 0; i <= 33 + 1; i++) {
		for (k = 0; k <= 33 + 1; k++) {
			phi1[i][k] = 0.0;
			phi2[i][k] = 0.0;
		}
	}
	jglob = jbeg;
	if (jglob == ji1) {
		for (i = ibeg; i <= ifin; i++) {
			iglob = i;
			for (k = ki1; k <= ki2; k++) {
				phi1[i][k] =
				    0.40e+00 * (u[i][jbeg][k][4] -
						0.50 *
						(((u[i][jbeg][k][1]) *
						  (u[i][jbeg][k][1])) +
						 ((u[i][jbeg][k][2]) *
						  (u[i][jbeg][k][2])) +
						 ((u[i][jbeg][k][3]) *
						  (u[i][jbeg][k][3]))) /
						u[i][jbeg][k][0]);
			}
		}
	}
	jglob = jfin;
	if (jglob == ji2) {
		for (i = ibeg; i <= ifin; i++) {
			iglob = i;
			for (k = ki1; k <= ki2; k++) {
				phi2[i][k] =
				    0.40e+00 * (u[i][jfin][k][4] -
						0.50 *
						(((u[i][jfin][k][1]) *
						  (u[i][jfin][k][1])) +
						 ((u[i][jfin][k][2]) *
						  (u[i][jfin][k][2])) +
						 ((u[i][jfin][k][3]) *
						  (u[i][jfin][k][3]))) /
						u[i][jfin][k][0]);
			}
		}
	}
	frc2 = 0.0;
	for (i = ibeg; i <= ifin1; i++) {
		for (k = ki1; k <= ki2 - 1; k++) {
			frc2 =
			    frc2 + (phi1[i][k] + phi1[i + 1][k] +
				    phi1[i][k + 1] + phi1[i + 1][k + 1] +
				    phi2[i][k] + phi2[i + 1][k] + phi2[i][k +
									  1] +
				    phi2[i + 1][k + 1]);
		}
	}
	frc2 = dxi * dzeta * frc2;
	for (i = 0; i <= 33 + 1; i++) {
		for (k = 0; k <= 33 + 1; k++) {
			phi1[i][k] = 0.0;
			phi2[i][k] = 0.0;
		}
	}
	iglob = ibeg;
	if (iglob == ii1) {
		for (j = jbeg; j <= jfin; j++) {
			jglob = j;
			for (k = ki1; k <= ki2; k++) {
				phi1[j][k] =
				    0.40e+00 * (u[ibeg][j][k][4] -
						0.50 *
						(((u[ibeg][j][k][1]) *
						  (u[ibeg][j][k][1])) +
						 ((u[ibeg][j][k][2]) *
						  (u[ibeg][j][k][2])) +
						 ((u[ibeg][j][k][3]) *
						  (u[ibeg][j][k][3]))) /
						u[ibeg][j][k][0]);
			}
		}
	}
	iglob = ifin;
	if (iglob == ii2) {
		for (j = jbeg; j <= jfin; j++) {
			jglob = j;
			for (k = ki1; k <= ki2; k++) {
				phi2[j][k] =
				    0.40e+00 * (u[ifin][j][k][4] -
						0.50 *
						(((u[ifin][j][k][1]) *
						  (u[ifin][j][k][1])) +
						 ((u[ifin][j][k][2]) *
						  (u[ifin][j][k][2])) +
						 ((u[ifin][j][k][3]) *
						  (u[ifin][j][k][3]))) /
						u[ifin][j][k][0]);
			}
		}
	}
	frc3 = 0.0;
	for (j = jbeg; j <= jfin1; j++) {
		for (k = ki1; k <= ki2 - 1; k++) {
			frc3 =
			    frc3 + (phi1[j][k] + phi1[j + 1][k] +
				    phi1[j][k + 1] + phi1[j + 1][k + 1] +
				    phi2[j][k] + phi2[j + 1][k] + phi2[j][k +
									  1] +
				    phi2[j + 1][k + 1]);
		}
	}
	frc3 = deta * dzeta * frc3;
	frc = 0.25 * (frc1 + frc2 + frc3);
}

static void read_input()
{
	FILE *fp;
	printf
	    ("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version - LU Benchmark\n\n");
	fp = fopen("inputlu.data", "r");
	if (fp != ((void *)0)) {
		printf(" Reading from input file inputlu.data\n");
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%d%d", &ipr, &inorm);
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%d", &itmax);
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%lf", &dt);
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%lf", &omega);
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%lf%lf%lf%lf%lf", &tolrsd[0], &tolrsd[1],
		       &tolrsd[2], &tolrsd[3], &tolrsd[4]);
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		while (fgetc(fp) != '\n') ;
		fscanf(fp, "%d%d%d", &nx0, &ny0, &nz0);
		while (fgetc(fp) != '\n') ;
		fclose(fp);
	} else {
		ipr = 1;
		inorm = 300;
		itmax = 300;
		dt = 1.5e-3;
		omega = 1.2;
		tolrsd[0] = 1.0e-8;
		tolrsd[1] = 1.0e-8;
		tolrsd[2] = 1.0e-8;
		tolrsd[3] = 1.0e-8;
		tolrsd[4] = 1.0e-8;
		nx0 = 33;
		ny0 = 33;
		nz0 = 33;
	}
	if (nx0 < 4 || ny0 < 4 || nz0 < 4) {
		printf
		    ("     PROBLEM SIZE IS TOO SMALL - \n     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5\n");
		exit(1);
	}
	if (nx0 > 33 || ny0 > 33 || nz0 > 33) {
		printf
		    ("     PROBLEM SIZE IS TOO LARGE - \n     NX, NY AND NZ SHOULD BE EQUAL TO \n     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY\n");
		exit(1);
	}
	printf(" Size: %3dx%3dx%3d\n", nx0, ny0, nz0);
	printf(" Iterations: %3d\n", itmax);
}

static void rhs()
{
	int i, j, k, m;
	int L1, L2;
	int ist1, iend1;
	int jst1, jend1;
	double q;
	double u21, u31, u41;
	double tmp;
	double u21i, u31i, u41i, u51i;
	double u21j, u31j, u41j, u51j;
	double u21k, u31k, u41k, u51k;
	double u21im1, u31im1, u41im1, u51im1;
	double u21jm1, u31jm1, u41jm1, u51jm1;
	double u21km1, u31km1, u41km1, u51km1;
	//--------------------------------------------------------------
	//Loop defined at line 2364 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz - 1) - (0) + 1;
		_ocl_gws[1] = (ny - 1) - (0) + 1;
		_ocl_gws[2] = (nx - 1) - (0) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_rhs_0, 0, __ocl_buffer_rsd);
		oclSetKernelArgBuffer(__ocl_rhs_0, 1, __ocl_buffer_frct);
		int __ocl_k_bound = nz - 1;
		oclSetKernelArg(__ocl_rhs_0, 2, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = ny - 1;
		oclSetKernelArg(__ocl_rhs_0, 3, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = nx - 1;
		oclSetKernelArg(__ocl_rhs_0, 4, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rsd);
		//------------------------------------------
		//Write set (END) 
		//------------------------------------------
		//------------------------------------------
		//Read only variables (BEGIN) 
		//------------------------------------------
		oclDevReads(__ocl_buffer_frct);
		//------------------------------------------
		//Read only variables (END) 
		//------------------------------------------

		oclRunKernel(__ocl_rhs_0, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	L1 = 0;
	L2 = nx - 1;
	//--------------------------------------------------------------
	//Loop defined at line 2382 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (jend) - (jst) + 1;
		_ocl_gws[2] = (L2) - (L1) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_rhs_1, 0, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_rhs_1, 1, __ocl_buffer_u);
		oclSetKernelArg(__ocl_rhs_1, 2, sizeof(int), &jst);
		oclSetKernelArg(__ocl_rhs_1, 3, sizeof(int), &L1);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_rhs_1, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_rhs_1, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = L2;
		oclSetKernelArg(__ocl_rhs_1, 6, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
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

		oclRunKernel(__ocl_rhs_1, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 2403 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (jend) - (jst) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_rhs_2, 0, sizeof(int), &ist);
		oclSetKernelArg(__ocl_rhs_2, 1, sizeof(int), &iend);
		oclSetKernelArgBuffer(__ocl_rhs_2, 2, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_rhs_2, 3, sizeof(double), &tx2);
		oclSetKernelArgBuffer(__ocl_rhs_2, 4, __ocl_buffer_flux);
		oclSetKernelArg(__ocl_rhs_2, 5, sizeof(int), &nx);
		oclSetKernelArgBuffer(__ocl_rhs_2, 6, __ocl_buffer_u);
		oclSetKernelArg(__ocl_rhs_2, 7, sizeof(double), &tx3);
		oclSetKernelArg(__ocl_rhs_2, 8, sizeof(double), &dx1);
		oclSetKernelArg(__ocl_rhs_2, 9, sizeof(double), &tx1);
		oclSetKernelArg(__ocl_rhs_2, 10, sizeof(double), &dx2);
		oclSetKernelArg(__ocl_rhs_2, 11, sizeof(double), &dx3);
		oclSetKernelArg(__ocl_rhs_2, 12, sizeof(double), &dx4);
		oclSetKernelArg(__ocl_rhs_2, 13, sizeof(double), &dx5);
		oclSetKernelArg(__ocl_rhs_2, 14, sizeof(double), &dssp);
		oclSetKernelArg(__ocl_rhs_2, 15, sizeof(int), &jst);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_rhs_2, 16, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_rhs_2, 17, sizeof(int), &__ocl_j_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rsd);
		oclDevWrites(__ocl_buffer_flux);
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

		oclRunKernel(__ocl_rhs_2, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	L1 = 0;
	L2 = ny - 1;
	//--------------------------------------------------------------
	//Loop defined at line 2520 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[3];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (L2) - (L1) + 1;
		_ocl_gws[2] = (iend) - (ist) + 1;

		oclGetWorkSize(3, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_rhs_3, 0, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_rhs_3, 1, __ocl_buffer_u);
		oclSetKernelArg(__ocl_rhs_3, 2, sizeof(int), &L1);
		oclSetKernelArg(__ocl_rhs_3, 3, sizeof(int), &ist);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_rhs_3, 4, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = L2;
		oclSetKernelArg(__ocl_rhs_3, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_rhs_3, 6, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
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

		oclRunKernel(__ocl_rhs_3, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 2541 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz - 2) - (1) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_rhs_4, 0, sizeof(int), &jst);
		oclSetKernelArg(__ocl_rhs_4, 1, sizeof(int), &jend);
		oclSetKernelArgBuffer(__ocl_rhs_4, 2, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_rhs_4, 3, sizeof(double), &ty2);
		oclSetKernelArgBuffer(__ocl_rhs_4, 4, __ocl_buffer_flux);
		oclSetKernelArg(__ocl_rhs_4, 5, sizeof(int), &ny);
		oclSetKernelArgBuffer(__ocl_rhs_4, 6, __ocl_buffer_u);
		oclSetKernelArg(__ocl_rhs_4, 7, sizeof(double), &ty3);
		oclSetKernelArg(__ocl_rhs_4, 8, sizeof(double), &dy1);
		oclSetKernelArg(__ocl_rhs_4, 9, sizeof(double), &ty1);
		oclSetKernelArg(__ocl_rhs_4, 10, sizeof(double), &dy2);
		oclSetKernelArg(__ocl_rhs_4, 11, sizeof(double), &dy3);
		oclSetKernelArg(__ocl_rhs_4, 12, sizeof(double), &dy4);
		oclSetKernelArg(__ocl_rhs_4, 13, sizeof(double), &dy5);
		oclSetKernelArg(__ocl_rhs_4, 14, sizeof(double), &dssp);
		oclSetKernelArg(__ocl_rhs_4, 15, sizeof(int), &ist);
		int __ocl_k_bound = nz - 2;
		oclSetKernelArg(__ocl_rhs_4, 16, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_rhs_4, 17, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_rsd);
		oclDevWrites(__ocl_buffer_flux);
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

		oclRunKernel(__ocl_rhs_4, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

	//--------------------------------------------------------------
	//Loop defined at line 2656 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (jend) - (jst) + 1;
		_ocl_gws[1] = (iend) - (ist) + 1;

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_rhs_5, 0, sizeof(int), &nz);
		oclSetKernelArgBuffer(__ocl_rhs_5, 1, __ocl_buffer_flux);
		oclSetKernelArgBuffer(__ocl_rhs_5, 2, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_rhs_5, 3, __ocl_buffer_rsd);
		oclSetKernelArg(__ocl_rhs_5, 4, sizeof(double), &tz2);
		oclSetKernelArg(__ocl_rhs_5, 5, sizeof(double), &tz3);
		oclSetKernelArg(__ocl_rhs_5, 6, sizeof(double), &dz1);
		oclSetKernelArg(__ocl_rhs_5, 7, sizeof(double), &tz1);
		oclSetKernelArg(__ocl_rhs_5, 8, sizeof(double), &dz2);
		oclSetKernelArg(__ocl_rhs_5, 9, sizeof(double), &dz3);
		oclSetKernelArg(__ocl_rhs_5, 10, sizeof(double), &dz4);
		oclSetKernelArg(__ocl_rhs_5, 11, sizeof(double), &dz5);
		oclSetKernelArg(__ocl_rhs_5, 12, sizeof(double), &dssp);
		oclSetKernelArg(__ocl_rhs_5, 13, sizeof(int), &jst);
		oclSetKernelArg(__ocl_rhs_5, 14, sizeof(int), &ist);
		int __ocl_j_bound = jend;
		oclSetKernelArg(__ocl_rhs_5, 15, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = iend;
		oclSetKernelArg(__ocl_rhs_5, 16, sizeof(int), &__ocl_i_bound);
		//------------------------------------------
		//OpenCL kernel arguments (END) 
		//------------------------------------------

		//------------------------------------------
		//Write set (BEGIN) 
		//------------------------------------------
		oclDevWrites(__ocl_buffer_flux);
		oclDevWrites(__ocl_buffer_rsd);
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

		oclRunKernel(__ocl_rhs_5, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
	}

}

static void setbv()
{
	int i, j, k;
	double tmp[5];
	DECLARE_LOCALVAR_OCL_BUFFER(tmp, double, (5));
	//--------------------------------------------------------------
	//Loop defined at line 2795 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (ny) - (0);
		_ocl_gws[1] = (nx) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_setbv_0, 0, __ocl_buffer_u);
		oclSetKernelArg(__ocl_setbv_0, 1, sizeof(int), &nz);
		oclSetKernelArgBuffer(__ocl_setbv_0, 2, __ocl_buffer_ce);
		oclSetKernelArg(__ocl_setbv_0, 3, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_setbv_0, 4, sizeof(int), &ny0);
		int __ocl_j_bound = ny;
		oclSetKernelArg(__ocl_setbv_0, 5, sizeof(int), &__ocl_j_bound);
		int __ocl_i_bound = nx;
		oclSetKernelArg(__ocl_setbv_0, 6, sizeof(int), &__ocl_i_bound);
		oclSetKernelArgBuffer(__ocl_setbv_0, 7, rd_oclb_u);
		oclSetKernelArgBuffer(__ocl_setbv_0, 8, wr_oclb_u);
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

		oclRunKernel(__ocl_setbv_0, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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
	//Loop defined at line 2819 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz) - (0);
		_ocl_gws[1] = (nx) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_setbv_1, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_setbv_1, 1, __ocl_buffer_ce);
		oclSetKernelArg(__ocl_setbv_1, 2, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_setbv_1, 3, sizeof(int), &ny0);
		oclSetKernelArg(__ocl_setbv_1, 4, sizeof(int), &nz);
		int __ocl_k_bound = nz;
		oclSetKernelArg(__ocl_setbv_1, 5, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = nx;
		oclSetKernelArg(__ocl_setbv_1, 6, sizeof(int), &__ocl_i_bound);
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

		oclRunKernel(__ocl_setbv_1, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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
	//Loop defined at line 2833 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz) - (0);
		_ocl_gws[1] = (nx) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_setbv_2, 0, sizeof(int), &ny0);
		oclSetKernelArgBuffer(__ocl_setbv_2, 1, __ocl_buffer_u);
		oclSetKernelArg(__ocl_setbv_2, 2, sizeof(int), &ny);
		oclSetKernelArgBuffer(__ocl_setbv_2, 3, __ocl_buffer_ce);
		oclSetKernelArg(__ocl_setbv_2, 4, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_setbv_2, 5, sizeof(int), &nz);
		int __ocl_k_bound = nz;
		oclSetKernelArg(__ocl_setbv_2, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_i_bound = nx;
		oclSetKernelArg(__ocl_setbv_2, 7, sizeof(int), &__ocl_i_bound);
		oclSetKernelArgBuffer(__ocl_setbv_2, 8, rd_oclb_u);
		oclSetKernelArgBuffer(__ocl_setbv_2, 9, wr_oclb_u);
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

		oclRunKernel(__ocl_setbv_2, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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
	//Loop defined at line 2850 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz) - (0);
		_ocl_gws[1] = (ny) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArgBuffer(__ocl_setbv_3, 0, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_setbv_3, 1, __ocl_buffer_ce);
		oclSetKernelArg(__ocl_setbv_3, 2, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_setbv_3, 3, sizeof(int), &ny0);
		oclSetKernelArg(__ocl_setbv_3, 4, sizeof(int), &nz);
		int __ocl_k_bound = nz;
		oclSetKernelArg(__ocl_setbv_3, 5, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = ny;
		oclSetKernelArg(__ocl_setbv_3, 6, sizeof(int), &__ocl_j_bound);
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

		oclRunKernel(__ocl_setbv_3, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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
	//Loop defined at line 2864 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[2];
		_ocl_gws[0] = (nz) - (0);
		_ocl_gws[1] = (ny) - (0);

		oclGetWorkSize(2, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_setbv_4, 0, sizeof(int), &nx0);
		oclSetKernelArgBuffer(__ocl_setbv_4, 1, __ocl_buffer_u);
		oclSetKernelArg(__ocl_setbv_4, 2, sizeof(int), &nx);
		oclSetKernelArgBuffer(__ocl_setbv_4, 3, __ocl_buffer_ce);
		oclSetKernelArg(__ocl_setbv_4, 4, sizeof(int), &ny0);
		oclSetKernelArg(__ocl_setbv_4, 5, sizeof(int), &nz);
		int __ocl_k_bound = nz;
		oclSetKernelArg(__ocl_setbv_4, 6, sizeof(int), &__ocl_k_bound);
		int __ocl_j_bound = ny;
		oclSetKernelArg(__ocl_setbv_4, 7, sizeof(int), &__ocl_j_bound);
		oclSetKernelArgBuffer(__ocl_setbv_4, 8, rd_oclb_u);
		oclSetKernelArgBuffer(__ocl_setbv_4, 9, wr_oclb_u);
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

		oclRunKernel(__ocl_setbv_4, 2, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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

}

static void setcoeff()
{
	dxi = 1.0 / (nx0 - 1);
	deta = 1.0 / (ny0 - 1);
	dzeta = 1.0 / (nz0 - 1);
	tx1 = 1.0 / (dxi * dxi);
	tx2 = 1.0 / (2.0 * dxi);
	tx3 = 1.0 / dxi;
	ty1 = 1.0 / (deta * deta);
	ty2 = 1.0 / (2.0 * deta);
	ty3 = 1.0 / deta;
	tz1 = 1.0 / (dzeta * dzeta);
	tz2 = 1.0 / (2.0 * dzeta);
	tz3 = 1.0 / dzeta;
	ii1 = 1;
	ii2 = nx0 - 2;
	ji1 = 1;
	ji2 = ny0 - 3;
	ki1 = 2;
	ki2 = nz0 - 2;
	dx1 = 0.75;
	dx2 = dx1;
	dx3 = dx1;
	dx4 = dx1;
	dx5 = dx1;
	dy1 = 0.75;
	dy2 = dy1;
	dy3 = dy1;
	dy4 = dy1;
	dy5 = dy1;
	dz1 = 1.00;
	dz2 = dz1;
	dz3 = dz1;
	dz4 = dz1;
	dz5 = dz1;
	dssp =
	    ((((dx1) >
	       ((((dy1) > (dz1)) ? (dy1) : (dz1)))) ? (dx1) : ((((dy1) >
								 (dz1)) ? (dy1)
								: (dz1))))) /
	    4.0;
	ce[0][0] = 2.0;
	ce[0][1] = 0.0;
	ce[0][2] = 0.0;
	ce[0][3] = 4.0;
	ce[0][4] = 5.0;
	ce[0][5] = 3.0;
	ce[0][6] = 5.0e-01;
	ce[0][7] = 2.0e-02;
	ce[0][8] = 1.0e-02;
	ce[0][9] = 3.0e-02;
	ce[0][10] = 5.0e-01;
	ce[0][11] = 4.0e-01;
	ce[0][12] = 3.0e-01;
	ce[1][0] = 1.0;
	ce[1][1] = 0.0;
	ce[1][2] = 0.0;
	ce[1][3] = 0.0;
	ce[1][4] = 1.0;
	ce[1][5] = 2.0;
	ce[1][6] = 3.0;
	ce[1][7] = 1.0e-02;
	ce[1][8] = 3.0e-02;
	ce[1][9] = 2.0e-02;
	ce[1][10] = 4.0e-01;
	ce[1][11] = 3.0e-01;
	ce[1][12] = 5.0e-01;
	ce[2][0] = 2.0;
	ce[2][1] = 2.0;
	ce[2][2] = 0.0;
	ce[2][3] = 0.0;
	ce[2][4] = 0.0;
	ce[2][5] = 2.0;
	ce[2][6] = 3.0;
	ce[2][7] = 4.0e-02;
	ce[2][8] = 3.0e-02;
	ce[2][9] = 5.0e-02;
	ce[2][10] = 3.0e-01;
	ce[2][11] = 5.0e-01;
	ce[2][12] = 4.0e-01;
	ce[3][0] = 2.0;
	ce[3][1] = 2.0;
	ce[3][2] = 0.0;
	ce[3][3] = 0.0;
	ce[3][4] = 0.0;
	ce[3][5] = 2.0;
	ce[3][6] = 3.0;
	ce[3][7] = 3.0e-02;
	ce[3][8] = 5.0e-02;
	ce[3][9] = 4.0e-02;
	ce[3][10] = 2.0e-01;
	ce[3][11] = 1.0e-01;
	ce[3][12] = 3.0e-01;
	ce[4][0] = 5.0;
	ce[4][1] = 4.0;
	ce[4][2] = 3.0;
	ce[4][3] = 2.0;
	ce[4][4] = 1.0e-01;
	ce[4][5] = 4.0e-01;
	ce[4][6] = 3.0e-01;
	ce[4][7] = 5.0e-02;
	ce[4][8] = 4.0e-02;
	ce[4][9] = 3.0e-02;
	ce[4][10] = 1.0e-01;
	ce[4][11] = 3.0e-01;
	ce[4][12] = 2.0e-01;
}

static void setiv()
{
	int i, j, k, m;
	int iglob, jglob;
	double xi, eta, zeta;
	double pxi, peta, pzeta;
	double ue_1jk[5], ue_nx0jk[5], ue_i1k[5], ue_iny0k[5], ue_ij1[5],
	    ue_ijnz[5];
	DECLARE_LOCALVAR_OCL_BUFFER(ue_1jk, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(ue_nx0jk, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(ue_i1k, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(ue_iny0k, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(ue_ij1, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(ue_ijnz, double, (5));
	//--------------------------------------------------------------
	//Loop defined at line 3045 of lu.c
	//--------------------------------------------------------------
	{
		//------------------------------------------
		//OpenCL kernel arguments (BEGIN) 
		//------------------------------------------
		size_t _ocl_gws[1];
		_ocl_gws[0] = (ny) - (0);

		oclGetWorkSize(1, _ocl_gws, NULL);
		oclSetKernelArg(__ocl_setiv_0, 0, sizeof(int), &nz);
		oclSetKernelArg(__ocl_setiv_0, 1, sizeof(int), &ny0);
		oclSetKernelArg(__ocl_setiv_0, 2, sizeof(int), &nx);
		oclSetKernelArg(__ocl_setiv_0, 3, sizeof(int), &nx0);
		oclSetKernelArg(__ocl_setiv_0, 4, sizeof(int), &m);
		oclSetKernelArgBuffer(__ocl_setiv_0, 5, __ocl_buffer_u);
		oclSetKernelArgBuffer(__ocl_setiv_0, 6, __ocl_buffer_ce);
		int __ocl_j_bound = ny;
		oclSetKernelArg(__ocl_setiv_0, 7, sizeof(int), &__ocl_j_bound);
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

		oclRunKernel(__ocl_setiv_0, 1, _ocl_gws);
#ifdef __STRICT_SYNC__
		oclSync();
#endif
		//---------------------------------------
		// GPU TLS Checking (BEGIN)
		//---------------------------------------
		{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
			{
				size_t __ocl_gws_u = (5 * 33 * 33 * 33);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned), &__ocl_gws_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_u);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_u);
				oclDevWrites(rd_oclb_u);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_u);
			}
#endif
#ifdef __RUN_CHECKING_KERNEL__
//Checking ce
			{
				size_t __ocl_gws_ce = (13 * 5);
				oclSetKernelArg(__ocl_tls_1D_checking, 0,
						sizeof(unsigned),
						&__ocl_gws_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
						      rd_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
						      wr_oclb_ce);
				oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
						      __oclb_gpu_tls_conflict_flag);
				oclDevWrites(wr_oclb_ce);
				oclDevWrites(rd_oclb_ce);
				oclDevWrites(__oclb_gpu_tls_conflict_flag);
				oclRunKernel(__ocl_tls_1D_checking, 1,
					     &__ocl_gws_ce);
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

}

static void ssor()
{
	int i, j, k, m;
	int istep;
	double tmp;
	double delunm[5], tv[33][33][5];
	DECLARE_LOCALVAR_OCL_BUFFER(delunm, double, (5));
	DECLARE_LOCALVAR_OCL_BUFFER(tv, double, (33 * 33 * 5));
	tmp = 1.0 / (omega * (2.0 - omega));
	{
		//--------------------------------------------------------------
		//Loop defined at line 3111 of lu.c
		//--------------------------------------------------------------
		{
			//------------------------------------------
			//OpenCL kernel arguments (BEGIN) 
			//------------------------------------------
			size_t _ocl_gws[3];
			_ocl_gws[0] = (5) - (0);
			_ocl_gws[1] = (33) - (0);
			_ocl_gws[2] = (33) - (0);

			oclGetWorkSize(3, _ocl_gws, NULL);
			oclSetKernelArgBuffer(__ocl_ssor_0, 0, __ocl_buffer_a);
			oclSetKernelArgBuffer(__ocl_ssor_0, 1, __ocl_buffer_b);
			oclSetKernelArgBuffer(__ocl_ssor_0, 2, __ocl_buffer_c);
			oclSetKernelArgBuffer(__ocl_ssor_0, 3, __ocl_buffer_d);
			//------------------------------------------
			//OpenCL kernel arguments (END) 
			//------------------------------------------

			//------------------------------------------
			//Write set (BEGIN) 
			//------------------------------------------
			oclDevWrites(__ocl_buffer_a);
			oclDevWrites(__ocl_buffer_b);
			oclDevWrites(__ocl_buffer_c);
			oclDevWrites(__ocl_buffer_d);
			//------------------------------------------
			//Write set (END) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (BEGIN) 
			//------------------------------------------
			//------------------------------------------
			//Read only variables (END) 
			//------------------------------------------

			oclRunKernel(__ocl_ssor_0, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
			oclSync();
#endif
		}

		rhs();
		l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd,
		       __ocl_buffer_rsd, rsdnm, NULL);
	}
	timer_clear(1);
	timer_start(1);
	{
		for (istep = 1; istep <= itmax; istep++) {
			if (istep % 20 == 0 || istep == itmax || istep == 1) {
				printf(" Time step %4d\n", istep);
			}
			//--------------------------------------------------------------
			//Loop defined at line 3157 of lu.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[3];
				_ocl_gws[0] = (nz - 2) - (1) + 1;
				_ocl_gws[1] = (jend) - (jst) + 1;
				_ocl_gws[2] = (iend) - (ist) + 1;

				oclGetWorkSize(3, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_ssor_1, 0,
						      __ocl_buffer_rsd);
				oclSetKernelArg(__ocl_ssor_1, 1, sizeof(double),
						&dt);
				oclSetKernelArg(__ocl_ssor_1, 2, sizeof(int),
						&jst);
				oclSetKernelArg(__ocl_ssor_1, 3, sizeof(int),
						&ist);
				int __ocl_k_bound = nz - 2;
				oclSetKernelArg(__ocl_ssor_1, 4, sizeof(int),
						&__ocl_k_bound);
				int __ocl_j_bound = jend;
				oclSetKernelArg(__ocl_ssor_1, 5, sizeof(int),
						&__ocl_j_bound);
				int __ocl_i_bound = iend;
				oclSetKernelArg(__ocl_ssor_1, 6, sizeof(int),
						&__ocl_i_bound);
				//------------------------------------------
				//OpenCL kernel arguments (END) 
				//------------------------------------------

				//------------------------------------------
				//Write set (BEGIN) 
				//------------------------------------------
				oclDevWrites(__ocl_buffer_rsd);
				//------------------------------------------
				//Write set (END) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (BEGIN) 
				//------------------------------------------
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_ssor_1, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
				oclSync();
#endif
			}

			{
				for (k = 1; k <= nz - 2; k++) {
					jacld(k);
					blts(nx, ny, nz, k, omega, rsd,
					     __ocl_buffer_rsd, a,
					     __ocl_buffer_a, b, __ocl_buffer_b,
					     c, __ocl_buffer_c, d,
					     __ocl_buffer_d, ist, iend, jst,
					     jend, nx0, ny0);
				}
			}
			{
				for (k = nz - 2; k >= 1; k--) {
					jacu(k);

					buts(nx, ny, nz, k, omega, rsd,
					     __ocl_buffer_rsd, tv,
					     __ocl_buffer_tv, d, __ocl_buffer_d,
					     a, __ocl_buffer_a, b,
					     __ocl_buffer_b, c, __ocl_buffer_c,
					     ist, iend, jst, jend, nx0, ny0);
				}
			}
			//--------------------------------------------------------------
			//Loop defined at line 3216 of lu.c
			//--------------------------------------------------------------
			{
				//------------------------------------------
				//OpenCL kernel arguments (BEGIN) 
				//------------------------------------------
				size_t _ocl_gws[3];
				_ocl_gws[0] = (nz - 2) - (1) + 1;
				_ocl_gws[1] = (jend) - (jst) + 1;
				_ocl_gws[2] = (iend) - (ist) + 1;

				oclGetWorkSize(3, _ocl_gws, NULL);
				oclSetKernelArgBuffer(__ocl_ssor_2, 0,
						      __ocl_buffer_u);
				oclSetKernelArg(__ocl_ssor_2, 1, sizeof(double),
						&tmp);
				oclSetKernelArgBuffer(__ocl_ssor_2, 2,
						      __ocl_buffer_rsd);
				oclSetKernelArg(__ocl_ssor_2, 3, sizeof(int),
						&jst);
				oclSetKernelArg(__ocl_ssor_2, 4, sizeof(int),
						&ist);
				int __ocl_k_bound = nz - 2;
				oclSetKernelArg(__ocl_ssor_2, 5, sizeof(int),
						&__ocl_k_bound);
				int __ocl_j_bound = jend;
				oclSetKernelArg(__ocl_ssor_2, 6, sizeof(int),
						&__ocl_j_bound);
				int __ocl_i_bound = iend;
				oclSetKernelArg(__ocl_ssor_2, 7, sizeof(int),
						&__ocl_i_bound);
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
				oclDevReads(__ocl_buffer_rsd);
				//------------------------------------------
				//Read only variables (END) 
				//------------------------------------------

				oclRunKernel(__ocl_ssor_2, 3, _ocl_gws);
#ifdef __STRICT_SYNC__
				oclSync();
#endif
			}

			if (istep % inorm == 0) {

				l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd,
				       __ocl_buffer_rsd, delunm,
				       __ocl_buffer_delunm);
			}
			rhs();
			if ((istep % inorm == 0) || (istep == itmax)) {
				l2norm(nx0, ny0, nz0, ist, iend, jst, jend, rsd,
				       __ocl_buffer_rsd, rsdnm, NULL);
			}
			if ((rsdnm[0] < tolrsd[0]) && (rsdnm[1] < tolrsd[1])
			    && (rsdnm[2] < tolrsd[2]) && (rsdnm[3] < tolrsd[3])
			    && (rsdnm[4] < tolrsd[4])) {
				exit(1);
			}
		}
	}
	timer_stop(1);
	maxtime = timer_read(1);
}

static void verify(double xcr[5], ocl_buffer * __ocl_buffer_xcr, double xce[5],
		   ocl_buffer * __ocl_buffer_xce, double xci, char *class,
		   ocl_buffer * __ocl_buffer_class, boolean * verified,
		   ocl_buffer * __ocl_buffer_verified)
{
	{
		double xcrref[5], xceref[5], xciref, xcrdif[5], xcedif[5],
		    xcidif, epsilon, dtref;
		DECLARE_LOCALVAR_OCL_BUFFER(xcrref, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xceref, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xcrdif, double, (5));
		DECLARE_LOCALVAR_OCL_BUFFER(xcedif, double, (5));
		int m;
		epsilon = 1.0e-08;
		*class = 'U';
		*verified = 1;
		for (m = 0; m < 5; m++) {
			xcrref[m] = 1.0;
			xceref[m] = 1.0;
		}
		xciref = 1.0;
		if (nx0 == 12 && ny0 == 12 && nz0 == 12 && itmax == 50) {
			*class = 'S';
			dtref = 5.0e-1;
			xcrref[0] = 1.6196343210976702e-02;
			xcrref[1] = 2.1976745164821318e-03;
			xcrref[2] = 1.5179927653399185e-03;
			xcrref[3] = 1.5029584435994323e-03;
			xcrref[4] = 3.4264073155896461e-02;
			xceref[0] = 6.4223319957960924e-04;
			xceref[1] = 8.4144342047347926e-05;
			xceref[2] = 5.8588269616485186e-05;
			xceref[3] = 5.8474222595157350e-05;
			xceref[4] = 1.3103347914111294e-03;
			xciref = 7.8418928865937083;
		} else if (nx0 == 33 && ny0 == 33 && nz0 == 33 && itmax == 300) {
			*class = 'W';
			dtref = 1.5e-3;
			xcrref[0] = 0.1236511638192e+02;
			xcrref[1] = 0.1317228477799e+01;
			xcrref[2] = 0.2550120713095e+01;
			xcrref[3] = 0.2326187750252e+01;
			xcrref[4] = 0.2826799444189e+02;
			xceref[0] = 0.4867877144216;
			xceref[1] = 0.5064652880982e-01;
			xceref[2] = 0.9281818101960e-01;
			xceref[3] = 0.8570126542733e-01;
			xceref[4] = 0.1084277417792e+01;
			xciref = 0.1161399311023e+02;
		} else if (nx0 == 64 && ny0 == 64 && nz0 == 64 && itmax == 250) {
			*class = 'A';
			dtref = 2.0e+0;
			xcrref[0] = 7.7902107606689367e+02;
			xcrref[1] = 6.3402765259692870e+01;
			xcrref[2] = 1.9499249727292479e+02;
			xcrref[3] = 1.7845301160418537e+02;
			xcrref[4] = 1.8384760349464247e+03;
			xceref[0] = 2.9964085685471943e+01;
			xceref[1] = 2.8194576365003349;
			xceref[2] = 7.3473412698774742;
			xceref[3] = 6.7139225687777051;
			xceref[4] = 7.0715315688392578e+01;
			xciref = 2.6030925604886277e+01;
		} else if (nx0 == 102 && ny0 == 102 && nz0 == 102
			   && itmax == 250) {
			*class = 'B';
			dtref = 2.0e+0;
			xcrref[0] = 3.5532672969982736e+03;
			xcrref[1] = 2.6214750795310692e+02;
			xcrref[2] = 8.8333721850952190e+02;
			xcrref[3] = 7.7812774739425265e+02;
			xcrref[4] = 7.3087969592545314e+03;
			xceref[0] = 1.1401176380212709e+02;
			xceref[1] = 8.1098963655421574;
			xceref[2] = 2.8480597317698308e+01;
			xceref[3] = 2.5905394567832939e+01;
			xceref[4] = 2.6054907504857413e+02;
			xciref = 4.7887162703308227e+01;
		} else if (nx0 == 162 && ny0 == 162 && nz0 == 162
			   && itmax == 250) {
			*class = 'C';
			dtref = 2.0e+0;
			xcrref[0] = 1.03766980323537846e+04;
			xcrref[1] = 8.92212458801008552e+02;
			xcrref[2] = 2.56238814582660871e+03;
			xcrref[3] = 2.19194343857831427e+03;
			xcrref[4] = 1.78078057261061185e+04;
			xceref[0] = 2.15986399716949279e+02;
			xceref[1] = 1.55789559239863600e+01;
			xceref[2] = 5.41318863077207766e+01;
			xceref[3] = 4.82262643154045421e+01;
			xceref[4] = 4.55902910043250358e+02;
			xciref = 6.66404553572181300e+01;
		} else {
			*verified = 0;
		}
		for (m = 0; m < 5; m++) {
			xcrdif[m] = fabs((xcr[m] - xcrref[m]) / xcrref[m]);
			xcedif[m] = fabs((xce[m] - xceref[m]) / xceref[m]);
		}
		xcidif = fabs((xci - xciref) / xciref);
		if (*class != 'U') {
			printf
			    ("\n Verification being performed for class %1c\n",
			     *class);
			printf(" Accuracy setting for epsilon = %20.13e\n",
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
				printf("          %2d  %20.13e\n", m, xcr[m]);
			} else if (xcrdif[m] > epsilon) {
				*verified = 0;
				printf(" FAILURE: %2d  %20.13e%20.13e%20.13e\n",
				       m, xcr[m], xcrref[m], xcrdif[m]);
			} else {
				printf("          %2d  %20.13e%20.13e%20.13e\n",
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
				printf("          %2d  %20.13e\n", m, xce[m]);
			} else if (xcedif[m] > epsilon) {
				*verified = 0;
				printf(" FAILURE: %2d  %20.13e%20.13e%20.13e\n",
				       m, xce[m], xceref[m], xcedif[m]);
			} else {
				printf("          %2d  %20.13e%20.13e%20.13e\n",
				       m, xce[m], xceref[m], xcedif[m]);
			}
		}
		if (*class != 'U') {
			printf(" Comparison of surface integral\n");
		} else {
			printf(" Surface integral\n");
		}
		if (*class == 'U') {
			printf("              %20.13e\n", xci);
		} else if (xcidif > epsilon) {
			*verified = 0;
			printf(" FAILURE:     %20.13e%20.13e%20.13e\n", xci,
			       xciref, xcidif);
		} else {
			printf("              %20.13e%20.13e%20.13e\n", xci,
			       xciref, xcidif);
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

	__ocl_program = oclBuildProgram("lu.cl");
	if (unlikely(!__ocl_program)) {
		fprintf(stderr, "Failed to build the program:%d.\n", err);
		exit(err);
	}

	__ocl_blts_0 = oclCreateKernel(__ocl_program, "blts_0");
	DYN_PROGRAM_CHECK(__ocl_blts_0);
	__ocl_blts_1 = oclCreateKernel(__ocl_program, "blts_1");
	DYN_PROGRAM_CHECK(__ocl_blts_1);
	__ocl_buts_0 = oclCreateKernel(__ocl_program, "buts_0");
	DYN_PROGRAM_CHECK(__ocl_buts_0);
	__ocl_buts_1 = oclCreateKernel(__ocl_program, "buts_1");
	DYN_PROGRAM_CHECK(__ocl_buts_1);
	__ocl_erhs_0 = oclCreateKernel(__ocl_program, "erhs_0");
	DYN_PROGRAM_CHECK(__ocl_erhs_0);
	__ocl_erhs_1 = oclCreateKernel(__ocl_program, "erhs_1");
	DYN_PROGRAM_CHECK(__ocl_erhs_1);
	__ocl_erhs_2 = oclCreateKernel(__ocl_program, "erhs_2");
	DYN_PROGRAM_CHECK(__ocl_erhs_2);
	__ocl_erhs_3 = oclCreateKernel(__ocl_program, "erhs_3");
	DYN_PROGRAM_CHECK(__ocl_erhs_3);
	__ocl_erhs_4 = oclCreateKernel(__ocl_program, "erhs_4");
	DYN_PROGRAM_CHECK(__ocl_erhs_4);
	__ocl_erhs_5 = oclCreateKernel(__ocl_program, "erhs_5");
	DYN_PROGRAM_CHECK(__ocl_erhs_5);
	__ocl_erhs_6 = oclCreateKernel(__ocl_program, "erhs_6");
	DYN_PROGRAM_CHECK(__ocl_erhs_6);
	__ocl_jacld_0 = oclCreateKernel(__ocl_program, "jacld_0");
	DYN_PROGRAM_CHECK(__ocl_jacld_0);
	__ocl_jacu_0 = oclCreateKernel(__ocl_program, "jacu_0");
	DYN_PROGRAM_CHECK(__ocl_jacu_0);
	__ocl_l2norm_0_reduction_step0 =
	    oclCreateKernel(__ocl_program, "l2norm_0_reduction_step0");
	DYN_PROGRAM_CHECK(__ocl_l2norm_0_reduction_step0);
	__ocl_l2norm_0_reduction_step1 =
	    oclCreateKernel(__ocl_program, "l2norm_0_reduction_step1");
	DYN_PROGRAM_CHECK(__ocl_l2norm_0_reduction_step1);
	__ocl_l2norm_0_reduction_step2 =
	    oclCreateKernel(__ocl_program, "l2norm_0_reduction_step2");
	DYN_PROGRAM_CHECK(__ocl_l2norm_0_reduction_step2);
	__ocl_rhs_0 = oclCreateKernel(__ocl_program, "rhs_0");
	DYN_PROGRAM_CHECK(__ocl_rhs_0);
	__ocl_rhs_1 = oclCreateKernel(__ocl_program, "rhs_1");
	DYN_PROGRAM_CHECK(__ocl_rhs_1);
	__ocl_rhs_2 = oclCreateKernel(__ocl_program, "rhs_2");
	DYN_PROGRAM_CHECK(__ocl_rhs_2);
	__ocl_rhs_3 = oclCreateKernel(__ocl_program, "rhs_3");
	DYN_PROGRAM_CHECK(__ocl_rhs_3);
	__ocl_rhs_4 = oclCreateKernel(__ocl_program, "rhs_4");
	DYN_PROGRAM_CHECK(__ocl_rhs_4);
	__ocl_rhs_5 = oclCreateKernel(__ocl_program, "rhs_5");
	DYN_PROGRAM_CHECK(__ocl_rhs_5);
	__ocl_setbv_0 = oclCreateKernel(__ocl_program, "setbv_0");
	DYN_PROGRAM_CHECK(__ocl_setbv_0);
	__ocl_setbv_1 = oclCreateKernel(__ocl_program, "setbv_1");
	DYN_PROGRAM_CHECK(__ocl_setbv_1);
	__ocl_setbv_2 = oclCreateKernel(__ocl_program, "setbv_2");
	DYN_PROGRAM_CHECK(__ocl_setbv_2);
	__ocl_setbv_3 = oclCreateKernel(__ocl_program, "setbv_3");
	DYN_PROGRAM_CHECK(__ocl_setbv_3);
	__ocl_setbv_4 = oclCreateKernel(__ocl_program, "setbv_4");
	DYN_PROGRAM_CHECK(__ocl_setbv_4);
	__ocl_setiv_0 = oclCreateKernel(__ocl_program, "setiv_0");
	DYN_PROGRAM_CHECK(__ocl_setiv_0);
	__ocl_ssor_0 = oclCreateKernel(__ocl_program, "ssor_0");
	DYN_PROGRAM_CHECK(__ocl_ssor_0);
	__ocl_ssor_1 = oclCreateKernel(__ocl_program, "ssor_1");
	DYN_PROGRAM_CHECK(__ocl_ssor_1);
	__ocl_ssor_2 = oclCreateKernel(__ocl_program, "ssor_2");
	DYN_PROGRAM_CHECK(__ocl_ssor_2);
	__ocl_tls_1D_checking =
	    oclCreateKernel(__ocl_program, "TLS_Checking_1D");
	DYN_PROGRAM_CHECK(__ocl_tls_1D_checking);
	create_ocl_buffers();
}

static void create_ocl_buffers()
{
	__ocl_buffer_frct =
	    oclCreateBuffer(frct, (33 * 33 * 33 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_frct, -1);
	__ocl_buffer_rsd =
	    oclCreateBuffer(rsd, (33 * 33 * 33 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_rsd, -1);
	__ocl_buffer_ce = oclCreateBuffer(ce, (5 * 13) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_ce, -1);
	__ocl_buffer_flux =
	    oclCreateBuffer(flux, (33 * 33 * 33 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_flux, -1);
	__ocl_buffer_u =
	    oclCreateBuffer(u, (33 * 33 * 33 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_u, -1);
	__ocl_buffer_d = oclCreateBuffer(d, (33 * 33 * 5 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_d, -1);
	__ocl_buffer_a = oclCreateBuffer(a, (33 * 33 * 5 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_a, -1);
	__ocl_buffer_b = oclCreateBuffer(b, (33 * 33 * 5 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_b, -1);
	__ocl_buffer_c = oclCreateBuffer(c, (33 * 33 * 5 * 5) * sizeof(double));
	DYN_BUFFER_CHECK(__ocl_buffer_c, -1);

	//------------------------------------------
	// GPU TLS wr/rd buffers (BEGIN)
	//------------------------------------------
	rd_oclb_u = oclCreateBuffer(rd_log_u, (33 * 33 * 33 * 5) * sizeof(int));
	wr_oclb_u = oclCreateBuffer(wr_log_u, (33 * 33 * 33 * 5) * sizeof(int));
	oclHostWrites(rd_oclb_u);
	oclHostWrites(wr_oclb_u);
	DYN_BUFFER_CHECK(rd_oclb_u, -1);
	DYN_BUFFER_CHECK(wr_oclb_u, -1);
	rd_oclb_ce = oclCreateBuffer(rd_log_ce, (5 * 13) * sizeof(int));
	wr_oclb_ce = oclCreateBuffer(wr_log_ce, (5 * 13) * sizeof(int));
	oclHostWrites(rd_oclb_ce);
	oclHostWrites(wr_oclb_ce);
	DYN_BUFFER_CHECK(rd_oclb_ce, -1);
	DYN_BUFFER_CHECK(wr_oclb_ce, -1);
	__oclb_gpu_tls_conflict_flag =
	    oclCreateBuffer(&gpu_tls_conflict_flag, 1 * sizeof(int));

	//------------------------------------------
	// GPU TLS wr/rd buffers (END)
	//------------------------------------------
}

static void sync_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_frct);
	oclHostWrites(__ocl_buffer_rsd);
	oclHostWrites(__ocl_buffer_ce);
	oclHostWrites(__ocl_buffer_flux);
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_d);
	oclHostWrites(__ocl_buffer_a);
	oclHostWrites(__ocl_buffer_b);
	oclHostWrites(__ocl_buffer_c);
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

static void release_ocl_buffers()
{
	oclReleaseBuffer(__ocl_buffer_frct);
	oclReleaseBuffer(__ocl_buffer_rsd);
	oclReleaseBuffer(__ocl_buffer_ce);
	oclReleaseBuffer(__ocl_buffer_flux);
	oclReleaseBuffer(__ocl_buffer_u);
	oclReleaseBuffer(__ocl_buffer_d);
	oclReleaseBuffer(__ocl_buffer_a);
	oclReleaseBuffer(__ocl_buffer_b);
	oclReleaseBuffer(__ocl_buffer_c);
	if (__ocl_buffer_sum0_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum0_l2norm_0);
		__ocl_buffer_sum0_l2norm_0_size = 0;
	}
	if (__ocl_output_sum0_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum0_l2norm_0);
		free(__ocl_output_sum0_l2norm_0);
		__ocl_output_sum0_l2norm_0_size = 0;
	}
	if (__ocl_buffer_sum1_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum1_l2norm_0);
		__ocl_buffer_sum1_l2norm_0_size = 0;
	}
	if (__ocl_output_sum1_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum1_l2norm_0);
		free(__ocl_output_sum1_l2norm_0);
		__ocl_output_sum1_l2norm_0_size = 0;
	}
	if (__ocl_buffer_sum2_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum2_l2norm_0);
		__ocl_buffer_sum2_l2norm_0_size = 0;
	}
	if (__ocl_output_sum2_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum2_l2norm_0);
		free(__ocl_output_sum2_l2norm_0);
		__ocl_output_sum2_l2norm_0_size = 0;
	}
	if (__ocl_buffer_sum3_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum3_l2norm_0);
		__ocl_buffer_sum3_l2norm_0_size = 0;
	}
	if (__ocl_output_sum3_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum3_l2norm_0);
		free(__ocl_output_sum3_l2norm_0);
		__ocl_output_sum3_l2norm_0_size = 0;
	}
	if (__ocl_buffer_sum4_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_buffer_sum4_l2norm_0);
		__ocl_buffer_sum4_l2norm_0_size = 0;
	}
	if (__ocl_output_sum4_l2norm_0_size > 0) {
		oclReleaseBuffer(__ocl_output_buffer_sum4_l2norm_0);
		free(__ocl_output_sum4_l2norm_0);
		__ocl_output_sum4_l2norm_0_size = 0;
	}
	RELEASE_LOCALVAR_OCL_BUFFERS();
}

static void flush_ocl_buffers()
{
	oclHostWrites(__ocl_buffer_frct);
	oclHostWrites(__ocl_buffer_rsd);
	oclHostWrites(__ocl_buffer_ce);
	oclHostWrites(__ocl_buffer_flux);
	oclHostWrites(__ocl_buffer_u);
	oclHostWrites(__ocl_buffer_d);
	oclHostWrites(__ocl_buffer_a);
	oclHostWrites(__ocl_buffer_b);
	oclHostWrites(__ocl_buffer_c);
	if (__ocl_buffer_v_blts) {
		oclHostWrites(__ocl_buffer_v_blts);
	}
	if (__ocl_buffer_ldz_blts) {
		oclHostWrites(__ocl_buffer_ldz_blts);
	}
	if (__ocl_buffer_ldy_blts) {
		oclHostWrites(__ocl_buffer_ldy_blts);
	}
	if (__ocl_buffer_ldx_blts) {
		oclHostWrites(__ocl_buffer_ldx_blts);
	}
	if (__ocl_buffer_d_blts) {
		oclHostWrites(__ocl_buffer_d_blts);
	}
	if (__ocl_buffer_tv_buts) {
		oclHostWrites(__ocl_buffer_tv_buts);
	}
	if (__ocl_buffer_udz_buts) {
		oclHostWrites(__ocl_buffer_udz_buts);
	}
	if (__ocl_buffer_v_buts) {
		oclHostWrites(__ocl_buffer_v_buts);
	}
	if (__ocl_buffer_udy_buts) {
		oclHostWrites(__ocl_buffer_udy_buts);
	}
	if (__ocl_buffer_udx_buts) {
		oclHostWrites(__ocl_buffer_udx_buts);
	}
	if (__ocl_buffer_d_buts) {
		oclHostWrites(__ocl_buffer_d_buts);
	}
	if (__ocl_buffer_v_l2norm) {
		oclHostWrites(__ocl_buffer_v_l2norm);
	}
//SYNC_LOCALVAR_OCL_BUFFERS();
	oclSync();
}

void ocl_gputls_checking()
{
	//---------------------------------------
	// GPU TLS Checking (BEGIN)
	//---------------------------------------
	{
#ifdef __RUN_CHECKING_KERNEL__
//Checking u
		{
			size_t __ocl_gws_u = (5 * 33 * 33 * 33);
			oclSetKernelArg(__ocl_tls_1D_checking, 0,
					sizeof(unsigned), &__ocl_gws_u);
			oclSetKernelArgBuffer(__ocl_tls_1D_checking, 1,
					      rd_oclb_u);
			oclSetKernelArgBuffer(__ocl_tls_1D_checking, 2,
					      wr_oclb_u);
			oclSetKernelArgBuffer(__ocl_tls_1D_checking, 3,
					      __oclb_gpu_tls_conflict_flag);
			oclDevWrites(wr_oclb_u);
			oclDevWrites(rd_oclb_u);
			oclDevWrites(__oclb_gpu_tls_conflict_flag);
			oclRunKernel(__ocl_tls_1D_checking, 1, &__ocl_gws_u);
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
	FILE *prof = fopen("profiling-lu", "w");
	float kernel = 0.0f, buffer = 0.0f;

	kernel += oclDumpKernelProfiling(__ocl_blts_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_blts_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_buts_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_buts_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_2, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_3, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_4, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_5, prof);
	kernel += oclDumpKernelProfiling(__ocl_erhs_6, prof);
	kernel += oclDumpKernelProfiling(__ocl_jacld_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_jacu_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_l2norm_0_reduction_step0, prof);
	kernel += oclDumpKernelProfiling(__ocl_l2norm_0_reduction_step1, prof);
	kernel += oclDumpKernelProfiling(__ocl_l2norm_0_reduction_step2, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_2, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_3, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_4, prof);
	kernel += oclDumpKernelProfiling(__ocl_rhs_5, prof);
	kernel += oclDumpKernelProfiling(__ocl_setbv_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_setbv_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_setbv_2, prof);
	kernel += oclDumpKernelProfiling(__ocl_setbv_3, prof);
	kernel += oclDumpKernelProfiling(__ocl_setbv_4, prof);
	kernel += oclDumpKernelProfiling(__ocl_setiv_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_ssor_0, prof);
	kernel += oclDumpKernelProfiling(__ocl_ssor_1, prof);
	kernel += oclDumpKernelProfiling(__ocl_ssor_2, prof);

	buffer += oclDumpBufferProfiling(__ocl_buffer_frct, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_rsd, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_ce, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_flux, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_u, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_d, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_a, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_b, prof);
	buffer += oclDumpBufferProfiling(__ocl_buffer_c, prof);
	PROFILE_LOCALVAR_OCL_BUFFERS(buffer, prof);

	fprintf(stderr, "-- kernel: %.3fms\n", kernel);
	fprintf(stderr, "-- buffer: %.3fms\n", buffer);
	fclose(prof);
}
#endif

//---------------------------------------------------------------------------
//OCL related routines (END)
//---------------------------------------------------------------------------
