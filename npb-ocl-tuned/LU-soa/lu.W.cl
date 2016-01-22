//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Mon Aug  6 14:04:43 2012
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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define GROUP_SIZE 128

//-------------------------------------------
//Array linearize macros (BEGIN)
//-------------------------------------------
#define CALC_2D_IDX(M1,M2,m1,m2) (((m1)*(M2))+((m2)))
#define CALC_3D_IDX(M1,M2,M3,m1,m2,m3) (((m1)*(M2)*(M3))+((m2)*(M3))+((m3)))
#define CALC_4D_IDX(M1,M2,M3,M4,m1,m2,m3,m4) (((m1)*(M2)*(M3)*(M4))+((m2)*(M3)*(M4))+((m3)*(M4))+((m4)))
#define CALC_5D_IDX(M1,M2,M3,M4,M5,m1,m2,m3,m4,m5) (((m1)*(M2)*(M3)*(M4)*(M5))+((m2)*(M3)*(M4)*(M5))+((m3)*(M4)*(M5))+((m4)*(M5))+((m5)))
#define CALC_6D_IDX(M1,M2,M3,M4,M5,M6,m1,m2,m3,m4,m5,m6) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6))+((m2)*(M3)*(M4)*(M5)*(M6))+((m3)*(M4)*(M5)*(M6))+((m4)*(M5)*(M6))+((m5)*(M6))+((m6)))
#define CALC_7D_IDX(M1,M2,M3,M4,M5,M6,M7,m1,m2,m3,m4,m5,m6,m7) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m3)*(M4)*(M5)*(M6)*(M7))+((m4)*(M5)*(M6)*(M7))+((m5)*(M6)*(M7))+((m6)*(M7))+((m7)))
#define CALC_8D_IDX(M1,M2,M3,M4,M5,M6,M7,M8,m1,m2,m3,m4,m5,m6,m7,m8) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m4)*(M5)*(M6)*(M7)*(M8))+((m5)*(M6)*(M7)*(M8))+((m6)*(M7)*(M8))+((m7)*(M8))+((m8)))
//-------------------------------------------
//Array linearize macros (END)
//-------------------------------------------

//-------------------------------------------------------------------------------
//Functions (BEGIN)
//-------------------------------------------------------------------------------
void exact_g7_e4_e5_e6(int i, int j, int k, double u000ijk[5], int nx0, int ny0,
		       int nz, __global double (*ce)[13]);

//-------------------------------------------------------------------------------
//This is an alias of function: exact
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_g7_e4_e5_e6(int i, int j, int k, double u000ijk[5], int nx0, int ny0,
		       int nz, __global double (*ce)[13])
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
		    ce[m][7] * xi * xi * xi + ce[m][8] * eta * eta * eta +
		    ce[m][9] * zeta * zeta * zeta +
		    ce[m][10] * xi * xi * xi * xi +
		    ce[m][11] * eta * eta * eta * eta +
		    ce[m][12] * zeta * zeta * zeta * zeta;
	}

}

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 217 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void blts_0(__global double *g_v, int k, double omega,
		     __global double *g_ldz, int jst, int ist,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	int m = get_global_id(2);
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	if (!(m < 5)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*v)[33][33][33] = (__global double (*)[33][33][33])g_v;
	__global double (*ldz)[5][33][33] =
	    (__global double (*)[5][33][33])g_ldz;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		v[m][i][j][k] =
		    v[m][i][j][k] -
		    omega * (ldz[m][0][i][j] * v[0][i][j][k - 1] +
			     ldz[m][1][i][j] * v[1][i][j][k - 1] +
			     ldz[m][2][i][j] * v[2][i][j][k - 1] +
			     ldz[m][3][i][j] * v[3][i][j][k - 1] +
			     ldz[m][4][i][j] * v[4][i][j][k - 1]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 462 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void buts_0(__global double *g_tv, double omega,
		     __global double *g_udz, __global double *g_v, int k,
		     int jst, int ist, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	int m = get_global_id(2);
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	if (!(m < 5)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*tv)[33][33] = (__global double (*)[33][33])g_tv;
	__global double (*udz)[5][33][33] =
	    (__global double (*)[5][33][33])g_udz;
	__global double (*v)[33][33][33] = (__global double (*)[33][33][33])g_v;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		tv[m][i][j] =
		    omega * (udz[m][0][i][j] * v[0][i][j][k + 1] +
			     udz[m][1][i][j] * v[1][i][j][k + 1] +
			     udz[m][2][i][j] * v[2][i][j][k + 1] +
			     udz[m][3][i][j] * v[3][i][j][k + 1] +
			     udz[m][4][i][j] * v[4][i][j][k + 1]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 754 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_0(__global double *g_frct, int m, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	int i = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[33][33][33] =
	    (__global double (*)[33][33][33])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		frct[m][i][j][k] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 765 of lu.c
//-------------------------------------------------------------------------------
__kernel void erhs_1(int nx0, int j, int ny, int ny0, int k, int nz, int m,
		     __global double *g_rsd, __global double *g_ce,
		     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int iglob;		/* Defined at lu.c : 735 */
	double xi;		/* Defined at lu.c : 740 */
	int jglob;		/* Defined at lu.c : 735 */
	double eta;		/* Defined at lu.c : 740 */
	double zeta;		/* Defined at lu.c : 740 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		iglob = i;
		xi = ((double)(iglob)) / (nx0 - 1);
		for (j = 0; j < ny; j++) {
			jglob = j;
			eta = ((double)(jglob)) / (ny0 - 1);
			for (k = 0; k < nz; k++) {
				zeta = ((double)(k)) / (nz - 1);
				for (m = 0; m < 5; m++) {
					//-------------------------------------------
					//Declare prefetching Buffers (BEGIN) : 773
					//-------------------------------------------
					double4 ce_0[3];
					//-------------------------------------------
					//Declare prefetching buffers (END)
					//-------------------------------------------
					//-------------------------------------------
					//Prefetching (BEGIN) : 773
					//Candidates:
					//      ce[m][0]
					//      ce[m][1]
					//      ce[m][2]
					//      ce[m][3]
					//      ce[m][4]
					//      ce[m][5]
					//      ce[m][6]
					//      ce[m][7]
					//      ce[m][8]
					//      ce[m][9]
					//      ce[m][10]
					//      ce[m][11]
					//-------------------------------------------
					__global double *p_ce_0_0 =
					    (__global double *)&ce[m][0];
					if ((unsigned long)p_ce_0_0 % 64 == 0) {
						ce_0[0] = vload4(0, p_ce_0_0);
					} else {
						ce_0[0].x = p_ce_0_0[0];
						p_ce_0_0++;
						ce_0[0].y = p_ce_0_0[0];
						p_ce_0_0++;
						ce_0[0].z = p_ce_0_0[0];
						p_ce_0_0++;
						ce_0[0].w = p_ce_0_0[0];
						p_ce_0_0++;
					}
					__global double *p_ce_0_1 =
					    (__global double *)&ce[m][4];
					if ((unsigned long)p_ce_0_1 % 64 == 0) {
						ce_0[1] = vload4(0, p_ce_0_1);
					} else {
						ce_0[1].x = p_ce_0_1[0];
						p_ce_0_1++;
						ce_0[1].y = p_ce_0_1[0];
						p_ce_0_1++;
						ce_0[1].z = p_ce_0_1[0];
						p_ce_0_1++;
						ce_0[1].w = p_ce_0_1[0];
						p_ce_0_1++;
					}
					__global double *p_ce_0_2 =
					    (__global double *)&ce[m][8];
					if ((unsigned long)p_ce_0_2 % 64 == 0) {
						ce_0[2] = vload4(0, p_ce_0_2);
					} else {
						ce_0[2].x = p_ce_0_2[0];
						p_ce_0_2++;
						ce_0[2].y = p_ce_0_2[0];
						p_ce_0_2++;
						ce_0[2].z = p_ce_0_2[0];
						p_ce_0_2++;
						ce_0[2].w = p_ce_0_2[0];
						p_ce_0_2++;
					}
					//-------------------------------------------
					//Prefetching (END)
					//-------------------------------------------

					rsd[m][i][j][k] =
					    ce_0[0].x /*ce[m][0] */  +
					    ce_0[0].y /*ce[m][1] */  * xi +
					    ce_0[0].z /*ce[m][2] */  * eta +
					    ce_0[0].w /*ce[m][3] */  * zeta +
					    ce_0[1].x /*ce[m][4] */  * xi * xi +
					    ce_0[1].y /*ce[m][5] */  * eta *
					    eta +
					    ce_0[1].z /*ce[m][6] */  * zeta *
					    zeta +
					    ce_0[1].w /*ce[m][7] */  * xi * xi *
					    xi +
					    ce_0[2].x /*ce[m][8] */  * eta *
					    eta * eta +
					    ce_0[2].y /*ce[m][9] */  * zeta *
					    zeta * zeta +
					    ce_0[2].z /*ce[m][10] */  * xi *
					    xi * xi * xi +
					    ce_0[2].w /*ce[m][11] */  * eta *
					    eta * eta * eta +
					    ce[m][12] * zeta * zeta * zeta *
					    zeta;
				}
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 800 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_2(__global double *g_flux, __global double *g_rsd, int jst,
		     int L1, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + L1;
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u21;		/* Defined at lu.c : 742 */
	double q;		/* Defined at lu.c : 741 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 802
		//-------------------------------------------
		double rsd_0[4];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 802
		//Candidates:
		//      rsd[0][i][j][k]
		//      rsd[1][i][j][k]
		//      rsd[2][i][j][k]
		//      rsd[3][i][j][k]
		//-------------------------------------------
		rsd_0[0] = rsd[0][i][j][k];
		rsd_0[1] = rsd[1][i][j][k];
		rsd_0[2] = rsd[2][i][j][k];
		rsd_0[3] = rsd[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[0][i][j][k] = rsd_0[1] /*rsd[1][i][j][k] */ ;
		u21 =
		    rsd_0[1] /*rsd[1][i][j][k] */ /rsd_0[0] /*rsd[0][i][j][k] */
		    ;
		q = 0.50 *
		    (rsd_0[1] /*rsd[1][i][j][k] */ *rsd_0[1]
		     /*rsd[1][i][j][k] */ +rsd_0[2] /*rsd[2][i][j][k] */
		     *rsd_0[2] /*rsd[2][i][j][k] */ +rsd_0[3]
		     /*rsd[3][i][j][k] */ *rsd_0[3] /*rsd[3][i][j][k] */ ) /
		    rsd_0[0] /*rsd[0][i][j][k] */ ;
		flux[1][i][j][k] =
		    rsd_0[1] /*rsd[1][i][j][k] */ *u21 +
		    0.40e+00 * (rsd[4][i][j][k] - q);
		flux[2][i][j][k] = rsd_0[2] /*rsd[2][i][j][k] */ *u21;
		flux[3][i][j][k] = rsd_0[3] /*rsd[3][i][j][k] */ *u21;
		flux[4][i][j][k] =
		    (1.40e+00 * rsd[4][i][j][k] - 0.40e+00 * q) * u21;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 819 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_3(int i, int ist, int iend, int m, __global double *g_frct,
		     double tx2, __global double *g_flux, int L2,
		     __global double *g_rsd, double tx3, double dx1, double tx1,
		     double dx2, double dx3, double dx4, double dx5,
		     double dsspm, int nx, int jst, int __ocl_k_bound,
		     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp;		/* Defined at lu.c : 743 */
	double u21i;		/* Defined at lu.c : 744 */
	double u31i;		/* Defined at lu.c : 744 */
	double u41i;		/* Defined at lu.c : 744 */
	double u51i;		/* Defined at lu.c : 744 */
	double u21im1;		/* Defined at lu.c : 747 */
	double u31im1;		/* Defined at lu.c : 747 */
	double u41im1;		/* Defined at lu.c : 747 */
	double u51im1;		/* Defined at lu.c : 747 */
	int ist1;		/* Defined at lu.c : 737 */
	int iend1;		/* Defined at lu.c : 737 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[33][33][33] =
	    (__global double (*)[33][33][33])g_frct;
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = ist; i <= iend; i++) {
			for (m = 0; m < 5; m++) {
				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    tx2 * (flux[m][i + 1][j][k] -
					   flux[m][i - 1][j][k]);
			}
		}
		for (i = ist; i <= L2; i++) {
			tmp = 1.0 / rsd[0][i][j][k];
			u21i = tmp * rsd[1][i][j][k];
			u31i = tmp * rsd[2][i][j][k];
			u41i = tmp * rsd[3][i][j][k];
			u51i = tmp * rsd[4][i][j][k];
			tmp = 1.0 / rsd[0][i - 1][j][k];
			u21im1 = tmp * rsd[1][i - 1][j][k];
			u31im1 = tmp * rsd[2][i - 1][j][k];
			u41im1 = tmp * rsd[3][i - 1][j][k];
			u51im1 = tmp * rsd[4][i - 1][j][k];
			flux[1][i][j][k] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
			flux[2][i][j][k] = tx3 * (u31i - u31im1);
			flux[3][i][j][k] = tx3 * (u41i - u41im1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * tx3 * ((u21i * u21i +
								   u31i * u31i +
								   u41i *
								   u41i) -
								  (u21im1 *
								   u21im1 +
								   u31im1 *
								   u31im1 +
								   u41im1 *
								   u41im1)) +
			    (1.0 / 6.0) * tx3 * (u21i * u21i -
						 u21im1 * u21im1) +
			    1.40e+00 * 1.40e+00 * tx3 * (u51i - u51im1);
		}
		for (i = ist; i <= iend; i++) {
			frct[0][i][j][k] =
			    frct[0][i][j][k] +
			    dx1 * tx1 * (rsd[0][i - 1][j][k] -
					 2.0 * rsd[0][i][j][k] + rsd[0][i +
									1][j]
					 [k]);
			frct[1][i][j][k] =
			    frct[1][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[1][i + 1][j][k] -
							 flux[1][i][j][k]) +
			    dx2 * tx1 * (rsd[1][i - 1][j][k] -
					 2.0 * rsd[1][i][j][k] + rsd[1][i +
									1][j]
					 [k]);
			frct[2][i][j][k] =
			    frct[2][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[2][i + 1][j][k] -
							 flux[2][i][j][k]) +
			    dx3 * tx1 * (rsd[2][i - 1][j][k] -
					 2.0 * rsd[2][i][j][k] + rsd[2][i +
									1][j]
					 [k]);
			frct[3][i][j][k] =
			    frct[3][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[3][i + 1][j][k] -
							 flux[3][i][j][k]) +
			    dx4 * tx1 * (rsd[3][i - 1][j][k] -
					 2.0 * rsd[3][i][j][k] + rsd[3][i +
									1][j]
					 [k]);
			frct[4][i][j][k] =
			    frct[4][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[4][i + 1][j][k] -
							 flux[4][i][j][k]) +
			    dx5 * tx1 * (rsd[4][i - 1][j][k] -
					 2.0 * rsd[4][i][j][k] + rsd[4][i +
									1][j]
					 [k]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 884
			//-------------------------------------------
			double rsd_1[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 884
			//Candidates:
			//      rsd[m][1][j][k]
			//      rsd[m][2][j][k]
			//      rsd[m][3][j][k]
			//-------------------------------------------
			rsd_1[0] = rsd[m][1][j][k];
			rsd_1[1] = rsd[m][2][j][k];
			rsd_1[2] = rsd[m][3][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][1][j][k] =
			    frct[m][1][j][k] -
			    dsspm * (+5.0 * rsd_1[0] /*rsd[m][1][j][k] */ -4.0 *
				     rsd_1[1] /*rsd[m][2][j][k] */ +rsd_1[2]
				     /*rsd[m][3][j][k] */ );
			frct[m][2][j][k] =
			    frct[m][2][j][k] -
			    dsspm * (-4.0 * rsd_1[0] /*rsd[m][1][j][k] */ +6.0 *
				     rsd_1[1] /*rsd[m][2][j][k] */ -4.0 *
				     rsd_1[2] /*rsd[m][3][j][k] */
				     +rsd[m][4][j][k]);
		}
		ist1 = 3;
		iend1 = nx - 4;
		for (i = ist1; i <= iend1; i++) {
			for (m = 0; m < 5; m++) {
				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    dsspm * (rsd[m][i - 2][j][k] -
					     4.0 * rsd[m][i - 1][j][k] +
					     6.0 * rsd[m][i][j][k] -
					     4.0 * rsd[m][i + 1][j][k] +
					     rsd[m][i + 2][j][k]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 909
			//-------------------------------------------
			double rsd_2[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 909
			//Candidates:
			//      rsd[m][nx - 4][j][k]
			//      rsd[m][nx - 3][j][k]
			//-------------------------------------------
			rsd_2[0] = rsd[m][nx - 4][j][k];
			rsd_2[1] = rsd[m][nx - 3][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][nx - 3][j][k] =
			    frct[m][nx - 3][j][k] -
			    dsspm * (rsd[m][nx - 5][j][k] -
				     4.0 *
				     rsd_2[0] /*rsd[m][nx - 4][j][k] */ +6.0 *
				     rsd_2[1] /*rsd[m][nx - 3][j][k] */ -4.0 *
				     rsd[m][nx - 2][j][k]);
			frct[m][nx - 2][j][k] =
			    frct[m][nx - 2][j][k] -
			    dsspm * (rsd_2[0] /*rsd[m][nx - 4][j][k] */ -4.0 *
				     rsd_2[1] /*rsd[m][nx - 3][j][k] */ +5.0 *
				     rsd[m][nx - 2][j][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 931 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_4(__global double *g_flux, __global double *g_rsd, int L1,
		     int ist, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + L1;
	int i = get_global_id(2) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u31;		/* Defined at lu.c : 742 */
	double q;		/* Defined at lu.c : 741 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 933
		//-------------------------------------------
		double rsd_3[4];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 933
		//Candidates:
		//      rsd[0][i][j][k]
		//      rsd[1][i][j][k]
		//      rsd[2][i][j][k]
		//      rsd[3][i][j][k]
		//-------------------------------------------
		rsd_3[0] = rsd[0][i][j][k];
		rsd_3[1] = rsd[1][i][j][k];
		rsd_3[2] = rsd[2][i][j][k];
		rsd_3[3] = rsd[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[0][i][j][k] = rsd_3[2] /*rsd[2][i][j][k] */ ;
		u31 =
		    rsd_3[2] /*rsd[2][i][j][k] */ /rsd_3[0] /*rsd[0][i][j][k] */
		    ;
		q = 0.50 *
		    (rsd_3[1] /*rsd[1][i][j][k] */ *rsd_3[1]
		     /*rsd[1][i][j][k] */ +rsd_3[2] /*rsd[2][i][j][k] */
		     *rsd_3[2] /*rsd[2][i][j][k] */ +rsd_3[3]
		     /*rsd[3][i][j][k] */ *rsd_3[3] /*rsd[3][i][j][k] */ ) /
		    rsd_3[0] /*rsd[0][i][j][k] */ ;
		flux[1][i][j][k] = rsd_3[1] /*rsd[1][i][j][k] */ *u31;
		flux[2][i][j][k] =
		    rsd_3[2] /*rsd[2][i][j][k] */ *u31 +
		    0.40e+00 * (rsd[4][i][j][k] - q);
		flux[3][i][j][k] = rsd_3[3] /*rsd[3][i][j][k] */ *u31;
		flux[4][i][j][k] =
		    (1.40e+00 * rsd[4][i][j][k] - 0.40e+00 * q) * u31;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 950 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_5(int j, int jst, int jend, int m, __global double *g_frct,
		     double ty2, __global double *g_flux, int L2,
		     __global double *g_rsd, double ty3, double dy1, double ty1,
		     double dy2, double dy3, double dy4, double dy5,
		     double dsspm, int ny, int ist, int __ocl_k_bound,
		     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp;		/* Defined at lu.c : 743 */
	double u21j;		/* Defined at lu.c : 745 */
	double u31j;		/* Defined at lu.c : 745 */
	double u41j;		/* Defined at lu.c : 745 */
	double u51j;		/* Defined at lu.c : 745 */
	double u21jm1;		/* Defined at lu.c : 748 */
	double u31jm1;		/* Defined at lu.c : 748 */
	double u41jm1;		/* Defined at lu.c : 748 */
	double u51jm1;		/* Defined at lu.c : 748 */
	int jst1;		/* Defined at lu.c : 738 */
	int jend1;		/* Defined at lu.c : 738 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[33][33][33] =
	    (__global double (*)[33][33][33])g_frct;
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (j = jst; j <= jend; j++) {
			for (m = 0; m < 5; m++) {
				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    ty2 * (flux[m][i][j + 1][k] -
					   flux[m][i][j - 1][k]);
			}
		}
		for (j = jst; j <= L2; j++) {
			tmp = 1.0 / rsd[0][i][j][k];
			u21j = tmp * rsd[1][i][j][k];
			u31j = tmp * rsd[2][i][j][k];
			u41j = tmp * rsd[3][i][j][k];
			u51j = tmp * rsd[4][i][j][k];
			tmp = 1.0 / rsd[0][i][j - 1][k];
			u21jm1 = tmp * rsd[1][i][j - 1][k];
			u31jm1 = tmp * rsd[2][i][j - 1][k];
			u41jm1 = tmp * rsd[3][i][j - 1][k];
			u51jm1 = tmp * rsd[4][i][j - 1][k];
			flux[1][i][j][k] = ty3 * (u21j - u21jm1);
			flux[2][i][j][k] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
			flux[3][i][j][k] = ty3 * (u41j - u41jm1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * ty3 * ((u21j * u21j +
								   u31j * u31j +
								   u41j *
								   u41j) -
								  (u21jm1 *
								   u21jm1 +
								   u31jm1 *
								   u31jm1 +
								   u41jm1 *
								   u41jm1)) +
			    (1.0 / 6.0) * ty3 * (u31j * u31j -
						 u31jm1 * u31jm1) +
			    1.40e+00 * 1.40e+00 * ty3 * (u51j - u51jm1);
		}
		for (j = jst; j <= jend; j++) {
			frct[0][i][j][k] =
			    frct[0][i][j][k] +
			    dy1 * ty1 * (rsd[0][i][j - 1][k] -
					 2.0 * rsd[0][i][j][k] + rsd[0][i][j +
									   1]
					 [k]);
			frct[1][i][j][k] =
			    frct[1][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[1][i][j + 1][k] -
							 flux[1][i][j][k]) +
			    dy2 * ty1 * (rsd[1][i][j - 1][k] -
					 2.0 * rsd[1][i][j][k] + rsd[1][i][j +
									   1]
					 [k]);
			frct[2][i][j][k] =
			    frct[2][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[2][i][j + 1][k] -
							 flux[2][i][j][k]) +
			    dy3 * ty1 * (rsd[2][i][j - 1][k] -
					 2.0 * rsd[2][i][j][k] + rsd[2][i][j +
									   1]
					 [k]);
			frct[3][i][j][k] =
			    frct[3][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[3][i][j + 1][k] -
							 flux[3][i][j][k]) +
			    dy4 * ty1 * (rsd[3][i][j - 1][k] -
					 2.0 * rsd[3][i][j][k] + rsd[3][i][j +
									   1]
					 [k]);
			frct[4][i][j][k] =
			    frct[4][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[4][i][j + 1][k] -
							 flux[4][i][j][k]) +
			    dy5 * ty1 * (rsd[4][i][j - 1][k] -
					 2.0 * rsd[4][i][j][k] + rsd[4][i][j +
									   1]
					 [k]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1015
			//-------------------------------------------
			double rsd_4[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1015
			//Candidates:
			//      rsd[m][i][1][k]
			//      rsd[m][i][2][k]
			//      rsd[m][i][3][k]
			//-------------------------------------------
			rsd_4[0] = rsd[m][i][1][k];
			rsd_4[1] = rsd[m][i][2][k];
			rsd_4[2] = rsd[m][i][3][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][i][1][k] =
			    frct[m][i][1][k] -
			    dsspm * (+5.0 * rsd_4[0] /*rsd[m][i][1][k] */ -4.0 *
				     rsd_4[1] /*rsd[m][i][2][k] */ +rsd_4[2]
				     /*rsd[m][i][3][k] */ );
			frct[m][i][2][k] =
			    frct[m][i][2][k] -
			    dsspm * (-4.0 * rsd_4[0] /*rsd[m][i][1][k] */ +6.0 *
				     rsd_4[1] /*rsd[m][i][2][k] */ -4.0 *
				     rsd_4[2] /*rsd[m][i][3][k] */
				     +rsd[m][i][4][k]);
		}
		jst1 = 3;
		jend1 = ny - 4;
		for (j = jst1; j <= jend1; j++) {
			for (m = 0; m < 5; m++) {
				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    dsspm * (rsd[m][i][j - 2][k] -
					     4.0 * rsd[m][i][j - 1][k] +
					     6.0 * rsd[m][i][j][k] -
					     4.0 * rsd[m][i][j + 1][k] +
					     rsd[m][i][j + 2][k]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1041
			//-------------------------------------------
			double rsd_5[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1041
			//Candidates:
			//      rsd[m][i][ny - 4][k]
			//      rsd[m][i][ny - 3][k]
			//-------------------------------------------
			rsd_5[0] = rsd[m][i][ny - 4][k];
			rsd_5[1] = rsd[m][i][ny - 3][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][i][ny - 3][k] =
			    frct[m][i][ny - 3][k] -
			    dsspm * (rsd[m][i][ny - 5][k] -
				     4.0 *
				     rsd_5[0] /*rsd[m][i][ny - 4][k] */ +6.0 *
				     rsd_5[1] /*rsd[m][i][ny - 3][k] */ -4.0 *
				     rsd[m][i][ny - 2][k]);
			frct[m][i][ny - 2][k] =
			    frct[m][i][ny - 2][k] -
			    dsspm * (rsd_5[0] /*rsd[m][i][ny - 4][k] */ -4.0 *
				     rsd_5[1] /*rsd[m][i][ny - 3][k] */ +5.0 *
				     rsd[m][i][ny - 2][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1060 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void erhs_6(int k, int nz, __global double *g_flux,
		     __global double *g_rsd, int m, __global double *g_frct,
		     double tz2, double tz3, double dz1, double tz1, double dz2,
		     double dz3, double dz4, double dz5, double dsspm, int jst,
		     int ist, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u41;		/* Defined at lu.c : 742 */
	double q;		/* Defined at lu.c : 741 */
	double tmp;		/* Defined at lu.c : 743 */
	double u21k;		/* Defined at lu.c : 746 */
	double u31k;		/* Defined at lu.c : 746 */
	double u41k;		/* Defined at lu.c : 746 */
	double u51k;		/* Defined at lu.c : 746 */
	double u21km1;		/* Defined at lu.c : 749 */
	double u31km1;		/* Defined at lu.c : 749 */
	double u41km1;		/* Defined at lu.c : 749 */
	double u51km1;		/* Defined at lu.c : 749 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	__global double (*frct)[33][33][33] =
	    (__global double (*)[33][33][33])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1062
			//-------------------------------------------
			double rsd_6[4];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1062
			//Candidates:
			//      rsd[0][i][j][k]
			//      rsd[1][i][j][k]
			//      rsd[2][i][j][k]
			//      rsd[3][i][j][k]
			//-------------------------------------------
			rsd_6[0] = rsd[0][i][j][k];
			rsd_6[1] = rsd[1][i][j][k];
			rsd_6[2] = rsd[2][i][j][k];
			rsd_6[3] = rsd[3][i][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			flux[0][i][j][k] = rsd_6[3] /*rsd[3][i][j][k] */ ;
			u41 =
			    rsd_6[3] /*rsd[3][i][j][k] */ /rsd_6[0]
			    /*rsd[0][i][j][k] */ ;
			q = 0.50 *
			    (rsd_6[1] /*rsd[1][i][j][k] */ *rsd_6[1]
			     /*rsd[1][i][j][k] */ +rsd_6[2] /*rsd[2][i][j][k] */
			     *rsd_6[2] /*rsd[2][i][j][k] */ +rsd_6[3]
			     /*rsd[3][i][j][k] */ *rsd_6[3] /*rsd[3][i][j][k] */
			     ) / rsd_6[0] /*rsd[0][i][j][k] */ ;
			flux[1][i][j][k] = rsd_6[1] /*rsd[1][i][j][k] */ *u41;
			flux[2][i][j][k] = rsd_6[2] /*rsd[2][i][j][k] */ *u41;
			flux[3][i][j][k] =
			    rsd_6[3] /*rsd[3][i][j][k] */ *u41 +
			    0.40e+00 * (rsd[4][i][j][k] - q);
			flux[4][i][j][k] =
			    (1.40e+00 * rsd[4][i][j][k] - 0.40e+00 * q) * u41;
		}
		for (k = 1; k <= nz - 2; k++) {
			for (m = 0; m < 5; m++) {
				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    tz2 * (flux[m][i][j][k + 1] -
					   flux[m][i][j][k - 1]);
			}
		}
		for (k = 1; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1082
			//-------------------------------------------
			double2 rsd_7[5];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1082
			//Candidates:
			//      rsd[0][i][j][k - 1]
			//      rsd[0][i][j][k]
			//      rsd[1][i][j][k - 1]
			//      rsd[1][i][j][k]
			//      rsd[2][i][j][k - 1]
			//      rsd[2][i][j][k]
			//      rsd[3][i][j][k - 1]
			//      rsd[3][i][j][k]
			//      rsd[4][i][j][k - 1]
			//      rsd[4][i][j][k]
			//-------------------------------------------
			__global double *p_rsd_7_0 =
			    (__global double *)&rsd[0][i][j][k - 1];
			if ((unsigned long)p_rsd_7_0 % 64 == 0) {
				rsd_7[0] = vload2(0, p_rsd_7_0);
			} else {
				rsd_7[0].x = p_rsd_7_0[0];
				p_rsd_7_0++;
				rsd_7[0].y = p_rsd_7_0[0];
				p_rsd_7_0++;
			}
			__global double *p_rsd_7_1 =
			    (__global double *)&rsd[1][i][j][k - 1];
			if ((unsigned long)p_rsd_7_1 % 64 == 0) {
				rsd_7[1] = vload2(0, p_rsd_7_1);
			} else {
				rsd_7[1].x = p_rsd_7_1[0];
				p_rsd_7_1++;
				rsd_7[1].y = p_rsd_7_1[0];
				p_rsd_7_1++;
			}
			__global double *p_rsd_7_2 =
			    (__global double *)&rsd[2][i][j][k - 1];
			if ((unsigned long)p_rsd_7_2 % 64 == 0) {
				rsd_7[2] = vload2(0, p_rsd_7_2);
			} else {
				rsd_7[2].x = p_rsd_7_2[0];
				p_rsd_7_2++;
				rsd_7[2].y = p_rsd_7_2[0];
				p_rsd_7_2++;
			}
			__global double *p_rsd_7_3 =
			    (__global double *)&rsd[3][i][j][k - 1];
			if ((unsigned long)p_rsd_7_3 % 64 == 0) {
				rsd_7[3] = vload2(0, p_rsd_7_3);
			} else {
				rsd_7[3].x = p_rsd_7_3[0];
				p_rsd_7_3++;
				rsd_7[3].y = p_rsd_7_3[0];
				p_rsd_7_3++;
			}
			__global double *p_rsd_7_4 =
			    (__global double *)&rsd[4][i][j][k - 1];
			if ((unsigned long)p_rsd_7_4 % 64 == 0) {
				rsd_7[4] = vload2(0, p_rsd_7_4);
			} else {
				rsd_7[4].x = p_rsd_7_4[0];
				p_rsd_7_4++;
				rsd_7[4].y = p_rsd_7_4[0];
				p_rsd_7_4++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmp = 1.0 / rsd_7[0].y /*rsd[0][i][j][k] */ ;
			u21k = tmp * rsd_7[1].y /*rsd[1][i][j][k] */ ;
			u31k = tmp * rsd_7[2].y /*rsd[2][i][j][k] */ ;
			u41k = tmp * rsd_7[3].y /*rsd[3][i][j][k] */ ;
			u51k = tmp * rsd_7[4].y /*rsd[4][i][j][k] */ ;
			tmp = 1.0 / rsd_7[0].x /*rsd[0][i][j][k - 1] */ ;
			u21km1 = tmp * rsd_7[1].x /*rsd[1][i][j][k - 1] */ ;
			u31km1 = tmp * rsd_7[2].x /*rsd[2][i][j][k - 1] */ ;
			u41km1 = tmp * rsd_7[3].x /*rsd[3][i][j][k - 1] */ ;
			u51km1 = tmp * rsd_7[4].x /*rsd[4][i][j][k - 1] */ ;
			flux[1][i][j][k] = tz3 * (u21k - u21km1);
			flux[2][i][j][k] = tz3 * (u31k - u31km1);
			flux[3][i][j][k] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * tz3 * ((u21k * u21k +
								   u31k * u31k +
								   u41k *
								   u41k) -
								  (u21km1 *
								   u21km1 +
								   u31km1 *
								   u31km1 +
								   u41km1 *
								   u41km1)) +
			    (1.0 / 6.0) * tz3 * (u41k * u41k -
						 u41km1 * u41km1) +
			    1.40e+00 * 1.40e+00 * tz3 * (u51k - u51km1);
		}
		for (k = 1; k <= nz - 2; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1109
			//-------------------------------------------
			double2 rsd_8[5];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1109
			//Candidates:
			//      rsd[0][i][j][k - 1]
			//      rsd[0][i][j][k]
			//      rsd[1][i][j][k - 1]
			//      rsd[1][i][j][k]
			//      rsd[2][i][j][k - 1]
			//      rsd[2][i][j][k]
			//      rsd[3][i][j][k - 1]
			//      rsd[3][i][j][k]
			//      rsd[4][i][j][k - 1]
			//      rsd[4][i][j][k]
			//-------------------------------------------
			__global double *p_rsd_8_0 =
			    (__global double *)&rsd[0][i][j][k - 1];
			if ((unsigned long)p_rsd_8_0 % 64 == 0) {
				rsd_8[0] = vload2(0, p_rsd_8_0);
			} else {
				rsd_8[0].x = p_rsd_8_0[0];
				p_rsd_8_0++;
				rsd_8[0].y = p_rsd_8_0[0];
				p_rsd_8_0++;
			}
			__global double *p_rsd_8_1 =
			    (__global double *)&rsd[1][i][j][k - 1];
			if ((unsigned long)p_rsd_8_1 % 64 == 0) {
				rsd_8[1] = vload2(0, p_rsd_8_1);
			} else {
				rsd_8[1].x = p_rsd_8_1[0];
				p_rsd_8_1++;
				rsd_8[1].y = p_rsd_8_1[0];
				p_rsd_8_1++;
			}
			__global double *p_rsd_8_2 =
			    (__global double *)&rsd[2][i][j][k - 1];
			if ((unsigned long)p_rsd_8_2 % 64 == 0) {
				rsd_8[2] = vload2(0, p_rsd_8_2);
			} else {
				rsd_8[2].x = p_rsd_8_2[0];
				p_rsd_8_2++;
				rsd_8[2].y = p_rsd_8_2[0];
				p_rsd_8_2++;
			}
			__global double *p_rsd_8_3 =
			    (__global double *)&rsd[3][i][j][k - 1];
			if ((unsigned long)p_rsd_8_3 % 64 == 0) {
				rsd_8[3] = vload2(0, p_rsd_8_3);
			} else {
				rsd_8[3].x = p_rsd_8_3[0];
				p_rsd_8_3++;
				rsd_8[3].y = p_rsd_8_3[0];
				p_rsd_8_3++;
			}
			__global double *p_rsd_8_4 =
			    (__global double *)&rsd[4][i][j][k - 1];
			if ((unsigned long)p_rsd_8_4 % 64 == 0) {
				rsd_8[4] = vload2(0, p_rsd_8_4);
			} else {
				rsd_8[4].x = p_rsd_8_4[0];
				p_rsd_8_4++;
				rsd_8[4].y = p_rsd_8_4[0];
				p_rsd_8_4++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[0][i][j][k] =
			    frct[0][i][j][k] +
			    dz1 * tz1 * (rsd[0][i][j][k + 1] -
					 2.0 *
					 rsd_8[0].y /*rsd[0][i][j][k] */  +
					 rsd_8[0].x /*rsd[0][i][j][k - 1] */ );
			frct[1][i][j][k] =
			    frct[1][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[1][i][j][k + 1] -
							 flux[1][i][j][k]) +
			    dz2 * tz1 * (rsd[1][i][j][k + 1] -
					 2.0 *
					 rsd_8[1].y /*rsd[1][i][j][k] */  +
					 rsd_8[1].x /*rsd[1][i][j][k - 1] */ );
			frct[2][i][j][k] =
			    frct[2][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[2][i][j][k + 1] -
							 flux[2][i][j][k]) +
			    dz3 * tz1 * (rsd[2][i][j][k + 1] -
					 2.0 *
					 rsd_8[2].y /*rsd[2][i][j][k] */  +
					 rsd_8[2].x /*rsd[2][i][j][k - 1] */ );
			frct[3][i][j][k] =
			    frct[3][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[3][i][j][k + 1] -
							 flux[3][i][j][k]) +
			    dz4 * tz1 * (rsd[3][i][j][k + 1] -
					 2.0 *
					 rsd_8[3].y /*rsd[3][i][j][k] */  +
					 rsd_8[3].x /*rsd[3][i][j][k - 1] */ );
			frct[4][i][j][k] =
			    frct[4][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[4][i][j][k + 1] -
							 flux[4][i][j][k]) +
			    dz5 * tz1 * (rsd[4][i][j][k + 1] -
					 2.0 *
					 rsd_8[4].y /*rsd[4][i][j][k] */  +
					 rsd_8[4].x /*rsd[4][i][j][k - 1] */ );
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1139
			//-------------------------------------------
			double4 rsd_9;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1139
			//Candidates:
			//      rsd[m][i][j][1]
			//      rsd[m][i][j][2]
			//      rsd[m][i][j][3]
			//      rsd[m][i][j][4]
			//-------------------------------------------
			__global double *p_rsd_9_0 =
			    (__global double *)&rsd[m][i][j][1];
			if ((unsigned long)p_rsd_9_0 % 64 == 0) {
				rsd_9 = vload4(0, p_rsd_9_0);
			} else {
				rsd_9.x = p_rsd_9_0[0];
				p_rsd_9_0++;
				rsd_9.y = p_rsd_9_0[0];
				p_rsd_9_0++;
				rsd_9.z = p_rsd_9_0[0];
				p_rsd_9_0++;
				rsd_9.w = p_rsd_9_0[0];
				p_rsd_9_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][i][j][1] =
			    frct[m][i][j][1] -
			    dsspm * (+5.0 * rsd_9.x /*rsd[m][i][j][1] */  -
				     4.0 * rsd_9.y /*rsd[m][i][j][2] */  +
				     rsd_9.z /*rsd[m][i][j][3] */ );
			frct[m][i][j][2] =
			    frct[m][i][j][2] -
			    dsspm * (-4.0 * rsd_9.x /*rsd[m][i][j][1] */  +
				     6.0 * rsd_9.y /*rsd[m][i][j][2] */  -
				     4.0 * rsd_9.z /*rsd[m][i][j][3] */  +
				     rsd_9.w /*rsd[m][i][j][4] */ );
		}
		for (k = 3; k <= nz - 4; k++) {
			for (m = 0; m < 5; m++) {
				//-------------------------------------------
				//Declare prefetching Buffers (BEGIN) : 1152
				//-------------------------------------------
				double4 rsd_10;
				//-------------------------------------------
				//Declare prefetching buffers (END)
				//-------------------------------------------
				//-------------------------------------------
				//Prefetching (BEGIN) : 1152
				//Candidates:
				//      rsd[m][i][j][k - 2]
				//      rsd[m][i][j][k - 1]
				//      rsd[m][i][j][k]
				//      rsd[m][i][j][k + 1]
				//-------------------------------------------
				__global double *p_rsd_10_0 =
				    (__global double *)&rsd[m][i][j][k - 2];
				if ((unsigned long)p_rsd_10_0 % 64 == 0) {
					rsd_10 = vload4(0, p_rsd_10_0);
				} else {
					rsd_10.x = p_rsd_10_0[0];
					p_rsd_10_0++;
					rsd_10.y = p_rsd_10_0[0];
					p_rsd_10_0++;
					rsd_10.z = p_rsd_10_0[0];
					p_rsd_10_0++;
					rsd_10.w = p_rsd_10_0[0];
					p_rsd_10_0++;
				}
				//-------------------------------------------
				//Prefetching (END)
				//-------------------------------------------

				frct[m][i][j][k] =
				    frct[m][i][j][k] -
				    dsspm *
				    (rsd_10.x /*rsd[m][i][j][k - 2] */  -
				     4.0 * rsd_10.y /*rsd[m][i][j][k - 1] */  +
				     6.0 * rsd_10.z /*rsd[m][i][j][k] */  -
				     4.0 * rsd_10.w /*rsd[m][i][j][k + 1] */  +
				     rsd[m][i][j][k + 2]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1162
			//-------------------------------------------
			double4 rsd_11;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1162
			//Candidates:
			//      rsd[m][i][j][nz - 5]
			//      rsd[m][i][j][nz - 4]
			//      rsd[m][i][j][nz - 3]
			//      rsd[m][i][j][nz - 2]
			//-------------------------------------------
			__global double *p_rsd_11_0 =
			    (__global double *)&rsd[m][i][j][nz - 5];
			if ((unsigned long)p_rsd_11_0 % 64 == 0) {
				rsd_11 = vload4(0, p_rsd_11_0);
			} else {
				rsd_11.x = p_rsd_11_0[0];
				p_rsd_11_0++;
				rsd_11.y = p_rsd_11_0[0];
				p_rsd_11_0++;
				rsd_11.z = p_rsd_11_0[0];
				p_rsd_11_0++;
				rsd_11.w = p_rsd_11_0[0];
				p_rsd_11_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[m][i][j][nz - 3] =
			    frct[m][i][j][nz - 3] -
			    dsspm * (rsd_11.x /*rsd[m][i][j][nz - 5] */  -
				     4.0 * rsd_11.y /*rsd[m][i][j][nz - 4] */  +
				     6.0 * rsd_11.z /*rsd[m][i][j][nz - 3] */  -
				     4.0 * rsd_11.w /*rsd[m][i][j][nz - 2] */ );
			frct[m][i][j][nz - 2] =
			    frct[m][i][j][nz - 2] -
			    dsspm * (rsd_11.y /*rsd[m][i][j][nz - 4] */  -
				     4.0 * rsd_11.z /*rsd[m][i][j][nz - 3] */  +
				     5.0 * rsd_11.w /*rsd[m][i][j][nz - 2] */ );
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1281 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void jacld_0(__global double *g_u, int k, __global double *g_d,
		      double dt, double tx1, double dx1, double ty1, double dy1,
		      double tz1, double dz1, double r43, double c34,
		      double dx2, double dy2, double dz2, double dx3,
		      double dy3, double dz3, double dx4, double dy4,
		      double dz4, double c1345, double dx5, double dy5,
		      double dz5, __global double *g_a, double tz2,
		      __global double *g_b, double ty2, __global double *g_c,
		      double tx2, int jst, int ist, int __ocl_j_bound,
		      int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp1;		/* Defined at lu.c : 1274 */
	double tmp2;		/* Defined at lu.c : 1274 */
	double tmp3;		/* Defined at lu.c : 1274 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*d)[5][33][33] = (__global double (*)[5][33][33])g_d;
	__global double (*a)[5][33][33] = (__global double (*)[5][33][33])g_a;
	__global double (*b)[5][33][33] = (__global double (*)[5][33][33])g_b;
	__global double (*c)[5][33][33] = (__global double (*)[5][33][33])g_c;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1282
		//-------------------------------------------
		double u_0[15];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1282
		//Candidates:
		//      u[1][i - 1][j][k]
		//      u[2][i - 1][j][k]
		//      u[3][i - 1][j][k]
		//      u[4][i - 1][j][k]
		//      u[1][i][j][k - 1]
		//      u[2][i][j][k - 1]
		//      u[3][i][j][k - 1]
		//      u[4][i][j][k - 1]
		//      u[1][i][j - 1][k]
		//      u[2][i][j - 1][k]
		//      u[3][i][j - 1][k]
		//      u[4][i][j - 1][k]
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_0[0] = u[1][i - 1][j][k];
		u_0[1] = u[2][i - 1][j][k];
		u_0[2] = u[3][i - 1][j][k];
		u_0[3] = u[4][i - 1][j][k];
		u_0[4] = u[1][i][j][k - 1];
		u_0[5] = u[2][i][j][k - 1];
		u_0[6] = u[3][i][j][k - 1];
		u_0[7] = u[4][i][j][k - 1];
		u_0[8] = u[1][i][j - 1][k];
		u_0[9] = u[2][i][j - 1][k];
		u_0[10] = u[3][i][j - 1][k];
		u_0[11] = u[4][i][j - 1][k];
		u_0[12] = u[1][i][j][k];
		u_0[13] = u[2][i][j][k];
		u_0[14] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u[0][i][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		d[0][0][i][j] =
		    1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
		d[0][1][i][j] = 0.0;
		d[0][2][i][j] = 0.0;
		d[0][3][i][j] = 0.0;
		d[0][4][i][j] = 0.0;
		d[1][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-r43 * c34 * tmp2 *
				 u_0[12] /*u[1][i][j][k] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_0[12] /*u[1][i][j][k] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_0[12] /*u[1][i][j][k] */ ));
		d[1][1][i][j] =
		    1.0 + dt * 2.0 * (tx1 * r43 * c34 * tmp1 +
				      ty1 * c34 * tmp1 + tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
		d[1][2][i][j] = 0.0;
		d[1][3][i][j] = 0.0;
		d[1][4][i][j] = 0.0;
		d[2][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_0[13] /*u[2][i][j][k] */ ) +
				ty1 * (-r43 * c34 * tmp2 *
				       u_0[13] /*u[2][i][j][k] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_0[13] /*u[2][i][j][k] */ ));
		d[2][1][i][j] = 0.0;
		d[2][2][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 +
				      ty1 * r43 * c34 * tmp1 +
				      tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
		d[2][3][i][j] = 0.0;
		d[2][4][i][j] = 0.0;
		d[3][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_0[14] /*u[3][i][j][k] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_0[14] /*u[3][i][j][k] */ ) +
				tz1 * (-r43 * c34 * tmp2 *
				       u_0[14] /*u[3][i][j][k] */ ));
		d[3][1][i][j] = 0.0;
		d[3][2][i][j] = 0.0;
		d[3][3][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 +
				      tz1 * r43 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
		d[3][4][i][j] = 0.0;
		d[4][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-(r43 * c34 - c1345) * tmp3 *
				 (((u_0[12] /*u[1][i][j][k] */ ) *
				   (u_0[12] /*u[1][i][j][k] */ ))) - (c34 -
								      c1345) *
				 tmp3 *
				 (((u_0[13] /*u[2][i][j][k] */ ) *
				   (u_0[13] /*u[2][i][j][k] */ ))) - (c34 -
								      c1345) *
				 tmp3 *
				 (((u_0[14] /*u[3][i][j][k] */ ) *
				   (u_0[14] /*u[3][i][j][k] */ ))) -
				 (c1345) * tmp2 * u[4][i][j][k]) +
				ty1 * (-(c34 - c1345) * tmp3 *
				       (((u_0[12] /*u[1][i][j][k] */ ) *
					 (u_0[12] /*u[1][i][j][k] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_0[13] /*u[2][i][j][k] */ ) *
					 (u_0[13] /*u[2][i][j][k] */ ))) -
				       (c34 -
					c1345) * tmp3 *
				       (((u_0[14] /*u[3][i][j][k] */ ) *
					 (u_0[14] /*u[3][i][j][k] */ ))) -
				       (c1345) * tmp2 * u[4][i][j][k]) +
				tz1 * (-(c34 - c1345) * tmp3 *
				       (((u_0[12] /*u[1][i][j][k] */ ) *
					 (u_0[12] /*u[1][i][j][k] */ ))) -
				       (c34 -
					c1345) * tmp3 *
				       (((u_0[13] /*u[2][i][j][k] */ ) *
					 (u_0[13] /*u[2][i][j][k] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_0[14] /*u[3][i][j][k] */ ) *
					 (u_0[14] /*u[3][i][j][k] */ ))) -
				       (c1345) * tmp2 * u[4][i][j][k]));
		d[4][1][i][j] =
		    dt * 2.0 * (tx1 * (r43 * c34 - c1345) * tmp2 *
				u_0[12] /*u[1][i][j][k] */ +ty1 * (c34 -
								   c1345) *
				tmp2 * u_0[12] /*u[1][i][j][k] */ +tz1 * (c34 -
									  c1345)
				* tmp2 * u_0[12] /*u[1][i][j][k] */ );
		d[4][2][i][j] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_0[13] /*u[2][i][j][k] */ +ty1 * (r43 * c34 -
								   c1345) *
				tmp2 * u_0[13] /*u[2][i][j][k] */ +tz1 * (c34 -
									  c1345)
				* tmp2 * u_0[13] /*u[2][i][j][k] */ );
		d[4][3][i][j] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_0[14] /*u[3][i][j][k] */ +ty1 * (c34 -
								   c1345) *
				tmp2 * u_0[14] /*u[3][i][j][k] */ +tz1 * (r43 *
									  c34 -
									  c1345)
				* tmp2 * u_0[14] /*u[3][i][j][k] */ );
		d[4][4][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 +
				      tz1 * c1345 * tmp1) +
		    dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
		tmp1 = 1.0 / u[0][i][j][k - 1];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		a[0][0][i][j] = -dt * tz1 * dz1;
		a[0][1][i][j] = 0.0;
		a[0][2][i][j] = 0.0;
		a[0][3][i][j] = -dt * tz2;
		a[0][4][i][j] = 0.0;
		a[1][0][i][j] =
		    -dt * tz2 *
		    (-
		     (u_0[4] /*u[1][i][j][k - 1] */ *u_0[6]
		      /*u[3][i][j][k - 1] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_0[4] /*u[1][i][j][k - 1] */ );
		a[1][1][i][j] =
		    -dt * tz2 * (u_0[6] /*u[3][i][j][k - 1] */ *tmp1) -
		    dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
		a[1][2][i][j] = 0.0;
		a[1][3][i][j] =
		    -dt * tz2 * (u_0[4] /*u[1][i][j][k - 1] */ *tmp1);
		a[1][4][i][j] = 0.0;
		a[2][0][i][j] =
		    -dt * tz2 *
		    (-
		     (u_0[5] /*u[2][i][j][k - 1] */ *u_0[6]
		      /*u[3][i][j][k - 1] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_0[5] /*u[2][i][j][k - 1] */ );
		a[2][1][i][j] = 0.0;
		a[2][2][i][j] =
		    -dt * tz2 * (u_0[6] /*u[3][i][j][k - 1] */ *tmp1) -
		    dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
		a[2][3][i][j] =
		    -dt * tz2 * (u_0[5] /*u[2][i][j][k - 1] */ *tmp1);
		a[2][4][i][j] = 0.0;
		a[3][0][i][j] =
		    -dt * tz2 * (-(u_0[6] /*u[3][i][j][k - 1] */ *tmp1) *
				 (u_0[6] /*u[3][i][j][k - 1] */ *tmp1) +
				 0.50 * 0.40e+00 *
				 ((u_0[4] /*u[1][i][j][k - 1] */ *u_0[4]
				   /*u[1][i][j][k - 1] */ +u_0[5]
				   /*u[2][i][j][k - 1] */ *u_0[5]
				   /*u[2][i][j][k - 1] */ +u_0[6]
				   /*u[3][i][j][k - 1] */ *u_0[6]
				   /*u[3][i][j][k - 1] */ ) * tmp2)) -
		    dt * tz1 * (-r43 * c34 * tmp2 *
				u_0[6] /*u[3][i][j][k - 1] */ );
		a[3][1][i][j] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_0[4] /*u[1][i][j][k - 1] */ *tmp1));
		a[3][2][i][j] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_0[5] /*u[2][i][j][k - 1] */ *tmp1));
		a[3][3][i][j] =
		    -dt * tz2 * (2.0 -
				 0.40e+00) *
		    (u_0[6] /*u[3][i][j][k - 1] */ *tmp1) -
		    dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
		a[3][4][i][j] = -dt * tz2 * 0.40e+00;
		a[4][0][i][j] =
		    -dt * tz2 *
		    ((0.40e+00 *
		      (u_0[4] /*u[1][i][j][k - 1] */ *u_0[4]
		       /*u[1][i][j][k - 1] */ +u_0[5] /*u[2][i][j][k - 1] */
		       *u_0[5] /*u[2][i][j][k - 1] */ +u_0[6]
		       /*u[3][i][j][k - 1] */ *u_0[6] /*u[3][i][j][k - 1] */ ) *
		      tmp2 -
		      1.40e+00 * (u_0[7] /*u[4][i][j][k - 1] */ *tmp1)) *
		     (u_0[6] /*u[3][i][j][k - 1] */ *tmp1)) -
		    dt * tz1 * (-(c34 - c1345) * tmp3 *
				(u_0[4] /*u[1][i][j][k - 1] */ *u_0[4]
				 /*u[1][i][j][k - 1] */ ) - (c34 -
							     c1345) * tmp3 *
				(u_0[5] /*u[2][i][j][k - 1] */ *u_0[5]
				 /*u[2][i][j][k - 1] */ ) - (r43 * c34 -
							     c1345) * tmp3 *
				(u_0[6] /*u[3][i][j][k - 1] */ *u_0[6]
				 /*u[3][i][j][k - 1] */ ) -
				c1345 * tmp2 * u_0[7] /*u[4][i][j][k - 1] */ );
		a[4][1][i][j] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_0[4] /*u[1][i][j][k - 1] */ *u_0[6]
				  /*u[3][i][j][k - 1] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 * u_0[4] /*u[1][i][j][k - 1] */ ;
		a[4][2][i][j] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_0[5] /*u[2][i][j][k - 1] */ *u_0[6]
				  /*u[3][i][j][k - 1] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 * u_0[5] /*u[2][i][j][k - 1] */ ;
		a[4][3][i][j] =
		    -dt * tz2 * (1.40e+00 *
				 (u_0[7] /*u[4][i][j][k - 1] */ *tmp1) -
				 0.50 * 0.40e+00 *
				 ((u_0[4] /*u[1][i][j][k - 1] */ *u_0[4]
				   /*u[1][i][j][k - 1] */ +u_0[5]
				   /*u[2][i][j][k - 1] */ *u_0[5]
				   /*u[2][i][j][k - 1] */ +3.0 *
				   u_0[6] /*u[3][i][j][k - 1] */ *u_0[6]
				   /*u[3][i][j][k - 1] */ ) * tmp2)) -
		    dt * tz1 * (r43 * c34 -
				c1345) * tmp2 * u_0[6] /*u[3][i][j][k - 1] */ ;
		a[4][4][i][j] =
		    -dt * tz2 * (1.40e+00 *
				 (u_0[6] /*u[3][i][j][k - 1] */ *tmp1)) -
		    dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
		tmp1 = 1.0 / u[0][i][j - 1][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		b[0][0][i][j] = -dt * ty1 * dy1;
		b[0][1][i][j] = 0.0;
		b[0][2][i][j] = -dt * ty2;
		b[0][3][i][j] = 0.0;
		b[0][4][i][j] = 0.0;
		b[1][0][i][j] =
		    -dt * ty2 *
		    (-
		     (u_0[8] /*u[1][i][j - 1][k] */ *u_0[9]
		      /*u[2][i][j - 1][k] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_0[8] /*u[1][i][j - 1][k] */ );
		b[1][1][i][j] =
		    -dt * ty2 * (u_0[9] /*u[2][i][j - 1][k] */ *tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
		b[1][2][i][j] =
		    -dt * ty2 * (u_0[8] /*u[1][i][j - 1][k] */ *tmp1);
		b[1][3][i][j] = 0.0;
		b[1][4][i][j] = 0.0;
		b[2][0][i][j] =
		    -dt * ty2 * (-(u_0[9] /*u[2][i][j - 1][k] */ *tmp1) *
				 (u_0[9] /*u[2][i][j - 1][k] */ *tmp1) +
				 0.50 * 0.40e+00 *
				 ((u_0[8] /*u[1][i][j - 1][k] */ *u_0[8]
				   /*u[1][i][j - 1][k] */ +u_0[9]
				   /*u[2][i][j - 1][k] */ *u_0[9]
				   /*u[2][i][j - 1][k] */ +u_0[10]
				   /*u[3][i][j - 1][k] */ *u_0[10]
				   /*u[3][i][j - 1][k] */ ) * tmp2)) -
		    dt * ty1 * (-r43 * c34 * tmp2 *
				u_0[9] /*u[2][i][j - 1][k] */ );
		b[2][1][i][j] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_0[8] /*u[1][i][j - 1][k] */ *tmp1));
		b[2][2][i][j] =
		    -dt * ty2 * ((2.0 - 0.40e+00) *
				 (u_0[9] /*u[2][i][j - 1][k] */ *tmp1)) -
		    dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
		b[2][3][i][j] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_0[10] /*u[3][i][j - 1][k] */ *tmp1));
		b[2][4][i][j] = -dt * ty2 * 0.40e+00;
		b[3][0][i][j] =
		    -dt * ty2 *
		    (-
		     (u_0[9] /*u[2][i][j - 1][k] */ *u_0[10]
		      /*u[3][i][j - 1][k] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_0[10] /*u[3][i][j - 1][k] */ );
		b[3][1][i][j] = 0.0;
		b[3][2][i][j] =
		    -dt * ty2 * (u_0[10] /*u[3][i][j - 1][k] */ *tmp1);
		b[3][3][i][j] =
		    -dt * ty2 * (u_0[9] /*u[2][i][j - 1][k] */ *tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
		b[3][4][i][j] = 0.0;
		b[4][0][i][j] =
		    -dt * ty2 *
		    ((0.40e+00 *
		      (u_0[8] /*u[1][i][j - 1][k] */ *u_0[8]
		       /*u[1][i][j - 1][k] */ +u_0[9] /*u[2][i][j - 1][k] */
		       *u_0[9] /*u[2][i][j - 1][k] */ +u_0[10]
		       /*u[3][i][j - 1][k] */ *u_0[10] /*u[3][i][j - 1][k] */ )
		      * tmp2 -
		      1.40e+00 * (u_0[11] /*u[4][i][j - 1][k] */ *tmp1)) *
		     (u_0[9] /*u[2][i][j - 1][k] */ *tmp1)) -
		    dt * ty1 * (-(c34 - c1345) * tmp3 *
				(((u_0[8] /*u[1][i][j - 1][k] */ ) *
				  (u_0[8] /*u[1][i][j - 1][k] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_0[9] /*u[2][i][j - 1][k] */ ) *
				  (u_0[9] /*u[2][i][j - 1][k] */ ))) - (c34 -
									c1345) *
				tmp3 *
				(((u_0[10] /*u[3][i][j - 1][k] */ ) *
				  (u_0[10] /*u[3][i][j - 1][k] */ ))) -
				c1345 * tmp2 * u_0[11] /*u[4][i][j - 1][k] */ );
		b[4][1][i][j] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_0[8] /*u[1][i][j - 1][k] */ *u_0[9]
				  /*u[2][i][j - 1][k] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_0[8] /*u[1][i][j - 1][k] */ ;
		b[4][2][i][j] =
		    -dt * ty2 * (1.40e+00 *
				 (u_0[11] /*u[4][i][j - 1][k] */ *tmp1) -
				 0.50 * 0.40e+00 *
				 ((u_0[8] /*u[1][i][j - 1][k] */ *u_0[8]
				   /*u[1][i][j - 1][k] */ +3.0 *
				   u_0[9] /*u[2][i][j - 1][k] */ *u_0[9]
				   /*u[2][i][j - 1][k] */ +u_0[10]
				   /*u[3][i][j - 1][k] */ *u_0[10]
				   /*u[3][i][j - 1][k] */ ) * tmp2)) -
		    dt * ty1 * (r43 * c34 -
				c1345) * tmp2 * u_0[9] /*u[2][i][j - 1][k] */ ;
		b[4][3][i][j] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_0[9] /*u[2][i][j - 1][k] */ *u_0[10]
				  /*u[3][i][j - 1][k] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_0[10] /*u[3][i][j - 1][k] */ ;
		b[4][4][i][j] =
		    -dt * ty2 * (1.40e+00 *
				 (u_0[9] /*u[2][i][j - 1][k] */ *tmp1)) -
		    dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
		tmp1 = 1.0 / u[0][i - 1][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		c[0][0][i][j] = -dt * tx1 * dx1;
		c[0][1][i][j] = -dt * tx2;
		c[0][2][i][j] = 0.0;
		c[0][3][i][j] = 0.0;
		c[0][4][i][j] = 0.0;
		c[1][0][i][j] =
		    -dt * tx2 * (-(u_0[0] /*u[1][i - 1][j][k] */ *tmp1) *
				 (u_0[0] /*u[1][i - 1][j][k] */ *tmp1) +
				 0.40e+00 * 0.50 *
				 (u_0[0] /*u[1][i - 1][j][k] */ *u_0[0]
				  /*u[1][i - 1][j][k] */ +u_0[1]
				  /*u[2][i - 1][j][k] */ *u_0[1]
				  /*u[2][i - 1][j][k] */ +u_0[2]
				  /*u[3][i - 1][j][k] */ *u_0[2]
				  /*u[3][i - 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-r43 * c34 * tmp2 *
				u_0[0] /*u[1][i - 1][j][k] */ );
		c[1][1][i][j] =
		    -dt * tx2 * ((2.0 - 0.40e+00) *
				 (u_0[0] /*u[1][i - 1][j][k] */ *tmp1)) -
		    dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
		c[1][2][i][j] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_0[1] /*u[2][i - 1][j][k] */ *tmp1));
		c[1][3][i][j] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_0[2] /*u[3][i - 1][j][k] */ *tmp1));
		c[1][4][i][j] = -dt * tx2 * 0.40e+00;
		c[2][0][i][j] =
		    -dt * tx2 *
		    (-
		     (u_0[0] /*u[1][i - 1][j][k] */ *u_0[1]
		      /*u[2][i - 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_0[1] /*u[2][i - 1][j][k] */ );
		c[2][1][i][j] =
		    -dt * tx2 * (u_0[1] /*u[2][i - 1][j][k] */ *tmp1);
		c[2][2][i][j] =
		    -dt * tx2 * (u_0[0] /*u[1][i - 1][j][k] */ *tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
		c[2][3][i][j] = 0.0;
		c[2][4][i][j] = 0.0;
		c[3][0][i][j] =
		    -dt * tx2 *
		    (-
		     (u_0[0] /*u[1][i - 1][j][k] */ *u_0[2]
		      /*u[3][i - 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_0[2] /*u[3][i - 1][j][k] */ );
		c[3][1][i][j] =
		    -dt * tx2 * (u_0[2] /*u[3][i - 1][j][k] */ *tmp1);
		c[3][2][i][j] = 0.0;
		c[3][3][i][j] =
		    -dt * tx2 * (u_0[0] /*u[1][i - 1][j][k] */ *tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
		c[3][4][i][j] = 0.0;
		c[4][0][i][j] =
		    -dt * tx2 *
		    ((0.40e+00 *
		      (u_0[0] /*u[1][i - 1][j][k] */ *u_0[0]
		       /*u[1][i - 1][j][k] */ +u_0[1] /*u[2][i - 1][j][k] */
		       *u_0[1] /*u[2][i - 1][j][k] */ +u_0[2]
		       /*u[3][i - 1][j][k] */ *u_0[2] /*u[3][i - 1][j][k] */ ) *
		      tmp2 -
		      1.40e+00 * (u_0[3] /*u[4][i - 1][j][k] */ *tmp1)) *
		     (u_0[0] /*u[1][i - 1][j][k] */ *tmp1)) -
		    dt * tx1 * (-(r43 * c34 - c1345) * tmp3 *
				(((u_0[0] /*u[1][i - 1][j][k] */ ) *
				  (u_0[0] /*u[1][i - 1][j][k] */ ))) - (c34 -
									c1345) *
				tmp3 *
				(((u_0[1] /*u[2][i - 1][j][k] */ ) *
				  (u_0[1] /*u[2][i - 1][j][k] */ ))) - (c34 -
									c1345) *
				tmp3 *
				(((u_0[2] /*u[3][i - 1][j][k] */ ) *
				  (u_0[2] /*u[3][i - 1][j][k] */ ))) -
				c1345 * tmp2 * u_0[3] /*u[4][i - 1][j][k] */ );
		c[4][1][i][j] =
		    -dt * tx2 * (1.40e+00 *
				 (u_0[3] /*u[4][i - 1][j][k] */ *tmp1) -
				 0.50 * 0.40e+00 *
				 ((3.0 *
				   u_0[0] /*u[1][i - 1][j][k] */ *u_0[0]
				   /*u[1][i - 1][j][k] */ +u_0[1]
				   /*u[2][i - 1][j][k] */ *u_0[1]
				   /*u[2][i - 1][j][k] */ +u_0[2]
				   /*u[3][i - 1][j][k] */ *u_0[2]
				   /*u[3][i - 1][j][k] */ ) * tmp2)) -
		    dt * tx1 * (r43 * c34 -
				c1345) * tmp2 * u_0[0] /*u[1][i - 1][j][k] */ ;
		c[4][2][i][j] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_0[1] /*u[2][i - 1][j][k] */ *u_0[0]
				  /*u[1][i - 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 * u_0[1] /*u[2][i - 1][j][k] */ ;
		c[4][3][i][j] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_0[2] /*u[3][i - 1][j][k] */ *u_0[0]
				  /*u[1][i - 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 * u_0[2] /*u[3][i - 1][j][k] */ ;
		c[4][4][i][j] =
		    -dt * tx2 * (1.40e+00 *
				 (u_0[0] /*u[1][i - 1][j][k] */ *tmp1)) -
		    dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1653 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void jacu_0(__global double *g_u, int k, __global double *g_d,
		     double dt, double tx1, double dx1, double ty1, double dy1,
		     double tz1, double dz1, double r43, double c34, double dx2,
		     double dy2, double dz2, double dx3, double dy3, double dz3,
		     double dx4, double dy4, double dz4, double c1345,
		     double dx5, double dy5, double dz5, __global double *g_a,
		     double tx2, __global double *g_b, double ty2,
		     __global double *g_c, double tz2, int jst, int ist,
		     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp1;		/* Defined at lu.c : 1642 */
	double tmp2;		/* Defined at lu.c : 1642 */
	double tmp3;		/* Defined at lu.c : 1642 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*d)[5][33][33] = (__global double (*)[5][33][33])g_d;
	__global double (*a)[5][33][33] = (__global double (*)[5][33][33])g_a;
	__global double (*b)[5][33][33] = (__global double (*)[5][33][33])g_b;
	__global double (*c)[5][33][33] = (__global double (*)[5][33][33])g_c;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1654
		//-------------------------------------------
		double u_1[15];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1654
		//Candidates:
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//      u[4][i][j][k]
		//      u[1][i][j][k + 1]
		//      u[2][i][j][k + 1]
		//      u[3][i][j][k + 1]
		//      u[1][i][j + 1][k]
		//      u[2][i][j + 1][k]
		//      u[3][i][j + 1][k]
		//      u[4][i][j + 1][k]
		//      u[1][i + 1][j][k]
		//      u[2][i + 1][j][k]
		//      u[3][i + 1][j][k]
		//      u[4][i + 1][j][k]
		//-------------------------------------------
		u_1[0] = u[1][i][j][k];
		u_1[1] = u[2][i][j][k];
		u_1[2] = u[3][i][j][k];
		u_1[3] = u[4][i][j][k];
		u_1[4] = u[1][i][j][k + 1];
		u_1[5] = u[2][i][j][k + 1];
		u_1[6] = u[3][i][j][k + 1];
		u_1[7] = u[1][i][j + 1][k];
		u_1[8] = u[2][i][j + 1][k];
		u_1[9] = u[3][i][j + 1][k];
		u_1[10] = u[4][i][j + 1][k];
		u_1[11] = u[1][i + 1][j][k];
		u_1[12] = u[2][i + 1][j][k];
		u_1[13] = u[3][i + 1][j][k];
		u_1[14] = u[4][i + 1][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u[0][i][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		d[0][0][i][j] =
		    1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
		d[0][1][i][j] = 0.0;
		d[0][2][i][j] = 0.0;
		d[0][3][i][j] = 0.0;
		d[0][4][i][j] = 0.0;
		d[1][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-r43 * c34 * tmp2 *
				 u_1[0] /*u[1][i][j][k] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_1[0] /*u[1][i][j][k] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_1[0] /*u[1][i][j][k] */ ));
		d[1][1][i][j] =
		    1.0 + dt * 2.0 * (tx1 * r43 * c34 * tmp1 +
				      ty1 * c34 * tmp1 + tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
		d[1][2][i][j] = 0.0;
		d[1][3][i][j] = 0.0;
		d[1][4][i][j] = 0.0;
		d[2][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_1[1] /*u[2][i][j][k] */ ) +
				ty1 * (-r43 * c34 * tmp2 *
				       u_1[1] /*u[2][i][j][k] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_1[1] /*u[2][i][j][k] */ ));
		d[2][1][i][j] = 0.0;
		d[2][2][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 +
				      ty1 * r43 * c34 * tmp1 +
				      tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
		d[2][3][i][j] = 0.0;
		d[2][4][i][j] = 0.0;
		d[3][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_1[2] /*u[3][i][j][k] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_1[2] /*u[3][i][j][k] */ ) +
				tz1 * (-r43 * c34 * tmp2 *
				       u_1[2] /*u[3][i][j][k] */ ));
		d[3][1][i][j] = 0.0;
		d[3][2][i][j] = 0.0;
		d[3][3][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 +
				      tz1 * r43 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
		d[3][4][i][j] = 0.0;
		d[4][0][i][j] =
		    dt * 2.0 * (tx1 *
				(-(r43 * c34 - c1345) * tmp3 *
				 (((u_1[0] /*u[1][i][j][k] */ ) *
				   (u_1[0] /*u[1][i][j][k] */ ))) - (c34 -
								     c1345) *
				 tmp3 *
				 (((u_1[1] /*u[2][i][j][k] */ ) *
				   (u_1[1] /*u[2][i][j][k] */ ))) - (c34 -
								     c1345) *
				 tmp3 *
				 (((u_1[2] /*u[3][i][j][k] */ ) *
				   (u_1[2] /*u[3][i][j][k] */ ))) -
				 (c1345) * tmp2 * u_1[3] /*u[4][i][j][k] */ ) +
				ty1 * (-(c34 - c1345) * tmp3 *
				       (((u_1[0] /*u[1][i][j][k] */ ) *
					 (u_1[0] /*u[1][i][j][k] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_1[1] /*u[2][i][j][k] */ ) *
					 (u_1[1] /*u[2][i][j][k] */ ))) - (c34 -
									   c1345)
				       * tmp3 *
				       (((u_1[2] /*u[3][i][j][k] */ ) *
					 (u_1[2] /*u[3][i][j][k] */ ))) -
				       (c1345) * tmp2 *
				       u_1[3] /*u[4][i][j][k] */ ) +
				tz1 * (-(c34 - c1345) * tmp3 *
				       (((u_1[0] /*u[1][i][j][k] */ ) *
					 (u_1[0] /*u[1][i][j][k] */ ))) - (c34 -
									   c1345)
				       * tmp3 *
				       (((u_1[1] /*u[2][i][j][k] */ ) *
					 (u_1[1] /*u[2][i][j][k] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_1[2] /*u[3][i][j][k] */ ) *
					 (u_1[2] /*u[3][i][j][k] */ ))) -
				       (c1345) * tmp2 *
				       u_1[3] /*u[4][i][j][k] */ ));
		d[4][1][i][j] =
		    dt * 2.0 * (tx1 * (r43 * c34 - c1345) * tmp2 *
				u_1[0] /*u[1][i][j][k] */ +ty1 * (c34 -
								  c1345) *
				tmp2 * u_1[0] /*u[1][i][j][k] */ +tz1 * (c34 -
									 c1345)
				* tmp2 * u_1[0] /*u[1][i][j][k] */ );
		d[4][2][i][j] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_1[1] /*u[2][i][j][k] */ +ty1 * (r43 * c34 -
								  c1345) *
				tmp2 * u_1[1] /*u[2][i][j][k] */ +tz1 * (c34 -
									 c1345)
				* tmp2 * u_1[1] /*u[2][i][j][k] */ );
		d[4][3][i][j] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_1[2] /*u[3][i][j][k] */ +ty1 * (c34 -
								  c1345) *
				tmp2 * u_1[2] /*u[3][i][j][k] */ +tz1 * (r43 *
									 c34 -
									 c1345)
				* tmp2 * u_1[2] /*u[3][i][j][k] */ );
		d[4][4][i][j] =
		    1.0 + dt * 2.0 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 +
				      tz1 * c1345 * tmp1) +
		    dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
		tmp1 = 1.0 / u[0][i + 1][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		a[0][0][i][j] = -dt * tx1 * dx1;
		a[0][1][i][j] = dt * tx2;
		a[0][2][i][j] = 0.0;
		a[0][3][i][j] = 0.0;
		a[0][4][i][j] = 0.0;
		a[1][0][i][j] =
		    dt * tx2 * (-(u_1[11] /*u[1][i + 1][j][k] */ *tmp1) *
				(u_1[11] /*u[1][i + 1][j][k] */ *tmp1) +
				0.40e+00 * 0.50 *
				(u_1[11] /*u[1][i + 1][j][k] */ *u_1[11]
				 /*u[1][i + 1][j][k] */ +u_1[12]
				 /*u[2][i + 1][j][k] */ *u_1[12]
				 /*u[2][i + 1][j][k] */ +u_1[13]
				 /*u[3][i + 1][j][k] */ *u_1[13]
				 /*u[3][i + 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-r43 * c34 * tmp2 *
				u_1[11] /*u[1][i + 1][j][k] */ );
		a[1][1][i][j] =
		    dt * tx2 * ((2.0 - 0.40e+00) *
				(u_1[11] /*u[1][i + 1][j][k] */ *tmp1)) -
		    dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
		a[1][2][i][j] =
		    dt * tx2 * (-0.40e+00 *
				(u_1[12] /*u[2][i + 1][j][k] */ *tmp1));
		a[1][3][i][j] =
		    dt * tx2 * (-0.40e+00 *
				(u_1[13] /*u[3][i + 1][j][k] */ *tmp1));
		a[1][4][i][j] = dt * tx2 * 0.40e+00;
		a[2][0][i][j] =
		    dt * tx2 *
		    (-
		     (u_1[11] /*u[1][i + 1][j][k] */ *u_1[12]
		      /*u[2][i + 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_1[12] /*u[2][i + 1][j][k] */ );
		a[2][1][i][j] =
		    dt * tx2 * (u_1[12] /*u[2][i + 1][j][k] */ *tmp1);
		a[2][2][i][j] =
		    dt * tx2 * (u_1[11] /*u[1][i + 1][j][k] */ *tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
		a[2][3][i][j] = 0.0;
		a[2][4][i][j] = 0.0;
		a[3][0][i][j] =
		    dt * tx2 *
		    (-
		     (u_1[11] /*u[1][i + 1][j][k] */ *u_1[13]
		      /*u[3][i + 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_1[13] /*u[3][i + 1][j][k] */ );
		a[3][1][i][j] =
		    dt * tx2 * (u_1[13] /*u[3][i + 1][j][k] */ *tmp1);
		a[3][2][i][j] = 0.0;
		a[3][3][i][j] =
		    dt * tx2 * (u_1[11] /*u[1][i + 1][j][k] */ *tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
		a[3][4][i][j] = 0.0;
		a[4][0][i][j] =
		    dt * tx2 *
		    ((0.40e+00 *
		      (u_1[11] /*u[1][i + 1][j][k] */ *u_1[11]
		       /*u[1][i + 1][j][k] */ +u_1[12] /*u[2][i + 1][j][k] */
		       *u_1[12] /*u[2][i + 1][j][k] */ +u_1[13]
		       /*u[3][i + 1][j][k] */ *u_1[13] /*u[3][i + 1][j][k] */ )
		      * tmp2 -
		      1.40e+00 * (u_1[14] /*u[4][i + 1][j][k] */ *tmp1)) *
		     (u_1[11] /*u[1][i + 1][j][k] */ *tmp1)) -
		    dt * tx1 * (-(r43 * c34 - c1345) * tmp3 *
				(((u_1[11] /*u[1][i + 1][j][k] */ ) *
				  (u_1[11] /*u[1][i + 1][j][k] */ ))) - (c34 -
									 c1345)
				* tmp3 *
				(((u_1[12] /*u[2][i + 1][j][k] */ ) *
				  (u_1[12] /*u[2][i + 1][j][k] */ ))) - (c34 -
									 c1345)
				* tmp3 *
				(((u_1[13] /*u[3][i + 1][j][k] */ ) *
				  (u_1[13] /*u[3][i + 1][j][k] */ ))) -
				c1345 * tmp2 * u_1[14] /*u[4][i + 1][j][k] */ );
		a[4][1][i][j] =
		    dt * tx2 * (1.40e+00 *
				(u_1[14] /*u[4][i + 1][j][k] */ *tmp1) -
				0.50 * 0.40e+00 *
				((3.0 *
				  u_1[11] /*u[1][i + 1][j][k] */ *u_1[11]
				  /*u[1][i + 1][j][k] */ +u_1[12]
				  /*u[2][i + 1][j][k] */ *u_1[12]
				  /*u[2][i + 1][j][k] */ +u_1[13]
				  /*u[3][i + 1][j][k] */ *u_1[13]
				  /*u[3][i + 1][j][k] */ ) * tmp2)) -
		    dt * tx1 * (r43 * c34 -
				c1345) * tmp2 * u_1[11] /*u[1][i + 1][j][k] */ ;
		a[4][2][i][j] =
		    dt * tx2 * (-0.40e+00 *
				(u_1[12] /*u[2][i + 1][j][k] */ *u_1[11]
				 /*u[1][i + 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 * u_1[12] /*u[2][i + 1][j][k] */ ;
		a[4][3][i][j] =
		    dt * tx2 * (-0.40e+00 *
				(u_1[13] /*u[3][i + 1][j][k] */ *u_1[11]
				 /*u[1][i + 1][j][k] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 * u_1[13] /*u[3][i + 1][j][k] */ ;
		a[4][4][i][j] =
		    dt * tx2 * (1.40e+00 *
				(u_1[11] /*u[1][i + 1][j][k] */ *tmp1)) -
		    dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
		tmp1 = 1.0 / u[0][i][j + 1][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		b[0][0][i][j] = -dt * ty1 * dy1;
		b[0][1][i][j] = 0.0;
		b[0][2][i][j] = dt * ty2;
		b[0][3][i][j] = 0.0;
		b[0][4][i][j] = 0.0;
		b[1][0][i][j] =
		    dt * ty2 *
		    (-
		     (u_1[7] /*u[1][i][j + 1][k] */ *u_1[8]
		      /*u[2][i][j + 1][k] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_1[7] /*u[1][i][j + 1][k] */ );
		b[1][1][i][j] =
		    dt * ty2 * (u_1[8] /*u[2][i][j + 1][k] */ *tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
		b[1][2][i][j] =
		    dt * ty2 * (u_1[7] /*u[1][i][j + 1][k] */ *tmp1);
		b[1][3][i][j] = 0.0;
		b[1][4][i][j] = 0.0;
		b[2][0][i][j] =
		    dt * ty2 * (-(u_1[8] /*u[2][i][j + 1][k] */ *tmp1) *
				(u_1[8] /*u[2][i][j + 1][k] */ *tmp1) +
				0.50 * 0.40e+00 *
				((u_1[7] /*u[1][i][j + 1][k] */ *u_1[7]
				  /*u[1][i][j + 1][k] */ +u_1[8]
				  /*u[2][i][j + 1][k] */ *u_1[8]
				  /*u[2][i][j + 1][k] */ +u_1[9]
				  /*u[3][i][j + 1][k] */ *u_1[9]
				  /*u[3][i][j + 1][k] */ ) * tmp2)) -
		    dt * ty1 * (-r43 * c34 * tmp2 *
				u_1[8] /*u[2][i][j + 1][k] */ );
		b[2][1][i][j] =
		    dt * ty2 * (-0.40e+00 *
				(u_1[7] /*u[1][i][j + 1][k] */ *tmp1));
		b[2][2][i][j] =
		    dt * ty2 * ((2.0 - 0.40e+00) *
				(u_1[8] /*u[2][i][j + 1][k] */ *tmp1)) -
		    dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
		b[2][3][i][j] =
		    dt * ty2 * (-0.40e+00 *
				(u_1[9] /*u[3][i][j + 1][k] */ *tmp1));
		b[2][4][i][j] = dt * ty2 * 0.40e+00;
		b[3][0][i][j] =
		    dt * ty2 *
		    (-
		     (u_1[8] /*u[2][i][j + 1][k] */ *u_1[9]
		      /*u[3][i][j + 1][k] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_1[9] /*u[3][i][j + 1][k] */ );
		b[3][1][i][j] = 0.0;
		b[3][2][i][j] =
		    dt * ty2 * (u_1[9] /*u[3][i][j + 1][k] */ *tmp1);
		b[3][3][i][j] =
		    dt * ty2 * (u_1[8] /*u[2][i][j + 1][k] */ *tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
		b[3][4][i][j] = 0.0;
		b[4][0][i][j] =
		    dt * ty2 *
		    ((0.40e+00 *
		      (u_1[7] /*u[1][i][j + 1][k] */ *u_1[7]
		       /*u[1][i][j + 1][k] */ +u_1[8] /*u[2][i][j + 1][k] */
		       *u_1[8] /*u[2][i][j + 1][k] */ +u_1[9]
		       /*u[3][i][j + 1][k] */ *u_1[9] /*u[3][i][j + 1][k] */ ) *
		      tmp2 -
		      1.40e+00 * (u_1[10] /*u[4][i][j + 1][k] */ *tmp1)) *
		     (u_1[8] /*u[2][i][j + 1][k] */ *tmp1)) -
		    dt * ty1 * (-(c34 - c1345) * tmp3 *
				(((u_1[7] /*u[1][i][j + 1][k] */ ) *
				  (u_1[7] /*u[1][i][j + 1][k] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_1[8] /*u[2][i][j + 1][k] */ ) *
				  (u_1[8] /*u[2][i][j + 1][k] */ ))) - (c34 -
									c1345) *
				tmp3 *
				(((u_1[9] /*u[3][i][j + 1][k] */ ) *
				  (u_1[9] /*u[3][i][j + 1][k] */ ))) -
				c1345 * tmp2 * u_1[10] /*u[4][i][j + 1][k] */ );
		b[4][1][i][j] =
		    dt * ty2 * (-0.40e+00 *
				(u_1[7] /*u[1][i][j + 1][k] */ *u_1[8]
				 /*u[2][i][j + 1][k] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_1[7] /*u[1][i][j + 1][k] */ ;
		b[4][2][i][j] =
		    dt * ty2 * (1.40e+00 *
				(u_1[10] /*u[4][i][j + 1][k] */ *tmp1) -
				0.50 * 0.40e+00 *
				((u_1[7] /*u[1][i][j + 1][k] */ *u_1[7]
				  /*u[1][i][j + 1][k] */ +3.0 *
				  u_1[8] /*u[2][i][j + 1][k] */ *u_1[8]
				  /*u[2][i][j + 1][k] */ +u_1[9]
				  /*u[3][i][j + 1][k] */ *u_1[9]
				  /*u[3][i][j + 1][k] */ ) * tmp2)) -
		    dt * ty1 * (r43 * c34 -
				c1345) * tmp2 * u_1[8] /*u[2][i][j + 1][k] */ ;
		b[4][3][i][j] =
		    dt * ty2 * (-0.40e+00 *
				(u_1[8] /*u[2][i][j + 1][k] */ *u_1[9]
				 /*u[3][i][j + 1][k] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_1[9] /*u[3][i][j + 1][k] */ ;
		b[4][4][i][j] =
		    dt * ty2 * (1.40e+00 *
				(u_1[8] /*u[2][i][j + 1][k] */ *tmp1)) -
		    dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
		tmp1 = 1.0 / u[0][i][j][k + 1];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		c[0][0][i][j] = -dt * tz1 * dz1;
		c[0][1][i][j] = 0.0;
		c[0][2][i][j] = 0.0;
		c[0][3][i][j] = dt * tz2;
		c[0][4][i][j] = 0.0;
		c[1][0][i][j] =
		    dt * tz2 *
		    (-
		     (u_1[4] /*u[1][i][j][k + 1] */ *u_1[6]
		      /*u[3][i][j][k + 1] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_1[4] /*u[1][i][j][k + 1] */ );
		c[1][1][i][j] =
		    dt * tz2 * (u_1[6] /*u[3][i][j][k + 1] */ *tmp1) -
		    dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
		c[1][2][i][j] = 0.0;
		c[1][3][i][j] =
		    dt * tz2 * (u_1[4] /*u[1][i][j][k + 1] */ *tmp1);
		c[1][4][i][j] = 0.0;
		c[2][0][i][j] =
		    dt * tz2 *
		    (-
		     (u_1[5] /*u[2][i][j][k + 1] */ *u_1[6]
		      /*u[3][i][j][k + 1] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_1[5] /*u[2][i][j][k + 1] */ );
		c[2][1][i][j] = 0.0;
		c[2][2][i][j] =
		    dt * tz2 * (u_1[6] /*u[3][i][j][k + 1] */ *tmp1) -
		    dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
		c[2][3][i][j] =
		    dt * tz2 * (u_1[5] /*u[2][i][j][k + 1] */ *tmp1);
		c[2][4][i][j] = 0.0;
		c[3][0][i][j] =
		    dt * tz2 * (-(u_1[6] /*u[3][i][j][k + 1] */ *tmp1) *
				(u_1[6] /*u[3][i][j][k + 1] */ *tmp1) +
				0.50 * 0.40e+00 *
				((u_1[4] /*u[1][i][j][k + 1] */ *u_1[4]
				  /*u[1][i][j][k + 1] */ +u_1[5]
				  /*u[2][i][j][k + 1] */ *u_1[5]
				  /*u[2][i][j][k + 1] */ +u_1[6]
				  /*u[3][i][j][k + 1] */ *u_1[6]
				  /*u[3][i][j][k + 1] */ ) * tmp2)) -
		    dt * tz1 * (-r43 * c34 * tmp2 *
				u_1[6] /*u[3][i][j][k + 1] */ );
		c[3][1][i][j] =
		    dt * tz2 * (-0.40e+00 *
				(u_1[4] /*u[1][i][j][k + 1] */ *tmp1));
		c[3][2][i][j] =
		    dt * tz2 * (-0.40e+00 *
				(u_1[5] /*u[2][i][j][k + 1] */ *tmp1));
		c[3][3][i][j] =
		    dt * tz2 * (2.0 -
				0.40e+00) *
		    (u_1[6] /*u[3][i][j][k + 1] */ *tmp1) -
		    dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
		c[3][4][i][j] = dt * tz2 * 0.40e+00;
		c[4][0][i][j] =
		    dt * tz2 *
		    ((0.40e+00 *
		      (u_1[4] /*u[1][i][j][k + 1] */ *u_1[4]
		       /*u[1][i][j][k + 1] */ +u_1[5] /*u[2][i][j][k + 1] */
		       *u_1[5] /*u[2][i][j][k + 1] */ +u_1[6]
		       /*u[3][i][j][k + 1] */ *u_1[6] /*u[3][i][j][k + 1] */ ) *
		      tmp2 -
		      1.40e+00 * (u[4][i][j][k + 1] * tmp1)) *
		     (u_1[6] /*u[3][i][j][k + 1] */ *tmp1)) -
		    dt * tz1 * (-(c34 - c1345) * tmp3 *
				(((u_1[4] /*u[1][i][j][k + 1] */ ) *
				  (u_1[4] /*u[1][i][j][k + 1] */ ))) - (c34 -
									c1345) *
				tmp3 *
				(((u_1[5] /*u[2][i][j][k + 1] */ ) *
				  (u_1[5] /*u[2][i][j][k + 1] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_1[6] /*u[3][i][j][k + 1] */ ) *
				  (u_1[6] /*u[3][i][j][k + 1] */ ))) -
				c1345 * tmp2 * u[4][i][j][k + 1]);
		c[4][1][i][j] =
		    dt * tz2 * (-0.40e+00 *
				(u_1[4] /*u[1][i][j][k + 1] */ *u_1[6]
				 /*u[3][i][j][k + 1] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 * u_1[4] /*u[1][i][j][k + 1] */ ;
		c[4][2][i][j] =
		    dt * tz2 * (-0.40e+00 *
				(u_1[5] /*u[2][i][j][k + 1] */ *u_1[6]
				 /*u[3][i][j][k + 1] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 * u_1[5] /*u[2][i][j][k + 1] */ ;
		c[4][3][i][j] =
		    dt * tz2 * (1.40e+00 * (u[4][i][j][k + 1] * tmp1) -
				0.50 * 0.40e+00 *
				((u_1[4] /*u[1][i][j][k + 1] */ *u_1[4]
				  /*u[1][i][j][k + 1] */ +u_1[5]
				  /*u[2][i][j][k + 1] */ *u_1[5]
				  /*u[2][i][j][k + 1] */ +3.0 *
				  u_1[6] /*u[3][i][j][k + 1] */ *u_1[6]
				  /*u[3][i][j][k + 1] */ ) * tmp2)) -
		    dt * tz1 * (r43 * c34 -
				c1345) * tmp2 * u_1[6] /*u[3][i][j][k + 1] */ ;
		c[4][4][i][j] =
		    dt * tz2 * (1.40e+00 *
				(u_1[6] /*u[3][i][j][k + 1] */ *tmp1)) -
		    dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void l2norm_0_reduction_step0(__global double *__ocl_part_sum0,
				       __global double *__ocl_part_sum1,
				       __global double *__ocl_part_sum2,
				       __global double *__ocl_part_sum3,
				       __global double *__ocl_part_sum4,
				       unsigned int offset, unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_sum0[i] = 0.0;
	__ocl_part_sum1[i] = 0.0;
	__ocl_part_sum2[i] = 0.0;
	__ocl_part_sum3[i] = 0.0;
	__ocl_part_sum4[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 2026 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void l2norm_0_reduction_step1(__global double *g_v, int jst, int ist,
				       int __ocl_k_bound, int __ocl_j_bound,
				       int __ocl_i_bound,
				       __global double *__ocl_part_sum0,
				       __global double *__ocl_part_sum1,
				       __global double *__ocl_part_sum2,
				       __global double *__ocl_part_sum3,
				       __global double *__ocl_part_sum4)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*v)[33][33][33] = (__global double (*)[33][33][33])g_v;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double sum0 = 0.0;	/* reduction variable, defined at: lu.c : 2019 */
	double sum1 = 0.0;	/* reduction variable, defined at: lu.c : 2019 */
	double sum2 = 0.0;	/* reduction variable, defined at: lu.c : 2019 */
	double sum3 = 0.0;	/* reduction variable, defined at: lu.c : 2019 */
	double sum4 = 0.0;	/* reduction variable, defined at: lu.c : 2019 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2028
		//-------------------------------------------
		double v_0[4];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2028
		//Candidates:
		//      v[0][i][j][k]
		//      v[1][i][j][k]
		//      v[2][i][j][k]
		//      v[3][i][j][k]
		//-------------------------------------------
		v_0[0] = v[0][i][j][k];
		v_0[1] = v[1][i][j][k];
		v_0[2] = v[2][i][j][k];
		v_0[3] = v[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		sum0 =
		    sum0 +
		    v_0[0] /*v[0][i][j][k] */ *v_0[0] /*v[0][i][j][k] */ ;
		sum1 =
		    sum1 +
		    v_0[1] /*v[1][i][j][k] */ *v_0[1] /*v[1][i][j][k] */ ;
		sum2 =
		    sum2 +
		    v_0[2] /*v[2][i][j][k] */ *v_0[2] /*v[2][i][j][k] */ ;
		sum3 =
		    sum3 +
		    v_0[3] /*v[3][i][j][k] */ *v_0[3] /*v[3][i][j][k] */ ;
		sum4 = sum4 + v[4][i][j][k] * v[4][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx =
		    CALC_3D_IDX(get_global_size(2), get_global_size(1),
				get_global_size(0), get_global_id(2),
				get_global_id(1), get_global_id(0));
		__ocl_part_sum0[__ocl_wb_idx] = sum0;
		__ocl_part_sum1[__ocl_wb_idx] = sum1;
		__ocl_part_sum2[__ocl_wb_idx] = sum2;
		__ocl_part_sum3[__ocl_wb_idx] = sum3;
		__ocl_part_sum4[__ocl_wb_idx] = sum4;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void l2norm_0_reduction_step2(__global double4 * input_sum0,
				       __global double *output_sum0,
				       __global double4 * input_sum1,
				       __global double *output_sum1,
				       __global double4 * input_sum2,
				       __global double *output_sum2,
				       __global double4 * input_sum3,
				       __global double *output_sum3,
				       __global double4 * input_sum4,
				       __global double *output_sum4)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_sum0[GROUP_SIZE];
	__local double4 sdata_sum1[GROUP_SIZE];
	__local double4 sdata_sum2[GROUP_SIZE];
	__local double4 sdata_sum3[GROUP_SIZE];
	__local double4 sdata_sum4[GROUP_SIZE];
	sdata_sum0[tid] = input_sum0[gid];
	sdata_sum1[tid] = input_sum1[gid];
	sdata_sum2[tid] = input_sum2[gid];
	sdata_sum3[tid] = input_sum3[gid];
	sdata_sum4[tid] = input_sum4[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum0[tid] += sdata_sum0[tid + s];
			sdata_sum1[tid] += sdata_sum1[tid + s];
			sdata_sum2[tid] += sdata_sum2[tid + s];
			sdata_sum3[tid] += sdata_sum3[tid + s];
			sdata_sum4[tid] += sdata_sum4[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_sum0[bid] =
		    (sdata_sum0[0].x + sdata_sum0[0].y + sdata_sum0[0].z +
		     sdata_sum0[0].w);
		output_sum1[bid] =
		    (sdata_sum1[0].x + sdata_sum1[0].y + sdata_sum1[0].z +
		     sdata_sum1[0].w);
		output_sum2[bid] =
		    (sdata_sum2[0].x + sdata_sum2[0].y + sdata_sum2[0].z +
		     sdata_sum2[0].w);
		output_sum3[bid] =
		    (sdata_sum3[0].x + sdata_sum3[0].y + sdata_sum3[0].z +
		     sdata_sum3[0].w);
		output_sum4[bid] =
		    (sdata_sum4[0].x + sdata_sum4[0].y + sdata_sum4[0].z +
		     sdata_sum4[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 2366 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_0(__global double *g_rsd, int m, __global double *g_frct,
		    int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	int i = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	__global double (*frct)[33][33][33] =
	    (__global double (*)[33][33][33])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rsd[m][i][j][k] = -frct[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2384 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_1(__global double *g_flux, __global double *g_u, int jst,
		    int L1, int __ocl_k_bound, int __ocl_j_bound,
		    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + L1;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u21;		/* Defined at lu.c : 2356 */
	double q;		/* Defined at lu.c : 2355 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2386
		//-------------------------------------------
		double u_2[4];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2386
		//Candidates:
		//      u[0][i][j][k]
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_2[0] = u[0][i][j][k];
		u_2[1] = u[1][i][j][k];
		u_2[2] = u[2][i][j][k];
		u_2[3] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[0][i][j][k] = u_2[1] /*u[1][i][j][k] */ ;
		u21 = u_2[1] /*u[1][i][j][k] */ /u_2[0] /*u[0][i][j][k] */ ;
		q = 0.50 *
		    (u_2[1] /*u[1][i][j][k] */ *u_2[1] /*u[1][i][j][k] */
		     +u_2[2] /*u[2][i][j][k] */ *u_2[2] /*u[2][i][j][k] */
		     +u_2[3] /*u[3][i][j][k] */ *u_2[3] /*u[3][i][j][k] */ ) /
		    u_2[0] /*u[0][i][j][k] */ ;
		flux[1][i][j][k] =
		    u_2[1] /*u[1][i][j][k] */ *u21 + 0.40e+00 * (u[4][i][j][k] -
								 q);
		flux[2][i][j][k] = u_2[2] /*u[2][i][j][k] */ *u21;
		flux[3][i][j][k] = u_2[3] /*u[3][i][j][k] */ *u21;
		flux[4][i][j][k] =
		    (1.40e+00 * u[4][i][j][k] - 0.40e+00 * q) * u21;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2405 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_2(int i, int ist, int iend, int m, __global double *g_rsd,
		    double tx2, __global double *g_flux, int nx,
		    __global double *g_u, double tx3, double dx1, double tx1,
		    double dx2, double dx3, double dx4, double dx5, double dssp,
		    int jst, int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int L2;			/* Defined at lu.c : 2352 */
	double tmp;		/* Defined at lu.c : 2357 */
	double u21i;		/* Defined at lu.c : 2358 */
	double u31i;		/* Defined at lu.c : 2358 */
	double u41i;		/* Defined at lu.c : 2358 */
	double u51i;		/* Defined at lu.c : 2358 */
	double u21im1;		/* Defined at lu.c : 2361 */
	double u31im1;		/* Defined at lu.c : 2361 */
	double u41im1;		/* Defined at lu.c : 2361 */
	double u51im1;		/* Defined at lu.c : 2361 */
	int ist1;		/* Defined at lu.c : 2353 */
	int iend1;		/* Defined at lu.c : 2353 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = ist; i <= iend; i++) {
			for (m = 0; m < 5; m++) {
				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    tx2 * (flux[m][i + 1][j][k] -
					   flux[m][i - 1][j][k]);
			}
		}
		L2 = nx - 1;
		for (i = ist; i <= L2; i++) {
			tmp = 1.0 / u[0][i][j][k];
			u21i = tmp * u[1][i][j][k];
			u31i = tmp * u[2][i][j][k];
			u41i = tmp * u[3][i][j][k];
			u51i = tmp * u[4][i][j][k];
			tmp = 1.0 / u[0][i - 1][j][k];
			u21im1 = tmp * u[1][i - 1][j][k];
			u31im1 = tmp * u[2][i - 1][j][k];
			u41im1 = tmp * u[3][i - 1][j][k];
			u51im1 = tmp * u[4][i - 1][j][k];
			flux[1][i][j][k] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
			flux[2][i][j][k] = tx3 * (u31i - u31im1);
			flux[3][i][j][k] = tx3 * (u41i - u41im1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * tx3 *
			    ((((u21i) * (u21i)) + ((u31i) * (u31i)) +
			      ((u41i) * (u41i))) - (((u21im1) * (u21im1)) +
						    ((u31im1) * (u31im1)) +
						    ((u41im1) * (u41im1)))) +
			    (1.0 / 6.0) * tx3 * (((u21i) * (u21i)) -
						 ((u21im1) * (u21im1))) +
			    1.40e+00 * 1.40e+00 * tx3 * (u51i - u51im1);
		}
		for (i = ist; i <= iend; i++) {
			rsd[0][i][j][k] =
			    rsd[0][i][j][k] + dx1 * tx1 * (u[0][i - 1][j][k] -
							   2.0 * u[0][i][j][k] +
							   u[0][i + 1][j][k]);
			rsd[1][i][j][k] =
			    rsd[1][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[1][i + 1][j][k] -
							 flux[1][i][j][k]) +
			    dx2 * tx1 * (u[1][i - 1][j][k] -
					 2.0 * u[1][i][j][k] + u[1][i +
								    1][j][k]);
			rsd[2][i][j][k] =
			    rsd[2][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[2][i + 1][j][k] -
							 flux[2][i][j][k]) +
			    dx3 * tx1 * (u[2][i - 1][j][k] -
					 2.0 * u[2][i][j][k] + u[2][i +
								    1][j][k]);
			rsd[3][i][j][k] =
			    rsd[3][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[3][i + 1][j][k] -
							 flux[3][i][j][k]) +
			    dx4 * tx1 * (u[3][i - 1][j][k] -
					 2.0 * u[3][i][j][k] + u[3][i +
								    1][j][k]);
			rsd[4][i][j][k] =
			    rsd[4][i][j][k] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[4][i + 1][j][k] -
							 flux[4][i][j][k]) +
			    dx5 * tx1 * (u[4][i - 1][j][k] -
					 2.0 * u[4][i][j][k] + u[4][i +
								    1][j][k]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2472
			//-------------------------------------------
			double u_3[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2472
			//Candidates:
			//      u[m][1][j][k]
			//      u[m][2][j][k]
			//      u[m][3][j][k]
			//-------------------------------------------
			u_3[0] = u[m][1][j][k];
			u_3[1] = u[m][2][j][k];
			u_3[2] = u[m][3][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][1][j][k] =
			    rsd[m][1][j][k] -
			    dssp * (+5.0 * u_3[0] /*u[m][1][j][k] */ -4.0 *
				    u_3[1] /*u[m][2][j][k] */ +u_3[2]
				    /*u[m][3][j][k] */ );
			rsd[m][2][j][k] =
			    rsd[m][2][j][k] -
			    dssp * (-4.0 * u_3[0] /*u[m][1][j][k] */ +6.0 *
				    u_3[1] /*u[m][2][j][k] */ -4.0 *
				    u_3[2] /*u[m][3][j][k] */ +u[m][4][j][k]);
		}
		ist1 = 3;
		iend1 = nx - 4;
		for (i = ist1; i <= iend1; i++) {
			for (m = 0; m < 5; m++) {
				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    dssp * (u[m][i - 2][j][k] -
					    4.0 * u[m][i - 1][j][k] +
					    6.0 * u[m][i][j][k] - 4.0 * u[m][i +
									     1]
					    [j][k] + u[m][i + 2][j][k]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2499
			//-------------------------------------------
			double u_4[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2499
			//Candidates:
			//      u[m][nx - 4][j][k]
			//      u[m][nx - 3][j][k]
			//-------------------------------------------
			u_4[0] = u[m][nx - 4][j][k];
			u_4[1] = u[m][nx - 3][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][nx - 3][j][k] =
			    rsd[m][nx - 3][j][k] - dssp * (u[m][nx - 5][j][k] -
							   4.0 *
							   u_4[0]
							   /*u[m][nx - 4][j][k] */
							   +6.0 *
							   u_4[1]
							   /*u[m][nx - 3][j][k] */
							   -4.0 * u[m][nx -
								       2][j]
							   [k]);
			rsd[m][nx - 2][j][k] =
			    rsd[m][nx - 2][j][k] -
			    dssp * (u_4[0] /*u[m][nx - 4][j][k] */ -4.0 *
				    u_4[1] /*u[m][nx - 3][j][k] */ +5.0 *
				    u[m][nx - 2][j][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2521 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_3(__global double *g_flux, __global double *g_u, int L1,
		    int ist, int __ocl_k_bound, int __ocl_j_bound,
		    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + L1;
	int i = get_global_id(2) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u31;		/* Defined at lu.c : 2356 */
	double q;		/* Defined at lu.c : 2355 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2523
		//-------------------------------------------
		double u_5[4];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2523
		//Candidates:
		//      u[0][i][j][k]
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_5[0] = u[0][i][j][k];
		u_5[1] = u[1][i][j][k];
		u_5[2] = u[2][i][j][k];
		u_5[3] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[0][i][j][k] = u_5[2] /*u[2][i][j][k] */ ;
		u31 = u_5[2] /*u[2][i][j][k] */ /u_5[0] /*u[0][i][j][k] */ ;
		q = 0.50 *
		    (u_5[1] /*u[1][i][j][k] */ *u_5[1] /*u[1][i][j][k] */
		     +u_5[2] /*u[2][i][j][k] */ *u_5[2] /*u[2][i][j][k] */
		     +u_5[3] /*u[3][i][j][k] */ *u_5[3] /*u[3][i][j][k] */ ) /
		    u_5[0] /*u[0][i][j][k] */ ;
		flux[1][i][j][k] = u_5[1] /*u[1][i][j][k] */ *u31;
		flux[2][i][j][k] =
		    u_5[2] /*u[2][i][j][k] */ *u31 + 0.40e+00 * (u[4][i][j][k] -
								 q);
		flux[3][i][j][k] = u_5[3] /*u[3][i][j][k] */ *u31;
		flux[4][i][j][k] =
		    (1.40e+00 * u[4][i][j][k] - 0.40e+00 * q) * u31;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2541 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_4(int j, int jst, int jend, int m, __global double *g_rsd,
		    double ty2, __global double *g_flux, int ny,
		    __global double *g_u, double ty3, double dy1, double ty1,
		    double dy2, double dy3, double dy4, double dy5, double dssp,
		    int ist, int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int L2;			/* Defined at lu.c : 2352 */
	double tmp;		/* Defined at lu.c : 2357 */
	double u21j;		/* Defined at lu.c : 2359 */
	double u31j;		/* Defined at lu.c : 2359 */
	double u41j;		/* Defined at lu.c : 2359 */
	double u51j;		/* Defined at lu.c : 2359 */
	double u21jm1;		/* Defined at lu.c : 2362 */
	double u31jm1;		/* Defined at lu.c : 2362 */
	double u41jm1;		/* Defined at lu.c : 2362 */
	double u51jm1;		/* Defined at lu.c : 2362 */
	int jst1;		/* Defined at lu.c : 2354 */
	int jend1;		/* Defined at lu.c : 2354 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (j = jst; j <= jend; j++) {
			for (m = 0; m < 5; m++) {
				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    ty2 * (flux[m][i][j + 1][k] -
					   flux[m][i][j - 1][k]);
			}
		}
		L2 = ny - 1;
		for (j = jst; j <= L2; j++) {
			tmp = 1.0 / u[0][i][j][k];
			u21j = tmp * u[1][i][j][k];
			u31j = tmp * u[2][i][j][k];
			u41j = tmp * u[3][i][j][k];
			u51j = tmp * u[4][i][j][k];
			tmp = 1.0 / u[0][i][j - 1][k];
			u21jm1 = tmp * u[1][i][j - 1][k];
			u31jm1 = tmp * u[2][i][j - 1][k];
			u41jm1 = tmp * u[3][i][j - 1][k];
			u51jm1 = tmp * u[4][i][j - 1][k];
			flux[1][i][j][k] = ty3 * (u21j - u21jm1);
			flux[2][i][j][k] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
			flux[3][i][j][k] = ty3 * (u41j - u41jm1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * ty3 *
			    ((((u21j) * (u21j)) + ((u31j) * (u31j)) +
			      ((u41j) * (u41j))) - (((u21jm1) * (u21jm1)) +
						    ((u31jm1) * (u31jm1)) +
						    ((u41jm1) * (u41jm1)))) +
			    (1.0 / 6.0) * ty3 * (((u31j) * (u31j)) -
						 ((u31jm1) * (u31jm1))) +
			    1.40e+00 * 1.40e+00 * ty3 * (u51j - u51jm1);
		}
		for (j = jst; j <= jend; j++) {
			rsd[0][i][j][k] =
			    rsd[0][i][j][k] + dy1 * ty1 * (u[0][i][j - 1][k] -
							   2.0 * u[0][i][j][k] +
							   u[0][i][j + 1][k]);
			rsd[1][i][j][k] =
			    rsd[1][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[1][i][j + 1][k] -
							 flux[1][i][j][k]) +
			    dy2 * ty1 * (u[1][i][j - 1][k] -
					 2.0 * u[1][i][j][k] + u[1][i][j +
								       1][k]);
			rsd[2][i][j][k] =
			    rsd[2][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[2][i][j + 1][k] -
							 flux[2][i][j][k]) +
			    dy3 * ty1 * (u[2][i][j - 1][k] -
					 2.0 * u[2][i][j][k] + u[2][i][j +
								       1][k]);
			rsd[3][i][j][k] =
			    rsd[3][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[3][i][j + 1][k] -
							 flux[3][i][j][k]) +
			    dy4 * ty1 * (u[3][i][j - 1][k] -
					 2.0 * u[3][i][j][k] + u[3][i][j +
								       1][k]);
			rsd[4][i][j][k] =
			    rsd[4][i][j][k] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[4][i][j + 1][k] -
							 flux[4][i][j][k]) +
			    dy5 * ty1 * (u[4][i][j - 1][k] -
					 2.0 * u[4][i][j][k] + u[4][i][j +
								       1][k]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2612
			//-------------------------------------------
			double u_6[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2612
			//Candidates:
			//      u[m][i][1][k]
			//      u[m][i][2][k]
			//      u[m][i][3][k]
			//-------------------------------------------
			u_6[0] = u[m][i][1][k];
			u_6[1] = u[m][i][2][k];
			u_6[2] = u[m][i][3][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][i][1][k] =
			    rsd[m][i][1][k] -
			    dssp * (+5.0 * u_6[0] /*u[m][i][1][k] */ -4.0 *
				    u_6[1] /*u[m][i][2][k] */ +u_6[2]
				    /*u[m][i][3][k] */ );
			rsd[m][i][2][k] =
			    rsd[m][i][2][k] -
			    dssp * (-4.0 * u_6[0] /*u[m][i][1][k] */ +6.0 *
				    u_6[1] /*u[m][i][2][k] */ -4.0 *
				    u_6[2] /*u[m][i][3][k] */ +u[m][i][4][k]);
		}
		jst1 = 3;
		jend1 = ny - 4;
		for (j = jst1; j <= jend1; j++) {
			for (m = 0; m < 5; m++) {
				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    dssp * (u[m][i][j - 2][k] -
					    4.0 * u[m][i][j - 1][k] +
					    6.0 * u[m][i][j][k] -
					    4.0 * u[m][i][j + 1][k] +
					    u[m][i][j + 2][k]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2637
			//-------------------------------------------
			double u_7[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2637
			//Candidates:
			//      u[m][i][ny - 4][k]
			//      u[m][i][ny - 3][k]
			//-------------------------------------------
			u_7[0] = u[m][i][ny - 4][k];
			u_7[1] = u[m][i][ny - 3][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][i][ny - 3][k] =
			    rsd[m][i][ny - 3][k] - dssp * (u[m][i][ny - 5][k] -
							   4.0 *
							   u_7[0]
							   /*u[m][i][ny - 4][k] */
							   +6.0 *
							   u_7[1]
							   /*u[m][i][ny - 3][k] */
							   -4.0 * u[m][i][ny -
									  2]
							   [k]);
			rsd[m][i][ny - 2][k] =
			    rsd[m][i][ny - 2][k] -
			    dssp * (u_7[0] /*u[m][i][ny - 4][k] */ -4.0 *
				    u_7[1] /*u[m][i][ny - 3][k] */ +5.0 *
				    u[m][i][ny - 2][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2655 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void rhs_5(int k, int nz, __global double *g_flux,
		    __global double *g_u, int m, __global double *g_rsd,
		    double tz2, double tz3, double dz1, double tz1, double dz2,
		    double dz3, double dz4, double dz5, double dssp, int jst,
		    int ist, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + jst;
	int i = get_global_id(1) + ist;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double u41;		/* Defined at lu.c : 2356 */
	double q;		/* Defined at lu.c : 2355 */
	double tmp;		/* Defined at lu.c : 2357 */
	double u21k;		/* Defined at lu.c : 2360 */
	double u31k;		/* Defined at lu.c : 2360 */
	double u41k;		/* Defined at lu.c : 2360 */
	double u51k;		/* Defined at lu.c : 2360 */
	double u21km1;		/* Defined at lu.c : 2363 */
	double u31km1;		/* Defined at lu.c : 2363 */
	double u41km1;		/* Defined at lu.c : 2363 */
	double u51km1;		/* Defined at lu.c : 2363 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[33][33][33] =
	    (__global double (*)[33][33][33])g_flux;
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2657
			//-------------------------------------------
			double u_8[4];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2657
			//Candidates:
			//      u[0][i][j][k]
			//      u[1][i][j][k]
			//      u[2][i][j][k]
			//      u[3][i][j][k]
			//-------------------------------------------
			u_8[0] = u[0][i][j][k];
			u_8[1] = u[1][i][j][k];
			u_8[2] = u[2][i][j][k];
			u_8[3] = u[3][i][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			flux[0][i][j][k] = u_8[3] /*u[3][i][j][k] */ ;
			u41 =
			    u_8[3] /*u[3][i][j][k] */ /u_8[0] /*u[0][i][j][k] */
			    ;
			q = 0.50 *
			    (u_8[1] /*u[1][i][j][k] */ *u_8[1]
			     /*u[1][i][j][k] */ +u_8[2] /*u[2][i][j][k] */
			     *u_8[2] /*u[2][i][j][k] */ +u_8[3]
			     /*u[3][i][j][k] */ *u_8[3] /*u[3][i][j][k] */ ) /
			    u_8[0] /*u[0][i][j][k] */ ;
			flux[1][i][j][k] = u_8[1] /*u[1][i][j][k] */ *u41;
			flux[2][i][j][k] = u_8[2] /*u[2][i][j][k] */ *u41;
			flux[3][i][j][k] =
			    u_8[3] /*u[3][i][j][k] */ *u41 +
			    0.40e+00 * (u[4][i][j][k] - q);
			flux[4][i][j][k] =
			    (1.40e+00 * u[4][i][j][k] - 0.40e+00 * q) * u41;
		}
		for (k = 1; k <= nz - 2; k++) {
			for (m = 0; m < 5; m++) {
				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    tz2 * (flux[m][i][j][k + 1] -
					   flux[m][i][j][k - 1]);
			}
		}
		for (k = 1; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2679
			//-------------------------------------------
			double2 u_9[5];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2679
			//Candidates:
			//      u[0][i][j][k - 1]
			//      u[0][i][j][k]
			//      u[1][i][j][k - 1]
			//      u[1][i][j][k]
			//      u[2][i][j][k - 1]
			//      u[2][i][j][k]
			//      u[3][i][j][k - 1]
			//      u[3][i][j][k]
			//      u[4][i][j][k - 1]
			//      u[4][i][j][k]
			//-------------------------------------------
			__global double *p_u_9_0 =
			    (__global double *)&u[0][i][j][k - 1];
			if ((unsigned long)p_u_9_0 % 64 == 0) {
				u_9[0] = vload2(0, p_u_9_0);
			} else {
				u_9[0].x = p_u_9_0[0];
				p_u_9_0++;
				u_9[0].y = p_u_9_0[0];
				p_u_9_0++;
			}
			__global double *p_u_9_1 =
			    (__global double *)&u[1][i][j][k - 1];
			if ((unsigned long)p_u_9_1 % 64 == 0) {
				u_9[1] = vload2(0, p_u_9_1);
			} else {
				u_9[1].x = p_u_9_1[0];
				p_u_9_1++;
				u_9[1].y = p_u_9_1[0];
				p_u_9_1++;
			}
			__global double *p_u_9_2 =
			    (__global double *)&u[2][i][j][k - 1];
			if ((unsigned long)p_u_9_2 % 64 == 0) {
				u_9[2] = vload2(0, p_u_9_2);
			} else {
				u_9[2].x = p_u_9_2[0];
				p_u_9_2++;
				u_9[2].y = p_u_9_2[0];
				p_u_9_2++;
			}
			__global double *p_u_9_3 =
			    (__global double *)&u[3][i][j][k - 1];
			if ((unsigned long)p_u_9_3 % 64 == 0) {
				u_9[3] = vload2(0, p_u_9_3);
			} else {
				u_9[3].x = p_u_9_3[0];
				p_u_9_3++;
				u_9[3].y = p_u_9_3[0];
				p_u_9_3++;
			}
			__global double *p_u_9_4 =
			    (__global double *)&u[4][i][j][k - 1];
			if ((unsigned long)p_u_9_4 % 64 == 0) {
				u_9[4] = vload2(0, p_u_9_4);
			} else {
				u_9[4].x = p_u_9_4[0];
				p_u_9_4++;
				u_9[4].y = p_u_9_4[0];
				p_u_9_4++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmp = 1.0 / u_9[0].y /*u[0][i][j][k] */ ;
			u21k = tmp * u_9[1].y /*u[1][i][j][k] */ ;
			u31k = tmp * u_9[2].y /*u[2][i][j][k] */ ;
			u41k = tmp * u_9[3].y /*u[3][i][j][k] */ ;
			u51k = tmp * u_9[4].y /*u[4][i][j][k] */ ;
			tmp = 1.0 / u_9[0].x /*u[0][i][j][k - 1] */ ;
			u21km1 = tmp * u_9[1].x /*u[1][i][j][k - 1] */ ;
			u31km1 = tmp * u_9[2].x /*u[2][i][j][k - 1] */ ;
			u41km1 = tmp * u_9[3].x /*u[3][i][j][k - 1] */ ;
			u51km1 = tmp * u_9[4].x /*u[4][i][j][k - 1] */ ;
			flux[1][i][j][k] = tz3 * (u21k - u21km1);
			flux[2][i][j][k] = tz3 * (u31k - u31km1);
			flux[3][i][j][k] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
			flux[4][i][j][k] =
			    0.50 * (1.0 -
				    1.40e+00 * 1.40e+00) * tz3 *
			    ((((u21k) * (u21k)) + ((u31k) * (u31k)) +
			      ((u41k) * (u41k))) - (((u21km1) * (u21km1)) +
						    ((u31km1) * (u31km1)) +
						    ((u41km1) * (u41km1)))) +
			    (1.0 / 6.0) * tz3 * (((u41k) * (u41k)) -
						 ((u41km1) * (u41km1))) +
			    1.40e+00 * 1.40e+00 * tz3 * (u51k - u51km1);
		}
		for (k = 1; k <= nz - 2; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2705
			//-------------------------------------------
			double2 u_10[5];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2705
			//Candidates:
			//      u[0][i][j][k - 1]
			//      u[0][i][j][k]
			//      u[1][i][j][k - 1]
			//      u[1][i][j][k]
			//      u[2][i][j][k - 1]
			//      u[2][i][j][k]
			//      u[3][i][j][k - 1]
			//      u[3][i][j][k]
			//      u[4][i][j][k - 1]
			//      u[4][i][j][k]
			//-------------------------------------------
			__global double *p_u_10_0 =
			    (__global double *)&u[0][i][j][k - 1];
			if ((unsigned long)p_u_10_0 % 64 == 0) {
				u_10[0] = vload2(0, p_u_10_0);
			} else {
				u_10[0].x = p_u_10_0[0];
				p_u_10_0++;
				u_10[0].y = p_u_10_0[0];
				p_u_10_0++;
			}
			__global double *p_u_10_1 =
			    (__global double *)&u[1][i][j][k - 1];
			if ((unsigned long)p_u_10_1 % 64 == 0) {
				u_10[1] = vload2(0, p_u_10_1);
			} else {
				u_10[1].x = p_u_10_1[0];
				p_u_10_1++;
				u_10[1].y = p_u_10_1[0];
				p_u_10_1++;
			}
			__global double *p_u_10_2 =
			    (__global double *)&u[2][i][j][k - 1];
			if ((unsigned long)p_u_10_2 % 64 == 0) {
				u_10[2] = vload2(0, p_u_10_2);
			} else {
				u_10[2].x = p_u_10_2[0];
				p_u_10_2++;
				u_10[2].y = p_u_10_2[0];
				p_u_10_2++;
			}
			__global double *p_u_10_3 =
			    (__global double *)&u[3][i][j][k - 1];
			if ((unsigned long)p_u_10_3 % 64 == 0) {
				u_10[3] = vload2(0, p_u_10_3);
			} else {
				u_10[3].x = p_u_10_3[0];
				p_u_10_3++;
				u_10[3].y = p_u_10_3[0];
				p_u_10_3++;
			}
			__global double *p_u_10_4 =
			    (__global double *)&u[4][i][j][k - 1];
			if ((unsigned long)p_u_10_4 % 64 == 0) {
				u_10[4] = vload2(0, p_u_10_4);
			} else {
				u_10[4].x = p_u_10_4[0];
				p_u_10_4++;
				u_10[4].y = p_u_10_4[0];
				p_u_10_4++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[0][i][j][k] =
			    rsd[0][i][j][k] +
			    dz1 * tz1 * (u_10[0].x /*u[0][i][j][k - 1] */  -
					 2.0 * u_10[0].y /*u[0][i][j][k] */  +
					 u[0][i][j][k + 1]);
			rsd[1][i][j][k] =
			    rsd[1][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[1][i][j][k + 1] -
							 flux[1][i][j][k]) +
			    dz2 * tz1 * (u_10[1].x /*u[1][i][j][k - 1] */  -
					 2.0 * u_10[1].y /*u[1][i][j][k] */  +
					 u[1][i][j][k + 1]);
			rsd[2][i][j][k] =
			    rsd[2][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[2][i][j][k + 1] -
							 flux[2][i][j][k]) +
			    dz3 * tz1 * (u_10[2].x /*u[2][i][j][k - 1] */  -
					 2.0 * u_10[2].y /*u[2][i][j][k] */  +
					 u[2][i][j][k + 1]);
			rsd[3][i][j][k] =
			    rsd[3][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[3][i][j][k + 1] -
							 flux[3][i][j][k]) +
			    dz4 * tz1 * (u_10[3].x /*u[3][i][j][k - 1] */  -
					 2.0 * u_10[3].y /*u[3][i][j][k] */  +
					 u[3][i][j][k + 1]);
			rsd[4][i][j][k] =
			    rsd[4][i][j][k] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[4][i][j][k + 1] -
							 flux[4][i][j][k]) +
			    dz5 * tz1 * (u_10[4].x /*u[4][i][j][k - 1] */  -
					 2.0 * u_10[4].y /*u[4][i][j][k] */  +
					 u[4][i][j][k + 1]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2735
			//-------------------------------------------
			double4 u_11;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2735
			//Candidates:
			//      u[m][i][j][1]
			//      u[m][i][j][2]
			//      u[m][i][j][3]
			//      u[m][i][j][4]
			//-------------------------------------------
			__global double *p_u_11_0 =
			    (__global double *)&u[m][i][j][1];
			if ((unsigned long)p_u_11_0 % 64 == 0) {
				u_11 = vload4(0, p_u_11_0);
			} else {
				u_11.x = p_u_11_0[0];
				p_u_11_0++;
				u_11.y = p_u_11_0[0];
				p_u_11_0++;
				u_11.z = p_u_11_0[0];
				p_u_11_0++;
				u_11.w = p_u_11_0[0];
				p_u_11_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][i][j][1] =
			    rsd[m][i][j][1] -
			    dssp * (+5.0 * u_11.x /*u[m][i][j][1] */  -
				    4.0 * u_11.y /*u[m][i][j][2] */  +
				    u_11.z /*u[m][i][j][3] */ );
			rsd[m][i][j][2] =
			    rsd[m][i][j][2] -
			    dssp * (-4.0 * u_11.x /*u[m][i][j][1] */  +
				    6.0 * u_11.y /*u[m][i][j][2] */  -
				    4.0 * u_11.z /*u[m][i][j][3] */  +
				    u_11.w /*u[m][i][j][4] */ );
		}
		for (k = 3; k <= nz - 4; k++) {
			for (m = 0; m < 5; m++) {
				//-------------------------------------------
				//Declare prefetching Buffers (BEGIN) : 2748
				//-------------------------------------------
				double4 u_12;
				//-------------------------------------------
				//Declare prefetching buffers (END)
				//-------------------------------------------
				//-------------------------------------------
				//Prefetching (BEGIN) : 2748
				//Candidates:
				//      u[m][i][j][k - 2]
				//      u[m][i][j][k - 1]
				//      u[m][i][j][k]
				//      u[m][i][j][k + 1]
				//-------------------------------------------
				__global double *p_u_12_0 =
				    (__global double *)&u[m][i][j][k - 2];
				if ((unsigned long)p_u_12_0 % 64 == 0) {
					u_12 = vload4(0, p_u_12_0);
				} else {
					u_12.x = p_u_12_0[0];
					p_u_12_0++;
					u_12.y = p_u_12_0[0];
					p_u_12_0++;
					u_12.z = p_u_12_0[0];
					p_u_12_0++;
					u_12.w = p_u_12_0[0];
					p_u_12_0++;
				}
				//-------------------------------------------
				//Prefetching (END)
				//-------------------------------------------

				rsd[m][i][j][k] =
				    rsd[m][i][j][k] -
				    dssp * (u_12.x /*u[m][i][j][k - 2] */  -
					    4.0 *
					    u_12.y /*u[m][i][j][k - 1] */  +
					    6.0 * u_12.z /*u[m][i][j][k] */  -
					    4.0 *
					    u_12.w /*u[m][i][j][k + 1] */  +
					    u[m][i][j][k + 2]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2758
			//-------------------------------------------
			double4 u_13;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2758
			//Candidates:
			//      u[m][i][j][nz - 5]
			//      u[m][i][j][nz - 4]
			//      u[m][i][j][nz - 3]
			//      u[m][i][j][nz - 2]
			//-------------------------------------------
			__global double *p_u_13_0 =
			    (__global double *)&u[m][i][j][nz - 5];
			if ((unsigned long)p_u_13_0 % 64 == 0) {
				u_13 = vload4(0, p_u_13_0);
			} else {
				u_13.x = p_u_13_0[0];
				p_u_13_0++;
				u_13.y = p_u_13_0[0];
				p_u_13_0++;
				u_13.z = p_u_13_0[0];
				p_u_13_0++;
				u_13.w = p_u_13_0[0];
				p_u_13_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[m][i][j][nz - 3] =
			    rsd[m][i][j][nz - 3] -
			    dssp * (u_13.x /*u[m][i][j][nz - 5] */  -
				    4.0 * u_13.y /*u[m][i][j][nz - 4] */  +
				    6.0 * u_13.z /*u[m][i][j][nz - 3] */  -
				    4.0 * u_13.w /*u[m][i][j][nz - 2] */ );
			rsd[m][i][j][nz - 2] =
			    rsd[m][i][j][nz - 2] -
			    dssp * (u_13.y /*u[m][i][j][nz - 4] */  -
				    4.0 * u_13.z /*u[m][i][j][nz - 3] */  +
				    5.0 * u_13.w /*u[m][i][j][nz - 2] */ );
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2793 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_0(__global double *g_u, int nz, __global double *g_ce,
		      int nx0, int ny0, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0);
	int i = get_global_id(1);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp[5];		/* Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6(i, j, 0, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][i][j][0] = tmp[0];
		u[1][i][j][0] = tmp[1];
		u[2][i][j][0] = tmp[2];
		u[3][i][j][0] = tmp[3];
		u[4][i][j][0] = tmp[4];
		exact_g7_e4_e5_e6(i, j, nz - 1, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][i][j][nz - 1] = tmp[0];
		u[1][i][j][nz - 1] = tmp[1];
		u[2][i][j][nz - 1] = tmp[2];
		u[3][i][j][nz - 1] = tmp[3];
		u[4][i][j][nz - 1] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2814 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_1(__global double *g_u, __global double *g_ce, int nx0,
		      int ny0, int nz, int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int i = get_global_id(1);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp[5];		/* Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6(i, 0, k, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][i][0][k] = tmp[0];
		u[1][i][0][k] = tmp[1];
		u[2][i][0][k] = tmp[2];
		u[3][i][0][k] = tmp[3];
		u[4][i][0][k] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2826 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_2(int ny0, __global double *g_u, int ny,
		      __global double *g_ce, int nx0, int nz, int __ocl_k_bound,
		      int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int i = get_global_id(1);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp[5];		/* Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6(i, ny0 - 1, k, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][i][ny - 1][k] = tmp[0];
		u[1][i][ny - 1][k] = tmp[1];
		u[2][i][ny - 1][k] = tmp[2];
		u[3][i][ny - 1][k] = tmp[3];
		u[4][i][ny - 1][k] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2841 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_3(__global double *g_u, __global double *g_ce, int nx0,
		      int ny0, int nz, int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp[5];		/* Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6(0, j, k, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][0][j][k] = tmp[0];
		u[1][0][j][k] = tmp[1];
		u[2][0][j][k] = tmp[2];
		u[3][0][j][k] = tmp[3];
		u[4][0][j][k] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2854 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_4(int nx0, __global double *g_u, int nx,
		      __global double *g_ce, int ny0, int nz, int __ocl_k_bound,
		      int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double tmp[5];		/* Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6(nx0 - 1, j, k, tmp, nx0, ny0, nz,
				  ce) /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][nx - 1][j][k] = tmp[0];
		u[1][nx - 1][j][k] = tmp[1];
		u[2][nx - 1][j][k] = tmp[2];
		u[3][nx - 1][j][k] = tmp[3];
		u[4][nx - 1][j][k] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3034 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setiv_0(int nz, int ny0, int i, int nx, int nx0, int m,
		      __global double *g_u, __global double *g_ce,
		      int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int jglob;		/* Defined at lu.c : 3027 */
	double zeta;		/* Defined at lu.c : 3028 */
	double eta;		/* Defined at lu.c : 3028 */
	int iglob;		/* Defined at lu.c : 3027 */
	double xi;		/* Defined at lu.c : 3028 */
	double ue_1jk[5];	/* Defined at lu.c : 3030 */
	double ue_nx0jk[5];	/* Defined at lu.c : 3030 */
	double ue_i1k[5];	/* Defined at lu.c : 3030 */
	double ue_iny0k[5];	/* Defined at lu.c : 3031 */
	double ue_ij1[5];	/* Defined at lu.c : 3031 */
	double ue_ijnz[5];	/* Defined at lu.c : 3031 */
	double pxi;		/* Defined at lu.c : 3029 */
	double peta;		/* Defined at lu.c : 3029 */
	double pzeta;		/* Defined at lu.c : 3029 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		jglob = j;
		zeta = ((double)k) / (nz - 1);
		if (jglob != 0 && jglob != ny0 - 1) {
			eta = ((double)(jglob)) / (ny0 - 1);
			for (i = 0; i < nx; i++) {
				iglob = i;
				if (iglob != 0 && iglob != nx0 - 1) {
					xi = ((double)(iglob)) / (nx0 - 1);
					exact_g7_e4_e5_e6(0, jglob, k, ue_1jk,
							  nx0, ny0, nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					exact_g7_e4_e5_e6(nx0 - 1, jglob, k,
							  ue_nx0jk, nx0, ny0,
							  nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					exact_g7_e4_e5_e6(iglob, 0, k, ue_i1k,
							  nx0, ny0, nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					exact_g7_e4_e5_e6(iglob, ny0 - 1, k,
							  ue_iny0k, nx0, ny0,
							  nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					exact_g7_e4_e5_e6(iglob, jglob, 0,
							  ue_ij1, nx0, ny0, nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					exact_g7_e4_e5_e6(iglob, jglob, nz - 1,
							  ue_ijnz, nx0, ny0, nz,
							  ce)
					    /*ARGEXP: nx0,ny0,nz,ce */ ;
					for (m = 0; m < 5; m++) {
						pxi =
						    (1.0 - xi) * ue_1jk[m] +
						    xi * ue_nx0jk[m];
						peta =
						    (1.0 - eta) * ue_i1k[m] +
						    eta * ue_iny0k[m];
						pzeta =
						    (1.0 - zeta) * ue_ij1[m] +
						    zeta * ue_ijnz[m];
						u[m][i][j][k] =
						    pxi + peta + pzeta -
						    pxi * peta - peta * pzeta -
						    pzeta * pxi +
						    pxi * peta * pzeta;
					}
				}
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3098 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void ssor_0(__global double *g_a, int k, __global double *g_b,
		     __global double *g_c, __global double *g_d)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0);
	int i = get_global_id(1);
	int m = get_global_id(2);
	if (!(j < 33)) {
		return;
	}
	if (!(i < 33)) {
		return;
	}
	if (!(m < 5)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*a)[5][33][33] = (__global double (*)[5][33][33])g_a;
	__global double (*b)[5][33][33] = (__global double (*)[5][33][33])g_b;
	__global double (*c)[5][33][33] = (__global double (*)[5][33][33])g_c;
	__global double (*d)[5][33][33] = (__global double (*)[5][33][33])g_d;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (k = 0; k < 5; k++) {
		a[k][m][i][j] = 0.0;
		b[k][m][i][j] = 0.0;
		c[k][m][i][j] = 0.0;
		d[k][m][i][j] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3142 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void ssor_1(__global double *g_rsd, int m, double dt, int jst, int ist,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rsd[m][i][j][k] = dt * rsd[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3192 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void ssor_2(__global double *g_u, int m, double tmp,
		     __global double *g_rsd, int jst, int ist,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + ist;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[33][33][33] = (__global double (*)[33][33][33])g_u;
	__global double (*rsd)[33][33][33] =
	    (__global double (*)[33][33][33])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		u[m][i][j][k] = u[m][i][j][k] + tmp * rsd[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
