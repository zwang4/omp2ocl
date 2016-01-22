//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Thu Oct 25 14:33:04 2012
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
//      Strict TLS Checking     false
//      Check TLS Conflict at the end of function       true
//      Use OCL TLS     true
//-------------------------------------------------------------------------------

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
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
//TLS Checking Routines (BEGIN)
//-------------------------------------------------------------------------------
int calc_thread_id_1()
{
	return get_global_id(0);
}

int calc_thread_id_2()
{
	return (get_global_id(1) * get_global_size(0) + get_global_id(0));
}

int calc_thread_id_3()
{
	return (get_global_id(2) * (get_global_size(1) * get_global_size(0)) +
		(get_global_id(1) * get_global_size(0) + get_global_id(0)));
}

double spec_read_double(__global double *a, __global int *wr_log,
			__global int *read_log, int thread_id,
			__global int *invalid)
{
	double value;
	atom_max((__global int *)read_log, thread_id);
	value = a[0];

	if (wr_log[0] > thread_id) {
		*invalid = 1;
	}
	return value;
}

double spec_write_double(__global double *a, __global int *wr_log,
			 __global int *read_log, int thread_id,
			 __global int *invalid, double value)
{
	if (atom_max((__global int *)wr_log, thread_id) > thread_id) {
		*invalid = 1;
	}

	a[0] = value;
	if (read_log[0] > thread_id) {
		*invalid = 1;
	}
	return value;
}

//-------------------------------------------------------------------------------
//TLS Checking Routines (END)
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//Functions (BEGIN)
//-------------------------------------------------------------------------------
void exact_g7_e4_e5_e6(int i, int j, int k, double u000ijk[5], int nx0, int ny0,
		       int nz, __global double (*ce)[13],
		       __global int *tls_validflag, int tls_thread_id);
void exact_g7_e4_e5_e6_no_spec(int i, int j, int k, double u000ijk[5], int nx0,
			       int ny0, int nz, __global double (*ce)[13],
			       __global int *tls_validflag, int tls_thread_id);

//-------------------------------------------------------------------------------
//This is an alias of function: exact
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_g7_e4_e5_e6(int i, int j, int k, double u000ijk[5], int nx0, int ny0,
		       int nz, __global double (*ce)[13],
		       __global int *tls_validflag, int tls_thread_id)
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
//This is an alias of function: exact
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_g7_e4_e5_e6_no_spec(int i, int j, int k, double u000ijk[5], int nx0,
			       int ny0, int nz, __global double (*ce)[13],
			       __global int *tls_validflag, int tls_thread_id)
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
//Loop defined at line 214 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void blts_0(__global double *g_v, int k, double omega,
		     __global double *g_ldz, int jst, int ist,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int m = get_global_id(0);
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + ist;
	if (!(m < 5)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*v)[65][65][5] = (__global double (*)[65][65][5])g_v;
	__global double (*ldz)[64][5][5] = (__global double (*)[64][5][5])g_ldz;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 216
		//-------------------------------------------
		double4 ldz_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 216
		//Candidates:
		//      ldz[i][j][m][0]
		//      ldz[i][j][m][1]
		//      ldz[i][j][m][2]
		//      ldz[i][j][m][3]
		//-------------------------------------------
		__global double *p_ldz_1_0 =
		    (__global double *)&ldz[i][j][m][0];
		if ((unsigned long)p_ldz_1_0 % 64 == 0) {
			ldz_1 = vload4(0, p_ldz_1_0);
		} else {
			ldz_1.x = p_ldz_1_0[0];
			p_ldz_1_0++;
			ldz_1.y = p_ldz_1_0[0];
			p_ldz_1_0++;
			ldz_1.z = p_ldz_1_0[0];
			p_ldz_1_0++;
			ldz_1.w = p_ldz_1_0[0];
			p_ldz_1_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		v[i][j][k][m] =
		    v[i][j][k][m] -
		    omega * (ldz_1.x /*ldz[i][j][m][0] */  * v[i][j][k - 1][0] +
			     ldz_1.y /*ldz[i][j][m][1] */  * v[i][j][k - 1][1] +
			     ldz_1.z /*ldz[i][j][m][2] */  * v[i][j][k - 1][2] +
			     ldz_1.w /*ldz[i][j][m][3] */  * v[i][j][k - 1][3] +
			     ldz[i][j][m][4] * v[i][j][k - 1][4]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 228 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void blts_1(int m, __global double *g_v, int k, double omega,
		     __global double *g_ldy, __global double *g_ldx,
		     __global double *g_d, int jst, int ist, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *tls_validflag)
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
	double tmat[5][5];	/* (User-defined privated variables) : Defined at lu.c : 211 */
	double tmp1;		/* (User-defined privated variables) : Defined at lu.c : 210 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 210 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*v)[65][65][5] = (__global double (*)[65][65][5])g_v;
	__global double (*ldy)[64][5][5] = (__global double (*)[64][5][5])g_ldy;
	__global double (*ldx)[64][5][5] = (__global double (*)[64][5][5])g_ldx;
	__global double (*d)[64][5][5] = (__global double (*)[64][5][5])g_d;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 246
			//-------------------------------------------
			double4 ldy_1;
			double4 ldx_1;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 246
			//Candidates:
			//      ldy[i][j][m][0]
			//      ldy[i][j][m][1]
			//      ldy[i][j][m][2]
			//      ldy[i][j][m][3]
			//      ldx[i][j][m][0]
			//      ldx[i][j][m][1]
			//      ldx[i][j][m][2]
			//      ldx[i][j][m][3]
			//-------------------------------------------
			__global double *p_ldy_1_0 =
			    (__global double *)&ldy[i][j][m][0];
			if ((unsigned long)p_ldy_1_0 % 64 == 0) {
				ldy_1 = vload4(0, p_ldy_1_0);
			} else {
				ldy_1.x = p_ldy_1_0[0];
				p_ldy_1_0++;
				ldy_1.y = p_ldy_1_0[0];
				p_ldy_1_0++;
				ldy_1.z = p_ldy_1_0[0];
				p_ldy_1_0++;
				ldy_1.w = p_ldy_1_0[0];
				p_ldy_1_0++;
			}
			__global double *p_ldx_1_0 =
			    (__global double *)&ldx[i][j][m][0];
			if ((unsigned long)p_ldx_1_0 % 64 == 0) {
				ldx_1 = vload4(0, p_ldx_1_0);
			} else {
				ldx_1.x = p_ldx_1_0[0];
				p_ldx_1_0++;
				ldx_1.y = p_ldx_1_0[0];
				p_ldx_1_0++;
				ldx_1.z = p_ldx_1_0[0];
				p_ldx_1_0++;
				ldx_1.w = p_ldx_1_0[0];
				p_ldx_1_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			v[i][j][k][m] =
			    v[i][j][k][m] -
			    omega * (ldy_1.x /*ldy[i][j][m][0] */  *
				     v[i][j - 1][k][0] +
				     ldx_1.x /*ldx[i][j][m][0] */  * v[i -
								       1][j][k]
				     [0] +
				     ldy_1.y /*ldy[i][j][m][1] */  * v[i][j -
									  1][k]
				     [1] + ldx_1.y /*ldx[i][j][m][1] */  * v[i -
									     1]
				     [j][k][1] +
				     ldy_1.z /*ldy[i][j][m][2] */  * v[i][j -
									  1][k]
				     [2] + ldx_1.z /*ldx[i][j][m][2] */  * v[i -
									     1]
				     [j][k][2] +
				     ldy_1.w /*ldy[i][j][m][3] */  * v[i][j -
									  1][k]
				     [3] + ldx_1.w /*ldx[i][j][m][3] */  * v[i -
									     1]
				     [j][k][3] + ldy[i][j][m][4] * v[i][j -
									1][k][4]
				     + ldx[i][j][m][4] * v[i - 1][j][k][4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 266
			//-------------------------------------------
			double4 d_1;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 266
			//Candidates:
			//      d[i][j][m][0]
			//      d[i][j][m][1]
			//      d[i][j][m][2]
			//      d[i][j][m][3]
			//-------------------------------------------
			__global double *p_d_1_0 =
			    (__global double *)&d[i][j][m][0];
			if ((unsigned long)p_d_1_0 % 64 == 0) {
				d_1 = vload4(0, p_d_1_0);
			} else {
				d_1.x = p_d_1_0[0];
				p_d_1_0++;
				d_1.y = p_d_1_0[0];
				p_d_1_0++;
				d_1.z = p_d_1_0[0];
				p_d_1_0++;
				d_1.w = p_d_1_0[0];
				p_d_1_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmat[m][0] = d_1.x /*d[i][j][m][0] */ ;
			tmat[m][1] = d_1.y /*d[i][j][m][1] */ ;
			tmat[m][2] = d_1.z /*d[i][j][m][2] */ ;
			tmat[m][3] = d_1.w /*d[i][j][m][3] */ ;
			tmat[m][4] = d[i][j][m][4];
		}
		tmp1 = 1.0 / tmat[0][0];
		tmp = tmp1 * tmat[1][0];
		tmat[1][1] = tmat[1][1] - tmp * tmat[0][1];
		tmat[1][2] = tmat[1][2] - tmp * tmat[0][2];
		tmat[1][3] = tmat[1][3] - tmp * tmat[0][3];
		tmat[1][4] = tmat[1][4] - tmp * tmat[0][4];
		v[i][j][k][1] = v[i][j][k][1] - v[i][j][k][0] * tmp;
		tmp = tmp1 * tmat[2][0];
		tmat[2][1] = tmat[2][1] - tmp * tmat[0][1];
		tmat[2][2] = tmat[2][2] - tmp * tmat[0][2];
		tmat[2][3] = tmat[2][3] - tmp * tmat[0][3];
		tmat[2][4] = tmat[2][4] - tmp * tmat[0][4];
		v[i][j][k][2] = v[i][j][k][2] - v[i][j][k][0] * tmp;
		tmp = tmp1 * tmat[3][0];
		tmat[3][1] = tmat[3][1] - tmp * tmat[0][1];
		tmat[3][2] = tmat[3][2] - tmp * tmat[0][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[0][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[0][4];
		v[i][j][k][3] = v[i][j][k][3] - v[i][j][k][0] * tmp;
		tmp = tmp1 * tmat[4][0];
		tmat[4][1] = tmat[4][1] - tmp * tmat[0][1];
		tmat[4][2] = tmat[4][2] - tmp * tmat[0][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[0][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[0][4];
		v[i][j][k][4] = v[i][j][k][4] - v[i][j][k][0] * tmp;
		tmp1 = 1.0 / tmat[1][1];
		tmp = tmp1 * tmat[2][1];
		tmat[2][2] = tmat[2][2] - tmp * tmat[1][2];
		tmat[2][3] = tmat[2][3] - tmp * tmat[1][3];
		tmat[2][4] = tmat[2][4] - tmp * tmat[1][4];
		v[i][j][k][2] = v[i][j][k][2] - v[i][j][k][1] * tmp;
		tmp = tmp1 * tmat[3][1];
		tmat[3][2] = tmat[3][2] - tmp * tmat[1][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[1][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[1][4];
		v[i][j][k][3] = v[i][j][k][3] - v[i][j][k][1] * tmp;
		tmp = tmp1 * tmat[4][1];
		tmat[4][2] = tmat[4][2] - tmp * tmat[1][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[1][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[1][4];
		v[i][j][k][4] = v[i][j][k][4] - v[i][j][k][1] * tmp;
		tmp1 = 1.0 / tmat[2][2];
		tmp = tmp1 * tmat[3][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[2][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[2][4];
		v[i][j][k][3] = v[i][j][k][3] - v[i][j][k][2] * tmp;
		tmp = tmp1 * tmat[4][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[2][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[2][4];
		v[i][j][k][4] = v[i][j][k][4] - v[i][j][k][2] * tmp;
		tmp1 = 1.0 / tmat[3][3];
		tmp = tmp1 * tmat[4][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[3][4];
		v[i][j][k][4] = v[i][j][k][4] - v[i][j][k][3] * tmp;
		v[i][j][k][4] = v[i][j][k][4] / tmat[4][4];
		v[i][j][k][3] = v[i][j][k][3] - tmat[3][4] * v[i][j][k][4];
		v[i][j][k][3] = v[i][j][k][3] / tmat[3][3];
		v[i][j][k][2] =
		    v[i][j][k][2] - tmat[2][3] * v[i][j][k][3] -
		    tmat[2][4] * v[i][j][k][4];
		v[i][j][k][2] = v[i][j][k][2] / tmat[2][2];
		v[i][j][k][1] =
		    v[i][j][k][1] - tmat[1][2] * v[i][j][k][2] -
		    tmat[1][3] * v[i][j][k][3] - tmat[1][4] * v[i][j][k][4];
		v[i][j][k][1] = v[i][j][k][1] / tmat[1][1];
		v[i][j][k][0] =
		    v[i][j][k][0] - tmat[0][1] * v[i][j][k][1] -
		    tmat[0][2] * v[i][j][k][2] - tmat[0][3] * v[i][j][k][3] -
		    tmat[0][4] * v[i][j][k][4];
		v[i][j][k][0] = v[i][j][k][0] / tmat[0][0];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 454 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void buts_0(__global double *g_tv, double omega,
		     __global double *g_udz, __global double *g_v, int k,
		     int jst, int ist, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int m = get_global_id(0);
	int j = get_global_id(1) + jst;
	int i = get_global_id(2) + ist;
	if (!(m < 5)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*tv)[64][5] = (__global double (*)[64][5])g_tv;
	__global double (*udz)[64][5][5] = (__global double (*)[64][5][5])g_udz;
	__global double (*v)[65][65][5] = (__global double (*)[65][65][5])g_v;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 457
		//-------------------------------------------
		double4 udz_1;
		double4 v_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 457
		//Candidates:
		//      udz[i][j][m][0]
		//      udz[i][j][m][1]
		//      udz[i][j][m][2]
		//      udz[i][j][m][3]
		//      v[i][j][k + 1][0]
		//      v[i][j][k + 1][1]
		//      v[i][j][k + 1][2]
		//      v[i][j][k + 1][3]
		//-------------------------------------------
		__global double *p_udz_1_0 =
		    (__global double *)&udz[i][j][m][0];
		if ((unsigned long)p_udz_1_0 % 64 == 0) {
			udz_1 = vload4(0, p_udz_1_0);
		} else {
			udz_1.x = p_udz_1_0[0];
			p_udz_1_0++;
			udz_1.y = p_udz_1_0[0];
			p_udz_1_0++;
			udz_1.z = p_udz_1_0[0];
			p_udz_1_0++;
			udz_1.w = p_udz_1_0[0];
			p_udz_1_0++;
		}
		__global double *p_v_1_0 =
		    (__global double *)&v[i][j][k + 1][0];
		if ((unsigned long)p_v_1_0 % 64 == 0) {
			v_1 = vload4(0, p_v_1_0);
		} else {
			v_1.x = p_v_1_0[0];
			p_v_1_0++;
			v_1.y = p_v_1_0[0];
			p_v_1_0++;
			v_1.z = p_v_1_0[0];
			p_v_1_0++;
			v_1.w = p_v_1_0[0];
			p_v_1_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tv[i][j][m] =
		    omega * (udz_1.x /*udz[i][j][m][0] */  *
			     v_1.x /*v[i][j][k + 1][0] */  +
			     udz_1.y /*udz[i][j][m][1] */  *
			     v_1.y /*v[i][j][k + 1][1] */  +
			     udz_1.z /*udz[i][j][m][2] */  *
			     v_1.z /*v[i][j][k + 1][2] */  +
			     udz_1.w /*udz[i][j][m][3] */  *
			     v_1.w /*v[i][j][k + 1][3] */  +
			     udz[i][j][m][4] * v[i][j][k + 1][4]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 470 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void buts_1(int m, __global double *g_tv, double omega,
		     __global double *g_udy, __global double *g_v, int k,
		     __global double *g_udx, __global double *g_d, int jst,
		     int ist, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	double tmat[5][5];	/* (User-defined privated variables) : Defined at lu.c : 449 */
	double tmp1;		/* (User-defined privated variables) : Defined at lu.c : 448 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 448 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*tv)[64][5] = (__global double (*)[64][5])g_tv;
	__global double (*udy)[64][5][5] = (__global double (*)[64][5][5])g_udy;
	__global double (*v)[65][65][5] = (__global double (*)[65][65][5])g_v;
	__global double (*udx)[64][5][5] = (__global double (*)[64][5][5])g_udx;
	__global double (*d)[64][5][5] = (__global double (*)[64][5][5])g_d;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 488
			//-------------------------------------------
			double4 udy_1;
			double4 udx_1;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 488
			//Candidates:
			//      udy[i][j][m][0]
			//      udy[i][j][m][1]
			//      udy[i][j][m][2]
			//      udy[i][j][m][3]
			//      udx[i][j][m][0]
			//      udx[i][j][m][1]
			//      udx[i][j][m][2]
			//      udx[i][j][m][3]
			//-------------------------------------------
			__global double *p_udy_1_0 =
			    (__global double *)&udy[i][j][m][0];
			if ((unsigned long)p_udy_1_0 % 64 == 0) {
				udy_1 = vload4(0, p_udy_1_0);
			} else {
				udy_1.x = p_udy_1_0[0];
				p_udy_1_0++;
				udy_1.y = p_udy_1_0[0];
				p_udy_1_0++;
				udy_1.z = p_udy_1_0[0];
				p_udy_1_0++;
				udy_1.w = p_udy_1_0[0];
				p_udy_1_0++;
			}
			__global double *p_udx_1_0 =
			    (__global double *)&udx[i][j][m][0];
			if ((unsigned long)p_udx_1_0 % 64 == 0) {
				udx_1 = vload4(0, p_udx_1_0);
			} else {
				udx_1.x = p_udx_1_0[0];
				p_udx_1_0++;
				udx_1.y = p_udx_1_0[0];
				p_udx_1_0++;
				udx_1.z = p_udx_1_0[0];
				p_udx_1_0++;
				udx_1.w = p_udx_1_0[0];
				p_udx_1_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tv[i][j][m] =
			    tv[i][j][m] +
			    omega * (udy_1.x /*udy[i][j][m][0] */  *
				     v[i][j + 1][k][0] +
				     udx_1.x /*udx[i][j][m][0] */  * v[i +
								       1][j][k]
				     [0] +
				     udy_1.y /*udy[i][j][m][1] */  * v[i][j +
									  1][k]
				     [1] + udx_1.y /*udx[i][j][m][1] */  * v[i +
									     1]
				     [j][k][1] +
				     udy_1.z /*udy[i][j][m][2] */  * v[i][j +
									  1][k]
				     [2] + udx_1.z /*udx[i][j][m][2] */  * v[i +
									     1]
				     [j][k][2] +
				     udy_1.w /*udy[i][j][m][3] */  * v[i][j +
									  1][k]
				     [3] + udx_1.w /*udx[i][j][m][3] */  * v[i +
									     1]
				     [j][k][3] + udy[i][j][m][4] * v[i][j +
									1][k][4]
				     + udx[i][j][m][4] * v[i + 1][j][k][4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 505
			//-------------------------------------------
			double4 d_3;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 505
			//Candidates:
			//      d[i][j][m][0]
			//      d[i][j][m][1]
			//      d[i][j][m][2]
			//      d[i][j][m][3]
			//-------------------------------------------
			__global double *p_d_3_0 =
			    (__global double *)&d[i][j][m][0];
			if ((unsigned long)p_d_3_0 % 64 == 0) {
				d_3 = vload4(0, p_d_3_0);
			} else {
				d_3.x = p_d_3_0[0];
				p_d_3_0++;
				d_3.y = p_d_3_0[0];
				p_d_3_0++;
				d_3.z = p_d_3_0[0];
				p_d_3_0++;
				d_3.w = p_d_3_0[0];
				p_d_3_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmat[m][0] = d_3.x /*d[i][j][m][0] */ ;
			tmat[m][1] = d_3.y /*d[i][j][m][1] */ ;
			tmat[m][2] = d_3.z /*d[i][j][m][2] */ ;
			tmat[m][3] = d_3.w /*d[i][j][m][3] */ ;
			tmat[m][4] = d[i][j][m][4];
		}
		tmp1 = 1.0 / tmat[0][0];
		tmp = tmp1 * tmat[1][0];
		tmat[1][1] = tmat[1][1] - tmp * tmat[0][1];
		tmat[1][2] = tmat[1][2] - tmp * tmat[0][2];
		tmat[1][3] = tmat[1][3] - tmp * tmat[0][3];
		tmat[1][4] = tmat[1][4] - tmp * tmat[0][4];
		tv[i][j][1] = tv[i][j][1] - tv[i][j][0] * tmp;
		tmp = tmp1 * tmat[2][0];
		tmat[2][1] = tmat[2][1] - tmp * tmat[0][1];
		tmat[2][2] = tmat[2][2] - tmp * tmat[0][2];
		tmat[2][3] = tmat[2][3] - tmp * tmat[0][3];
		tmat[2][4] = tmat[2][4] - tmp * tmat[0][4];
		tv[i][j][2] = tv[i][j][2] - tv[i][j][0] * tmp;
		tmp = tmp1 * tmat[3][0];
		tmat[3][1] = tmat[3][1] - tmp * tmat[0][1];
		tmat[3][2] = tmat[3][2] - tmp * tmat[0][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[0][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[0][4];
		tv[i][j][3] = tv[i][j][3] - tv[i][j][0] * tmp;
		tmp = tmp1 * tmat[4][0];
		tmat[4][1] = tmat[4][1] - tmp * tmat[0][1];
		tmat[4][2] = tmat[4][2] - tmp * tmat[0][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[0][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[0][4];
		tv[i][j][4] = tv[i][j][4] - tv[i][j][0] * tmp;
		tmp1 = 1.0 / tmat[1][1];
		tmp = tmp1 * tmat[2][1];
		tmat[2][2] = tmat[2][2] - tmp * tmat[1][2];
		tmat[2][3] = tmat[2][3] - tmp * tmat[1][3];
		tmat[2][4] = tmat[2][4] - tmp * tmat[1][4];
		tv[i][j][2] = tv[i][j][2] - tv[i][j][1] * tmp;
		tmp = tmp1 * tmat[3][1];
		tmat[3][2] = tmat[3][2] - tmp * tmat[1][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[1][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[1][4];
		tv[i][j][3] = tv[i][j][3] - tv[i][j][1] * tmp;
		tmp = tmp1 * tmat[4][1];
		tmat[4][2] = tmat[4][2] - tmp * tmat[1][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[1][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[1][4];
		tv[i][j][4] = tv[i][j][4] - tv[i][j][1] * tmp;
		tmp1 = 1.0 / tmat[2][2];
		tmp = tmp1 * tmat[3][2];
		tmat[3][3] = tmat[3][3] - tmp * tmat[2][3];
		tmat[3][4] = tmat[3][4] - tmp * tmat[2][4];
		tv[i][j][3] = tv[i][j][3] - tv[i][j][2] * tmp;
		tmp = tmp1 * tmat[4][2];
		tmat[4][3] = tmat[4][3] - tmp * tmat[2][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[2][4];
		tv[i][j][4] = tv[i][j][4] - tv[i][j][2] * tmp;
		tmp1 = 1.0 / tmat[3][3];
		tmp = tmp1 * tmat[4][3];
		tmat[4][4] = tmat[4][4] - tmp * tmat[3][4];
		tv[i][j][4] = tv[i][j][4] - tv[i][j][3] * tmp;
		tv[i][j][4] = tv[i][j][4] / tmat[4][4];
		tv[i][j][3] = tv[i][j][3] - tmat[3][4] * tv[i][j][4];
		tv[i][j][3] = tv[i][j][3] / tmat[3][3];
		tv[i][j][2] =
		    tv[i][j][2] - tmat[2][3] * tv[i][j][3] -
		    tmat[2][4] * tv[i][j][4];
		tv[i][j][2] = tv[i][j][2] / tmat[2][2];
		tv[i][j][1] =
		    tv[i][j][1] - tmat[1][2] * tv[i][j][2] -
		    tmat[1][3] * tv[i][j][3] - tmat[1][4] * tv[i][j][4];
		tv[i][j][1] = tv[i][j][1] / tmat[1][1];
		tv[i][j][0] =
		    tv[i][j][0] - tmat[0][1] * tv[i][j][1] -
		    tmat[0][2] * tv[i][j][2] - tmat[0][3] * tv[i][j][3] -
		    tmat[0][4] * tv[i][j][4];
		tv[i][j][0] = tv[i][j][0] / tmat[0][0];
		v[i][j][k][0] = v[i][j][k][0] - tv[i][j][0];
		v[i][j][k][1] = v[i][j][k][1] - tv[i][j][1];
		v[i][j][k][2] = v[i][j][k][2] - tv[i][j][2];
		v[i][j][k][3] = v[i][j][k][3] - tv[i][j][3];
		v[i][j][k][4] = v[i][j][k][4] - tv[i][j][4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 741 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_0(__global double *g_frct, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	int m;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[65][65][5] =
	    (__global double (*)[65][65][5])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			frct[i][j][k][m] = 0.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 752 of lu.c
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_1(int nx0, int ny, int ny0, int nz, __global double *g_rsd,
		     __global double *g_ce, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	int iglob;		/* (User-defined privated variables) : Defined at lu.c : 722 */
	double xi;		/* (User-defined privated variables) : Defined at lu.c : 727 */
	int j;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	int jglob;		/* (User-defined privated variables) : Defined at lu.c : 722 */
	double eta;		/* (User-defined privated variables) : Defined at lu.c : 727 */
	int k;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	double zeta;		/* (User-defined privated variables) : Defined at lu.c : 727 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_1();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
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
					//Declare prefetching Buffers (BEGIN) : 760
					//-------------------------------------------
					double4 ce_1[3];
					//-------------------------------------------
					//Declare prefetching buffers (END)
					//-------------------------------------------
					//-------------------------------------------
					//Prefetching (BEGIN) : 760
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
					__global double *p_ce_1_0 =
					    (__global double *)&ce[m][0];
					if ((unsigned long)p_ce_1_0 % 64 == 0) {
						ce_1[0] = vload4(0, p_ce_1_0);
					} else {
						ce_1[0].x = p_ce_1_0[0];
						p_ce_1_0++;
						ce_1[0].y = p_ce_1_0[0];
						p_ce_1_0++;
						ce_1[0].z = p_ce_1_0[0];
						p_ce_1_0++;
						ce_1[0].w = p_ce_1_0[0];
						p_ce_1_0++;
					}
					__global double *p_ce_1_1 =
					    (__global double *)&ce[m][4];
					if ((unsigned long)p_ce_1_1 % 64 == 0) {
						ce_1[1] = vload4(0, p_ce_1_1);
					} else {
						ce_1[1].x = p_ce_1_1[0];
						p_ce_1_1++;
						ce_1[1].y = p_ce_1_1[0];
						p_ce_1_1++;
						ce_1[1].z = p_ce_1_1[0];
						p_ce_1_1++;
						ce_1[1].w = p_ce_1_1[0];
						p_ce_1_1++;
					}
					__global double *p_ce_1_2 =
					    (__global double *)&ce[m][8];
					if ((unsigned long)p_ce_1_2 % 64 == 0) {
						ce_1[2] = vload4(0, p_ce_1_2);
					} else {
						ce_1[2].x = p_ce_1_2[0];
						p_ce_1_2++;
						ce_1[2].y = p_ce_1_2[0];
						p_ce_1_2++;
						ce_1[2].z = p_ce_1_2[0];
						p_ce_1_2++;
						ce_1[2].w = p_ce_1_2[0];
						p_ce_1_2++;
					}
					//-------------------------------------------
					//Prefetching (END)
					//-------------------------------------------

					rsd[i][j][k][m] =
					    ce_1[0].x /*ce[m][0] */  +
					    ce_1[0].y /*ce[m][1] */  * xi +
					    ce_1[0].z /*ce[m][2] */  * eta +
					    ce_1[0].w /*ce[m][3] */  * zeta +
					    ce_1[1].x /*ce[m][4] */  * xi * xi +
					    ce_1[1].y /*ce[m][5] */  * eta *
					    eta +
					    ce_1[1].z /*ce[m][6] */  * zeta *
					    zeta +
					    ce_1[1].w /*ce[m][7] */  * xi * xi *
					    xi +
					    ce_1[2].x /*ce[m][8] */  * eta *
					    eta * eta +
					    ce_1[2].y /*ce[m][9] */  * zeta *
					    zeta * zeta +
					    ce_1[2].z /*ce[m][10] */  * xi *
					    xi * xi * xi +
					    ce_1[2].w /*ce[m][11] */  * eta *
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
//Loop defined at line 787 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_2(__global double *g_flux, __global double *g_rsd, int jst,
		     int L1, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *tls_validflag)
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
	double u21;		/* (User-defined privated variables) : Defined at lu.c : 729 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 728 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 789
		//-------------------------------------------
		double4 rsd_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 789
		//Candidates:
		//      rsd[i][j][k][0]
		//      rsd[i][j][k][1]
		//      rsd[i][j][k][2]
		//      rsd[i][j][k][3]
		//-------------------------------------------
		__global double *p_rsd_1_0 =
		    (__global double *)&rsd[i][j][k][0];
		if ((unsigned long)p_rsd_1_0 % 64 == 0) {
			rsd_1 = vload4(0, p_rsd_1_0);
		} else {
			rsd_1.x = p_rsd_1_0[0];
			p_rsd_1_0++;
			rsd_1.y = p_rsd_1_0[0];
			p_rsd_1_0++;
			rsd_1.z = p_rsd_1_0[0];
			p_rsd_1_0++;
			rsd_1.w = p_rsd_1_0[0];
			p_rsd_1_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[i][j][k][0] = rsd_1.y /*rsd[i][j][k][1] */ ;
		u21 =
		    rsd_1.y /*rsd[i][j][k][1] */  /
		    rsd_1.x /*rsd[i][j][k][0] */ ;
		q = 0.50 * (rsd_1.y /*rsd[i][j][k][1] */  *
			    rsd_1.y /*rsd[i][j][k][1] */  +
			    rsd_1.z /*rsd[i][j][k][2] */  *
			    rsd_1.z /*rsd[i][j][k][2] */  +
			    rsd_1.w /*rsd[i][j][k][3] */  *
			    rsd_1.w /*rsd[i][j][k][3] */ ) /
		    rsd_1.x /*rsd[i][j][k][0] */ ;
		flux[i][j][k][1] =
		    rsd_1.y /*rsd[i][j][k][1] */  * u21 +
		    0.40e+00 * (rsd[i][j][k][4] - q);
		flux[i][j][k][2] = rsd_1.z /*rsd[i][j][k][2] */  * u21;
		flux[i][j][k][3] = rsd_1.w /*rsd[i][j][k][3] */  * u21;
		flux[i][j][k][4] =
		    (1.40e+00 * rsd[i][j][k][4] - 0.40e+00 * q) * u21;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 806 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_3(int ist, int iend, __global double *g_frct, double tx2,
		     __global double *g_flux, int L2, __global double *g_rsd,
		     double tx3, double dx1, double tx1, double dx2, double dx3,
		     double dx4, double dx5, double dsspm, int nx, int jst,
		     int __ocl_k_bound, int __ocl_j_bound,
		     __global int *tls_validflag)
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
	int i;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 730 */
	double u21i;		/* (User-defined privated variables) : Defined at lu.c : 731 */
	double u31i;		/* (User-defined privated variables) : Defined at lu.c : 731 */
	double u41i;		/* (User-defined privated variables) : Defined at lu.c : 731 */
	double u51i;		/* (User-defined privated variables) : Defined at lu.c : 731 */
	double u21im1;		/* (User-defined privated variables) : Defined at lu.c : 734 */
	double u31im1;		/* (User-defined privated variables) : Defined at lu.c : 734 */
	double u41im1;		/* (User-defined privated variables) : Defined at lu.c : 734 */
	double u51im1;		/* (User-defined privated variables) : Defined at lu.c : 734 */
	int ist1;		/* (User-defined privated variables) : Defined at lu.c : 724 */
	int iend1;		/* (User-defined privated variables) : Defined at lu.c : 724 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[65][65][5] =
	    (__global double (*)[65][65][5])g_frct;
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = ist; i <= iend; i++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    tx2 * (flux[i + 1][j][k][m] -
					   flux[i - 1][j][k][m]);
			}
		}
		for (i = ist; i <= L2; i++) {
			tmp = 1.0 / rsd[i][j][k][0];
			u21i = tmp * rsd[i][j][k][1];
			u31i = tmp * rsd[i][j][k][2];
			u41i = tmp * rsd[i][j][k][3];
			u51i = tmp * rsd[i][j][k][4];
			tmp = 1.0 / rsd[i - 1][j][k][0];
			u21im1 = tmp * rsd[i - 1][j][k][1];
			u31im1 = tmp * rsd[i - 1][j][k][2];
			u41im1 = tmp * rsd[i - 1][j][k][3];
			u51im1 = tmp * rsd[i - 1][j][k][4];
			flux[i][j][k][1] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
			flux[i][j][k][2] = tx3 * (u31i - u31im1);
			flux[i][j][k][3] = tx3 * (u41i - u41im1);
			flux[i][j][k][4] =
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
			frct[i][j][k][0] =
			    frct[i][j][k][0] +
			    dx1 * tx1 * (rsd[i - 1][j][k][0] -
					 2.0 * rsd[i][j][k][0] + rsd[i +
								     1][j][k]
					 [0]);
			frct[i][j][k][1] =
			    frct[i][j][k][1] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][1] -
							 flux[i][j][k][1]) +
			    dx2 * tx1 * (rsd[i - 1][j][k][1] -
					 2.0 * rsd[i][j][k][1] + rsd[i +
								     1][j][k]
					 [1]);
			frct[i][j][k][2] =
			    frct[i][j][k][2] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][2] -
							 flux[i][j][k][2]) +
			    dx3 * tx1 * (rsd[i - 1][j][k][2] -
					 2.0 * rsd[i][j][k][2] + rsd[i +
								     1][j][k]
					 [2]);
			frct[i][j][k][3] =
			    frct[i][j][k][3] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][3] -
							 flux[i][j][k][3]) +
			    dx4 * tx1 * (rsd[i - 1][j][k][3] -
					 2.0 * rsd[i][j][k][3] + rsd[i +
								     1][j][k]
					 [3]);
			frct[i][j][k][4] =
			    frct[i][j][k][4] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][4] -
							 flux[i][j][k][4]) +
			    dx5 * tx1 * (rsd[i - 1][j][k][4] -
					 2.0 * rsd[i][j][k][4] + rsd[i +
								     1][j][k]
					 [4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 871
			//-------------------------------------------
			double rsd_4[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 871
			//Candidates:
			//      rsd[1][j][k][m]
			//      rsd[2][j][k][m]
			//      rsd[3][j][k][m]
			//-------------------------------------------
			rsd_4[0] = rsd[1][j][k][m];
			rsd_4[1] = rsd[2][j][k][m];
			rsd_4[2] = rsd[3][j][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[1][j][k][m] =
			    frct[1][j][k][m] -
			    dsspm * (+5.0 * rsd_4[0] /*rsd[1][j][k][m] */ -4.0 *
				     rsd_4[1] /*rsd[2][j][k][m] */ +rsd_4[2]
				     /*rsd[3][j][k][m] */ );
			frct[2][j][k][m] =
			    frct[2][j][k][m] -
			    dsspm * (-4.0 * rsd_4[0] /*rsd[1][j][k][m] */ +6.0 *
				     rsd_4[1] /*rsd[2][j][k][m] */ -4.0 *
				     rsd_4[2] /*rsd[3][j][k][m] */
				     +rsd[4][j][k][m]);
		}
		ist1 = 3;
		iend1 = nx - 4;
		for (i = ist1; i <= iend1; i++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    dsspm * (rsd[i - 2][j][k][m] -
					     4.0 * rsd[i - 1][j][k][m] +
					     6.0 * rsd[i][j][k][m] -
					     4.0 * rsd[i + 1][j][k][m] + rsd[i +
									     2]
					     [j][k][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 896
			//-------------------------------------------
			double rsd_5[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 896
			//Candidates:
			//      rsd[nx - 4][j][k][m]
			//      rsd[nx - 3][j][k][m]
			//-------------------------------------------
			rsd_5[0] = rsd[nx - 4][j][k][m];
			rsd_5[1] = rsd[nx - 3][j][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[nx - 3][j][k][m] =
			    frct[nx - 3][j][k][m] -
			    dsspm * (rsd[nx - 5][j][k][m] -
				     4.0 *
				     rsd_5[0] /*rsd[nx - 4][j][k][m] */ +6.0 *
				     rsd_5[1] /*rsd[nx - 3][j][k][m] */ -4.0 *
				     rsd[nx - 2][j][k][m]);
			frct[nx - 2][j][k][m] =
			    frct[nx - 2][j][k][m] -
			    dsspm * (rsd_5[0] /*rsd[nx - 4][j][k][m] */ -4.0 *
				     rsd_5[1] /*rsd[nx - 3][j][k][m] */ +5.0 *
				     rsd[nx - 2][j][k][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 918 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_4(__global double *g_flux, __global double *g_rsd, int L1,
		     int ist, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *tls_validflag)
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
	double u31;		/* (User-defined privated variables) : Defined at lu.c : 729 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 728 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 920
		//-------------------------------------------
		double4 rsd_7;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 920
		//Candidates:
		//      rsd[i][j][k][0]
		//      rsd[i][j][k][1]
		//      rsd[i][j][k][2]
		//      rsd[i][j][k][3]
		//-------------------------------------------
		__global double *p_rsd_7_0 =
		    (__global double *)&rsd[i][j][k][0];
		if ((unsigned long)p_rsd_7_0 % 64 == 0) {
			rsd_7 = vload4(0, p_rsd_7_0);
		} else {
			rsd_7.x = p_rsd_7_0[0];
			p_rsd_7_0++;
			rsd_7.y = p_rsd_7_0[0];
			p_rsd_7_0++;
			rsd_7.z = p_rsd_7_0[0];
			p_rsd_7_0++;
			rsd_7.w = p_rsd_7_0[0];
			p_rsd_7_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[i][j][k][0] = rsd_7.z /*rsd[i][j][k][2] */ ;
		u31 =
		    rsd_7.z /*rsd[i][j][k][2] */  /
		    rsd_7.x /*rsd[i][j][k][0] */ ;
		q = 0.50 * (rsd_7.y /*rsd[i][j][k][1] */  *
			    rsd_7.y /*rsd[i][j][k][1] */  +
			    rsd_7.z /*rsd[i][j][k][2] */  *
			    rsd_7.z /*rsd[i][j][k][2] */  +
			    rsd_7.w /*rsd[i][j][k][3] */  *
			    rsd_7.w /*rsd[i][j][k][3] */ ) /
		    rsd_7.x /*rsd[i][j][k][0] */ ;
		flux[i][j][k][1] = rsd_7.y /*rsd[i][j][k][1] */  * u31;
		flux[i][j][k][2] =
		    rsd_7.z /*rsd[i][j][k][2] */  * u31 +
		    0.40e+00 * (rsd[i][j][k][4] - q);
		flux[i][j][k][3] = rsd_7.w /*rsd[i][j][k][3] */  * u31;
		flux[i][j][k][4] =
		    (1.40e+00 * rsd[i][j][k][4] - 0.40e+00 * q) * u31;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 937 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_5(int jst, int jend, __global double *g_frct, double ty2,
		     __global double *g_flux, int L2, __global double *g_rsd,
		     double ty3, double dy1, double ty1, double dy2, double dy3,
		     double dy4, double dy5, double dsspm, int ny, int ist,
		     int __ocl_k_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	int j;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 730 */
	double u21j;		/* (User-defined privated variables) : Defined at lu.c : 732 */
	double u31j;		/* (User-defined privated variables) : Defined at lu.c : 732 */
	double u41j;		/* (User-defined privated variables) : Defined at lu.c : 732 */
	double u51j;		/* (User-defined privated variables) : Defined at lu.c : 732 */
	double u21jm1;		/* (User-defined privated variables) : Defined at lu.c : 735 */
	double u31jm1;		/* (User-defined privated variables) : Defined at lu.c : 735 */
	double u41jm1;		/* (User-defined privated variables) : Defined at lu.c : 735 */
	double u51jm1;		/* (User-defined privated variables) : Defined at lu.c : 735 */
	int jst1;		/* (User-defined privated variables) : Defined at lu.c : 725 */
	int jend1;		/* (User-defined privated variables) : Defined at lu.c : 725 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*frct)[65][65][5] =
	    (__global double (*)[65][65][5])g_frct;
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (j = jst; j <= jend; j++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    ty2 * (flux[i][j + 1][k][m] -
					   flux[i][j - 1][k][m]);
			}
		}
		for (j = jst; j <= L2; j++) {
			tmp = 1.0 / rsd[i][j][k][0];
			u21j = tmp * rsd[i][j][k][1];
			u31j = tmp * rsd[i][j][k][2];
			u41j = tmp * rsd[i][j][k][3];
			u51j = tmp * rsd[i][j][k][4];
			tmp = 1.0 / rsd[i][j - 1][k][0];
			u21jm1 = tmp * rsd[i][j - 1][k][1];
			u31jm1 = tmp * rsd[i][j - 1][k][2];
			u41jm1 = tmp * rsd[i][j - 1][k][3];
			u51jm1 = tmp * rsd[i][j - 1][k][4];
			flux[i][j][k][1] = ty3 * (u21j - u21jm1);
			flux[i][j][k][2] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
			flux[i][j][k][3] = ty3 * (u41j - u41jm1);
			flux[i][j][k][4] =
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
			frct[i][j][k][0] =
			    frct[i][j][k][0] +
			    dy1 * ty1 * (rsd[i][j - 1][k][0] -
					 2.0 * rsd[i][j][k][0] + rsd[i][j +
									1][k]
					 [0]);
			frct[i][j][k][1] =
			    frct[i][j][k][1] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][1] -
							 flux[i][j][k][1]) +
			    dy2 * ty1 * (rsd[i][j - 1][k][1] -
					 2.0 * rsd[i][j][k][1] + rsd[i][j +
									1][k]
					 [1]);
			frct[i][j][k][2] =
			    frct[i][j][k][2] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][2] -
							 flux[i][j][k][2]) +
			    dy3 * ty1 * (rsd[i][j - 1][k][2] -
					 2.0 * rsd[i][j][k][2] + rsd[i][j +
									1][k]
					 [2]);
			frct[i][j][k][3] =
			    frct[i][j][k][3] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][3] -
							 flux[i][j][k][3]) +
			    dy4 * ty1 * (rsd[i][j - 1][k][3] -
					 2.0 * rsd[i][j][k][3] + rsd[i][j +
									1][k]
					 [3]);
			frct[i][j][k][4] =
			    frct[i][j][k][4] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][4] -
							 flux[i][j][k][4]) +
			    dy5 * ty1 * (rsd[i][j - 1][k][4] -
					 2.0 * rsd[i][j][k][4] + rsd[i][j +
									1][k]
					 [4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1002
			//-------------------------------------------
			double rsd_10[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1002
			//Candidates:
			//      rsd[i][1][k][m]
			//      rsd[i][2][k][m]
			//      rsd[i][3][k][m]
			//-------------------------------------------
			rsd_10[0] = rsd[i][1][k][m];
			rsd_10[1] = rsd[i][2][k][m];
			rsd_10[2] = rsd[i][3][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[i][1][k][m] =
			    frct[i][1][k][m] -
			    dsspm * (+5.0 *
				     rsd_10[0] /*rsd[i][1][k][m] */ -4.0 *
				     rsd_10[1] /*rsd[i][2][k][m] */ +rsd_10[2]
				     /*rsd[i][3][k][m] */ );
			frct[i][2][k][m] =
			    frct[i][2][k][m] -
			    dsspm * (-4.0 *
				     rsd_10[0] /*rsd[i][1][k][m] */ +6.0 *
				     rsd_10[1] /*rsd[i][2][k][m] */ -4.0 *
				     rsd_10[2] /*rsd[i][3][k][m] */
				     +rsd[i][4][k][m]);
		}
		jst1 = 3;
		jend1 = ny - 4;
		for (j = jst1; j <= jend1; j++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    dsspm * (rsd[i][j - 2][k][m] -
					     4.0 * rsd[i][j - 1][k][m] +
					     6.0 * rsd[i][j][k][m] -
					     4.0 * rsd[i][j + 1][k][m] +
					     rsd[i][j + 2][k][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1028
			//-------------------------------------------
			double rsd_11[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1028
			//Candidates:
			//      rsd[i][ny - 4][k][m]
			//      rsd[i][ny - 3][k][m]
			//-------------------------------------------
			rsd_11[0] = rsd[i][ny - 4][k][m];
			rsd_11[1] = rsd[i][ny - 3][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[i][ny - 3][k][m] =
			    frct[i][ny - 3][k][m] -
			    dsspm * (rsd[i][ny - 5][k][m] -
				     4.0 *
				     rsd_11[0] /*rsd[i][ny - 4][k][m] */ +6.0 *
				     rsd_11[1] /*rsd[i][ny - 3][k][m] */ -4.0 *
				     rsd[i][ny - 2][k][m]);
			frct[i][ny - 2][k][m] =
			    frct[i][ny - 2][k][m] -
			    dsspm * (rsd_11[0] /*rsd[i][ny - 4][k][m] */ -4.0 *
				     rsd_11[1] /*rsd[i][ny - 3][k][m] */ +5.0 *
				     rsd[i][ny - 2][k][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1047 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void erhs_6(int nz, __global double *g_flux, __global double *g_rsd,
		     __global double *g_frct, double tz2, double tz3,
		     double dz1, double tz1, double dz2, double dz3, double dz4,
		     double dz5, double dsspm, int jst, int ist,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	int k;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	double u41;		/* (User-defined privated variables) : Defined at lu.c : 729 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 728 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 721 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 730 */
	double u21k;		/* (User-defined privated variables) : Defined at lu.c : 733 */
	double u31k;		/* (User-defined privated variables) : Defined at lu.c : 733 */
	double u41k;		/* (User-defined privated variables) : Defined at lu.c : 733 */
	double u51k;		/* (User-defined privated variables) : Defined at lu.c : 733 */
	double u21km1;		/* (User-defined privated variables) : Defined at lu.c : 736 */
	double u31km1;		/* (User-defined privated variables) : Defined at lu.c : 736 */
	double u41km1;		/* (User-defined privated variables) : Defined at lu.c : 736 */
	double u51km1;		/* (User-defined privated variables) : Defined at lu.c : 736 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	__global double (*frct)[65][65][5] =
	    (__global double (*)[65][65][5])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1049
			//-------------------------------------------
			double4 rsd_15;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1049
			//Candidates:
			//      rsd[i][j][k][0]
			//      rsd[i][j][k][1]
			//      rsd[i][j][k][2]
			//      rsd[i][j][k][3]
			//-------------------------------------------
			__global double *p_rsd_15_0 =
			    (__global double *)&rsd[i][j][k][0];
			if ((unsigned long)p_rsd_15_0 % 64 == 0) {
				rsd_15 = vload4(0, p_rsd_15_0);
			} else {
				rsd_15.x = p_rsd_15_0[0];
				p_rsd_15_0++;
				rsd_15.y = p_rsd_15_0[0];
				p_rsd_15_0++;
				rsd_15.z = p_rsd_15_0[0];
				p_rsd_15_0++;
				rsd_15.w = p_rsd_15_0[0];
				p_rsd_15_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			flux[i][j][k][0] = rsd_15.w /*rsd[i][j][k][3] */ ;
			u41 =
			    rsd_15.w /*rsd[i][j][k][3] */  /
			    rsd_15.x /*rsd[i][j][k][0] */ ;
			q = 0.50 * (rsd_15.y /*rsd[i][j][k][1] */  *
				    rsd_15.y /*rsd[i][j][k][1] */  +
				    rsd_15.z /*rsd[i][j][k][2] */  *
				    rsd_15.z /*rsd[i][j][k][2] */  +
				    rsd_15.w /*rsd[i][j][k][3] */  *
				    rsd_15.w /*rsd[i][j][k][3] */ ) /
			    rsd_15.x /*rsd[i][j][k][0] */ ;
			flux[i][j][k][1] = rsd_15.y /*rsd[i][j][k][1] */  * u41;
			flux[i][j][k][2] = rsd_15.z /*rsd[i][j][k][2] */  * u41;
			flux[i][j][k][3] =
			    rsd_15.w /*rsd[i][j][k][3] */  * u41 +
			    0.40e+00 * (rsd[i][j][k][4] - q);
			flux[i][j][k][4] =
			    (1.40e+00 * rsd[i][j][k][4] - 0.40e+00 * q) * u41;
		}
		for (k = 1; k <= nz - 2; k++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    tz2 * (flux[i][j][k + 1][m] -
					   flux[i][j][k - 1][m]);
			}
		}
		for (k = 1; k <= nz - 1; k++) {
			tmp = 1.0 / rsd[i][j][k][0];
			u21k = tmp * rsd[i][j][k][1];
			u31k = tmp * rsd[i][j][k][2];
			u41k = tmp * rsd[i][j][k][3];
			u51k = tmp * rsd[i][j][k][4];
			tmp = 1.0 / rsd[i][j][k - 1][0];
			u21km1 = tmp * rsd[i][j][k - 1][1];
			u31km1 = tmp * rsd[i][j][k - 1][2];
			u41km1 = tmp * rsd[i][j][k - 1][3];
			u51km1 = tmp * rsd[i][j][k - 1][4];
			flux[i][j][k][1] = tz3 * (u21k - u21km1);
			flux[i][j][k][2] = tz3 * (u31k - u31km1);
			flux[i][j][k][3] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
			flux[i][j][k][4] =
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
			frct[i][j][k][0] =
			    frct[i][j][k][0] +
			    dz1 * tz1 * (rsd[i][j][k + 1][0] -
					 2.0 * rsd[i][j][k][0] + rsd[i][j][k -
									   1]
					 [0]);
			frct[i][j][k][1] =
			    frct[i][j][k][1] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][1] -
							 flux[i][j][k][1]) +
			    dz2 * tz1 * (rsd[i][j][k + 1][1] -
					 2.0 * rsd[i][j][k][1] + rsd[i][j][k -
									   1]
					 [1]);
			frct[i][j][k][2] =
			    frct[i][j][k][2] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][2] -
							 flux[i][j][k][2]) +
			    dz3 * tz1 * (rsd[i][j][k + 1][2] -
					 2.0 * rsd[i][j][k][2] + rsd[i][j][k -
									   1]
					 [2]);
			frct[i][j][k][3] =
			    frct[i][j][k][3] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][3] -
							 flux[i][j][k][3]) +
			    dz4 * tz1 * (rsd[i][j][k + 1][3] -
					 2.0 * rsd[i][j][k][3] + rsd[i][j][k -
									   1]
					 [3]);
			frct[i][j][k][4] =
			    frct[i][j][k][4] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][4] -
							 flux[i][j][k][4]) +
			    dz5 * tz1 * (rsd[i][j][k + 1][4] -
					 2.0 * rsd[i][j][k][4] + rsd[i][j][k -
									   1]
					 [4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1126
			//-------------------------------------------
			double rsd_16[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1126
			//Candidates:
			//      rsd[i][j][1][m]
			//      rsd[i][j][2][m]
			//      rsd[i][j][3][m]
			//-------------------------------------------
			rsd_16[0] = rsd[i][j][1][m];
			rsd_16[1] = rsd[i][j][2][m];
			rsd_16[2] = rsd[i][j][3][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[i][j][1][m] =
			    frct[i][j][1][m] -
			    dsspm * (+5.0 *
				     rsd_16[0] /*rsd[i][j][1][m] */ -4.0 *
				     rsd_16[1] /*rsd[i][j][2][m] */ +rsd_16[2]
				     /*rsd[i][j][3][m] */ );
			frct[i][j][2][m] =
			    frct[i][j][2][m] -
			    dsspm * (-4.0 *
				     rsd_16[0] /*rsd[i][j][1][m] */ +6.0 *
				     rsd_16[1] /*rsd[i][j][2][m] */ -4.0 *
				     rsd_16[2] /*rsd[i][j][3][m] */
				     +rsd[i][j][4][m]);
		}
		for (k = 3; k <= nz - 4; k++) {
			for (m = 0; m < 5; m++) {
				frct[i][j][k][m] =
				    frct[i][j][k][m] -
				    dsspm * (rsd[i][j][k - 2][m] -
					     4.0 * rsd[i][j][k - 1][m] +
					     6.0 * rsd[i][j][k][m] -
					     4.0 * rsd[i][j][k + 1][m] +
					     rsd[i][j][k + 2][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1149
			//-------------------------------------------
			double rsd_17[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1149
			//Candidates:
			//      rsd[i][j][nz - 4][m]
			//      rsd[i][j][nz - 3][m]
			//-------------------------------------------
			rsd_17[0] = rsd[i][j][nz - 4][m];
			rsd_17[1] = rsd[i][j][nz - 3][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			frct[i][j][nz - 3][m] =
			    frct[i][j][nz - 3][m] -
			    dsspm * (rsd[i][j][nz - 5][m] -
				     4.0 *
				     rsd_17[0] /*rsd[i][j][nz - 4][m] */ +6.0 *
				     rsd_17[1] /*rsd[i][j][nz - 3][m] */ -4.0 *
				     rsd[i][j][nz - 2][m]);
			frct[i][j][nz - 2][m] =
			    frct[i][j][nz - 2][m] -
			    dsspm * (rsd_17[0] /*rsd[i][j][nz - 4][m] */ -4.0 *
				     rsd_17[1] /*rsd[i][j][nz - 3][m] */ +5.0 *
				     rsd[i][j][nz - 2][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1269 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
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
		      int __ocl_i_bound, __global int *tls_validflag)
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
	double tmp1;		/* (User-defined privated variables) : Defined at lu.c : 1261 */
	double tmp2;		/* (User-defined privated variables) : Defined at lu.c : 1261 */
	double tmp3;		/* (User-defined privated variables) : Defined at lu.c : 1261 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*d)[64][5][5] = (__global double (*)[64][5][5])g_d;
	__global double (*a)[64][5][5] = (__global double (*)[64][5][5])g_a;
	__global double (*b)[64][5][5] = (__global double (*)[64][5][5])g_b;
	__global double (*c)[64][5][5] = (__global double (*)[64][5][5])g_c;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1270
		//-------------------------------------------
		double4 u_3[3];
		double u_4[2];
		double2 u_5;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1270
		//Candidates:
		//      u[i - 1][j][k][0]
		//      u[i - 1][j][k][1]
		//      u[i - 1][j][k][2]
		//      u[i - 1][j][k][3]
		//      u[i][j][k - 1][1]
		//      u[i][j][k - 1][2]
		//      u[i][j][k - 1][3]
		//      u[i][j][k - 1][4]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//      u[i][j][k][4]
		//      u[i - 1][j][k][4]
		//      u[i][j - 1][k][3]
		//      u[i][j - 1][k][1]
		//      u[i][j - 1][k][2]
		//-------------------------------------------
		__global double *p_u_3_0 =
		    (__global double *)&u[i - 1][j][k][0];
		if ((unsigned long)p_u_3_0 % 64 == 0) {
			u_3[0] = vload4(0, p_u_3_0);
		} else {
			u_3[0].x = p_u_3_0[0];
			p_u_3_0++;
			u_3[0].y = p_u_3_0[0];
			p_u_3_0++;
			u_3[0].z = p_u_3_0[0];
			p_u_3_0++;
			u_3[0].w = p_u_3_0[0];
			p_u_3_0++;
		}
		__global double *p_u_3_1 =
		    (__global double *)&u[i][j][k - 1][1];
		if ((unsigned long)p_u_3_1 % 64 == 0) {
			u_3[1] = vload4(0, p_u_3_1);
		} else {
			u_3[1].x = p_u_3_1[0];
			p_u_3_1++;
			u_3[1].y = p_u_3_1[0];
			p_u_3_1++;
			u_3[1].z = p_u_3_1[0];
			p_u_3_1++;
			u_3[1].w = p_u_3_1[0];
			p_u_3_1++;
		}
		__global double *p_u_3_2 = (__global double *)&u[i][j][k][1];
		if ((unsigned long)p_u_3_2 % 64 == 0) {
			u_3[2] = vload4(0, p_u_3_2);
		} else {
			u_3[2].x = p_u_3_2[0];
			p_u_3_2++;
			u_3[2].y = p_u_3_2[0];
			p_u_3_2++;
			u_3[2].z = p_u_3_2[0];
			p_u_3_2++;
			u_3[2].w = p_u_3_2[0];
			p_u_3_2++;
		}
		u_4[0] = u[i - 1][j][k][4];
		u_4[1] = u[i][j - 1][k][3];
		__global double *p_u_5_0 =
		    (__global double *)&u[i][j - 1][k][1];
		if ((unsigned long)p_u_5_0 % 64 == 0) {
			u_5 = vload2(0, p_u_5_0);
		} else {
			u_5.x = p_u_5_0[0];
			p_u_5_0++;
			u_5.y = p_u_5_0[0];
			p_u_5_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u[i][j][k][0];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		d[i][j][0][0] =
		    1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
		d[i][j][0][1] = 0.0;
		d[i][j][0][2] = 0.0;
		d[i][j][0][3] = 0.0;
		d[i][j][0][4] = 0.0;
		d[i][j][1][0] =
		    dt * 2.0 * (tx1 *
				(-r43 * c34 * tmp2 *
				 u_3[2].x /*u[i][j][k][1] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_3[2].x /*u[i][j][k][1] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_3[2].x /*u[i][j][k][1] */ ));
		d[i][j][1][1] =
		    1.0 + dt * 2.0 * (tx1 * r43 * c34 * tmp1 +
				      ty1 * c34 * tmp1 + tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
		d[i][j][1][2] = 0.0;
		d[i][j][1][3] = 0.0;
		d[i][j][1][4] = 0.0;
		d[i][j][2][0] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_3[2].y /*u[i][j][k][2] */ ) +
				ty1 * (-r43 * c34 * tmp2 *
				       u_3[2].y /*u[i][j][k][2] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_3[2].y /*u[i][j][k][2] */ ));
		d[i][j][2][1] = 0.0;
		d[i][j][2][2] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 +
				      ty1 * r43 * c34 * tmp1 +
				      tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
		d[i][j][2][3] = 0.0;
		d[i][j][2][4] = 0.0;
		d[i][j][3][0] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_3[2].z /*u[i][j][k][3] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_3[2].z /*u[i][j][k][3] */ ) +
				tz1 * (-r43 * c34 * tmp2 *
				       u_3[2].z /*u[i][j][k][3] */ ));
		d[i][j][3][1] = 0.0;
		d[i][j][3][2] = 0.0;
		d[i][j][3][3] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 +
				      tz1 * r43 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
		d[i][j][3][4] = 0.0;
		d[i][j][4][0] =
		    dt * 2.0 * (tx1 *
				(-(r43 * c34 - c1345) * tmp3 *
				 (((u_3[2].x /*u[i][j][k][1] */ ) *
				   (u_3[2].x /*u[i][j][k][1] */ ))) - (c34 -
								       c1345) *
				 tmp3 *
				 (((u_3[2].y /*u[i][j][k][2] */ ) *
				   (u_3[2].y /*u[i][j][k][2] */ ))) - (c34 -
								       c1345) *
				 tmp3 *
				 (((u_3[2].z /*u[i][j][k][3] */ ) *
				   (u_3[2].z /*u[i][j][k][3] */ ))) -
				 (c1345) * tmp2 *
				 u_3[2].w /*u[i][j][k][4] */ ) + ty1 * (-(c34 -
									  c1345)
									* tmp3 *
									(((u_3
									   [2].
									   x
									   /*u[i][j][k][1] */
									   ) *
									  (u_3
									   [2].
									   x
									   /*u[i][j][k][1] */
									   ))) -
									(r43 *
									 c34 -
									 c1345)
									* tmp3 *
									(((u_3
									   [2].
									   y
									   /*u[i][j][k][2] */
									   ) *
									  (u_3
									   [2].
									   y
									   /*u[i][j][k][2] */
									   ))) -
									(c34 -
									 c1345)
									* tmp3 *
									(((u_3
									   [2].
									   z
									   /*u[i][j][k][3] */
									   ) *
									  (u_3
									   [2].
									   z
									   /*u[i][j][k][3] */
									   ))) -
									(c1345)
									* tmp2 *
									u_3[2].
									w
									/*u[i][j][k][4] */
									) +
				tz1 * (-(c34 - c1345) * tmp3 *
				       (((u_3[2].x /*u[i][j][k][1] */ ) *
					 (u_3[2].x /*u[i][j][k][1] */ ))) -
				       (c34 -
					c1345) * tmp3 *
				       (((u_3[2].y /*u[i][j][k][2] */ ) *
					 (u_3[2].y /*u[i][j][k][2] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_3[2].z /*u[i][j][k][3] */ ) *
					 (u_3[2].z /*u[i][j][k][3] */ ))) -
				       (c1345) * tmp2 *
				       u_3[2].w /*u[i][j][k][4] */ ));
		d[i][j][4][1] =
		    dt * 2.0 * (tx1 * (r43 * c34 - c1345) * tmp2 *
				u_3[2].x /*u[i][j][k][1] */  + ty1 * (c34 -
								      c1345) *
				tmp2 * u_3[2].x /*u[i][j][k][1] */  +
				tz1 * (c34 -
				       c1345) * tmp2 *
				u_3[2].x /*u[i][j][k][1] */ );
		d[i][j][4][2] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_3[2].y /*u[i][j][k][2] */  +
				ty1 * (r43 * c34 -
				       c1345) * tmp2 *
				u_3[2].y /*u[i][j][k][2] */  + tz1 * (c34 -
								      c1345) *
				tmp2 * u_3[2].y /*u[i][j][k][2] */ );
		d[i][j][4][3] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_3[2].z /*u[i][j][k][3] */  + ty1 * (c34 -
								      c1345) *
				tmp2 * u_3[2].z /*u[i][j][k][3] */  +
				tz1 * (r43 * c34 -
				       c1345) * tmp2 *
				u_3[2].z /*u[i][j][k][3] */ );
		d[i][j][4][4] =
		    1.0 + dt * 2.0 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 +
				      tz1 * c1345 * tmp1) +
		    dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
		tmp1 = 1.0 / u[i][j][k - 1][0];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		a[i][j][0][0] = -dt * tz1 * dz1;
		a[i][j][0][1] = 0.0;
		a[i][j][0][2] = 0.0;
		a[i][j][0][3] = -dt * tz2;
		a[i][j][0][4] = 0.0;
		a[i][j][1][0] =
		    -dt * tz2 *
		    (-
		     (u_3[1].x /*u[i][j][k - 1][1] */  *
		      u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_3[1].x /*u[i][j][k - 1][1] */ );
		a[i][j][1][1] =
		    -dt * tz2 * (u_3[1].z /*u[i][j][k - 1][3] */  * tmp1) -
		    dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
		a[i][j][1][2] = 0.0;
		a[i][j][1][3] =
		    -dt * tz2 * (u_3[1].x /*u[i][j][k - 1][1] */  * tmp1);
		a[i][j][1][4] = 0.0;
		a[i][j][2][0] =
		    -dt * tz2 *
		    (-
		     (u_3[1].y /*u[i][j][k - 1][2] */  *
		      u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_3[1].y /*u[i][j][k - 1][2] */ );
		a[i][j][2][1] = 0.0;
		a[i][j][2][2] =
		    -dt * tz2 * (u_3[1].z /*u[i][j][k - 1][3] */  * tmp1) -
		    dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
		a[i][j][2][3] =
		    -dt * tz2 * (u_3[1].y /*u[i][j][k - 1][2] */  * tmp1);
		a[i][j][2][4] = 0.0;
		a[i][j][3][0] =
		    -dt * tz2 * (-(u_3[1].z /*u[i][j][k - 1][3] */  * tmp1) *
				 (u_3[1].z /*u[i][j][k - 1][3] */  * tmp1) +
				 0.50 * 0.40e+00 *
				 ((u_3[1].x /*u[i][j][k - 1][1] */  *
				   u_3[1].x /*u[i][j][k - 1][1] */  +
				   u_3[1].y /*u[i][j][k - 1][2] */  *
				   u_3[1].y /*u[i][j][k - 1][2] */  +
				   u_3[1].z /*u[i][j][k - 1][3] */  *
				   u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2)) -
		    dt * tz1 * (-r43 * c34 * tmp2 *
				u_3[1].z /*u[i][j][k - 1][3] */ );
		a[i][j][3][1] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_3[1].x /*u[i][j][k - 1][1] */  * tmp1));
		a[i][j][3][2] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_3[1].y /*u[i][j][k - 1][2] */  * tmp1));
		a[i][j][3][3] =
		    -dt * tz2 * (2.0 -
				 0.40e+00) * (u_3[1].z /*u[i][j][k - 1][3] */  *
					      tmp1) -
		    dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
		a[i][j][3][4] = -dt * tz2 * 0.40e+00;
		a[i][j][4][0] =
		    -dt * tz2 *
		    ((0.40e+00 *
		      (u_3[1].x /*u[i][j][k - 1][1] */  *
		       u_3[1].x /*u[i][j][k - 1][1] */  +
		       u_3[1].y /*u[i][j][k - 1][2] */  *
		       u_3[1].y /*u[i][j][k - 1][2] */  +
		       u_3[1].z /*u[i][j][k - 1][3] */  *
		       u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2 -
		      1.40e+00 * (u_3[1].w /*u[i][j][k - 1][4] */  * tmp1)) *
		     (u_3[1].z /*u[i][j][k - 1][3] */  * tmp1)) -
		    dt * tz1 * (-(c34 - c1345) * tmp3 *
				(u_3[1].x /*u[i][j][k - 1][1] */  *
				 u_3[1].x /*u[i][j][k - 1][1] */ ) - (c34 -
								      c1345) *
				tmp3 * (u_3[1].y /*u[i][j][k - 1][2] */  *
					u_3[1].y /*u[i][j][k - 1][2] */ ) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(u_3[1].z /*u[i][j][k - 1][3] */  *
				 u_3[1].z /*u[i][j][k - 1][3] */ ) -
				c1345 * tmp2 *
				u_3[1].w /*u[i][j][k - 1][4] */ );
		a[i][j][4][1] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_3[1].x /*u[i][j][k - 1][1] */  *
				  u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 *
		    u_3[1].x /*u[i][j][k - 1][1] */ ;
		a[i][j][4][2] =
		    -dt * tz2 * (-0.40e+00 *
				 (u_3[1].y /*u[i][j][k - 1][2] */  *
				  u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 *
		    u_3[1].y /*u[i][j][k - 1][2] */ ;
		a[i][j][4][3] =
		    -dt * tz2 * (1.40e+00 *
				 (u_3[1].w /*u[i][j][k - 1][4] */  * tmp1) -
				 0.50 * 0.40e+00 *
				 ((u_3[1].x /*u[i][j][k - 1][1] */  *
				   u_3[1].x /*u[i][j][k - 1][1] */  +
				   u_3[1].y /*u[i][j][k - 1][2] */  *
				   u_3[1].y /*u[i][j][k - 1][2] */  +
				   3.0 * u_3[1].z /*u[i][j][k - 1][3] */  *
				   u_3[1].z /*u[i][j][k - 1][3] */ ) * tmp2)) -
		    dt * tz1 * (r43 * c34 -
				c1345) * tmp2 *
		    u_3[1].z /*u[i][j][k - 1][3] */ ;
		a[i][j][4][4] =
		    -dt * tz2 * (1.40e+00 *
				 (u_3[1].z /*u[i][j][k - 1][3] */  * tmp1)) -
		    dt * tz1 * c1345 * tmp1 - dt * tz1 * dz5;
		tmp1 = 1.0 / u[i][j - 1][k][0];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		b[i][j][0][0] = -dt * ty1 * dy1;
		b[i][j][0][1] = 0.0;
		b[i][j][0][2] = -dt * ty2;
		b[i][j][0][3] = 0.0;
		b[i][j][0][4] = 0.0;
		b[i][j][1][0] =
		    -dt * ty2 *
		    (-
		     (u_5.x /*u[i][j - 1][k][1] */  *
		      u_5.y /*u[i][j - 1][k][2] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_5.x /*u[i][j - 1][k][1] */ );
		b[i][j][1][1] =
		    -dt * ty2 * (u_5.y /*u[i][j - 1][k][2] */  * tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
		b[i][j][1][2] =
		    -dt * ty2 * (u_5.x /*u[i][j - 1][k][1] */  * tmp1);
		b[i][j][1][3] = 0.0;
		b[i][j][1][4] = 0.0;
		b[i][j][2][0] =
		    -dt * ty2 * (-(u_5.y /*u[i][j - 1][k][2] */  * tmp1) *
				 (u_5.y /*u[i][j - 1][k][2] */  * tmp1) +
				 0.50 * 0.40e+00 *
				 ((u_5.x /*u[i][j - 1][k][1] */  *
				   u_5.x /*u[i][j - 1][k][1] */  +
				   u_5.y /*u[i][j - 1][k][2] */  *
				   u_5.y /*u[i][j - 1][k][2] */  +
				   u_4[1] /*u[i][j - 1][k][3] */ *u_4[1]
				   /*u[i][j - 1][k][3] */ ) * tmp2)) -
		    dt * ty1 * (-r43 * c34 * tmp2 *
				u_5.y /*u[i][j - 1][k][2] */ );
		b[i][j][2][1] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_5.x /*u[i][j - 1][k][1] */  * tmp1));
		b[i][j][2][2] =
		    -dt * ty2 * ((2.0 - 0.40e+00) *
				 (u_5.y /*u[i][j - 1][k][2] */  * tmp1)) -
		    dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
		b[i][j][2][3] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_4[1] /*u[i][j - 1][k][3] */ *tmp1));
		b[i][j][2][4] = -dt * ty2 * 0.40e+00;
		b[i][j][3][0] =
		    -dt * ty2 *
		    (-
		     (u_5.y /*u[i][j - 1][k][2] */  *
		      u_4[1] /*u[i][j - 1][k][3] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_4[1] /*u[i][j - 1][k][3] */ );
		b[i][j][3][1] = 0.0;
		b[i][j][3][2] =
		    -dt * ty2 * (u_4[1] /*u[i][j - 1][k][3] */ *tmp1);
		b[i][j][3][3] =
		    -dt * ty2 * (u_5.y /*u[i][j - 1][k][2] */  * tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
		b[i][j][3][4] = 0.0;
		b[i][j][4][0] =
		    -dt * ty2 *
		    ((0.40e+00 *
		      (u_5.x /*u[i][j - 1][k][1] */  *
		       u_5.x /*u[i][j - 1][k][1] */  +
		       u_5.y /*u[i][j - 1][k][2] */  *
		       u_5.y /*u[i][j - 1][k][2] */  +
		       u_4[1] /*u[i][j - 1][k][3] */ *u_4[1]
		       /*u[i][j - 1][k][3] */ ) * tmp2 - 1.40e+00 * (u[i][j -
									  1][k]
								     [4] *
								     tmp1)) *
		     (u_5.y /*u[i][j - 1][k][2] */  * tmp1)) -
		    dt * ty1 * (-(c34 - c1345) * tmp3 *
				(((u_5.x /*u[i][j - 1][k][1] */ ) *
				  (u_5.x /*u[i][j - 1][k][1] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_5.y /*u[i][j - 1][k][2] */ ) *
				  (u_5.y /*u[i][j - 1][k][2] */ ))) - (c34 -
								       c1345) *
				tmp3 *
				(((u_4[1] /*u[i][j - 1][k][3] */ ) *
				  (u_4[1] /*u[i][j - 1][k][3] */ ))) -
				c1345 * tmp2 * u[i][j - 1][k][4]);
		b[i][j][4][1] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_5.x /*u[i][j - 1][k][1] */  *
				  u_5.y /*u[i][j - 1][k][2] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_5.x /*u[i][j - 1][k][1] */ ;
		b[i][j][4][2] =
		    -dt * ty2 * (1.40e+00 * (u[i][j - 1][k][4] * tmp1) -
				 0.50 * 0.40e+00 *
				 ((u_5.x /*u[i][j - 1][k][1] */  *
				   u_5.x /*u[i][j - 1][k][1] */  +
				   3.0 * u_5.y /*u[i][j - 1][k][2] */  *
				   u_5.y /*u[i][j - 1][k][2] */  +
				   u_4[1] /*u[i][j - 1][k][3] */ *u_4[1]
				   /*u[i][j - 1][k][3] */ ) * tmp2)) -
		    dt * ty1 * (r43 * c34 -
				c1345) * tmp2 * u_5.y /*u[i][j - 1][k][2] */ ;
		b[i][j][4][3] =
		    -dt * ty2 * (-0.40e+00 *
				 (u_5.y /*u[i][j - 1][k][2] */  *
				  u_4[1] /*u[i][j - 1][k][3] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 * u_4[1] /*u[i][j - 1][k][3] */ ;
		b[i][j][4][4] =
		    -dt * ty2 * (1.40e+00 *
				 (u_5.y /*u[i][j - 1][k][2] */  * tmp1)) -
		    dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
		tmp1 = 1.0 / u_3[0].x /*u[i - 1][j][k][0] */ ;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		c[i][j][0][0] = -dt * tx1 * dx1;
		c[i][j][0][1] = -dt * tx2;
		c[i][j][0][2] = 0.0;
		c[i][j][0][3] = 0.0;
		c[i][j][0][4] = 0.0;
		c[i][j][1][0] =
		    -dt * tx2 * (-(u_3[0].y /*u[i - 1][j][k][1] */  * tmp1) *
				 (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1) +
				 0.40e+00 * 0.50 *
				 (u_3[0].y /*u[i - 1][j][k][1] */  *
				  u_3[0].y /*u[i - 1][j][k][1] */  +
				  u_3[0].z /*u[i - 1][j][k][2] */  *
				  u_3[0].z /*u[i - 1][j][k][2] */  +
				  u_3[0].w /*u[i - 1][j][k][3] */  *
				  u_3[0].w /*u[i - 1][j][k][3] */ ) * tmp2) -
		    dt * tx1 * (-r43 * c34 * tmp2 *
				u_3[0].y /*u[i - 1][j][k][1] */ );
		c[i][j][1][1] =
		    -dt * tx2 * ((2.0 - 0.40e+00) *
				 (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
		c[i][j][1][2] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_3[0].z /*u[i - 1][j][k][2] */  * tmp1));
		c[i][j][1][3] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_3[0].w /*u[i - 1][j][k][3] */  * tmp1));
		c[i][j][1][4] = -dt * tx2 * 0.40e+00;
		c[i][j][2][0] =
		    -dt * tx2 *
		    (-
		     (u_3[0].y /*u[i - 1][j][k][1] */  *
		      u_3[0].z /*u[i - 1][j][k][2] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_3[0].z /*u[i - 1][j][k][2] */ );
		c[i][j][2][1] =
		    -dt * tx2 * (u_3[0].z /*u[i - 1][j][k][2] */  * tmp1);
		c[i][j][2][2] =
		    -dt * tx2 * (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
		c[i][j][2][3] = 0.0;
		c[i][j][2][4] = 0.0;
		c[i][j][3][0] =
		    -dt * tx2 *
		    (-
		     (u_3[0].y /*u[i - 1][j][k][1] */  *
		      u_3[0].w /*u[i - 1][j][k][3] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_3[0].w /*u[i - 1][j][k][3] */ );
		c[i][j][3][1] =
		    -dt * tx2 * (u_3[0].w /*u[i - 1][j][k][3] */  * tmp1);
		c[i][j][3][2] = 0.0;
		c[i][j][3][3] =
		    -dt * tx2 * (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
		c[i][j][3][4] = 0.0;
		c[i][j][4][0] =
		    -dt * tx2 *
		    ((0.40e+00 *
		      (u_3[0].y /*u[i - 1][j][k][1] */  *
		       u_3[0].y /*u[i - 1][j][k][1] */  +
		       u_3[0].z /*u[i - 1][j][k][2] */  *
		       u_3[0].z /*u[i - 1][j][k][2] */  +
		       u_3[0].w /*u[i - 1][j][k][3] */  *
		       u_3[0].w /*u[i - 1][j][k][3] */ ) * tmp2 -
		      1.40e+00 * (u_4[0] /*u[i - 1][j][k][4] */ *tmp1)) *
		     (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * (-(r43 * c34 - c1345) * tmp3 *
				(((u_3[0].y /*u[i - 1][j][k][1] */ ) *
				  (u_3[0].y /*u[i - 1][j][k][1] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_3[0].z /*u[i - 1][j][k][2] */ ) *
				  (u_3[0].z /*u[i - 1][j][k][2] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_3[0].w /*u[i - 1][j][k][3] */ ) *
				  (u_3[0].w /*u[i - 1][j][k][3] */ ))) -
				c1345 * tmp2 * u_4[0] /*u[i - 1][j][k][4] */ );
		c[i][j][4][1] =
		    -dt * tx2 * (1.40e+00 *
				 (u_4[0] /*u[i - 1][j][k][4] */ *tmp1) -
				 0.50 * 0.40e+00 *
				 ((3.0 * u_3[0].y /*u[i - 1][j][k][1] */  *
				   u_3[0].y /*u[i - 1][j][k][1] */  +
				   u_3[0].z /*u[i - 1][j][k][2] */  *
				   u_3[0].z /*u[i - 1][j][k][2] */  +
				   u_3[0].w /*u[i - 1][j][k][3] */  *
				   u_3[0].w /*u[i - 1][j][k][3] */ ) * tmp2)) -
		    dt * tx1 * (r43 * c34 -
				c1345) * tmp2 *
		    u_3[0].y /*u[i - 1][j][k][1] */ ;
		c[i][j][4][2] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_3[0].z /*u[i - 1][j][k][2] */  *
				  u_3[0].y /*u[i - 1][j][k][1] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 *
		    u_3[0].z /*u[i - 1][j][k][2] */ ;
		c[i][j][4][3] =
		    -dt * tx2 * (-0.40e+00 *
				 (u_3[0].w /*u[i - 1][j][k][3] */  *
				  u_3[0].y /*u[i - 1][j][k][1] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 *
		    u_3[0].w /*u[i - 1][j][k][3] */ ;
		c[i][j][4][4] =
		    -dt * tx2 * (1.40e+00 *
				 (u_3[0].y /*u[i - 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1643 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void jacu_0(__global double *g_u, int k, __global double *g_d,
		     double dt, double tx1, double dx1, double ty1, double dy1,
		     double tz1, double dz1, double r43, double c34, double dx2,
		     double dy2, double dz2, double dx3, double dy3, double dz3,
		     double dx4, double dy4, double dz4, double c1345,
		     double dx5, double dy5, double dz5, __global double *g_a,
		     double tx2, __global double *g_b, double ty2,
		     __global double *g_c, double tz2, int jst, int ist,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	double tmp1;		/* (User-defined privated variables) : Defined at lu.c : 1630 */
	double tmp2;		/* (User-defined privated variables) : Defined at lu.c : 1630 */
	double tmp3;		/* (User-defined privated variables) : Defined at lu.c : 1630 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*d)[64][5][5] = (__global double (*)[64][5][5])g_d;
	__global double (*a)[64][5][5] = (__global double (*)[64][5][5])g_a;
	__global double (*b)[64][5][5] = (__global double (*)[64][5][5])g_b;
	__global double (*c)[64][5][5] = (__global double (*)[64][5][5])g_c;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1644
		//-------------------------------------------
		double4 u_8[4];
		double u_9[2];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1644
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//      u[i][j][k + 1][0]
		//      u[i][j][k + 1][1]
		//      u[i][j][k + 1][2]
		//      u[i][j][k + 1][3]
		//      u[i][j + 1][k][0]
		//      u[i][j + 1][k][1]
		//      u[i][j + 1][k][2]
		//      u[i][j + 1][k][3]
		//      u[i + 1][j][k][0]
		//      u[i + 1][j][k][1]
		//      u[i + 1][j][k][2]
		//      u[i + 1][j][k][3]
		//      u[i][j][k + 1][4]
		//      u[i + 1][j][k][4]
		//-------------------------------------------
		__global double *p_u_8_0 = (__global double *)&u[i][j][k][0];
		if ((unsigned long)p_u_8_0 % 64 == 0) {
			u_8[0] = vload4(0, p_u_8_0);
		} else {
			u_8[0].x = p_u_8_0[0];
			p_u_8_0++;
			u_8[0].y = p_u_8_0[0];
			p_u_8_0++;
			u_8[0].z = p_u_8_0[0];
			p_u_8_0++;
			u_8[0].w = p_u_8_0[0];
			p_u_8_0++;
		}
		__global double *p_u_8_1 =
		    (__global double *)&u[i][j][k + 1][0];
		if ((unsigned long)p_u_8_1 % 64 == 0) {
			u_8[1] = vload4(0, p_u_8_1);
		} else {
			u_8[1].x = p_u_8_1[0];
			p_u_8_1++;
			u_8[1].y = p_u_8_1[0];
			p_u_8_1++;
			u_8[1].z = p_u_8_1[0];
			p_u_8_1++;
			u_8[1].w = p_u_8_1[0];
			p_u_8_1++;
		}
		__global double *p_u_8_2 =
		    (__global double *)&u[i][j + 1][k][0];
		if ((unsigned long)p_u_8_2 % 64 == 0) {
			u_8[2] = vload4(0, p_u_8_2);
		} else {
			u_8[2].x = p_u_8_2[0];
			p_u_8_2++;
			u_8[2].y = p_u_8_2[0];
			p_u_8_2++;
			u_8[2].z = p_u_8_2[0];
			p_u_8_2++;
			u_8[2].w = p_u_8_2[0];
			p_u_8_2++;
		}
		__global double *p_u_8_3 =
		    (__global double *)&u[i + 1][j][k][0];
		if ((unsigned long)p_u_8_3 % 64 == 0) {
			u_8[3] = vload4(0, p_u_8_3);
		} else {
			u_8[3].x = p_u_8_3[0];
			p_u_8_3++;
			u_8[3].y = p_u_8_3[0];
			p_u_8_3++;
			u_8[3].z = p_u_8_3[0];
			p_u_8_3++;
			u_8[3].w = p_u_8_3[0];
			p_u_8_3++;
		}
		u_9[0] = u[i][j][k + 1][4];
		u_9[1] = u[i + 1][j][k][4];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u_8[0].x /*u[i][j][k][0] */ ;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		d[i][j][0][0] =
		    1.0 + dt * 2.0 * (tx1 * dx1 + ty1 * dy1 + tz1 * dz1);
		d[i][j][0][1] = 0.0;
		d[i][j][0][2] = 0.0;
		d[i][j][0][3] = 0.0;
		d[i][j][0][4] = 0.0;
		d[i][j][1][0] =
		    dt * 2.0 * (tx1 *
				(-r43 * c34 * tmp2 *
				 u_8[0].y /*u[i][j][k][1] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_8[0].y /*u[i][j][k][1] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_8[0].y /*u[i][j][k][1] */ ));
		d[i][j][1][1] =
		    1.0 + dt * 2.0 * (tx1 * r43 * c34 * tmp1 +
				      ty1 * c34 * tmp1 + tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx2 + ty1 * dy2 + tz1 * dz2);
		d[i][j][1][2] = 0.0;
		d[i][j][1][3] = 0.0;
		d[i][j][1][4] = 0.0;
		d[i][j][2][0] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_8[0].z /*u[i][j][k][2] */ ) +
				ty1 * (-r43 * c34 * tmp2 *
				       u_8[0].z /*u[i][j][k][2] */ ) +
				tz1 * (-c34 * tmp2 *
				       u_8[0].z /*u[i][j][k][2] */ ));
		d[i][j][2][1] = 0.0;
		d[i][j][2][2] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 +
				      ty1 * r43 * c34 * tmp1 +
				      tz1 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx3 + ty1 * dy3 + tz1 * dz3);
		d[i][j][2][3] = 0.0;
		d[i][j][2][4] = 0.0;
		d[i][j][3][0] =
		    dt * 2.0 * (tx1 *
				(-c34 * tmp2 * u_8[0].w /*u[i][j][k][3] */ ) +
				ty1 * (-c34 * tmp2 *
				       u_8[0].w /*u[i][j][k][3] */ ) +
				tz1 * (-r43 * c34 * tmp2 *
				       u_8[0].w /*u[i][j][k][3] */ ));
		d[i][j][3][1] = 0.0;
		d[i][j][3][2] = 0.0;
		d[i][j][3][3] =
		    1.0 + dt * 2.0 * (tx1 * c34 * tmp1 + ty1 * c34 * tmp1 +
				      tz1 * r43 * c34 * tmp1) +
		    dt * 2.0 * (tx1 * dx4 + ty1 * dy4 + tz1 * dz4);
		d[i][j][3][4] = 0.0;
		d[i][j][4][0] =
		    dt * 2.0 * (tx1 *
				(-(r43 * c34 - c1345) * tmp3 *
				 (((u_8[0].y /*u[i][j][k][1] */ ) *
				   (u_8[0].y /*u[i][j][k][1] */ ))) - (c34 -
								       c1345) *
				 tmp3 *
				 (((u_8[0].z /*u[i][j][k][2] */ ) *
				   (u_8[0].z /*u[i][j][k][2] */ ))) - (c34 -
								       c1345) *
				 tmp3 *
				 (((u_8[0].w /*u[i][j][k][3] */ ) *
				   (u_8[0].w /*u[i][j][k][3] */ ))) -
				 (c1345) * tmp2 * u[i][j][k][4]) +
				ty1 * (-(c34 - c1345) * tmp3 *
				       (((u_8[0].y /*u[i][j][k][1] */ ) *
					 (u_8[0].y /*u[i][j][k][1] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_8[0].z /*u[i][j][k][2] */ ) *
					 (u_8[0].z /*u[i][j][k][2] */ ))) -
				       (c34 -
					c1345) * tmp3 *
				       (((u_8[0].w /*u[i][j][k][3] */ ) *
					 (u_8[0].w /*u[i][j][k][3] */ ))) -
				       (c1345) * tmp2 * u[i][j][k][4]) +
				tz1 * (-(c34 - c1345) * tmp3 *
				       (((u_8[0].y /*u[i][j][k][1] */ ) *
					 (u_8[0].y /*u[i][j][k][1] */ ))) -
				       (c34 -
					c1345) * tmp3 *
				       (((u_8[0].z /*u[i][j][k][2] */ ) *
					 (u_8[0].z /*u[i][j][k][2] */ ))) -
				       (r43 * c34 -
					c1345) * tmp3 *
				       (((u_8[0].w /*u[i][j][k][3] */ ) *
					 (u_8[0].w /*u[i][j][k][3] */ ))) -
				       (c1345) * tmp2 * u[i][j][k][4]));
		d[i][j][4][1] =
		    dt * 2.0 * (tx1 * (r43 * c34 - c1345) * tmp2 *
				u_8[0].y /*u[i][j][k][1] */  + ty1 * (c34 -
								      c1345) *
				tmp2 * u_8[0].y /*u[i][j][k][1] */  +
				tz1 * (c34 -
				       c1345) * tmp2 *
				u_8[0].y /*u[i][j][k][1] */ );
		d[i][j][4][2] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_8[0].z /*u[i][j][k][2] */  +
				ty1 * (r43 * c34 -
				       c1345) * tmp2 *
				u_8[0].z /*u[i][j][k][2] */  + tz1 * (c34 -
								      c1345) *
				tmp2 * u_8[0].z /*u[i][j][k][2] */ );
		d[i][j][4][3] =
		    dt * 2.0 * (tx1 * (c34 - c1345) * tmp2 *
				u_8[0].w /*u[i][j][k][3] */  + ty1 * (c34 -
								      c1345) *
				tmp2 * u_8[0].w /*u[i][j][k][3] */  +
				tz1 * (r43 * c34 -
				       c1345) * tmp2 *
				u_8[0].w /*u[i][j][k][3] */ );
		d[i][j][4][4] =
		    1.0 + dt * 2.0 * (tx1 * c1345 * tmp1 + ty1 * c1345 * tmp1 +
				      tz1 * c1345 * tmp1) +
		    dt * 2.0 * (tx1 * dx5 + ty1 * dy5 + tz1 * dz5);
		tmp1 = 1.0 / u_8[3].x /*u[i + 1][j][k][0] */ ;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		a[i][j][0][0] = -dt * tx1 * dx1;
		a[i][j][0][1] = dt * tx2;
		a[i][j][0][2] = 0.0;
		a[i][j][0][3] = 0.0;
		a[i][j][0][4] = 0.0;
		a[i][j][1][0] =
		    dt * tx2 * (-(u_8[3].y /*u[i + 1][j][k][1] */  * tmp1) *
				(u_8[3].y /*u[i + 1][j][k][1] */  * tmp1) +
				0.40e+00 * 0.50 *
				(u_8[3].y /*u[i + 1][j][k][1] */  *
				 u_8[3].y /*u[i + 1][j][k][1] */  +
				 u_8[3].z /*u[i + 1][j][k][2] */  *
				 u_8[3].z /*u[i + 1][j][k][2] */  +
				 u_8[3].w /*u[i + 1][j][k][3] */  *
				 u_8[3].w /*u[i + 1][j][k][3] */ ) * tmp2) -
		    dt * tx1 * (-r43 * c34 * tmp2 *
				u_8[3].y /*u[i + 1][j][k][1] */ );
		a[i][j][1][1] =
		    dt * tx2 * ((2.0 - 0.40e+00) *
				(u_8[3].y /*u[i + 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * (r43 * c34 * tmp1) - dt * tx1 * dx2;
		a[i][j][1][2] =
		    dt * tx2 * (-0.40e+00 *
				(u_8[3].z /*u[i + 1][j][k][2] */  * tmp1));
		a[i][j][1][3] =
		    dt * tx2 * (-0.40e+00 *
				(u_8[3].w /*u[i + 1][j][k][3] */  * tmp1));
		a[i][j][1][4] = dt * tx2 * 0.40e+00;
		a[i][j][2][0] =
		    dt * tx2 *
		    (-
		     (u_8[3].y /*u[i + 1][j][k][1] */  *
		      u_8[3].z /*u[i + 1][j][k][2] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_8[3].z /*u[i + 1][j][k][2] */ );
		a[i][j][2][1] =
		    dt * tx2 * (u_8[3].z /*u[i + 1][j][k][2] */  * tmp1);
		a[i][j][2][2] =
		    dt * tx2 * (u_8[3].y /*u[i + 1][j][k][1] */  * tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx3;
		a[i][j][2][3] = 0.0;
		a[i][j][2][4] = 0.0;
		a[i][j][3][0] =
		    dt * tx2 *
		    (-
		     (u_8[3].y /*u[i + 1][j][k][1] */  *
		      u_8[3].w /*u[i + 1][j][k][3] */ ) * tmp2) -
		    dt * tx1 * (-c34 * tmp2 * u_8[3].w /*u[i + 1][j][k][3] */ );
		a[i][j][3][1] =
		    dt * tx2 * (u_8[3].w /*u[i + 1][j][k][3] */  * tmp1);
		a[i][j][3][2] = 0.0;
		a[i][j][3][3] =
		    dt * tx2 * (u_8[3].y /*u[i + 1][j][k][1] */  * tmp1) -
		    dt * tx1 * (c34 * tmp1) - dt * tx1 * dx4;
		a[i][j][3][4] = 0.0;
		a[i][j][4][0] =
		    dt * tx2 *
		    ((0.40e+00 *
		      (u_8[3].y /*u[i + 1][j][k][1] */  *
		       u_8[3].y /*u[i + 1][j][k][1] */  +
		       u_8[3].z /*u[i + 1][j][k][2] */  *
		       u_8[3].z /*u[i + 1][j][k][2] */  +
		       u_8[3].w /*u[i + 1][j][k][3] */  *
		       u_8[3].w /*u[i + 1][j][k][3] */ ) * tmp2 -
		      1.40e+00 * (u_9[1] /*u[i + 1][j][k][4] */ *tmp1)) *
		     (u_8[3].y /*u[i + 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * (-(r43 * c34 - c1345) * tmp3 *
				(((u_8[3].y /*u[i + 1][j][k][1] */ ) *
				  (u_8[3].y /*u[i + 1][j][k][1] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_8[3].z /*u[i + 1][j][k][2] */ ) *
				  (u_8[3].z /*u[i + 1][j][k][2] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_8[3].w /*u[i + 1][j][k][3] */ ) *
				  (u_8[3].w /*u[i + 1][j][k][3] */ ))) -
				c1345 * tmp2 * u_9[1] /*u[i + 1][j][k][4] */ );
		a[i][j][4][1] =
		    dt * tx2 * (1.40e+00 *
				(u_9[1] /*u[i + 1][j][k][4] */ *tmp1) -
				0.50 * 0.40e+00 *
				((3.0 * u_8[3].y /*u[i + 1][j][k][1] */  *
				  u_8[3].y /*u[i + 1][j][k][1] */  +
				  u_8[3].z /*u[i + 1][j][k][2] */  *
				  u_8[3].z /*u[i + 1][j][k][2] */  +
				  u_8[3].w /*u[i + 1][j][k][3] */  *
				  u_8[3].w /*u[i + 1][j][k][3] */ ) * tmp2)) -
		    dt * tx1 * (r43 * c34 -
				c1345) * tmp2 *
		    u_8[3].y /*u[i + 1][j][k][1] */ ;
		a[i][j][4][2] =
		    dt * tx2 * (-0.40e+00 *
				(u_8[3].z /*u[i + 1][j][k][2] */  *
				 u_8[3].y /*u[i + 1][j][k][1] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 *
		    u_8[3].z /*u[i + 1][j][k][2] */ ;
		a[i][j][4][3] =
		    dt * tx2 * (-0.40e+00 *
				(u_8[3].w /*u[i + 1][j][k][3] */  *
				 u_8[3].y /*u[i + 1][j][k][1] */ ) * tmp2) -
		    dt * tx1 * (c34 -
				c1345) * tmp2 *
		    u_8[3].w /*u[i + 1][j][k][3] */ ;
		a[i][j][4][4] =
		    dt * tx2 * (1.40e+00 *
				(u_8[3].y /*u[i + 1][j][k][1] */  * tmp1)) -
		    dt * tx1 * c1345 * tmp1 - dt * tx1 * dx5;
		tmp1 = 1.0 / u_8[2].x /*u[i][j + 1][k][0] */ ;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		b[i][j][0][0] = -dt * ty1 * dy1;
		b[i][j][0][1] = 0.0;
		b[i][j][0][2] = dt * ty2;
		b[i][j][0][3] = 0.0;
		b[i][j][0][4] = 0.0;
		b[i][j][1][0] =
		    dt * ty2 *
		    (-
		     (u_8[2].y /*u[i][j + 1][k][1] */  *
		      u_8[2].z /*u[i][j + 1][k][2] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_8[2].y /*u[i][j + 1][k][1] */ );
		b[i][j][1][1] =
		    dt * ty2 * (u_8[2].z /*u[i][j + 1][k][2] */  * tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy2;
		b[i][j][1][2] =
		    dt * ty2 * (u_8[2].y /*u[i][j + 1][k][1] */  * tmp1);
		b[i][j][1][3] = 0.0;
		b[i][j][1][4] = 0.0;
		b[i][j][2][0] =
		    dt * ty2 * (-(u_8[2].z /*u[i][j + 1][k][2] */  * tmp1) *
				(u_8[2].z /*u[i][j + 1][k][2] */  * tmp1) +
				0.50 * 0.40e+00 *
				((u_8[2].y /*u[i][j + 1][k][1] */  *
				  u_8[2].y /*u[i][j + 1][k][1] */  +
				  u_8[2].z /*u[i][j + 1][k][2] */  *
				  u_8[2].z /*u[i][j + 1][k][2] */  +
				  u_8[2].w /*u[i][j + 1][k][3] */  *
				  u_8[2].w /*u[i][j + 1][k][3] */ ) * tmp2)) -
		    dt * ty1 * (-r43 * c34 * tmp2 *
				u_8[2].z /*u[i][j + 1][k][2] */ );
		b[i][j][2][1] =
		    dt * ty2 * (-0.40e+00 *
				(u_8[2].y /*u[i][j + 1][k][1] */  * tmp1));
		b[i][j][2][2] =
		    dt * ty2 * ((2.0 - 0.40e+00) *
				(u_8[2].z /*u[i][j + 1][k][2] */  * tmp1)) -
		    dt * ty1 * (r43 * c34 * tmp1) - dt * ty1 * dy3;
		b[i][j][2][3] =
		    dt * ty2 * (-0.40e+00 *
				(u_8[2].w /*u[i][j + 1][k][3] */  * tmp1));
		b[i][j][2][4] = dt * ty2 * 0.40e+00;
		b[i][j][3][0] =
		    dt * ty2 *
		    (-
		     (u_8[2].z /*u[i][j + 1][k][2] */  *
		      u_8[2].w /*u[i][j + 1][k][3] */ ) * tmp2) -
		    dt * ty1 * (-c34 * tmp2 * u_8[2].w /*u[i][j + 1][k][3] */ );
		b[i][j][3][1] = 0.0;
		b[i][j][3][2] =
		    dt * ty2 * (u_8[2].w /*u[i][j + 1][k][3] */  * tmp1);
		b[i][j][3][3] =
		    dt * ty2 * (u_8[2].z /*u[i][j + 1][k][2] */  * tmp1) -
		    dt * ty1 * (c34 * tmp1) - dt * ty1 * dy4;
		b[i][j][3][4] = 0.0;
		b[i][j][4][0] =
		    dt * ty2 *
		    ((0.40e+00 *
		      (u_8[2].y /*u[i][j + 1][k][1] */  *
		       u_8[2].y /*u[i][j + 1][k][1] */  +
		       u_8[2].z /*u[i][j + 1][k][2] */  *
		       u_8[2].z /*u[i][j + 1][k][2] */  +
		       u_8[2].w /*u[i][j + 1][k][3] */  *
		       u_8[2].w /*u[i][j + 1][k][3] */ ) * tmp2 -
		      1.40e+00 * (u[i][j + 1][k][4] * tmp1)) *
		     (u_8[2].z /*u[i][j + 1][k][2] */  * tmp1)) -
		    dt * ty1 * (-(c34 - c1345) * tmp3 *
				(((u_8[2].y /*u[i][j + 1][k][1] */ ) *
				  (u_8[2].y /*u[i][j + 1][k][1] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_8[2].z /*u[i][j + 1][k][2] */ ) *
				  (u_8[2].z /*u[i][j + 1][k][2] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_8[2].w /*u[i][j + 1][k][3] */ ) *
				  (u_8[2].w /*u[i][j + 1][k][3] */ ))) -
				c1345 * tmp2 * u[i][j + 1][k][4]);
		b[i][j][4][1] =
		    dt * ty2 * (-0.40e+00 *
				(u_8[2].y /*u[i][j + 1][k][1] */  *
				 u_8[2].z /*u[i][j + 1][k][2] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 *
		    u_8[2].y /*u[i][j + 1][k][1] */ ;
		b[i][j][4][2] =
		    dt * ty2 * (1.40e+00 * (u[i][j + 1][k][4] * tmp1) -
				0.50 * 0.40e+00 *
				((u_8[2].y /*u[i][j + 1][k][1] */  *
				  u_8[2].y /*u[i][j + 1][k][1] */  +
				  3.0 * u_8[2].z /*u[i][j + 1][k][2] */  *
				  u_8[2].z /*u[i][j + 1][k][2] */  +
				  u_8[2].w /*u[i][j + 1][k][3] */  *
				  u_8[2].w /*u[i][j + 1][k][3] */ ) * tmp2)) -
		    dt * ty1 * (r43 * c34 -
				c1345) * tmp2 *
		    u_8[2].z /*u[i][j + 1][k][2] */ ;
		b[i][j][4][3] =
		    dt * ty2 * (-0.40e+00 *
				(u_8[2].z /*u[i][j + 1][k][2] */  *
				 u_8[2].w /*u[i][j + 1][k][3] */ ) * tmp2) -
		    dt * ty1 * (c34 -
				c1345) * tmp2 *
		    u_8[2].w /*u[i][j + 1][k][3] */ ;
		b[i][j][4][4] =
		    dt * ty2 * (1.40e+00 *
				(u_8[2].z /*u[i][j + 1][k][2] */  * tmp1)) -
		    dt * ty1 * c1345 * tmp1 - dt * ty1 * dy5;
		tmp1 = 1.0 / u_8[1].x /*u[i][j][k + 1][0] */ ;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		c[i][j][0][0] = -dt * tz1 * dz1;
		c[i][j][0][1] = 0.0;
		c[i][j][0][2] = 0.0;
		c[i][j][0][3] = dt * tz2;
		c[i][j][0][4] = 0.0;
		c[i][j][1][0] =
		    dt * tz2 *
		    (-
		     (u_8[1].y /*u[i][j][k + 1][1] */  *
		      u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_8[1].y /*u[i][j][k + 1][1] */ );
		c[i][j][1][1] =
		    dt * tz2 * (u_8[1].w /*u[i][j][k + 1][3] */  * tmp1) -
		    dt * tz1 * c34 * tmp1 - dt * tz1 * dz2;
		c[i][j][1][2] = 0.0;
		c[i][j][1][3] =
		    dt * tz2 * (u_8[1].y /*u[i][j][k + 1][1] */  * tmp1);
		c[i][j][1][4] = 0.0;
		c[i][j][2][0] =
		    dt * tz2 *
		    (-
		     (u_8[1].z /*u[i][j][k + 1][2] */  *
		      u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2) -
		    dt * tz1 * (-c34 * tmp2 * u_8[1].z /*u[i][j][k + 1][2] */ );
		c[i][j][2][1] = 0.0;
		c[i][j][2][2] =
		    dt * tz2 * (u_8[1].w /*u[i][j][k + 1][3] */  * tmp1) -
		    dt * tz1 * (c34 * tmp1) - dt * tz1 * dz3;
		c[i][j][2][3] =
		    dt * tz2 * (u_8[1].z /*u[i][j][k + 1][2] */  * tmp1);
		c[i][j][2][4] = 0.0;
		c[i][j][3][0] =
		    dt * tz2 * (-(u_8[1].w /*u[i][j][k + 1][3] */  * tmp1) *
				(u_8[1].w /*u[i][j][k + 1][3] */  * tmp1) +
				0.50 * 0.40e+00 *
				((u_8[1].y /*u[i][j][k + 1][1] */  *
				  u_8[1].y /*u[i][j][k + 1][1] */  +
				  u_8[1].z /*u[i][j][k + 1][2] */  *
				  u_8[1].z /*u[i][j][k + 1][2] */  +
				  u_8[1].w /*u[i][j][k + 1][3] */  *
				  u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2)) -
		    dt * tz1 * (-r43 * c34 * tmp2 *
				u_8[1].w /*u[i][j][k + 1][3] */ );
		c[i][j][3][1] =
		    dt * tz2 * (-0.40e+00 *
				(u_8[1].y /*u[i][j][k + 1][1] */  * tmp1));
		c[i][j][3][2] =
		    dt * tz2 * (-0.40e+00 *
				(u_8[1].z /*u[i][j][k + 1][2] */  * tmp1));
		c[i][j][3][3] =
		    dt * tz2 * (2.0 -
				0.40e+00) * (u_8[1].w /*u[i][j][k + 1][3] */  *
					     tmp1) -
		    dt * tz1 * (r43 * c34 * tmp1) - dt * tz1 * dz4;
		c[i][j][3][4] = dt * tz2 * 0.40e+00;
		c[i][j][4][0] =
		    dt * tz2 *
		    ((0.40e+00 *
		      (u_8[1].y /*u[i][j][k + 1][1] */  *
		       u_8[1].y /*u[i][j][k + 1][1] */  +
		       u_8[1].z /*u[i][j][k + 1][2] */  *
		       u_8[1].z /*u[i][j][k + 1][2] */  +
		       u_8[1].w /*u[i][j][k + 1][3] */  *
		       u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2 -
		      1.40e+00 * (u_9[0] /*u[i][j][k + 1][4] */ *tmp1)) *
		     (u_8[1].w /*u[i][j][k + 1][3] */  * tmp1)) -
		    dt * tz1 * (-(c34 - c1345) * tmp3 *
				(((u_8[1].y /*u[i][j][k + 1][1] */ ) *
				  (u_8[1].y /*u[i][j][k + 1][1] */ ))) - (c34 -
									  c1345)
				* tmp3 *
				(((u_8[1].z /*u[i][j][k + 1][2] */ ) *
				  (u_8[1].z /*u[i][j][k + 1][2] */ ))) -
				(r43 * c34 -
				 c1345) * tmp3 *
				(((u_8[1].w /*u[i][j][k + 1][3] */ ) *
				  (u_8[1].w /*u[i][j][k + 1][3] */ ))) -
				c1345 * tmp2 * u_9[0] /*u[i][j][k + 1][4] */ );
		c[i][j][4][1] =
		    dt * tz2 * (-0.40e+00 *
				(u_8[1].y /*u[i][j][k + 1][1] */  *
				 u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 *
		    u_8[1].y /*u[i][j][k + 1][1] */ ;
		c[i][j][4][2] =
		    dt * tz2 * (-0.40e+00 *
				(u_8[1].z /*u[i][j][k + 1][2] */  *
				 u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2) -
		    dt * tz1 * (c34 -
				c1345) * tmp2 *
		    u_8[1].z /*u[i][j][k + 1][2] */ ;
		c[i][j][4][3] =
		    dt * tz2 * (1.40e+00 *
				(u_9[0] /*u[i][j][k + 1][4] */ *tmp1) -
				0.50 * 0.40e+00 *
				((u_8[1].y /*u[i][j][k + 1][1] */  *
				  u_8[1].y /*u[i][j][k + 1][1] */  +
				  u_8[1].z /*u[i][j][k + 1][2] */  *
				  u_8[1].z /*u[i][j][k + 1][2] */  +
				  3.0 * u_8[1].w /*u[i][j][k + 1][3] */  *
				  u_8[1].w /*u[i][j][k + 1][3] */ ) * tmp2)) -
		    dt * tz1 * (r43 * c34 -
				c1345) * tmp2 *
		    u_8[1].w /*u[i][j][k + 1][3] */ ;
		c[i][j][4][4] =
		    dt * tz2 * (1.40e+00 *
				(u_8[1].w /*u[i][j][k + 1][3] */  * tmp1)) -
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
//Loop defined at line 2020 of lu.c
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*v)[65][65][5] = (__global double (*)[65][65][5])g_v;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double sum0 = 0.0;	/* reduction variable, defined at: lu.c : 2011 */
	double sum1 = 0.0;	/* reduction variable, defined at: lu.c : 2011 */
	double sum2 = 0.0;	/* reduction variable, defined at: lu.c : 2011 */
	double sum3 = 0.0;	/* reduction variable, defined at: lu.c : 2011 */
	double sum4 = 0.0;	/* reduction variable, defined at: lu.c : 2011 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2022
		//-------------------------------------------
		double4 v_3;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2022
		//Candidates:
		//      v[i][j][k][0]
		//      v[i][j][k][1]
		//      v[i][j][k][2]
		//      v[i][j][k][3]
		//-------------------------------------------
		__global double *p_v_3_0 = (__global double *)&v[i][j][k][0];
		if ((unsigned long)p_v_3_0 % 64 == 0) {
			v_3 = vload4(0, p_v_3_0);
		} else {
			v_3.x = p_v_3_0[0];
			p_v_3_0++;
			v_3.y = p_v_3_0[0];
			p_v_3_0++;
			v_3.z = p_v_3_0[0];
			p_v_3_0++;
			v_3.w = p_v_3_0[0];
			p_v_3_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		sum0 =
		    sum0 +
		    v_3.x /*v[i][j][k][0] */  * v_3.x /*v[i][j][k][0] */ ;
		sum1 =
		    sum1 +
		    v_3.y /*v[i][j][k][1] */  * v_3.y /*v[i][j][k][1] */ ;
		sum2 =
		    sum2 +
		    v_3.z /*v[i][j][k][2] */  * v_3.z /*v[i][j][k][2] */ ;
		sum3 =
		    sum3 +
		    v_3.w /*v[i][j][k][3] */  * v_3.w /*v[i][j][k][3] */ ;
		sum4 = sum4 + v[i][j][k][4] * v[i][j][k][4];
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
//Loop defined at line 2364 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_0(__global double *g_rsd, __global double *g_frct,
		    int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		    __global int *tls_validflag)
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
	int m;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	__global double (*frct)[65][65][5] =
	    (__global double (*)[65][65][5])g_frct;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rsd[i][j][k][m] = -frct[i][j][k][m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2382 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_1(__global double *g_flux, __global double *g_u, int jst,
		    int L1, int __ocl_k_bound, int __ocl_j_bound,
		    int __ocl_i_bound, __global int *tls_validflag)
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
	double u21;		/* (User-defined privated variables) : Defined at lu.c : 2353 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 2352 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2384
		//-------------------------------------------
		double4 u_11;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2384
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//-------------------------------------------
		__global double *p_u_11_0 = (__global double *)&u[i][j][k][0];
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

		flux[i][j][k][0] = u_11.y /*u[i][j][k][1] */ ;
		u21 = u_11.y /*u[i][j][k][1] */  / u_11.x /*u[i][j][k][0] */ ;
		q = 0.50 * (u_11.y /*u[i][j][k][1] */  *
			    u_11.y /*u[i][j][k][1] */  +
			    u_11.z /*u[i][j][k][2] */  *
			    u_11.z /*u[i][j][k][2] */  +
			    u_11.w /*u[i][j][k][3] */  *
			    u_11.w /*u[i][j][k][3] */ ) /
		    u_11.x /*u[i][j][k][0] */ ;
		flux[i][j][k][1] =
		    u_11.y /*u[i][j][k][1] */  * u21 +
		    0.40e+00 * (u[i][j][k][4] - q);
		flux[i][j][k][2] = u_11.z /*u[i][j][k][2] */  * u21;
		flux[i][j][k][3] = u_11.w /*u[i][j][k][3] */  * u21;
		flux[i][j][k][4] =
		    (1.40e+00 * u[i][j][k][4] - 0.40e+00 * q) * u21;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2403 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_2(int ist, int iend, __global double *g_rsd, double tx2,
		    __global double *g_flux, int nx, __global double *g_u,
		    double tx3, double dx1, double tx1, double dx2, double dx3,
		    double dx4, double dx5, double dssp, int jst,
		    int __ocl_k_bound, int __ocl_j_bound,
		    __global int *tls_validflag)
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
	int i;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	int L2;			/* (User-defined privated variables) : Defined at lu.c : 2349 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 2354 */
	double u21i;		/* (User-defined privated variables) : Defined at lu.c : 2355 */
	double u31i;		/* (User-defined privated variables) : Defined at lu.c : 2355 */
	double u41i;		/* (User-defined privated variables) : Defined at lu.c : 2355 */
	double u51i;		/* (User-defined privated variables) : Defined at lu.c : 2355 */
	double u21im1;		/* (User-defined privated variables) : Defined at lu.c : 2358 */
	double u31im1;		/* (User-defined privated variables) : Defined at lu.c : 2358 */
	double u41im1;		/* (User-defined privated variables) : Defined at lu.c : 2358 */
	double u51im1;		/* (User-defined privated variables) : Defined at lu.c : 2358 */
	int ist1;		/* (User-defined privated variables) : Defined at lu.c : 2350 */
	int iend1;		/* (User-defined privated variables) : Defined at lu.c : 2350 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = ist; i <= iend; i++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    tx2 * (flux[i + 1][j][k][m] -
					   flux[i - 1][j][k][m]);
			}
		}
		L2 = nx - 1;
		for (i = ist; i <= L2; i++) {
			tmp = 1.0 / u[i][j][k][0];
			u21i = tmp * u[i][j][k][1];
			u31i = tmp * u[i][j][k][2];
			u41i = tmp * u[i][j][k][3];
			u51i = tmp * u[i][j][k][4];
			tmp = 1.0 / u[i - 1][j][k][0];
			u21im1 = tmp * u[i - 1][j][k][1];
			u31im1 = tmp * u[i - 1][j][k][2];
			u41im1 = tmp * u[i - 1][j][k][3];
			u51im1 = tmp * u[i - 1][j][k][4];
			flux[i][j][k][1] = (4.0 / 3.0) * tx3 * (u21i - u21im1);
			flux[i][j][k][2] = tx3 * (u31i - u31im1);
			flux[i][j][k][3] = tx3 * (u41i - u41im1);
			flux[i][j][k][4] =
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
			rsd[i][j][k][0] =
			    rsd[i][j][k][0] + dx1 * tx1 * (u[i - 1][j][k][0] -
							   2.0 * u[i][j][k][0] +
							   u[i + 1][j][k][0]);
			rsd[i][j][k][1] =
			    rsd[i][j][k][1] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][1] -
							 flux[i][j][k][1]) +
			    dx2 * tx1 * (u[i - 1][j][k][1] -
					 2.0 * u[i][j][k][1] + u[i +
								 1][j][k][1]);
			rsd[i][j][k][2] =
			    rsd[i][j][k][2] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][2] -
							 flux[i][j][k][2]) +
			    dx3 * tx1 * (u[i - 1][j][k][2] -
					 2.0 * u[i][j][k][2] + u[i +
								 1][j][k][2]);
			rsd[i][j][k][3] =
			    rsd[i][j][k][3] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][3] -
							 flux[i][j][k][3]) +
			    dx4 * tx1 * (u[i - 1][j][k][3] -
					 2.0 * u[i][j][k][3] + u[i +
								 1][j][k][3]);
			rsd[i][j][k][4] =
			    rsd[i][j][k][4] +
			    tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][j][k][4] -
							 flux[i][j][k][4]) +
			    dx5 * tx1 * (u[i - 1][j][k][4] -
					 2.0 * u[i][j][k][4] + u[i +
								 1][j][k][4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2470
			//-------------------------------------------
			double u_14[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2470
			//Candidates:
			//      u[1][j][k][m]
			//      u[2][j][k][m]
			//      u[3][j][k][m]
			//-------------------------------------------
			u_14[0] = u[1][j][k][m];
			u_14[1] = u[2][j][k][m];
			u_14[2] = u[3][j][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[1][j][k][m] =
			    rsd[1][j][k][m] -
			    dssp * (+5.0 * u_14[0] /*u[1][j][k][m] */ -4.0 *
				    u_14[1] /*u[2][j][k][m] */ +u_14[2]
				    /*u[3][j][k][m] */ );
			rsd[2][j][k][m] =
			    rsd[2][j][k][m] -
			    dssp * (-4.0 * u_14[0] /*u[1][j][k][m] */ +6.0 *
				    u_14[1] /*u[2][j][k][m] */ -4.0 *
				    u_14[2] /*u[3][j][k][m] */ +u[4][j][k][m]);
		}
		ist1 = 3;
		iend1 = nx - 4;
		for (i = ist1; i <= iend1; i++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    dssp * (u[i - 2][j][k][m] -
					    4.0 * u[i - 1][j][k][m] +
					    6.0 * u[i][j][k][m] - 4.0 * u[i +
									  1][j]
					    [k][m] + u[i + 2][j][k][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2497
			//-------------------------------------------
			double u_15[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2497
			//Candidates:
			//      u[nx - 4][j][k][m]
			//      u[nx - 3][j][k][m]
			//-------------------------------------------
			u_15[0] = u[nx - 4][j][k][m];
			u_15[1] = u[nx - 3][j][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[nx - 3][j][k][m] =
			    rsd[nx - 3][j][k][m] - dssp * (u[nx - 5][j][k][m] -
							   4.0 *
							   u_15[0]
							   /*u[nx - 4][j][k][m] */
							   +6.0 *
							   u_15[1]
							   /*u[nx - 3][j][k][m] */
							   -4.0 * u[nx -
								    2][j][k]
							   [m]);
			rsd[nx - 2][j][k][m] =
			    rsd[nx - 2][j][k][m] -
			    dssp * (u_15[0] /*u[nx - 4][j][k][m] */ -4.0 *
				    u_15[1] /*u[nx - 3][j][k][m] */ +5.0 *
				    u[nx - 2][j][k][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2520 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_3(__global double *g_flux, __global double *g_u, int L1,
		    int ist, int __ocl_k_bound, int __ocl_j_bound,
		    int __ocl_i_bound, __global int *tls_validflag)
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
	double u31;		/* (User-defined privated variables) : Defined at lu.c : 2353 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 2352 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2522
		//-------------------------------------------
		double4 u_17;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2522
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//-------------------------------------------
		__global double *p_u_17_0 = (__global double *)&u[i][j][k][0];
		if ((unsigned long)p_u_17_0 % 64 == 0) {
			u_17 = vload4(0, p_u_17_0);
		} else {
			u_17.x = p_u_17_0[0];
			p_u_17_0++;
			u_17.y = p_u_17_0[0];
			p_u_17_0++;
			u_17.z = p_u_17_0[0];
			p_u_17_0++;
			u_17.w = p_u_17_0[0];
			p_u_17_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		flux[i][j][k][0] = u_17.z /*u[i][j][k][2] */ ;
		u31 = u_17.z /*u[i][j][k][2] */  / u_17.x /*u[i][j][k][0] */ ;
		q = 0.50 * (u_17.y /*u[i][j][k][1] */  *
			    u_17.y /*u[i][j][k][1] */  +
			    u_17.z /*u[i][j][k][2] */  *
			    u_17.z /*u[i][j][k][2] */  +
			    u_17.w /*u[i][j][k][3] */  *
			    u_17.w /*u[i][j][k][3] */ ) /
		    u_17.x /*u[i][j][k][0] */ ;
		flux[i][j][k][1] = u_17.y /*u[i][j][k][1] */  * u31;
		flux[i][j][k][2] =
		    u_17.z /*u[i][j][k][2] */  * u31 +
		    0.40e+00 * (u[i][j][k][4] - q);
		flux[i][j][k][3] = u_17.w /*u[i][j][k][3] */  * u31;
		flux[i][j][k][4] =
		    (1.40e+00 * u[i][j][k][4] - 0.40e+00 * q) * u31;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2541 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_4(int jst, int jend, __global double *g_rsd, double ty2,
		    __global double *g_flux, int ny, __global double *g_u,
		    double ty3, double dy1, double ty1, double dy2, double dy3,
		    double dy4, double dy5, double dssp, int ist,
		    int __ocl_k_bound, int __ocl_i_bound,
		    __global int *tls_validflag)
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
	int j;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	int L2;			/* (User-defined privated variables) : Defined at lu.c : 2349 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 2354 */
	double u21j;		/* (User-defined privated variables) : Defined at lu.c : 2356 */
	double u31j;		/* (User-defined privated variables) : Defined at lu.c : 2356 */
	double u41j;		/* (User-defined privated variables) : Defined at lu.c : 2356 */
	double u51j;		/* (User-defined privated variables) : Defined at lu.c : 2356 */
	double u21jm1;		/* (User-defined privated variables) : Defined at lu.c : 2359 */
	double u31jm1;		/* (User-defined privated variables) : Defined at lu.c : 2359 */
	double u41jm1;		/* (User-defined privated variables) : Defined at lu.c : 2359 */
	double u51jm1;		/* (User-defined privated variables) : Defined at lu.c : 2359 */
	int jst1;		/* (User-defined privated variables) : Defined at lu.c : 2351 */
	int jend1;		/* (User-defined privated variables) : Defined at lu.c : 2351 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (j = jst; j <= jend; j++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    ty2 * (flux[i][j + 1][k][m] -
					   flux[i][j - 1][k][m]);
			}
		}
		L2 = ny - 1;
		for (j = jst; j <= L2; j++) {
			tmp = 1.0 / u[i][j][k][0];
			u21j = tmp * u[i][j][k][1];
			u31j = tmp * u[i][j][k][2];
			u41j = tmp * u[i][j][k][3];
			u51j = tmp * u[i][j][k][4];
			tmp = 1.0 / u[i][j - 1][k][0];
			u21jm1 = tmp * u[i][j - 1][k][1];
			u31jm1 = tmp * u[i][j - 1][k][2];
			u41jm1 = tmp * u[i][j - 1][k][3];
			u51jm1 = tmp * u[i][j - 1][k][4];
			flux[i][j][k][1] = ty3 * (u21j - u21jm1);
			flux[i][j][k][2] = (4.0 / 3.0) * ty3 * (u31j - u31jm1);
			flux[i][j][k][3] = ty3 * (u41j - u41jm1);
			flux[i][j][k][4] =
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
			rsd[i][j][k][0] =
			    rsd[i][j][k][0] + dy1 * ty1 * (u[i][j - 1][k][0] -
							   2.0 * u[i][j][k][0] +
							   u[i][j + 1][k][0]);
			rsd[i][j][k][1] =
			    rsd[i][j][k][1] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][1] -
							 flux[i][j][k][1]) +
			    dy2 * ty1 * (u[i][j - 1][k][1] -
					 2.0 * u[i][j][k][1] + u[i][j +
								    1][k][1]);
			rsd[i][j][k][2] =
			    rsd[i][j][k][2] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][2] -
							 flux[i][j][k][2]) +
			    dy3 * ty1 * (u[i][j - 1][k][2] -
					 2.0 * u[i][j][k][2] + u[i][j +
								    1][k][2]);
			rsd[i][j][k][3] =
			    rsd[i][j][k][3] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][3] -
							 flux[i][j][k][3]) +
			    dy4 * ty1 * (u[i][j - 1][k][3] -
					 2.0 * u[i][j][k][3] + u[i][j +
								    1][k][3]);
			rsd[i][j][k][4] =
			    rsd[i][j][k][4] +
			    ty3 * 1.00e-01 * 1.00e+00 * (flux[i][j + 1][k][4] -
							 flux[i][j][k][4]) +
			    dy5 * ty1 * (u[i][j - 1][k][4] -
					 2.0 * u[i][j][k][4] + u[i][j +
								    1][k][4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2612
			//-------------------------------------------
			double u_20[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2612
			//Candidates:
			//      u[i][1][k][m]
			//      u[i][2][k][m]
			//      u[i][3][k][m]
			//-------------------------------------------
			u_20[0] = u[i][1][k][m];
			u_20[1] = u[i][2][k][m];
			u_20[2] = u[i][3][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[i][1][k][m] =
			    rsd[i][1][k][m] -
			    dssp * (+5.0 * u_20[0] /*u[i][1][k][m] */ -4.0 *
				    u_20[1] /*u[i][2][k][m] */ +u_20[2]
				    /*u[i][3][k][m] */ );
			rsd[i][2][k][m] =
			    rsd[i][2][k][m] -
			    dssp * (-4.0 * u_20[0] /*u[i][1][k][m] */ +6.0 *
				    u_20[1] /*u[i][2][k][m] */ -4.0 *
				    u_20[2] /*u[i][3][k][m] */ +u[i][4][k][m]);
		}
		jst1 = 3;
		jend1 = ny - 4;
		for (j = jst1; j <= jend1; j++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    dssp * (u[i][j - 2][k][m] -
					    4.0 * u[i][j - 1][k][m] +
					    6.0 * u[i][j][k][m] - 4.0 * u[i][j +
									     1]
					    [k][m] + u[i][j + 2][k][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2637
			//-------------------------------------------
			double u_21[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2637
			//Candidates:
			//      u[i][ny - 4][k][m]
			//      u[i][ny - 3][k][m]
			//-------------------------------------------
			u_21[0] = u[i][ny - 4][k][m];
			u_21[1] = u[i][ny - 3][k][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[i][ny - 3][k][m] =
			    rsd[i][ny - 3][k][m] - dssp * (u[i][ny - 5][k][m] -
							   4.0 *
							   u_21[0]
							   /*u[i][ny - 4][k][m] */
							   +6.0 *
							   u_21[1]
							   /*u[i][ny - 3][k][m] */
							   -4.0 * u[i][ny -
								       2][k]
							   [m]);
			rsd[i][ny - 2][k][m] =
			    rsd[i][ny - 2][k][m] -
			    dssp * (u_21[0] /*u[i][ny - 4][k][m] */ -4.0 *
				    u_21[1] /*u[i][ny - 3][k][m] */ +5.0 *
				    u[i][ny - 2][k][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2656 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void rhs_5(int nz, __global double *g_flux, __global double *g_u,
		    __global double *g_rsd, double tz2, double tz3, double dz1,
		    double tz1, double dz2, double dz3, double dz4, double dz5,
		    double dssp, int jst, int ist, int __ocl_j_bound,
		    int __ocl_i_bound, __global int *tls_validflag)
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
	int k;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	double u41;		/* (User-defined privated variables) : Defined at lu.c : 2353 */
	double q;		/* (User-defined privated variables) : Defined at lu.c : 2352 */
	int m;			/* (User-defined privated variables) : Defined at lu.c : 2348 */
	double tmp;		/* (User-defined privated variables) : Defined at lu.c : 2354 */
	double u21k;		/* (User-defined privated variables) : Defined at lu.c : 2357 */
	double u31k;		/* (User-defined privated variables) : Defined at lu.c : 2357 */
	double u41k;		/* (User-defined privated variables) : Defined at lu.c : 2357 */
	double u51k;		/* (User-defined privated variables) : Defined at lu.c : 2357 */
	double u21km1;		/* (User-defined privated variables) : Defined at lu.c : 2360 */
	double u31km1;		/* (User-defined privated variables) : Defined at lu.c : 2360 */
	double u41km1;		/* (User-defined privated variables) : Defined at lu.c : 2360 */
	double u51km1;		/* (User-defined privated variables) : Defined at lu.c : 2360 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*flux)[65][65][5] =
	    (__global double (*)[65][65][5])g_flux;
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= nz - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2658
			//-------------------------------------------
			double4 u_25;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2658
			//Candidates:
			//      u[i][j][k][0]
			//      u[i][j][k][1]
			//      u[i][j][k][2]
			//      u[i][j][k][3]
			//-------------------------------------------
			__global double *p_u_25_0 =
			    (__global double *)&u[i][j][k][0];
			if ((unsigned long)p_u_25_0 % 64 == 0) {
				u_25 = vload4(0, p_u_25_0);
			} else {
				u_25.x = p_u_25_0[0];
				p_u_25_0++;
				u_25.y = p_u_25_0[0];
				p_u_25_0++;
				u_25.z = p_u_25_0[0];
				p_u_25_0++;
				u_25.w = p_u_25_0[0];
				p_u_25_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			flux[i][j][k][0] = u_25.w /*u[i][j][k][3] */ ;
			u41 =
			    u_25.w /*u[i][j][k][3] */  /
			    u_25.x /*u[i][j][k][0] */ ;
			q = 0.50 * (u_25.y /*u[i][j][k][1] */  *
				    u_25.y /*u[i][j][k][1] */  +
				    u_25.z /*u[i][j][k][2] */  *
				    u_25.z /*u[i][j][k][2] */  +
				    u_25.w /*u[i][j][k][3] */  *
				    u_25.w /*u[i][j][k][3] */ ) /
			    u_25.x /*u[i][j][k][0] */ ;
			flux[i][j][k][1] = u_25.y /*u[i][j][k][1] */  * u41;
			flux[i][j][k][2] = u_25.z /*u[i][j][k][2] */  * u41;
			flux[i][j][k][3] =
			    u_25.w /*u[i][j][k][3] */  * u41 +
			    0.40e+00 * (u[i][j][k][4] - q);
			flux[i][j][k][4] =
			    (1.40e+00 * u[i][j][k][4] - 0.40e+00 * q) * u41;
		}
		for (k = 1; k <= nz - 2; k++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    tz2 * (flux[i][j][k + 1][m] -
					   flux[i][j][k - 1][m]);
			}
		}
		for (k = 1; k <= nz - 1; k++) {
			tmp = 1.0 / u[i][j][k][0];
			u21k = tmp * u[i][j][k][1];
			u31k = tmp * u[i][j][k][2];
			u41k = tmp * u[i][j][k][3];
			u51k = tmp * u[i][j][k][4];
			tmp = 1.0 / u[i][j][k - 1][0];
			u21km1 = tmp * u[i][j][k - 1][1];
			u31km1 = tmp * u[i][j][k - 1][2];
			u41km1 = tmp * u[i][j][k - 1][3];
			u51km1 = tmp * u[i][j][k - 1][4];
			flux[i][j][k][1] = tz3 * (u21k - u21km1);
			flux[i][j][k][2] = tz3 * (u31k - u31km1);
			flux[i][j][k][3] = (4.0 / 3.0) * tz3 * (u41k - u41km1);
			flux[i][j][k][4] =
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
			rsd[i][j][k][0] =
			    rsd[i][j][k][0] + dz1 * tz1 * (u[i][j][k - 1][0] -
							   2.0 * u[i][j][k][0] +
							   u[i][j][k + 1][0]);
			rsd[i][j][k][1] =
			    rsd[i][j][k][1] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][1] -
							 flux[i][j][k][1]) +
			    dz2 * tz1 * (u[i][j][k - 1][1] -
					 2.0 * u[i][j][k][1] + u[i][j][k +
								       1][1]);
			rsd[i][j][k][2] =
			    rsd[i][j][k][2] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][2] -
							 flux[i][j][k][2]) +
			    dz3 * tz1 * (u[i][j][k - 1][2] -
					 2.0 * u[i][j][k][2] + u[i][j][k +
								       1][2]);
			rsd[i][j][k][3] =
			    rsd[i][j][k][3] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][3] -
							 flux[i][j][k][3]) +
			    dz4 * tz1 * (u[i][j][k - 1][3] -
					 2.0 * u[i][j][k][3] + u[i][j][k +
								       1][3]);
			rsd[i][j][k][4] =
			    rsd[i][j][k][4] +
			    tz3 * 1.00e-01 * 1.00e+00 * (flux[i][j][k + 1][4] -
							 flux[i][j][k][4]) +
			    dz5 * tz1 * (u[i][j][k - 1][4] -
					 2.0 * u[i][j][k][4] + u[i][j][k +
								       1][4]);
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2736
			//-------------------------------------------
			double u_26[3];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2736
			//Candidates:
			//      u[i][j][1][m]
			//      u[i][j][2][m]
			//      u[i][j][3][m]
			//-------------------------------------------
			u_26[0] = u[i][j][1][m];
			u_26[1] = u[i][j][2][m];
			u_26[2] = u[i][j][3][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[i][j][1][m] =
			    rsd[i][j][1][m] -
			    dssp * (+5.0 * u_26[0] /*u[i][j][1][m] */ -4.0 *
				    u_26[1] /*u[i][j][2][m] */ +u_26[2]
				    /*u[i][j][3][m] */ );
			rsd[i][j][2][m] =
			    rsd[i][j][2][m] -
			    dssp * (-4.0 * u_26[0] /*u[i][j][1][m] */ +6.0 *
				    u_26[1] /*u[i][j][2][m] */ -4.0 *
				    u_26[2] /*u[i][j][3][m] */ +u[i][j][4][m]);
		}
		for (k = 3; k <= nz - 4; k++) {
			for (m = 0; m < 5; m++) {
				rsd[i][j][k][m] =
				    rsd[i][j][k][m] -
				    dssp * (u[i][j][k - 2][m] -
					    4.0 * u[i][j][k - 1][m] +
					    6.0 * u[i][j][k][m] -
					    4.0 * u[i][j][k + 1][m] +
					    u[i][j][k + 2][m]);
			}
		}
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2759
			//-------------------------------------------
			double u_27[2];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2759
			//Candidates:
			//      u[i][j][nz - 4][m]
			//      u[i][j][nz - 3][m]
			//-------------------------------------------
			u_27[0] = u[i][j][nz - 4][m];
			u_27[1] = u[i][j][nz - 3][m];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rsd[i][j][nz - 3][m] =
			    rsd[i][j][nz - 3][m] - dssp * (u[i][j][nz - 5][m] -
							   4.0 *
							   u_27[0]
							   /*u[i][j][nz - 4][m] */
							   +6.0 *
							   u_27[1]
							   /*u[i][j][nz - 3][m] */
							   -4.0 * u[i][j][nz -
									  2]
							   [m]);
			rsd[i][j][nz - 2][m] =
			    rsd[i][j][nz - 2][m] -
			    dssp * (u_27[0] /*u[i][j][nz - 4][m] */ -4.0 *
				    u_27[1] /*u[i][j][nz - 3][m] */ +5.0 *
				    u[i][j][nz - 2][m]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2795 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_0(__global double *g_u, int nz, __global double *g_ce,
		      int nx0, int ny0, int __ocl_j_bound, int __ocl_i_bound,
		      __global int *tls_validflag)
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
	double tmp[5];		/* (User-defined privated variables) : Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6_no_spec(i, j, 0, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[i][j][0][0] = tmp[0];
		u[i][j][0][1] = tmp[1];
		u[i][j][0][2] = tmp[2];
		u[i][j][0][3] = tmp[3];
		u[i][j][0][4] = tmp[4];
		exact_g7_e4_e5_e6_no_spec(i, j, nz - 1, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[i][j][nz - 1][0] = tmp[0];
		u[i][j][nz - 1][1] = tmp[1];
		u[i][j][nz - 1][2] = tmp[2];
		u[i][j][nz - 1][3] = tmp[3];
		u[i][j][nz - 1][4] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2819 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_1(__global double *g_u, __global double *g_ce, int nx0,
		      int ny0, int nz, int __ocl_k_bound, int __ocl_i_bound,
		      __global int *tls_validflag)
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
	double tmp[5];		/* (User-defined privated variables) : Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6_no_spec(i, 0, k, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[i][0][k][0] = tmp[0];
		u[i][0][k][1] = tmp[1];
		u[i][0][k][2] = tmp[2];
		u[i][0][k][3] = tmp[3];
		u[i][0][k][4] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2833 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_2(int ny0, __global double *g_u, int ny,
		      __global double *g_ce, int nx0, int nz, int __ocl_k_bound,
		      int __ocl_i_bound, __global int *tls_validflag)
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
	double tmp[5];		/* (User-defined privated variables) : Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6_no_spec(i, ny0 - 1, k, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[i][ny - 1][k][0] = tmp[0];
		u[i][ny - 1][k][1] = tmp[1];
		u[i][ny - 1][k][2] = tmp[2];
		u[i][ny - 1][k][3] = tmp[3];
		u[i][ny - 1][k][4] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2850 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_3(__global double *g_u, __global double *g_ce, int nx0,
		      int ny0, int nz, int __ocl_k_bound, int __ocl_j_bound,
		      __global int *tls_validflag)
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
	double tmp[5];		/* (User-defined privated variables) : Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6_no_spec(0, j, k, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[0][j][k][0] = tmp[0];
		u[0][j][k][1] = tmp[1];
		u[0][j][k][2] = tmp[2];
		u[0][j][k][3] = tmp[3];
		u[0][j][k][4] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2864 of lu.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void setbv_4(int nx0, __global double *g_u, int nx,
		      __global double *g_ce, int ny0, int nz, int __ocl_k_bound,
		      int __ocl_j_bound, __global int *tls_validflag)
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
	double tmp[5];		/* (User-defined privated variables) : Defined at lu.c : 2787 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		exact_g7_e4_e5_e6_no_spec(nx0 - 1, j, k, tmp, nx0, ny0, nz, ce,
					  tls_validflag,
					  tls_thread_id)
		    /*ARGEXP: nx0,ny0,nz,ce */ ;
		u[nx - 1][j][k][0] = tmp[0];
		u[nx - 1][j][k][1] = tmp[1];
		u[nx - 1][j][k][2] = tmp[2];
		u[nx - 1][j][k][3] = tmp[3];
		u[nx - 1][j][k][4] = tmp[4];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3045 of lu.c
//-------------------------------------------------------------------------------
__kernel void setiv_0(int nz, int ny0, int nx, int nx0, int m,
		      __global double *g_u, __global double *g_ce,
		      int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int jglob;		/* (User-defined privated variables) : Defined at lu.c : 3037 */
	int k;			/* (User-defined privated variables) : Defined at lu.c : 3036 */
	double zeta;		/* (User-defined privated variables) : Defined at lu.c : 3038 */
	double eta;		/* (User-defined privated variables) : Defined at lu.c : 3038 */
	int i;			/* (User-defined privated variables) : Defined at lu.c : 3036 */
	int iglob;		/* (User-defined privated variables) : Defined at lu.c : 3037 */
	double xi;		/* (User-defined privated variables) : Defined at lu.c : 3038 */
	double ue_1jk[5];	/* (User-defined privated variables) : Defined at lu.c : 3040 */
	double ue_nx0jk[5];	/* (User-defined privated variables) : Defined at lu.c : 3040 */
	double ue_i1k[5];	/* (User-defined privated variables) : Defined at lu.c : 3040 */
	double ue_iny0k[5];	/* (User-defined privated variables) : Defined at lu.c : 3041 */
	double ue_ij1[5];	/* (User-defined privated variables) : Defined at lu.c : 3041 */
	double ue_ijnz[5];	/* (User-defined privated variables) : Defined at lu.c : 3041 */
	double pxi;		/* (User-defined privated variables) : Defined at lu.c : 3039 */
	double peta;		/* (User-defined privated variables) : Defined at lu.c : 3039 */
	double pzeta;		/* (User-defined privated variables) : Defined at lu.c : 3039 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_1();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		jglob = j;
		for (k = 1; k < nz - 1; k++) {
			zeta = ((double)k) / (nz - 1);
			if (jglob != 0 && jglob != ny0 - 1) {
				eta = ((double)(jglob)) / (ny0 - 1);
				for (i = 0; i < nx; i++) {
					iglob = i;
					if (iglob != 0 && iglob != nx0 - 1) {
						xi = ((double)(iglob)) / (nx0 -
									  1);
						exact_g7_e4_e5_e6_no_spec(0,
									  jglob,
									  k,
									  ue_1jk,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						exact_g7_e4_e5_e6_no_spec(nx0 -
									  1,
									  jglob,
									  k,
									  ue_nx0jk,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						exact_g7_e4_e5_e6_no_spec(iglob,
									  0, k,
									  ue_i1k,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						exact_g7_e4_e5_e6_no_spec(iglob,
									  ny0 -
									  1, k,
									  ue_iny0k,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						exact_g7_e4_e5_e6_no_spec(iglob,
									  jglob,
									  0,
									  ue_ij1,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						exact_g7_e4_e5_e6_no_spec(iglob,
									  jglob,
									  nz -
									  1,
									  ue_ijnz,
									  nx0,
									  ny0,
									  nz,
									  ce,
									  tls_validflag,
									  tls_thread_id)
						    /*ARGEXP: nx0,ny0,nz,ce */ ;
						for (m = 0; m < 5; m++) {
							pxi =
							    (1.0 -
							     xi) * ue_1jk[m] +
							    xi * ue_nx0jk[m];
							peta =
							    (1.0 -
							     eta) * ue_i1k[m] +
							    eta * ue_iny0k[m];
							pzeta =
							    (1.0 -
							     zeta) * ue_ij1[m] +
							    zeta * ue_ijnz[m];
							u[i][j][k][m] =
							    pxi + peta + pzeta -
							    pxi * peta -
							    peta * pzeta -
							    pzeta * pxi +
							    pxi * peta * pzeta;
						}
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
//Loop defined at line 3111 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void ssor_0(__global double *g_a, __global double *g_b,
		     __global double *g_c, __global double *g_d,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	int i = get_global_id(2);
	if (!(k < 5)) {
		return;
	}
	if (!(j < 64)) {
		return;
	}
	if (!(i < 64)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int m;			/* (User-defined privated variables) : Defined at lu.c : 3093 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*a)[64][5][5] = (__global double (*)[64][5][5])g_a;
	__global double (*b)[64][5][5] = (__global double (*)[64][5][5])g_b;
	__global double (*c)[64][5][5] = (__global double (*)[64][5][5])g_c;
	__global double (*d)[64][5][5] = (__global double (*)[64][5][5])g_d;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			a[i][j][k][m] = 0.0;
			b[i][j][k][m] = 0.0;
			c[i][j][k][m] = 0.0;
			d[i][j][k][m] = 0.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3157 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void ssor_1(__global double *g_rsd, double dt, int jst, int ist,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
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
	int m;			/* (User-defined privated variables) : Defined at lu.c : 3093 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rsd[i][j][k][m] = dt * rsd[i][j][k][m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3216 of lu.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void ssor_2(__global double *g_u, double tmp, __global double *g_rsd,
		     int jst, int ist, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *tls_validflag)
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
	int m;			/* (User-defined privated variables) : Defined at lu.c : 3093 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][5] = (__global double (*)[65][65][5])g_u;
	__global double (*rsd)[65][65][5] =
	    (__global double (*)[65][65][5])g_rsd;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[i][j][k][m] = u[i][j][k][m] + tmp * rsd[i][j][k][m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
