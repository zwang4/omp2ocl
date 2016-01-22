//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Thu Oct 25 14:32:11 2012
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
__kernel void TLS_Checking_1D(unsigned dim0, __global int *rd_log,
			      __global int *wr_log, __global int *conflict_flag)
{
	int wr, rd, index;
	int conflict = 0;
	if (get_global_id(0) >= dim0) {
		return;
	}
	index = get_global_id(0);
	wr = wr_log[index];
	rd = rd_log[index];
	conflict = (wr > 1) | (rd & wr);
	wr_log[index] = 0;
	rd_log[index] = 0;

	if (conflict) {
		*conflict_flag = 1;
	}
}

//-------------------------------------------------------------------------------
//TLS Checking Routines (END)
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//Functions (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 190 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void add_0(__global double *g_u, __global double *g_rhs,
		    int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		    __global int *g_rd_log_u, __global int *g_wr_log_u)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at sp.c : 180 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_u)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_u;
	__global int (*wr_log_u)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_u;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = u[m][i][j][k] + rhs[m][i][j][k];
			//-------------------------------------------
			// GPU TLS logs (BEGIN) 
			//-------------------------------------------
			atom_inc(&wr_log_u[m][i][j][k]);
			rd_log_u[m][i][j][k] = 1;
			//-------------------------------------------
			// GPU TLS logs (END)
			//-------------------------------------------
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 857 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsinit_0(__global double *g_lhs, int n, int __ocl_k_bound,
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[n][i][j][k] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 874 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsinit_1(__global double *g_lhs, int n, int __ocl_k_bound,
			int __ocl_j_bound, int __ocl_i_bound,
			__global int *g_rd_log_lhs, __global int *g_wr_log_lhs)
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[5 * n + 2][i][j][k] = 1.0;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[5 * n + 2][i][j][k]);
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 937 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_0(__global double *g_lhs, int i, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_k_bound,
		     int __ocl_j_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz5;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		lhs[1][i + 1][j][k] = lhs[1][i + 1][j][k] - comz4;
		lhs[2][i + 1][j][k] = lhs[2][i + 1][j][k] + comz6;
		lhs[3][i + 1][j][k] = lhs[3][i + 1][j][k] - comz4;
		lhs[4][i + 1][j][k] = lhs[4][i + 1][j][k] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		atom_inc(&wr_log_lhs[1][i + 1][j][k]);
		atom_inc(&wr_log_lhs[2][i + 1][j][k]);
		atom_inc(&wr_log_lhs[3][i + 1][j][k]);
		atom_inc(&wr_log_lhs[4][i + 1][j][k]);
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		rd_log_lhs[1][i + 1][j][k] = 1;
		rd_log_lhs[2][i + 1][j][k] = 1;
		rd_log_lhs[3][i + 1][j][k] = 1;
		rd_log_lhs[4][i + 1][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 953 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_1(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 3;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 970 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_2(__global double *g_lhs, int i, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_k_bound,
		     int __ocl_j_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[0][i + 1][j][k] = lhs[0][i + 1][j][k] + comz1;
		lhs[1][i + 1][j][k] = lhs[1][i + 1][j][k] - comz4;
		lhs[2][i + 1][j][k] = lhs[2][i + 1][j][k] + comz5;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[0][i + 1][j][k]);
		atom_inc(&wr_log_lhs[1][i + 1][j][k]);
		atom_inc(&wr_log_lhs[2][i + 1][j][k]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[0][i + 1][j][k] = 1;
		rd_log_lhs[1][i + 1][j][k] = 1;
		rd_log_lhs[2][i + 1][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 991 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_3(__global double *g_lhs, double dttx2,
		     __global double *g_speed, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *g_rd_log_lhs, __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 993
		//-------------------------------------------
		double speed_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 993
		//Candidates:
		//      speed[i - 1][j][k]
		//-------------------------------------------
		speed_1 = speed[i - 1][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dttx2 * speed_1 /*speed[i - 1][j][k] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dttx2 * speed[i + 1][j][k];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dttx2 * speed_1 /*speed[i - 1][j][k] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dttx2 * speed[i + 1][j][k];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[0 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 10][i][j][k]);
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1067 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_0(__global double *g_lhs, int j, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_k_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz5;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		lhs[1][i][j + 1][k] = lhs[1][i][j + 1][k] - comz4;
		lhs[2][i][j + 1][k] = lhs[2][i][j + 1][k] + comz6;
		lhs[3][i][j + 1][k] = lhs[3][i][j + 1][k] - comz4;
		lhs[4][i][j + 1][k] = lhs[4][i][j + 1][k] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j + 1][k]);
		atom_inc(&wr_log_lhs[2][i][j + 1][k]);
		atom_inc(&wr_log_lhs[3][i][j + 1][k]);
		atom_inc(&wr_log_lhs[4][i][j + 1][k]);
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		rd_log_lhs[1][i][j + 1][k] = 1;
		rd_log_lhs[2][i][j + 1][k] = 1;
		rd_log_lhs[3][i][j + 1][k] = 1;
		rd_log_lhs[4][i][j + 1][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1084 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_1(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 3;
	int i = get_global_id(2) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1100 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_2(__global double *g_lhs, int j, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_k_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[0][i][j + 1][k] = lhs[0][i][j + 1][k] + comz1;
		lhs[1][i][j + 1][k] = lhs[1][i][j + 1][k] - comz4;
		lhs[2][i][j + 1][k] = lhs[2][i][j + 1][k] + comz5;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[0][i][j + 1][k]);
		atom_inc(&wr_log_lhs[1][i][j + 1][k]);
		atom_inc(&wr_log_lhs[2][i][j + 1][k]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[0][i][j + 1][k] = 1;
		rd_log_lhs[1][i][j + 1][k] = 1;
		rd_log_lhs[2][i][j + 1][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1120 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_3(__global double *g_lhs, double dtty2,
		     __global double *g_speed, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *g_rd_log_lhs, __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1122
		//-------------------------------------------
		double speed_3;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1122
		//Candidates:
		//      speed[i][j - 1][k]
		//-------------------------------------------
		speed_3 = speed[i][j - 1][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dtty2 * speed_3 /*speed[i][j - 1][k] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dtty2 * speed[i][j + 1][k];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dtty2 * speed_3 /*speed[i][j - 1][k] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dtty2 * speed[i][j + 1][k];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[0 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 10][i][j][k]);
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1198 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_0(__global double *g_lhs, int k, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz5;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		lhs[1][i][j][k + 1] = lhs[1][i][j][k + 1] - comz4;
		lhs[2][i][j][k + 1] = lhs[2][i][j][k + 1] + comz6;
		lhs[3][i][j][k + 1] = lhs[3][i][j][k + 1] - comz4;
		lhs[4][i][j][k + 1] = lhs[4][i][j][k + 1] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k + 1]);
		atom_inc(&wr_log_lhs[2][i][j][k + 1]);
		atom_inc(&wr_log_lhs[3][i][j][k + 1]);
		atom_inc(&wr_log_lhs[4][i][j][k + 1]);
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		rd_log_lhs[1][i][j][k + 1] = 1;
		rd_log_lhs[2][i][j][k + 1] = 1;
		rd_log_lhs[3][i][j][k + 1] = 1;
		rd_log_lhs[4][i][j][k + 1] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1215 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_1(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 3;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[4][i][j][k] = lhs[4][i][j][k] + comz1;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[4][i][j][k]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[4][i][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1232 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_2(__global double *g_lhs, int k, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_j_bound,
		     int __ocl_i_bound, __global int *g_rd_log_lhs,
		     __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = lhs[0][i][j][k] + comz1;
		lhs[1][i][j][k] = lhs[1][i][j][k] - comz4;
		lhs[2][i][j][k] = lhs[2][i][j][k] + comz6;
		lhs[3][i][j][k] = lhs[3][i][j][k] - comz4;
		lhs[0][i][j][k + 1] = lhs[0][i][j][k + 1] + comz1;
		lhs[1][i][j][k + 1] = lhs[1][i][j][k + 1] - comz4;
		lhs[2][i][j][k + 1] = lhs[2][i][j][k + 1] + comz5;
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0][i][j][k]);
		atom_inc(&wr_log_lhs[1][i][j][k]);
		atom_inc(&wr_log_lhs[2][i][j][k]);
		atom_inc(&wr_log_lhs[3][i][j][k]);
		atom_inc(&wr_log_lhs[0][i][j][k + 1]);
		atom_inc(&wr_log_lhs[1][i][j][k + 1]);
		atom_inc(&wr_log_lhs[2][i][j][k + 1]);
		rd_log_lhs[0][i][j][k] = 1;
		rd_log_lhs[1][i][j][k] = 1;
		rd_log_lhs[2][i][j][k] = 1;
		rd_log_lhs[3][i][j][k] = 1;
		rd_log_lhs[0][i][j][k + 1] = 1;
		rd_log_lhs[1][i][j][k + 1] = 1;
		rd_log_lhs[2][i][j][k + 1] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1252 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_3(__global double *g_lhs, double dttz2,
		     __global double *g_speed, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *g_rd_log_lhs, __global int *g_wr_log_lhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_lhs;
	__global int (*wr_log_lhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_lhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1254
		//-------------------------------------------
		double speed_5;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1254
		//Candidates:
		//      speed[i][j][k - 1]
		//-------------------------------------------
		speed_5 = speed[i][j][k - 1];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dttz2 * speed_5 /*speed[i][j][k - 1] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dttz2 * speed[i][j][k + 1];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dttz2 * speed_5 /*speed[i][j][k - 1] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dttz2 * speed[i][j][k + 1];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_lhs[0 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 5][i][j][k]);
		atom_inc(&wr_log_lhs[0 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[1 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[2 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[3 + 10][i][j][k]);
		atom_inc(&wr_log_lhs[4 + 10][i][j][k]);
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1292 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void ninvr_0(__global double *g_rhs, double bt, int __ocl_k_bound,
		      int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double r1;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double r2;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double r3;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double r4;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double r5;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double t1;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	double t2;		/* (User-defined privated variables) : Defined at sp.c : 1288 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		r1 = rhs[0][i][j][k];
		r2 = rhs[1][i][j][k];
		r3 = rhs[2][i][j][k];
		r4 = rhs[3][i][j][k];
		r5 = rhs[4][i][j][k];
		t1 = bt * r3;
		t2 = 0.5 * (r4 + r5);
		rhs[0][i][j][k] = -r2;
		rhs[1][i][j][k] = r1;
		rhs[2][i][j][k] = bt * (r4 - r5);
		rhs[3][i][j][k] = -t1 + t2;
		rhs[4][i][j][k] = t1 + t2;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1334 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void pinvr_0(__global double *g_rhs, double bt, int __ocl_k_bound,
		      int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double r1;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double r2;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double r3;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double r4;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double r5;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double t1;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	double t2;		/* (User-defined privated variables) : Defined at sp.c : 1329 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		r1 = rhs[0][i][j][k];
		r2 = rhs[1][i][j][k];
		r3 = rhs[2][i][j][k];
		r4 = rhs[3][i][j][k];
		r5 = rhs[4][i][j][k];
		t1 = bt * r1;
		t2 = 0.5 * (r4 + r5);
		rhs[0][i][j][k] = bt * (r4 - r5);
		rhs[1][i][j][k] = -r3;
		rhs[2][i][j][k] = r2;
		rhs[3][i][j][k] = -t1 + t2;
		rhs[4][i][j][k] = t1 + t2;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1378 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_0(__global double *g_u, __global double *g_rho_i,
			    __global double *g_us, __global double *g_vs,
			    __global double *g_ws, __global double *g_square,
			    __global double *g_qs, double c1c2,
			    __global double *g_speed, __global double *g_ainv,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
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
	double rho_inv;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	double aux;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*rho_i)[37][37] = (__global double (*)[37][37])g_rho_i;
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*square)[37][37] =
	    (__global double (*)[37][37])g_square;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;
	__global double (*ainv)[37][37] = (__global double (*)[37][37])g_ainv;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1380
		//-------------------------------------------
		double u_0[3];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1380
		//Candidates:
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_0[0] = u[1][i][j][k];
		u_0[1] = u[2][i][j][k];
		u_0[2] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rho_inv = 1.0 / u[0][i][j][k];
		rho_i[i][j][k] = rho_inv;
		us[i][j][k] = u_0[0] /*u[1][i][j][k] */ *rho_inv;
		vs[i][j][k] = u_0[1] /*u[2][i][j][k] */ *rho_inv;
		ws[i][j][k] = u_0[2] /*u[3][i][j][k] */ *rho_inv;
		square[i][j][k] =
		    0.5 *
		    (u_0[0] /*u[1][i][j][k] */ *u_0[0] /*u[1][i][j][k] */
		     +u_0[1] /*u[2][i][j][k] */ *u_0[1] /*u[2][i][j][k] */
		     +u_0[2] /*u[3][i][j][k] */ *u_0[2] /*u[3][i][j][k] */ ) *
		    rho_inv;
		qs[i][j][k] = square[i][j][k] * rho_inv;
		aux = c1c2 * rho_inv * (u[4][i][j][k] - square[i][j][k]);
		aux = sqrt(aux);
		speed[i][j][k] = aux;
		ainv[i][j][k] = 1.0 / aux;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1410 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_1(__global int *grid_points, __global double *g_rhs,
			    __global double *g_forcing, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0);
	int i = get_global_id(1);
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
	int k;			/* (User-defined privated variables) : Defined at sp.c : 1366 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*forcing)[37][37][37] =
	    (__global double (*)[37][37][37])g_forcing;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= grid_points[2] - 1; k++) {
			rhs[m][i][j][k] = forcing[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1427 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_2(__global double *g_us, __global double *g_rhs,
			    double dx1tx1, __global double *g_u, double tx2,
			    double dx2tx1, double xxcon2, double con43,
			    __global double *g_square, double c2, double dx3tx1,
			    __global double *g_vs, double dx4tx1,
			    __global double *g_ws, double dx5tx1, double xxcon3,
			    __global double *g_qs, double xxcon4, double xxcon5,
			    __global double *g_rho_i, double c1,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double uijk;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	double up1;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	double um1;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*square)[37][37] =
	    (__global double (*)[37][37])g_square;
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	__global double (*rho_i)[37][37] = (__global double (*)[37][37])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1429
		//-------------------------------------------
		double u_1[8];
		double square_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1429
		//Candidates:
		//      u[1][i - 1][j][k]
		//      u[1][i + 1][j][k]
		//      u[2][i - 1][j][k]
		//      u[4][i - 1][j][k]
		//      u[2][i + 1][j][k]
		//      u[3][i - 1][j][k]
		//      u[3][i + 1][j][k]
		//      u[4][i][j][k]
		//      square[i - 1][j][k]
		//-------------------------------------------
		u_1[0] = u[1][i - 1][j][k];
		u_1[1] = u[1][i + 1][j][k];
		u_1[2] = u[2][i - 1][j][k];
		u_1[3] = u[4][i - 1][j][k];
		u_1[4] = u[2][i + 1][j][k];
		u_1[5] = u[3][i - 1][j][k];
		u_1[6] = u[3][i + 1][j][k];
		u_1[7] = u[4][i][j][k];
		square_0 = square[i - 1][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		uijk = us[i][j][k];
		up1 = us[i + 1][j][k];
		um1 = us[i - 1][j][k];
		rhs[0][i][j][k] =
		    rhs[0][i][j][k] + dx1tx1 * (u[0][i + 1][j][k] -
						2.0 * u[0][i][j][k] + u[0][i -
									   1][j]
						[k]) -
		    tx2 *
		    (u_1[1] /*u[1][i + 1][j][k] */ -u_1[0]
		     /*u[1][i - 1][j][k] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] +
		    dx2tx1 * (u_1[1] /*u[1][i + 1][j][k] */ -2.0 *
			      u[1][i][j][k] + u_1[0] /*u[1][i - 1][j][k] */ ) +
		    xxcon2 * con43 * (up1 - 2.0 * uijk + um1) -
		    tx2 * (u_1[1] /*u[1][i + 1][j][k] */ *up1 -
			   u_1[0] /*u[1][i - 1][j][k] */ *um1 +
			   (u[4][i + 1][j][k] - square[i + 1][j][k] -
			    u_1[3] /*u[4][i - 1][j][k] */ +square_0
			    /*square[i - 1][j][k] */ ) * c2);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dx3tx1 * (u_1[4] /*u[2][i + 1][j][k] */ -2.0 *
			      u[2][i][j][k] + u_1[2] /*u[2][i - 1][j][k] */ ) +
		    xxcon2 * (vs[i + 1][j][k] - 2.0 * vs[i][j][k] +
			      vs[i - 1][j][k]) -
		    tx2 * (u_1[4] /*u[2][i + 1][j][k] */ *up1 -
			   u_1[2] /*u[2][i - 1][j][k] */ *um1);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dx4tx1 * (u_1[6] /*u[3][i + 1][j][k] */ -2.0 *
			      u[3][i][j][k] + u_1[5] /*u[3][i - 1][j][k] */ ) +
		    xxcon2 * (ws[i + 1][j][k] - 2.0 * ws[i][j][k] +
			      ws[i - 1][j][k]) -
		    tx2 * (u_1[6] /*u[3][i + 1][j][k] */ *up1 -
			   u_1[5] /*u[3][i - 1][j][k] */ *um1);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dx5tx1 * (u[4][i + 1][j][k] -
						2.0 *
						u_1[7] /*u[4][i][j][k] */
						+u_1[3] /*u[4][i - 1][j][k] */ )
		    + xxcon3 * (qs[i + 1][j][k] - 2.0 * qs[i][j][k] +
				qs[i - 1][j][k]) + xxcon4 * (up1 * up1 -
							     2.0 * uijk * uijk +
							     um1 * um1) +
		    xxcon5 * (u[4][i + 1][j][k] * rho_i[i + 1][j][k] -
			      2.0 * u_1[7] /*u[4][i][j][k] */ *rho_i[i][j][k] +
			      u_1[3] /*u[4][i - 1][j][k] */ *rho_i[i -
								   1][j][k]) -
		    tx2 * ((c1 * u[4][i + 1][j][k] - c2 * square[i + 1][j][k]) *
			   up1 -
			   (c1 * u_1[3] /*u[4][i - 1][j][k] */ -c2 *
			    square_0 /*square[i - 1][j][k] */ ) * um1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1490 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_3(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (5.0 * u[m][i][j][k] -
					      4.0 * u[m][i + 1][j][k] + u[m][i +
									     2]
					      [j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1504 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_4(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (-4.0 * u[m][i - 1][j][k] +
					      6.0 * u[m][i][j][k] -
					      4.0 * u[m][i + 1][j][k] + u[m][i +
									     2]
					      [j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1517 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_5(__global double *g_rhs, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 3 * 1;
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
	int m;			/* (User-defined privated variables) : Defined at sp.c : 1366 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[m][i][j][k] =
			    rhs[m][i][j][k] - dssp * (u[m][i - 2][j][k] -
						      4.0 * u[m][i - 1][j][k] +
						      6.0 * u[m][i][j][k] -
						      4.0 * u[m][i + 1][j][k] +
						      u[m][i + 2][j][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1534 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_6(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i - 2][j][k] -
					      4.0 * u[m][i - 1][j][k] +
					      6.0 * u[m][i][j][k] -
					      4.0 * u[m][i + 1][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1548 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_7(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i - 2][j][k] -
					      4.0 * u[m][i - 1][j][k] +
					      5.0 * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1566 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_8(__global double *g_vs, __global double *g_rhs,
			    double dy1ty1, __global double *g_u, double ty2,
			    double dy2ty1, double yycon2, __global double *g_us,
			    double dy3ty1, double con43,
			    __global double *g_square, double c2, double dy4ty1,
			    __global double *g_ws, double dy5ty1, double yycon3,
			    __global double *g_qs, double yycon4, double yycon5,
			    __global double *g_rho_i, double c1,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double vijk;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	double vp1;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	double vm1;		/* (User-defined privated variables) : Defined at sp.c : 1367 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*square)[37][37] =
	    (__global double (*)[37][37])g_square;
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	__global double (*rho_i)[37][37] = (__global double (*)[37][37])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1568
		//-------------------------------------------
		double u_2[8];
		double square_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1568
		//Candidates:
		//      u[1][i][j - 1][k]
		//      u[2][i][j - 1][k]
		//      u[1][i][j + 1][k]
		//      u[2][i][j + 1][k]
		//      u[3][i][j - 1][k]
		//      u[4][i][j - 1][k]
		//      u[3][i][j + 1][k]
		//      u[4][i][j][k]
		//      square[i][j - 1][k]
		//-------------------------------------------
		u_2[0] = u[1][i][j - 1][k];
		u_2[1] = u[2][i][j - 1][k];
		u_2[2] = u[1][i][j + 1][k];
		u_2[3] = u[2][i][j + 1][k];
		u_2[4] = u[3][i][j - 1][k];
		u_2[5] = u[4][i][j - 1][k];
		u_2[6] = u[3][i][j + 1][k];
		u_2[7] = u[4][i][j][k];
		square_1 = square[i][j - 1][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		vijk = vs[i][j][k];
		vp1 = vs[i][j + 1][k];
		vm1 = vs[i][j - 1][k];
		rhs[0][i][j][k] =
		    rhs[0][i][j][k] + dy1ty1 * (u[0][i][j + 1][k] -
						2.0 * u[0][i][j][k] +
						u[0][i][j - 1][k]) -
		    ty2 *
		    (u_2[3] /*u[2][i][j + 1][k] */ -u_2[1]
		     /*u[2][i][j - 1][k] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] +
		    dy2ty1 * (u_2[2] /*u[1][i][j + 1][k] */ -2.0 *
			      u[1][i][j][k] + u_2[0] /*u[1][i][j - 1][k] */ ) +
		    yycon2 * (us[i][j + 1][k] - 2.0 * us[i][j][k] +
			      us[i][j - 1][k]) -
		    ty2 * (u_2[2] /*u[1][i][j + 1][k] */ *vp1 -
			   u_2[0] /*u[1][i][j - 1][k] */ *vm1);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dy3ty1 * (u_2[3] /*u[2][i][j + 1][k] */ -2.0 *
			      u[2][i][j][k] + u_2[1] /*u[2][i][j - 1][k] */ ) +
		    yycon2 * con43 * (vp1 - 2.0 * vijk + vm1) -
		    ty2 * (u_2[3] /*u[2][i][j + 1][k] */ *vp1 -
			   u_2[1] /*u[2][i][j - 1][k] */ *vm1 +
			   (u[4][i][j + 1][k] - square[i][j + 1][k] -
			    u_2[5] /*u[4][i][j - 1][k] */ +square_1
			    /*square[i][j - 1][k] */ ) * c2);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dy4ty1 * (u_2[6] /*u[3][i][j + 1][k] */ -2.0 *
			      u[3][i][j][k] + u_2[4] /*u[3][i][j - 1][k] */ ) +
		    yycon2 * (ws[i][j + 1][k] - 2.0 * ws[i][j][k] +
			      ws[i][j - 1][k]) -
		    ty2 * (u_2[6] /*u[3][i][j + 1][k] */ *vp1 -
			   u_2[4] /*u[3][i][j - 1][k] */ *vm1);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dy5ty1 * (u[4][i][j + 1][k] -
						2.0 *
						u_2[7] /*u[4][i][j][k] */
						+u_2[5] /*u[4][i][j - 1][k] */ )
		    + yycon3 * (qs[i][j + 1][k] - 2.0 * qs[i][j][k] +
				qs[i][j - 1][k]) + yycon4 * (vp1 * vp1 -
							     2.0 * vijk * vijk +
							     vm1 * vm1) +
		    yycon5 * (u[4][i][j + 1][k] * rho_i[i][j + 1][k] -
			      2.0 * u_2[7] /*u[4][i][j][k] */ *rho_i[i][j][k] +
			      u_2[5] /*u[4][i][j - 1][k] */ *rho_i[i][j -
								      1][k]) -
		    ty2 * ((c1 * u[4][i][j + 1][k] - c2 * square[i][j + 1][k]) *
			   vp1 -
			   (c1 * u_2[5] /*u[4][i][j - 1][k] */ -c2 *
			    square_1 /*square[i][j - 1][k] */ ) * vm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1625 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_9(__global double *g_rhs, int j, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (5.0 * u[m][i][j][k] -
					      4.0 * u[m][i][j + 1][k] +
					      u[m][i][j + 2][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1639 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_10(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (-4.0 * u[m][i][j - 1][k] +
					      6.0 * u[m][i][j][k] -
					      4.0 * u[m][i][j + 1][k] +
					      u[m][i][j + 2][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1652 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_11(__global int *grid_points, __global double *g_rhs,
			     double dssp, __global double *g_u,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 3 * 1;
	int i = get_global_id(1) + 1;
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
	int k;			/* (User-defined privated variables) : Defined at sp.c : 1366 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 1; k <= grid_points[2] - 2; k++) {
			rhs[m][i][j][k] =
			    rhs[m][i][j][k] - dssp * (u[m][i][j - 2][k] -
						      4.0 * u[m][i][j - 1][k] +
						      6.0 * u[m][i][j][k] -
						      4.0 * u[m][i][j + 1][k] +
						      u[m][i][j + 2][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1669 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_12(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i][j - 2][k] -
					      4.0 * u[m][i][j - 1][k] +
					      6.0 * u[m][i][j][k] -
					      4.0 * u[m][i][j + 1][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1683 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_13(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i][j - 2][k] -
					      4.0 * u[m][i][j - 1][k] +
					      5.0 * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1701 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_14(__global double *g_ws, __global double *g_rhs,
			     double dz1tz1, __global double *g_u, double tz2,
			     double dz2tz1, double zzcon2,
			     __global double *g_us, double dz3tz1,
			     __global double *g_vs, double dz4tz1, double con43,
			     __global double *g_square, double c2,
			     double dz5tz1, double zzcon3,
			     __global double *g_qs, double zzcon4,
			     double zzcon5, __global double *g_rho_i, double c1,
			     int __ocl_k_bound, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double wijk;		/* (User-defined privated variables) : Defined at sp.c : 1368 */
	double wp1;		/* (User-defined privated variables) : Defined at sp.c : 1368 */
	double wm1;		/* (User-defined privated variables) : Defined at sp.c : 1368 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*square)[37][37] =
	    (__global double (*)[37][37])g_square;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	__global double (*rho_i)[37][37] = (__global double (*)[37][37])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1703
		//-------------------------------------------
		double2 ws_0;
		double2 u_3[4];
		double u_4[3];
		double2 us_0;
		double2 vs_0;
		double square_2;
		double2 qs_0;
		double2 rho_i_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1703
		//Candidates:
		//      ws[i][j][k - 1]
		//      ws[i][j][k]
		//      u[0][i][j][k - 1]
		//      u[0][i][j][k]
		//      u[2][i][j][k - 1]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//      u[3][i][j][k + 1]
		//      u[4][i][j][k - 1]
		//      u[4][i][j][k]
		//      u[1][i][j][k - 1]
		//      u[3][i][j][k - 1]
		//      u[2][i][j][k + 1]
		//      us[i][j][k - 1]
		//      us[i][j][k]
		//      vs[i][j][k - 1]
		//      vs[i][j][k]
		//      square[i][j][k - 1]
		//      qs[i][j][k - 1]
		//      qs[i][j][k]
		//      rho_i[i][j][k - 1]
		//      rho_i[i][j][k]
		//-------------------------------------------
		__global double *p_ws_0_0 = (__global double *)&ws[i][j][k - 1];
		if ((unsigned long)p_ws_0_0 % 64 == 0) {
			ws_0 = vload2(0, p_ws_0_0);
		} else {
			ws_0.x = p_ws_0_0[0];
			p_ws_0_0++;
			ws_0.y = p_ws_0_0[0];
			p_ws_0_0++;
		}
		__global double *p_u_3_0 =
		    (__global double *)&u[0][i][j][k - 1];
		if ((unsigned long)p_u_3_0 % 64 == 0) {
			u_3[0] = vload2(0, p_u_3_0);
		} else {
			u_3[0].x = p_u_3_0[0];
			p_u_3_0++;
			u_3[0].y = p_u_3_0[0];
			p_u_3_0++;
		}
		__global double *p_u_3_1 =
		    (__global double *)&u[2][i][j][k - 1];
		if ((unsigned long)p_u_3_1 % 64 == 0) {
			u_3[1] = vload2(0, p_u_3_1);
		} else {
			u_3[1].x = p_u_3_1[0];
			p_u_3_1++;
			u_3[1].y = p_u_3_1[0];
			p_u_3_1++;
		}
		__global double *p_u_3_2 = (__global double *)&u[3][i][j][k];
		if ((unsigned long)p_u_3_2 % 64 == 0) {
			u_3[2] = vload2(0, p_u_3_2);
		} else {
			u_3[2].x = p_u_3_2[0];
			p_u_3_2++;
			u_3[2].y = p_u_3_2[0];
			p_u_3_2++;
		}
		__global double *p_u_3_3 =
		    (__global double *)&u[4][i][j][k - 1];
		if ((unsigned long)p_u_3_3 % 64 == 0) {
			u_3[3] = vload2(0, p_u_3_3);
		} else {
			u_3[3].x = p_u_3_3[0];
			p_u_3_3++;
			u_3[3].y = p_u_3_3[0];
			p_u_3_3++;
		}
		u_4[0] = u[1][i][j][k - 1];
		u_4[1] = u[3][i][j][k - 1];
		u_4[2] = u[2][i][j][k + 1];
		__global double *p_us_0_0 = (__global double *)&us[i][j][k - 1];
		if ((unsigned long)p_us_0_0 % 64 == 0) {
			us_0 = vload2(0, p_us_0_0);
		} else {
			us_0.x = p_us_0_0[0];
			p_us_0_0++;
			us_0.y = p_us_0_0[0];
			p_us_0_0++;
		}
		__global double *p_vs_0_0 = (__global double *)&vs[i][j][k - 1];
		if ((unsigned long)p_vs_0_0 % 64 == 0) {
			vs_0 = vload2(0, p_vs_0_0);
		} else {
			vs_0.x = p_vs_0_0[0];
			p_vs_0_0++;
			vs_0.y = p_vs_0_0[0];
			p_vs_0_0++;
		}
		square_2 = square[i][j][k - 1];
		__global double *p_qs_0_0 = (__global double *)&qs[i][j][k - 1];
		if ((unsigned long)p_qs_0_0 % 64 == 0) {
			qs_0 = vload2(0, p_qs_0_0);
		} else {
			qs_0.x = p_qs_0_0[0];
			p_qs_0_0++;
			qs_0.y = p_qs_0_0[0];
			p_qs_0_0++;
		}
		__global double *p_rho_i_0_0 =
		    (__global double *)&rho_i[i][j][k - 1];
		if ((unsigned long)p_rho_i_0_0 % 64 == 0) {
			rho_i_0 = vload2(0, p_rho_i_0_0);
		} else {
			rho_i_0.x = p_rho_i_0_0[0];
			p_rho_i_0_0++;
			rho_i_0.y = p_rho_i_0_0[0];
			p_rho_i_0_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		wijk = ws_0.y /*ws[i][j][k] */ ;
		wp1 = ws[i][j][k + 1];
		wm1 = ws_0.x /*ws[i][j][k - 1] */ ;
		rhs[0][i][j][k] =
		    rhs[0][i][j][k] + dz1tz1 * (u[0][i][j][k + 1] -
						2.0 *
						u_3[0].y /*u[0][i][j][k] */  +
						u_3[0].
						x /*u[0][i][j][k - 1] */ ) -
		    tz2 * (u_3[2].y /*u[3][i][j][k + 1] */  -
			   u_4[1] /*u[3][i][j][k - 1] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] + dz2tz1 * (u[1][i][j][k + 1] -
						2.0 * u[1][i][j][k] +
						u_4[0] /*u[1][i][j][k - 1] */ )
		    + zzcon2 * (us[i][j][k + 1] -
				2.0 * us_0.y /*us[i][j][k] */  +
				us_0.x /*us[i][j][k - 1] */ ) -
		    tz2 * (u[1][i][j][k + 1] * wp1 -
			   u_4[0] /*u[1][i][j][k - 1] */ *wm1);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dz3tz1 * (u_4[2] /*u[2][i][j][k + 1] */ -2.0 *
			      u_3[1].y /*u[2][i][j][k] */  +
			      u_3[1].x /*u[2][i][j][k - 1] */ ) +
		    zzcon2 * (vs[i][j][k + 1] - 2.0 * vs_0.y /*vs[i][j][k] */  +
			      vs_0.x /*vs[i][j][k - 1] */ ) -
		    tz2 * (u_4[2] /*u[2][i][j][k + 1] */ *wp1 -
			   u_3[1].x /*u[2][i][j][k - 1] */  * wm1);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dz4tz1 * (u_3[2].y /*u[3][i][j][k + 1] */  -
			      2.0 * u_3[2].x /*u[3][i][j][k] */  +
			      u_4[1] /*u[3][i][j][k - 1] */ ) +
		    zzcon2 * con43 * (wp1 - 2.0 * wijk + wm1) -
		    tz2 * (u_3[2].y /*u[3][i][j][k + 1] */  * wp1 -
			   u_4[1] /*u[3][i][j][k - 1] */ *wm1 +
			   (u[4][i][j][k + 1] - square[i][j][k + 1] -
			    u_3[3].x /*u[4][i][j][k - 1] */  +
			    square_2 /*square[i][j][k - 1] */ ) * c2);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dz5tz1 * (u[4][i][j][k + 1] -
						2.0 *
						u_3[3].y /*u[4][i][j][k] */  +
						u_3[3].
						x /*u[4][i][j][k - 1] */ ) +
		    zzcon3 * (qs[i][j][k + 1] - 2.0 * qs_0.y /*qs[i][j][k] */  +
			      qs_0.x /*qs[i][j][k - 1] */ ) +
		    zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
		    zzcon5 * (u[4][i][j][k + 1] * rho_i[i][j][k + 1] -
			      2.0 * u_3[3].y /*u[4][i][j][k] */  *
			      rho_i_0.y /*rho_i[i][j][k] */  +
			      u_3[3].x /*u[4][i][j][k - 1] */  *
			      rho_i_0.x /*rho_i[i][j][k - 1] */ ) -
		    tz2 * ((c1 * u[4][i][j][k + 1] - c2 * square[i][j][k + 1]) *
			   wp1 - (c1 * u_3[3].x /*u[4][i][j][k - 1] */  -
				  c2 * square_2 /*square[i][j][k - 1] */ ) *
			   wm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1761 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_15(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1763
		//-------------------------------------------
		double2 u_5;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1763
		//Candidates:
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//-------------------------------------------
		__global double *p_u_5_0 = (__global double *)&u[m][i][j][k];
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

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (5.0 * u_5.x /*u[m][i][j][k] */  -
					      4.0 *
					      u_5.y /*u[m][i][j][k + 1] */  +
					      u[m][i][j][k + 2]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1775 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_16(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1777
		//-------------------------------------------
		double4 u_6;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1777
		//Candidates:
		//      u[m][i][j][k - 1]
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//      u[m][i][j][k + 2]
		//-------------------------------------------
		__global double *p_u_6_0 =
		    (__global double *)&u[m][i][j][k - 1];
		if ((unsigned long)p_u_6_0 % 64 == 0) {
			u_6 = vload4(0, p_u_6_0);
		} else {
			u_6.x = p_u_6_0[0];
			p_u_6_0++;
			u_6.y = p_u_6_0[0];
			p_u_6_0++;
			u_6.z = p_u_6_0[0];
			p_u_6_0++;
			u_6.w = p_u_6_0[0];
			p_u_6_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] -
		    dssp * (-4.0 * u_6.x /*u[m][i][j][k - 1] */  +
			    6.0 * u_6.y /*u[m][i][j][k] */  -
			    4.0 * u_6.z /*u[m][i][j][k + 1] */  +
			    u_6.w /*u[m][i][j][k + 2] */ );
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1788 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_17(__global int *grid_points, __global double *g_rhs,
			     double dssp, __global double *g_u,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	int k;			/* (User-defined privated variables) : Defined at sp.c : 1366 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 3 * 1; k <= grid_points[2] - 3 * 1 - 1; k++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 1791
			//-------------------------------------------
			double4 u_7;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 1791
			//Candidates:
			//      u[m][i][j][k - 2]
			//      u[m][i][j][k - 1]
			//      u[m][i][j][k]
			//      u[m][i][j][k + 1]
			//-------------------------------------------
			__global double *p_u_7_0 =
			    (__global double *)&u[m][i][j][k - 2];
			if ((unsigned long)p_u_7_0 % 64 == 0) {
				u_7 = vload4(0, p_u_7_0);
			} else {
				u_7.x = p_u_7_0[0];
				p_u_7_0++;
				u_7.y = p_u_7_0[0];
				p_u_7_0++;
				u_7.z = p_u_7_0[0];
				p_u_7_0++;
				u_7.w = p_u_7_0[0];
				p_u_7_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rhs[m][i][j][k] =
			    rhs[m][i][j][k] -
			    dssp * (u_7.x /*u[m][i][j][k - 2] */  -
				    4.0 * u_7.y /*u[m][i][j][k - 1] */  +
				    6.0 * u_7.z /*u[m][i][j][k] */  -
				    4.0 * u_7.w /*u[m][i][j][k + 1] */  +
				    u[m][i][j][k + 2]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1805 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_18(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1807
		//-------------------------------------------
		double4 u_8;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1807
		//Candidates:
		//      u[m][i][j][k - 2]
		//      u[m][i][j][k - 1]
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//-------------------------------------------
		__global double *p_u_8_0 =
		    (__global double *)&u[m][i][j][k - 2];
		if ((unsigned long)p_u_8_0 % 64 == 0) {
			u_8 = vload4(0, p_u_8_0);
		} else {
			u_8.x = p_u_8_0[0];
			p_u_8_0++;
			u_8.y = p_u_8_0[0];
			p_u_8_0++;
			u_8.z = p_u_8_0[0];
			p_u_8_0++;
			u_8.w = p_u_8_0[0];
			p_u_8_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u_8.x /*u[m][i][j][k - 2] */  -
					      4.0 *
					      u_8.y /*u[m][i][j][k - 1] */  +
					      6.0 * u_8.z /*u[m][i][j][k] */  -
					      4.0 *
					      u_8.w /*u[m][i][j][k + 1] */ );
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1819 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_19(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1821
		//-------------------------------------------
		double2 u_9;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1821
		//Candidates:
		//      u[m][i][j][k - 2]
		//      u[m][i][j][k - 1]
		//-------------------------------------------
		__global double *p_u_9_0 =
		    (__global double *)&u[m][i][j][k - 2];
		if ((unsigned long)p_u_9_0 % 64 == 0) {
			u_9 = vload2(0, p_u_9_0);
		} else {
			u_9.x = p_u_9_0[0];
			p_u_9_0++;
			u_9.y = p_u_9_0[0];
			p_u_9_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u_9.x /*u[m][i][j][k - 2] */  -
					      4.0 *
					      u_9.y /*u[m][i][j][k - 1] */  +
					      5.0 * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1832 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_20(__global int *grid_points, __global double *g_rhs,
			     double dt, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	int k;			/* (User-defined privated variables) : Defined at sp.c : 1366 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 1; k <= grid_points[2] - 2; k++) {
			rhs[m][i][j][k] = rhs[m][i][j][k] * dt;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2061 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void txinvr_0(__global double *g_rho_i, __global double *g_us,
		       __global double *g_vs, __global double *g_ws,
		       __global double *g_speed, __global double *g_ainv,
		       __global double *g_rhs, double c2, __global double *g_qs,
		       double bt, int __ocl_k_bound, int __ocl_j_bound,
		       int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double ru1;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double uu;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double vv;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double ww;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double ac;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double ac2inv;		/* (User-defined privated variables) : Defined at sp.c : 2057 */
	double r1;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double r2;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double r3;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double r4;		/* (User-defined privated variables) : Defined at sp.c : 2057 */
	double r5;		/* (User-defined privated variables) : Defined at sp.c : 2057 */
	double t1;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double t2;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	double t3;		/* (User-defined privated variables) : Defined at sp.c : 2056 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rho_i)[37][37] = (__global double (*)[37][37])g_rho_i;
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;
	__global double (*ainv)[37][37] = (__global double (*)[37][37])g_ainv;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		ru1 = rho_i[i][j][k];
		uu = us[i][j][k];
		vv = vs[i][j][k];
		ww = ws[i][j][k];
		ac = speed[i][j][k];
		ac2inv = ainv[i][j][k] * ainv[i][j][k];
		r1 = rhs[0][i][j][k];
		r2 = rhs[1][i][j][k];
		r3 = rhs[2][i][j][k];
		r4 = rhs[3][i][j][k];
		r5 = rhs[4][i][j][k];
		t1 = c2 * ac2inv * (qs[i][j][k] * r1 - uu * r2 - vv * r3 -
				    ww * r4 + r5);
		t2 = bt * ru1 * (uu * r1 - r2);
		t3 = (bt * ru1 * ac) * t1;
		rhs[0][i][j][k] = r1 - t1;
		rhs[1][i][j][k] = -ru1 * (ww * r1 - r4);
		rhs[2][i][j][k] = ru1 * (vv * r1 - r3);
		rhs[3][i][j][k] = -t2 + t3;
		rhs[4][i][j][k] = t2 + t3;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2112 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void tzetar_0(__global double *g_us, __global double *g_vs,
		       __global double *g_ws, __global double *g_speed,
		       __global double *g_ainv, __global double *g_rhs,
		       __global double *g_u, double bt, __global double *g_qs,
		       double c2iv, int __ocl_k_bound, int __ocl_j_bound,
		       int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double xvel;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double yvel;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double zvel;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double ac;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double acinv;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double ac2u;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double r1;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double r2;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double r3;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double r4;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double r5;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double uzik1;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double btuz;		/* (User-defined privated variables) : Defined at sp.c : 2108 */
	double t1;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double t2;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	double t3;		/* (User-defined privated variables) : Defined at sp.c : 2107 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*us)[37][37] = (__global double (*)[37][37])g_us;
	__global double (*vs)[37][37] = (__global double (*)[37][37])g_vs;
	__global double (*ws)[37][37] = (__global double (*)[37][37])g_ws;
	__global double (*speed)[37][37] = (__global double (*)[37][37])g_speed;
	__global double (*ainv)[37][37] = (__global double (*)[37][37])g_ainv;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*u)[37][37][37] = (__global double (*)[37][37][37])g_u;
	__global double (*qs)[37][37] = (__global double (*)[37][37])g_qs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xvel = us[i][j][k];
		yvel = vs[i][j][k];
		zvel = ws[i][j][k];
		ac = speed[i][j][k];
		acinv = ainv[i][j][k];
		ac2u = ac * ac;
		r1 = rhs[0][i][j][k];
		r2 = rhs[1][i][j][k];
		r3 = rhs[2][i][j][k];
		r4 = rhs[3][i][j][k];
		r5 = rhs[4][i][j][k];
		uzik1 = u[0][i][j][k];
		btuz = bt * uzik1;
		t1 = btuz * acinv * (r4 + r5);
		t2 = r3 + t1;
		t3 = btuz * (r4 - r5);
		rhs[0][i][j][k] = t2;
		rhs[1][i][j][k] = -uzik1 * r2 + xvel * t2;
		rhs[2][i][j][k] = uzik1 * r1 + yvel * t2;
		rhs[3][i][j][k] = zvel * t2 + t3;
		rhs[4][i][j][k] =
		    uzik1 * (-xvel * r2 + yvel * r1) + qs[i][j][k] * t2 +
		    c2iv * ac2u * t1 + zvel * t3;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2458 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_0(__global double *g_lhs, int n, int i,
			__global double *g_rhs, int i1, int i2,
			int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2442 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2441 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i1][j][k] =
		    lhs[n + 2][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i1][j][k] =
		    lhs[n + 3][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i1][j][k] =
			    rhs[m][i1][j][k] - lhs[n +
						   1][i1][j][k] *
			    rhs[m][i][j][k];
		}
		lhs[n + 1][i2][j][k] =
		    lhs[n + 1][i2][j][k] - lhs[n + 0][i2][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 2][i2][j][k] =
		    lhs[n + 2][i2][j][k] - lhs[n + 0][i2][j][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i2][j][k] =
			    rhs[m][i2][j][k] - lhs[n +
						   0][i2][j][k] *
			    rhs[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2498 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_1(__global double *g_lhs, int n, int i,
			__global double *g_rhs, int i1, int __ocl_k_bound,
			int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2442 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2441 */
	double fac2;		/* (User-defined privated variables) : Defined at sp.c : 2442 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1.0 / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i1][j][k] =
		    lhs[n + 2][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i1][j][k] =
		    lhs[n + 3][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i1][j][k] =
			    rhs[m][i1][j][k] - lhs[n +
						   1][i1][j][k] *
			    rhs[m][i][j][k];
		}
		fac2 = 1. / lhs[n + 2][i1][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i1][j][k] = fac2 * rhs[m][i1][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2538 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_2(double fac1, __global double *g_lhs, int n, int i,
			__global double *g_rhs, int m, int i1, int i2,
			int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		lhs[n + 2][i1][j][k] =
		    lhs[n + 2][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i1][j][k] =
		    lhs[n + 3][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i1][j][k] =
		    rhs[m][i1][j][k] - lhs[n + 1][i1][j][k] * rhs[m][i][j][k];
		lhs[n + 1][i2][j][k] =
		    lhs[n + 1][i2][j][k] - lhs[n + 0][i2][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 2][i2][j][k] =
		    lhs[n + 2][i2][j][k] - lhs[n + 0][i2][j][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i2][j][k] =
		    rhs[m][i2][j][k] - lhs[n + 0][i2][j][k] * rhs[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2569 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_3(__global double *g_lhs, int n, int i,
			__global double *g_rhs, int m, int i1,
			int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2442 */
	double fac2;		/* (User-defined privated variables) : Defined at sp.c : 2442 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		lhs[n + 2][i1][j][k] =
		    lhs[n + 2][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i1][j][k] =
		    lhs[n + 3][i1][j][k] - lhs[n + 1][i1][j][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i1][j][k] =
		    rhs[m][i1][j][k] - lhs[n + 1][i1][j][k] * rhs[m][i][j][k];
		fac2 = 1. / lhs[n + 2][i1][j][k];
		rhs[m][i1][j][k] = fac2 * rhs[m][i1][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2601 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_4(__global double *g_rhs, int i, __global double *g_lhs,
			int n, int i1, int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(m < 3)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i1][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2613 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_5(__global double *g_rhs, int i, __global double *g_lhs,
			int i1, int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2) + 3;
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
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
	int n;			/* (User-defined privated variables) : Defined at sp.c : 2441 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		n = (m - 3 + 1) * 5;
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i1][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2634 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_6(__global double *g_rhs, int i, __global double *g_lhs,
			int n, int i1, int i2, int __ocl_k_bound,
			int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(m < 3)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i1][j][k] -
		    lhs[n + 4][i][j][k] * rhs[m][i2][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2656 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void x_solve_7(__global double *g_rhs, int m, int i,
			__global double *g_lhs, int n, int i1, int i2,
			int __ocl_k_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i1][j][k] -
		    lhs[n + 4][i][j][k] * rhs[m][i2][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2703 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_0(__global double *g_lhs, int n, int j,
			__global double *g_rhs, int j1, int j2,
			int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2688 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i][j1][k] =
		    lhs[n + 2][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j1][k] =
		    lhs[n + 3][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j1][k] =
			    rhs[m][i][j1][k] - lhs[n +
						   1][i][j1][k] *
			    rhs[m][i][j][k];
		}
		lhs[n + 1][i][j2][k] =
		    lhs[n + 1][i][j2][k] - lhs[n + 0][i][j2][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 2][i][j2][k] =
		    lhs[n + 2][i][j2][k] - lhs[n + 0][i][j2][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j2][k] =
			    rhs[m][i][j2][k] - lhs[n +
						   0][i][j2][k] *
			    rhs[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2743 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_1(__global double *g_lhs, int n, int j,
			__global double *g_rhs, int j1, int __ocl_k_bound,
			int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2688 */
	double fac2;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i][j1][k] =
		    lhs[n + 2][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j1][k] =
		    lhs[n + 3][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j1][k] =
			    rhs[m][i][j1][k] - lhs[n +
						   1][i][j1][k] *
			    rhs[m][i][j][k];
		}
		fac2 = 1. / lhs[n + 2][i][j1][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j1][k] = fac2 * rhs[m][i][j1][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2782 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_2(__global double *g_lhs, int n, int j,
			__global double *g_rhs, int m, int j1, int j2,
			int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		lhs[n + 2][i][j1][k] =
		    lhs[n + 2][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j1][k] =
		    lhs[n + 3][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i][j1][k] =
		    rhs[m][i][j1][k] - lhs[n + 1][i][j1][k] * rhs[m][i][j][k];
		lhs[n + 1][i][j2][k] =
		    lhs[n + 1][i][j2][k] - lhs[n + 0][i][j2][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 2][i][j2][k] =
		    lhs[n + 2][i][j2][k] - lhs[n + 0][i][j2][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i][j2][k] =
		    rhs[m][i][j2][k] - lhs[n + 0][i][j2][k] * rhs[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2813 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_3(__global double *g_lhs, int n, int j,
			__global double *g_rhs, int m, int j1,
			int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	double fac2;		/* (User-defined privated variables) : Defined at sp.c : 2689 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		lhs[n + 2][i][j1][k] =
		    lhs[n + 2][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j1][k] =
		    lhs[n + 3][i][j1][k] - lhs[n + 1][i][j1][k] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i][j1][k] =
		    rhs[m][i][j1][k] - lhs[n + 1][i][j1][k] * rhs[m][i][j][k];
		fac2 = 1. / lhs[n + 2][i][j1][k];
		rhs[m][i][j1][k] = fac2 * rhs[m][i][j1][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2844 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_4(__global double *g_rhs, int j, __global double *g_lhs,
			int n, int j1, int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	if (!(m < 3)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j1][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2856 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void y_solve_5(__global double *g_rhs, int j, __global double *g_lhs,
			int j1, int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2) + 3;
	if (!(k <= __ocl_k_bound)) {
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
	int n;			/* (User-defined privated variables) : Defined at sp.c : 2688 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		n = (m - 3 + 1) * 5;
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j1][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2877 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_6(__global double *g_rhs, int m, int j,
			__global double *g_lhs, int n, int j1, int j2,
			int __ocl_k_bound, int __ocl_i_bound,
			__global int *g_rd_log_rhs, __global int *g_wr_log_rhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_rhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_rhs;
	__global int (*wr_log_rhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j1][k] -
		    lhs[n + 4][i][j][k] * rhs[m][i][j2][k];
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_rhs[m][i][j][k]);
		rd_log_rhs[m][i][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2899 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_7(__global double *g_rhs, int m, int j,
			__global double *g_lhs, int n, int j1, int j2,
			int __ocl_k_bound, int __ocl_i_bound,
			__global int *g_rd_log_rhs, __global int *g_wr_log_rhs)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_rhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_rd_log_rhs;
	__global int (*wr_log_rhs)[37][37][37] =
	    (__global int (*)[37][37][37])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j1][k] -
		    lhs[n + 4][i][j][k] * rhs[m][i][j2][k];
		//-------------------------------------------
		// GPU TLS logs (BEGIN) 
		//-------------------------------------------
		atom_inc(&wr_log_rhs[m][i][j][k]);
		rd_log_rhs[m][i][j][k] = 1;
		//-------------------------------------------
		// GPU TLS logs (END)
		//-------------------------------------------
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2942 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_0(__global double *g_lhs, int n, __global double *g_rhs,
			int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int k1;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k2;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2929 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		k1 = k + 1;
		k2 = k + 2;
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i][j][k1] =
		    lhs[n + 2][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j][k1] =
		    lhs[n + 3][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k1] =
			    rhs[m][i][j][k1] - lhs[n +
						   1][i][j][k1] *
			    rhs[m][i][j][k];
		}
		lhs[n + 1][i][j][k2] =
		    lhs[n + 1][i][j][k2] - lhs[n + 0][i][j][k2] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 2][i][j][k2] =
		    lhs[n + 2][i][j][k2] - lhs[n + 0][i][j][k2] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k2] =
			    rhs[m][i][j][k2] - lhs[n +
						   0][i][j][k2] *
			    rhs[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2984 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_1(__global double *g_lhs, int n, int k,
			__global double *g_rhs, int k1, int __ocl_j_bound,
			int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2929 */
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	double fac2;		/* (User-defined privated variables) : Defined at sp.c : 2929 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		}
		lhs[n + 2][i][j][k1] =
		    lhs[n + 2][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j][k1] =
		    lhs[n + 3][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      4][i][j]
		    [k];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k1] =
			    rhs[m][i][j][k1] - lhs[n +
						   1][i][j][k1] *
			    rhs[m][i][j][k];
		}
		fac2 = 1. / lhs[n + 2][i][j][k1];
		for (m = 0; m < 3; m++) {
			rhs[m][i][j][k1] = fac2 * rhs[m][i][j][k1];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3017 of sp.c
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_2(__global int *grid_points, __global double *g_lhs,
			__global double *g_rhs, double fac2)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int m = get_global_id(0) + 3;
	if (!(m < 5)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int n;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int i;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int j;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k1;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k2;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	double fac1;		/* (User-defined privated variables) : Defined at sp.c : 2929 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 3017
		//-------------------------------------------
		int grid_points_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 3017
		//Candidates:
		//      grid_points[0]
		//-------------------------------------------
		grid_points_0 = grid_points[0];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		n = (m - 3 + 1) * 5;
		for (i = 1; i <= grid_points_0 /*grid_points[0] */  - 2; i++) {
			for (j = 1; j <= grid_points[1] - 2; j++) {
				for (k = 0; k <= grid_points[2] - 3; k++) {
					k1 = k + 1;
					k2 = k + 2;
					fac1 = 1. / lhs[n + 2][i][j][k];
					lhs[n + 3][i][j][k] =
					    fac1 * lhs[n + 3][i][j][k];
					lhs[n + 4][i][j][k] =
					    fac1 * lhs[n + 4][i][j][k];
					rhs[m][i][j][k] =
					    fac1 * rhs[m][i][j][k];
					lhs[n + 2][i][j][k1] =
					    lhs[n + 2][i][j][k1] - lhs[n +
								       1][i][j]
					    [k1] * lhs[n + 3][i][j][k];
					lhs[n + 3][i][j][k1] =
					    lhs[n + 3][i][j][k1] - lhs[n +
								       1][i][j]
					    [k1] * lhs[n + 4][i][j][k];
					rhs[m][i][j][k1] =
					    rhs[m][i][j][k1] - lhs[n +
								   1][i][j][k1]
					    * rhs[m][i][j][k];
					lhs[n + 1][i][j][k2] =
					    lhs[n + 1][i][j][k2] - lhs[n +
								       0][i][j]
					    [k2] * lhs[n + 3][i][j][k];
					lhs[n + 2][i][j][k2] =
					    lhs[n + 2][i][j][k2] - lhs[n +
								       0][i][j]
					    [k2] * lhs[n + 4][i][j][k];
					rhs[m][i][j][k2] =
					    rhs[m][i][j][k2] - lhs[n +
								   0][i][j][k2]
					    * rhs[m][i][j][k];
				}
			}
		}
		k = grid_points[2] - 2;
		k1 = grid_points[2] - 1;
		for (i = 1; i <= grid_points_0 /*grid_points[0] */  - 2; i++) {
			for (j = 1; j <= grid_points[1] - 2; j++) {
				fac1 = 1. / lhs[n + 2][i][j][k];
				lhs[n + 3][i][j][k] =
				    fac1 * lhs[n + 3][i][j][k];
				lhs[n + 4][i][j][k] =
				    fac1 * lhs[n + 4][i][j][k];
				rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
				lhs[n + 2][i][j][k1] =
				    lhs[n + 2][i][j][k1] - lhs[n +
							       1][i][j][k1] *
				    lhs[n + 3][i][j][k];
				lhs[n + 3][i][j][k1] =
				    lhs[n + 3][i][j][k1] - lhs[n +
							       1][i][j][k1] *
				    lhs[n + 4][i][j][k];
				rhs[m][i][j][k1] =
				    rhs[m][i][j][k1] - lhs[n +
							   1][i][j][k1] *
				    rhs[m][i][j][k];
				fac2 = 1. / lhs[n + 2][i][j][k1];
				rhs[m][i][j][k1] = fac2 * rhs[m][i][j][k1];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3082 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_3(__global double *g_rhs, int k, __global double *g_lhs,
			int n, int k1, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	if (!(m < 3)) {
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
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j][k1];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3094 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_4(__global double *g_rhs, int k, __global double *g_lhs,
			int k1, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2) + 3;
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
	int n;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		n = (m - 3 + 1) * 5;
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[n + 3][i][j][k] * rhs[m][i][j][k1];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3116 of sp.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_5(__global double *g_rhs, __global double *g_lhs, int n,
			int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k1;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k2;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 3; m++) {
			k1 = k + 1;
			k2 = k + 2;
			rhs[m][i][j][k] =
			    rhs[m][i][j][k] - lhs[n +
						  3][i][j][k] *
			    rhs[m][i][j][k1] - lhs[n +
						   4][i][j][k] *
			    rhs[m][i][j][k2];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3137 of sp.c
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void z_solve_6(__global double *g_rhs, __global double *g_lhs,
			int __ocl_i_bound, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2);
	if (!(i <= __ocl_i_bound)) {
		return;
	}
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(k <= __ocl_k_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int m;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int n;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k1;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	int k2;			/* (User-defined privated variables) : Defined at sp.c : 2928 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_rhs;
	__global double (*lhs)[37][37][37] =
	    (__global double (*)[37][37][37])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 3; m < 5; m++) {
			n = (m - 3 + 1) * 5;
			k1 = k + 1;
			k2 = k + 2;
			rhs[m][i][j][k] =
			    rhs[m][i][j][k] - lhs[n +
						  3][i][j][k] *
			    rhs[m][i][j][k1] - lhs[n +
						   4][i][j][k] *
			    rhs[m][i][j][k2];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
