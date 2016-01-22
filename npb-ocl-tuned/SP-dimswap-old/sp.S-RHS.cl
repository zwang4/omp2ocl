//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Tue Aug  7 13:28:29 2012
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
//Loop defined at line 203 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void add_0(__global double *g_u, __global double *g_rhs,
		    int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
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
	int m;			/* Defined at sp.c : 193 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		u[m][i][j][k] = u[m][i][j][k] + rhs[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 870 of sp.c
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
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 887 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsinit_1(__global double *g_lhs, int n, int __ocl_k_bound,
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
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[5 * n + 2][i][j][k] = 1.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 916 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_0(double c3c4, __global double *g_rho_i,
		     __global double *g_cv, __global double *g_us,
		     __global double *g_rhon, double dx2, double con43,
		     double dx5, double c1c5, double dxmax, double dx1,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double ru1;		/* Defined at sp.c : 909 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*rhon)[12][12] = (__global double (*)[12][12])g_rhon;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		ru1 = c3c4 * rho_i[i][j][k];
		cv[i][j][k] = us[i][j][k];
		rhon[i][j][k] =
		    (((dx2 + con43 * ru1) >
		      ((((dx5 + c1c5 * ru1) >
			 ((((dxmax + ru1) >
			    (dx1)) ? (dxmax + ru1) : (dx1)))) ? (dx5 +
								 c1c5 *
								 ru1)
			: ((((dxmax + ru1) >
			     (dx1)) ? (dxmax + ru1) : (dx1)))))) ? (dx2 +
								    con43 *
								    ru1)
		     : ((((dx5 + c1c5 * ru1) >
			  ((((dxmax + ru1) >
			     (dx1)) ? (dxmax + ru1) : (dx1)))) ? (dx5 +
								  c1c5 *
								  ru1)
			 : ((((dxmax + ru1) >
			      (dx1)) ? (dxmax + ru1) : (dx1))))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 931 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_1(__global double *g_lhs, double dttx2,
		     __global double *g_cv, double dttx1,
		     __global double *g_rhon, double c2dttx1, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*rhon)[12][12] = (__global double (*)[12][12])g_rhon;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = 0.0;
		lhs[1][i][j][k] =
		    -dttx2 * cv[i - 1][j][k] - dttx1 * rhon[i - 1][j][k];
		lhs[2][i][j][k] = 1.0 + c2dttx1 * rhon[i][j][k];
		lhs[3][i][j][k] =
		    dttx2 * cv[i + 1][j][k] - dttx1 * rhon[i + 1][j][k];
		lhs[4][i][j][k] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 952 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_2(__global double *g_lhs, int i, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 968 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_3(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
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
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 985 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_4(__global double *g_lhs, int i, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1006 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_5(__global double *g_lhs, double dttx2,
		     __global double *g_speed, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1008
		//-------------------------------------------
		double speed_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1008
		//Candidates:
		//      speed[i - 1][j][k]
		//-------------------------------------------
		speed_0 = speed[i - 1][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dttx2 * speed_0 /*speed[i - 1][j][k] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dttx2 * speed[i + 1][j][k];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dttx2 * speed_0 /*speed[i - 1][j][k] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dttx2 * speed[i + 1][j][k];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1048 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_0(double c3c4, __global double *g_rho_i,
		     __global double *g_cv, __global double *g_vs,
		     __global double *g_rhoq, double dy3, double con43,
		     double dy5, double c1c5, double dymax, double dy1,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1);
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
	double ru1;		/* Defined at sp.c : 1041 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*rhoq)[12][12] = (__global double (*)[12][12])g_rhoq;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		ru1 = c3c4 * rho_i[i][j][k];
		cv[i][j][k] = vs[i][j][k];
		rhoq[i][j][k] =
		    (((dy3 + con43 * ru1) >
		      ((((dy5 + c1c5 * ru1) >
			 ((((dymax + ru1) >
			    (dy1)) ? (dymax + ru1) : (dy1)))) ? (dy5 +
								 c1c5 *
								 ru1)
			: ((((dymax + ru1) >
			     (dy1)) ? (dymax + ru1) : (dy1)))))) ? (dy3 +
								    con43 *
								    ru1)
		     : ((((dy5 + c1c5 * ru1) >
			  ((((dymax + ru1) >
			     (dy1)) ? (dymax + ru1) : (dy1)))) ? (dy5 +
								  c1c5 *
								  ru1)
			 : ((((dymax + ru1) >
			      (dy1)) ? (dymax + ru1) : (dy1))))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1063 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_1(__global double *g_lhs, double dtty2,
		     __global double *g_cv, double dtty1,
		     __global double *g_rhoq, double c2dtty1, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*rhoq)[12][12] = (__global double (*)[12][12])g_rhoq;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		lhs[0][i][j][k] = 0.0;
		lhs[1][i][j][k] =
		    -dtty2 * cv[i][j - 1][k] - dtty1 * rhoq[i][j - 1][k];
		lhs[2][i][j][k] = 1.0 + c2dtty1 * rhoq[i][j][k];
		lhs[3][i][j][k] =
		    dtty2 * cv[i][j + 1][k] - dtty1 * rhoq[i][j + 1][k];
		lhs[4][i][j][k] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1084 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_2(__global double *g_lhs, int j, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1101 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_3(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
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
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1117 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_4(__global double *g_lhs, int j, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1137 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_5(__global double *g_lhs, double dtty2,
		     __global double *g_speed, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1139
		//-------------------------------------------
		double speed_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1139
		//Candidates:
		//      speed[i][j - 1][k]
		//-------------------------------------------
		speed_1 = speed[i][j - 1][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dtty2 * speed_1 /*speed[i][j - 1][k] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dtty2 * speed[i][j + 1][k];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dtty2 * speed_1 /*speed[i][j - 1][k] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dtty2 * speed[i][j + 1][k];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1179 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_0(double c3c4, __global double *g_rho_i,
		     __global double *g_cv, __global double *g_ws,
		     __global double *g_rhos, double dz4, double con43,
		     double dz5, double c1c5, double dzmax, double dz1,
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
	double ru1;		/* Defined at sp.c : 1172 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*rhos)[12][12] = (__global double (*)[12][12])g_rhos;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		ru1 = c3c4 * rho_i[i][j][k];
		cv[i][j][k] = ws[i][j][k];
		rhos[i][j][k] =
		    (((dz4 + con43 * ru1) >
		      ((((dz5 + c1c5 * ru1) >
			 ((((dzmax + ru1) >
			    (dz1)) ? (dzmax + ru1) : (dz1)))) ? (dz5 +
								 c1c5 *
								 ru1)
			: ((((dzmax + ru1) >
			     (dz1)) ? (dzmax + ru1) : (dz1)))))) ? (dz4 +
								    con43 *
								    ru1)
		     : ((((dz5 + c1c5 * ru1) >
			  ((((dzmax + ru1) >
			     (dz1)) ? (dzmax + ru1) : (dz1)))) ? (dz5 +
								  c1c5 *
								  ru1)
			 : ((((dzmax + ru1) >
			      (dz1)) ? (dzmax + ru1) : (dz1))))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1194 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_1(__global double *g_lhs, double dttz2,
		     __global double *g_cv, double dttz1,
		     __global double *g_rhos, double c2dttz1, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*cv)[12][12] = (__global double (*)[12][12])g_cv;
	__global double (*rhos)[12][12] = (__global double (*)[12][12])g_rhos;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1196
		//-------------------------------------------
		double2 rhos_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1196
		//Candidates:
		//      rhos[i][j][k - 1]
		//      rhos[i][j][k]
		//-------------------------------------------
		__global double *p_rhos_0_0 =
		    (__global double *)&rhos[i][j][k - 1];
		if ((unsigned long)p_rhos_0_0 % 64 == 0) {
			rhos_0 = vload2(0, p_rhos_0_0);
		} else {
			rhos_0.x = p_rhos_0_0[0];
			p_rhos_0_0++;
			rhos_0.y = p_rhos_0_0[0];
			p_rhos_0_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0][i][j][k] = 0.0;
		lhs[1][i][j][k] =
		    -dttz2 * cv[i][j][k - 1] -
		    dttz1 * rhos_0.x /*rhos[i][j][k - 1] */ ;
		lhs[2][i][j][k] = 1.0 + c2dttz1 * rhos_0.y /*rhos[i][j][k] */ ;
		lhs[3][i][j][k] =
		    dttz2 * cv[i][j][k + 1] - dttz1 * rhos[i][j][k + 1];
		lhs[4][i][j][k] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1215 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_2(__global double *g_lhs, int k, double comz5, double comz4,
		     double comz1, double comz6, int __ocl_j_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1232 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_3(__global double *g_lhs, double comz1, double comz4,
		     double comz6, int __ocl_k_bound, int __ocl_j_bound,
		     int __ocl_i_bound)
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
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1249 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_4(__global double *g_lhs, int k, double comz1, double comz4,
		     double comz6, double comz5, int __ocl_j_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1269 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_5(__global double *g_lhs, double dttz2,
		     __global double *g_speed, int __ocl_k_bound,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1271
		//-------------------------------------------
		double speed_2;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1271
		//Candidates:
		//      speed[i][j][k - 1]
		//-------------------------------------------
		speed_2 = speed[i][j][k - 1];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		lhs[0 + 5][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 5][i][j][k] =
		    lhs[1][i][j][k] - dttz2 * speed_2 /*speed[i][j][k - 1] */ ;
		lhs[2 + 5][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 5][i][j][k] =
		    lhs[3][i][j][k] + dttz2 * speed[i][j][k + 1];
		lhs[4 + 5][i][j][k] = lhs[4][i][j][k];
		lhs[0 + 10][i][j][k] = lhs[0][i][j][k];
		lhs[1 + 10][i][j][k] =
		    lhs[1][i][j][k] + dttz2 * speed_2 /*speed[i][j][k - 1] */ ;
		lhs[2 + 10][i][j][k] = lhs[2][i][j][k];
		lhs[3 + 10][i][j][k] =
		    lhs[3][i][j][k] - dttz2 * speed[i][j][k + 1];
		lhs[4 + 10][i][j][k] = lhs[4][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1309 of sp.c
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
	double r1;		/* Defined at sp.c : 1305 */
	double r2;		/* Defined at sp.c : 1305 */
	double r3;		/* Defined at sp.c : 1305 */
	double r4;		/* Defined at sp.c : 1305 */
	double r5;		/* Defined at sp.c : 1305 */
	double t1;		/* Defined at sp.c : 1305 */
	double t2;		/* Defined at sp.c : 1305 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 1351 of sp.c
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
	double r1;		/* Defined at sp.c : 1346 */
	double r2;		/* Defined at sp.c : 1346 */
	double r3;		/* Defined at sp.c : 1346 */
	double r4;		/* Defined at sp.c : 1346 */
	double r5;		/* Defined at sp.c : 1346 */
	double t1;		/* Defined at sp.c : 1346 */
	double t2;		/* Defined at sp.c : 1346 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 1395 of sp.c
//The nested loops were swaped. 
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
	double rho_inv;		/* Defined at sp.c : 1384 */
	double aux;		/* Defined at sp.c : 1384 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*square)[13][13] =
	    (__global double (*)[13][13])g_square;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	__global double (*ainv)[13][13] = (__global double (*)[13][13])g_ainv;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1397
		//-------------------------------------------
		double u_0[3];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1397
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
//Loop defined at line 1427 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_1(__global double *g_rhs, __global double *g_forcing,
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
	int m;			/* Defined at sp.c : 1383 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*forcing)[13][13][13] =
	    (__global double (*)[13][13][13])g_forcing;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rhs[m][i][j][k] = forcing[m][i][j][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1444 of sp.c
//The nested loops were swaped. 
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
	double uijk;		/* Defined at sp.c : 1384 */
	double up1;		/* Defined at sp.c : 1384 */
	double um1;		/* Defined at sp.c : 1384 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*square)[13][13] =
	    (__global double (*)[13][13])g_square;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1446
		//-------------------------------------------
		double u_1[8];
		double square_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1446
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
//Loop defined at line 1507 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1521 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1534 of sp.c
//The nested loops were swaped. 
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
	int m;			/* Defined at sp.c : 1383 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i - 2][j][k] -
					      4.0 * u[m][i - 1][j][k] +
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
//Loop defined at line 1551 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1565 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1583 of sp.c
//The nested loops were swaped. 
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
	double vijk;		/* Defined at sp.c : 1384 */
	double vp1;		/* Defined at sp.c : 1384 */
	double vm1;		/* Defined at sp.c : 1384 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*square)[13][13] =
	    (__global double (*)[13][13])g_square;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1585
		//-------------------------------------------
		double u_2[8];
		double square_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1585
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
//Loop defined at line 1642 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1656 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1669 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_11(__global double *g_rhs, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_j_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 3 * 1;
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
	int m;			/* Defined at sp.c : 1383 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i][j - 2][k] -
					      4.0 * u[m][i][j - 1][k] +
					      6.0 * u[m][i][j][k] -
					      4.0 * u[m][i][j + 1][k] +
					      u[m][i][j + 2][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1686 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1700 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
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
//Loop defined at line 1718 of sp.c
//The nested loops were swaped. 
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
	double wijk;		/* Defined at sp.c : 1385 */
	double wp1;		/* Defined at sp.c : 1385 */
	double wm1;		/* Defined at sp.c : 1385 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*square)[13][13] =
	    (__global double (*)[13][13])g_square;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1720
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
		//Prefetching (BEGIN) : 1720
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
//Loop defined at line 1783 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][k][j] =
		    rhs[m][i][k][j] - dssp * (5.0 * u[m][i][k][j] -
					      4.0 * u[m][i][k + 1][j] +
					      u[m][i][k + 2][j]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1803 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][k][j] =
		    rhs[m][i][k][j] - dssp * (-4.0 * u[m][i][k - 1][j] +
					      6.0 * u[m][i][k][j] -
					      4.0 * u[m][i][k + 1][j] +
					      u[m][i][k + 2][j]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1822 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_17(__global double *g_rhs, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_k_bound, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 3 * 1;
	int i = get_global_id(2) + 1;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
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
	int m;			/* Defined at sp.c : 1383 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rhs[m][i][k][j] =
		    rhs[m][i][k][j] - dssp * (u[m][i][k - 2][j] -
					      4.0 * u[m][i][k - 1][j] +
					      6.0 * u[m][i][k][j] -
					      4.0 * u[m][i][k + 1][j] +
					      u[m][i][k + 2][j]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1846 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][k][j] =
		    rhs[m][i][k][j] - dssp * (u[m][i][k - 2][j] -
					      4.0 * u[m][i][k - 1][j] +
					      6.0 * u[m][i][k][j] -
					      4.0 * u[m][i][k + 1][j]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1866 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][k][j] =
		    rhs[m][i][k][j] - dssp * (u[m][i][k - 2][j] -
					      4.0 * u[m][i][k - 1][j] +
					      5.0 * u[m][i][k][j]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1890 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_20(__global double *g_rhs, double dt,
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
	int m;			/* Defined at sp.c : 1383 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (m = 0; m < 5; m++) {
		rhs[m][i][j][k] = rhs[m][i][j][k] * dt;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2119 of sp.c
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
	double ru1;		/* Defined at sp.c : 2114 */
	double uu;		/* Defined at sp.c : 2114 */
	double vv;		/* Defined at sp.c : 2114 */
	double ww;		/* Defined at sp.c : 2114 */
	double ac;		/* Defined at sp.c : 2114 */
	double ac2inv;		/* Defined at sp.c : 2115 */
	double r1;		/* Defined at sp.c : 2114 */
	double r2;		/* Defined at sp.c : 2114 */
	double r3;		/* Defined at sp.c : 2114 */
	double r4;		/* Defined at sp.c : 2115 */
	double r5;		/* Defined at sp.c : 2115 */
	double t1;		/* Defined at sp.c : 2114 */
	double t2;		/* Defined at sp.c : 2114 */
	double t3;		/* Defined at sp.c : 2114 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rho_i)[13][13] = (__global double (*)[13][13])g_rho_i;
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	__global double (*ainv)[13][13] = (__global double (*)[13][13])g_ainv;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
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
//Loop defined at line 2170 of sp.c
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
	double xvel;		/* Defined at sp.c : 2165 */
	double yvel;		/* Defined at sp.c : 2165 */
	double zvel;		/* Defined at sp.c : 2165 */
	double ac;		/* Defined at sp.c : 2165 */
	double acinv;		/* Defined at sp.c : 2166 */
	double ac2u;		/* Defined at sp.c : 2166 */
	double r1;		/* Defined at sp.c : 2165 */
	double r2;		/* Defined at sp.c : 2165 */
	double r3;		/* Defined at sp.c : 2165 */
	double r4;		/* Defined at sp.c : 2166 */
	double r5;		/* Defined at sp.c : 2166 */
	double uzik1;		/* Defined at sp.c : 2166 */
	double btuz;		/* Defined at sp.c : 2166 */
	double t1;		/* Defined at sp.c : 2165 */
	double t2;		/* Defined at sp.c : 2165 */
	double t3;		/* Defined at sp.c : 2165 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*us)[13][13] = (__global double (*)[13][13])g_us;
	__global double (*vs)[13][13] = (__global double (*)[13][13])g_vs;
	__global double (*ws)[13][13] = (__global double (*)[13][13])g_ws;
	__global double (*speed)[13][13] = (__global double (*)[13][13])g_speed;
	__global double (*ainv)[13][13] = (__global double (*)[13][13])g_ainv;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*u)[13][13][13] = (__global double (*)[13][13][13])g_u;
	__global double (*qs)[13][13] = (__global double (*)[13][13])g_qs;
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
//Loop defined at line 2517 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2501 */
	int m;			/* Defined at sp.c : 2500 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2557 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2501 */
	int m;			/* Defined at sp.c : 2500 */
	double fac2;		/* Defined at sp.c : 2501 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2597 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void x_solve_2(__global double *g_lhs, int n, int i,
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
	double fac1;		/* Defined at sp.c : 2501 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2628 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2501 */
	double fac2;		/* Defined at sp.c : 2501 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2660 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2672 of sp.c
//The nested loops were swaped. 
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
	int n;			/* Defined at sp.c : 2500 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2693 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2715 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2762 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2748 */
	int m;			/* Defined at sp.c : 2747 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2802 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2748 */
	int m;			/* Defined at sp.c : 2747 */
	double fac2;		/* Defined at sp.c : 2748 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2841 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2748 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2872 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2748 */
	double fac2;		/* Defined at sp.c : 2748 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 2903 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2915 of sp.c
//The nested loops were swaped. 
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
	int n;			/* Defined at sp.c : 2747 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 2936 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_6(__global double *g_rhs, int m, int j,
			__global double *g_lhs, int n, int j1, int j2,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2958 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_7(__global double *g_rhs, int m, int j,
			__global double *g_lhs, int n, int j1, int j2,
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3006 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_0(__global int *grid_points, __global double *g_lhs,
			int n, __global double *g_rhs, int __ocl_j_bound,
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
	int k;			/* Defined at sp.c : 2987 */
	int k1;			/* Defined at sp.c : 2987 */
	int k2;			/* Defined at sp.c : 2987 */
	double fac1;		/* Defined at sp.c : 2988 */
	int m;			/* Defined at sp.c : 2987 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = 0; k <= grid_points[2] - 3; k++) {
			k1 = k + 1;
			k2 = k + 2;
			fac1 = 1. / lhs[n + 2][i][j][k];
			lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
			lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
			for (m = 0; m < 3; m++) {
				rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
			}
			lhs[n + 2][i][j][k1] =
			    lhs[n + 2][i][j][k1] - lhs[n +
						       1][i][j][k1] * lhs[n +
									  3][i]
			    [j][k];
			lhs[n + 3][i][j][k1] =
			    lhs[n + 3][i][j][k1] - lhs[n +
						       1][i][j][k1] * lhs[n +
									  4][i]
			    [j][k];
			for (m = 0; m < 3; m++) {
				rhs[m][i][j][k1] =
				    rhs[m][i][j][k1] - lhs[n +
							   1][i][j][k1] *
				    rhs[m][i][j][k];
			}
			lhs[n + 1][i][j][k2] =
			    lhs[n + 1][i][j][k2] - lhs[n +
						       0][i][j][k2] * lhs[n +
									  3][i]
			    [j][k];
			lhs[n + 2][i][j][k2] =
			    lhs[n + 2][i][j][k2] - lhs[n +
						       0][i][j][k2] * lhs[n +
									  4][i]
			    [j][k];
			for (m = 0; m < 3; m++) {
				rhs[m][i][j][k2] =
				    rhs[m][i][j][k2] - lhs[n +
							   0][i][j][k2] *
				    rhs[m][i][j][k];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3073 of sp.c
//The nested loops were swaped. 
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
	double fac1;		/* Defined at sp.c : 2988 */
	int m;			/* Defined at sp.c : 2987 */
	double fac2;		/* Defined at sp.c : 2988 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
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
//Loop defined at line 3131 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_2(__global int *grid_points, __global double *g_lhs,
			__global double *g_rhs, double fac2, int __ocl_j_bound,
			int __ocl_i_bound)
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
	int n;			/* Defined at sp.c : 2987 */
	int k;			/* Defined at sp.c : 2987 */
	int k1;			/* Defined at sp.c : 2987 */
	int k2;			/* Defined at sp.c : 2987 */
	double fac1;		/* Defined at sp.c : 2988 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		n = (m - 3 + 1) * 5;
		for (k = 0; k <= grid_points[2] - 3; k++) {
			k1 = k + 1;
			k2 = k + 2;
			fac1 = 1. / lhs[n + 2][i][j][k];
			lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
			lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
			rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
			lhs[n + 2][i][j][k1] =
			    lhs[n + 2][i][j][k1] - lhs[n +
						       1][i][j][k1] * lhs[n +
									  3][i]
			    [j][k];
			lhs[n + 3][i][j][k1] =
			    lhs[n + 3][i][j][k1] - lhs[n +
						       1][i][j][k1] * lhs[n +
									  4][i]
			    [j][k];
			rhs[m][i][j][k1] =
			    rhs[m][i][j][k1] - lhs[n +
						   1][i][j][k1] *
			    rhs[m][i][j][k];
			lhs[n + 1][i][j][k2] =
			    lhs[n + 1][i][j][k2] - lhs[n +
						       0][i][j][k2] * lhs[n +
									  3][i]
			    [j][k];
			lhs[n + 2][i][j][k2] =
			    lhs[n + 2][i][j][k2] - lhs[n +
						       0][i][j][k2] * lhs[n +
									  4][i]
			    [j][k];
			rhs[m][i][j][k2] =
			    rhs[m][i][j][k2] - lhs[n +
						   0][i][j][k2] *
			    rhs[m][i][j][k];
		}
		k = grid_points[2] - 2;
		k1 = grid_points[2] - 1;
		fac1 = 1. / lhs[n + 2][i][j][k];
		lhs[n + 3][i][j][k] = fac1 * lhs[n + 3][i][j][k];
		lhs[n + 4][i][j][k] = fac1 * lhs[n + 4][i][j][k];
		rhs[m][i][j][k] = fac1 * rhs[m][i][j][k];
		lhs[n + 2][i][j][k1] =
		    lhs[n + 2][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      3][i][j]
		    [k];
		lhs[n + 3][i][j][k1] =
		    lhs[n + 3][i][j][k1] - lhs[n + 1][i][j][k1] * lhs[n +
								      4][i][j]
		    [k];
		rhs[m][i][j][k1] =
		    rhs[m][i][j][k1] - lhs[n + 1][i][j][k1] * rhs[m][i][j][k];
		fac2 = 1. / lhs[n + 2][i][j][k1];
		rhs[m][i][j][k1] = fac2 * rhs[m][i][j][k1];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3236 of sp.c
//The nested loops were swaped. 
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
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 3253 of sp.c
//The nested loops were swaped. 
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
	int n;			/* Defined at sp.c : 2987 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
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
//Loop defined at line 3280 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_5(__global int *grid_points, __global double *g_rhs,
			__global double *g_lhs, int n, int __ocl_j_bound,
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
	int k;			/* Defined at sp.c : 2987 */
	int m;			/* Defined at sp.c : 2987 */
	int k1;			/* Defined at sp.c : 2987 */
	int k2;			/* Defined at sp.c : 2987 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = grid_points[2] - 3; k >= 0; k--) {
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3307 of sp.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_6(__global int *grid_points, __global double *g_rhs,
			__global double *g_lhs, int __ocl_j_bound,
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
	int k;			/* Defined at sp.c : 2987 */
	int m;			/* Defined at sp.c : 2987 */
	int n;			/* Defined at sp.c : 2987 */
	int k1;			/* Defined at sp.c : 2987 */
	int k2;			/* Defined at sp.c : 2987 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_rhs;
	__global double (*lhs)[13][13][13] =
	    (__global double (*)[13][13][13])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = grid_points[2] - 3; k >= 0; k--) {
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
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
