//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Mon Aug  6 14:02:21 2012
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

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 312 of cg.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void main_0(__global int *colidx, int firstcol, __global int *rowstr,
		     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + rowstr[j];
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		colidx[k] = colidx[k] - firstcol + 1;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 322 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_1(__global double *x, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		x[i] = 1.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void main_2_reduction_step0(__global double *__ocl_part_norm_temp11,
				     __global double *__ocl_part_norm_temp12,
				     unsigned int offset, unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_norm_temp11[i] = 0.0;
	__ocl_part_norm_temp12[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 354 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_2_reduction_step1(__global double *x, __global double *z,
				     int __ocl_j_bound,
				     __global double *__ocl_part_norm_temp11,
				     __global double *__ocl_part_norm_temp12)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double norm_temp11 = 0.0;	/* reduction variable, defined at: cg.c : 246 */
	double norm_temp12 = 0.0;	/* reduction variable, defined at: cg.c : 247 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		norm_temp11 = norm_temp11 + x[j] * z[j];
		norm_temp12 = norm_temp12 + z[j] * z[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_norm_temp11[__ocl_wb_idx] = norm_temp11;
		__ocl_part_norm_temp12[__ocl_wb_idx] = norm_temp12;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void main_2_reduction_step2(__global double4 * input_norm_temp11,
				     __global double *output_norm_temp11,
				     __global double4 * input_norm_temp12,
				     __global double *output_norm_temp12)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_norm_temp11[GROUP_SIZE];
	__local double4 sdata_norm_temp12[GROUP_SIZE];
	sdata_norm_temp11[tid] = input_norm_temp11[gid];
	sdata_norm_temp12[tid] = input_norm_temp12[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_norm_temp11[tid] += sdata_norm_temp11[tid + s];
			sdata_norm_temp12[tid] += sdata_norm_temp12[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_norm_temp11[bid] =
		    (sdata_norm_temp11[0].x + sdata_norm_temp11[0].y +
		     sdata_norm_temp11[0].z + sdata_norm_temp11[0].w);
		output_norm_temp12[bid] =
		    (sdata_norm_temp12[0].x + sdata_norm_temp12[0].y +
		     sdata_norm_temp12[0].z + sdata_norm_temp12[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 365 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_3(__global double *x, double norm_temp12, __global double *z,
		     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		x[j] = norm_temp12 * z[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 375 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_4(__global double *x, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		x[i] = 1.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void main_5_reduction_step0(__global double *__ocl_part_norm_temp11,
				     __global double *__ocl_part_norm_temp12,
				     unsigned int offset, unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_norm_temp11[i] = 0.0;
	__ocl_part_norm_temp12[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 415 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_5_reduction_step1(__global double *x, __global double *z,
				     int __ocl_j_bound,
				     __global double *__ocl_part_norm_temp11,
				     __global double *__ocl_part_norm_temp12)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double norm_temp11 = 0.0;	/* reduction variable, defined at: cg.c : 246 */
	double norm_temp12 = 0.0;	/* reduction variable, defined at: cg.c : 247 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		norm_temp11 = norm_temp11 + x[j] * z[j];
		norm_temp12 = norm_temp12 + z[j] * z[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_norm_temp11[__ocl_wb_idx] = norm_temp11;
		__ocl_part_norm_temp12[__ocl_wb_idx] = norm_temp12;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void main_5_reduction_step2(__global double4 * input_norm_temp11,
				     __global double *output_norm_temp11,
				     __global double4 * input_norm_temp12,
				     __global double *output_norm_temp12)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_norm_temp11[GROUP_SIZE];
	__local double4 sdata_norm_temp12[GROUP_SIZE];
	sdata_norm_temp11[tid] = input_norm_temp11[gid];
	sdata_norm_temp12[tid] = input_norm_temp12[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_norm_temp11[tid] += sdata_norm_temp11[tid + s];
			sdata_norm_temp12[tid] += sdata_norm_temp12[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_norm_temp11[bid] =
		    (sdata_norm_temp11[0].x + sdata_norm_temp11[0].y +
		     sdata_norm_temp11[0].z + sdata_norm_temp11[0].w);
		output_norm_temp12[bid] =
		    (sdata_norm_temp12[0].x + sdata_norm_temp12[0].y +
		     sdata_norm_temp12[0].z + sdata_norm_temp12[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 439 of cg.c
//-------------------------------------------------------------------------------
__kernel void main_6(__global double *x, double norm_temp12, __global double *z,
		     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		x[j] = norm_temp12 * z[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 519 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_0(__global double *q, __global double *z,
			  __global double *r, __global double *x,
			  __global double *p, __global double *w,
			  int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
		w[j] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void conj_grad_1_reduction_step0(__global double *__ocl_part_rho,
					  unsigned int offset,
					  unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_rho[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 532 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_1_reduction_step1(__global double *x, int __ocl_j_bound,
					  __global double *__ocl_part_rho)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double rho = 0.0;	/* reduction variable, defined at: cg.c : 508 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rho = rho + x[j] * x[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_rho[__ocl_wb_idx] = rho;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void conj_grad_1_reduction_step2(__global double4 * input_rho,
					  __global double *output_rho)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_rho[GROUP_SIZE];
	sdata_rho[tid] = input_rho[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_rho[tid] += sdata_rho[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_rho[bid] =
		    (sdata_rho[0].x + sdata_rho[0].y + sdata_rho[0].z +
		     sdata_rho[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 564 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_2(__global int *rowstr, __global double *a,
			  __global double *p, __global int *colidx,
			  __global double *w, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double sum;		/* Defined at cg.c : 508 */
	int k;			/* Defined at cg.c : 509 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 564
		//-------------------------------------------
		int2 rowstr_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 564
		//Candidates:
		//      rowstr[j]
		//      rowstr[j + 1]
		//-------------------------------------------
		__global int *p_rowstr_0_0 = (__global int *)&rowstr[j];
		if ((unsigned long)p_rowstr_0_0 % 32 == 0) {
			rowstr_0 = vload2(0, p_rowstr_0_0);
		} else {
			rowstr_0.x = p_rowstr_0_0[0];
			p_rowstr_0_0++;
			rowstr_0.y = p_rowstr_0_0[0];
			p_rowstr_0_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		sum = 0.0;
		for (k = rowstr_0.x /*rowstr[j] */ ;
		     k < rowstr_0.y /*rowstr[j + 1] */ ; k++) {
			sum = sum + a[k] * p[colidx[k]];
		}
		w[j] = sum;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 573 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_3(__global double *q, __global double *w,
			  int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		q[j] = w[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 581 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_4(__global double *w, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		w[j] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void conj_grad_5_reduction_step0(__global double *__ocl_part_d,
					  unsigned int offset,
					  unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_d[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 589 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_5_reduction_step1(__global double *p,
					  __global double *q, int __ocl_j_bound,
					  __global double *__ocl_part_d)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double d = 0.0;		/* reduction variable, defined at: cg.c : 508 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		d = d + p[j] * q[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_d[__ocl_wb_idx] = d;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void conj_grad_5_reduction_step2(__global double4 * input_d,
					  __global double *output_d)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_d[GROUP_SIZE];
	sdata_d[tid] = input_d[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_d[tid] += sdata_d[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_d[bid] =
		    (sdata_d[0].x + sdata_d[0].y + sdata_d[0].z + sdata_d[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 609 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_6(__global double *z, double alpha, __global double *p,
			  __global double *r, __global double *q,
			  int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		z[j] = z[j] + alpha * p[j];
		r[j] = r[j] - alpha * q[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void conj_grad_7_reduction_step0(__global double *__ocl_part_rho,
					  unsigned int offset,
					  unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_rho[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 619 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_7_reduction_step1(__global double *r, int __ocl_j_bound,
					  __global double *__ocl_part_rho)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double rho = 0.0;	/* reduction variable, defined at: cg.c : 508 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rho = rho + r[j] * r[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_rho[__ocl_wb_idx] = rho;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void conj_grad_7_reduction_step2(__global double4 * input_rho,
					  __global double *output_rho)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_rho[GROUP_SIZE];
	sdata_rho[tid] = input_rho[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_rho[tid] += sdata_rho[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_rho[bid] =
		    (sdata_rho[0].x + sdata_rho[0].y + sdata_rho[0].z +
		     sdata_rho[0].w);
	}
}

//-------------------------------------------------------------------------------
//Loop defined at line 633 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_8(__global double *p, __global double *r, double beta,
			  int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		p[j] = r[j] + beta * p[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 647 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_9(__global int *rowstr, __global double *a,
			  __global double *z, __global int *colidx,
			  __global double *w, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double d;		/* Defined at cg.c : 508 */
	int k;			/* Defined at cg.c : 509 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 647
		//-------------------------------------------
		int2 rowstr_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 647
		//Candidates:
		//      rowstr[j]
		//      rowstr[j + 1]
		//-------------------------------------------
		__global int *p_rowstr_1_0 = (__global int *)&rowstr[j];
		if ((unsigned long)p_rowstr_1_0 % 32 == 0) {
			rowstr_1 = vload2(0, p_rowstr_1_0);
		} else {
			rowstr_1.x = p_rowstr_1_0[0];
			p_rowstr_1_0++;
			rowstr_1.y = p_rowstr_1_0[0];
			p_rowstr_1_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		d = 0.0;
		for (k = rowstr_1.x /*rowstr[j] */ ;
		     k <= rowstr_1.y /*rowstr[j + 1] */  - 1; k++) {
			d = d + a[k] * z[colidx[k]];
		}
		w[j] = d;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 656 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_10(__global double *r, __global double *w,
			   int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
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
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		r[j] = w[j];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void conj_grad_11_reduction_step0(__global double *__ocl_part_sum,
					   unsigned int offset,
					   unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_sum[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 664 of cg.c
//-------------------------------------------------------------------------------
__kernel void conj_grad_11_reduction_step1(__global double *x,
					   __global double *r,
					   int __ocl_j_bound,
					   __global double *__ocl_part_sum)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	if (!(j <= __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double d;		/* Defined at cg.c : 508 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double sum = 0.0;	/* reduction variable, defined at: cg.c : 508 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		d = x[j] - r[j];
		sum = sum + d * d;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_sum[__ocl_wb_idx] = sum;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void conj_grad_11_reduction_step2(__global double4 * input_sum,
					   __global double *output_sum)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_sum[GROUP_SIZE];
	sdata_sum[tid] = input_sum[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_sum[tid] += sdata_sum[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_sum[bid] =
		    (sdata_sum[0].x + sdata_sum[0].y + sdata_sum[0].z +
		     sdata_sum[0].w);
	}
}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
