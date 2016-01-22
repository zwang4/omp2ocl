//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Mon Aug 13 10:13:42 2012
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
//Loop defined at line 643 of mg.c
//-------------------------------------------------------------------------------
__kernel void psinv_0(int i1, int n1, __global double *r1,
		      __global double *r_data, __global unsigned *r_idx,
		      unsigned int r_offset, __global double *r2,
		      __global double *u_data, __global unsigned *u_idx,
		      unsigned int u_offset, __global double *c,
		      int __ocl_i3_bound, int __ocl_i2_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + 1;
	int _ocl_i3_init = i3;
	int i2 = get_global_id(1) + 1;
	int _ocl_i2_init = i2;
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
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

	unsigned gsize_0 = get_global_size(0);
	unsigned gid_0 = get_global_id(0);
	unsigned gsize_1 = get_global_size(1);
	unsigned gid_1 = get_global_id(1);

	unsigned __ocl_mult_factor = (gsize_0 * gsize_1);
	unsigned __ocl_add_offset = ((gid_1 * gsize_0) + (gid_0));

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (i3 = _ocl_i3_init; i3 < __ocl_i3_bound; i3 += get_global_size(0))
		for (i2 = _ocl_i2_init; i2 < __ocl_i2_bound;
		     i2 += get_global_size(1)) {
			for (i1 = 0; i1 < n1; i1++) {
				r1[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 - 1)] + i3)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 + 1)] + i3)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)] + i3 - 1)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)] + i3 + 1)];
				r2[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 - 1)] + i3 - 1)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 + 1)] + i3 - 1)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 - 1)] + i3 + 1)] +
	 r_data[(r_idx[(r_idx[(r_offset + i1)] + i2 + 1)] + i3 + 1)];
			}
			for (i1 = 1; i1 < n1 - 1; i1++) {
				//-------------------------------------------
				//Declare prefetching Buffers (BEGIN) : 655
				//-------------------------------------------
				double2 c_0;
				//-------------------------------------------
				//Declare prefetching buffers (END)
				//-------------------------------------------
				//-------------------------------------------
				//Prefetching (BEGIN) : 655
				//Candidates:
				//      c[0]
				//      c[1]
				//-------------------------------------------
				__global double *p_c_0_0 =
				    c;// + (0) * __ocl_mult_factor +
				    __ocl_add_offset;
				if ((unsigned long)p_c_0_0 % 64 == 0) {
					c_0 = vload2(0, p_c_0_0);
				} else {
					c_0.x = p_c_0_0[0];
					p_c_0_0++;
					c_0.y = p_c_0_0[0];
					p_c_0_0++;
				}
				//-------------------------------------------
				//Prefetching (END)
				//-------------------------------------------

				u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] +
					i3)] =
				    u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)]
					    + i3)] +
				    c_0.x /*c[0] */  *
				    r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)]
					    + i3)] +
				    c_0.y /*c[1] */  *
				    (r_data
				     [(r_idx[(r_idx[(r_offset + i1 - 1)] + i2)]
				       + i3)] +
				     r_data[(r_idx
					     [(r_idx[(r_offset + i1 + 1)] +
					       i2)] + i3)] +
				     r1[(i1) * __ocl_mult_factor +
					__ocl_add_offset]) +
				    c[(2)] *
				    (r2
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + r1[(i1 -
							      1) *
							     __ocl_mult_factor +
							     __ocl_add_offset] +
				     r1[(i1 + 1) * __ocl_mult_factor +
					__ocl_add_offset]);
			}
		}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 716 of mg.c
//-------------------------------------------------------------------------------
__kernel void resid_0(int i1, int n1, __global double *u1,
		      __global double *u_data, __global unsigned *u_idx,
		      unsigned int u_offset, __global double *u2,
		      __global double *r_data, __global unsigned *r_idx,
		      unsigned int r_offset, __global double *v_data,
		      __global unsigned *v_idx, unsigned int v_offset,
		      __global double *a, int __ocl_i3_bound,
		      int __ocl_i2_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + 1;
	int _ocl_i3_init = i3;
	int i2 = get_global_id(1) + 1;
	int _ocl_i2_init = i2;
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
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

	unsigned gsize_0 = get_global_size(0);
	unsigned gid_0 = get_global_id(0);
	unsigned gsize_1 = get_global_size(1);
	unsigned gid_1 = get_global_id(1);

	unsigned __ocl_mult_factor = (gsize_0 * gsize_1);
	unsigned __ocl_add_offset = ((gid_1 * gsize_0) + (gid_0));

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (i3 = _ocl_i3_init; i3 < __ocl_i3_bound; i3 += get_global_size(0))
		for (i2 = _ocl_i2_init; i2 < __ocl_i2_bound;
		     i2 += get_global_size(1)) {
			for (i1 = 0; i1 < n1; i1++) {
				u1[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 - 1)] + i3)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 + 1)] + i3)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + i3 - 1)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + i3 + 1)];
				u2[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 - 1)] + i3 - 1)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 + 1)] + i3 - 1)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 - 1)] + i3 + 1)] +
	 u_data[(u_idx[(u_idx[(u_offset + i1)] + i2 + 1)] + i3 + 1)];
			}
			for (i1 = 1; i1 < n1 - 1; i1++) {
				r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)] +
					i3)] =
				    v_data[(v_idx[(v_idx[(v_offset + i1)] + i2)]
					    + i3)] -
				    a[(0)] *
				    u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)]
					    + i3)] -
				    a[(2)] *
				    (u2
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + u1[(i1 -
							      1) *
							     __ocl_mult_factor +
							     __ocl_add_offset] +
				     u1[(i1 + 1) * __ocl_mult_factor +
					__ocl_add_offset]) - a[(3)] * (u2[(i1 -
									   1) *
									  __ocl_mult_factor
									  +
									  __ocl_add_offset]
								       +
								       u2[(i1 +
									   1) *
									  __ocl_mult_factor
									  +
									  __ocl_add_offset]);
			}
		}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 804 of mg.c
//-------------------------------------------------------------------------------
__kernel void rprj3_0(int d3, int j2, int m2j, int d2, int j1, int m1j, int d1,
		      __global double *xx1, __global double *r_data,
		      __global unsigned *r_idx, unsigned int r_offset,
		      __global double *yy1, __global double *s_data,
		      __global unsigned *s_idx, unsigned int s_offset,
		      int __ocl_j3_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j3 = get_global_id(0) + 1;
	if (!(j3 < __ocl_j3_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int i3;			/* Defined at mg.c : 781 */
	int i2;			/* Defined at mg.c : 781 */
	int i1;			/* Defined at mg.c : 781 */
	double y2;		/* Defined at mg.c : 783 */
	double x2;		/* Defined at mg.c : 783 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	unsigned gsize_0 = get_global_size(0);
	unsigned gid_0 = get_global_id(0);

	unsigned __ocl_mult_factor = (gsize_0);
	unsigned __ocl_add_offset = ((gid_0));

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		i3 = 2 * j3 - d3;
		for (j2 = 1; j2 < m2j - 1; j2++) {
			i2 = 2 * j2 - d2;
			for (j1 = 1; j1 < m1j; j1++) {
				i1 = 2 * j1 - d1;
				xx1[(i1) * __ocl_mult_factor +
				    __ocl_add_offset] =
				    r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)]
					    + i3 + 1)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1)] + i2 +
					      2)] + i3 + 1)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1)] + i2 +
					      1)] + i3)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1)] + i2 +
					      1)] + i3 + 2)];
				yy1[(i1) * __ocl_mult_factor +
				    __ocl_add_offset] =
				    r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)]
					    + i3)] +
				    r_data[(r_idx[(r_idx[(r_offset + i1)] + i2)]
					    + i3 + 2)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1)] + i2 +
					      2)] + i3)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1)] + i2 +
					      2)] + i3 + 2)];
			}
			for (j1 = 1; j1 < m1j - 1; j1++) {
				i1 = 2 * j1 - d1;
				y2 = r_data[(r_idx
					     [(r_idx[(r_offset + i1 + 1)] +
					       i2)] + i3)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] +
					      i2)] + i3 + 2)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      2)] + i3)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      2)] + i3 + 2)];
				x2 = r_data[(r_idx
					     [(r_idx[(r_offset + i1 + 1)] +
					       i2)] + i3 + 1)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      2)] + i3 + 1)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      1)] + i3)] +
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      1)] + i3 + 2)];
				s_data[(s_idx[(s_idx[(s_offset + j1)] + j2)] +
					j3)] =
				    0.5 *
				    r_data[(r_idx
					    [(r_idx[(r_offset + i1 + 1)] + i2 +
					      1)] + i3 + 1)] +
				    0.25 *
				    (r_data
				     [(r_idx[(r_idx[(r_offset + i1)] + i2 + 1)]
				       + i3 + 1)] +
				     r_data[(r_idx
					     [(r_idx[(r_offset + i1 + 2)] + i2 +
					       1)] + i3 + 1)] + x2) +
				    0.125 *
				    (xx1
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + xx1[(i1 +
							       2) *
							      __ocl_mult_factor
							      +
							      __ocl_add_offset]
				     + y2) +
				    0.0625 *
				    (yy1
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + yy1[(i1 +
							       2) *
							      __ocl_mult_factor
							      +
							      __ocl_add_offset]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 893 of mg.c
//-------------------------------------------------------------------------------
__kernel void interp_0(int i1, int mm1, __global double *z1,
		       __global double *z_data, __global unsigned *z_idx,
		       unsigned int z_offset, __global double *z2,
		       __global double *z3, __global double *u_data,
		       __global unsigned *u_idx, unsigned int u_offset,
		       int __ocl_i3_bound, int __ocl_i2_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0);
	int _ocl_i3_init = i3;
	int i2 = get_global_id(1);
	int _ocl_i2_init = i2;
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
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

	unsigned gsize_0 = get_global_size(0);
	unsigned gid_0 = get_global_id(0);
	unsigned gsize_1 = get_global_size(1);
	unsigned gid_1 = get_global_id(1);

	unsigned __ocl_mult_factor = (gsize_0 * gsize_1);
	unsigned __ocl_add_offset = ((gid_1 * gsize_0) + (gid_0));

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (i3 = _ocl_i3_init; i3 < __ocl_i3_bound; i3 += get_global_size(0))
		for (i2 = _ocl_i2_init; i2 < __ocl_i2_bound;
		     i2 += get_global_size(1)) {
			for (i1 = 0; i1 < mm1; i1++) {
				z1[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2 + 1)] + i3)] +
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)] + i3)];
				z2[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)] + i3 + 1)] +
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)] + i3)];
				z3[(i1) * __ocl_mult_factor +
				   __ocl_add_offset] =
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2 + 1)] + i3 + 1)] +
	 z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)] + i3 + 1)] +
	 z1[(i1) * __ocl_mult_factor + __ocl_add_offset];
			}
			for (i1 = 0; i1 < mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1)] +
					  2 * i2)] + 2 * i3)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1)] +
					      2 * i2)] + 2 * i3)] +
				    z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)]
					    + i3)];
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 + 1)] +
					  2 * i2)] + 2 * i3)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1 + 1)] +
					      2 * i2)] + 2 * i3)] +
				    0.5 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1 + 1)] + i2)]
				       + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2)] +
					     i3)]);
			}
			for (i1 = 0; i1 < mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1)] + 2 * i2 +
					  1)] + 2 * i3)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1)] +
					      2 * i2 + 1)] + 2 * i3)] +
				    0.5 * z1[(i1) * __ocl_mult_factor +
					     __ocl_add_offset];
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 + 1)] +
					  2 * i2 + 1)] + 2 * i3)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1 + 1)] +
					      2 * i2 + 1)] + 2 * i3)] +
				    0.25 *
				    (z1
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + z1[(i1 +
							      1) *
							     __ocl_mult_factor +
							     __ocl_add_offset]);
			}
			for (i1 = 0; i1 < mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1)] +
					  2 * i2)] + 2 * i3 + 1)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1)] +
					      2 * i2)] + 2 * i3 + 1)] +
				    0.5 * z2[(i1) * __ocl_mult_factor +
					     __ocl_add_offset];
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 + 1)] +
					  2 * i2)] + 2 * i3 + 1)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1 + 1)] +
					      2 * i2)] + 2 * i3 + 1)] +
				    0.25 *
				    (z2
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + z2[(i1 +
							      1) *
							     __ocl_mult_factor +
							     __ocl_add_offset]);
			}
			for (i1 = 0; i1 < mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1)] + 2 * i2 +
					  1)] + 2 * i3 + 1)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1)] +
					      2 * i2 + 1)] + 2 * i3 + 1)] +
				    0.25 * z3[(i1) * __ocl_mult_factor +
					      __ocl_add_offset];
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 + 1)] +
					  2 * i2 + 1)] + 2 * i3 + 1)] =
				    u_data[(u_idx
					    [(u_idx[(u_offset + 2 * i1 + 1)] +
					      2 * i2 + 1)] + 2 * i3 + 1)] +
				    0.125 *
				    (z3
				     [(i1) * __ocl_mult_factor +
				      __ocl_add_offset] + z3[(i1 +
							      1) *
							     __ocl_mult_factor +
							     __ocl_add_offset]);
			}
		}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 971 of mg.c
//-------------------------------------------------------------------------------
__kernel void interp_1(int i2, int d2, int mm2, int i1, int d1, int mm1,
		       __global double *u_data, __global unsigned *u_idx,
		       unsigned int u_offset, int d3, __global double *z_data,
		       __global unsigned *z_idx, unsigned int z_offset, int t1,
		       int t2, int __ocl_i3_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + d3;
	if (!(i3 <= __ocl_i3_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i2 = d2; i2 <= mm2 - 1; i2++) {
			for (i1 = d1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - d1 - 1)] +
					  2 * i2 - d2 - 1)] + 2 * i3 - d3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - d1 - 1)] +
					      2 * i2 - d2 - 1)] + 2 * i3 - d3 -
					    1)] +
				    z_data[(z_idx
					    [(z_idx[(z_offset + i1 - 1)] + i2 -
					      1)] + i3 - 1)];
			}
			for (i1 = 1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - t1 - 1)] +
					  2 * i2 - d2 - 1)] + 2 * i3 - d3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - t1 - 1)] +
					      2 * i2 - d2 - 1)] + 2 * i3 - d3 -
					    1)] +
				    0.5 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1)] + i2 - 1)]
				       + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
		}
		for (i2 = 1; i2 <= mm2 - 1; i2++) {
			for (i1 = d1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - d1 - 1)] +
					  2 * i2 - t2 - 1)] + 2 * i3 - d3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - d1 - 1)] +
					      2 * i2 - t2 - 1)] + 2 * i3 - d3 -
					    1)] +
				    0.5 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1 - 1)] + i2)]
				       + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
			for (i1 = 1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - t1 - 1)] +
					  2 * i2 - t2 - 1)] + 2 * i3 - d3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - t1 - 1)] +
					      2 * i2 - t2 - 1)] + 2 * i3 - d3 -
					    1)] +
				    0.25 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1)] + i2)] +
				       i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2 -
					       1)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] +
					       i2)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1012 of mg.c
//-------------------------------------------------------------------------------
__kernel void interp_2(int i2, int d2, int mm2, int i1, int d1, int mm1,
		       __global double *u_data, __global unsigned *u_idx,
		       unsigned int u_offset, int t3, __global double *z_data,
		       __global unsigned *z_idx, unsigned int z_offset, int t1,
		       int t2, int __ocl_i3_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + 1;
	if (!(i3 <= __ocl_i3_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i2 = d2; i2 <= mm2 - 1; i2++) {
			for (i1 = d1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - d1 - 1)] +
					  2 * i2 - d2 - 1)] + 2 * i3 - t3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - d1 - 1)] +
					      2 * i2 - d2 - 1)] + 2 * i3 - t3 -
					    1)] +
				    0.5 *
				    (z_data
				     [(z_idx
				       [(z_idx[(z_offset + i1 - 1)] + i2 - 1)] +
				       i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
			for (i1 = 1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - t1 - 1)] +
					  2 * i2 - d2 - 1)] + 2 * i3 - t3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - t1 - 1)] +
					      2 * i2 - d2 - 1)] + 2 * i3 - t3 -
					    1)] +
				    0.25 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1)] + i2 - 1)]
				       + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2 -
					       1)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
		}
		for (i2 = 1; i2 <= mm2 - 1; i2++) {
			for (i1 = d1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - d1 - 1)] +
					  2 * i2 - t2 - 1)] + 2 * i3 - t3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - d1 - 1)] +
					      2 * i2 - t2 - 1)] + 2 * i3 - t3 -
					    1)] +
				    0.25 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1 - 1)] + i2)]
				       + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] +
					       i2)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
			for (i1 = 1; i1 <= mm1 - 1; i1++) {
				u_data[(u_idx
					[(u_idx[(u_offset + 2 * i1 - t1 - 1)] +
					  2 * i2 - t2 - 1)] + 2 * i3 - t3 -
					1)] =
				    u_data[(u_idx
					    [(u_idx
					      [(u_offset + 2 * i1 - t1 - 1)] +
					      2 * i2 - t2 - 1)] + 2 * i3 - t3 -
					    1)] +
				    0.125 *
				    (z_data
				     [(z_idx[(z_idx[(z_offset + i1)] + i2)] +
				       i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2 -
					       1)] + i2)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] +
					       i2)] + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2)] +
					     i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1)] + i2 -
					       1)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] +
					       i2)] + i3 - 1)] +
				     z_data[(z_idx
					     [(z_idx[(z_offset + i1 - 1)] + i2 -
					       1)] + i3 - 1)]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1168 of mg.c
//-------------------------------------------------------------------------------
__kernel void comm3_0(__global double *u_data, __global unsigned *u_idx,
		      unsigned int u_offset, int n1, int __ocl_i3_bound,
		      int __ocl_i2_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + 1;
	int i2 = get_global_id(1) + 1;
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		u_data[(u_idx[(u_idx[(u_offset + n1 - 1)] + i2)] + i3)] =
		    u_data[(u_idx[(u_idx[(u_offset + 1)] + i2)] + i3)];
		u_data[(u_idx[(u_idx[(u_offset + 0)] + i2)] + i3)] =
		    u_data[(u_idx[(u_idx[(u_offset + n1 - 2)] + i2)] + i3)];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1181 of mg.c
//-------------------------------------------------------------------------------
__kernel void comm3_1(__global double *u_data, __global unsigned *u_idx,
		      unsigned int u_offset, int n2, int __ocl_i3_bound,
		      int __ocl_i1_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0) + 1;
	int i1 = get_global_id(1);
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i1 < __ocl_i1_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		u_data[(u_idx[(u_idx[(u_offset + i1)] + n2 - 1)] + i3)] =
		    u_data[(u_idx[(u_idx[(u_offset + i1)] + 1)] + i3)];
		u_data[(u_idx[(u_idx[(u_offset + i1)] + 0)] + i3)] =
		    u_data[(u_idx[(u_idx[(u_offset + i1)] + n2 - 2)] + i3)];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1194 of mg.c
//-------------------------------------------------------------------------------
__kernel void comm3_2(__global double *u_data, __global unsigned *u_idx,
		      unsigned int u_offset, int n3, int __ocl_i2_bound,
		      int __ocl_i1_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i2 = get_global_id(0);
	int i1 = get_global_id(1);
	if (!(i2 < __ocl_i2_bound)) {
		return;
	}
	if (!(i1 < __ocl_i1_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + n3 - 1)] =
		    u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + 1)];
		u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + 0)] =
		    u_data[(u_idx[(u_idx[(u_offset + i1)] + i2)] + n3 - 2)];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1387 of mg.c
//-------------------------------------------------------------------------------
__kernel void zran3_0(__global double *z_data, __global unsigned *z_idx,
		      int __ocl_i3_bound, int __ocl_i2_bound,
		      int __ocl_i1_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0);
	int i2 = get_global_id(1);
	int i1 = get_global_id(2);
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
		return;
	}
	if (!(i1 < __ocl_i1_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		z_data[(z_idx[(z_idx[(i1)] + i2)] + i3)] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1397 of mg.c
//-------------------------------------------------------------------------------
__kernel void zran3_1(__global double *z_data, __global unsigned *z_idx,
		      __global int *zran3_j1, __global int *zran3_j2,
		      __global int *zran3_j3, int m0, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + m0;
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		z_data[(z_idx
			[(z_idx[(zran3_j1[CALC_2D_IDX(10, 2, (i), (0))])] +
			  zran3_j2[CALC_2D_IDX(10, 2, (i), (0))])] +
			zran3_j3[CALC_2D_IDX(10, 2, (i), (0))])] = -1.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1404 of mg.c
//-------------------------------------------------------------------------------
__kernel void zran3_2(__global double *z_data, __global unsigned *z_idx,
		      __global int *zran3_j1, __global int *zran3_j2,
		      __global int *zran3_j3, int m1, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + m1;
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		z_data[(z_idx
			[(z_idx[(zran3_j1[CALC_2D_IDX(10, 2, (i), (1))])] +
			  zran3_j2[CALC_2D_IDX(10, 2, (i), (1))])] +
			zran3_j3[CALC_2D_IDX(10, 2, (i), (1))])] = 1.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1552 of mg.c
//-------------------------------------------------------------------------------
__kernel void zero3_0(__global double *z_data, __global unsigned *z_idx,
		      unsigned int z_offset, int __ocl_i3_bound,
		      int __ocl_i2_bound, int __ocl_i1_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i3 = get_global_id(0);
	int i2 = get_global_id(1);
	int i1 = get_global_id(2);
	if (!(i3 < __ocl_i3_bound)) {
		return;
	}
	if (!(i2 < __ocl_i2_bound)) {
		return;
	}
	if (!(i1 < __ocl_i1_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		z_data[(z_idx[(z_idx[(z_offset + i1)] + i2)] + i3)] = 0.0;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
