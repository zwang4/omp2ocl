//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Wed Aug  8 16:02:10 2012
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
void cfftz_g3_g4_g5_g6_g7_g10_e8_e9_gtp(int is, int m, int n,
					__global double *x_real,
					__global double *x_imag,
					__global double *y_real,
					__global double *y_imag,
					__global double *u_real, int fftblock,
					int fftblockpad,
					__global double *u_imag,
					unsigned int __ocl_mult_factor,
					unsigned int __ocl_add_offset);
void fftz2_g6_g7_g8_g9_g10_g11_gtp(int is, int l, int m, int n, int ny, int ny1,
				   __global double *u_real,
				   __global double *u_imag,
				   __global double *x_real,
				   __global double *x_imag,
				   __global double *y_real,
				   __global double *y_imag,
				   unsigned int __ocl_mult_factor,
				   unsigned int __ocl_add_offset);
void fftz2(int is, int l, int m, int n, int ny, int ny1, double u_real[64],
	   double u_imag[64], double x_real[64][18], double x_imag[64][18],
	   double y_real[64][18], double y_imag[64][18]);

void fftz2(int is, int l, int m, int n, int ny, int ny1, double u_real[64],
	   double u_imag[64], double x_real[64][18], double x_imag[64][18],
	   double y_real[64][18], double y_imag[64][18])
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

//-------------------------------------------------------------------------------
//This is an alias of function: cfftz
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: x_real
//      1: x_imag
//      2: y_real
//      3: y_imag
//      4: u_real
//      5: u_imag
//-------------------------------------------------------------------------------
void cfftz_g3_g4_g5_g6_g7_g10_e8_e9_gtp(int is, int m, int n,
					__global double *x_real,
					__global double *x_imag,
					__global double *y_real,
					__global double *y_imag,
					__global double *u_real, int fftblock,
					int fftblockpad,
					__global double *u_imag,
					unsigned int __ocl_mult_factor,
					unsigned int __ocl_add_offset)
{

	int i, j, l, mx;
	mx = (int)(u_real[(0)]);
	for (l = 1; l <= m; l += 2) {
		fftz2_g6_g7_g8_g9_g10_g11_gtp(is, l, m, n, fftblock,
					      fftblockpad, u_real, u_imag,
					      x_real, x_imag, y_real, y_imag,
					      __ocl_mult_factor,
					      __ocl_add_offset);
		if (l == m)
			break;
		fftz2_g6_g7_g8_g9_g10_g11_gtp(is, l + 1, m, n, fftblock,
					      fftblockpad, u_real, u_imag,
					      y_real, y_imag, x_real, x_imag,
					      __ocl_mult_factor,
					      __ocl_add_offset);
	}
	if (m % 2 == 1) {
		for (j = 0; j < n; j++) {
			for (i = 0; i < fftblock; i++) {
				x_real[CALC_2D_IDX(64, 18, (j), (i)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    y_real[CALC_2D_IDX(64, 18, (j), (i)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
				x_imag[CALC_2D_IDX(64, 18, (j), (i)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    y_imag[CALC_2D_IDX(64, 18, (j), (i)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
			}
		}
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: fftz2
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: u_real
//      1: u_imag
//      2: x_real
//      3: x_imag
//      4: y_real
//      5: y_imag
//-------------------------------------------------------------------------------
void fftz2_g6_g7_g8_g9_g10_g11_gtp(int is, int l, int m, int n, int ny, int ny1,
				   __global double *u_real,
				   __global double *u_imag,
				   __global double *x_real,
				   __global double *x_imag,
				   __global double *y_real,
				   __global double *y_imag,
				   unsigned int __ocl_mult_factor,
				   unsigned int __ocl_add_offset)
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
			u1_real = u_real[(ku + i)];
			u1_imag = u_imag[(ku + i)];
		} else {
			u1_real = u_real[(ku + i)];
			u1_imag = -u_imag[(ku + i)];
		}
		for (k = 0; k < lk; k++) {
			for (j = 0; j < ny; j++) {
				double x11real, x11imag;
				double x21real, x21imag;
				x11real =
				    x_real[CALC_2D_IDX(64, 18, (i11 + k), (j)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
				x11imag =
				    x_imag[CALC_2D_IDX(64, 18, (i11 + k), (j)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
				x21real =
				    x_real[CALC_2D_IDX(64, 18, (i12 + k), (j)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
				x21imag =
				    x_imag[CALC_2D_IDX(64, 18, (i12 + k), (j)) *
					   __ocl_mult_factor +
					   __ocl_add_offset];
				y_real[CALC_2D_IDX(64, 18, (i21 + k), (j)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    x11real + x21real;
				y_imag[CALC_2D_IDX(64, 18, (i21 + k), (j)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    x11imag + x21imag;
				y_real[CALC_2D_IDX(64, 18, (i22 + k), (j)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    u1_real * (x11real - x21real) -
				    u1_imag * (x11imag - x21imag);
				y_imag[CALC_2D_IDX(64, 18, (i22 + k), (j)) *
				       __ocl_mult_factor + __ocl_add_offset] =
				    u1_real * (x11imag - x21imag) +
				    u1_imag * (x11real - x21real);
			}
		}
	}

}

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 395 of ft.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void evolve_0(__global double *u1_real, __global double *u0_real,
		       __global double *ex, int t, __global int *indexmap,
		       __global double *u1_imag, __global double *u0_imag,
		       int __ocl_i_bound, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		u1_real[CALC_3D_IDX(64, 64, 64, (k), (j), (i))] =
		    u0_real[CALC_3D_IDX(64, 64, 64, (k), (j), (i))] *
		    ex[(t * indexmap[CALC_3D_IDX(64, 64, 64, (k), (j), (i))])];
		u1_imag[CALC_3D_IDX(64, 64, 64, (k), (j), (i))] =
		    u0_imag[CALC_3D_IDX(64, 64, 64, (k), (j), (i))] *
		    ex[(t * indexmap[CALC_3D_IDX(64, 64, 64, (k), (j), (i))])];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 579 of ft.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void compute_indexmap_0(int xstart_i, int ystart_i, int zstart_i,
				 __global int *indexmap, int __ocl_i_bound,
				 int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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
	int ii;			/* Defined at ft.c : 562 */
	int ii2;		/* Defined at ft.c : 562 */
	int jj;			/* Defined at ft.c : 562 */
	int ij2;		/* Defined at ft.c : 562 */
	int kk;			/* Defined at ft.c : 562 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		ii = (i + 1 + xstart_i - 2 + 64 / 2) % 64 - 64 / 2;
		ii2 = ii * ii;
		jj = (j + 1 + ystart_i - 2 + 64 / 2) % 64 - 64 / 2;
		ij2 = jj * jj + ii2;
		kk = (k + 1 + zstart_i - 2 + 64 / 2) % 64 - 64 / 2;
		indexmap[CALC_3D_IDX(64, 64, 64, (k), (j), (i))] =
		    kk * kk + ij2;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 690 of ft.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void cffts1_0(int fftblock, __global int *d, __global double *yy0_real,
		       __global double *x_real, __global double *yy0_imag,
		       __global double *x_imag, int is, int logd_0,
		       __global double *yy1_real, __global double *yy1_imag,
		       __global double *xout_real, __global double *xout_imag,
		       __global double *u_real, __global double *u_imag,
		       int fftblockpad, int __ocl_jj_inc_fftblock,
		       int __ocl_jj_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int jj = get_global_id(0);
	jj = jj * __ocl_jj_inc_fftblock;
	int k = get_global_id(1);
	if (!(jj <= __ocl_jj_bound)) {
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
	int j;			/* Defined at ft.c : 681 */
	int i;			/* Defined at ft.c : 681 */
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
	{
		for (j = 0; j < fftblock; j++) {
			for (i = 0; i < d[(0)]; i++) {
				yy0_real[CALC_2D_IDX(64, 18, (i), (j)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_real[CALC_3D_IDX
					   (64, 64, 64, (k), (j + jj), (i))];
				yy0_imag[CALC_2D_IDX(64, 18, (i), (j)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_imag[CALC_3D_IDX
					   (64, 64, 64, (k), (j + jj), (i))];
			}
		}
		cfftz_g3_g4_g5_g6_g7_g10_e8_e9_gtp(is, logd_0, d[(0)], yy0_real,
						   yy0_imag, yy1_real, yy1_imag,
						   u_real, fftblock,
						   fftblockpad, u_imag,
						   __ocl_mult_factor,
						   __ocl_add_offset)
		    /*ARGEXP: u_real,fftblock,fftblockpad,u_imag */ ;
		for (j = 0; j < fftblock; j++) {
			for (i = 0; i < d[(0)]; i++) {
				xout_real[CALC_3D_IDX
					  (64, 64, 64, (k), (j + jj), (i))] =
				    yy0_real[CALC_2D_IDX(64, 18, (i), (j)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
				xout_imag[CALC_3D_IDX
					  (64, 64, 64, (k), (j + jj), (i))] =
				    yy0_imag[CALC_2D_IDX(64, 18, (i), (j)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 738 of ft.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void cffts2_0(__global int *d, int fftblock, __global double *yy0_real,
		       __global double *x_real, __global double *yy0_imag,
		       __global double *x_imag, int is, int logd_1,
		       __global double *yy1_real, __global double *yy1_imag,
		       __global double *xout_real, __global double *xout_imag,
		       __global double *u_real, __global double *u_imag,
		       int fftblockpad, int __ocl_ii_inc_fftblock,
		       int __ocl_ii_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int ii = get_global_id(0);
	ii = ii * __ocl_ii_inc_fftblock;
	int k = get_global_id(1);
	if (!(ii <= __ocl_ii_bound)) {
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
	int j;			/* Defined at ft.c : 729 */
	int i;			/* Defined at ft.c : 729 */
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
	{
		for (j = 0; j < d[(1)]; j++) {
			for (i = 0; i < fftblock; i++) {
				yy0_real[CALC_2D_IDX(64, 18, (j), (i)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_real[CALC_3D_IDX
					   (64, 64, 64, (k), (j), (i + ii))];
				yy0_imag[CALC_2D_IDX(64, 18, (j), (i)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_imag[CALC_3D_IDX
					   (64, 64, 64, (k), (j), (i + ii))];
			}
		}
		cfftz_g3_g4_g5_g6_g7_g10_e8_e9_gtp(is, logd_1, d[(1)], yy0_real,
						   yy0_imag, yy1_real, yy1_imag,
						   u_real, fftblock,
						   fftblockpad, u_imag,
						   __ocl_mult_factor,
						   __ocl_add_offset)
		    /*ARGEXP: u_real,fftblock,fftblockpad,u_imag */ ;
		for (j = 0; j < d[(1)]; j++) {
			for (i = 0; i < fftblock; i++) {
				xout_real[CALC_3D_IDX
					  (64, 64, 64, (k), (j), (i + ii))] =
				    yy0_real[CALC_2D_IDX(64, 18, (j), (i)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
				xout_imag[CALC_3D_IDX
					  (64, 64, 64, (k), (j), (i + ii))] =
				    yy0_imag[CALC_2D_IDX(64, 18, (j), (i)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 785 of ft.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void cffts3_0(__global int *d, int fftblock, __global double *yy0_real,
		       __global double *x_real, __global double *yy0_imag,
		       __global double *x_imag, int is, int logd_2,
		       __global double *yy1_real, __global double *yy1_imag,
		       __global double *xout_real, __global double *xout_imag,
		       __global double *u_real, __global double *u_imag,
		       int fftblockpad, int __ocl_ii_inc_fftblock,
		       int __ocl_ii_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int ii = get_global_id(0);
	ii = ii * __ocl_ii_inc_fftblock;
	int j = get_global_id(1);
	if (!(ii <= __ocl_ii_bound)) {
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
	int k;			/* Defined at ft.c : 776 */
	int i;			/* Defined at ft.c : 776 */
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
	{
		for (k = 0; k < d[(2)]; k++) {
			for (i = 0; i < fftblock; i++) {
				yy0_real[CALC_2D_IDX(64, 18, (k), (i)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_real[CALC_3D_IDX
					   (64, 64, 64, (k), (j), (i + ii))];
				yy0_imag[CALC_2D_IDX(64, 18, (k), (i)) *
					 __ocl_mult_factor + __ocl_add_offset] =
				    x_imag[CALC_3D_IDX
					   (64, 64, 64, (k), (j), (i + ii))];
			}
		}
		cfftz_g3_g4_g5_g6_g7_g10_e8_e9_gtp(is, logd_2, d[(2)], yy0_real,
						   yy0_imag, yy1_real, yy1_imag,
						   u_real, fftblock,
						   fftblockpad, u_imag,
						   __ocl_mult_factor,
						   __ocl_add_offset)
		    /*ARGEXP: u_real,fftblock,fftblockpad,u_imag */ ;
		for (k = 0; k < d[(2)]; k++) {
			for (i = 0; i < fftblock; i++) {
				xout_real[CALC_3D_IDX
					  (64, 64, 64, (k), (j), (i + ii))] =
				    yy0_real[CALC_2D_IDX(64, 18, (k), (i)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
				xout_imag[CALC_3D_IDX
					  (64, 64, 64, (k), (j), (i + ii))] =
				    yy0_imag[CALC_2D_IDX(64, 18, (k), (i)) *
					     __ocl_mult_factor +
					     __ocl_add_offset];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

__kernel void checksum_0_reduction_step0(__global double *__ocl_part_chk_real,
					 __global double *__ocl_part_chk_imag,
					 unsigned int offset,
					 unsigned int bound)
{
	unsigned int i = get_global_id(0);
	if (i >= bound)
		return;
	i = i + offset;
	__ocl_part_chk_real[i] = 0.0;
	__ocl_part_chk_imag[i] = 0.0;
}

//-------------------------------------------------------------------------------
//Loop defined at line 1019 of ft.c
//-------------------------------------------------------------------------------
__kernel void checksum_0_reduction_step1(__global int *xstart,
					 __global int *xend,
					 __global int *ystart,
					 __global int *yend,
					 __global int *zstart,
					 __global int *zend,
					 __global double *u1_real,
					 __global double *u1_imag,
					 __global double *__ocl_part_chk_real,
					 __global double *__ocl_part_chk_imag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	if (!(j <= 1024)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int q;			/* Defined at ft.c : 1007 */
	int r;			/* Defined at ft.c : 1007 */
	int s;			/* Defined at ft.c : 1007 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	//__global double (*u1_real)[64][64] =
	//    (__global double (*)[64][64])g_u1_real;
	//__global double (*u1_imag)[64][64] =
	//    (__global double (*)[64][64])g_u1_imag;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//Declare reduction variables (BEGIN)
	//-------------------------------------------
	double chk_real = 0.0;	/* reduction variable, defined at: ft.c : 1015 */
	double chk_imag = 0.0;	/* reduction variable, defined at: ft.c : 1016 */
	//-------------------------------------------
	//Declare reduction variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		q = j % 64 + 1;
		if (q >= xstart[(0)] && q <= xend[(0)]) {
			r = (3 * j) % 64 + 1;
			if (r >= ystart[(0)] && r <= yend[(0)]) {
				s = (5 * j) % 64 + 1;
				if (s >= zstart[(0)] && s <= zend[(0)]) {
					chk_real =
					    chk_real +
					    u1_real[CALC_3D_IDX
						    (64, 64, 64,
						     (s - zstart[(0)]),
						     (r - ystart[(0)]),
						     (q - xstart[(0)]))];
					chk_imag =
					    chk_imag +
					    u1_imag[CALC_3D_IDX
						    (64, 64, 64,
						     (s - zstart[(0)]),
						     (r - ystart[(0)]),
						     (q - xstart[(0)]))];
				}
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

	//-------------------------------------------
	//Write back to the global buffer (BEGIN)
	//-------------------------------------------
	{
		unsigned int __ocl_wb_idx = get_global_id(0);
		__ocl_part_chk_real[__ocl_wb_idx] = chk_real;
		__ocl_part_chk_imag[__ocl_wb_idx] = chk_imag;
	}
	//-------------------------------------------
	//Write back to the global buffer (END)
	//-------------------------------------------
}

__kernel void checksum_0_reduction_step2(__global double4 * input_chk_real,
					 __global double *output_chk_real,
					 __global double4 * input_chk_imag,
					 __global double *output_chk_imag)
{
	unsigned int tid = get_local_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int gid = get_global_id(0);
	unsigned int local_size = get_local_size(0);

	__local double4 sdata_chk_real[GROUP_SIZE];
	__local double4 sdata_chk_imag[GROUP_SIZE];
	sdata_chk_real[tid] = input_chk_real[gid];
	sdata_chk_imag[tid] = input_chk_imag[gid];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int s = local_size / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata_chk_real[tid] += sdata_chk_real[tid + s];
			sdata_chk_imag[tid] += sdata_chk_imag[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tid == 0) {
		output_chk_real[bid] =
		    (sdata_chk_real[0].x + sdata_chk_real[0].y +
		     sdata_chk_real[0].z + sdata_chk_real[0].w);
		output_chk_imag[bid] =
		    (sdata_chk_imag[0].x + sdata_chk_imag[0].y +
		     sdata_chk_imag[0].z + sdata_chk_imag[0].w);
	}
}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
