//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Fri Sep 20 18:27:28 2013
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
void exact_solution_g4(double xi, double eta, double zeta, double dtemp[5],
		       __global double (*ce)[13], __global int *tls_validflag,
		       int tls_thread_id);
void binvcrhs_g0_g5_g10(__global double (*lhs)[5][5][65][65][65], int lhs_0,
			int lhs_1, int lhs_2, int lhs_3,
			__global double (*c)[5][5][65][65][65], int c_0,
			int c_1, int c_2, int c_3,
			__global double (*r)[65][65][65], int r_0, int r_1,
			int r_2, __global int (*rd_log_lhs)[5][5][65][65][65],
			__global int (*wr_log_lhs)[5][5][65][65][65],
			__global int (*rd_log_c)[5][5][65][65][65],
			__global int (*wr_log_c)[5][5][65][65][65],
			__global int (*rd_log_r)[65][65][65],
			__global int (*wr_log_r)[65][65][65],
			__global int *tls_validflag, int tls_thread_id);
void matvec_sub_g0_g5_g9(__global double (*ablock)[5][5][65][65][65],
			 int ablock_0, int ablock_1, int ablock_2, int ablock_3,
			 __global double (*avec)[65][65][65], int avec_0,
			 int avec_1, int avec_2,
			 __global double (*bvec)[65][65][65], int bvec_0,
			 int bvec_1, int bvec_2,
			 __global int (*rd_log_bvec)[65][65][65],
			 __global int (*wr_log_bvec)[65][65][65],
			 __global int *tls_validflag, int tls_thread_id);
void matmul_sub_g0_g5_g10(__global double (*ablock)[5][5][65][65][65],
			  int ablock_0, int ablock_1, int ablock_2,
			  int ablock_3,
			  __global double (*bblock)[5][5][65][65][65],
			  int bblock_0, int bblock_1, int bblock_2,
			  int bblock_3,
			  __global double (*cblock)[5][5][65][65][65],
			  int cblock_0, int cblock_1, int cblock_2,
			  int cblock_3,
			  __global int (*rd_log_cblock)[5][5][65][65][65],
			  __global int (*wr_log_cblock)[5][5][65][65][65],
			  __global int *tls_validflag, int tls_thread_id);
void binvrhs_g0_g5(__global double (*lhs)[5][5][65][65][65], int lhs_0,
		   int lhs_1, int lhs_2, int lhs_3,
		   __global double (*r)[65][65][65], int r_0, int r_1, int r_2,
		   __global int (*rd_log_lhs)[5][5][65][65][65],
		   __global int (*wr_log_lhs)[5][5][65][65][65],
		   __global int (*rd_log_r)[65][65][65],
		   __global int (*wr_log_r)[65][65][65],
		   __global int *tls_validflag, int tls_thread_id);
void exact_solution_g4_no_spec(double xi, double eta, double zeta,
			       double dtemp[5], __global double (*ce)[13],
			       __global int *tls_validflag, int tls_thread_id);
void binvcrhs_g0_g5_g10_no_spec(__global double (*lhs)[5][5][65][65][65],
				int lhs_0, int lhs_1, int lhs_2, int lhs_3,
				__global double (*c)[5][5][65][65][65], int c_0,
				int c_1, int c_2, int c_3,
				__global double (*r)[65][65][65], int r_0,
				int r_1, int r_2, __global int *tls_validflag,
				int tls_thread_id);
void matvec_sub_g0_g5_g9_no_spec(__global double (*ablock)[5][5][65][65][65],
				 int ablock_0, int ablock_1, int ablock_2,
				 int ablock_3,
				 __global double (*avec)[65][65][65],
				 int avec_0, int avec_1, int avec_2,
				 __global double (*bvec)[65][65][65],
				 int bvec_0, int bvec_1, int bvec_2,
				 __global int *tls_validflag,
				 int tls_thread_id);
void matmul_sub_g0_g5_g10_no_spec(__global double (*ablock)[5][5][65][65][65],
				  int ablock_0, int ablock_1, int ablock_2,
				  int ablock_3,
				  __global double (*bblock)[5][5][65][65][65],
				  int bblock_0, int bblock_1, int bblock_2,
				  int bblock_3,
				  __global double (*cblock)[5][5][65][65][65],
				  int cblock_0, int cblock_1, int cblock_2,
				  int cblock_3, __global int *tls_validflag,
				  int tls_thread_id);
void binvrhs_g0_g5_no_spec(__global double (*lhs)[5][5][65][65][65], int lhs_0,
			   int lhs_1, int lhs_2, int lhs_3,
			   __global double (*r)[65][65][65], int r_0, int r_1,
			   int r_2, __global int *tls_validflag,
			   int tls_thread_id);
void binvcrhs(double lhs[3][5][5][65][65][65], int lhs_0, int lhs_1, int lhs_2,
	      int lhs_3, double c[3][5][5][65][65][65], int c_0, int c_1,
	      int c_2, int c_3, double r[5][65][65][65], int r_0, int r_1,
	      int r_2);
void matvec_sub(double ablock[3][5][5][65][65][65], int ablock_0, int ablock_1,
		int ablock_2, int ablock_3, double avec[5][65][65][65],
		int avec_0, int avec_1, int avec_2, double bvec[5][65][65][65],
		int bvec_0, int bvec_1, int bvec_2);
void matmul_sub(double ablock[3][5][5][65][65][65], int ablock_0, int ablock_1,
		int ablock_2, int ablock_3, double bblock[3][5][5][65][65][65],
		int bblock_0, int bblock_1, int bblock_2, int bblock_3,
		double cblock[3][5][5][65][65][65], int cblock_0, int cblock_1,
		int cblock_2, int cblock_3);
void binvrhs(double lhs[3][5][5][65][65][65], int lhs_0, int lhs_1, int lhs_2,
	     int lhs_3, double r[5][65][65][65], int r_0, int r_1, int r_2);

void binvcrhs(double lhs[3][5][5][65][65][65], int lhs_0, int lhs_1, int lhs_2,
	      int lhs_3, double c[3][5][5][65][65][65], int c_0, int c_1,
	      int c_2, int c_3, double r[5][65][65][65], int r_0, int r_1,
	      int r_2)
{
	double pivot, coeff;
	pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] * pivot;
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] * pivot;
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] * pivot;
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] * pivot;
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] * pivot;
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] * pivot;
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] * pivot;
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] * pivot;
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] * pivot;
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] * pivot;
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] * pivot;
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] * pivot;
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] * pivot;
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] * pivot;
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] * pivot;
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] * pivot;
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] * pivot;
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] * pivot;
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] * pivot;
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] * pivot;
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] * pivot;
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] * pivot;
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] * pivot;
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] * pivot;
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] * pivot;
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
}

void matvec_sub(double ablock[3][5][5][65][65][65], int ablock_0, int ablock_1,
		int ablock_2, int ablock_3, double avec[5][65][65][65],
		int avec_0, int avec_1, int avec_2, double bvec[5][65][65][65],
		int bvec_0, int bvec_1, int bvec_2)
{
	int i;
	for (i = 0; i < 5; i++) {
		bvec[i][bvec_0][bvec_1][bvec_2] =
		    bvec[i][bvec_0][bvec_1][bvec_2] -
		    ablock[ablock_3][i][0][ablock_0][ablock_1][ablock_2] *
		    avec[0][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][1][ablock_0][ablock_1][ablock_2] *
		    avec[1][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][2][ablock_0][ablock_1][ablock_2] *
		    avec[2][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][3][ablock_0][ablock_1][ablock_2] *
		    avec[3][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][4][ablock_0][ablock_1][ablock_2] *
		    avec[4][avec_0][avec_1][avec_2];
	}
}

void matmul_sub(double ablock[3][5][5][65][65][65], int ablock_0, int ablock_1,
		int ablock_2, int ablock_3, double bblock[3][5][5][65][65][65],
		int bblock_0, int bblock_1, int bblock_2, int bblock_3,
		double cblock[3][5][5][65][65][65], int cblock_0, int cblock_1,
		int cblock_2, int cblock_3)
{
	int j;
	for (j = 0; j < 5; j++) {
		cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][0][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][1][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][2][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][3][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][4][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
	}
}

void binvrhs(double lhs[3][5][5][65][65][65], int lhs_0, int lhs_1, int lhs_2,
	     int lhs_3, double r[5][65][65][65], int r_0, int r_1, int r_2)
{
	double pivot, coeff;
	pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
}

//-------------------------------------------------------------------------------
//This is an alias of function: exact_solution
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_solution_g4(double xi, double eta, double zeta, double dtemp[5],
		       __global double (*ce)[13], __global int *tls_validflag,
		       int tls_thread_id)
{
	int m;
	for (m = 0; m < 5; m++) {
		dtemp[m] =
		    ce[m][0] + xi * (ce[m][1] +
				     xi * (ce[m][4] +
					   xi * (ce[m][7] + xi * ce[m][10]))) +
		    eta * (ce[m][2] +
			   eta * (ce[m][5] +
				  eta * (ce[m][8] + eta * ce[m][11]))) +
		    zeta * (ce[m][3] +
			    zeta * (ce[m][6] +
				    zeta * (ce[m][9] + zeta * ce[m][12])));
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: binvcrhs
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: lhs
//      1: c
//      2: r
//-------------------------------------------------------------------------------
void binvcrhs_g0_g5_g10(__global double (*lhs)[5][5][65][65][65], int lhs_0,
			int lhs_1, int lhs_2, int lhs_3,
			__global double (*c)[5][5][65][65][65], int c_0,
			int c_1, int c_2, int c_3,
			__global double (*r)[65][65][65], int r_0, int r_1,
			int r_2, __global int (*rd_log_lhs)[5][5][65][65][65],
			__global int (*wr_log_lhs)[5][5][65][65][65],
			__global int (*rd_log_c)[5][5][65][65][65],
			__global int (*wr_log_c)[5][5][65][65][65],
			__global int (*rd_log_r)[65][65][65],
			__global int (*wr_log_r)[65][65][65],
			__global int *tls_validflag, int tls_thread_id)
{
	double pivot, coeff;
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][0][0][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][0][0][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&c[c_3][0][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][0][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][0][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][0][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][0][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][1][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][2][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][3][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][4][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][0][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][0][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&c[c_3][1][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][1][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][1][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][1][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][1][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][0][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][2][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][3][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][4][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][1][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][1][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&c[c_3][2][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][2][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][2][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][2][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][2][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][0][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][1][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][3][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][4][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][2][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][2][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&c[c_3][3][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][3][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][3][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][3][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][3][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][0][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][1][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][2][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&c[c_3][4][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][4][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][3][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][3][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&c[c_3][4][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][4][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][4][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][4][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&c[c_3][4][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][4][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][4][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) * pivot));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&c[c_3][0][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][0][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][0][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][0][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][0][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&c[c_3][1][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][1][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][1][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][1][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][1][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&c[c_3][2][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][2][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][2][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][2][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][2][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&c[c_3][3][0][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][0][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][0][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][0][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][0][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][0][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][0][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][1][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][1][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][1][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][1][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][1][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][1][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][1][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][2][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][2][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][2][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][2][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][2][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][2][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][2][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][3][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][3][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][3][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][3][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][3][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][3][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][3][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&c[c_3][3][4][c_0][c_1][c_2],
			  &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			  &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&c[c_3][3][4][c_0][c_1][c_2],
			    &wr_log_c[c_3][3][4][c_0][c_1][c_2],
			    &rd_log_c[c_3][3][4][c_0][c_1][c_2], tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&c[c_3][4][4][c_0][c_1][c_2],
					    &wr_log_c[c_3][4][4][c_0][c_1][c_2],
					    &rd_log_c[c_3][4][4][c_0][c_1][c_2],
					    tls_thread_id, tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));

}

//-------------------------------------------------------------------------------
//This is an alias of function: matvec_sub
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ablock
//      1: avec
//      2: bvec
//-------------------------------------------------------------------------------
void matvec_sub_g0_g5_g9(__global double (*ablock)[5][5][65][65][65],
			 int ablock_0, int ablock_1, int ablock_2, int ablock_3,
			 __global double (*avec)[65][65][65], int avec_0,
			 int avec_1, int avec_2,
			 __global double (*bvec)[65][65][65], int bvec_0,
			 int bvec_1, int bvec_2,
			 __global int (*rd_log_bvec)[65][65][65],
			 __global int (*wr_log_bvec)[65][65][65],
			 __global int *tls_validflag, int tls_thread_id)
{
	int i;
	for (i = 0; i < 5; i++) {
		spec_write_double(&bvec[i][bvec_0][bvec_1][bvec_2],
				  &wr_log_bvec[i][bvec_0][bvec_1][bvec_2],
				  &rd_log_bvec[i][bvec_0][bvec_1][bvec_2],
				  tls_thread_id, tls_validflag,
				  (spec_read_double
				   (&bvec[i][bvec_0][bvec_1][bvec_2],
				    &wr_log_bvec[i][bvec_0][bvec_1][bvec_2],
				    &rd_log_bvec[i][bvec_0][bvec_1][bvec_2],
				    tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][i][0][ablock_0][ablock_1]
				   [ablock_2] *
				   avec[0][avec_0][avec_1][avec_2] -
				   ablock[ablock_3][i][1][ablock_0][ablock_1]
				   [ablock_2] *
				   avec[1][avec_0][avec_1][avec_2] -
				   ablock[ablock_3][i][2][ablock_0][ablock_1]
				   [ablock_2] *
				   avec[2][avec_0][avec_1][avec_2] -
				   ablock[ablock_3][i][3][ablock_0][ablock_1]
				   [ablock_2] *
				   avec[3][avec_0][avec_1][avec_2] -
				   ablock[ablock_3][i][4][ablock_0][ablock_1]
				   [ablock_2] *
				   avec[4][avec_0][avec_1][avec_2]));
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: matmul_sub
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ablock
//      1: bblock
//      2: cblock
//-------------------------------------------------------------------------------
void matmul_sub_g0_g5_g10(__global double (*ablock)[5][5][65][65][65],
			  int ablock_0, int ablock_1, int ablock_2,
			  int ablock_3,
			  __global double (*bblock)[5][5][65][65][65],
			  int bblock_0, int bblock_1, int bblock_2,
			  int bblock_3,
			  __global double (*cblock)[5][5][65][65][65],
			  int cblock_0, int cblock_1, int cblock_2,
			  int cblock_3,
			  __global int (*rd_log_cblock)[5][5][65][65][65],
			  __global int (*wr_log_cblock)[5][5][65][65][65],
			  __global int *tls_validflag, int tls_thread_id)
{
	int j;
	for (j = 0; j < 5; j++) {
		spec_write_double(&cblock[cblock_3][0][j][cblock_0][cblock_1]
				  [cblock_2],
				  &wr_log_cblock[cblock_3][0][j][cblock_0]
				  [cblock_1][cblock_2],
				  &rd_log_cblock[cblock_3][0][j][cblock_0]
				  [cblock_1][cblock_2], tls_thread_id,
				  tls_validflag,
				  (spec_read_double
				   (&cblock[cblock_3][0][j][cblock_0][cblock_1]
				    [cblock_2],
				    &wr_log_cblock[cblock_3][0][j][cblock_0]
				    [cblock_1][cblock_2],
				    &rd_log_cblock[cblock_3][0][j][cblock_0]
				    [cblock_1][cblock_2], tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][0][0][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][0][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][0][1][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][1][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][0][2][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][2][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][0][3][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][3][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][0][4][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][4][j][bblock_0][bblock_1]
				   [bblock_2]));
		spec_write_double(&cblock[cblock_3][1][j][cblock_0][cblock_1]
				  [cblock_2],
				  &wr_log_cblock[cblock_3][1][j][cblock_0]
				  [cblock_1][cblock_2],
				  &rd_log_cblock[cblock_3][1][j][cblock_0]
				  [cblock_1][cblock_2], tls_thread_id,
				  tls_validflag,
				  (spec_read_double
				   (&cblock[cblock_3][1][j][cblock_0][cblock_1]
				    [cblock_2],
				    &wr_log_cblock[cblock_3][1][j][cblock_0]
				    [cblock_1][cblock_2],
				    &rd_log_cblock[cblock_3][1][j][cblock_0]
				    [cblock_1][cblock_2], tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][1][0][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][0][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][1][1][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][1][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][1][2][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][2][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][1][3][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][3][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][1][4][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][4][j][bblock_0][bblock_1]
				   [bblock_2]));
		spec_write_double(&cblock[cblock_3][2][j][cblock_0][cblock_1]
				  [cblock_2],
				  &wr_log_cblock[cblock_3][2][j][cblock_0]
				  [cblock_1][cblock_2],
				  &rd_log_cblock[cblock_3][2][j][cblock_0]
				  [cblock_1][cblock_2], tls_thread_id,
				  tls_validflag,
				  (spec_read_double
				   (&cblock[cblock_3][2][j][cblock_0][cblock_1]
				    [cblock_2],
				    &wr_log_cblock[cblock_3][2][j][cblock_0]
				    [cblock_1][cblock_2],
				    &rd_log_cblock[cblock_3][2][j][cblock_0]
				    [cblock_1][cblock_2], tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][2][0][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][0][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][2][1][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][1][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][2][2][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][2][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][2][3][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][3][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][2][4][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][4][j][bblock_0][bblock_1]
				   [bblock_2]));
		spec_write_double(&cblock[cblock_3][3][j][cblock_0][cblock_1]
				  [cblock_2],
				  &wr_log_cblock[cblock_3][3][j][cblock_0]
				  [cblock_1][cblock_2],
				  &rd_log_cblock[cblock_3][3][j][cblock_0]
				  [cblock_1][cblock_2], tls_thread_id,
				  tls_validflag,
				  (spec_read_double
				   (&cblock[cblock_3][3][j][cblock_0][cblock_1]
				    [cblock_2],
				    &wr_log_cblock[cblock_3][3][j][cblock_0]
				    [cblock_1][cblock_2],
				    &rd_log_cblock[cblock_3][3][j][cblock_0]
				    [cblock_1][cblock_2], tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][3][0][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][0][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][3][1][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][1][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][3][2][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][2][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][3][3][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][3][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][3][4][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][4][j][bblock_0][bblock_1]
				   [bblock_2]));
		spec_write_double(&cblock[cblock_3][4][j][cblock_0][cblock_1]
				  [cblock_2],
				  &wr_log_cblock[cblock_3][4][j][cblock_0]
				  [cblock_1][cblock_2],
				  &rd_log_cblock[cblock_3][4][j][cblock_0]
				  [cblock_1][cblock_2], tls_thread_id,
				  tls_validflag,
				  (spec_read_double
				   (&cblock[cblock_3][4][j][cblock_0][cblock_1]
				    [cblock_2],
				    &wr_log_cblock[cblock_3][4][j][cblock_0]
				    [cblock_1][cblock_2],
				    &rd_log_cblock[cblock_3][4][j][cblock_0]
				    [cblock_1][cblock_2], tls_thread_id,
				    tls_validflag) -
				   ablock[ablock_3][4][0][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][0][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][4][1][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][1][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][4][2][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][2][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][4][3][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][3][j][bblock_0][bblock_1]
				   [bblock_2] -
				   ablock[ablock_3][4][4][ablock_0][ablock_1]
				   [ablock_2] *
				   bblock[bblock_3][4][j][bblock_0][bblock_1]
				   [bblock_2]));
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: binvrhs
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: lhs
//      1: r
//-------------------------------------------------------------------------------
void binvrhs_g0_g5(__global double (*lhs)[5][5][65][65][65], int lhs_0,
		   int lhs_1, int lhs_2, int lhs_3,
		   __global double (*r)[65][65][65], int r_0, int r_1, int r_2,
		   __global int (*rd_log_lhs)[5][5][65][65][65],
		   __global int (*wr_log_lhs)[5][5][65][65][65],
		   __global int (*rd_log_r)[65][65][65],
		   __global int (*wr_log_r)[65][65][65],
		   __global int *tls_validflag, int tls_thread_id)
{
	double pivot, coeff;
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][0][0][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][0][0][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][1][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][0][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[0][r_0][r_1][r_2],
						    &wr_log_r[0][r_0][r_1][r_2],
						    &rd_log_r[0][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][1][1][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][1][1][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][2][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][1][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[1][r_0][r_1][r_2],
						    &wr_log_r[1][r_0][r_1][r_2],
						    &rd_log_r[1][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][2][2][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][2][2][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][3][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][2][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[2][r_0][r_1][r_2],
						    &wr_log_r[2][r_0][r_1][r_2],
						    &rd_log_r[2][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][3][3][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][3][3][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id, tls_validflag) * pivot));
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			  tls_thread_id, tls_validflag,
			  (spec_read_double
			   (&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
			    tls_thread_id,
			    tls_validflag) -
			   coeff *
			   spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1]
					    [lhs_2],
					    &wr_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2],
					    &rd_log_lhs[lhs_3][3][4][lhs_0]
					    [lhs_1][lhs_2], tls_thread_id,
					    tls_validflag)));
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[3][r_0][r_1][r_2],
						    &wr_log_r[3][r_0][r_1][r_2],
						    &rd_log_r[3][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	pivot =
	    1.00 / spec_read_double(&lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2],
				    &wr_log_lhs[lhs_3][4][4][lhs_0][lhs_1]
				    [lhs_2],
				    &rd_log_lhs[lhs_3][4][4][lhs_0][lhs_1]
				    [lhs_2], tls_thread_id, tls_validflag);
	spec_write_double(&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			  &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[4][r_0][r_1][r_2], &wr_log_r[4][r_0][r_1][r_2],
			    &rd_log_r[4][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) * pivot));
	coeff =
	    spec_read_double(&lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			  &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[0][r_0][r_1][r_2], &wr_log_r[0][r_0][r_1][r_2],
			    &rd_log_r[0][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			  &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[1][r_0][r_1][r_2], &wr_log_r[1][r_0][r_1][r_2],
			    &rd_log_r[1][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			  &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[2][r_0][r_1][r_2], &wr_log_r[2][r_0][r_1][r_2],
			    &rd_log_r[2][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));
	coeff =
	    spec_read_double(&lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     &wr_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     &rd_log_lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2],
			     tls_thread_id, tls_validflag);
	spec_write_double(&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			  &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			  tls_validflag,
			  (spec_read_double
			   (&r[3][r_0][r_1][r_2], &wr_log_r[3][r_0][r_1][r_2],
			    &rd_log_r[3][r_0][r_1][r_2], tls_thread_id,
			    tls_validflag) -
			   coeff * spec_read_double(&r[4][r_0][r_1][r_2],
						    &wr_log_r[4][r_0][r_1][r_2],
						    &rd_log_r[4][r_0][r_1][r_2],
						    tls_thread_id,
						    tls_validflag)));

}

//-------------------------------------------------------------------------------
//This is an alias of function: exact_solution
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_solution_g4_no_spec(double xi, double eta, double zeta,
			       double dtemp[5], __global double (*ce)[13],
			       __global int *tls_validflag, int tls_thread_id)
{
	int m;
	for (m = 0; m < 5; m++) {
		dtemp[m] =
		    ce[m][0] + xi * (ce[m][1] +
				     xi * (ce[m][4] +
					   xi * (ce[m][7] + xi * ce[m][10]))) +
		    eta * (ce[m][2] +
			   eta * (ce[m][5] +
				  eta * (ce[m][8] + eta * ce[m][11]))) +
		    zeta * (ce[m][3] +
			    zeta * (ce[m][6] +
				    zeta * (ce[m][9] + zeta * ce[m][12])));
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: binvcrhs
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: lhs
//      1: c
//      2: r
//-------------------------------------------------------------------------------
void binvcrhs_g0_g5_g10_no_spec(__global double (*lhs)[5][5][65][65][65],
				int lhs_0, int lhs_1, int lhs_2, int lhs_3,
				__global double (*c)[5][5][65][65][65], int c_0,
				int c_1, int c_2, int c_3,
				__global double (*r)[65][65][65], int r_0,
				int r_1, int r_2, __global int *tls_validflag,
				int tls_thread_id)
{
	double pivot, coeff;
	pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][0][0][c_0][c_1][c_2] = c[c_3][0][0][c_0][c_1][c_2] * pivot;
	c[c_3][0][1][c_0][c_1][c_2] = c[c_3][0][1][c_0][c_1][c_2] * pivot;
	c[c_3][0][2][c_0][c_1][c_2] = c[c_3][0][2][c_0][c_1][c_2] * pivot;
	c[c_3][0][3][c_0][c_1][c_2] = c[c_3][0][3][c_0][c_1][c_2] * pivot;
	c[c_3][0][4][c_0][c_1][c_2] = c[c_3][0][4][c_0][c_1][c_2] * pivot;
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][0][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][0][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][0][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][0][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][0][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][1][0][c_0][c_1][c_2] = c[c_3][1][0][c_0][c_1][c_2] * pivot;
	c[c_3][1][1][c_0][c_1][c_2] = c[c_3][1][1][c_0][c_1][c_2] * pivot;
	c[c_3][1][2][c_0][c_1][c_2] = c[c_3][1][2][c_0][c_1][c_2] * pivot;
	c[c_3][1][3][c_0][c_1][c_2] = c[c_3][1][3][c_0][c_1][c_2] * pivot;
	c[c_3][1][4][c_0][c_1][c_2] = c[c_3][1][4][c_0][c_1][c_2] * pivot;
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][1][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][1][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][1][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][1][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][1][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][2][0][c_0][c_1][c_2] = c[c_3][2][0][c_0][c_1][c_2] * pivot;
	c[c_3][2][1][c_0][c_1][c_2] = c[c_3][2][1][c_0][c_1][c_2] * pivot;
	c[c_3][2][2][c_0][c_1][c_2] = c[c_3][2][2][c_0][c_1][c_2] * pivot;
	c[c_3][2][3][c_0][c_1][c_2] = c[c_3][2][3][c_0][c_1][c_2] * pivot;
	c[c_3][2][4][c_0][c_1][c_2] = c[c_3][2][4][c_0][c_1][c_2] * pivot;
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][2][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][2][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][2][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][2][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][2][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
	c[c_3][3][0][c_0][c_1][c_2] = c[c_3][3][0][c_0][c_1][c_2] * pivot;
	c[c_3][3][1][c_0][c_1][c_2] = c[c_3][3][1][c_0][c_1][c_2] * pivot;
	c[c_3][3][2][c_0][c_1][c_2] = c[c_3][3][2][c_0][c_1][c_2] * pivot;
	c[c_3][3][3][c_0][c_1][c_2] = c[c_3][3][3][c_0][c_1][c_2] * pivot;
	c[c_3][3][4][c_0][c_1][c_2] = c[c_3][3][4][c_0][c_1][c_2] * pivot;
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] =
	    c[c_3][4][0][c_0][c_1][c_2] - coeff * c[c_3][3][0][c_0][c_1][c_2];
	c[c_3][4][1][c_0][c_1][c_2] =
	    c[c_3][4][1][c_0][c_1][c_2] - coeff * c[c_3][3][1][c_0][c_1][c_2];
	c[c_3][4][2][c_0][c_1][c_2] =
	    c[c_3][4][2][c_0][c_1][c_2] - coeff * c[c_3][3][2][c_0][c_1][c_2];
	c[c_3][4][3][c_0][c_1][c_2] =
	    c[c_3][4][3][c_0][c_1][c_2] - coeff * c[c_3][3][3][c_0][c_1][c_2];
	c[c_3][4][4][c_0][c_1][c_2] =
	    c[c_3][4][4][c_0][c_1][c_2] - coeff * c[c_3][3][4][c_0][c_1][c_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	c[c_3][4][0][c_0][c_1][c_2] = c[c_3][4][0][c_0][c_1][c_2] * pivot;
	c[c_3][4][1][c_0][c_1][c_2] = c[c_3][4][1][c_0][c_1][c_2] * pivot;
	c[c_3][4][2][c_0][c_1][c_2] = c[c_3][4][2][c_0][c_1][c_2] * pivot;
	c[c_3][4][3][c_0][c_1][c_2] = c[c_3][4][3][c_0][c_1][c_2] * pivot;
	c[c_3][4][4][c_0][c_1][c_2] = c[c_3][4][4][c_0][c_1][c_2] * pivot;
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	c[c_3][0][0][c_0][c_1][c_2] =
	    c[c_3][0][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][0][1][c_0][c_1][c_2] =
	    c[c_3][0][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][0][2][c_0][c_1][c_2] =
	    c[c_3][0][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][0][3][c_0][c_1][c_2] =
	    c[c_3][0][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][0][4][c_0][c_1][c_2] =
	    c[c_3][0][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	c[c_3][1][0][c_0][c_1][c_2] =
	    c[c_3][1][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][1][1][c_0][c_1][c_2] =
	    c[c_3][1][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][1][2][c_0][c_1][c_2] =
	    c[c_3][1][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][1][3][c_0][c_1][c_2] =
	    c[c_3][1][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][1][4][c_0][c_1][c_2] =
	    c[c_3][1][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	c[c_3][2][0][c_0][c_1][c_2] =
	    c[c_3][2][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][2][1][c_0][c_1][c_2] =
	    c[c_3][2][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][2][2][c_0][c_1][c_2] =
	    c[c_3][2][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][2][3][c_0][c_1][c_2] =
	    c[c_3][2][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][2][4][c_0][c_1][c_2] =
	    c[c_3][2][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	c[c_3][3][0][c_0][c_1][c_2] =
	    c[c_3][3][0][c_0][c_1][c_2] - coeff * c[c_3][4][0][c_0][c_1][c_2];
	c[c_3][3][1][c_0][c_1][c_2] =
	    c[c_3][3][1][c_0][c_1][c_2] - coeff * c[c_3][4][1][c_0][c_1][c_2];
	c[c_3][3][2][c_0][c_1][c_2] =
	    c[c_3][3][2][c_0][c_1][c_2] - coeff * c[c_3][4][2][c_0][c_1][c_2];
	c[c_3][3][3][c_0][c_1][c_2] =
	    c[c_3][3][3][c_0][c_1][c_2] - coeff * c[c_3][4][3][c_0][c_1][c_2];
	c[c_3][3][4][c_0][c_1][c_2] =
	    c[c_3][3][4][c_0][c_1][c_2] - coeff * c[c_3][4][4][c_0][c_1][c_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];

}

//-------------------------------------------------------------------------------
//This is an alias of function: matvec_sub
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ablock
//      1: avec
//      2: bvec
//-------------------------------------------------------------------------------
void matvec_sub_g0_g5_g9_no_spec(__global double (*ablock)[5][5][65][65][65],
				 int ablock_0, int ablock_1, int ablock_2,
				 int ablock_3,
				 __global double (*avec)[65][65][65],
				 int avec_0, int avec_1, int avec_2,
				 __global double (*bvec)[65][65][65],
				 int bvec_0, int bvec_1, int bvec_2,
				 __global int *tls_validflag, int tls_thread_id)
{
	int i;
	for (i = 0; i < 5; i++) {
		bvec[i][bvec_0][bvec_1][bvec_2] =
		    bvec[i][bvec_0][bvec_1][bvec_2] -
		    ablock[ablock_3][i][0][ablock_0][ablock_1][ablock_2] *
		    avec[0][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][1][ablock_0][ablock_1][ablock_2] *
		    avec[1][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][2][ablock_0][ablock_1][ablock_2] *
		    avec[2][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][3][ablock_0][ablock_1][ablock_2] *
		    avec[3][avec_0][avec_1][avec_2] -
		    ablock[ablock_3][i][4][ablock_0][ablock_1][ablock_2] *
		    avec[4][avec_0][avec_1][avec_2];
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: matmul_sub
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ablock
//      1: bblock
//      2: cblock
//-------------------------------------------------------------------------------
void matmul_sub_g0_g5_g10_no_spec(__global double (*ablock)[5][5][65][65][65],
				  int ablock_0, int ablock_1, int ablock_2,
				  int ablock_3,
				  __global double (*bblock)[5][5][65][65][65],
				  int bblock_0, int bblock_1, int bblock_2,
				  int bblock_3,
				  __global double (*cblock)[5][5][65][65][65],
				  int cblock_0, int cblock_1, int cblock_2,
				  int cblock_3, __global int *tls_validflag,
				  int tls_thread_id)
{
	int j;
	for (j = 0; j < 5; j++) {
		cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][0][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][0][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][0][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][1][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][1][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][1][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][2][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][2][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][2][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][3][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][3][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][3][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
		cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] =
		    cblock[cblock_3][4][j][cblock_0][cblock_1][cblock_2] -
		    ablock[ablock_3][4][0][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][0][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][1][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][1][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][2][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][2][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][3][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][3][j][bblock_0][bblock_1][bblock_2] -
		    ablock[ablock_3][4][4][ablock_0][ablock_1][ablock_2] *
		    bblock[bblock_3][4][j][bblock_0][bblock_1][bblock_2];
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: binvrhs
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: lhs
//      1: r
//-------------------------------------------------------------------------------
void binvrhs_g0_g5_no_spec(__global double (*lhs)[5][5][65][65][65], int lhs_0,
			   int lhs_1, int lhs_2, int lhs_3,
			   __global double (*r)[65][65][65], int r_0, int r_1,
			   int r_2, __global int *tls_validflag,
			   int tls_thread_id)
{
	double pivot, coeff;
	pivot = 1.00 / lhs[lhs_3][0][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][1][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][0][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[0][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][1][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][1][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[1][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][2][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2] * pivot;
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][2][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[2][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][3][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2] * pivot;
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	coeff = lhs[lhs_3][4][3][lhs_0][lhs_1][lhs_2];
	lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] =
	    lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2] -
	    coeff * lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] - coeff * r[3][r_0][r_1][r_2];
	pivot = 1.00 / lhs[lhs_3][4][4][lhs_0][lhs_1][lhs_2];
	r[4][r_0][r_1][r_2] = r[4][r_0][r_1][r_2] * pivot;
	coeff = lhs[lhs_3][0][4][lhs_0][lhs_1][lhs_2];
	r[0][r_0][r_1][r_2] = r[0][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][1][4][lhs_0][lhs_1][lhs_2];
	r[1][r_0][r_1][r_2] = r[1][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][2][4][lhs_0][lhs_1][lhs_2];
	r[2][r_0][r_1][r_2] = r[2][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];
	coeff = lhs[lhs_3][3][4][lhs_0][lhs_1][lhs_2];
	r[3][r_0][r_1][r_2] = r[3][r_0][r_1][r_2] - coeff * r[4][r_0][r_1][r_2];

}

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 225 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void add_0(__global double *g_u, __global double *g_rhs,
		    int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		    __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 221 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = u[m][i][j][k] + rhs[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 348 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void exact_rhs_0(__global double *g_forcing, int __ocl_k_bound,
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			forcing[m][i][j][k] = 0.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 364 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void exact_rhs_1(double dnym1, double dnzm1, __global int *grid_points,
			  double dnxm1, __global double *g_forcing, double tx2,
			  double dx1tx1, double c2, double xxcon1,
			  double dx2tx1, double xxcon2, double dx3tx1,
			  double dx4tx1, double c1, double xxcon3,
			  double xxcon4, double xxcon5, double dx5tx1,
			  double dssp, __global double *g_ce, int __ocl_k_bound,
			  int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	int i;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double dtemp[5];	/* (User-defined privated variables) : Defined at bt.c : 340 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double ue[5][64];	/* threadprivate: defined at ./header.h : 74 */
	double dtpp;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double buf[5][64];	/* threadprivate: defined at ./header.h : 75 */
	double cuf[64];		/* threadprivate: defined at ./header.h : 72 */
	double q[64];		/* threadprivate: defined at ./header.h : 73 */
	int im1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	int ip1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		zeta = (double)k *dnzm1;
		for (i = 0; i < grid_points[0]; i++) {
			xi = (double)i *dnxm1;
			exact_solution_g4_no_spec(xi, eta, zeta, dtemp, ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
			for (m = 0; m < 5; m++) {
				ue[m][i] = dtemp[m];
			}
			dtpp = 1.0 / dtemp[0];
			for (m = 1; m <= 4; m++) {
				buf[m][i] = dtpp * dtemp[m];
			}
			cuf[i] = buf[1][i] * buf[1][i];
			buf[0][i] =
			    cuf[i] + buf[2][i] * buf[2][i] +
			    buf[3][i] * buf[3][i];
			q[i] =
			    0.5 * (buf[1][i] * ue[1][i] + buf[2][i] * ue[2][i] +
				   buf[3][i] * ue[3][i]);
		}
		for (i = 1; i < grid_points[0] - 1; i++) {
			im1 = i - 1;
			ip1 = i + 1;
			forcing[0][i][j][k] =
			    forcing[0][i][j][k] - tx2 * (ue[1][ip1] -
							 ue[1][im1]) +
			    dx1tx1 * (ue[0][ip1] - 2.0 * ue[0][i] + ue[0][im1]);
			forcing[1][i][j][k] =
			    forcing[1][i][j][k] -
			    tx2 *
			    ((ue[1][ip1] * buf[1][ip1] +
			      c2 * (ue[4][ip1] - q[ip1])) -
			     (ue[1][im1] * buf[1][im1] +
			      c2 * (ue[4][im1] - q[im1]))) +
			    xxcon1 * (buf[1][ip1] - 2.0 * buf[1][i] +
				      buf[1][im1]) + dx2tx1 * (ue[1][ip1] -
							       2.0 * ue[1][i] +
							       ue[1][im1]);
			forcing[2][i][j][k] =
			    forcing[2][i][j][k] -
			    tx2 * (ue[2][ip1] * buf[1][ip1] -
				   ue[2][im1] * buf[1][im1]) +
			    xxcon2 * (buf[2][ip1] - 2.0 * buf[2][i] +
				      buf[2][im1]) + dx3tx1 * (ue[2][ip1] -
							       2.0 * ue[2][i] +
							       ue[2][im1]);
			forcing[3][i][j][k] =
			    forcing[3][i][j][k] -
			    tx2 * (ue[3][ip1] * buf[1][ip1] -
				   ue[3][im1] * buf[1][im1]) +
			    xxcon2 * (buf[3][ip1] - 2.0 * buf[3][i] +
				      buf[3][im1]) + dx4tx1 * (ue[3][ip1] -
							       2.0 * ue[3][i] +
							       ue[3][im1]);
			forcing[4][i][j][k] =
			    forcing[4][i][j][k] -
			    tx2 * (buf[1][ip1] *
				   (c1 * ue[4][ip1] - c2 * q[ip1]) -
				   buf[1][im1] * (c1 * ue[4][im1] -
						  c2 * q[im1])) +
			    0.5 * xxcon3 * (buf[0][ip1] - 2.0 * buf[0][i] +
					    buf[0][im1]) + xxcon4 * (cuf[ip1] -
								     2.0 *
								     cuf[i] +
								     cuf[im1]) +
			    xxcon5 * (buf[4][ip1] - 2.0 * buf[4][i] +
				      buf[4][im1]) + dx5tx1 * (ue[4][ip1] -
							       2.0 * ue[4][i] +
							       ue[4][im1]);
		}
		for (m = 0; m < 5; m++) {
			i = 1;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (5.0 * ue[m][i] -
							  4.0 * ue[m][i + 1] +
							  ue[m][i + 2]);
			i = 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (-4.0 * ue[m][i - 1] +
							  6.0 * ue[m][i] -
							  4.0 * ue[m][i + 1] +
							  ue[m][i + 2]);
		}
		for (m = 0; m < 5; m++) {
			for (i = 1 * 3; i <= grid_points[0] - 3 * 1 - 1; i++) {
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][i - 2] -
								  4.0 *
								  ue[m][i - 1] +
								  6.0 *
								  ue[m][i] -
								  4.0 *
								  ue[m][i + 1] +
								  ue[m][i + 2]);
			}
		}
		for (m = 0; m < 5; m++) {
			i = grid_points[0] - 3;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][i - 2] -
							  4.0 * ue[m][i - 1] +
							  6.0 * ue[m][i] -
							  4.0 * ue[m][i + 1]);
			i = grid_points[0] - 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][i - 2] -
							  4.0 * ue[m][i - 1] +
							  5.0 * ue[m][i]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 464 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void exact_rhs_2(double dnxm1, double dnzm1, __global int *grid_points,
			  double dnym1, __global double *g_forcing, double ty2,
			  double dy1ty1, double yycon2, double dy2ty1,
			  double c2, double yycon1, double dy3ty1,
			  double dy4ty1, double c1, double yycon3,
			  double yycon4, double yycon5, double dy5ty1,
			  double dssp, __global double *g_ce, int __ocl_k_bound,
			  int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	int j;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double dtemp[5];	/* (User-defined privated variables) : Defined at bt.c : 340 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double ue[5][64];	/* threadprivate: defined at ./header.h : 74 */
	double dtpp;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double buf[5][64];	/* threadprivate: defined at ./header.h : 75 */
	double cuf[64];		/* threadprivate: defined at ./header.h : 72 */
	double q[64];		/* threadprivate: defined at ./header.h : 73 */
	int jm1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	int jp1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		zeta = (double)k *dnzm1;
		for (j = 0; j < grid_points[1]; j++) {
			eta = (double)j *dnym1;
			exact_solution_g4_no_spec(xi, eta, zeta, dtemp, ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
			for (m = 0; m < 5; m++) {
				ue[m][j] = dtemp[m];
			}
			dtpp = 1.0 / dtemp[0];
			for (m = 1; m <= 4; m++) {
				buf[m][j] = dtpp * dtemp[m];
			}
			cuf[j] = buf[2][j] * buf[2][j];
			buf[0][j] =
			    cuf[j] + buf[1][j] * buf[1][j] +
			    buf[3][j] * buf[3][j];
			q[j] =
			    0.5 * (buf[1][j] * ue[1][j] + buf[2][j] * ue[2][j] +
				   buf[3][j] * ue[3][j]);
		}
		for (j = 1; j < grid_points[1] - 1; j++) {
			jm1 = j - 1;
			jp1 = j + 1;
			forcing[0][i][j][k] =
			    forcing[0][i][j][k] - ty2 * (ue[2][jp1] -
							 ue[2][jm1]) +
			    dy1ty1 * (ue[0][jp1] - 2.0 * ue[0][j] + ue[0][jm1]);
			forcing[1][i][j][k] =
			    forcing[1][i][j][k] -
			    ty2 * (ue[1][jp1] * buf[2][jp1] -
				   ue[1][jm1] * buf[2][jm1]) +
			    yycon2 * (buf[1][jp1] - 2.0 * buf[1][j] +
				      buf[1][jm1]) + dy2ty1 * (ue[1][jp1] -
							       2.0 * ue[1][j] +
							       ue[1][jm1]);
			forcing[2][i][j][k] =
			    forcing[2][i][j][k] -
			    ty2 *
			    ((ue[2][jp1] * buf[2][jp1] +
			      c2 * (ue[4][jp1] - q[jp1])) -
			     (ue[2][jm1] * buf[2][jm1] +
			      c2 * (ue[4][jm1] - q[jm1]))) +
			    yycon1 * (buf[2][jp1] - 2.0 * buf[2][j] +
				      buf[2][jm1]) + dy3ty1 * (ue[2][jp1] -
							       2.0 * ue[2][j] +
							       ue[2][jm1]);
			forcing[3][i][j][k] =
			    forcing[3][i][j][k] -
			    ty2 * (ue[3][jp1] * buf[2][jp1] -
				   ue[3][jm1] * buf[2][jm1]) +
			    yycon2 * (buf[3][jp1] - 2.0 * buf[3][j] +
				      buf[3][jm1]) + dy4ty1 * (ue[3][jp1] -
							       2.0 * ue[3][j] +
							       ue[3][jm1]);
			forcing[4][i][j][k] =
			    forcing[4][i][j][k] -
			    ty2 * (buf[2][jp1] *
				   (c1 * ue[4][jp1] - c2 * q[jp1]) -
				   buf[2][jm1] * (c1 * ue[4][jm1] -
						  c2 * q[jm1])) +
			    0.5 * yycon3 * (buf[0][jp1] - 2.0 * buf[0][j] +
					    buf[0][jm1]) + yycon4 * (cuf[jp1] -
								     2.0 *
								     cuf[j] +
								     cuf[jm1]) +
			    yycon5 * (buf[4][jp1] - 2.0 * buf[4][j] +
				      buf[4][jm1]) + dy5ty1 * (ue[4][jp1] -
							       2.0 * ue[4][j] +
							       ue[4][jm1]);
		}
		for (m = 0; m < 5; m++) {
			j = 1;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (5.0 * ue[m][j] -
							  4.0 * ue[m][j + 1] +
							  ue[m][j + 2]);
			j = 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (-4.0 * ue[m][j - 1] +
							  6.0 * ue[m][j] -
							  4.0 * ue[m][j + 1] +
							  ue[m][j + 2]);
		}
		for (m = 0; m < 5; m++) {
			for (j = 1 * 3; j <= grid_points[1] - 3 * 1 - 1; j++) {
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][j - 2] -
								  4.0 *
								  ue[m][j - 1] +
								  6.0 *
								  ue[m][j] -
								  4.0 *
								  ue[m][j + 1] +
								  ue[m][j + 2]);
			}
		}
		for (m = 0; m < 5; m++) {
			j = grid_points[1] - 3;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][j - 2] -
							  4.0 * ue[m][j - 1] +
							  6.0 * ue[m][j] -
							  4.0 * ue[m][j + 1]);
			j = grid_points[1] - 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][j - 2] -
							  4.0 * ue[m][j - 1] +
							  5.0 * ue[m][j]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 566 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void exact_rhs_3(double dnxm1, double dnym1, __global int *grid_points,
			  double dnzm1, __global double *g_forcing, double tz2,
			  double dz1tz1, double zzcon2, double dz2tz1,
			  double dz3tz1, double c2, double zzcon1,
			  double dz4tz1, double c1, double zzcon3,
			  double zzcon4, double zzcon5, double dz5tz1,
			  double dssp, __global double *g_ce, int __ocl_j_bound,
			  int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	int k;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double dtemp[5];	/* (User-defined privated variables) : Defined at bt.c : 340 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	double ue[5][64];	/* threadprivate: defined at ./header.h : 74 */
	double dtpp;		/* (User-defined privated variables) : Defined at bt.c : 340 */
	double buf[5][64];	/* threadprivate: defined at ./header.h : 75 */
	double cuf[64];		/* threadprivate: defined at ./header.h : 72 */
	double q[64];		/* threadprivate: defined at ./header.h : 73 */
	int km1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	int kp1;		/* (User-defined privated variables) : Defined at bt.c : 341 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[2]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4_no_spec(xi, eta, zeta, dtemp, ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
			for (m = 0; m < 5; m++) {
				ue[m][k] = dtemp[m];
			}
			dtpp = 1.0 / dtemp[0];
			for (m = 1; m <= 4; m++) {
				buf[m][k] = dtpp * dtemp[m];
			}
			cuf[k] = buf[3][k] * buf[3][k];
			buf[0][k] =
			    cuf[k] + buf[1][k] * buf[1][k] +
			    buf[2][k] * buf[2][k];
			q[k] =
			    0.5 * (buf[1][k] * ue[1][k] + buf[2][k] * ue[2][k] +
				   buf[3][k] * ue[3][k]);
		}
		for (k = 1; k < grid_points[2] - 1; k++) {
			km1 = k - 1;
			kp1 = k + 1;
			forcing[0][i][j][k] =
			    forcing[0][i][j][k] - tz2 * (ue[3][kp1] -
							 ue[3][km1]) +
			    dz1tz1 * (ue[0][kp1] - 2.0 * ue[0][k] + ue[0][km1]);
			forcing[1][i][j][k] =
			    forcing[1][i][j][k] -
			    tz2 * (ue[1][kp1] * buf[3][kp1] -
				   ue[1][km1] * buf[3][km1]) +
			    zzcon2 * (buf[1][kp1] - 2.0 * buf[1][k] +
				      buf[1][km1]) + dz2tz1 * (ue[1][kp1] -
							       2.0 * ue[1][k] +
							       ue[1][km1]);
			forcing[2][i][j][k] =
			    forcing[2][i][j][k] -
			    tz2 * (ue[2][kp1] * buf[3][kp1] -
				   ue[2][km1] * buf[3][km1]) +
			    zzcon2 * (buf[2][kp1] - 2.0 * buf[2][k] +
				      buf[2][km1]) + dz3tz1 * (ue[2][kp1] -
							       2.0 * ue[2][k] +
							       ue[2][km1]);
			forcing[3][i][j][k] =
			    forcing[3][i][j][k] -
			    tz2 *
			    ((ue[3][kp1] * buf[3][kp1] +
			      c2 * (ue[4][kp1] - q[kp1])) -
			     (ue[3][km1] * buf[3][km1] +
			      c2 * (ue[4][km1] - q[km1]))) +
			    zzcon1 * (buf[3][kp1] - 2.0 * buf[3][k] +
				      buf[3][km1]) + dz4tz1 * (ue[3][kp1] -
							       2.0 * ue[3][k] +
							       ue[3][km1]);
			forcing[4][i][j][k] =
			    forcing[4][i][j][k] -
			    tz2 * (buf[3][kp1] *
				   (c1 * ue[4][kp1] - c2 * q[kp1]) -
				   buf[3][km1] * (c1 * ue[4][km1] -
						  c2 * q[km1])) +
			    0.5 * zzcon3 * (buf[0][kp1] - 2.0 * buf[0][k] +
					    buf[0][km1]) + zzcon4 * (cuf[kp1] -
								     2.0 *
								     cuf[k] +
								     cuf[km1]) +
			    zzcon5 * (buf[4][kp1] - 2.0 * buf[4][k] +
				      buf[4][km1]) + dz5tz1 * (ue[4][kp1] -
							       2.0 * ue[4][k] +
							       ue[4][km1]);
		}
		for (m = 0; m < 5; m++) {
			k = 1;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (5.0 * ue[m][k] -
							  4.0 * ue[m][k + 1] +
							  ue[m][k + 2]);
			k = 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (-4.0 * ue[m][k - 1] +
							  6.0 * ue[m][k] -
							  4.0 * ue[m][k + 1] +
							  ue[m][k + 2]);
		}
		for (m = 0; m < 5; m++) {
			for (k = 1 * 3; k <= grid_points[2] - 3 * 1 - 1; k++) {
				forcing[m][i][j][k] =
				    forcing[m][i][j][k] - dssp * (ue[m][k - 2] -
								  4.0 *
								  ue[m][k - 1] +
								  6.0 *
								  ue[m][k] -
								  4.0 *
								  ue[m][k + 1] +
								  ue[m][k + 2]);
			}
		}
		for (m = 0; m < 5; m++) {
			k = grid_points[2] - 3;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][k - 2] -
							  4.0 * ue[m][k - 1] +
							  6.0 * ue[m][k] -
							  4.0 * ue[m][k + 1]);
			k = grid_points[2] - 2;
			forcing[m][i][j][k] =
			    forcing[m][i][j][k] - dssp * (ue[m][k - 2] -
							  4.0 * ue[m][k - 1] +
							  5.0 * ue[m][k]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 666 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void exact_rhs_4(__global double *g_forcing, int __ocl_k_bound,
			  int __ocl_j_bound, int __ocl_i_bound,
			  __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 341 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			forcing[m][i][j][k] = -1.0 * forcing[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 728 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void initialize_0(__global double *g_u, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1);
	int i = get_global_id(2);
	if (!(k < 64)) {
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = 1.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 745 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_1(double dnym1, double dnxm1, double dnzm1,
			   __global double *g_u, __global double *g_ce,
			   int __ocl_k_bound, int __ocl_j_bound,
			   int __ocl_i_bound, __global int *tls_validflag)
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
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int ix;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	double Pface[2][3][5];	/* (User-defined privated variables) : Defined at bt.c : 718 */
	int iy;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	int iz;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	double Pxi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double Peta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double Pzeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		xi = (double)i *dnxm1;
		zeta = (double)k *dnzm1;
		for (ix = 0; ix < 2; ix++) {
			exact_solution_g4_no_spec((double)ix, eta, zeta,
						  &(Pface[ix][0][0]), ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
		}
		for (iy = 0; iy < 2; iy++) {
			exact_solution_g4_no_spec(xi, (double)iy, zeta,
						  &Pface[iy][1][0], ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
		}
		for (iz = 0; iz < 2; iz++) {
			exact_solution_g4_no_spec(xi, eta, (double)iz,
						  &Pface[iz][2][0], ce,
						  tls_validflag,
						  tls_thread_id) /*ARGEXP: ce */
			    ;
		}
		for (m = 0; m < 5; m++) {
			Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
			Peta =
			    eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
			Pzeta =
			    zeta * Pface[1][2][m] + (1.0 -
						     zeta) * Pface[0][2][m];
			u[m][i][j][k] =
			    Pxi + Peta + Pzeta - Pxi * Peta - Pxi * Pzeta -
			    Peta * Pzeta + Pxi * Peta * Pzeta;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 795 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_2(double dnym1, double dnzm1, double xi,
			   __global double *g_u, int i, __global double *g_ce,
			   int __ocl_k_bound, int __ocl_j_bound,
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
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		zeta = (double)k *dnzm1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 815 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_3(double dnym1, double dnzm1, double xi,
			   __global double *g_u, int i, __global double *g_ce,
			   int __ocl_k_bound, int __ocl_j_bound,
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
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		zeta = (double)k *dnzm1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 834 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_4(double dnxm1, double dnzm1, double eta,
			   __global double *g_u, int j, __global double *g_ce,
			   int __ocl_k_bound, int __ocl_i_bound,
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		zeta = (double)k *dnzm1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 853 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_5(double dnxm1, double dnzm1, double eta,
			   __global double *g_u, int j, __global double *g_ce,
			   int __ocl_k_bound, int __ocl_i_bound,
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double zeta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		zeta = (double)k *dnzm1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 872 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_6(double dnxm1, double dnym1, double zeta,
			   __global double *g_u, int k, __global double *g_ce,
			   int __ocl_j_bound, int __ocl_i_bound,
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		eta = (double)j *dnym1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 891 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void initialize_7(double dnxm1, double dnym1, double zeta,
			   __global double *g_u, int k, __global double *g_ce,
			   int __ocl_j_bound, int __ocl_i_bound,
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
	double xi;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double eta;		/* (User-defined privated variables) : Defined at bt.c : 718 */
	double temp[5];		/* (User-defined privated variables) : Defined at bt.c : 718 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 717 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*ce)[13] = (__global double (*)[13])g_ce;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		eta = (double)j *dnym1;
		exact_solution_g4_no_spec(xi, eta, zeta, temp, ce,
					  tls_validflag,
					  tls_thread_id) /*ARGEXP: ce */ ;
		for (m = 0; m < 5; m++) {
			u[m][i][j][k] = temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 919 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsinit_0(__global double *g_lhs, int __ocl_k_bound,
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 909 */
	int n;			/* (User-defined privated variables) : Defined at bt.c : 909 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			for (n = 0; n < 5; n++) {
				lhs[0][m][n][i][j][k] = 0.0;
				lhs[1][m][n][i][j][k] = 0.0;
				lhs[2][m][n][i][j][k] = 0.0;
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 939 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsinit_1(__global double *g_lhs, int __ocl_k_bound,
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 909 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			lhs[1][m][m][i][j][k] = 1.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 970 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsx_0(__global int *grid_points, __global double *g_u,
		     __global double *g_fjac, double c2, double c1,
		     __global double *g_njac, double con43, double c3c4,
		     double c1345, double dt, double tx1, double tx2,
		     __global double *g_lhs, double dx1, double dx2, double dx3,
		     double dx4, double dx5, int __ocl_k_bound,
		     int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	int i;			/* (User-defined privated variables) : Defined at bt.c : 963 */
	double tmp1;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp2;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp3;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*fjac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_fjac;
	__global double (*njac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_njac;
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = 0; i < grid_points[0]; i++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 972
			//-------------------------------------------
			double u_3[4];
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 972
			//Candidates:
			//      u[0][i][j][k]
			//      u[1][i][j][k]
			//      u[2][i][j][k]
			//      u[3][i][j][k]
			//-------------------------------------------
			u_3[0] = u[0][i][j][k];
			u_3[1] = u[1][i][j][k];
			u_3[2] = u[2][i][j][k];
			u_3[3] = u[3][i][j][k];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmp1 = 1.0 / u_3[0] /*u[0][i][j][k] */ ;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[0][0][i][j][k] = 0.0;
			fjac[0][1][i][j][k] = 1.0;
			fjac[0][2][i][j][k] = 0.0;
			fjac[0][3][i][j][k] = 0.0;
			fjac[0][4][i][j][k] = 0.0;
			fjac[1][0][i][j][k] =
			    -(u_3[1] /*u[1][i][j][k] */ *tmp2 *
			      u_3[1] /*u[1][i][j][k] */ ) +
			    c2 * 0.50 *
			    (u_3[1] /*u[1][i][j][k] */ *u_3[1]
			     /*u[1][i][j][k] */ +u_3[2] /*u[2][i][j][k] */
			     *u_3[2] /*u[2][i][j][k] */ +u_3[3]
			     /*u[3][i][j][k] */ *u_3[3] /*u[3][i][j][k] */ ) *
			    tmp2;
			fjac[1][1][i][j][k] =
			    (2.0 -
			     c2) *
			    (u_3[1] /*u[1][i][j][k] */ /u_3[0]
			     /*u[0][i][j][k] */ );
			fjac[1][2][i][j][k] =
			    -c2 * (u_3[2] /*u[2][i][j][k] */ *tmp1);
			fjac[1][3][i][j][k] =
			    -c2 * (u_3[3] /*u[3][i][j][k] */ *tmp1);
			fjac[1][4][i][j][k] = c2;
			fjac[2][0][i][j][k] =
			    -(u_3[1] /*u[1][i][j][k] */ *u_3[2]
			      /*u[2][i][j][k] */ ) * tmp2;
			fjac[2][1][i][j][k] = u_3[2] /*u[2][i][j][k] */ *tmp1;
			fjac[2][2][i][j][k] = u_3[1] /*u[1][i][j][k] */ *tmp1;
			fjac[2][3][i][j][k] = 0.0;
			fjac[2][4][i][j][k] = 0.0;
			fjac[3][0][i][j][k] =
			    -(u_3[1] /*u[1][i][j][k] */ *u_3[3]
			      /*u[3][i][j][k] */ ) * tmp2;
			fjac[3][1][i][j][k] = u_3[3] /*u[3][i][j][k] */ *tmp1;
			fjac[3][2][i][j][k] = 0.0;
			fjac[3][3][i][j][k] = u_3[1] /*u[1][i][j][k] */ *tmp1;
			fjac[3][4][i][j][k] = 0.0;
			fjac[4][0][i][j][k] =
			    (c2 *
			     (u_3[1] /*u[1][i][j][k] */ *u_3[1]
			      /*u[1][i][j][k] */ +u_3[2] /*u[2][i][j][k] */
			      *u_3[2] /*u[2][i][j][k] */ +u_3[3]
			      /*u[3][i][j][k] */ *u_3[3] /*u[3][i][j][k] */ ) *
			     tmp2 -
			     c1 * (u[4][i][j][k] * tmp1)) *
			    (u_3[1] /*u[1][i][j][k] */ *tmp1);
			fjac[4][1][i][j][k] =
			    c1 * u[4][i][j][k] * tmp1 -
			    0.50 * c2 * (3.0 *
					 u_3[1] /*u[1][i][j][k] */ *u_3[1]
					 /*u[1][i][j][k] */ +u_3[2]
					 /*u[2][i][j][k] */ *u_3[2]
					 /*u[2][i][j][k] */ +u_3[3]
					 /*u[3][i][j][k] */ *u_3[3]
					 /*u[3][i][j][k] */ ) * tmp2;
			fjac[4][2][i][j][k] =
			    -c2 *
			    (u_3[2] /*u[2][i][j][k] */ *u_3[1]
			     /*u[1][i][j][k] */ ) * tmp2;
			fjac[4][3][i][j][k] =
			    -c2 *
			    (u_3[3] /*u[3][i][j][k] */ *u_3[1]
			     /*u[1][i][j][k] */ ) * tmp2;
			fjac[4][4][i][j][k] =
			    c1 * (u_3[1] /*u[1][i][j][k] */ *tmp1);
			njac[0][0][i][j][k] = 0.0;
			njac[0][1][i][j][k] = 0.0;
			njac[0][2][i][j][k] = 0.0;
			njac[0][3][i][j][k] = 0.0;
			njac[0][4][i][j][k] = 0.0;
			njac[1][0][i][j][k] =
			    -con43 * c3c4 * tmp2 * u_3[1] /*u[1][i][j][k] */ ;
			njac[1][1][i][j][k] = con43 * c3c4 * tmp1;
			njac[1][2][i][j][k] = 0.0;
			njac[1][3][i][j][k] = 0.0;
			njac[1][4][i][j][k] = 0.0;
			njac[2][0][i][j][k] =
			    -c3c4 * tmp2 * u_3[2] /*u[2][i][j][k] */ ;
			njac[2][1][i][j][k] = 0.0;
			njac[2][2][i][j][k] = c3c4 * tmp1;
			njac[2][3][i][j][k] = 0.0;
			njac[2][4][i][j][k] = 0.0;
			njac[3][0][i][j][k] =
			    -c3c4 * tmp2 * u_3[3] /*u[3][i][j][k] */ ;
			njac[3][1][i][j][k] = 0.0;
			njac[3][2][i][j][k] = 0.0;
			njac[3][3][i][j][k] = c3c4 * tmp1;
			njac[3][4][i][j][k] = 0.0;
			njac[4][0][i][j][k] =
			    -(con43 * c3c4 -
			      c1345) * tmp3 * (((u_3[1] /*u[1][i][j][k] */ ) *
						(u_3[1] /*u[1][i][j][k] */ ))) -
			    (c3c4 -
			     c1345) * tmp3 * (((u_3[2] /*u[2][i][j][k] */ ) *
					       (u_3[2] /*u[2][i][j][k] */ ))) -
			    (c3c4 -
			     c1345) * tmp3 * (((u_3[3] /*u[3][i][j][k] */ ) *
					       (u_3[3] /*u[3][i][j][k] */ ))) -
			    c1345 * tmp2 * u[4][i][j][k];
			njac[4][1][i][j][k] =
			    (con43 * c3c4 -
			     c1345) * tmp2 * u_3[1] /*u[1][i][j][k] */ ;
			njac[4][2][i][j][k] =
			    (c3c4 - c1345) * tmp2 * u_3[2] /*u[2][i][j][k] */ ;
			njac[4][3][i][j][k] =
			    (c3c4 - c1345) * tmp2 * u_3[3] /*u[3][i][j][k] */ ;
			njac[4][4][i][j][k] = (c1345) * tmp1;
		}
		for (i = 1; i < grid_points[0] - 1; i++) {
			tmp1 = dt * tx1;
			tmp2 = dt * tx2;
			lhs[0][0][0][i][j][k] =
			    -tmp2 * fjac[0][0][i - 1][j][k] -
			    tmp1 * njac[0][0][i - 1][j][k] - tmp1 * dx1;
			lhs[0][0][1][i][j][k] =
			    -tmp2 * fjac[0][1][i - 1][j][k] -
			    tmp1 * njac[0][1][i - 1][j][k];
			lhs[0][0][2][i][j][k] =
			    -tmp2 * fjac[0][2][i - 1][j][k] -
			    tmp1 * njac[0][2][i - 1][j][k];
			lhs[0][0][3][i][j][k] =
			    -tmp2 * fjac[0][3][i - 1][j][k] -
			    tmp1 * njac[0][3][i - 1][j][k];
			lhs[0][0][4][i][j][k] =
			    -tmp2 * fjac[0][4][i - 1][j][k] -
			    tmp1 * njac[0][4][i - 1][j][k];
			lhs[0][1][0][i][j][k] =
			    -tmp2 * fjac[1][0][i - 1][j][k] -
			    tmp1 * njac[1][0][i - 1][j][k];
			lhs[0][1][1][i][j][k] =
			    -tmp2 * fjac[1][1][i - 1][j][k] -
			    tmp1 * njac[1][1][i - 1][j][k] - tmp1 * dx2;
			lhs[0][1][2][i][j][k] =
			    -tmp2 * fjac[1][2][i - 1][j][k] -
			    tmp1 * njac[1][2][i - 1][j][k];
			lhs[0][1][3][i][j][k] =
			    -tmp2 * fjac[1][3][i - 1][j][k] -
			    tmp1 * njac[1][3][i - 1][j][k];
			lhs[0][1][4][i][j][k] =
			    -tmp2 * fjac[1][4][i - 1][j][k] -
			    tmp1 * njac[1][4][i - 1][j][k];
			lhs[0][2][0][i][j][k] =
			    -tmp2 * fjac[2][0][i - 1][j][k] -
			    tmp1 * njac[2][0][i - 1][j][k];
			lhs[0][2][1][i][j][k] =
			    -tmp2 * fjac[2][1][i - 1][j][k] -
			    tmp1 * njac[2][1][i - 1][j][k];
			lhs[0][2][2][i][j][k] =
			    -tmp2 * fjac[2][2][i - 1][j][k] -
			    tmp1 * njac[2][2][i - 1][j][k] - tmp1 * dx3;
			lhs[0][2][3][i][j][k] =
			    -tmp2 * fjac[2][3][i - 1][j][k] -
			    tmp1 * njac[2][3][i - 1][j][k];
			lhs[0][2][4][i][j][k] =
			    -tmp2 * fjac[2][4][i - 1][j][k] -
			    tmp1 * njac[2][4][i - 1][j][k];
			lhs[0][3][0][i][j][k] =
			    -tmp2 * fjac[3][0][i - 1][j][k] -
			    tmp1 * njac[3][0][i - 1][j][k];
			lhs[0][3][1][i][j][k] =
			    -tmp2 * fjac[3][1][i - 1][j][k] -
			    tmp1 * njac[3][1][i - 1][j][k];
			lhs[0][3][2][i][j][k] =
			    -tmp2 * fjac[3][2][i - 1][j][k] -
			    tmp1 * njac[3][2][i - 1][j][k];
			lhs[0][3][3][i][j][k] =
			    -tmp2 * fjac[3][3][i - 1][j][k] -
			    tmp1 * njac[3][3][i - 1][j][k] - tmp1 * dx4;
			lhs[0][3][4][i][j][k] =
			    -tmp2 * fjac[3][4][i - 1][j][k] -
			    tmp1 * njac[3][4][i - 1][j][k];
			lhs[0][4][0][i][j][k] =
			    -tmp2 * fjac[4][0][i - 1][j][k] -
			    tmp1 * njac[4][0][i - 1][j][k];
			lhs[0][4][1][i][j][k] =
			    -tmp2 * fjac[4][1][i - 1][j][k] -
			    tmp1 * njac[4][1][i - 1][j][k];
			lhs[0][4][2][i][j][k] =
			    -tmp2 * fjac[4][2][i - 1][j][k] -
			    tmp1 * njac[4][2][i - 1][j][k];
			lhs[0][4][3][i][j][k] =
			    -tmp2 * fjac[4][3][i - 1][j][k] -
			    tmp1 * njac[4][3][i - 1][j][k];
			lhs[0][4][4][i][j][k] =
			    -tmp2 * fjac[4][4][i - 1][j][k] -
			    tmp1 * njac[4][4][i - 1][j][k] - tmp1 * dx5;
			lhs[1][0][0][i][j][k] =
			    1.0 + tmp1 * 2.0 * njac[0][0][i][j][k] +
			    tmp1 * 2.0 * dx1;
			lhs[1][0][1][i][j][k] =
			    tmp1 * 2.0 * njac[0][1][i][j][k];
			lhs[1][0][2][i][j][k] =
			    tmp1 * 2.0 * njac[0][2][i][j][k];
			lhs[1][0][3][i][j][k] =
			    tmp1 * 2.0 * njac[0][3][i][j][k];
			lhs[1][0][4][i][j][k] =
			    tmp1 * 2.0 * njac[0][4][i][j][k];
			lhs[1][1][0][i][j][k] =
			    tmp1 * 2.0 * njac[1][0][i][j][k];
			lhs[1][1][1][i][j][k] =
			    1.0 + tmp1 * 2.0 * njac[1][1][i][j][k] +
			    tmp1 * 2.0 * dx2;
			lhs[1][1][2][i][j][k] =
			    tmp1 * 2.0 * njac[1][2][i][j][k];
			lhs[1][1][3][i][j][k] =
			    tmp1 * 2.0 * njac[1][3][i][j][k];
			lhs[1][1][4][i][j][k] =
			    tmp1 * 2.0 * njac[1][4][i][j][k];
			lhs[1][2][0][i][j][k] =
			    tmp1 * 2.0 * njac[2][0][i][j][k];
			lhs[1][2][1][i][j][k] =
			    tmp1 * 2.0 * njac[2][1][i][j][k];
			lhs[1][2][2][i][j][k] =
			    1.0 + tmp1 * 2.0 * njac[2][2][i][j][k] +
			    tmp1 * 2.0 * dx3;
			lhs[1][2][3][i][j][k] =
			    tmp1 * 2.0 * njac[2][3][i][j][k];
			lhs[1][2][4][i][j][k] =
			    tmp1 * 2.0 * njac[2][4][i][j][k];
			lhs[1][3][0][i][j][k] =
			    tmp1 * 2.0 * njac[3][0][i][j][k];
			lhs[1][3][1][i][j][k] =
			    tmp1 * 2.0 * njac[3][1][i][j][k];
			lhs[1][3][2][i][j][k] =
			    tmp1 * 2.0 * njac[3][2][i][j][k];
			lhs[1][3][3][i][j][k] =
			    1.0 + tmp1 * 2.0 * njac[3][3][i][j][k] +
			    tmp1 * 2.0 * dx4;
			lhs[1][3][4][i][j][k] =
			    tmp1 * 2.0 * njac[3][4][i][j][k];
			lhs[1][4][0][i][j][k] =
			    tmp1 * 2.0 * njac[4][0][i][j][k];
			lhs[1][4][1][i][j][k] =
			    tmp1 * 2.0 * njac[4][1][i][j][k];
			lhs[1][4][2][i][j][k] =
			    tmp1 * 2.0 * njac[4][2][i][j][k];
			lhs[1][4][3][i][j][k] =
			    tmp1 * 2.0 * njac[4][3][i][j][k];
			lhs[1][4][4][i][j][k] =
			    1.0 + tmp1 * 2.0 * njac[4][4][i][j][k] +
			    tmp1 * 2.0 * dx5;
			lhs[2][0][0][i][j][k] =
			    tmp2 * fjac[0][0][i + 1][j][k] -
			    tmp1 * njac[0][0][i + 1][j][k] - tmp1 * dx1;
			lhs[2][0][1][i][j][k] =
			    tmp2 * fjac[0][1][i + 1][j][k] -
			    tmp1 * njac[0][1][i + 1][j][k];
			lhs[2][0][2][i][j][k] =
			    tmp2 * fjac[0][2][i + 1][j][k] -
			    tmp1 * njac[0][2][i + 1][j][k];
			lhs[2][0][3][i][j][k] =
			    tmp2 * fjac[0][3][i + 1][j][k] -
			    tmp1 * njac[0][3][i + 1][j][k];
			lhs[2][0][4][i][j][k] =
			    tmp2 * fjac[0][4][i + 1][j][k] -
			    tmp1 * njac[0][4][i + 1][j][k];
			lhs[2][1][0][i][j][k] =
			    tmp2 * fjac[1][0][i + 1][j][k] -
			    tmp1 * njac[1][0][i + 1][j][k];
			lhs[2][1][1][i][j][k] =
			    tmp2 * fjac[1][1][i + 1][j][k] -
			    tmp1 * njac[1][1][i + 1][j][k] - tmp1 * dx2;
			lhs[2][1][2][i][j][k] =
			    tmp2 * fjac[1][2][i + 1][j][k] -
			    tmp1 * njac[1][2][i + 1][j][k];
			lhs[2][1][3][i][j][k] =
			    tmp2 * fjac[1][3][i + 1][j][k] -
			    tmp1 * njac[1][3][i + 1][j][k];
			lhs[2][1][4][i][j][k] =
			    tmp2 * fjac[1][4][i + 1][j][k] -
			    tmp1 * njac[1][4][i + 1][j][k];
			lhs[2][2][0][i][j][k] =
			    tmp2 * fjac[2][0][i + 1][j][k] -
			    tmp1 * njac[2][0][i + 1][j][k];
			lhs[2][2][1][i][j][k] =
			    tmp2 * fjac[2][1][i + 1][j][k] -
			    tmp1 * njac[2][1][i + 1][j][k];
			lhs[2][2][2][i][j][k] =
			    tmp2 * fjac[2][2][i + 1][j][k] -
			    tmp1 * njac[2][2][i + 1][j][k] - tmp1 * dx3;
			lhs[2][2][3][i][j][k] =
			    tmp2 * fjac[2][3][i + 1][j][k] -
			    tmp1 * njac[2][3][i + 1][j][k];
			lhs[2][2][4][i][j][k] =
			    tmp2 * fjac[2][4][i + 1][j][k] -
			    tmp1 * njac[2][4][i + 1][j][k];
			lhs[2][3][0][i][j][k] =
			    tmp2 * fjac[3][0][i + 1][j][k] -
			    tmp1 * njac[3][0][i + 1][j][k];
			lhs[2][3][1][i][j][k] =
			    tmp2 * fjac[3][1][i + 1][j][k] -
			    tmp1 * njac[3][1][i + 1][j][k];
			lhs[2][3][2][i][j][k] =
			    tmp2 * fjac[3][2][i + 1][j][k] -
			    tmp1 * njac[3][2][i + 1][j][k];
			lhs[2][3][3][i][j][k] =
			    tmp2 * fjac[3][3][i + 1][j][k] -
			    tmp1 * njac[3][3][i + 1][j][k] - tmp1 * dx4;
			lhs[2][3][4][i][j][k] =
			    tmp2 * fjac[3][4][i + 1][j][k] -
			    tmp1 * njac[3][4][i + 1][j][k];
			lhs[2][4][0][i][j][k] =
			    tmp2 * fjac[4][0][i + 1][j][k] -
			    tmp1 * njac[4][0][i + 1][j][k];
			lhs[2][4][1][i][j][k] =
			    tmp2 * fjac[4][1][i + 1][j][k] -
			    tmp1 * njac[4][1][i + 1][j][k];
			lhs[2][4][2][i][j][k] =
			    tmp2 * fjac[4][2][i + 1][j][k] -
			    tmp1 * njac[4][2][i + 1][j][k];
			lhs[2][4][3][i][j][k] =
			    tmp2 * fjac[4][3][i + 1][j][k] -
			    tmp1 * njac[4][3][i + 1][j][k];
			lhs[2][4][4][i][j][k] =
			    tmp2 * fjac[4][4][i + 1][j][k] -
			    tmp1 * njac[4][4][i + 1][j][k] - tmp1 * dx5;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1256 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_0(__global double *g_u, __global double *g_fjac, double c2,
		     double c1, __global double *g_njac, double c3c4,
		     double con43, double c1345, int __ocl_k_bound,
		     int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1);
	int i = get_global_id(2) + 1;
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
	double tmp1;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp2;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp3;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*fjac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_fjac;
	__global double (*njac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_njac;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1258
		//-------------------------------------------
		double u_7[3];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1258
		//Candidates:
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_7[0] = u[1][i][j][k];
		u_7[1] = u[2][i][j][k];
		u_7[2] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u[0][i][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[0][0][i][j][k] = 0.0;
		fjac[0][1][i][j][k] = 0.0;
		fjac[0][2][i][j][k] = 1.0;
		fjac[0][3][i][j][k] = 0.0;
		fjac[0][4][i][j][k] = 0.0;
		fjac[1][0][i][j][k] =
		    -(u_7[0] /*u[1][i][j][k] */ *u_7[1] /*u[2][i][j][k] */ ) *
		    tmp2;
		fjac[1][1][i][j][k] = u_7[1] /*u[2][i][j][k] */ *tmp1;
		fjac[1][2][i][j][k] = u_7[0] /*u[1][i][j][k] */ *tmp1;
		fjac[1][3][i][j][k] = 0.0;
		fjac[1][4][i][j][k] = 0.0;
		fjac[2][0][i][j][k] =
		    -(u_7[1] /*u[2][i][j][k] */ *u_7[1] /*u[2][i][j][k] */
		      *tmp2) +
		    0.50 * c2 *
		    ((u_7[0] /*u[1][i][j][k] */ *u_7[0] /*u[1][i][j][k] */
		      +u_7[1] /*u[2][i][j][k] */ *u_7[1] /*u[2][i][j][k] */
		      +u_7[2] /*u[3][i][j][k] */ *u_7[2] /*u[3][i][j][k] */ ) *
		     tmp2);
		fjac[2][1][i][j][k] = -c2 * u_7[0] /*u[1][i][j][k] */ *tmp1;
		fjac[2][2][i][j][k] =
		    (2.0 - c2) * u_7[1] /*u[2][i][j][k] */ *tmp1;
		fjac[2][3][i][j][k] = -c2 * u_7[2] /*u[3][i][j][k] */ *tmp1;
		fjac[2][4][i][j][k] = c2;
		fjac[3][0][i][j][k] =
		    -(u_7[1] /*u[2][i][j][k] */ *u_7[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[3][1][i][j][k] = 0.0;
		fjac[3][2][i][j][k] = u_7[2] /*u[3][i][j][k] */ *tmp1;
		fjac[3][3][i][j][k] = u_7[1] /*u[2][i][j][k] */ *tmp1;
		fjac[3][4][i][j][k] = 0.0;
		fjac[4][0][i][j][k] =
		    (c2 *
		     (u_7[0] /*u[1][i][j][k] */ *u_7[0] /*u[1][i][j][k] */
		      +u_7[1] /*u[2][i][j][k] */ *u_7[1] /*u[2][i][j][k] */
		      +u_7[2] /*u[3][i][j][k] */ *u_7[2] /*u[3][i][j][k] */ ) *
		     tmp2 -
		     c1 * u[4][i][j][k] * tmp1) *
		    u_7[1] /*u[2][i][j][k] */ *tmp1;
		fjac[4][1][i][j][k] =
		    -c2 *
		    u_7[0] /*u[1][i][j][k] */ *u_7[1] /*u[2][i][j][k] */ *tmp2;
		fjac[4][2][i][j][k] =
		    c1 * u[4][i][j][k] * tmp1 -
		    0.50 * c2 *
		    ((u_7[0] /*u[1][i][j][k] */ *u_7[0] /*u[1][i][j][k] */ +3.0
		      *
		      u_7[1] /*u[2][i][j][k] */ *u_7[1] /*u[2][i][j][k] */
		      +u_7[2] /*u[3][i][j][k] */ *u_7[2] /*u[3][i][j][k] */ ) *
		     tmp2);
		fjac[4][3][i][j][k] =
		    -c2 *
		    (u_7[1] /*u[2][i][j][k] */ *u_7[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[4][4][i][j][k] = c1 * u_7[1] /*u[2][i][j][k] */ *tmp1;
		njac[0][0][i][j][k] = 0.0;
		njac[0][1][i][j][k] = 0.0;
		njac[0][2][i][j][k] = 0.0;
		njac[0][3][i][j][k] = 0.0;
		njac[0][4][i][j][k] = 0.0;
		njac[1][0][i][j][k] = -c3c4 * tmp2 * u_7[0] /*u[1][i][j][k] */ ;
		njac[1][1][i][j][k] = c3c4 * tmp1;
		njac[1][2][i][j][k] = 0.0;
		njac[1][3][i][j][k] = 0.0;
		njac[1][4][i][j][k] = 0.0;
		njac[2][0][i][j][k] =
		    -con43 * c3c4 * tmp2 * u_7[1] /*u[2][i][j][k] */ ;
		njac[2][1][i][j][k] = 0.0;
		njac[2][2][i][j][k] = con43 * c3c4 * tmp1;
		njac[2][3][i][j][k] = 0.0;
		njac[2][4][i][j][k] = 0.0;
		njac[3][0][i][j][k] = -c3c4 * tmp2 * u_7[2] /*u[3][i][j][k] */ ;
		njac[3][1][i][j][k] = 0.0;
		njac[3][2][i][j][k] = 0.0;
		njac[3][3][i][j][k] = c3c4 * tmp1;
		njac[3][4][i][j][k] = 0.0;
		njac[4][0][i][j][k] =
		    -(c3c4 -
		      c1345) * tmp3 * (((u_7[0] /*u[1][i][j][k] */ ) *
					(u_7[0] /*u[1][i][j][k] */ ))) -
		    (con43 * c3c4 -
		     c1345) * tmp3 * (((u_7[1] /*u[2][i][j][k] */ ) *
				       (u_7[1] /*u[2][i][j][k] */ ))) - (c3c4 -
									 c1345)
		    * tmp3 *
		    (((u_7[2] /*u[3][i][j][k] */ ) *
		      (u_7[2] /*u[3][i][j][k] */ ))) -
		    c1345 * tmp2 * u[4][i][j][k];
		njac[4][1][i][j][k] =
		    (c3c4 - c1345) * tmp2 * u_7[0] /*u[1][i][j][k] */ ;
		njac[4][2][i][j][k] =
		    (con43 * c3c4 - c1345) * tmp2 * u_7[1] /*u[2][i][j][k] */ ;
		njac[4][3][i][j][k] =
		    (c3c4 - c1345) * tmp2 * u_7[2] /*u[3][i][j][k] */ ;
		njac[4][4][i][j][k] = (c1345) * tmp1;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1360 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsy_1(double dt, double ty1, double ty2, __global double *g_lhs,
		     __global double *g_fjac, __global double *g_njac,
		     double dy1, double dy2, double dy3, double dy4, double dy5,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double tmp1;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp2;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*fjac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_fjac;
	__global double (*njac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_njac;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		tmp1 = dt * ty1;
		tmp2 = dt * ty2;
		lhs[0][0][0][i][j][k] =
		    -tmp2 * fjac[0][0][i][j - 1][k] - tmp1 * njac[0][0][i][j -
									   1][k]
		    - tmp1 * dy1;
		lhs[0][0][1][i][j][k] =
		    -tmp2 * fjac[0][1][i][j - 1][k] - tmp1 * njac[0][1][i][j -
									   1]
		    [k];
		lhs[0][0][2][i][j][k] =
		    -tmp2 * fjac[0][2][i][j - 1][k] - tmp1 * njac[0][2][i][j -
									   1]
		    [k];
		lhs[0][0][3][i][j][k] =
		    -tmp2 * fjac[0][3][i][j - 1][k] - tmp1 * njac[0][3][i][j -
									   1]
		    [k];
		lhs[0][0][4][i][j][k] =
		    -tmp2 * fjac[0][4][i][j - 1][k] - tmp1 * njac[0][4][i][j -
									   1]
		    [k];
		lhs[0][1][0][i][j][k] =
		    -tmp2 * fjac[1][0][i][j - 1][k] - tmp1 * njac[1][0][i][j -
									   1]
		    [k];
		lhs[0][1][1][i][j][k] =
		    -tmp2 * fjac[1][1][i][j - 1][k] - tmp1 * njac[1][1][i][j -
									   1][k]
		    - tmp1 * dy2;
		lhs[0][1][2][i][j][k] =
		    -tmp2 * fjac[1][2][i][j - 1][k] - tmp1 * njac[1][2][i][j -
									   1]
		    [k];
		lhs[0][1][3][i][j][k] =
		    -tmp2 * fjac[1][3][i][j - 1][k] - tmp1 * njac[1][3][i][j -
									   1]
		    [k];
		lhs[0][1][4][i][j][k] =
		    -tmp2 * fjac[1][4][i][j - 1][k] - tmp1 * njac[1][4][i][j -
									   1]
		    [k];
		lhs[0][2][0][i][j][k] =
		    -tmp2 * fjac[2][0][i][j - 1][k] - tmp1 * njac[2][0][i][j -
									   1]
		    [k];
		lhs[0][2][1][i][j][k] =
		    -tmp2 * fjac[2][1][i][j - 1][k] - tmp1 * njac[2][1][i][j -
									   1]
		    [k];
		lhs[0][2][2][i][j][k] =
		    -tmp2 * fjac[2][2][i][j - 1][k] - tmp1 * njac[2][2][i][j -
									   1][k]
		    - tmp1 * dy3;
		lhs[0][2][3][i][j][k] =
		    -tmp2 * fjac[2][3][i][j - 1][k] - tmp1 * njac[2][3][i][j -
									   1]
		    [k];
		lhs[0][2][4][i][j][k] =
		    -tmp2 * fjac[2][4][i][j - 1][k] - tmp1 * njac[2][4][i][j -
									   1]
		    [k];
		lhs[0][3][0][i][j][k] =
		    -tmp2 * fjac[3][0][i][j - 1][k] - tmp1 * njac[3][0][i][j -
									   1]
		    [k];
		lhs[0][3][1][i][j][k] =
		    -tmp2 * fjac[3][1][i][j - 1][k] - tmp1 * njac[3][1][i][j -
									   1]
		    [k];
		lhs[0][3][2][i][j][k] =
		    -tmp2 * fjac[3][2][i][j - 1][k] - tmp1 * njac[3][2][i][j -
									   1]
		    [k];
		lhs[0][3][3][i][j][k] =
		    -tmp2 * fjac[3][3][i][j - 1][k] - tmp1 * njac[3][3][i][j -
									   1][k]
		    - tmp1 * dy4;
		lhs[0][3][4][i][j][k] =
		    -tmp2 * fjac[3][4][i][j - 1][k] - tmp1 * njac[3][4][i][j -
									   1]
		    [k];
		lhs[0][4][0][i][j][k] =
		    -tmp2 * fjac[4][0][i][j - 1][k] - tmp1 * njac[4][0][i][j -
									   1]
		    [k];
		lhs[0][4][1][i][j][k] =
		    -tmp2 * fjac[4][1][i][j - 1][k] - tmp1 * njac[4][1][i][j -
									   1]
		    [k];
		lhs[0][4][2][i][j][k] =
		    -tmp2 * fjac[4][2][i][j - 1][k] - tmp1 * njac[4][2][i][j -
									   1]
		    [k];
		lhs[0][4][3][i][j][k] =
		    -tmp2 * fjac[4][3][i][j - 1][k] - tmp1 * njac[4][3][i][j -
									   1]
		    [k];
		lhs[0][4][4][i][j][k] =
		    -tmp2 * fjac[4][4][i][j - 1][k] - tmp1 * njac[4][4][i][j -
									   1][k]
		    - tmp1 * dy5;
		lhs[1][0][0][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[0][0][i][j][k] + tmp1 * 2.0 * dy1;
		lhs[1][0][1][i][j][k] = tmp1 * 2.0 * njac[0][1][i][j][k];
		lhs[1][0][2][i][j][k] = tmp1 * 2.0 * njac[0][2][i][j][k];
		lhs[1][0][3][i][j][k] = tmp1 * 2.0 * njac[0][3][i][j][k];
		lhs[1][0][4][i][j][k] = tmp1 * 2.0 * njac[0][4][i][j][k];
		lhs[1][1][0][i][j][k] = tmp1 * 2.0 * njac[1][0][i][j][k];
		lhs[1][1][1][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[1][1][i][j][k] + tmp1 * 2.0 * dy2;
		lhs[1][1][2][i][j][k] = tmp1 * 2.0 * njac[1][2][i][j][k];
		lhs[1][1][3][i][j][k] = tmp1 * 2.0 * njac[1][3][i][j][k];
		lhs[1][1][4][i][j][k] = tmp1 * 2.0 * njac[1][4][i][j][k];
		lhs[1][2][0][i][j][k] = tmp1 * 2.0 * njac[2][0][i][j][k];
		lhs[1][2][1][i][j][k] = tmp1 * 2.0 * njac[2][1][i][j][k];
		lhs[1][2][2][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[2][2][i][j][k] + tmp1 * 2.0 * dy3;
		lhs[1][2][3][i][j][k] = tmp1 * 2.0 * njac[2][3][i][j][k];
		lhs[1][2][4][i][j][k] = tmp1 * 2.0 * njac[2][4][i][j][k];
		lhs[1][3][0][i][j][k] = tmp1 * 2.0 * njac[3][0][i][j][k];
		lhs[1][3][1][i][j][k] = tmp1 * 2.0 * njac[3][1][i][j][k];
		lhs[1][3][2][i][j][k] = tmp1 * 2.0 * njac[3][2][i][j][k];
		lhs[1][3][3][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[3][3][i][j][k] + tmp1 * 2.0 * dy4;
		lhs[1][3][4][i][j][k] = tmp1 * 2.0 * njac[3][4][i][j][k];
		lhs[1][4][0][i][j][k] = tmp1 * 2.0 * njac[4][0][i][j][k];
		lhs[1][4][1][i][j][k] = tmp1 * 2.0 * njac[4][1][i][j][k];
		lhs[1][4][2][i][j][k] = tmp1 * 2.0 * njac[4][2][i][j][k];
		lhs[1][4][3][i][j][k] = tmp1 * 2.0 * njac[4][3][i][j][k];
		lhs[1][4][4][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[4][4][i][j][k] + tmp1 * 2.0 * dy5;
		lhs[2][0][0][i][j][k] =
		    tmp2 * fjac[0][0][i][j + 1][k] - tmp1 * njac[0][0][i][j +
									  1][k]
		    - tmp1 * dy1;
		lhs[2][0][1][i][j][k] =
		    tmp2 * fjac[0][1][i][j + 1][k] - tmp1 * njac[0][1][i][j +
									  1][k];
		lhs[2][0][2][i][j][k] =
		    tmp2 * fjac[0][2][i][j + 1][k] - tmp1 * njac[0][2][i][j +
									  1][k];
		lhs[2][0][3][i][j][k] =
		    tmp2 * fjac[0][3][i][j + 1][k] - tmp1 * njac[0][3][i][j +
									  1][k];
		lhs[2][0][4][i][j][k] =
		    tmp2 * fjac[0][4][i][j + 1][k] - tmp1 * njac[0][4][i][j +
									  1][k];
		lhs[2][1][0][i][j][k] =
		    tmp2 * fjac[1][0][i][j + 1][k] - tmp1 * njac[1][0][i][j +
									  1][k];
		lhs[2][1][1][i][j][k] =
		    tmp2 * fjac[1][1][i][j + 1][k] - tmp1 * njac[1][1][i][j +
									  1][k]
		    - tmp1 * dy2;
		lhs[2][1][2][i][j][k] =
		    tmp2 * fjac[1][2][i][j + 1][k] - tmp1 * njac[1][2][i][j +
									  1][k];
		lhs[2][1][3][i][j][k] =
		    tmp2 * fjac[1][3][i][j + 1][k] - tmp1 * njac[1][3][i][j +
									  1][k];
		lhs[2][1][4][i][j][k] =
		    tmp2 * fjac[1][4][i][j + 1][k] - tmp1 * njac[1][4][i][j +
									  1][k];
		lhs[2][2][0][i][j][k] =
		    tmp2 * fjac[2][0][i][j + 1][k] - tmp1 * njac[2][0][i][j +
									  1][k];
		lhs[2][2][1][i][j][k] =
		    tmp2 * fjac[2][1][i][j + 1][k] - tmp1 * njac[2][1][i][j +
									  1][k];
		lhs[2][2][2][i][j][k] =
		    tmp2 * fjac[2][2][i][j + 1][k] - tmp1 * njac[2][2][i][j +
									  1][k]
		    - tmp1 * dy3;
		lhs[2][2][3][i][j][k] =
		    tmp2 * fjac[2][3][i][j + 1][k] - tmp1 * njac[2][3][i][j +
									  1][k];
		lhs[2][2][4][i][j][k] =
		    tmp2 * fjac[2][4][i][j + 1][k] - tmp1 * njac[2][4][i][j +
									  1][k];
		lhs[2][3][0][i][j][k] =
		    tmp2 * fjac[3][0][i][j + 1][k] - tmp1 * njac[3][0][i][j +
									  1][k];
		lhs[2][3][1][i][j][k] =
		    tmp2 * fjac[3][1][i][j + 1][k] - tmp1 * njac[3][1][i][j +
									  1][k];
		lhs[2][3][2][i][j][k] =
		    tmp2 * fjac[3][2][i][j + 1][k] - tmp1 * njac[3][2][i][j +
									  1][k];
		lhs[2][3][3][i][j][k] =
		    tmp2 * fjac[3][3][i][j + 1][k] - tmp1 * njac[3][3][i][j +
									  1][k]
		    - tmp1 * dy4;
		lhs[2][3][4][i][j][k] =
		    tmp2 * fjac[3][4][i][j + 1][k] - tmp1 * njac[3][4][i][j +
									  1][k];
		lhs[2][4][0][i][j][k] =
		    tmp2 * fjac[4][0][i][j + 1][k] - tmp1 * njac[4][0][i][j +
									  1][k];
		lhs[2][4][1][i][j][k] =
		    tmp2 * fjac[4][1][i][j + 1][k] - tmp1 * njac[4][1][i][j +
									  1][k];
		lhs[2][4][2][i][j][k] =
		    tmp2 * fjac[4][2][i][j + 1][k] - tmp1 * njac[4][2][i][j +
									  1][k];
		lhs[2][4][3][i][j][k] =
		    tmp2 * fjac[4][3][i][j + 1][k] - tmp1 * njac[4][3][i][j +
									  1][k];
		lhs[2][4][4][i][j][k] =
		    tmp2 * fjac[4][4][i][j + 1][k] - tmp1 * njac[4][4][i][j +
									  1][k]
		    - tmp1 * dy5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1554 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_0(__global double *g_u, __global double *g_fjac, double c2,
		     double c1, __global double *g_njac, double c3c4,
		     double con43, double c3, double c4, double c1345,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0);
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double tmp1;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp2;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp3;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*fjac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_fjac;
	__global double (*njac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_njac;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1556
		//-------------------------------------------
		double u_11[3];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1556
		//Candidates:
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//      u[3][i][j][k]
		//-------------------------------------------
		u_11[0] = u[1][i][j][k];
		u_11[1] = u[2][i][j][k];
		u_11[2] = u[3][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u[0][i][j][k];
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[0][0][i][j][k] = 0.0;
		fjac[0][1][i][j][k] = 0.0;
		fjac[0][2][i][j][k] = 0.0;
		fjac[0][3][i][j][k] = 1.0;
		fjac[0][4][i][j][k] = 0.0;
		fjac[1][0][i][j][k] =
		    -(u_11[0] /*u[1][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[1][1][i][j][k] = u_11[2] /*u[3][i][j][k] */ *tmp1;
		fjac[1][2][i][j][k] = 0.0;
		fjac[1][3][i][j][k] = u_11[0] /*u[1][i][j][k] */ *tmp1;
		fjac[1][4][i][j][k] = 0.0;
		fjac[2][0][i][j][k] =
		    -(u_11[1] /*u[2][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[2][1][i][j][k] = 0.0;
		fjac[2][2][i][j][k] = u_11[2] /*u[3][i][j][k] */ *tmp1;
		fjac[2][3][i][j][k] = u_11[1] /*u[2][i][j][k] */ *tmp1;
		fjac[2][4][i][j][k] = 0.0;
		fjac[3][0][i][j][k] =
		    -(u_11[2] /*u[3][i][j][k] */ *u_11[2] /*u[3][i][j][k] */
		      *tmp2) +
		    0.50 * c2 *
		    ((u_11[0] /*u[1][i][j][k] */ *u_11[0] /*u[1][i][j][k] */
		      +u_11[1] /*u[2][i][j][k] */ *u_11[1] /*u[2][i][j][k] */
		      +u_11[2] /*u[3][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ )
		     * tmp2);
		fjac[3][1][i][j][k] = -c2 * u_11[0] /*u[1][i][j][k] */ *tmp1;
		fjac[3][2][i][j][k] = -c2 * u_11[1] /*u[2][i][j][k] */ *tmp1;
		fjac[3][3][i][j][k] =
		    (2.0 - c2) * u_11[2] /*u[3][i][j][k] */ *tmp1;
		fjac[3][4][i][j][k] = c2;
		fjac[4][0][i][j][k] =
		    (c2 *
		     (u_11[0] /*u[1][i][j][k] */ *u_11[0] /*u[1][i][j][k] */
		      +u_11[1] /*u[2][i][j][k] */ *u_11[1] /*u[2][i][j][k] */
		      +u_11[2] /*u[3][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ )
		     * tmp2 -
		     c1 * (u[4][i][j][k] * tmp1)) *
		    (u_11[2] /*u[3][i][j][k] */ *tmp1);
		fjac[4][1][i][j][k] =
		    -c2 *
		    (u_11[0] /*u[1][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[4][2][i][j][k] =
		    -c2 *
		    (u_11[1] /*u[2][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ ) *
		    tmp2;
		fjac[4][3][i][j][k] =
		    c1 * (u[4][i][j][k] * tmp1) -
		    0.50 * c2 *
		    ((u_11[0] /*u[1][i][j][k] */ *u_11[0] /*u[1][i][j][k] */
		      +u_11[1] /*u[2][i][j][k] */ *u_11[1] /*u[2][i][j][k] */
		      +3.0 *
		      u_11[2] /*u[3][i][j][k] */ *u_11[2] /*u[3][i][j][k] */ ) *
		     tmp2);
		fjac[4][4][i][j][k] = c1 * u_11[2] /*u[3][i][j][k] */ *tmp1;
		njac[0][0][i][j][k] = 0.0;
		njac[0][1][i][j][k] = 0.0;
		njac[0][2][i][j][k] = 0.0;
		njac[0][3][i][j][k] = 0.0;
		njac[0][4][i][j][k] = 0.0;
		njac[1][0][i][j][k] =
		    -c3c4 * tmp2 * u_11[0] /*u[1][i][j][k] */ ;
		njac[1][1][i][j][k] = c3c4 * tmp1;
		njac[1][2][i][j][k] = 0.0;
		njac[1][3][i][j][k] = 0.0;
		njac[1][4][i][j][k] = 0.0;
		njac[2][0][i][j][k] =
		    -c3c4 * tmp2 * u_11[1] /*u[2][i][j][k] */ ;
		njac[2][1][i][j][k] = 0.0;
		njac[2][2][i][j][k] = c3c4 * tmp1;
		njac[2][3][i][j][k] = 0.0;
		njac[2][4][i][j][k] = 0.0;
		njac[3][0][i][j][k] =
		    -con43 * c3c4 * tmp2 * u_11[2] /*u[3][i][j][k] */ ;
		njac[3][1][i][j][k] = 0.0;
		njac[3][2][i][j][k] = 0.0;
		njac[3][3][i][j][k] = con43 * c3 * c4 * tmp1;
		njac[3][4][i][j][k] = 0.0;
		njac[4][0][i][j][k] =
		    -(c3c4 -
		      c1345) * tmp3 * (((u_11[0] /*u[1][i][j][k] */ ) *
					(u_11[0] /*u[1][i][j][k] */ ))) -
		    (c3c4 -
		     c1345) * tmp3 * (((u_11[1] /*u[2][i][j][k] */ ) *
				       (u_11[1] /*u[2][i][j][k] */ ))) -
		    (con43 * c3c4 -
		     c1345) * tmp3 * (((u_11[2] /*u[3][i][j][k] */ ) *
				       (u_11[2] /*u[3][i][j][k] */ ))) -
		    c1345 * tmp2 * u[4][i][j][k];
		njac[4][1][i][j][k] =
		    (c3c4 - c1345) * tmp2 * u_11[0] /*u[1][i][j][k] */ ;
		njac[4][2][i][j][k] =
		    (c3c4 - c1345) * tmp2 * u_11[1] /*u[2][i][j][k] */ ;
		njac[4][3][i][j][k] =
		    (con43 * c3c4 - c1345) * tmp2 * u_11[2] /*u[3][i][j][k] */ ;
		njac[4][4][i][j][k] = (c1345) * tmp1;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1658 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void lhsz_1(double dt, double tz1, double tz2, __global double *g_lhs,
		     __global double *g_fjac, __global double *g_njac,
		     double dz1, double dz2, double dz3, double dz4, double dz5,
		     int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound,
		     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double tmp1;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	double tmp2;		/* (User-defined privated variables) : Defined at ./header.h : 88 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*fjac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_fjac;
	__global double (*njac)[5][65][65][64] =
	    (__global double (*)[5][65][65][64])g_njac;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1660
		//-------------------------------------------
		double2 njac_3[2];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1660
		//Candidates:
		//      njac[2][2][i][j][k - 1]
		//      njac[2][2][i][j][k]
		//      njac[0][3][i][j][k - 1]
		//      njac[0][3][i][j][k]
		//-------------------------------------------
		__global double *p_njac_3_0 =
		    (__global double *)&njac[2][2][i][j][k - 1];
		if ((unsigned long)p_njac_3_0 % 64 == 0) {
			njac_3[0] = vload2(0, p_njac_3_0);
		} else {
			njac_3[0].x = p_njac_3_0[0];
			p_njac_3_0++;
			njac_3[0].y = p_njac_3_0[0];
			p_njac_3_0++;
		}
		__global double *p_njac_3_1 =
		    (__global double *)&njac[0][3][i][j][k - 1];
		if ((unsigned long)p_njac_3_1 % 64 == 0) {
			njac_3[1] = vload2(0, p_njac_3_1);
		} else {
			njac_3[1].x = p_njac_3_1[0];
			p_njac_3_1++;
			njac_3[1].y = p_njac_3_1[0];
			p_njac_3_1++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = dt * tz1;
		tmp2 = dt * tz2;
		lhs[0][0][0][i][j][k] =
		    -tmp2 * fjac[0][0][i][j][k - 1] -
		    tmp1 * njac[0][0][i][j][k - 1] - tmp1 * dz1;
		lhs[0][0][1][i][j][k] =
		    -tmp2 * fjac[0][1][i][j][k - 1] -
		    tmp1 * njac[0][1][i][j][k - 1];
		lhs[0][0][2][i][j][k] =
		    -tmp2 * fjac[0][2][i][j][k - 1] -
		    tmp1 * njac[0][2][i][j][k - 1];
		lhs[0][0][3][i][j][k] =
		    -tmp2 * fjac[0][3][i][j][k - 1] -
		    tmp1 * njac_3[1].x /*njac[0][3][i][j][k - 1] */ ;
		lhs[0][0][4][i][j][k] =
		    -tmp2 * fjac[0][4][i][j][k - 1] -
		    tmp1 * njac[0][4][i][j][k - 1];
		lhs[0][1][0][i][j][k] =
		    -tmp2 * fjac[1][0][i][j][k - 1] -
		    tmp1 * njac[1][0][i][j][k - 1];
		lhs[0][1][1][i][j][k] =
		    -tmp2 * fjac[1][1][i][j][k - 1] -
		    tmp1 * njac[1][1][i][j][k - 1] - tmp1 * dz2;
		lhs[0][1][2][i][j][k] =
		    -tmp2 * fjac[1][2][i][j][k - 1] -
		    tmp1 * njac[1][2][i][j][k - 1];
		lhs[0][1][3][i][j][k] =
		    -tmp2 * fjac[1][3][i][j][k - 1] -
		    tmp1 * njac[1][3][i][j][k - 1];
		lhs[0][1][4][i][j][k] =
		    -tmp2 * fjac[1][4][i][j][k - 1] -
		    tmp1 * njac[1][4][i][j][k - 1];
		lhs[0][2][0][i][j][k] =
		    -tmp2 * fjac[2][0][i][j][k - 1] -
		    tmp1 * njac[2][0][i][j][k - 1];
		lhs[0][2][1][i][j][k] =
		    -tmp2 * fjac[2][1][i][j][k - 1] -
		    tmp1 * njac[2][1][i][j][k - 1];
		lhs[0][2][2][i][j][k] =
		    -tmp2 * fjac[2][2][i][j][k - 1] -
		    tmp1 * njac_3[0].x /*njac[2][2][i][j][k - 1] */  -
		    tmp1 * dz3;
		lhs[0][2][3][i][j][k] =
		    -tmp2 * fjac[2][3][i][j][k - 1] -
		    tmp1 * njac[2][3][i][j][k - 1];
		lhs[0][2][4][i][j][k] =
		    -tmp2 * fjac[2][4][i][j][k - 1] -
		    tmp1 * njac[2][4][i][j][k - 1];
		lhs[0][3][0][i][j][k] =
		    -tmp2 * fjac[3][0][i][j][k - 1] -
		    tmp1 * njac[3][0][i][j][k - 1];
		lhs[0][3][1][i][j][k] =
		    -tmp2 * fjac[3][1][i][j][k - 1] -
		    tmp1 * njac[3][1][i][j][k - 1];
		lhs[0][3][2][i][j][k] =
		    -tmp2 * fjac[3][2][i][j][k - 1] -
		    tmp1 * njac[3][2][i][j][k - 1];
		lhs[0][3][3][i][j][k] =
		    -tmp2 * fjac[3][3][i][j][k - 1] -
		    tmp1 * njac[3][3][i][j][k - 1] - tmp1 * dz4;
		lhs[0][3][4][i][j][k] =
		    -tmp2 * fjac[3][4][i][j][k - 1] -
		    tmp1 * njac[3][4][i][j][k - 1];
		lhs[0][4][0][i][j][k] =
		    -tmp2 * fjac[4][0][i][j][k - 1] -
		    tmp1 * njac[4][0][i][j][k - 1];
		lhs[0][4][1][i][j][k] =
		    -tmp2 * fjac[4][1][i][j][k - 1] -
		    tmp1 * njac[4][1][i][j][k - 1];
		lhs[0][4][2][i][j][k] =
		    -tmp2 * fjac[4][2][i][j][k - 1] -
		    tmp1 * njac[4][2][i][j][k - 1];
		lhs[0][4][3][i][j][k] =
		    -tmp2 * fjac[4][3][i][j][k - 1] -
		    tmp1 * njac[4][3][i][j][k - 1];
		lhs[0][4][4][i][j][k] =
		    -tmp2 * fjac[4][4][i][j][k - 1] -
		    tmp1 * njac[4][4][i][j][k - 1] - tmp1 * dz5;
		lhs[1][0][0][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[0][0][i][j][k] + tmp1 * 2.0 * dz1;
		lhs[1][0][1][i][j][k] = tmp1 * 2.0 * njac[0][1][i][j][k];
		lhs[1][0][2][i][j][k] = tmp1 * 2.0 * njac[0][2][i][j][k];
		lhs[1][0][3][i][j][k] =
		    tmp1 * 2.0 * njac_3[1].y /*njac[0][3][i][j][k] */ ;
		lhs[1][0][4][i][j][k] = tmp1 * 2.0 * njac[0][4][i][j][k];
		lhs[1][1][0][i][j][k] = tmp1 * 2.0 * njac[1][0][i][j][k];
		lhs[1][1][1][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[1][1][i][j][k] + tmp1 * 2.0 * dz2;
		lhs[1][1][2][i][j][k] = tmp1 * 2.0 * njac[1][2][i][j][k];
		lhs[1][1][3][i][j][k] = tmp1 * 2.0 * njac[1][3][i][j][k];
		lhs[1][1][4][i][j][k] = tmp1 * 2.0 * njac[1][4][i][j][k];
		lhs[1][2][0][i][j][k] = tmp1 * 2.0 * njac[2][0][i][j][k];
		lhs[1][2][1][i][j][k] = tmp1 * 2.0 * njac[2][1][i][j][k];
		lhs[1][2][2][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac_3[0].y /*njac[2][2][i][j][k] */  +
		    tmp1 * 2.0 * dz3;
		lhs[1][2][3][i][j][k] = tmp1 * 2.0 * njac[2][3][i][j][k];
		lhs[1][2][4][i][j][k] = tmp1 * 2.0 * njac[2][4][i][j][k];
		lhs[1][3][0][i][j][k] = tmp1 * 2.0 * njac[3][0][i][j][k];
		lhs[1][3][1][i][j][k] = tmp1 * 2.0 * njac[3][1][i][j][k];
		lhs[1][3][2][i][j][k] = tmp1 * 2.0 * njac[3][2][i][j][k];
		lhs[1][3][3][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[3][3][i][j][k] + tmp1 * 2.0 * dz4;
		lhs[1][3][4][i][j][k] = tmp1 * 2.0 * njac[3][4][i][j][k];
		lhs[1][4][0][i][j][k] = tmp1 * 2.0 * njac[4][0][i][j][k];
		lhs[1][4][1][i][j][k] = tmp1 * 2.0 * njac[4][1][i][j][k];
		lhs[1][4][2][i][j][k] = tmp1 * 2.0 * njac[4][2][i][j][k];
		lhs[1][4][3][i][j][k] = tmp1 * 2.0 * njac[4][3][i][j][k];
		lhs[1][4][4][i][j][k] =
		    1.0 + tmp1 * 2.0 * njac[4][4][i][j][k] + tmp1 * 2.0 * dz5;
		lhs[2][0][0][i][j][k] =
		    tmp2 * fjac[0][0][i][j][k + 1] - tmp1 * njac[0][0][i][j][k +
									     1]
		    - tmp1 * dz1;
		lhs[2][0][1][i][j][k] =
		    tmp2 * fjac[0][1][i][j][k + 1] - tmp1 * njac[0][1][i][j][k +
									     1];
		lhs[2][0][2][i][j][k] =
		    tmp2 * fjac[0][2][i][j][k + 1] - tmp1 * njac[0][2][i][j][k +
									     1];
		lhs[2][0][3][i][j][k] =
		    tmp2 * fjac[0][3][i][j][k + 1] - tmp1 * njac[0][3][i][j][k +
									     1];
		lhs[2][0][4][i][j][k] =
		    tmp2 * fjac[0][4][i][j][k + 1] - tmp1 * njac[0][4][i][j][k +
									     1];
		lhs[2][1][0][i][j][k] =
		    tmp2 * fjac[1][0][i][j][k + 1] - tmp1 * njac[1][0][i][j][k +
									     1];
		lhs[2][1][1][i][j][k] =
		    tmp2 * fjac[1][1][i][j][k + 1] - tmp1 * njac[1][1][i][j][k +
									     1]
		    - tmp1 * dz2;
		lhs[2][1][2][i][j][k] =
		    tmp2 * fjac[1][2][i][j][k + 1] - tmp1 * njac[1][2][i][j][k +
									     1];
		lhs[2][1][3][i][j][k] =
		    tmp2 * fjac[1][3][i][j][k + 1] - tmp1 * njac[1][3][i][j][k +
									     1];
		lhs[2][1][4][i][j][k] =
		    tmp2 * fjac[1][4][i][j][k + 1] - tmp1 * njac[1][4][i][j][k +
									     1];
		lhs[2][2][0][i][j][k] =
		    tmp2 * fjac[2][0][i][j][k + 1] - tmp1 * njac[2][0][i][j][k +
									     1];
		lhs[2][2][1][i][j][k] =
		    tmp2 * fjac[2][1][i][j][k + 1] - tmp1 * njac[2][1][i][j][k +
									     1];
		lhs[2][2][2][i][j][k] =
		    tmp2 * fjac[2][2][i][j][k + 1] - tmp1 * njac[2][2][i][j][k +
									     1]
		    - tmp1 * dz3;
		lhs[2][2][3][i][j][k] =
		    tmp2 * fjac[2][3][i][j][k + 1] - tmp1 * njac[2][3][i][j][k +
									     1];
		lhs[2][2][4][i][j][k] =
		    tmp2 * fjac[2][4][i][j][k + 1] - tmp1 * njac[2][4][i][j][k +
									     1];
		lhs[2][3][0][i][j][k] =
		    tmp2 * fjac[3][0][i][j][k + 1] - tmp1 * njac[3][0][i][j][k +
									     1];
		lhs[2][3][1][i][j][k] =
		    tmp2 * fjac[3][1][i][j][k + 1] - tmp1 * njac[3][1][i][j][k +
									     1];
		lhs[2][3][2][i][j][k] =
		    tmp2 * fjac[3][2][i][j][k + 1] - tmp1 * njac[3][2][i][j][k +
									     1];
		lhs[2][3][3][i][j][k] =
		    tmp2 * fjac[3][3][i][j][k + 1] - tmp1 * njac[3][3][i][j][k +
									     1]
		    - tmp1 * dz4;
		lhs[2][3][4][i][j][k] =
		    tmp2 * fjac[3][4][i][j][k + 1] - tmp1 * njac[3][4][i][j][k +
									     1];
		lhs[2][4][0][i][j][k] =
		    tmp2 * fjac[4][0][i][j][k + 1] - tmp1 * njac[4][0][i][j][k +
									     1];
		lhs[2][4][1][i][j][k] =
		    tmp2 * fjac[4][1][i][j][k + 1] - tmp1 * njac[4][1][i][j][k +
									     1];
		lhs[2][4][2][i][j][k] =
		    tmp2 * fjac[4][2][i][j][k + 1] - tmp1 * njac[4][2][i][j][k +
									     1];
		lhs[2][4][3][i][j][k] =
		    tmp2 * fjac[4][3][i][j][k + 1] - tmp1 * njac[4][3][i][j][k +
									     1];
		lhs[2][4][4][i][j][k] =
		    tmp2 * fjac[4][4][i][j][k + 1] - tmp1 * njac[4][4][i][j][k +
									     1]
		    - tmp1 * dz5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1845 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_0(__global double *g_u, __global double *g_rho_i,
			    __global double *g_us, __global double *g_vs,
			    __global double *g_ws, __global double *g_square,
			    __global double *g_qs, int __ocl_k_bound,
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
	double rho_inv;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*rho_i)[65][65] = (__global double (*)[65][65])g_rho_i;
	__global double (*us)[65][65] = (__global double (*)[65][65])g_us;
	__global double (*vs)[65][65] = (__global double (*)[65][65])g_vs;
	__global double (*ws)[65][65] = (__global double (*)[65][65])g_ws;
	__global double (*square)[65][65] =
	    (__global double (*)[65][65])g_square;
	__global double (*qs)[65][65] = (__global double (*)[65][65])g_qs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1847
		//-------------------------------------------
		double u_13[2];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1847
		//Candidates:
		//      u[1][i][j][k]
		//      u[2][i][j][k]
		//-------------------------------------------
		u_13[0] = u[1][i][j][k];
		u_13[1] = u[2][i][j][k];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rho_inv = 1.0 / u[0][i][j][k];
		rho_i[i][j][k] = rho_inv;
		us[i][j][k] = u_13[0] /*u[1][i][j][k] */ *rho_inv;
		vs[i][j][k] = u_13[1] /*u[2][i][j][k] */ *rho_inv;
		ws[i][j][k] = u[3][i][j][k] * rho_inv;
		square[i][j][k] =
		    0.5 *
		    (u_13[0] /*u[1][i][j][k] */ *u_13[0] /*u[1][i][j][k] */
		     +u_13[1] /*u[2][i][j][k] */ *u_13[1] /*u[2][i][j][k] */
		     +u[3][i][j][k] * u[3][i][j][k]) * rho_inv;
		qs[i][j][k] = square[i][j][k] * rho_inv;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1870 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_1(__global double *g_rhs, __global double *g_forcing,
			    int __ocl_k_bound, int __ocl_j_bound,
			    int __ocl_i_bound, __global int *tls_validflag)
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 1836 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*forcing)[65][65][65] =
	    (__global double (*)[65][65][65])g_forcing;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[m][i][j][k] = forcing[m][i][j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1886 of bt.c
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
			    int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double uijk;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double up1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double um1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*us)[65][65] = (__global double (*)[65][65])g_us;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*square)[65][65] =
	    (__global double (*)[65][65])g_square;
	__global double (*vs)[65][65] = (__global double (*)[65][65])g_vs;
	__global double (*ws)[65][65] = (__global double (*)[65][65])g_ws;
	__global double (*qs)[65][65] = (__global double (*)[65][65])g_qs;
	__global double (*rho_i)[65][65] = (__global double (*)[65][65])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1888
		//-------------------------------------------
		double u_15[8];
		double square_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1888
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
		u_15[0] = u[1][i - 1][j][k];
		u_15[1] = u[1][i + 1][j][k];
		u_15[2] = u[2][i - 1][j][k];
		u_15[3] = u[4][i - 1][j][k];
		u_15[4] = u[2][i + 1][j][k];
		u_15[5] = u[3][i - 1][j][k];
		u_15[6] = u[3][i + 1][j][k];
		u_15[7] = u[4][i][j][k];
		square_1 = square[i - 1][j][k];
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
		    (u_15[1] /*u[1][i + 1][j][k] */ -u_15[0]
		     /*u[1][i - 1][j][k] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] +
		    dx2tx1 * (u_15[1] /*u[1][i + 1][j][k] */ -2.0 *
			      u[1][i][j][k] + u_15[0] /*u[1][i - 1][j][k] */ ) +
		    xxcon2 * con43 * (up1 - 2.0 * uijk + um1) -
		    tx2 * (u_15[1] /*u[1][i + 1][j][k] */ *up1 -
			   u_15[0] /*u[1][i - 1][j][k] */ *um1 +
			   (u[4][i + 1][j][k] - square[i + 1][j][k] -
			    u_15[3] /*u[4][i - 1][j][k] */ +square_1
			    /*square[i - 1][j][k] */ ) * c2);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dx3tx1 * (u_15[4] /*u[2][i + 1][j][k] */ -2.0 *
			      u[2][i][j][k] + u_15[2] /*u[2][i - 1][j][k] */ ) +
		    xxcon2 * (vs[i + 1][j][k] - 2.0 * vs[i][j][k] +
			      vs[i - 1][j][k]) -
		    tx2 * (u_15[4] /*u[2][i + 1][j][k] */ *up1 -
			   u_15[2] /*u[2][i - 1][j][k] */ *um1);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dx4tx1 * (u_15[6] /*u[3][i + 1][j][k] */ -2.0 *
			      u[3][i][j][k] + u_15[5] /*u[3][i - 1][j][k] */ ) +
		    xxcon2 * (ws[i + 1][j][k] - 2.0 * ws[i][j][k] +
			      ws[i - 1][j][k]) -
		    tx2 * (u_15[6] /*u[3][i + 1][j][k] */ *up1 -
			   u_15[5] /*u[3][i - 1][j][k] */ *um1);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dx5tx1 * (u[4][i + 1][j][k] -
						2.0 *
						u_15[7] /*u[4][i][j][k] */
						+u_15[3] /*u[4][i - 1][j][k] */
						) + xxcon3 * (qs[i + 1][j][k] -
							      2.0 *
							      qs[i][j][k] +
							      qs[i - 1][j][k]) +
		    xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
		    xxcon5 * (u[4][i + 1][j][k] * rho_i[i + 1][j][k] -
			      2.0 * u_15[7] /*u[4][i][j][k] */ *rho_i[i][j][k] +
			      u_15[3] /*u[4][i - 1][j][k] */ *rho_i[i -
								    1][j][k]) -
		    tx2 * ((c1 * u[4][i + 1][j][k] - c2 * square[i + 1][j][k]) *
			   up1 -
			   (c1 * u_15[3] /*u[4][i - 1][j][k] */ -c2 *
			    square_1 /*square[i - 1][j][k] */ ) * um1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1949 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_3(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 1963 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_4(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 1976 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_5(__global double *g_rhs, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, int __ocl_i_bound,
			    __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 3;
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 1836 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 1993 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_6(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 2007 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_7(__global double *g_rhs, int i, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i - 2][j][k] -
					      4. * u[m][i - 1][j][k] +
					      5.0 * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2023 of bt.c
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
			    int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double vijk;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double vp1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double vm1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*vs)[65][65] = (__global double (*)[65][65])g_vs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*us)[65][65] = (__global double (*)[65][65])g_us;
	__global double (*square)[65][65] =
	    (__global double (*)[65][65])g_square;
	__global double (*ws)[65][65] = (__global double (*)[65][65])g_ws;
	__global double (*qs)[65][65] = (__global double (*)[65][65])g_qs;
	__global double (*rho_i)[65][65] = (__global double (*)[65][65])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2025
		//-------------------------------------------
		double u_17[8];
		double square_3;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2025
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
		u_17[0] = u[1][i][j - 1][k];
		u_17[1] = u[2][i][j - 1][k];
		u_17[2] = u[1][i][j + 1][k];
		u_17[3] = u[2][i][j + 1][k];
		u_17[4] = u[3][i][j - 1][k];
		u_17[5] = u[4][i][j - 1][k];
		u_17[6] = u[3][i][j + 1][k];
		u_17[7] = u[4][i][j][k];
		square_3 = square[i][j - 1][k];
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
		    (u_17[3] /*u[2][i][j + 1][k] */ -u_17[1]
		     /*u[2][i][j - 1][k] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] +
		    dy2ty1 * (u_17[2] /*u[1][i][j + 1][k] */ -2.0 *
			      u[1][i][j][k] + u_17[0] /*u[1][i][j - 1][k] */ ) +
		    yycon2 * (us[i][j + 1][k] - 2.0 * us[i][j][k] +
			      us[i][j - 1][k]) -
		    ty2 * (u_17[2] /*u[1][i][j + 1][k] */ *vp1 -
			   u_17[0] /*u[1][i][j - 1][k] */ *vm1);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dy3ty1 * (u_17[3] /*u[2][i][j + 1][k] */ -2.0 *
			      u[2][i][j][k] + u_17[1] /*u[2][i][j - 1][k] */ ) +
		    yycon2 * con43 * (vp1 - 2.0 * vijk + vm1) -
		    ty2 * (u_17[3] /*u[2][i][j + 1][k] */ *vp1 -
			   u_17[1] /*u[2][i][j - 1][k] */ *vm1 +
			   (u[4][i][j + 1][k] - square[i][j + 1][k] -
			    u_17[5] /*u[4][i][j - 1][k] */ +square_3
			    /*square[i][j - 1][k] */ ) * c2);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dy4ty1 * (u_17[6] /*u[3][i][j + 1][k] */ -2.0 *
			      u[3][i][j][k] + u_17[4] /*u[3][i][j - 1][k] */ ) +
		    yycon2 * (ws[i][j + 1][k] - 2.0 * ws[i][j][k] +
			      ws[i][j - 1][k]) -
		    ty2 * (u_17[6] /*u[3][i][j + 1][k] */ *vp1 -
			   u_17[4] /*u[3][i][j - 1][k] */ *vm1);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dy5ty1 * (u[4][i][j + 1][k] -
						2.0 *
						u_17[7] /*u[4][i][j][k] */
						+u_17[5] /*u[4][i][j - 1][k] */
						) + yycon3 * (qs[i][j + 1][k] -
							      2.0 *
							      qs[i][j][k] +
							      qs[i][j - 1][k]) +
		    yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
		    yycon5 * (u[4][i][j + 1][k] * rho_i[i][j + 1][k] -
			      2.0 * u_17[7] /*u[4][i][j][k] */ *rho_i[i][j][k] +
			      u_17[5] /*u[4][i][j - 1][k] */ *rho_i[i][j -
								       1][k]) -
		    ty2 * ((c1 * u[4][i][j + 1][k] - c2 * square[i][j + 1][k]) *
			   vp1 -
			   (c1 * u_17[5] /*u[4][i][j - 1][k] */ -c2 *
			    square_3 /*square[i][j - 1][k] */ ) * vm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2081 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_9(__global double *g_rhs, int j, double dssp,
			    __global double *g_u, int __ocl_k_bound,
			    int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 2095 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_10(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 2108 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_11(__global double *g_rhs, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_j_bound, int __ocl_i_bound,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 3;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 1836 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
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
//Loop defined at line 2125 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_12(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
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
//Loop defined at line 2139 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_13(__global double *g_rhs, int j, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u[m][i][j - 2][k] -
					      4. * u[m][i][j - 1][k] +
					      5. * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2155 of bt.c
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
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	double wijk;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double wp1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	double wm1;		/* (User-defined privated variables) : Defined at bt.c : 1837 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*ws)[65][65] = (__global double (*)[65][65])g_ws;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	__global double (*us)[65][65] = (__global double (*)[65][65])g_us;
	__global double (*vs)[65][65] = (__global double (*)[65][65])g_vs;
	__global double (*square)[65][65] =
	    (__global double (*)[65][65])g_square;
	__global double (*qs)[65][65] = (__global double (*)[65][65])g_qs;
	__global double (*rho_i)[65][65] = (__global double (*)[65][65])g_rho_i;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2157
		//-------------------------------------------
		double2 ws_1;
		double2 u_20[4];
		double u_21[3];
		double2 us_1;
		double2 vs_1;
		double square_5;
		double2 qs_1;
		double2 rho_i_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2157
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
		__global double *p_ws_1_0 = (__global double *)&ws[i][j][k - 1];
		if ((unsigned long)p_ws_1_0 % 64 == 0) {
			ws_1 = vload2(0, p_ws_1_0);
		} else {
			ws_1.x = p_ws_1_0[0];
			p_ws_1_0++;
			ws_1.y = p_ws_1_0[0];
			p_ws_1_0++;
		}
		__global double *p_u_20_0 =
		    (__global double *)&u[0][i][j][k - 1];
		if ((unsigned long)p_u_20_0 % 64 == 0) {
			u_20[0] = vload2(0, p_u_20_0);
		} else {
			u_20[0].x = p_u_20_0[0];
			p_u_20_0++;
			u_20[0].y = p_u_20_0[0];
			p_u_20_0++;
		}
		__global double *p_u_20_1 =
		    (__global double *)&u[2][i][j][k - 1];
		if ((unsigned long)p_u_20_1 % 64 == 0) {
			u_20[1] = vload2(0, p_u_20_1);
		} else {
			u_20[1].x = p_u_20_1[0];
			p_u_20_1++;
			u_20[1].y = p_u_20_1[0];
			p_u_20_1++;
		}
		__global double *p_u_20_2 = (__global double *)&u[3][i][j][k];
		if ((unsigned long)p_u_20_2 % 64 == 0) {
			u_20[2] = vload2(0, p_u_20_2);
		} else {
			u_20[2].x = p_u_20_2[0];
			p_u_20_2++;
			u_20[2].y = p_u_20_2[0];
			p_u_20_2++;
		}
		__global double *p_u_20_3 =
		    (__global double *)&u[4][i][j][k - 1];
		if ((unsigned long)p_u_20_3 % 64 == 0) {
			u_20[3] = vload2(0, p_u_20_3);
		} else {
			u_20[3].x = p_u_20_3[0];
			p_u_20_3++;
			u_20[3].y = p_u_20_3[0];
			p_u_20_3++;
		}
		u_21[0] = u[1][i][j][k - 1];
		u_21[1] = u[3][i][j][k - 1];
		u_21[2] = u[2][i][j][k + 1];
		__global double *p_us_1_0 = (__global double *)&us[i][j][k - 1];
		if ((unsigned long)p_us_1_0 % 64 == 0) {
			us_1 = vload2(0, p_us_1_0);
		} else {
			us_1.x = p_us_1_0[0];
			p_us_1_0++;
			us_1.y = p_us_1_0[0];
			p_us_1_0++;
		}
		__global double *p_vs_1_0 = (__global double *)&vs[i][j][k - 1];
		if ((unsigned long)p_vs_1_0 % 64 == 0) {
			vs_1 = vload2(0, p_vs_1_0);
		} else {
			vs_1.x = p_vs_1_0[0];
			p_vs_1_0++;
			vs_1.y = p_vs_1_0[0];
			p_vs_1_0++;
		}
		square_5 = square[i][j][k - 1];
		__global double *p_qs_1_0 = (__global double *)&qs[i][j][k - 1];
		if ((unsigned long)p_qs_1_0 % 64 == 0) {
			qs_1 = vload2(0, p_qs_1_0);
		} else {
			qs_1.x = p_qs_1_0[0];
			p_qs_1_0++;
			qs_1.y = p_qs_1_0[0];
			p_qs_1_0++;
		}
		__global double *p_rho_i_1_0 =
		    (__global double *)&rho_i[i][j][k - 1];
		if ((unsigned long)p_rho_i_1_0 % 64 == 0) {
			rho_i_1 = vload2(0, p_rho_i_1_0);
		} else {
			rho_i_1.x = p_rho_i_1_0[0];
			p_rho_i_1_0++;
			rho_i_1.y = p_rho_i_1_0[0];
			p_rho_i_1_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		wijk = ws_1.y /*ws[i][j][k] */ ;
		wp1 = ws[i][j][k + 1];
		wm1 = ws_1.x /*ws[i][j][k - 1] */ ;
		rhs[0][i][j][k] =
		    rhs[0][i][j][k] + dz1tz1 * (u[0][i][j][k + 1] -
						2.0 *
						u_20[0].y /*u[0][i][j][k] */  +
						u_20[0].
						x /*u[0][i][j][k - 1] */ ) -
		    tz2 * (u_20[2].y /*u[3][i][j][k + 1] */  -
			   u_21[1] /*u[3][i][j][k - 1] */ );
		rhs[1][i][j][k] =
		    rhs[1][i][j][k] + dz2tz1 * (u[1][i][j][k + 1] -
						2.0 * u[1][i][j][k] +
						u_21[0] /*u[1][i][j][k - 1] */ )
		    + zzcon2 * (us[i][j][k + 1] -
				2.0 * us_1.y /*us[i][j][k] */  +
				us_1.x /*us[i][j][k - 1] */ ) -
		    tz2 * (u[1][i][j][k + 1] * wp1 -
			   u_21[0] /*u[1][i][j][k - 1] */ *wm1);
		rhs[2][i][j][k] =
		    rhs[2][i][j][k] +
		    dz3tz1 * (u_21[2] /*u[2][i][j][k + 1] */ -2.0 *
			      u_20[1].y /*u[2][i][j][k] */  +
			      u_20[1].x /*u[2][i][j][k - 1] */ ) +
		    zzcon2 * (vs[i][j][k + 1] - 2.0 * vs_1.y /*vs[i][j][k] */  +
			      vs_1.x /*vs[i][j][k - 1] */ ) -
		    tz2 * (u_21[2] /*u[2][i][j][k + 1] */ *wp1 -
			   u_20[1].x /*u[2][i][j][k - 1] */  * wm1);
		rhs[3][i][j][k] =
		    rhs[3][i][j][k] +
		    dz4tz1 * (u_20[2].y /*u[3][i][j][k + 1] */  -
			      2.0 * u_20[2].x /*u[3][i][j][k] */  +
			      u_21[1] /*u[3][i][j][k - 1] */ ) +
		    zzcon2 * con43 * (wp1 - 2.0 * wijk + wm1) -
		    tz2 * (u_20[2].y /*u[3][i][j][k + 1] */  * wp1 -
			   u_21[1] /*u[3][i][j][k - 1] */ *wm1 +
			   (u[4][i][j][k + 1] - square[i][j][k + 1] -
			    u_20[3].x /*u[4][i][j][k - 1] */  +
			    square_5 /*square[i][j][k - 1] */ ) * c2);
		rhs[4][i][j][k] =
		    rhs[4][i][j][k] + dz5tz1 * (u[4][i][j][k + 1] -
						2.0 *
						u_20[3].y /*u[4][i][j][k] */  +
						u_20[3].
						x /*u[4][i][j][k - 1] */ ) +
		    zzcon3 * (qs[i][j][k + 1] - 2.0 * qs_1.y /*qs[i][j][k] */  +
			      qs_1.x /*qs[i][j][k - 1] */ ) +
		    zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
		    zzcon5 * (u[4][i][j][k + 1] * rho_i[i][j][k + 1] -
			      2.0 * u_20[3].y /*u[4][i][j][k] */  *
			      rho_i_1.y /*rho_i[i][j][k] */  +
			      u_20[3].x /*u[4][i][j][k - 1] */  *
			      rho_i_1.x /*rho_i[i][j][k - 1] */ ) -
		    tz2 * ((c1 * u[4][i][j][k + 1] - c2 * square[i][j][k + 1]) *
			   wp1 - (c1 * u_20[3].x /*u[4][i][j][k - 1] */  -
				  c2 * square_5 /*square[i][j][k - 1] */ ) *
			   wm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2220 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_15(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2222
		//-------------------------------------------
		double2 u_23;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2222
		//Candidates:
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//-------------------------------------------
		__global double *p_u_23_0 = (__global double *)&u[m][i][j][k];
		if ((unsigned long)p_u_23_0 % 64 == 0) {
			u_23 = vload2(0, p_u_23_0);
		} else {
			u_23.x = p_u_23_0[0];
			p_u_23_0++;
			u_23.y = p_u_23_0[0];
			p_u_23_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (5.0 * u_23.x /*u[m][i][j][k] */  -
					      4.0 *
					      u_23.y /*u[m][i][j][k + 1] */  +
					      u[m][i][j][k + 2]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2240 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_16(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2242
		//-------------------------------------------
		double4 u_25;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2242
		//Candidates:
		//      u[m][i][j][k - 1]
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//      u[m][i][j][k + 2]
		//-------------------------------------------
		__global double *p_u_25_0 =
		    (__global double *)&u[m][i][j][k - 1];
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

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] -
		    dssp * (-4.0 * u_25.x /*u[m][i][j][k - 1] */  +
			    6.0 * u_25.y /*u[m][i][j][k] */  -
			    4.0 * u_25.z /*u[m][i][j][k + 1] */  +
			    u_25.w /*u[m][i][j][k + 2] */ );
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2259 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_17(__global double *g_rhs, double dssp,
			     __global double *g_u, int __ocl_k_bound,
			     int __ocl_j_bound, int __ocl_i_bound,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 3;
	int j = get_global_id(1) + 1;
	int i = get_global_id(2) + 1;
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
	int m;			/* (User-defined privated variables) : Defined at bt.c : 1836 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 2262
			//-------------------------------------------
			double4 u_27;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 2262
			//Candidates:
			//      u[m][i][j][k - 2]
			//      u[m][i][j][k - 1]
			//      u[m][i][j][k]
			//      u[m][i][j][k + 1]
			//-------------------------------------------
			__global double *p_u_27_0 =
			    (__global double *)&u[m][i][j][k - 2];
			if ((unsigned long)p_u_27_0 % 64 == 0) {
				u_27 = vload4(0, p_u_27_0);
			} else {
				u_27.x = p_u_27_0[0];
				p_u_27_0++;
				u_27.y = p_u_27_0[0];
				p_u_27_0++;
				u_27.z = p_u_27_0[0];
				p_u_27_0++;
				u_27.w = p_u_27_0[0];
				p_u_27_0++;
			}
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			rhs[m][i][j][k] =
			    rhs[m][i][j][k] -
			    dssp * (u_27.x /*u[m][i][j][k - 2] */  -
				    4.0 * u_27.y /*u[m][i][j][k - 1] */  +
				    6.0 * u_27.z /*u[m][i][j][k] */  -
				    4.0 * u_27.w /*u[m][i][j][k + 1] */  +
				    u[m][i][j][k + 2]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2283 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_18(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2285
		//-------------------------------------------
		double4 u_29;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2285
		//Candidates:
		//      u[m][i][j][k - 2]
		//      u[m][i][j][k - 1]
		//      u[m][i][j][k]
		//      u[m][i][j][k + 1]
		//-------------------------------------------
		__global double *p_u_29_0 =
		    (__global double *)&u[m][i][j][k - 2];
		if ((unsigned long)p_u_29_0 % 64 == 0) {
			u_29 = vload4(0, p_u_29_0);
		} else {
			u_29.x = p_u_29_0[0];
			p_u_29_0++;
			u_29.y = p_u_29_0[0];
			p_u_29_0++;
			u_29.z = p_u_29_0[0];
			p_u_29_0++;
			u_29.w = p_u_29_0[0];
			p_u_29_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u_29.x /*u[m][i][j][k - 2] */  -
					      4.0 *
					      u_29.y /*u[m][i][j][k - 1] */  +
					      6.0 * u_29.z /*u[m][i][j][k] */  -
					      4.0 *
					      u_29.w /*u[m][i][j][k + 1] */ );
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2303 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_19(__global double *g_rhs, int k, double dssp,
			     __global double *g_u, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*u)[65][65][65] = (__global double (*)[65][65][65])g_u;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2305
		//-------------------------------------------
		double2 u_31;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2305
		//Candidates:
		//      u[m][i][j][k - 2]
		//      u[m][i][j][k - 1]
		//-------------------------------------------
		__global double *p_u_31_0 =
		    (__global double *)&u[m][i][j][k - 2];
		if ((unsigned long)p_u_31_0 % 64 == 0) {
			u_31 = vload2(0, p_u_31_0);
		} else {
			u_31.x = p_u_31_0[0];
			p_u_31_0++;
			u_31.y = p_u_31_0[0];
			p_u_31_0++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - dssp * (u_31.x /*u[m][i][j][k - 2] */  -
					      4.0 *
					      u_31.y /*u[m][i][j][k - 1] */  +
					      5.0 * u[m][i][j][k]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2327 of bt.c
//The nested loops were swaped. 
//GPU TLS Checking is disabled by the user. 
//-------------------------------------------------------------------------------
__kernel void compute_rhs_20(__global int *grid_points, __global double *g_rhs,
			     double dt, int __ocl_k_bound, int __ocl_j_bound,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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
	int i;			/* (User-defined privated variables) : Defined at bt.c : 1836 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = 1; i < grid_points[0] - 1; i++) {
			rhs[m][i][j][k] = rhs[m][i][j][k] * dt;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2853 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void x_backsubstitute_0(__global double *g_rhs, int i,
				 __global double *g_lhs, int __ocl_k_bound,
				 int __ocl_j_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(j < __ocl_j_bound)) {
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
	int n;			/* (User-defined privated variables) : Defined at bt.c : 2849 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (n = 0; n < 5; n++) {
			rhs[m][i][j][k] =
			    rhs[m][i][j][k] - lhs[2][m][n][i][j][k] * rhs[n][i +
									     1]
			    [j][k];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2891 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_0(__global double *g_lhs, __global double *g_rhs,
			     int __ocl_k_bound, int __ocl_j_bound,
			     __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		binvcrhs_g0_g5_g10(lhs, 0, j, k, 1, lhs, 0, j, k, 2, rhs, 0, j,
				   k, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2912 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_1(__global double *g_lhs, int i,
			     __global double *g_rhs, int __ocl_k_bound,
			     int __ocl_j_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, i, j, k, 0, rhs, i - 1, j, k, rhs, i,
				    j, k, rd_log_rhs, wr_log_rhs, tls_validflag,
				    tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, i, j, k, 0, lhs, i - 1, j, k, 2, lhs,
				     i, j, k, 1, rd_log_lhs, wr_log_lhs,
				     tls_validflag, tls_thread_id);
		binvcrhs_g0_g5_g10(lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j,
				   k, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2945 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_2(__global double *g_lhs, int isize,
			     __global double *g_rhs, int i, int __ocl_k_bound,
			     int __ocl_j_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, isize, j, k, 0, rhs, isize - 1, j, k,
				    rhs, isize, j, k, rd_log_rhs, wr_log_rhs,
				    tls_validflag, tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, isize, j, k, 0, lhs, isize - 1, j, k,
				     2, lhs, isize, j, k, 1, rd_log_lhs,
				     wr_log_lhs, tls_validflag, tls_thread_id);
		binvrhs_g0_g5(lhs, i, j, k, 1, rhs, i, j, k, rd_log_lhs,
			      wr_log_lhs, rd_log_rhs, wr_log_rhs, tls_validflag,
			      tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3497 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_backsubstitute_0(__global double *g_rhs, int j,
				 __global double *g_lhs, int __ocl_k_bound,
				 int __ocl_i_bound, __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(k < __ocl_k_bound)) {
		return;
	}
	if (!(i < __ocl_i_bound)) {
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
	int n;			/* (User-defined privated variables) : Defined at bt.c : 3492 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_3();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	for (n = 0; n < 5; n++) {
		rhs[m][i][j][k] =
		    rhs[m][i][j][k] - lhs[2][m][n][i][j][k] * rhs[n][i][j +
									1][k];
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3535 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_0(__global double *g_lhs, __global double *g_rhs,
			     int __ocl_k_bound, int __ocl_i_bound,
			     __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		binvcrhs_g0_g5_g10(lhs, i, 0, k, 1, lhs, i, 0, k, 2, rhs, i, 0,
				   k, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3556 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_1(__global double *g_lhs, int j,
			     __global double *g_rhs, int __ocl_k_bound,
			     int __ocl_i_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, i, j, k, 0, rhs, i, j - 1, k, rhs, i,
				    j, k, rd_log_rhs, wr_log_rhs, tls_validflag,
				    tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, i, j, k, 0, lhs, i, j - 1, k, 2, lhs,
				     i, j, k, 1, rd_log_lhs, wr_log_lhs,
				     tls_validflag, tls_thread_id);
		binvcrhs_g0_g5_g10(lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j,
				   k, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3589 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_2(__global double *g_lhs, int jsize,
			     __global double *g_rhs, int __ocl_k_bound,
			     int __ocl_i_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int k = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, i, jsize, k, 0, rhs, i, jsize - 1, k,
				    rhs, i, jsize, k, rd_log_rhs, wr_log_rhs,
				    tls_validflag, tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, i, jsize, k, 0, lhs, i, jsize - 1, k,
				     2, lhs, i, jsize, k, 1, rd_log_lhs,
				     wr_log_lhs, tls_validflag, tls_thread_id);
		binvrhs_g0_g5(lhs, i, jsize, k, 1, rhs, i, jsize, k, rd_log_lhs,
			      wr_log_lhs, rd_log_rhs, wr_log_rhs, tls_validflag,
			      tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3673 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_backsubstitute_0(__global int *grid_points,
				 __global double *g_rhs, __global double *g_lhs,
				 int __ocl_j_bound, int __ocl_i_bound,
				 __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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
	int k;			/* (User-defined privated variables) : Defined at bt.c : 3669 */
	int m;			/* (User-defined privated variables) : Defined at bt.c : 3669 */
	int n;			/* (User-defined privated variables) : Defined at bt.c : 3669 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (k = grid_points[2] - 2; k >= 0; k--) {
			for (m = 0; m < 5; m++) {
				for (n = 0; n < 5; n++) {
					rhs[m][i][j][k] =
					    rhs[m][i][j][k] -
					    lhs[2][m][n][i][j][k] *
					    rhs[n][i][j][k + 1];
				}
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3720 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_0(__global double *g_lhs, __global double *g_rhs,
			     int __ocl_j_bound, int __ocl_i_bound,
			     __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		binvcrhs_g0_g5_g10(lhs, i, j, 0, 1, lhs, i, j, 0, 2, rhs, i, j,
				   0, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3748 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_1(__global double *g_lhs, int k,
			     __global double *g_rhs, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, i, j, k, 0, rhs, i, j, k - 1, rhs, i,
				    j, k, rd_log_rhs, wr_log_rhs, tls_validflag,
				    tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, i, j, k, 0, lhs, i, j, k - 1, 2, lhs,
				     i, j, k, 1, rd_log_lhs, wr_log_lhs,
				     tls_validflag, tls_thread_id);
		binvcrhs_g0_g5_g10(lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j,
				   k, rd_log_lhs, wr_log_lhs, rd_log_lhs,
				   wr_log_lhs, rd_log_rhs, wr_log_rhs,
				   tls_validflag, tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3802 of bt.c
//The nested loops were swaped. 
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_2(__global double *g_lhs, int ksize,
			     __global double *g_rhs, int __ocl_j_bound,
			     int __ocl_i_bound, __global int *g_rd_log_lhs,
			     __global int *g_wr_log_lhs,
			     __global int *g_rd_log_rhs,
			     __global int *g_wr_log_rhs,
			     __global int *tls_validflag)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
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

	int tls_thread_id = calc_thread_id_2();
	//-------------------------------------------
	//Convert global memory objects (BEGIN)
	//-------------------------------------------
	__global double (*lhs)[5][5][65][65][65] =
	    (__global double (*)[5][5][65][65][65])g_lhs;
	__global double (*rhs)[65][65][65] =
	    (__global double (*)[65][65][65])g_rhs;

	//TLS Checking Buffers (BEGIN)
	__global int (*rd_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_rd_log_lhs;
	__global int (*wr_log_lhs)[5][5][65][65][65] =
	    (__global int (*)[5][5][65][65][65])g_wr_log_lhs;
	__global int (*rd_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_rd_log_rhs;
	__global int (*wr_log_rhs)[65][65][65] =
	    (__global int (*)[65][65][65])g_wr_log_rhs;
	//TLS Checking Buffers (END)

	//-------------------------------------------
	//Convert global memory objects (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_g0_g5_g9(lhs, i, j, ksize, 0, rhs, i, j, ksize - 1,
				    rhs, i, j, ksize, rd_log_rhs, wr_log_rhs,
				    tls_validflag, tls_thread_id);
		matmul_sub_g0_g5_g10(lhs, i, j, ksize, 0, lhs, i, j, ksize - 1,
				     2, lhs, i, j, ksize, 1, rd_log_lhs,
				     wr_log_lhs, tls_validflag, tls_thread_id);
		binvrhs_g0_g5(lhs, i, j, ksize, 1, rhs, i, j, ksize, rd_log_lhs,
			      wr_log_lhs, rd_log_rhs, wr_log_rhs, tls_validflag,
			      tls_thread_id);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
