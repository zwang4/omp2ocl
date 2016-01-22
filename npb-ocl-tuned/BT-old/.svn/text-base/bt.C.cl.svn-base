//====== OPENCL KERNEL START
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//These are some common routines
#define CALC_2D_ARRAY_INDEX(M1,M2,m1,m2) (((m1)*(M2))+((m2)))
#define CALC_3D_ARRAY_INDEX(M1,M2,M3,m1,m2,m3) (((m1)*(M2)*(M3))+((m2)*(M3))+((m3)))
#define CALC_4D_ARRAY_INDEX(M1,M2,M3,M4,m1,m2,m3,m4) (((m1)*(M2)*(M3)*(M4))+((m2)*(M3)*(M4))+((m3)*(M4))+((m4)))
#define CALC_5D_ARRAY_INDEX(M1,M2,M3,M4,M5,m1,m2,m3,m4,m5) (((m1)*(M2)*(M3)*(M4)*(M5))+((m2)*(M3)*(M4)*(M5))+((m3)*(M4)*(M5))+((m4)*(M5))+((m5)))
#define CALC_6D_ARRAY_INDEX(M1,M2,M3,M4,M5,M6,m1,m2,m3,m4,m5,m6) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6))+((m2)*(M3)*(M4)*(M5)*(M6))+((m3)*(M4)*(M5)*(M6))+((m4)*(M5)*(M6))+((m5)*(M6))+((m6)))
#define CALC_7D_ARRAY_INDEX(M1,M2,M3,M4,M5,M6,M7,m1,m2,m3,m4,m5,m6,m7) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7))+((m3)*(M4)*(M5)*(M6)*(M7))+((m4)*(M5)*(M6)*(M7))+((m5)*(M6)*(M7))+((m6)*(M7))+((m7)))
#define CALC_8D_ARRAY_INDEX(M1,M2,M3,M4,M5,M6,M7,M8,m1,m2,m3,m4,m5,m6,m7,m8) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m3)*(M4)*(M5)*(M6)*(M7)*(M8))+((m4)*(M5)*(M6)*(M7)*(M8))+((m5)*(M6)*(M7)*(M8))+((m6)*(M7)*(M8))+((m7)*(M8))+((m8)))
#define CALC_9D_ARRAY_INDEX(M1,M2,M3,M4,M5,M6,M7,M8,M9,m1,m2,m3,m4,m5,m6,m7,m8,m9) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9))+((m3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9))+((m4)*(M5)*(M6)*(M7)*(M8)*(M9))+((m5)*(M6)*(M7)*(M8)*(M9))+((m6)*(M7)*(M8)*(M9))+((m7)*(M8)*(M9))+((m8)*(M9))+((m9)))
#define CALC_10D_ARRAY_INDEX(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10) (((m1)*(M2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9)*(M10))+((m2)*(M3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9)*(M10))+((m3)*(M4)*(M5)*(M6)*(M7)*(M8)*(M9)*(M10))+((m4)*(M5)*(M6)*(M7)*(M8)*(M9)*(M10))+((m5)*(M6)*(M7)*(M8)*(M9)*(M10))+((m6)*(M7)*(M8)*(M9)*(M10))+((m7)*(M8)*(M9)*(M10))+((m8)*(M9)*(M10))+((m9)*(M10))+((m10)))
#define GROUP_SIZE	128




//Functions that will be used by the kernels (START)

static void exact_solution_g4 (double xi, double eta, double zeta,
			       double dtemp[5], __global double *ce);
static void exact_solution_g4_e4 (double xi, double eta, double zeta,
				  double dtemp[5], __global double *ce);
static void binvcrhs_g0_g5_g10 (__global double *lhs, int lhs_0, int lhs_1,
				int lhs_2, int lhs_3, __global double *c,
				int c_0, int c_1, int c_2, int c_3,
				__global double *r, int r_0, int r_1,
				int r_2);
static void matvec_sub_g0_g5_g9 (__global double *ablock, int ablock_0,
				 int ablock_1, int ablock_2, int ablock_3,
				 __global double *avec, int avec_0,
				 int avec_1, int avec_2,
				 __global double *bvec, int bvec_0,
				 int bvec_1, int bvec_2);
static void matmul_sub_g0_g5_g10 (__global double *ablock, int ablock_0,
				  int ablock_1, int ablock_2, int ablock_3,
				  __global double *bblock, int bblock_0,
				  int bblock_1, int bblock_2, int bblock_3,
				  __global double *cblock, int cblock_0,
				  int cblock_1, int cblock_2, int cblock_3);
static void binvrhs_g0_g5 (__global double *lhs, int lhs_0, int lhs_1,
			   int lhs_2, int lhs_3, __global double *r, int r_0,
			   int r_1, int r_2);
static void binvcrhs (double lhs[3][5][5][163][163][163], int lhs_0,
		      int lhs_1, int lhs_2, int lhs_3,
		      double c[3][5][5][163][163][163], int c_0, int c_1,
		      int c_2, int c_3, double r[5][163][163][163], int r_0,
		      int r_1, int r_2);
static void matvec_sub (double ablock[3][5][5][163][163][163], int ablock_0,
			int ablock_1, int ablock_2, int ablock_3,
			double avec[5][163][163][163], int avec_0, int avec_1,
			int avec_2, double bvec[5][163][163][163], int bvec_0,
			int bvec_1, int bvec_2);
static void matmul_sub (double ablock[3][5][5][163][163][163], int ablock_0,
			int ablock_1, int ablock_2, int ablock_3,
			double bblock[3][5][5][163][163][163], int bblock_0,
			int bblock_1, int bblock_2, int bblock_3,
			double cblock[3][5][5][163][163][163], int cblock_0,
			int cblock_1, int cblock_2, int cblock_3);
static void binvrhs (double lhs[3][5][5][163][163][163], int lhs_0, int lhs_1,
		     int lhs_2, int lhs_3, double r[5][163][163][163],
		     int r_0, int r_1, int r_2);

static void
binvcrhs (double lhs[3][5][5][163][163][163], int lhs_0, int lhs_1, int lhs_2,
	  int lhs_3, double c[3][5][5][163][163][163], int c_0, int c_1,
	  int c_2, int c_3, double r[5][163][163][163], int r_0, int r_1,
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

static void
matvec_sub (double ablock[3][5][5][163][163][163], int ablock_0, int ablock_1,
	    int ablock_2, int ablock_3, double avec[5][163][163][163],
	    int avec_0, int avec_1, int avec_2, double bvec[5][163][163][163],
	    int bvec_0, int bvec_1, int bvec_2)
{
  int i;
  for (i = 0; i < 5; i++)
    {
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

static void
matmul_sub (double ablock[3][5][5][163][163][163], int ablock_0, int ablock_1,
	    int ablock_2, int ablock_3, double bblock[3][5][5][163][163][163],
	    int bblock_0, int bblock_1, int bblock_2, int bblock_3,
	    double cblock[3][5][5][163][163][163], int cblock_0, int cblock_1,
	    int cblock_2, int cblock_3)
{
  int j;
  for (j = 0; j < 5; j++)
    {
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

static void
binvrhs (double lhs[3][5][5][163][163][163], int lhs_0, int lhs_1, int lhs_2,
	 int lhs_3, double r[5][163][163][163], int r_0, int r_1, int r_2)
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

static void
exact_solution_g4 (double xi, double eta, double zeta, double dtemp[5],
		   __global double *ce)
{
  int m;
  for (m = 0; m < 5; m++)
    {
      dtemp[m] =
	ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (0))] +
	xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (1))] +
	      xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (4))] +
		    xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (7))] +
			  xi * ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (10))]))) +
	eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (2))] +
	       eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (5))] +
		      eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (8))] +
			     eta *
			     ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (11))]))) +
	zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (3))] +
		zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (6))] +
			zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (9))] +
				zeta *
				ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (12))])));
    }
}

static void
exact_solution_g4_e4 (double xi, double eta, double zeta, double dtemp[5],
		   __global double *ce)
{
  int m;
  for (m = 0; m < 5; m++)
    {
      dtemp[m] =
	ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (0))] +
	xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (1))] +
	      xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (4))] +
		    xi * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (7))] +
			  xi * ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (10))]))) +
	eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (2))] +
	       eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (5))] +
		      eta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (8))] +
			     eta *
			     ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (11))]))) +
	zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (3))] +
		zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (6))] +
			zeta * (ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (9))] +
				zeta *
				ce[CALC_2D_ARRAY_INDEX (5, 13, (m), (12))])));
    }
}

static void
binvcrhs_g0_g5_g10 (__global double *lhs, int lhs_0, int lhs_1, int lhs_2,
		    int lhs_3, __global double *c, int c_0, int c_1, int c_2,
		    int c_3, __global double *r, int r_0, int r_1, int r_2)
{
  double pivot, coeff;
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(0), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(1), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(2), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(3), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (0), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (1), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (2), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (3), (c_0), (c_1), (c_2))] * pivot;
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (4), (4), (c_0), (c_1), (c_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (0), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (1), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (2), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (0), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (0), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (1), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (1), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (2), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (2), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (3), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (3), (c_0), (c_1),
						(c_2))];
  c[CALC_6D_ARRAY_INDEX
    (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1), (c_2))] =
    c[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (c_3), (3), (4), (c_0), (c_1),
       (c_2))] - coeff * c[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163, (c_3),
						(4), (4), (c_0), (c_1),
						(c_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
}

static void
matvec_sub_g0_g5_g9 (__global double *ablock, int ablock_0, int ablock_1,
		     int ablock_2, int ablock_3, __global double *avec,
		     int avec_0, int avec_1, int avec_2,
		     __global double *bvec, int bvec_0, int bvec_1,
		     int bvec_2)
{
  int i;
  for (i = 0; i < 5; i++)
    {
      bvec[CALC_4D_ARRAY_INDEX
	   (5, 163, 163, 163, (i), (bvec_0), (bvec_1), (bvec_2))] =
	bvec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (i), (bvec_0), (bvec_1),
	      (bvec_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						       (ablock_3), (i), (0),
						       (ablock_0), (ablock_1),
						       (ablock_2))] *
	avec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (0), (avec_0), (avec_1),
	      (avec_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						       (ablock_3), (i), (1),
						       (ablock_0), (ablock_1),
						       (ablock_2))] *
	avec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (1), (avec_0), (avec_1),
	      (avec_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						       (ablock_3), (i), (2),
						       (ablock_0), (ablock_1),
						       (ablock_2))] *
	avec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (2), (avec_0), (avec_1),
	      (avec_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						       (ablock_3), (i), (3),
						       (ablock_0), (ablock_1),
						       (ablock_2))] *
	avec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (3), (avec_0), (avec_1),
	      (avec_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						       (ablock_3), (i), (4),
						       (ablock_0), (ablock_1),
						       (ablock_2))] *
	avec[CALC_4D_ARRAY_INDEX
	     (5, 163, 163, 163, (4), (avec_0), (avec_1), (avec_2))];
    }
}

static void
matmul_sub_g0_g5_g10 (__global double *ablock, int ablock_0, int ablock_1,
		      int ablock_2, int ablock_3, __global double *bblock,
		      int bblock_0, int bblock_1, int bblock_2, int bblock_3,
		      __global double *cblock, int cblock_0, int cblock_1,
		      int cblock_2, int cblock_3)
{
  int j;
  for (j = 0; j < 5; j++)
    {
      cblock[CALC_6D_ARRAY_INDEX
	     (3, 5, 5, 163, 163, 163, (cblock_3), (0), (j), (cblock_0),
	      (cblock_1), (cblock_2))] =
	cblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (cblock_3), (0), (j), (cblock_0),
		(cblock_1), (cblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (0),
								       (0),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (0), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (0),
								       (1),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (1), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (0),
								       (2),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (2), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (0),
								       (3),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (3), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (0),
								       (4),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (4), (j), (bblock_0),
		(bblock_1), (bblock_2))];
      cblock[CALC_6D_ARRAY_INDEX
	     (3, 5, 5, 163, 163, 163, (cblock_3), (1), (j), (cblock_0),
	      (cblock_1), (cblock_2))] =
	cblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (cblock_3), (1), (j), (cblock_0),
		(cblock_1), (cblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (1),
								       (0),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (0), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (1),
								       (1),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (1), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (1),
								       (2),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (2), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (1),
								       (3),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (3), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (1),
								       (4),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (4), (j), (bblock_0),
		(bblock_1), (bblock_2))];
      cblock[CALC_6D_ARRAY_INDEX
	     (3, 5, 5, 163, 163, 163, (cblock_3), (2), (j), (cblock_0),
	      (cblock_1), (cblock_2))] =
	cblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (cblock_3), (2), (j), (cblock_0),
		(cblock_1), (cblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (2),
								       (0),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (0), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (2),
								       (1),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (1), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (2),
								       (2),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (2), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (2),
								       (3),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (3), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (2),
								       (4),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (4), (j), (bblock_0),
		(bblock_1), (bblock_2))];
      cblock[CALC_6D_ARRAY_INDEX
	     (3, 5, 5, 163, 163, 163, (cblock_3), (3), (j), (cblock_0),
	      (cblock_1), (cblock_2))] =
	cblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (cblock_3), (3), (j), (cblock_0),
		(cblock_1), (cblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (3),
								       (0),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (0), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (3),
								       (1),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (1), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (3),
								       (2),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (2), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (3),
								       (3),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (3), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (3),
								       (4),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (4), (j), (bblock_0),
		(bblock_1), (bblock_2))];
      cblock[CALC_6D_ARRAY_INDEX
	     (3, 5, 5, 163, 163, 163, (cblock_3), (4), (j), (cblock_0),
	      (cblock_1), (cblock_2))] =
	cblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (cblock_3), (4), (j), (cblock_0),
		(cblock_1), (cblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (4),
								       (0),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (0), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (4),
								       (1),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (1), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (4),
								       (2),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (2), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (4),
								       (3),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (3), (j), (bblock_0),
		(bblock_1), (bblock_2))] - ablock[CALC_6D_ARRAY_INDEX (3, 5,
								       5, 163,
								       163,
								       163,
								       (ablock_3),
								       (4),
								       (4),
								       (ablock_0),
								       (ablock_1),
								       (ablock_2))]
	*
	bblock[CALC_6D_ARRAY_INDEX
	       (3, 5, 5, 163, 163, 163, (bblock_3), (4), (j), (bblock_0),
		(bblock_1), (bblock_2))];
    }
}

static void
binvrhs_g0_g5 (__global double *lhs, int lhs_0, int lhs_1, int lhs_2,
	       int lhs_3, __global double *r, int r_0, int r_1, int r_2)
{
  double pivot, coeff;
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (0), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (1),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (0), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (1), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (2),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (1), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (2), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (3),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (2), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))] * pivot;
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (3), (lhs_0), (lhs_1),
	 (lhs_2))];
  lhs[CALC_6D_ARRAY_INDEX
      (3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
       (lhs_2))] =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))] - coeff * lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163, 163,
						      (lhs_3), (3), (4),
						      (lhs_0), (lhs_1),
						      (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))];
  pivot =
    1.00 /
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (4), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))] *
    pivot;
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (0), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (1), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (2), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
  coeff =
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (lhs_3), (3), (4), (lhs_0), (lhs_1),
	 (lhs_2))];
  r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] =
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (r_0), (r_1), (r_2))] -
    coeff *
    r[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (r_0), (r_1), (r_2))];
}


//Functions that will be used by the kernels (END)

/* This is origined from a loop of bt.c at line: 219 */
__kernel void
add_0 (__global double *u, __global double *rhs, int __ocl_k_bound,
       int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 215 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] +
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))];
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 342 */
__kernel void
exact_rhs_0 (__global double *forcing, int __ocl_k_bound, int __ocl_j_bound,
	     int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 335 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	0.0;
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 358 */
__kernel void
exact_rhs_1 (double dnym1, double dnzm1, __global int *grid_points,
	     double dnxm1, __global double *forcing, double tx2,
	     double dx1tx1, double c2, double xxcon1, double dx2tx1,
	     double xxcon2, double dx3tx1, double dx4tx1, double c1,
	     double xxcon3, double xxcon4, double xxcon5, double dx5tx1,
	     double dssp, __global double *ce, int __ocl_k_bound,
	     int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */
  double eta;			/* DEFINED AT bt.c : 334 */
  double zeta;			/* DEFINED AT bt.c : 334 */
  int i;			/* DEFINED AT bt.c : 335 */
  double xi;			/* DEFINED AT bt.c : 334 */
  double dtemp[5];		/* DEFINED AT bt.c : 334 */
  int m;			/* DEFINED AT bt.c : 335 */
  double ue[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 74 */
  double dtpp;			/* DEFINED AT bt.c : 334 */
  double buf[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 75 */
  double cuf[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 72 */
  double q[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 73 */
  int im1;			/* DEFINED AT bt.c : 335 */
  int ip1;			/* DEFINED AT bt.c : 335 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    eta = (double) j *dnym1;
    zeta = (double) k *dnzm1;
    for (i = 0; i < grid_points[(0)]; i++)
      {
	xi = (double) i *dnxm1;
	exact_solution_g4 (xi, eta, zeta, dtemp, ce) /*Arg Expension: ce */ ;
	for (m = 0; m < 5; m++)
	  {
	    ue[m][i] = dtemp[m];
	  }
	dtpp = 1.0 / dtemp[0];
	for (m = 1; m <= 4; m++)
	  {
	    buf[m][i] = dtpp * dtemp[m];
	  }
	cuf[i] = buf[1][i] * buf[1][i];
	buf[0][i] = cuf[i] + buf[2][i] * buf[2][i] + buf[3][i] * buf[3][i];
	q[i] =
	  0.5 * (buf[1][i] * ue[1][i] + buf[2][i] * ue[2][i] +
		 buf[3][i] * ue[3][i]);
      }
    for (i = 1; i < grid_points[(0)] - 1; i++)
      {
	im1 = i - 1;
	ip1 = i + 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))]
	  - tx2 * (ue[1][ip1] - ue[1][im1]) + dx1tx1 * (ue[0][ip1] -
							2.0 * ue[0][i] +
							ue[0][im1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))]
	  - tx2 * ((ue[1][ip1] * buf[1][ip1] + c2 * (ue[4][ip1] - q[ip1])) -
		   (ue[1][im1] * buf[1][im1] + c2 * (ue[4][im1] - q[im1]))) +
	  xxcon1 * (buf[1][ip1] - 2.0 * buf[1][i] + buf[1][im1]) +
	  dx2tx1 * (ue[1][ip1] - 2.0 * ue[1][i] + ue[1][im1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))]
	  - tx2 * (ue[2][ip1] * buf[1][ip1] - ue[2][im1] * buf[1][im1]) +
	  xxcon2 * (buf[2][ip1] - 2.0 * buf[2][i] + buf[2][im1]) +
	  dx3tx1 * (ue[2][ip1] - 2.0 * ue[2][i] + ue[2][im1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))]
	  - tx2 * (ue[3][ip1] * buf[1][ip1] - ue[3][im1] * buf[1][im1]) +
	  xxcon2 * (buf[3][ip1] - 2.0 * buf[3][i] + buf[3][im1]) +
	  dx4tx1 * (ue[3][ip1] - 2.0 * ue[3][i] + ue[3][im1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))]
	  - tx2 * (buf[1][ip1] * (c1 * ue[4][ip1] - c2 * q[ip1]) -
		   buf[1][im1] * (c1 * ue[4][im1] - c2 * q[im1])) +
	  0.5 * xxcon3 * (buf[0][ip1] - 2.0 * buf[0][i] + buf[0][im1]) +
	  xxcon4 * (cuf[ip1] - 2.0 * cuf[i] + cuf[im1]) +
	  xxcon5 * (buf[4][ip1] - 2.0 * buf[4][i] + buf[4][im1]) +
	  dx5tx1 * (ue[4][ip1] - 2.0 * ue[4][i] + ue[4][im1]);
      }
    for (m = 0; m < 5; m++)
      {
	i = 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (5.0 * ue[m][i] - 4.0 * ue[m][i + 1] + ue[m][i + 2]);
	i = 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (-4.0 * ue[m][i - 1] + 6.0 * ue[m][i] -
		    4.0 * ue[m][i + 1] + ue[m][i + 2]);
      }
    for (m = 0; m < 5; m++)
      {
	for (i = 1 * 3; i <= grid_points[(0)] - 3 * 1 - 1; i++)
	  {
	    forcing[CALC_4D_ARRAY_INDEX
		    (6, 163, 163, 163, (m), (i), (j), (k))] =
	      forcing[CALC_4D_ARRAY_INDEX
		      (6, 163, 163, 163, (m), (i), (j),
		       (k))] - dssp * (ue[m][i - 2] - 4.0 * ue[m][i - 1] +
				       6.0 * ue[m][i] - 4.0 * ue[m][i + 1] +
				       ue[m][i + 2]);
	  }
      }
    for (m = 0; m < 5; m++)
      {
	i = grid_points[(0)] - 3;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][i - 2] - 4.0 * ue[m][i - 1] + 6.0 * ue[m][i] -
		    4.0 * ue[m][i + 1]);
	i = grid_points[(0)] - 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][i - 2] - 4.0 * ue[m][i - 1] + 5.0 * ue[m][i]);
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 458 */
__kernel void
exact_rhs_2 (double dnxm1, double dnzm1, __global int *grid_points,
	     double dnym1, __global double *forcing, double ty2,
	     double dy1ty1, double yycon2, double dy2ty1, double c2,
	     double yycon1, double dy3ty1, double dy4ty1, double c1,
	     double yycon3, double yycon4, double yycon5, double dy5ty1,
	     double dssp, __global double *ce, int __ocl_k_bound,
	     int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 334 */
  double zeta;			/* DEFINED AT bt.c : 334 */
  int j;			/* DEFINED AT bt.c : 335 */
  double eta;			/* DEFINED AT bt.c : 334 */
  double dtemp[5];		/* DEFINED AT bt.c : 334 */
  int m;			/* DEFINED AT bt.c : 335 */
  double ue[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 74 */
  double dtpp;			/* DEFINED AT bt.c : 334 */
  double buf[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 75 */
  double cuf[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 72 */
  double q[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 73 */
  int jm1;			/* DEFINED AT bt.c : 335 */
  int jp1;			/* DEFINED AT bt.c : 335 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    zeta = (double) k *dnzm1;
    for (j = 0; j < grid_points[(1)]; j++)
      {
	eta = (double) j *dnym1;
	exact_solution_g4 (xi, eta, zeta, dtemp, ce) /*Arg Expension: ce */ ;
	for (m = 0; m < 5; m++)
	  {
	    ue[m][j] = dtemp[m];
	  }
	dtpp = 1.0 / dtemp[0];
	for (m = 1; m <= 4; m++)
	  {
	    buf[m][j] = dtpp * dtemp[m];
	  }
	cuf[j] = buf[2][j] * buf[2][j];
	buf[0][j] = cuf[j] + buf[1][j] * buf[1][j] + buf[3][j] * buf[3][j];
	q[j] =
	  0.5 * (buf[1][j] * ue[1][j] + buf[2][j] * ue[2][j] +
		 buf[3][j] * ue[3][j]);
      }
    for (j = 1; j < grid_points[(1)] - 1; j++)
      {
	jm1 = j - 1;
	jp1 = j + 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))]
	  - ty2 * (ue[2][jp1] - ue[2][jm1]) + dy1ty1 * (ue[0][jp1] -
							2.0 * ue[0][j] +
							ue[0][jm1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))]
	  - ty2 * (ue[1][jp1] * buf[2][jp1] - ue[1][jm1] * buf[2][jm1]) +
	  yycon2 * (buf[1][jp1] - 2.0 * buf[1][j] + buf[1][jm1]) +
	  dy2ty1 * (ue[1][jp1] - 2.0 * ue[1][j] + ue[1][jm1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))]
	  - ty2 * ((ue[2][jp1] * buf[2][jp1] + c2 * (ue[4][jp1] - q[jp1])) -
		   (ue[2][jm1] * buf[2][jm1] + c2 * (ue[4][jm1] - q[jm1]))) +
	  yycon1 * (buf[2][jp1] - 2.0 * buf[2][j] + buf[2][jm1]) +
	  dy3ty1 * (ue[2][jp1] - 2.0 * ue[2][j] + ue[2][jm1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))]
	  - ty2 * (ue[3][jp1] * buf[2][jp1] - ue[3][jm1] * buf[2][jm1]) +
	  yycon2 * (buf[3][jp1] - 2.0 * buf[3][j] + buf[3][jm1]) +
	  dy4ty1 * (ue[3][jp1] - 2.0 * ue[3][j] + ue[3][jm1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))]
	  - ty2 * (buf[2][jp1] * (c1 * ue[4][jp1] - c2 * q[jp1]) -
		   buf[2][jm1] * (c1 * ue[4][jm1] - c2 * q[jm1])) +
	  0.5 * yycon3 * (buf[0][jp1] - 2.0 * buf[0][j] + buf[0][jm1]) +
	  yycon4 * (cuf[jp1] - 2.0 * cuf[j] + cuf[jm1]) +
	  yycon5 * (buf[4][jp1] - 2.0 * buf[4][j] + buf[4][jm1]) +
	  dy5ty1 * (ue[4][jp1] - 2.0 * ue[4][j] + ue[4][jm1]);
      }
    for (m = 0; m < 5; m++)
      {
	j = 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (5.0 * ue[m][j] - 4.0 * ue[m][j + 1] + ue[m][j + 2]);
	j = 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (-4.0 * ue[m][j - 1] + 6.0 * ue[m][j] -
		    4.0 * ue[m][j + 1] + ue[m][j + 2]);
      }
    for (m = 0; m < 5; m++)
      {
	for (j = 1 * 3; j <= grid_points[(1)] - 3 * 1 - 1; j++)
	  {
	    forcing[CALC_4D_ARRAY_INDEX
		    (6, 163, 163, 163, (m), (i), (j), (k))] =
	      forcing[CALC_4D_ARRAY_INDEX
		      (6, 163, 163, 163, (m), (i), (j),
		       (k))] - dssp * (ue[m][j - 2] - 4.0 * ue[m][j - 1] +
				       6.0 * ue[m][j] - 4.0 * ue[m][j + 1] +
				       ue[m][j + 2]);
	  }
      }
    for (m = 0; m < 5; m++)
      {
	j = grid_points[(1)] - 3;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][j - 2] - 4.0 * ue[m][j - 1] + 6.0 * ue[m][j] -
		    4.0 * ue[m][j + 1]);
	j = grid_points[(1)] - 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][j - 2] - 4.0 * ue[m][j - 1] + 5.0 * ue[m][j]);
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 560 */
__kernel void
exact_rhs_3 (double dnxm1, double dnym1, __global int *grid_points,
	     double dnzm1, __global double *forcing, double tz2,
	     double dz1tz1, double zzcon2, double dz2tz1, double dz3tz1,
	     double c2, double zzcon1, double dz4tz1, double c1,
	     double zzcon3, double zzcon4, double zzcon5, double dz5tz1,
	     double dssp, __global double *ce, int __ocl_j_bound,
	     int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 334 */
  double eta;			/* DEFINED AT bt.c : 334 */
  int k;			/* DEFINED AT bt.c : 335 */
  double zeta;			/* DEFINED AT bt.c : 334 */
  double dtemp[5];		/* DEFINED AT bt.c : 334 */
  int m;			/* DEFINED AT bt.c : 335 */
  double ue[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 74 */
  double dtpp;			/* DEFINED AT bt.c : 334 */
  double buf[5][162];		/* THREADPRIVATE: DEFINED AT ./header.h : 75 */
  double cuf[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 72 */
  double q[162];		/* THREADPRIVATE: DEFINED AT ./header.h : 73 */
  int km1;			/* DEFINED AT bt.c : 335 */
  int kp1;			/* DEFINED AT bt.c : 335 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    eta = (double) j *dnym1;
    for (k = 0; k < grid_points[(2)]; k++)
      {
	zeta = (double) k *dnzm1;
	exact_solution_g4 (xi, eta, zeta, dtemp, ce) /*Arg Expension: ce */ ;
	for (m = 0; m < 5; m++)
	  {
	    ue[m][k] = dtemp[m];
	  }
	dtpp = 1.0 / dtemp[0];
	for (m = 1; m <= 4; m++)
	  {
	    buf[m][k] = dtpp * dtemp[m];
	  }
	cuf[k] = buf[3][k] * buf[3][k];
	buf[0][k] = cuf[k] + buf[1][k] * buf[1][k] + buf[2][k] * buf[2][k];
	q[k] =
	  0.5 * (buf[1][k] * ue[1][k] + buf[2][k] * ue[2][k] +
		 buf[3][k] * ue[3][k]);
      }
    for (k = 1; k < grid_points[(2)] - 1; k++)
      {
	km1 = k - 1;
	kp1 = k + 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (0), (i), (j), (k))]
	  - tz2 * (ue[3][kp1] - ue[3][km1]) + dz1tz1 * (ue[0][kp1] -
							2.0 * ue[0][k] +
							ue[0][km1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (1), (i), (j), (k))]
	  - tz2 * (ue[1][kp1] * buf[3][kp1] - ue[1][km1] * buf[3][km1]) +
	  zzcon2 * (buf[1][kp1] - 2.0 * buf[1][k] + buf[1][km1]) +
	  dz2tz1 * (ue[1][kp1] - 2.0 * ue[1][k] + ue[1][km1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (2), (i), (j), (k))]
	  - tz2 * (ue[2][kp1] * buf[3][kp1] - ue[2][km1] * buf[3][km1]) +
	  zzcon2 * (buf[2][kp1] - 2.0 * buf[2][k] + buf[2][km1]) +
	  dz3tz1 * (ue[2][kp1] - 2.0 * ue[2][k] + ue[2][km1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (3), (i), (j), (k))]
	  - tz2 * ((ue[3][kp1] * buf[3][kp1] + c2 * (ue[4][kp1] - q[kp1])) -
		   (ue[3][km1] * buf[3][km1] + c2 * (ue[4][km1] - q[km1]))) +
	  zzcon1 * (buf[3][kp1] - 2.0 * buf[3][k] + buf[3][km1]) +
	  dz4tz1 * (ue[3][kp1] - 2.0 * ue[3][k] + ue[3][km1]);
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (4), (i), (j), (k))]
	  - tz2 * (buf[3][kp1] * (c1 * ue[4][kp1] - c2 * q[kp1]) -
		   buf[3][km1] * (c1 * ue[4][km1] - c2 * q[km1])) +
	  0.5 * zzcon3 * (buf[0][kp1] - 2.0 * buf[0][k] + buf[0][km1]) +
	  zzcon4 * (cuf[kp1] - 2.0 * cuf[k] + cuf[km1]) +
	  zzcon5 * (buf[4][kp1] - 2.0 * buf[4][k] + buf[4][km1]) +
	  dz5tz1 * (ue[4][kp1] - 2.0 * ue[4][k] + ue[4][km1]);
      }
    for (m = 0; m < 5; m++)
      {
	k = 1;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (5.0 * ue[m][k] - 4.0 * ue[m][k + 1] + ue[m][k + 2]);
	k = 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (-4.0 * ue[m][k - 1] + 6.0 * ue[m][k] -
		    4.0 * ue[m][k + 1] + ue[m][k + 2]);
      }
    for (m = 0; m < 5; m++)
      {
	for (k = 1 * 3; k <= grid_points[(2)] - 3 * 1 - 1; k++)
	  {
	    forcing[CALC_4D_ARRAY_INDEX
		    (6, 163, 163, 163, (m), (i), (j), (k))] =
	      forcing[CALC_4D_ARRAY_INDEX
		      (6, 163, 163, 163, (m), (i), (j),
		       (k))] - dssp * (ue[m][k - 2] - 4.0 * ue[m][k - 1] +
				       6.0 * ue[m][k] - 4.0 * ue[m][k + 1] +
				       ue[m][k + 2]);
	  }
      }
    for (m = 0; m < 5; m++)
      {
	k = grid_points[(2)] - 3;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][k - 2] - 4.0 * ue[m][k - 1] + 6.0 * ue[m][k] -
		    4.0 * ue[m][k + 1]);
	k = grid_points[(2)] - 2;
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	  forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))]
	  - dssp * (ue[m][k - 2] - 4.0 * ue[m][k - 1] + 5.0 * ue[m][k]);
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 660 */
__kernel void
exact_rhs_4 (__global double *forcing, int __ocl_k_bound, int __ocl_j_bound,
	     int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 335 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))] =
	-1.0 *
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))];
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 722 */
__kernel void
initialize_0 (__global double *u)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(162))
    return;
  int j = get_global_id (1);
  if (!(162))
    return;
  int i = get_global_id (2);
  if (!(162))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] = 1.0;
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 739 */
__kernel void
initialize_1 (double dnym1, double dnxm1, double dnzm1, __global double *u,
	      __global double *ce, int __ocl_k_bound, int __ocl_j_bound,
	      int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double eta;			/* DEFINED AT bt.c : 712 */
  double xi;			/* DEFINED AT bt.c : 712 */
  double zeta;			/* DEFINED AT bt.c : 712 */
  int ix;			/* DEFINED AT bt.c : 711 */
  double Pface[2][3][5];	/* DEFINED AT bt.c : 712 */
  int iy;			/* DEFINED AT bt.c : 711 */
  int iz;			/* DEFINED AT bt.c : 711 */
  int m;			/* DEFINED AT bt.c : 711 */
  double Pxi;			/* DEFINED AT bt.c : 712 */
  double Peta;			/* DEFINED AT bt.c : 712 */
  double Pzeta;			/* DEFINED AT bt.c : 712 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    eta = (double) j *dnym1;
    xi = (double) i *dnxm1;
    zeta = (double) k *dnzm1;
    for (ix = 0; ix < 2; ix++)
      {
	exact_solution_g4_e4 ((double) ix, eta, zeta, &(Pface[ix][0][0]),
			      ce) /*Arg Expension: ce */ ;
      }
    for (iy = 0; iy < 2; iy++)
      {
	exact_solution_g4_e4 (xi, (double) iy, zeta, &Pface[iy][1][0],
			      ce) /*Arg Expension: ce */ ;
      }
    for (iz = 0; iz < 2; iz++)
      {
	exact_solution_g4_e4 (xi, eta, (double) iz, &Pface[iz][2][0],
			      ce) /*Arg Expension: ce */ ;
      }
    for (m = 0; m < 5; m++)
      {
	Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
	Peta = eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
	Pzeta = zeta * Pface[1][2][m] + (1.0 - zeta) * Pface[0][2][m];
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  Pxi + Peta + Pzeta - Pxi * Peta - Pxi * Pzeta - Peta * Pzeta +
	  Pxi * Peta * Pzeta;
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 789 */
__kernel void
initialize_2 (double dnym1, double dnzm1, double xi, __global double *u,
	      int i, __global double *ce, int __ocl_k_bound,
	      int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */
  double eta;			/* DEFINED AT bt.c : 712 */
  double zeta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    eta = (double) j *dnym1;
    zeta = (double) k *dnzm1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 809 */
__kernel void
initialize_3 (double dnym1, double dnzm1, double xi, __global double *u,
	      int i, __global double *ce, int __ocl_k_bound,
	      int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */
  double eta;			/* DEFINED AT bt.c : 712 */
  double zeta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    eta = (double) j *dnym1;
    zeta = (double) k *dnzm1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 828 */
__kernel void
initialize_4 (double dnxm1, double dnzm1, double eta, __global double *u,
	      int j, __global double *ce, int __ocl_k_bound,
	      int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 712 */
  double zeta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    zeta = (double) k *dnzm1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 847 */
__kernel void
initialize_5 (double dnxm1, double dnzm1, double eta, __global double *u,
	      int j, __global double *ce, int __ocl_k_bound,
	      int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 712 */
  double zeta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    zeta = (double) k *dnzm1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 866 */
__kernel void
initialize_6 (double dnxm1, double dnym1, double zeta, __global double *u,
	      int k, __global double *ce, int __ocl_j_bound,
	      int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 712 */
  double eta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    eta = (double) j *dnym1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 885 */
__kernel void
initialize_7 (double dnxm1, double dnym1, double zeta, __global double *u,
	      int k, __global double *ce, int __ocl_j_bound,
	      int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double xi;			/* DEFINED AT bt.c : 712 */
  double eta;			/* DEFINED AT bt.c : 712 */
  double temp[5];		/* DEFINED AT bt.c : 712 */
  int m;			/* DEFINED AT bt.c : 711 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    xi = (double) i *dnxm1;
    eta = (double) j *dnym1;
    exact_solution_g4_e4 (xi, eta, zeta, temp, ce) /*Arg Expension: ce */ ;
    for (m = 0; m < 5; m++)
      {
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  temp[m];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 913 */
__kernel void
lhsinit_0 (__global double *lhs, int __ocl_k_bound, int __ocl_j_bound,
	   int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 903 */
  int n;			/* DEFINED AT bt.c : 903 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (n = 0; n < 5; n++)
    for (m = 0; m < 5; m++)
      {
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (m), (n), (i), (j), (k))] = 0.0;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (m), (n), (i), (j), (k))] = 0.0;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (m), (n), (i), (j), (k))] = 0.0;
      }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 933 */
__kernel void
lhsinit_1 (__global double *lhs, int __ocl_k_bound, int __ocl_j_bound,
	   int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 903 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    for (m = 0; m < 5; m++)
      {
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (m), (m), (i), (j), (k))] = 1.0;
      }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 964 */
__kernel void
lhsx_0 (__global int *grid_points, __global double *u, __global double *fjac,
	double c2, double c1, __global double *njac, double con43,
	double c3c4, double c1345, double dt, double tx1, double tx2,
	__global double *lhs, double dx1, double dx2, double dx3, double dx4,
	double dx5, int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */
  int i;			/* DEFINED AT bt.c : 957 */
  double tmp1;			/* DEFINED AT ./header.h : 88 */
  double tmp2;			/* DEFINED AT ./header.h : 88 */
  double tmp3;			/* DEFINED AT ./header.h : 88 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    for (i = 0; i < grid_points[(0)]; i++)
      {
	tmp1 =
	  1.0 / u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))];
	tmp2 = tmp1 * tmp1;
	tmp3 = tmp1 * tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] = 1.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
	  -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	    tmp2 *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) +
	  c2 * 0.50 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	  tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
	  (2.0 -
	   c2) *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] /
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))]);
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] =
	  -c2 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	   tmp1);
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] =
	  -c2 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	   tmp1);
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] = c2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
	  -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]) *
	  tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] =
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	  tmp1;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] =
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	  tmp1;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
	  -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	  tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] =
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	  tmp1;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] =
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	  tmp1;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] = 0.0;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
	  (c2 *
	   (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	    u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	   tmp2 -
	   c1 *
	   (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
	    tmp1)) * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j),
					     (k))] * tmp1);
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
	  c1 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
	  tmp1 -
	  0.50 * c2 * (3.0 *
		       u[CALC_4D_ARRAY_INDEX
			 (5, 163, 163, 163, (1), (i), (j),
			  (k))] * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163,
							 (1), (i), (j),
							 (k))] +
		       u[CALC_4D_ARRAY_INDEX
			 (5, 163, 163, 163, (2), (i), (j),
			  (k))] * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163,
							 (2), (i), (j),
							 (k))] +
		       u[CALC_4D_ARRAY_INDEX
			 (5, 163, 163, 163, (3), (i), (j),
			  (k))] * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163,
							 (3), (i), (j),
							 (k))]) * tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
	  -c2 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) *
	  tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
	  -c2 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	   u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) *
	  tmp2;
	fjac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] =
	  c1 *
	  (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	   tmp1);
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
	  -con43 * c3c4 * tmp2 *
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
	  con43 * c3c4 * tmp1;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
	  -c3c4 * tmp2 *
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] = c3c4 * tmp1;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
	  -c3c4 * tmp2 *
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] = c3c4 * tmp1;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] = 0.0;
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
	  -(con43 * c3c4 -
	    c1345) * tmp3 *
	  (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) *
	    (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))])))
	  - (c3c4 -
	     c1345) * tmp3 *
	  (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]) *
	    (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))])))
	  - (c3c4 -
	     c1345) * tmp3 *
	  (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	    (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))])))
	  -
	  c1345 * tmp2 *
	  u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
	  (con43 * c3c4 -
	   c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i),
						  (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
	  (c3c4 -
	   c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i),
						  (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
	  (c3c4 -
	   c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i),
						  (j), (k))];
	njac[CALC_5D_ARRAY_INDEX
	     (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] = (c1345) * tmp1;
      }
    for (i = 1; i < grid_points[(0)] - 1; i++)
      {
	tmp1 = dt * tx1;
	tmp2 = dt * tx2;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (0), (0), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (0), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (0), (i - 1),
							 (j),
							 (k))] - tmp1 * dx1;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (0), (1), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (1), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (1), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (0), (2), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (2), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (2), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (0), (3), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (3), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (3), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (0), (4), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (4), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (4), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (1), (0), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (0), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (0), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (1), (1), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (1), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (1), (i - 1),
							 (j),
							 (k))] - tmp1 * dx2;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (1), (2), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (2), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (2), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (1), (3), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (3), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (3), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (1), (4), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (4), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (4), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (2), (0), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (0), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (0), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (2), (1), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (1), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (1), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (2), (2), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (2), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (2), (i - 1),
							 (j),
							 (k))] - tmp1 * dx3;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (2), (3), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (3), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (3), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (2), (4), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (4), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (4), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (3), (0), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (0), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (0), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (3), (1), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (1), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (1), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (3), (2), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (2), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (2), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (3), (3), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (3), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (3), (i - 1),
							 (j),
							 (k))] - tmp1 * dx4;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (3), (4), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (4), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (4), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (4), (0), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (0), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (0), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (4), (1), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (1), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (1), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (4), (2), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (2), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (2), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (4), (3), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (3), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (3), (i - 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (0), (4), (4), (i), (j), (k))] =
	  -tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (4), (i - 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (4), (i - 1),
							 (j),
							 (k))] - tmp1 * dx5;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (0), (0), (i), (j), (k))] =
	  1.0 +
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (0), (i), (j),
		(k))] + tmp1 * 2.0 * dx1;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (0), (1), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (0), (2), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (0), (3), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (0), (4), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (1), (0), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (1), (1), (i), (j), (k))] =
	  1.0 +
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (1), (i), (j),
		(k))] + tmp1 * 2.0 * dx2;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (1), (2), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (1), (3), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (1), (4), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (2), (0), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (2), (1), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (2), (2), (i), (j), (k))] =
	  1.0 +
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (2), (i), (j),
		(k))] + tmp1 * 2.0 * dx3;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (2), (3), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (2), (4), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (3), (0), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (3), (1), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (3), (2), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (3), (3), (i), (j), (k))] =
	  1.0 +
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (3), (i), (j),
		(k))] + tmp1 * 2.0 * dx4;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (3), (4), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (4), (0), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (4), (1), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (4), (2), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (4), (3), (i), (j), (k))] =
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (1), (4), (4), (i), (j), (k))] =
	  1.0 +
	  tmp1 * 2.0 *
	  njac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (4), (i), (j),
		(k))] + tmp1 * 2.0 * dx5;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (0), (0), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (0), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (0), (i + 1),
							 (j),
							 (k))] - tmp1 * dx1;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (0), (1), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (1), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (1), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (0), (2), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (2), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (2), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (0), (3), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (3), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (3), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (0), (4), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (0), (4), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (4), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (1), (0), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (0), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (0), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (1), (1), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (1), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (1), (i + 1),
							 (j),
							 (k))] - tmp1 * dx2;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (1), (2), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (2), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (2), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (1), (3), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (3), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (3), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (1), (4), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (1), (4), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (4), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (2), (0), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (0), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (0), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (2), (1), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (1), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (1), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (2), (2), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (2), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (2), (i + 1),
							 (j),
							 (k))] - tmp1 * dx3;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (2), (3), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (3), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (3), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (2), (4), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (2), (4), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (4), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (3), (0), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (0), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (0), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (3), (1), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (1), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (1), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (3), (2), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (2), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (2), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (3), (3), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (3), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (3), (i + 1),
							 (j),
							 (k))] - tmp1 * dx4;
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (3), (4), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (3), (4), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (4), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (4), (0), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (0), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (0), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (4), (1), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (1), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (1), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (4), (2), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (2), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (2), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (4), (3), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (3), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (3), (i + 1),
							 (j), (k))];
	lhs[CALC_6D_ARRAY_INDEX
	    (3, 5, 5, 163, 163, 163, (2), (4), (4), (i), (j), (k))] =
	  tmp2 *
	  fjac[CALC_5D_ARRAY_INDEX
	       (5, 5, 163, 163, 162, (4), (4), (i + 1), (j),
		(k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (4), (i + 1),
							 (j),
							 (k))] - tmp1 * dx5;
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1250 */
__kernel void
lhsy_0 (__global double *u, __global double *fjac, double c2, double c1,
	__global double *njac, double c3c4, double con43, double c1345,
	int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double tmp1;			/* DEFINED AT ./header.h : 88 */
  double tmp2;			/* DEFINED AT ./header.h : 88 */
  double tmp3;			/* DEFINED AT ./header.h : 88 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    tmp1 =
      1.0 / u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] =
      1.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]) * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	tmp2) +
      0.50 * c2 *
      ((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2);
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] =
      -c2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
      tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] =
      (2.0 -
       c2) * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j),
				    (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] =
      -c2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
      tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] =
      c2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
      (c2 *
       (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2 -
       c1 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
       tmp1) * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j),
				      (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
      -c2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
      c1 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
      tmp1 -
      0.50 * c2 *
      ((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	3.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2);
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
      -c2 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
      tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] =
      c1 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
      tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
      -c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
      c3c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
      -con43 * c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] =
      con43 * c3c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
      -c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] =
      c3c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
      -(c3c4 -
	c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]))) -
      (con43 * c3c4 -
       c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]))) -
      (c3c4 -
       c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]))) -
      c1345 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
      (c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
      (con43 * c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
      (c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] =
      (c1345) * tmp1;
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1354 */
__kernel void
lhsy_1 (double dt, double ty1, double ty2, __global double *lhs,
	__global double *fjac, __global double *njac, double dy1, double dy2,
	double dy3, double dy4, double dy5, int __ocl_k_bound,
	int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double tmp1;			/* DEFINED AT ./header.h : 88 */
  double tmp2;			/* DEFINED AT ./header.h : 88 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    tmp1 = dt * ty1;
    tmp2 = dt * ty2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (0), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (0), (i), (j - 1),
						     (k))] - tmp1 * dy1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (1), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (2), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (3), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (4), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (0), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (1), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (1), (i), (j - 1),
						     (k))] - tmp1 * dy2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (2), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (3), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (4), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (0), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (1), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (2), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (2), (i), (j - 1),
						     (k))] - tmp1 * dy3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (3), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (4), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (0), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (1), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (2), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (3), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (3), (i), (j - 1),
						     (k))] - tmp1 * dy4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (4), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (0), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (1), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (2), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (3), (i), (j - 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (4), (i), (j - 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (4), (i), (j - 1),
						     (k))] - tmp1 * dy5;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (0), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))]
      + tmp1 * 2.0 * dy1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (1), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))]
      + tmp1 * 2.0 * dy2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (2), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))]
      + tmp1 * 2.0 * dy3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (3), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))]
      + tmp1 * 2.0 * dy4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (4), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))]
      + tmp1 * 2.0 * dy5;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (0), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (0), (i), (j + 1),
						     (k))] - tmp1 * dy1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (1), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (2), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (3), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0),
						     (4), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (0), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (1), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (1), (i), (j + 1),
						     (k))] - tmp1 * dy2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (2), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (3), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1),
						     (4), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (0), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (1), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (2), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (2), (i), (j + 1),
						     (k))] - tmp1 * dy3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (3), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2),
						     (4), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (0), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (1), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (2), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (3), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (3), (i), (j + 1),
						     (k))] - tmp1 * dy4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3),
						     (4), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (0), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (1), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (2), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (3), (i), (j + 1), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (4), (i), (j + 1),
	    (k))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4),
						     (4), (i), (j + 1),
						     (k))] - tmp1 * dy5;
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1548 */
__kernel void
lhsz_0 (__global double *u, __global double *fjac, double c2, double c1,
	__global double *njac, double c3c4, double con43, double c3,
	double c4, double c1345, int __ocl_k_bound, int __ocl_j_bound,
	int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double tmp1;			/* DEFINED AT ./header.h : 88 */
  double tmp2;			/* DEFINED AT ./header.h : 88 */
  double tmp3;			/* DEFINED AT ./header.h : 88 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    tmp1 =
      1.0 / u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))];
    tmp2 = tmp1 * tmp1;
    tmp3 = tmp1 * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] =
      1.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) * tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] =
      0.0;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
      -(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	tmp2) +
      0.50 * c2 *
      ((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2);
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] =
      -c2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
      tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] =
      -c2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
      tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] =
      (2.0 -
       c2) * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j),
				    (k))] * tmp1;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] =
      c2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
      (c2 *
       (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2 -
       c1 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
	     tmp1)) * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j),
					      (k))] * tmp1);
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
      -c2 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
      tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
      -c2 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
      tmp2;
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
      c1 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
	    tmp1) -
      0.50 * c2 *
      ((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	3.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
       tmp2);
    fjac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] =
      c1 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
      tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))] =
      -c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))] =
      c3c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))] =
      -c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))] =
      c3c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))] =
      -con43 * c3c4 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))] =
      con43 * c3 * c4 * tmp1;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))] =
      0.0;
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))] =
      -(c3c4 -
	c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))]))) -
      (c3c4 -
       c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))]))) -
      (con43 * c3c4 -
       c1345) * tmp3 *
      (((u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]))) -
      c1345 * tmp2 *
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))] =
      (c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))] =
      (c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))] =
      (con43 * c3c4 -
       c1345) * tmp2 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j),
					      (k))];
    njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))] =
      (c1345) * tmp1;
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1652 */
__kernel void
lhsz_1 (double dt, double tz1, double tz2, __global double *lhs,
	__global double *fjac, __global double *njac, double dz1, double dz2,
	double dz3, double dz4, double dz5, int __ocl_k_bound,
	int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double tmp1;			/* DEFINED AT ./header.h : 88 */
  double tmp2;			/* DEFINED AT ./header.h : 88 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    tmp1 = dt * tz1;
    tmp2 = dt * tz2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (0), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (0), (i), (j),
							 (k - 1))] -
      tmp1 * dz1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (1), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (2), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (3), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (0), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (4), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (0), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (1), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (1), (i), (j),
							 (k - 1))] -
      tmp1 * dz2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (2), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (3), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (1), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (4), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (0), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (1), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (2), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (2), (i), (j),
							 (k - 1))] -
      tmp1 * dz3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (3), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (2), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (4), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (0), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (1), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (2), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (3), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (3), (i), (j),
							 (k - 1))] -
      tmp1 * dz4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (3), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (4), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (0), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (0), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (1), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (1), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (2), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (2), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (3), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (3), (i), (j),
							 (k - 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (0), (4), (4), (i), (j), (k))] =
      -tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (4), (i), (j),
	    (k - 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (4), (i), (j),
							 (k - 1))] -
      tmp1 * dz5;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (0), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (0), (0), (i), (j), (k))]
      + tmp1 * 2.0 * dz1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (0), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (1), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (1), (1), (i), (j), (k))]
      + tmp1 * 2.0 * dz2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (1), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (2), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (2), (2), (i), (j), (k))]
      + tmp1 * 2.0 * dz3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (2), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (3), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (3), (3), (i), (j), (k))]
      + tmp1 * 2.0 * dz4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (3), (4), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (0), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (1), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (2), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (3), (i), (j), (k))] =
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j), (k))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (1), (4), (4), (i), (j), (k))] =
      1.0 +
      tmp1 * 2.0 *
      njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162, (4), (4), (i), (j), (k))]
      + tmp1 * 2.0 * dz5;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (0), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (0), (i), (j),
							 (k + 1))] -
      tmp1 * dz1;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (1), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (1), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (2), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (2), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (3), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (3), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (0), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (0), (4), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (0), (4), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (0), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (0), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (1), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (1), (i), (j),
							 (k + 1))] -
      tmp1 * dz2;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (2), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (2), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (3), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (3), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (1), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (1), (4), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (1), (4), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (0), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (0), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (1), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (1), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (2), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (2), (i), (j),
							 (k + 1))] -
      tmp1 * dz3;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (3), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (3), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (2), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (2), (4), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (2), (4), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (0), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (0), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (1), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (1), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (2), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (2), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (3), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (3), (i), (j),
							 (k + 1))] -
      tmp1 * dz4;
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (3), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (3), (4), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (3), (4), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (0), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (0), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (0), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (1), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (1), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (1), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (2), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (2), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (2), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (3), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (3), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (3), (i), (j),
							 (k + 1))];
    lhs[CALC_6D_ARRAY_INDEX
	(3, 5, 5, 163, 163, 163, (2), (4), (4), (i), (j), (k))] =
      tmp2 *
      fjac[CALC_5D_ARRAY_INDEX
	   (5, 5, 163, 163, 162, (4), (4), (i), (j),
	    (k + 1))] - tmp1 * njac[CALC_5D_ARRAY_INDEX (5, 5, 163, 163, 162,
							 (4), (4), (i), (j),
							 (k + 1))] -
      tmp1 * dz5;
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1839 */
__kernel void
compute_rhs_0 (__global double *u, __global double *rho_i,
	       __global double *us, __global double *vs, __global double *ws,
	       __global double *square, __global double *qs,
	       int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double rho_inv;		/* DEFINED AT bt.c : 1831 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rho_inv =
      1.0 / u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))];
    rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] = rho_inv;
    us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] * rho_inv;
    vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] * rho_inv;
    ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] =
      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] * rho_inv;
    square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] =
      0.5 * (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] *
	     u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))]) *
      rho_inv;
    qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] =
      square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] * rho_inv;
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1864 */
__kernel void
compute_rhs_1 (__global double *rhs, __global double *forcing,
	       int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0);
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1);
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2);
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 1830 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	forcing[CALC_4D_ARRAY_INDEX (6, 163, 163, 163, (m), (i), (j), (k))];
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1880 */
__kernel void
compute_rhs_2 (__global double *us, __global double *rhs, double dx1tx1,
	       __global double *u, double tx2, double dx2tx1, double xxcon2,
	       double con43, __global double *square, double c2,
	       double dx3tx1, __global double *vs, double dx4tx1,
	       __global double *ws, double dx5tx1, double xxcon3,
	       __global double *qs, double xxcon4, double xxcon5,
	       __global double *rho_i, double c1, int __ocl_k_bound,
	       int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double uijk;			/* DEFINED AT bt.c : 1831 */
  double up1;			/* DEFINED AT bt.c : 1831 */
  double um1;			/* DEFINED AT bt.c : 1831 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    uijk = us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))];
    up1 = us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))];
    um1 = us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))];
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
      dx1tx1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i - 1), (j), (k))]) -
      tx2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i + 1), (j), (k))] -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i - 1), (j), (k))]);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
      dx2tx1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i - 1), (j), (k))]) +
      xxcon2 * con43 * (up1 - 2.0 * uijk + um1) -
      tx2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i + 1), (j), (k))] *
       up1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i - 1), (j), (k))] *
       um1 +
       (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i + 1), (j), (k))] -
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))] -
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i - 1), (j), (k))] +
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) * c2);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
      dx3tx1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i - 1), (j), (k))]) +
      xxcon2 * (vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))] -
		2.0 * vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) -
      tx2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i + 1), (j), (k))] *
       up1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i - 1), (j), (k))] *
       um1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
      dx4tx1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i - 1), (j), (k))]) +
      xxcon2 * (ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))] -
		2.0 * ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) -
      tx2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i + 1), (j), (k))] *
       up1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i - 1), (j), (k))] *
       um1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
      dx5tx1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i - 1), (j), (k))]) +
      xxcon3 * (qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))] -
		2.0 * qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) +
      xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
      xxcon5 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i + 1), (j), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i - 1), (j), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) -
      tx2 *
      ((c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i + 1), (j), (k))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i + 1), (j), (k))]) *
       up1 -
       (c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i - 1), (j), (k))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i - 1), (j), (k))]) *
       um1);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1943 */
__kernel void
compute_rhs_3 (__global double *rhs, int i, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (5.0 *
	      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i + 1), (j),
		 (k))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						(i + 2), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1957 */
__kernel void
compute_rhs_4 (__global double *rhs, int i, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (-4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i - 1), (j),
		 (k))] + 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						      (i), (j),
						      (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i + 1), (j),
		 (k))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						(i + 2), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1970 */
__kernel void
compute_rhs_5 (__global double *rhs, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 3;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 1830 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	dssp *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 2), (j), (k))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 1), (j), (k))] +
	 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i + 1), (j), (k))] +
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i + 2), (j), (k))]);
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 1987 */
__kernel void
compute_rhs_6 (__global double *rhs, int i, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 2), (j), (k))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 1), (j), (k))] +
       6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i + 1), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2001 */
__kernel void
compute_rhs_7 (__global double *rhs, int i, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 2), (j), (k))] -
       4. *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i - 1), (j), (k))] +
       5.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2017 */
__kernel void
compute_rhs_8 (__global double *vs, __global double *rhs, double dy1ty1,
	       __global double *u, double ty2, double dy2ty1, double yycon2,
	       __global double *us, double dy3ty1, double con43,
	       __global double *square, double c2, double dy4ty1,
	       __global double *ws, double dy5ty1, double yycon3,
	       __global double *qs, double yycon4, double yycon5,
	       __global double *rho_i, double c1, int __ocl_k_bound,
	       int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double vijk;			/* DEFINED AT bt.c : 1831 */
  double vp1;			/* DEFINED AT bt.c : 1831 */
  double vm1;			/* DEFINED AT bt.c : 1831 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    vijk = vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))];
    vp1 = vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))];
    vm1 = vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))];
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
      dy1ty1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j - 1), (k))]) -
      ty2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j + 1), (k))] -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j - 1), (k))]);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
      dy2ty1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j - 1), (k))]) +
      yycon2 * (us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))] -
		2.0 * us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) -
      ty2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j + 1), (k))] *
       vp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j - 1), (k))] *
       vm1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
      dy3ty1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j - 1), (k))]) +
      yycon2 * con43 * (vp1 - 2.0 * vijk + vm1) -
      ty2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j + 1), (k))] *
       vp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j - 1), (k))] *
       vm1 +
       (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j + 1), (k))] -
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))] -
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j - 1), (k))] +
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) * c2);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
      dy4ty1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j - 1), (k))]) +
      yycon2 * (ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))] -
		2.0 * ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) -
      ty2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j + 1), (k))] *
       vp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j - 1), (k))] *
       vm1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
      dy5ty1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j - 1), (k))]) +
      yycon3 * (qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))] -
		2.0 * qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) +
      yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
      yycon5 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j + 1), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j - 1), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) -
      ty2 *
      ((c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j + 1), (k))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j + 1), (k))]) *
       vp1 -
       (c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j - 1), (k))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j - 1), (k))]) *
       vm1);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2075 */
__kernel void
compute_rhs_9 (__global double *rhs, int j, double dssp, __global double *u,
	       int __ocl_k_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (5.0 *
	      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j + 1),
		 (k))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i),
						(j + 2), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2089 */
__kernel void
compute_rhs_10 (__global double *rhs, int j, double dssp, __global double *u,
		int __ocl_k_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (-4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j - 1),
		 (k))] + 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						      (i), (j),
						      (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j + 1),
		 (k))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i),
						(j + 2), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2102 */
__kernel void
compute_rhs_11 (__global double *rhs, double dssp, __global double *u,
		int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 3;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 1830 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	dssp *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 2), (k))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 1), (k))] +
	 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j + 1), (k))] +
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j + 2), (k))]);
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2119 */
__kernel void
compute_rhs_12 (__global double *rhs, int j, double dssp, __global double *u,
		int __ocl_k_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 2), (k))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 1), (k))] +
       6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j + 1), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2133 */
__kernel void
compute_rhs_13 (__global double *rhs, int j, double dssp, __global double *u,
		int __ocl_k_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 2), (k))] -
       4. *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j - 1), (k))] +
       5. * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2149 */
__kernel void
compute_rhs_14 (__global double *ws, __global double *rhs, double dz1tz1,
		__global double *u, double tz2, double dz2tz1, double zzcon2,
		__global double *us, double dz3tz1, __global double *vs,
		double dz4tz1, double con43, __global double *square,
		double c2, double dz5tz1, double zzcon3, __global double *qs,
		double zzcon4, double zzcon5, __global double *rho_i,
		double c1, int __ocl_k_bound, int __ocl_j_bound,
		int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  double wijk;			/* DEFINED AT bt.c : 1831 */
  double wp1;			/* DEFINED AT bt.c : 1831 */
  double wm1;			/* DEFINED AT bt.c : 1831 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    wijk = ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))];
    wp1 = ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))];
    wm1 = ws[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))];
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
      dz1tz1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (0), (i), (j), (k - 1))]) -
      tz2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k + 1))] -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k - 1))]);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
      dz2tz1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k - 1))]) +
      zzcon2 * (us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))] -
		2.0 * us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		us[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) -
      tz2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k + 1))] *
       wp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (1), (i), (j), (k - 1))] *
       wm1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
      dz3tz1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k - 1))]) +
      zzcon2 * (vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))] -
		2.0 * vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		vs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) -
      tz2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k + 1))] *
       wp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (2), (i), (j), (k - 1))] *
       wm1);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
      dz4tz1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k - 1))]) +
      zzcon2 * con43 * (wp1 - 2.0 * wijk + wm1) -
      tz2 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k + 1))] *
       wp1 -
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (3), (i), (j), (k - 1))] *
       wm1 +
       (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k + 1))] -
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))] -
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k - 1))] +
	square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) * c2);
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
      dz5tz1 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k - 1))]) +
      zzcon3 * (qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))] -
		2.0 * qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
		qs[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) +
      zzcon4 * (wp1 * wp1 - 2.0 * wijk * wijk + wm1 * wm1) +
      zzcon5 *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k + 1))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))] -
       2.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k))] +
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k - 1))] *
       rho_i[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) -
      tz2 *
      ((c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k + 1))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k + 1))]) *
       wp1 -
       (c1 *
	u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (4), (i), (j), (k - 1))] -
	c2 * square[CALC_3D_ARRAY_INDEX (163, 163, 163, (i), (j), (k - 1))]) *
       wm1);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2208 */
__kernel void
compute_rhs_15 (__global double *rhs, int k, double dssp, __global double *u,
		int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (5.0 *
	      u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j),
		 (k + 1))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						    (i), (j), (k + 2))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2222 */
__kernel void
compute_rhs_16 (__global double *rhs, int k, double dssp, __global double *u,
		int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp * (-4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j),
		 (k - 1))] + 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163,
							  (m), (i), (j),
							  (k))] -
	      4.0 *
	      u[CALC_4D_ARRAY_INDEX
		(5, 163, 163, 163, (m), (i), (j),
		 (k + 1))] + u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m),
						    (i), (j), (k + 2))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2235 */
__kernel void
compute_rhs_17 (__global double *rhs, double dssp, __global double *u,
		int __ocl_k_bound, int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 3;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 1830 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	dssp *
	(u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 2))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 1))] +
	 6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	 4.0 *
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k + 1))] +
	 u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k + 2))]);
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2252 */
__kernel void
compute_rhs_18 (__global double *rhs, int k, double dssp, __global double *u,
		int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 2))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 1))] +
       6.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k + 1))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2266 */
__kernel void
compute_rhs_19 (__global double *rhs, int k, double dssp, __global double *u,
		int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
      dssp *
      (u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 2))] -
       4.0 *
       u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k - 1))] +
       5.0 * u[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))]);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2279 */
__kernel void
compute_rhs_20 (__global double *rhs, double dt, int __ocl_k_bound,
		int __ocl_j_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (2) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int m;			/* DEFINED AT bt.c : 1830 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  for (m = 0; m < 5; m++)
    {
      rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] * dt;
    }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2805 */
__kernel void
x_backsubstitute_0 (__global double *rhs, int i, __global double *lhs,
		    int __ocl_k_bound, int __ocl_j_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */
  int n;			/* DEFINED AT bt.c : 2801 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    for (n = 0; n < 5; n++)
      {
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	  lhs[CALC_6D_ARRAY_INDEX
	      (3, 5, 5, 163, 163, 163, (2), (m), (n), (i), (j),
	       (k))] * rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (n),
						(i + 1), (j), (k))];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2843 */
__kernel void
x_solve_cell_0 (__global double *lhs, __global double *rhs, int __ocl_j_bound,
		int __ocl_k_bound)
{
  /*Global variables */
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    binvcrhs_g0_g5_g10 (lhs, 0, j, k, 1, lhs, 0, j, k, 2, rhs, 0, j, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2864 */
__kernel void
x_solve_cell_1 (__global double *lhs, int i, __global double *rhs,
		int __ocl_j_bound, int __ocl_k_bound)
{
  /*Global variables */
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, i, j, k, 0, rhs, i - 1, j, k, rhs, i, j, k);
    matmul_sub_g0_g5_g10 (lhs, i, j, k, 0, lhs, i - 1, j, k, 2, lhs, i, j, k,
			  1);
    binvcrhs_g0_g5_g10 (lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 2896 */
__kernel void
x_solve_cell_2 (__global double *lhs, int isize, __global double *rhs, int i,
		int __ocl_j_bound, int __ocl_k_bound)
{
  /*Global variables */
  int j = get_global_id (1) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, isize, j, k, 0, rhs, isize - 1, j, k, rhs,
			 isize, j, k);
    matmul_sub_g0_g5_g10 (lhs, isize, j, k, 0, lhs, isize - 1, j, k, 2, lhs,
			  isize, j, k, 1);
    binvrhs_g0_g5 (lhs, i, j, k, 1, rhs, i, j, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3453 */
__kernel void
y_backsubstitute_0 (__global double *rhs, int j, __global double *lhs,
		    int __ocl_k_bound, int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int m = get_global_id (2);
  if (!(5))
    return;


  /* Private Variables */
  int n;			/* DEFINED AT bt.c : 3448 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    for (n = 0; n < 5; n++)
      {
	rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] =
	  rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (m), (i), (j), (k))] -
	  lhs[CALC_6D_ARRAY_INDEX
	      (3, 5, 5, 163, 163, 163, (2), (m), (n), (i), (j),
	       (k))] * rhs[CALC_4D_ARRAY_INDEX (5, 163, 163, 163, (n), (i),
						(j + 1), (k))];
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3491 */
__kernel void
y_solve_cell_0 (__global double *lhs, __global double *rhs, int __ocl_i_bound,
		int __ocl_k_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    binvcrhs_g0_g5_g10 (lhs, i, 0, k, 1, lhs, i, 0, k, 2, rhs, i, 0, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3512 */
__kernel void
y_solve_cell_1 (__global double *lhs, int j, __global double *rhs,
		int __ocl_i_bound, int __ocl_k_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, i, j, k, 0, rhs, i, j - 1, k, rhs, i, j, k);
    matmul_sub_g0_g5_g10 (lhs, i, j, k, 0, lhs, i, j - 1, k, 2, lhs, i, j, k,
			  1);
    binvcrhs_g0_g5_g10 (lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3545 */
__kernel void
y_solve_cell_2 (__global double *lhs, int jsize, __global double *rhs,
		int __ocl_i_bound, int __ocl_k_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int k = get_global_id (0) + 1;
  if (!(k < __ocl_k_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, i, jsize, k, 0, rhs, i, jsize - 1, k, rhs, i,
			 jsize, k);
    matmul_sub_g0_g5_g10 (lhs, i, jsize, k, 0, lhs, i, jsize - 1, k, 2, lhs,
			  i, jsize, k, 1);
    binvrhs_g0_g5 (lhs, i, jsize, k, 1, rhs, i, jsize, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3617 */
__kernel void
z_backsubstitute_0 (__global int *grid_points, __global double *rhs,
		    __global double *lhs, int __ocl_j_bound,
		    int __ocl_i_bound)
{
  // The loops have been swaped. 
  /*Global variables */
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;


  /* Private Variables */
  int k;			/* DEFINED AT bt.c : 3613 */
  int m;			/* DEFINED AT bt.c : 3613 */
  int n;			/* DEFINED AT bt.c : 3613 */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    for (k = grid_points[(2)] - 2; k >= 0; k--)
      {
	for (m = 0; m < 5; m++)
	  {
	    for (n = 0; n < 5; n++)
	      {
		rhs[CALC_4D_ARRAY_INDEX
		    (5, 163, 163, 163, (m), (i), (k), (j))] =
		  rhs[CALC_4D_ARRAY_INDEX
		      (5, 163, 163, 163, (m), (i), (k),
		       (j))] - lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163,
							163, (2), (m), (n),
							(i), (k),
							(j))] *
		  rhs[CALC_4D_ARRAY_INDEX
		      (5, 163, 163, 163, (n), (i), (k + 1), (j))];

		////rhs[CALC_4D_ARRAY_INDEX
		////    (5, 163, 163, 163, (m), (i), (j), (k))] =
		////  rhs[CALC_4D_ARRAY_INDEX
		////      (5, 163, 163, 163, (m), (i), (j),
		////       (k))] - lhs[CALC_6D_ARRAY_INDEX (3, 5, 5, 163, 163,
		////					163, (2), (m), (n),
		////					(i), (j),
		////					(k))] *
		////  rhs[CALC_4D_ARRAY_INDEX
		////      (5, 163, 163, 163, (n), (i), (j), (k + 1))];
	      }
	  }
      }
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3659 */
__kernel void
z_solve_cell_0 (__global double *lhs, __global double *rhs, int __ocl_i_bound,
		int __ocl_j_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    binvcrhs_g0_g5_g10 (lhs, i, 0, j, 1, lhs, i, 0, j, 2, rhs, i, 0, j);
    ////binvcrhs_g0_g5_g10 (lhs, i, j, 0, 1, lhs, i, j, 0, 2, rhs, i, j, 0);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3681 */
__kernel void
z_solve_cell_1 (__global double *lhs, int k, __global double *rhs,
		int __ocl_i_bound, int __ocl_j_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, i, k, j, 0, rhs, i, k - 1, j, rhs, i, k, j);
    matmul_sub_g0_g5_g10 (lhs, i, k, j, 0, lhs, i, k - 1, j, 2, lhs, i, k, j,
			  1);
    binvcrhs_g0_g5_g10 (lhs, i, k, j, 1, lhs, i, k, j, 2, rhs, i, k, j);

    ////matvec_sub_g0_g5_g9 (lhs, i, j, k, 0, rhs, i, j, k - 1, rhs, i, j, k);
    ////matmul_sub_g0_g5_g10 (lhs, i, j, k, 0, lhs, i, j, k - 1, 2, lhs, i, j, k,
    ////    		  1);
    ////binvcrhs_g0_g5_g10 (lhs, i, j, k, 1, lhs, i, j, k, 2, rhs, i, j, k);
  }
  //OPENCL KERNEL END 
}

/* This is origined from a loop of bt.c at line: 3718 */
__kernel void
z_solve_cell_2 (__global double *lhs, int ksize, __global double *rhs,
		int __ocl_i_bound, int __ocl_j_bound)
{
  /*Global variables */
  int i = get_global_id (1) + 1;
  if (!(i < __ocl_i_bound))
    return;
  int j = get_global_id (0) + 1;
  if (!(j < __ocl_j_bound))
    return;


  /* Private Variables */

//COPYIN (START)
//COPYIN (END)

  //OPENCL KERNEL START 
  {
    matvec_sub_g0_g5_g9 (lhs, i, ksize, j, 0, rhs, i, ksize - 1, j, rhs, i,
			 ksize, j);
    matmul_sub_g0_g5_g10 (lhs, i, ksize, j, 0, lhs, i, ksize - 1, j, 2, lhs,
			  i, ksize, j, 1);
    binvrhs_g0_g5 (lhs, i, ksize, j, 1, rhs, i, ksize, j);

    ////matvec_sub_g0_g5_g9 (lhs, i, j, ksize, 0, rhs, i, j, ksize - 1, rhs, i, j,
    ////    		 ksize);
    ////matmul_sub_g0_g5_g10 (lhs, i, j, ksize, 0, lhs, i, j, ksize - 1, 2, lhs,
    ////    		  i, j, ksize, 1);
    ////binvrhs_g0_g5 (lhs, i, j, ksize, 1, rhs, i, j, ksize);
  }
  //OPENCL KERNEL END 
}
