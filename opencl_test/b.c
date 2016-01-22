//-------------------------------------------------------------------------------
//OpenCL Kernels 
//Generated at : Mon Apr 30 15:53:05 2012
//Compiler options: 
//      Software Cache  true
//      Local Memory    true
//      DefaultParallelDepth    3
//      UserDefParallelDepth    false
//      EnableLoopInterchange   false
//      Generating debug/profiling code false
//      EnableMLFeatureCollection       false
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
void exact_solution_g4(double xi, double eta, double zeta, double dtemp[5],
		       __global double *ce);
void binvcrhs_p0_p1_p2(__global double *lhs, __global double *c,
		       __global double *r, unsigned arg_0_offset,
		       unsigned arg_1_offset, unsigned arg_2_offset);
void matvec_sub_p0_p1_p2(__global double *ablock, __global double *avec,
			 __global double *bvec, unsigned arg_0_offset,
			 unsigned arg_1_offset, unsigned arg_2_offset);
void matmul_sub_p0_p1_p2(__global double *ablock, __global double *bblock,
			 __global double *cblock, unsigned arg_0_offset,
			 unsigned arg_1_offset, unsigned arg_2_offset);
void binvrhs_p0_p1(__global double *lhs, __global double *r,
		   unsigned arg_0_offset, unsigned arg_1_offset);
void binvcrhs(double lhs[5][5], double c[5][5], double r[5]);
void matvec_sub(double ablock[5][5], double avec[5], double bvec[5]);
void matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5]);
void binvrhs(double lhs[5][5], double r[5]);

void binvcrhs(double lhs[5][5], double c[5][5], double r[5])
{
	double pivot, coeff;
	pivot = 1.00 / lhs[0][0];
	lhs[0][1] = lhs[0][1] * pivot;
	lhs[0][2] = lhs[0][2] * pivot;
	lhs[0][3] = lhs[0][3] * pivot;
	lhs[0][4] = lhs[0][4] * pivot;
	c[0][0] = c[0][0] * pivot;
	c[0][1] = c[0][1] * pivot;
	c[0][2] = c[0][2] * pivot;
	c[0][3] = c[0][3] * pivot;
	c[0][4] = c[0][4] * pivot;
	r[0] = r[0] * pivot;
	coeff = lhs[1][0];
	lhs[1][1] = lhs[1][1] - coeff * lhs[0][1];
	lhs[1][2] = lhs[1][2] - coeff * lhs[0][2];
	lhs[1][3] = lhs[1][3] - coeff * lhs[0][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[0][4];
	c[1][0] = c[1][0] - coeff * c[0][0];
	c[1][1] = c[1][1] - coeff * c[0][1];
	c[1][2] = c[1][2] - coeff * c[0][2];
	c[1][3] = c[1][3] - coeff * c[0][3];
	c[1][4] = c[1][4] - coeff * c[0][4];
	r[1] = r[1] - coeff * r[0];
	coeff = lhs[2][0];
	lhs[2][1] = lhs[2][1] - coeff * lhs[0][1];
	lhs[2][2] = lhs[2][2] - coeff * lhs[0][2];
	lhs[2][3] = lhs[2][3] - coeff * lhs[0][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[0][4];
	c[2][0] = c[2][0] - coeff * c[0][0];
	c[2][1] = c[2][1] - coeff * c[0][1];
	c[2][2] = c[2][2] - coeff * c[0][2];
	c[2][3] = c[2][3] - coeff * c[0][3];
	c[2][4] = c[2][4] - coeff * c[0][4];
	r[2] = r[2] - coeff * r[0];
	coeff = lhs[3][0];
	lhs[3][1] = lhs[3][1] - coeff * lhs[0][1];
	lhs[3][2] = lhs[3][2] - coeff * lhs[0][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[0][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[0][4];
	c[3][0] = c[3][0] - coeff * c[0][0];
	c[3][1] = c[3][1] - coeff * c[0][1];
	c[3][2] = c[3][2] - coeff * c[0][2];
	c[3][3] = c[3][3] - coeff * c[0][3];
	c[3][4] = c[3][4] - coeff * c[0][4];
	r[3] = r[3] - coeff * r[0];
	coeff = lhs[4][0];
	lhs[4][1] = lhs[4][1] - coeff * lhs[0][1];
	lhs[4][2] = lhs[4][2] - coeff * lhs[0][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[0][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[0][4];
	c[4][0] = c[4][0] - coeff * c[0][0];
	c[4][1] = c[4][1] - coeff * c[0][1];
	c[4][2] = c[4][2] - coeff * c[0][2];
	c[4][3] = c[4][3] - coeff * c[0][3];
	c[4][4] = c[4][4] - coeff * c[0][4];
	r[4] = r[4] - coeff * r[0];
	pivot = 1.00 / lhs[1][1];
	lhs[1][2] = lhs[1][2] * pivot;
	lhs[1][3] = lhs[1][3] * pivot;
	lhs[1][4] = lhs[1][4] * pivot;
	c[1][0] = c[1][0] * pivot;
	c[1][1] = c[1][1] * pivot;
	c[1][2] = c[1][2] * pivot;
	c[1][3] = c[1][3] * pivot;
	c[1][4] = c[1][4] * pivot;
	r[1] = r[1] * pivot;
	coeff = lhs[0][1];
	lhs[0][2] = lhs[0][2] - coeff * lhs[1][2];
	lhs[0][3] = lhs[0][3] - coeff * lhs[1][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[1][4];
	c[0][0] = c[0][0] - coeff * c[1][0];
	c[0][1] = c[0][1] - coeff * c[1][1];
	c[0][2] = c[0][2] - coeff * c[1][2];
	c[0][3] = c[0][3] - coeff * c[1][3];
	c[0][4] = c[0][4] - coeff * c[1][4];
	r[0] = r[0] - coeff * r[1];
	coeff = lhs[2][1];
	lhs[2][2] = lhs[2][2] - coeff * lhs[1][2];
	lhs[2][3] = lhs[2][3] - coeff * lhs[1][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[1][4];
	c[2][0] = c[2][0] - coeff * c[1][0];
	c[2][1] = c[2][1] - coeff * c[1][1];
	c[2][2] = c[2][2] - coeff * c[1][2];
	c[2][3] = c[2][3] - coeff * c[1][3];
	c[2][4] = c[2][4] - coeff * c[1][4];
	r[2] = r[2] - coeff * r[1];
	coeff = lhs[3][1];
	lhs[3][2] = lhs[3][2] - coeff * lhs[1][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[1][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[1][4];
	c[3][0] = c[3][0] - coeff * c[1][0];
	c[3][1] = c[3][1] - coeff * c[1][1];
	c[3][2] = c[3][2] - coeff * c[1][2];
	c[3][3] = c[3][3] - coeff * c[1][3];
	c[3][4] = c[3][4] - coeff * c[1][4];
	r[3] = r[3] - coeff * r[1];
	coeff = lhs[4][1];
	lhs[4][2] = lhs[4][2] - coeff * lhs[1][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[1][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[1][4];
	c[4][0] = c[4][0] - coeff * c[1][0];
	c[4][1] = c[4][1] - coeff * c[1][1];
	c[4][2] = c[4][2] - coeff * c[1][2];
	c[4][3] = c[4][3] - coeff * c[1][3];
	c[4][4] = c[4][4] - coeff * c[1][4];
	r[4] = r[4] - coeff * r[1];
	pivot = 1.00 / lhs[2][2];
	lhs[2][3] = lhs[2][3] * pivot;
	lhs[2][4] = lhs[2][4] * pivot;
	c[2][0] = c[2][0] * pivot;
	c[2][1] = c[2][1] * pivot;
	c[2][2] = c[2][2] * pivot;
	c[2][3] = c[2][3] * pivot;
	c[2][4] = c[2][4] * pivot;
	r[2] = r[2] * pivot;
	coeff = lhs[0][2];
	lhs[0][3] = lhs[0][3] - coeff * lhs[2][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[2][4];
	c[0][0] = c[0][0] - coeff * c[2][0];
	c[0][1] = c[0][1] - coeff * c[2][1];
	c[0][2] = c[0][2] - coeff * c[2][2];
	c[0][3] = c[0][3] - coeff * c[2][3];
	c[0][4] = c[0][4] - coeff * c[2][4];
	r[0] = r[0] - coeff * r[2];
	coeff = lhs[1][2];
	lhs[1][3] = lhs[1][3] - coeff * lhs[2][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[2][4];
	c[1][0] = c[1][0] - coeff * c[2][0];
	c[1][1] = c[1][1] - coeff * c[2][1];
	c[1][2] = c[1][2] - coeff * c[2][2];
	c[1][3] = c[1][3] - coeff * c[2][3];
	c[1][4] = c[1][4] - coeff * c[2][4];
	r[1] = r[1] - coeff * r[2];
	coeff = lhs[3][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[2][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[2][4];
	c[3][0] = c[3][0] - coeff * c[2][0];
	c[3][1] = c[3][1] - coeff * c[2][1];
	c[3][2] = c[3][2] - coeff * c[2][2];
	c[3][3] = c[3][3] - coeff * c[2][3];
	c[3][4] = c[3][4] - coeff * c[2][4];
	r[3] = r[3] - coeff * r[2];
	coeff = lhs[4][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[2][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[2][4];
	c[4][0] = c[4][0] - coeff * c[2][0];
	c[4][1] = c[4][1] - coeff * c[2][1];
	c[4][2] = c[4][2] - coeff * c[2][2];
	c[4][3] = c[4][3] - coeff * c[2][3];
	c[4][4] = c[4][4] - coeff * c[2][4];
	r[4] = r[4] - coeff * r[2];
	pivot = 1.00 / lhs[3][3];
	lhs[3][4] = lhs[3][4] * pivot;
	c[3][0] = c[3][0] * pivot;
	c[3][1] = c[3][1] * pivot;
	c[3][2] = c[3][2] * pivot;
	c[3][3] = c[3][3] * pivot;
	c[3][4] = c[3][4] * pivot;
	r[3] = r[3] * pivot;
	coeff = lhs[0][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[3][4];
	c[0][0] = c[0][0] - coeff * c[3][0];
	c[0][1] = c[0][1] - coeff * c[3][1];
	c[0][2] = c[0][2] - coeff * c[3][2];
	c[0][3] = c[0][3] - coeff * c[3][3];
	c[0][4] = c[0][4] - coeff * c[3][4];
	r[0] = r[0] - coeff * r[3];
	coeff = lhs[1][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[3][4];
	c[1][0] = c[1][0] - coeff * c[3][0];
	c[1][1] = c[1][1] - coeff * c[3][1];
	c[1][2] = c[1][2] - coeff * c[3][2];
	c[1][3] = c[1][3] - coeff * c[3][3];
	c[1][4] = c[1][4] - coeff * c[3][4];
	r[1] = r[1] - coeff * r[3];
	coeff = lhs[2][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[3][4];
	c[2][0] = c[2][0] - coeff * c[3][0];
	c[2][1] = c[2][1] - coeff * c[3][1];
	c[2][2] = c[2][2] - coeff * c[3][2];
	c[2][3] = c[2][3] - coeff * c[3][3];
	c[2][4] = c[2][4] - coeff * c[3][4];
	r[2] = r[2] - coeff * r[3];
	coeff = lhs[4][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[3][4];
	c[4][0] = c[4][0] - coeff * c[3][0];
	c[4][1] = c[4][1] - coeff * c[3][1];
	c[4][2] = c[4][2] - coeff * c[3][2];
	c[4][3] = c[4][3] - coeff * c[3][3];
	c[4][4] = c[4][4] - coeff * c[3][4];
	r[4] = r[4] - coeff * r[3];
	pivot = 1.00 / lhs[4][4];
	c[4][0] = c[4][0] * pivot;
	c[4][1] = c[4][1] * pivot;
	c[4][2] = c[4][2] * pivot;
	c[4][3] = c[4][3] * pivot;
	c[4][4] = c[4][4] * pivot;
	r[4] = r[4] * pivot;
	coeff = lhs[0][4];
	c[0][0] = c[0][0] - coeff * c[4][0];
	c[0][1] = c[0][1] - coeff * c[4][1];
	c[0][2] = c[0][2] - coeff * c[4][2];
	c[0][3] = c[0][3] - coeff * c[4][3];
	c[0][4] = c[0][4] - coeff * c[4][4];
	r[0] = r[0] - coeff * r[4];
	coeff = lhs[1][4];
	c[1][0] = c[1][0] - coeff * c[4][0];
	c[1][1] = c[1][1] - coeff * c[4][1];
	c[1][2] = c[1][2] - coeff * c[4][2];
	c[1][3] = c[1][3] - coeff * c[4][3];
	c[1][4] = c[1][4] - coeff * c[4][4];
	r[1] = r[1] - coeff * r[4];
	coeff = lhs[2][4];
	c[2][0] = c[2][0] - coeff * c[4][0];
	c[2][1] = c[2][1] - coeff * c[4][1];
	c[2][2] = c[2][2] - coeff * c[4][2];
	c[2][3] = c[2][3] - coeff * c[4][3];
	c[2][4] = c[2][4] - coeff * c[4][4];
	r[2] = r[2] - coeff * r[4];
	coeff = lhs[3][4];
	c[3][0] = c[3][0] - coeff * c[4][0];
	c[3][1] = c[3][1] - coeff * c[4][1];
	c[3][2] = c[3][2] - coeff * c[4][2];
	c[3][3] = c[3][3] - coeff * c[4][3];
	c[3][4] = c[3][4] - coeff * c[4][4];
	r[3] = r[3] - coeff * r[4];
}

void matvec_sub(double ablock[5][5], double avec[5], double bvec[5])
{
	int i;
	for (i = 0; i < 5; i++) {
		bvec[i] =
		    bvec[i] - ablock[i][0] * avec[0] - ablock[i][1] * avec[1] -
		    ablock[i][2] * avec[2] - ablock[i][3] * avec[3] -
		    ablock[i][4] * avec[4];
	}
}

void matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5])
{
	int j;
	for (j = 0; j < 5; j++) {
		cblock[0][j] =
		    cblock[0][j] - ablock[0][0] * bblock[0][j] -
		    ablock[0][1] * bblock[1][j] - ablock[0][2] * bblock[2][j] -
		    ablock[0][3] * bblock[3][j] - ablock[0][4] * bblock[4][j];
		cblock[1][j] =
		    cblock[1][j] - ablock[1][0] * bblock[0][j] -
		    ablock[1][1] * bblock[1][j] - ablock[1][2] * bblock[2][j] -
		    ablock[1][3] * bblock[3][j] - ablock[1][4] * bblock[4][j];
		cblock[2][j] =
		    cblock[2][j] - ablock[2][0] * bblock[0][j] -
		    ablock[2][1] * bblock[1][j] - ablock[2][2] * bblock[2][j] -
		    ablock[2][3] * bblock[3][j] - ablock[2][4] * bblock[4][j];
		cblock[3][j] =
		    cblock[3][j] - ablock[3][0] * bblock[0][j] -
		    ablock[3][1] * bblock[1][j] - ablock[3][2] * bblock[2][j] -
		    ablock[3][3] * bblock[3][j] - ablock[3][4] * bblock[4][j];
		cblock[4][j] =
		    cblock[4][j] - ablock[4][0] * bblock[0][j] -
		    ablock[4][1] * bblock[1][j] - ablock[4][2] * bblock[2][j] -
		    ablock[4][3] * bblock[3][j] - ablock[4][4] * bblock[4][j];
	}
}

void binvrhs(double lhs[5][5], double r[5])
{
	double pivot, coeff;
	pivot = 1.00 / lhs[0][0];
	lhs[0][1] = lhs[0][1] * pivot;
	lhs[0][2] = lhs[0][2] * pivot;
	lhs[0][3] = lhs[0][3] * pivot;
	lhs[0][4] = lhs[0][4] * pivot;
	r[0] = r[0] * pivot;
	coeff = lhs[1][0];
	lhs[1][1] = lhs[1][1] - coeff * lhs[0][1];
	lhs[1][2] = lhs[1][2] - coeff * lhs[0][2];
	lhs[1][3] = lhs[1][3] - coeff * lhs[0][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[0][4];
	r[1] = r[1] - coeff * r[0];
	coeff = lhs[2][0];
	lhs[2][1] = lhs[2][1] - coeff * lhs[0][1];
	lhs[2][2] = lhs[2][2] - coeff * lhs[0][2];
	lhs[2][3] = lhs[2][3] - coeff * lhs[0][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[0][4];
	r[2] = r[2] - coeff * r[0];
	coeff = lhs[3][0];
	lhs[3][1] = lhs[3][1] - coeff * lhs[0][1];
	lhs[3][2] = lhs[3][2] - coeff * lhs[0][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[0][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[0][4];
	r[3] = r[3] - coeff * r[0];
	coeff = lhs[4][0];
	lhs[4][1] = lhs[4][1] - coeff * lhs[0][1];
	lhs[4][2] = lhs[4][2] - coeff * lhs[0][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[0][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[0][4];
	r[4] = r[4] - coeff * r[0];
	pivot = 1.00 / lhs[1][1];
	lhs[1][2] = lhs[1][2] * pivot;
	lhs[1][3] = lhs[1][3] * pivot;
	lhs[1][4] = lhs[1][4] * pivot;
	r[1] = r[1] * pivot;
	coeff = lhs[0][1];
	lhs[0][2] = lhs[0][2] - coeff * lhs[1][2];
	lhs[0][3] = lhs[0][3] - coeff * lhs[1][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[1][4];
	r[0] = r[0] - coeff * r[1];
	coeff = lhs[2][1];
	lhs[2][2] = lhs[2][2] - coeff * lhs[1][2];
	lhs[2][3] = lhs[2][3] - coeff * lhs[1][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[1][4];
	r[2] = r[2] - coeff * r[1];
	coeff = lhs[3][1];
	lhs[3][2] = lhs[3][2] - coeff * lhs[1][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[1][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[1][4];
	r[3] = r[3] - coeff * r[1];
	coeff = lhs[4][1];
	lhs[4][2] = lhs[4][2] - coeff * lhs[1][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[1][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[1][4];
	r[4] = r[4] - coeff * r[1];
	pivot = 1.00 / lhs[2][2];
	lhs[2][3] = lhs[2][3] * pivot;
	lhs[2][4] = lhs[2][4] * pivot;
	r[2] = r[2] * pivot;
	coeff = lhs[0][2];
	lhs[0][3] = lhs[0][3] - coeff * lhs[2][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[2][4];
	r[0] = r[0] - coeff * r[2];
	coeff = lhs[1][2];
	lhs[1][3] = lhs[1][3] - coeff * lhs[2][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[2][4];
	r[1] = r[1] - coeff * r[2];
	coeff = lhs[3][2];
	lhs[3][3] = lhs[3][3] - coeff * lhs[2][3];
	lhs[3][4] = lhs[3][4] - coeff * lhs[2][4];
	r[3] = r[3] - coeff * r[2];
	coeff = lhs[4][2];
	lhs[4][3] = lhs[4][3] - coeff * lhs[2][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[2][4];
	r[4] = r[4] - coeff * r[2];
	pivot = 1.00 / lhs[3][3];
	lhs[3][4] = lhs[3][4] * pivot;
	r[3] = r[3] * pivot;
	coeff = lhs[0][3];
	lhs[0][4] = lhs[0][4] - coeff * lhs[3][4];
	r[0] = r[0] - coeff * r[3];
	coeff = lhs[1][3];
	lhs[1][4] = lhs[1][4] - coeff * lhs[3][4];
	r[1] = r[1] - coeff * r[3];
	coeff = lhs[2][3];
	lhs[2][4] = lhs[2][4] - coeff * lhs[3][4];
	r[2] = r[2] - coeff * r[3];
	coeff = lhs[4][3];
	lhs[4][4] = lhs[4][4] - coeff * lhs[3][4];
	r[4] = r[4] - coeff * r[3];
	pivot = 1.00 / lhs[4][4];
	r[4] = r[4] * pivot;
	coeff = lhs[0][4];
	r[0] = r[0] - coeff * r[4];
	coeff = lhs[1][4];
	r[1] = r[1] - coeff * r[4];
	coeff = lhs[2][4];
	r[2] = r[2] - coeff * r[4];
	coeff = lhs[3][4];
	r[3] = r[3] - coeff * r[4];
}

//-------------------------------------------------------------------------------
//This is an alias of function: exact_solution
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ce
//-------------------------------------------------------------------------------
void exact_solution_g4(double xi, double eta, double zeta, double dtemp[5],
		       __global double *ce)
{

	int m;
	for (m = 0; m < 5; m++) {
		dtemp[m] =
		    ce[CALC_2D_IDX(5, 13, (m), (0))] +
		    xi * (ce[CALC_2D_IDX(5, 13, (m), (1))] +
			  xi * (ce[CALC_2D_IDX(5, 13, (m), (4))] +
				xi * (ce[CALC_2D_IDX(5, 13, (m), (7))] +
				      xi *
				      ce[CALC_2D_IDX(5, 13, (m), (10))]))) +
		    eta * (ce[CALC_2D_IDX(5, 13, (m), (2))] +
			   eta * (ce[CALC_2D_IDX(5, 13, (m), (5))] +
				  eta * (ce[CALC_2D_IDX(5, 13, (m), (8))] +
					 eta *
					 ce[CALC_2D_IDX(5, 13, (m), (11))]))) +
		    zeta * (ce[CALC_2D_IDX(5, 13, (m), (3))] +
			    zeta * (ce[CALC_2D_IDX(5, 13, (m), (6))] +
				    zeta * (ce[CALC_2D_IDX(5, 13, (m), (9))] +
					    zeta *
					    ce[CALC_2D_IDX
					       (5, 13, (m), (12))])));
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
void binvcrhs_p0_p1_p2(__global double *lhs, __global double *c,
		       __global double *r, unsigned arg_0_offset,
		       unsigned arg_1_offset, unsigned arg_2_offset)
{
	lhs += arg_0_offset;
	c += arg_1_offset;
	r += arg_2_offset;

	double pivot, coeff;
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (0), (0))];
	lhs[CALC_2D_IDX(5, 5, (0), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (1))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (2))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] * pivot;
	c[CALC_2D_IDX(5, 5, (0), (0))] = c[CALC_2D_IDX(5, 5, (0), (0))] * pivot;
	c[CALC_2D_IDX(5, 5, (0), (1))] = c[CALC_2D_IDX(5, 5, (0), (1))] * pivot;
	c[CALC_2D_IDX(5, 5, (0), (2))] = c[CALC_2D_IDX(5, 5, (0), (2))] * pivot;
	c[CALC_2D_IDX(5, 5, (0), (3))] = c[CALC_2D_IDX(5, 5, (0), (3))] * pivot;
	c[CALC_2D_IDX(5, 5, (0), (4))] = c[CALC_2D_IDX(5, 5, (0), (4))] * pivot;
	r[(0)] = r[(0)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (0))];
	lhs[CALC_2D_IDX(5, 5, (1), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (1), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	c[CALC_2D_IDX(5, 5, (1), (0))] =
	    c[CALC_2D_IDX(5, 5, (1), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (0))];
	c[CALC_2D_IDX(5, 5, (1), (1))] =
	    c[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (1))];
	c[CALC_2D_IDX(5, 5, (1), (2))] =
	    c[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (2))];
	c[CALC_2D_IDX(5, 5, (1), (3))] =
	    c[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (3))];
	c[CALC_2D_IDX(5, 5, (1), (4))] =
	    c[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (4))];
	r[(1)] = r[(1)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (0))];
	lhs[CALC_2D_IDX(5, 5, (2), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (2), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	c[CALC_2D_IDX(5, 5, (2), (0))] =
	    c[CALC_2D_IDX(5, 5, (2), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (0))];
	c[CALC_2D_IDX(5, 5, (2), (1))] =
	    c[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (1))];
	c[CALC_2D_IDX(5, 5, (2), (2))] =
	    c[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (2))];
	c[CALC_2D_IDX(5, 5, (2), (3))] =
	    c[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (3))];
	c[CALC_2D_IDX(5, 5, (2), (4))] =
	    c[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (4))];
	r[(2)] = r[(2)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (0))];
	lhs[CALC_2D_IDX(5, 5, (3), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (3), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	c[CALC_2D_IDX(5, 5, (3), (0))] =
	    c[CALC_2D_IDX(5, 5, (3), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (0))];
	c[CALC_2D_IDX(5, 5, (3), (1))] =
	    c[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (1))];
	c[CALC_2D_IDX(5, 5, (3), (2))] =
	    c[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (2))];
	c[CALC_2D_IDX(5, 5, (3), (3))] =
	    c[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (3))];
	c[CALC_2D_IDX(5, 5, (3), (4))] =
	    c[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (4))];
	r[(3)] = r[(3)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (0))];
	lhs[CALC_2D_IDX(5, 5, (4), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (4), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	c[CALC_2D_IDX(5, 5, (4), (0))] =
	    c[CALC_2D_IDX(5, 5, (4), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (0))];
	c[CALC_2D_IDX(5, 5, (4), (1))] =
	    c[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (1))];
	c[CALC_2D_IDX(5, 5, (4), (2))] =
	    c[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (2))];
	c[CALC_2D_IDX(5, 5, (4), (3))] =
	    c[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (3))];
	c[CALC_2D_IDX(5, 5, (4), (4))] =
	    c[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (0), (4))];
	r[(4)] = r[(4)] - coeff * r[(0)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (1), (1))];
	lhs[CALC_2D_IDX(5, 5, (1), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (2))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] * pivot;
	c[CALC_2D_IDX(5, 5, (1), (0))] = c[CALC_2D_IDX(5, 5, (1), (0))] * pivot;
	c[CALC_2D_IDX(5, 5, (1), (1))] = c[CALC_2D_IDX(5, 5, (1), (1))] * pivot;
	c[CALC_2D_IDX(5, 5, (1), (2))] = c[CALC_2D_IDX(5, 5, (1), (2))] * pivot;
	c[CALC_2D_IDX(5, 5, (1), (3))] = c[CALC_2D_IDX(5, 5, (1), (3))] * pivot;
	c[CALC_2D_IDX(5, 5, (1), (4))] = c[CALC_2D_IDX(5, 5, (1), (4))] * pivot;
	r[(1)] = r[(1)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (0), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	c[CALC_2D_IDX(5, 5, (0), (0))] =
	    c[CALC_2D_IDX(5, 5, (0), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (0))];
	c[CALC_2D_IDX(5, 5, (0), (1))] =
	    c[CALC_2D_IDX(5, 5, (0), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (1))];
	c[CALC_2D_IDX(5, 5, (0), (2))] =
	    c[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (2))];
	c[CALC_2D_IDX(5, 5, (0), (3))] =
	    c[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (3))];
	c[CALC_2D_IDX(5, 5, (0), (4))] =
	    c[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (4))];
	r[(0)] = r[(0)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (1))];
	lhs[CALC_2D_IDX(5, 5, (2), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	c[CALC_2D_IDX(5, 5, (2), (0))] =
	    c[CALC_2D_IDX(5, 5, (2), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (0))];
	c[CALC_2D_IDX(5, 5, (2), (1))] =
	    c[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (1))];
	c[CALC_2D_IDX(5, 5, (2), (2))] =
	    c[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (2))];
	c[CALC_2D_IDX(5, 5, (2), (3))] =
	    c[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (3))];
	c[CALC_2D_IDX(5, 5, (2), (4))] =
	    c[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (4))];
	r[(2)] = r[(2)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (1))];
	lhs[CALC_2D_IDX(5, 5, (3), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	c[CALC_2D_IDX(5, 5, (3), (0))] =
	    c[CALC_2D_IDX(5, 5, (3), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (0))];
	c[CALC_2D_IDX(5, 5, (3), (1))] =
	    c[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (1))];
	c[CALC_2D_IDX(5, 5, (3), (2))] =
	    c[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (2))];
	c[CALC_2D_IDX(5, 5, (3), (3))] =
	    c[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (3))];
	c[CALC_2D_IDX(5, 5, (3), (4))] =
	    c[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (4))];
	r[(3)] = r[(3)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (1))];
	lhs[CALC_2D_IDX(5, 5, (4), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	c[CALC_2D_IDX(5, 5, (4), (0))] =
	    c[CALC_2D_IDX(5, 5, (4), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (0))];
	c[CALC_2D_IDX(5, 5, (4), (1))] =
	    c[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (1))];
	c[CALC_2D_IDX(5, 5, (4), (2))] =
	    c[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (2))];
	c[CALC_2D_IDX(5, 5, (4), (3))] =
	    c[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (3))];
	c[CALC_2D_IDX(5, 5, (4), (4))] =
	    c[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (1), (4))];
	r[(4)] = r[(4)] - coeff * r[(1)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (2), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] * pivot;
	c[CALC_2D_IDX(5, 5, (2), (0))] = c[CALC_2D_IDX(5, 5, (2), (0))] * pivot;
	c[CALC_2D_IDX(5, 5, (2), (1))] = c[CALC_2D_IDX(5, 5, (2), (1))] * pivot;
	c[CALC_2D_IDX(5, 5, (2), (2))] = c[CALC_2D_IDX(5, 5, (2), (2))] * pivot;
	c[CALC_2D_IDX(5, 5, (2), (3))] = c[CALC_2D_IDX(5, 5, (2), (3))] * pivot;
	c[CALC_2D_IDX(5, 5, (2), (4))] = c[CALC_2D_IDX(5, 5, (2), (4))] * pivot;
	r[(2)] = r[(2)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	c[CALC_2D_IDX(5, 5, (0), (0))] =
	    c[CALC_2D_IDX(5, 5, (0), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (0))];
	c[CALC_2D_IDX(5, 5, (0), (1))] =
	    c[CALC_2D_IDX(5, 5, (0), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (1))];
	c[CALC_2D_IDX(5, 5, (0), (2))] =
	    c[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (2))];
	c[CALC_2D_IDX(5, 5, (0), (3))] =
	    c[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (3))];
	c[CALC_2D_IDX(5, 5, (0), (4))] =
	    c[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (4))];
	r[(0)] = r[(0)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	c[CALC_2D_IDX(5, 5, (1), (0))] =
	    c[CALC_2D_IDX(5, 5, (1), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (0))];
	c[CALC_2D_IDX(5, 5, (1), (1))] =
	    c[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (1))];
	c[CALC_2D_IDX(5, 5, (1), (2))] =
	    c[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (2))];
	c[CALC_2D_IDX(5, 5, (1), (3))] =
	    c[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (3))];
	c[CALC_2D_IDX(5, 5, (1), (4))] =
	    c[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (4))];
	r[(1)] = r[(1)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	c[CALC_2D_IDX(5, 5, (3), (0))] =
	    c[CALC_2D_IDX(5, 5, (3), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (0))];
	c[CALC_2D_IDX(5, 5, (3), (1))] =
	    c[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (1))];
	c[CALC_2D_IDX(5, 5, (3), (2))] =
	    c[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (2))];
	c[CALC_2D_IDX(5, 5, (3), (3))] =
	    c[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (3))];
	c[CALC_2D_IDX(5, 5, (3), (4))] =
	    c[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (4))];
	r[(3)] = r[(3)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	c[CALC_2D_IDX(5, 5, (4), (0))] =
	    c[CALC_2D_IDX(5, 5, (4), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (0))];
	c[CALC_2D_IDX(5, 5, (4), (1))] =
	    c[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (1))];
	c[CALC_2D_IDX(5, 5, (4), (2))] =
	    c[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (2))];
	c[CALC_2D_IDX(5, 5, (4), (3))] =
	    c[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (3))];
	c[CALC_2D_IDX(5, 5, (4), (4))] =
	    c[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (2), (4))];
	r[(4)] = r[(4)] - coeff * r[(2)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (3), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] * pivot;
	c[CALC_2D_IDX(5, 5, (3), (0))] = c[CALC_2D_IDX(5, 5, (3), (0))] * pivot;
	c[CALC_2D_IDX(5, 5, (3), (1))] = c[CALC_2D_IDX(5, 5, (3), (1))] * pivot;
	c[CALC_2D_IDX(5, 5, (3), (2))] = c[CALC_2D_IDX(5, 5, (3), (2))] * pivot;
	c[CALC_2D_IDX(5, 5, (3), (3))] = c[CALC_2D_IDX(5, 5, (3), (3))] * pivot;
	c[CALC_2D_IDX(5, 5, (3), (4))] = c[CALC_2D_IDX(5, 5, (3), (4))] * pivot;
	r[(3)] = r[(3)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	c[CALC_2D_IDX(5, 5, (0), (0))] =
	    c[CALC_2D_IDX(5, 5, (0), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (0))];
	c[CALC_2D_IDX(5, 5, (0), (1))] =
	    c[CALC_2D_IDX(5, 5, (0), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (1))];
	c[CALC_2D_IDX(5, 5, (0), (2))] =
	    c[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (2))];
	c[CALC_2D_IDX(5, 5, (0), (3))] =
	    c[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (3))];
	c[CALC_2D_IDX(5, 5, (0), (4))] =
	    c[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (4))];
	r[(0)] = r[(0)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	c[CALC_2D_IDX(5, 5, (1), (0))] =
	    c[CALC_2D_IDX(5, 5, (1), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (0))];
	c[CALC_2D_IDX(5, 5, (1), (1))] =
	    c[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (1))];
	c[CALC_2D_IDX(5, 5, (1), (2))] =
	    c[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (2))];
	c[CALC_2D_IDX(5, 5, (1), (3))] =
	    c[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (3))];
	c[CALC_2D_IDX(5, 5, (1), (4))] =
	    c[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (4))];
	r[(1)] = r[(1)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	c[CALC_2D_IDX(5, 5, (2), (0))] =
	    c[CALC_2D_IDX(5, 5, (2), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (0))];
	c[CALC_2D_IDX(5, 5, (2), (1))] =
	    c[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (1))];
	c[CALC_2D_IDX(5, 5, (2), (2))] =
	    c[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (2))];
	c[CALC_2D_IDX(5, 5, (2), (3))] =
	    c[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (3))];
	c[CALC_2D_IDX(5, 5, (2), (4))] =
	    c[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (4))];
	r[(2)] = r[(2)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	c[CALC_2D_IDX(5, 5, (4), (0))] =
	    c[CALC_2D_IDX(5, 5, (4), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (0))];
	c[CALC_2D_IDX(5, 5, (4), (1))] =
	    c[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (1))];
	c[CALC_2D_IDX(5, 5, (4), (2))] =
	    c[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (2))];
	c[CALC_2D_IDX(5, 5, (4), (3))] =
	    c[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (3))];
	c[CALC_2D_IDX(5, 5, (4), (4))] =
	    c[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (3), (4))];
	r[(4)] = r[(4)] - coeff * r[(3)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (4), (4))];
	c[CALC_2D_IDX(5, 5, (4), (0))] = c[CALC_2D_IDX(5, 5, (4), (0))] * pivot;
	c[CALC_2D_IDX(5, 5, (4), (1))] = c[CALC_2D_IDX(5, 5, (4), (1))] * pivot;
	c[CALC_2D_IDX(5, 5, (4), (2))] = c[CALC_2D_IDX(5, 5, (4), (2))] * pivot;
	c[CALC_2D_IDX(5, 5, (4), (3))] = c[CALC_2D_IDX(5, 5, (4), (3))] * pivot;
	c[CALC_2D_IDX(5, 5, (4), (4))] = c[CALC_2D_IDX(5, 5, (4), (4))] * pivot;
	r[(4)] = r[(4)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (4))];
	c[CALC_2D_IDX(5, 5, (0), (0))] =
	    c[CALC_2D_IDX(5, 5, (0), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (0))];
	c[CALC_2D_IDX(5, 5, (0), (1))] =
	    c[CALC_2D_IDX(5, 5, (0), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (1))];
	c[CALC_2D_IDX(5, 5, (0), (2))] =
	    c[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (2))];
	c[CALC_2D_IDX(5, 5, (0), (3))] =
	    c[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (3))];
	c[CALC_2D_IDX(5, 5, (0), (4))] =
	    c[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (4))];
	r[(0)] = r[(0)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (4))];
	c[CALC_2D_IDX(5, 5, (1), (0))] =
	    c[CALC_2D_IDX(5, 5, (1), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (0))];
	c[CALC_2D_IDX(5, 5, (1), (1))] =
	    c[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (1))];
	c[CALC_2D_IDX(5, 5, (1), (2))] =
	    c[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (2))];
	c[CALC_2D_IDX(5, 5, (1), (3))] =
	    c[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (3))];
	c[CALC_2D_IDX(5, 5, (1), (4))] =
	    c[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (4))];
	r[(1)] = r[(1)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (4))];
	c[CALC_2D_IDX(5, 5, (2), (0))] =
	    c[CALC_2D_IDX(5, 5, (2), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (0))];
	c[CALC_2D_IDX(5, 5, (2), (1))] =
	    c[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (1))];
	c[CALC_2D_IDX(5, 5, (2), (2))] =
	    c[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (2))];
	c[CALC_2D_IDX(5, 5, (2), (3))] =
	    c[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (3))];
	c[CALC_2D_IDX(5, 5, (2), (4))] =
	    c[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (4))];
	r[(2)] = r[(2)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (4))];
	c[CALC_2D_IDX(5, 5, (3), (0))] =
	    c[CALC_2D_IDX(5, 5, (3), (0))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (0))];
	c[CALC_2D_IDX(5, 5, (3), (1))] =
	    c[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (1))];
	c[CALC_2D_IDX(5, 5, (3), (2))] =
	    c[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (2))];
	c[CALC_2D_IDX(5, 5, (3), (3))] =
	    c[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (3))];
	c[CALC_2D_IDX(5, 5, (3), (4))] =
	    c[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * c[CALC_2D_IDX(5, 5, (4), (4))];
	r[(3)] = r[(3)] - coeff * r[(4)];

}

//-------------------------------------------------------------------------------
//This is an alias of function: matvec_sub
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: ablock
//      1: avec
//      2: bvec
//-------------------------------------------------------------------------------
void matvec_sub_p0_p1_p2(__global double *ablock, __global double *avec,
			 __global double *bvec, unsigned arg_0_offset,
			 unsigned arg_1_offset, unsigned arg_2_offset)
{
	ablock += arg_0_offset;
	avec += arg_1_offset;
	bvec += arg_2_offset;

	int i;
	for (i = 0; i < 5; i++) {
		bvec[(i)] =
		    bvec[(i)] -
		    ablock[CALC_2D_IDX(5, 5, (i), (0))] * avec[(0)] -
		    ablock[CALC_2D_IDX(5, 5, (i), (1))] * avec[(1)] -
		    ablock[CALC_2D_IDX(5, 5, (i), (2))] * avec[(2)] -
		    ablock[CALC_2D_IDX(5, 5, (i), (3))] * avec[(3)] -
		    ablock[CALC_2D_IDX(5, 5, (i), (4))] * avec[(4)];
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
void matmul_sub_p0_p1_p2(__global double *ablock, __global double *bblock,
			 __global double *cblock, unsigned arg_0_offset,
			 unsigned arg_1_offset, unsigned arg_2_offset)
{
	ablock += arg_0_offset;
	bblock += arg_1_offset;
	cblock += arg_2_offset;

	int j;
	for (j = 0; j < 5; j++) {
		cblock[CALC_2D_IDX(5, 5, (0), (j))] =
		    cblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (0), (0))] *
		    bblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (0), (1))] *
		    bblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (0), (2))] *
		    bblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (0), (3))] *
		    bblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (0), (4))] *
		    bblock[CALC_2D_IDX(5, 5, (4), (j))];
		cblock[CALC_2D_IDX(5, 5, (1), (j))] =
		    cblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (1), (0))] *
		    bblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (1), (1))] *
		    bblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (1), (2))] *
		    bblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (1), (3))] *
		    bblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (1), (4))] *
		    bblock[CALC_2D_IDX(5, 5, (4), (j))];
		cblock[CALC_2D_IDX(5, 5, (2), (j))] =
		    cblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (2), (0))] *
		    bblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (2), (1))] *
		    bblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (2), (2))] *
		    bblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (2), (3))] *
		    bblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (2), (4))] *
		    bblock[CALC_2D_IDX(5, 5, (4), (j))];
		cblock[CALC_2D_IDX(5, 5, (3), (j))] =
		    cblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (3), (0))] *
		    bblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (3), (1))] *
		    bblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (3), (2))] *
		    bblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (3), (3))] *
		    bblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (3), (4))] *
		    bblock[CALC_2D_IDX(5, 5, (4), (j))];
		cblock[CALC_2D_IDX(5, 5, (4), (j))] =
		    cblock[CALC_2D_IDX(5, 5, (4), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (4), (0))] *
		    bblock[CALC_2D_IDX(5, 5, (0), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (4), (1))] *
		    bblock[CALC_2D_IDX(5, 5, (1), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (4), (2))] *
		    bblock[CALC_2D_IDX(5, 5, (2), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (4), (3))] *
		    bblock[CALC_2D_IDX(5, 5, (3), (j))] -
		    ablock[CALC_2D_IDX(5, 5, (4), (4))] *
		    bblock[CALC_2D_IDX(5, 5, (4), (j))];
	}

}

//-------------------------------------------------------------------------------
//This is an alias of function: binvrhs
//The input arguments of this function are expanded. 
//Global memory variables:
//      0: lhs
//      1: r
//-------------------------------------------------------------------------------
void binvrhs_p0_p1(__global double *lhs, __global double *r,
		   unsigned arg_0_offset, unsigned arg_1_offset)
{
	lhs += arg_0_offset;
	r += arg_1_offset;

	double pivot, coeff;
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (0), (0))];
	lhs[CALC_2D_IDX(5, 5, (0), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (1))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (2))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] * pivot;
	r[(0)] = r[(0)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (0))];
	lhs[CALC_2D_IDX(5, 5, (1), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (1), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	r[(1)] = r[(1)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (0))];
	lhs[CALC_2D_IDX(5, 5, (2), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (2), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	r[(2)] = r[(2)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (0))];
	lhs[CALC_2D_IDX(5, 5, (3), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (3), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	r[(3)] = r[(3)] - coeff * r[(0)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (0))];
	lhs[CALC_2D_IDX(5, 5, (4), (1))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (1))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (4), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (0), (4))];
	r[(4)] = r[(4)] - coeff * r[(0)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (1), (1))];
	lhs[CALC_2D_IDX(5, 5, (1), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (2))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] * pivot;
	r[(1)] = r[(1)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (1))];
	lhs[CALC_2D_IDX(5, 5, (0), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	r[(0)] = r[(0)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (1))];
	lhs[CALC_2D_IDX(5, 5, (2), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	r[(2)] = r[(2)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (1))];
	lhs[CALC_2D_IDX(5, 5, (3), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	r[(3)] = r[(3)] - coeff * r[(1)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (1))];
	lhs[CALC_2D_IDX(5, 5, (4), (2))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (2))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (1), (4))];
	r[(4)] = r[(4)] - coeff * r[(1)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (2), (2))];
	lhs[CALC_2D_IDX(5, 5, (2), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (3))] * pivot;
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] * pivot;
	r[(2)] = r[(2)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (2))];
	lhs[CALC_2D_IDX(5, 5, (0), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	r[(0)] = r[(0)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (2))];
	lhs[CALC_2D_IDX(5, 5, (1), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	r[(1)] = r[(1)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (2))];
	lhs[CALC_2D_IDX(5, 5, (3), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	r[(3)] = r[(3)] - coeff * r[(2)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (2))];
	lhs[CALC_2D_IDX(5, 5, (4), (3))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (3))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (2), (4))];
	r[(4)] = r[(4)] - coeff * r[(2)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (3), (3))];
	lhs[CALC_2D_IDX(5, 5, (3), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (3), (4))] * pivot;
	r[(3)] = r[(3)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (3))];
	lhs[CALC_2D_IDX(5, 5, (0), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (0), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	r[(0)] = r[(0)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (3))];
	lhs[CALC_2D_IDX(5, 5, (1), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (1), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	r[(1)] = r[(1)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (3))];
	lhs[CALC_2D_IDX(5, 5, (2), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (2), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	r[(2)] = r[(2)] - coeff * r[(3)];
	coeff = lhs[CALC_2D_IDX(5, 5, (4), (3))];
	lhs[CALC_2D_IDX(5, 5, (4), (4))] =
	    lhs[CALC_2D_IDX(5, 5, (4), (4))] -
	    coeff * lhs[CALC_2D_IDX(5, 5, (3), (4))];
	r[(4)] = r[(4)] - coeff * r[(3)];
	pivot = 1.00 / lhs[CALC_2D_IDX(5, 5, (4), (4))];
	r[(4)] = r[(4)] * pivot;
	coeff = lhs[CALC_2D_IDX(5, 5, (0), (4))];
	r[(0)] = r[(0)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (1), (4))];
	r[(1)] = r[(1)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (2), (4))];
	r[(2)] = r[(2)] - coeff * r[(4)];
	coeff = lhs[CALC_2D_IDX(5, 5, (3), (4))];
	r[(3)] = r[(3)] - coeff * r[(4)];

}

//-------------------------------------------------------------------------------
//Functions (END)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//OpenCL Kernels (BEGIN)
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
//Loop defined at line 204 of bt.c
//-------------------------------------------------------------------------------
__kernel void add_0(__global double *u, __global double *rhs, int __ocl_i_bound,
		    int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	int m;			/* Defined at bt.c : 201 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] +
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 325 of bt.c
//-------------------------------------------------------------------------------
__kernel void exact_rhs_0(__global double *forcing, int __ocl_i_bound,
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
	int m;			/* Defined at bt.c : 319 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			forcing[CALC_4D_IDX(25, 25, 25, 6, (i), (j), (k), (m))]
			    = 0.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 339 of bt.c
//-------------------------------------------------------------------------------
__kernel void exact_rhs_1(double dnym1, __global int *grid_points, double dnzm1,
			  double dnxm1, __global double *forcing, double tx2,
			  double dx1tx1, double c2, double xxcon1,
			  double dx2tx1, double xxcon2, double dx3tx1,
			  double dx4tx1, double c1, double xxcon3,
			  double xxcon4, double xxcon5, double dx5tx1,
			  double dssp, __global double *ce, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	if (!(j < __ocl_j_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double eta;		/* Defined at bt.c : 318 */
	int k;			/* Defined at bt.c : 319 */
	double zeta;		/* Defined at bt.c : 318 */
	int i;			/* Defined at bt.c : 319 */
	double xi;		/* Defined at bt.c : 318 */
	double dtemp[5];	/* Defined at bt.c : 318 */
	int m;			/* Defined at bt.c : 319 */
	double ue[24][5];	/* threadprivate: defined at ./header.h : 75 */
	double dtpp;		/* Defined at bt.c : 318 */
	double buf[24][5];	/* threadprivate: defined at ./header.h : 76 */
	double cuf[24];		/* threadprivate: defined at ./header.h : 73 */
	double q[24];		/* threadprivate: defined at ./header.h : 74 */
	int im1;		/* Defined at bt.c : 319 */
	int ip1;		/* Defined at bt.c : 319 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		for (k = 1; k < grid_points[(2)] - 1; k++) {
			zeta = (double)k *dnzm1;
			for (i = 0; i < grid_points[(0)]; i++) {
				xi = (double)i *dnxm1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Exp: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[i][m] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[i][m] = dtpp * dtemp[m];
				}
				cuf[i] = buf[i][1] * buf[i][1];
				buf[i][0] =
				    cuf[i] + buf[i][2] * buf[i][2] +
				    buf[i][3] * buf[i][3];
				q[i] =
				    0.5 * (buf[i][1] * ue[i][1] +
					   buf[i][2] * ue[i][2] +
					   buf[i][3] * ue[i][3]);
			}
			for (i = 1; i < grid_points[(0)] - 1; i++) {
				im1 = i - 1;
				ip1 = i + 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (0))] - tx2 * (ue[ip1][1] -
							    ue[im1][1]) +
				    dx1tx1 * (ue[ip1][0] - 2.0 * ue[i][0] +
					      ue[im1][0]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (1))] -
				    tx2 *
				    ((ue[ip1][1] * buf[ip1][1] +
				      c2 * (ue[ip1][4] - q[ip1])) -
				     (ue[im1][1] * buf[im1][1] +
				      c2 * (ue[im1][4] - q[im1]))) +
				    xxcon1 * (buf[ip1][1] - 2.0 * buf[i][1] +
					      buf[im1][1]) +
				    dx2tx1 * (ue[ip1][1] - 2.0 * ue[i][1] +
					      ue[im1][1]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (2))] -
				    tx2 * (ue[ip1][2] * buf[ip1][1] -
					   ue[im1][2] * buf[im1][1]) +
				    xxcon2 * (buf[ip1][2] - 2.0 * buf[i][2] +
					      buf[im1][2]) +
				    dx3tx1 * (ue[ip1][2] - 2.0 * ue[i][2] +
					      ue[im1][2]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (3))] -
				    tx2 * (ue[ip1][3] * buf[ip1][1] -
					   ue[im1][3] * buf[im1][1]) +
				    xxcon2 * (buf[ip1][3] - 2.0 * buf[i][3] +
					      buf[im1][3]) +
				    dx4tx1 * (ue[ip1][3] - 2.0 * ue[i][3] +
					      ue[im1][3]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (4))] -
				    tx2 * (buf[ip1][1] *
					   (c1 * ue[ip1][4] - c2 * q[ip1]) -
					   buf[im1][1] * (c1 * ue[im1][4] -
							  c2 * q[im1])) +
				    0.5 * xxcon3 * (buf[ip1][0] -
						    2.0 * buf[i][0] +
						    buf[im1][0]) +
				    xxcon4 * (cuf[ip1] - 2.0 * cuf[i] +
					      cuf[im1]) +
				    xxcon5 * (buf[ip1][4] - 2.0 * buf[i][4] +
					      buf[im1][4]) +
				    dx5tx1 * (ue[ip1][4] - 2.0 * ue[i][4] +
					      ue[im1][4]);
			}
			for (m = 0; m < 5; m++) {
				i = 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (5.0 * ue[i][m] -
							     4.0 * ue[i +
								      1][m] +
							     ue[i + 2][m]);
				i = 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (-4.0 * ue[i -
								       1][m] +
							     6.0 * ue[i][m] -
							     4.0 * ue[i +
								      1][m] +
							     ue[i + 2][m]);
			}
			for (m = 0; m < 5; m++) {
				for (i = 1 * 3;
				     i <= grid_points[(0)] - 3 * 1 - 1; i++) {
					forcing[CALC_4D_IDX
						(25, 25, 25, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (25, 25, 25, 6, (i), (j),
						     (k),
						     (m))] - dssp * (ue[i -
									2][m] -
								     4.0 *
								     ue[i -
									1][m] +
								     6.0 *
								     ue[i][m] -
								     4.0 *
								     ue[i +
									1][m] +
								     ue[i +
									2][m]);
				}
			}
			for (m = 0; m < 5; m++) {
				i = grid_points[(0)] - 3;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[i - 2][m] -
							     4.0 * ue[i -
								      1][m] +
							     6.0 * ue[i][m] -
							     4.0 * ue[i +
								      1][m]);
				i = grid_points[(0)] - 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[i - 2][m] -
							     4.0 * ue[i -
								      1][m] +
							     5.0 * ue[i][m]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 440 of bt.c
//-------------------------------------------------------------------------------
__kernel void exact_rhs_2(double dnxm1, __global int *grid_points, double dnzm1,
			  double dnym1, __global double *forcing, double ty2,
			  double dy1ty1, double yycon2, double dy2ty1,
			  double c2, double yycon1, double dy3ty1,
			  double dy4ty1, double c1, double yycon3,
			  double yycon4, double yycon5, double dy5ty1,
			  double dssp, __global double *ce, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double xi;		/* Defined at bt.c : 318 */
	int k;			/* Defined at bt.c : 319 */
	double zeta;		/* Defined at bt.c : 318 */
	int j;			/* Defined at bt.c : 319 */
	double eta;		/* Defined at bt.c : 318 */
	double dtemp[5];	/* Defined at bt.c : 318 */
	int m;			/* Defined at bt.c : 319 */
	double ue[24][5];	/* threadprivate: defined at ./header.h : 75 */
	double dtpp;		/* Defined at bt.c : 318 */
	double buf[24][5];	/* threadprivate: defined at ./header.h : 76 */
	double cuf[24];		/* threadprivate: defined at ./header.h : 73 */
	double q[24];		/* threadprivate: defined at ./header.h : 74 */
	int jm1;		/* Defined at bt.c : 319 */
	int jp1;		/* Defined at bt.c : 319 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (k = 1; k < grid_points[(2)] - 1; k++) {
			zeta = (double)k *dnzm1;
			for (j = 0; j < grid_points[(1)]; j++) {
				eta = (double)j *dnym1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Exp: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[j][m] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[j][m] = dtpp * dtemp[m];
				}
				cuf[j] = buf[j][2] * buf[j][2];
				buf[j][0] =
				    cuf[j] + buf[j][1] * buf[j][1] +
				    buf[j][3] * buf[j][3];
				q[j] =
				    0.5 * (buf[j][1] * ue[j][1] +
					   buf[j][2] * ue[j][2] +
					   buf[j][3] * ue[j][3]);
			}
			for (j = 1; j < grid_points[(1)] - 1; j++) {
				jm1 = j - 1;
				jp1 = j + 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (0))] - ty2 * (ue[jp1][2] -
							    ue[jm1][2]) +
				    dy1ty1 * (ue[jp1][0] - 2.0 * ue[j][0] +
					      ue[jm1][0]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (1))] -
				    ty2 * (ue[jp1][1] * buf[jp1][2] -
					   ue[jm1][1] * buf[jm1][2]) +
				    yycon2 * (buf[jp1][1] - 2.0 * buf[j][1] +
					      buf[jm1][1]) +
				    dy2ty1 * (ue[jp1][1] - 2.0 * ue[j][1] +
					      ue[jm1][1]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (2))] -
				    ty2 *
				    ((ue[jp1][2] * buf[jp1][2] +
				      c2 * (ue[jp1][4] - q[jp1])) -
				     (ue[jm1][2] * buf[jm1][2] +
				      c2 * (ue[jm1][4] - q[jm1]))) +
				    yycon1 * (buf[jp1][2] - 2.0 * buf[j][2] +
					      buf[jm1][2]) +
				    dy3ty1 * (ue[jp1][2] - 2.0 * ue[j][2] +
					      ue[jm1][2]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (3))] -
				    ty2 * (ue[jp1][3] * buf[jp1][2] -
					   ue[jm1][3] * buf[jm1][2]) +
				    yycon2 * (buf[jp1][3] - 2.0 * buf[j][3] +
					      buf[jm1][3]) +
				    dy4ty1 * (ue[jp1][3] - 2.0 * ue[j][3] +
					      ue[jm1][3]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (4))] -
				    ty2 * (buf[jp1][2] *
					   (c1 * ue[jp1][4] - c2 * q[jp1]) -
					   buf[jm1][2] * (c1 * ue[jm1][4] -
							  c2 * q[jm1])) +
				    0.5 * yycon3 * (buf[jp1][0] -
						    2.0 * buf[j][0] +
						    buf[jm1][0]) +
				    yycon4 * (cuf[jp1] - 2.0 * cuf[j] +
					      cuf[jm1]) +
				    yycon5 * (buf[jp1][4] - 2.0 * buf[j][4] +
					      buf[jm1][4]) +
				    dy5ty1 * (ue[jp1][4] - 2.0 * ue[j][4] +
					      ue[jm1][4]);
			}
			for (m = 0; m < 5; m++) {
				j = 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (5.0 * ue[j][m] -
							     4.0 * ue[j +
								      1][m] +
							     ue[j + 2][m]);
				j = 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (-4.0 * ue[j -
								       1][m] +
							     6.0 * ue[j][m] -
							     4.0 * ue[j +
								      1][m] +
							     ue[j + 2][m]);
			}
			for (m = 0; m < 5; m++) {
				for (j = 1 * 3;
				     j <= grid_points[(1)] - 3 * 1 - 1; j++) {
					forcing[CALC_4D_IDX
						(25, 25, 25, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (25, 25, 25, 6, (i), (j),
						     (k),
						     (m))] - dssp * (ue[j -
									2][m] -
								     4.0 *
								     ue[j -
									1][m] +
								     6.0 *
								     ue[j][m] -
								     4.0 *
								     ue[j +
									1][m] +
								     ue[j +
									2][m]);
				}
			}
			for (m = 0; m < 5; m++) {
				j = grid_points[(1)] - 3;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[j - 2][m] -
							     4.0 * ue[j -
								      1][m] +
							     6.0 * ue[j][m] -
							     4.0 * ue[j +
								      1][m]);
				j = grid_points[(1)] - 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[j - 2][m] -
							     4.0 * ue[j -
								      1][m] +
							     5.0 * ue[j][m]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 542 of bt.c
//-------------------------------------------------------------------------------
__kernel void exact_rhs_3(double dnxm1, __global int *grid_points, double dnym1,
			  double dnzm1, __global double *forcing, double tz2,
			  double dz1tz1, double zzcon2, double dz2tz1,
			  double dz3tz1, double c2, double zzcon1,
			  double dz4tz1, double c1, double zzcon3,
			  double zzcon4, double zzcon5, double dz5tz1,
			  double dssp, __global double *ce, int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	double xi;		/* Defined at bt.c : 318 */
	int j;			/* Defined at bt.c : 319 */
	double eta;		/* Defined at bt.c : 318 */
	int k;			/* Defined at bt.c : 319 */
	double zeta;		/* Defined at bt.c : 318 */
	double dtemp[5];	/* Defined at bt.c : 318 */
	int m;			/* Defined at bt.c : 319 */
	double ue[24][5];	/* threadprivate: defined at ./header.h : 75 */
	double dtpp;		/* Defined at bt.c : 318 */
	double buf[24][5];	/* threadprivate: defined at ./header.h : 76 */
	double cuf[24];		/* threadprivate: defined at ./header.h : 73 */
	double q[24];		/* threadprivate: defined at ./header.h : 74 */
	int km1;		/* Defined at bt.c : 319 */
	int kp1;		/* Defined at bt.c : 319 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (j = 1; j < grid_points[(1)] - 1; j++) {
			eta = (double)j *dnym1;
			for (k = 0; k < grid_points[(2)]; k++) {
				zeta = (double)k *dnzm1;
				exact_solution_g4(xi, eta, zeta, dtemp,
						  ce) /*Arg Exp: ce */ ;
				for (m = 0; m < 5; m++) {
					ue[k][m] = dtemp[m];
				}
				dtpp = 1.0 / dtemp[0];
				for (m = 1; m <= 4; m++) {
					buf[k][m] = dtpp * dtemp[m];
				}
				cuf[k] = buf[k][3] * buf[k][3];
				buf[k][0] =
				    cuf[k] + buf[k][1] * buf[k][1] +
				    buf[k][2] * buf[k][2];
				q[k] =
				    0.5 * (buf[k][1] * ue[k][1] +
					   buf[k][2] * ue[k][2] +
					   buf[k][3] * ue[k][3]);
			}
			for (k = 1; k < grid_points[(2)] - 1; k++) {
				km1 = k - 1;
				kp1 = k + 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (0))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (0))] - tz2 * (ue[kp1][3] -
							    ue[km1][3]) +
				    dz1tz1 * (ue[kp1][0] - 2.0 * ue[k][0] +
					      ue[km1][0]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (1))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (1))] -
				    tz2 * (ue[kp1][1] * buf[kp1][3] -
					   ue[km1][1] * buf[km1][3]) +
				    zzcon2 * (buf[kp1][1] - 2.0 * buf[k][1] +
					      buf[km1][1]) +
				    dz2tz1 * (ue[kp1][1] - 2.0 * ue[k][1] +
					      ue[km1][1]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (2))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (2))] -
				    tz2 * (ue[kp1][2] * buf[kp1][3] -
					   ue[km1][2] * buf[km1][3]) +
				    zzcon2 * (buf[kp1][2] - 2.0 * buf[k][2] +
					      buf[km1][2]) +
				    dz3tz1 * (ue[kp1][2] - 2.0 * ue[k][2] +
					      ue[km1][2]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (3))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (3))] -
				    tz2 *
				    ((ue[kp1][3] * buf[kp1][3] +
				      c2 * (ue[kp1][4] - q[kp1])) -
				     (ue[km1][3] * buf[km1][3] +
				      c2 * (ue[km1][4] - q[km1]))) +
				    zzcon1 * (buf[kp1][3] - 2.0 * buf[k][3] +
					      buf[km1][3]) +
				    dz4tz1 * (ue[kp1][3] - 2.0 * ue[k][3] +
					      ue[km1][3]);
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (4))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (4))] -
				    tz2 * (buf[kp1][3] *
					   (c1 * ue[kp1][4] - c2 * q[kp1]) -
					   buf[km1][3] * (c1 * ue[km1][4] -
							  c2 * q[km1])) +
				    0.5 * zzcon3 * (buf[kp1][0] -
						    2.0 * buf[k][0] +
						    buf[km1][0]) +
				    zzcon4 * (cuf[kp1] - 2.0 * cuf[k] +
					      cuf[km1]) +
				    zzcon5 * (buf[kp1][4] - 2.0 * buf[k][4] +
					      buf[km1][4]) +
				    dz5tz1 * (ue[kp1][4] - 2.0 * ue[k][4] +
					      ue[km1][4]);
			}
			for (m = 0; m < 5; m++) {
				k = 1;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (5.0 * ue[k][m] -
							     4.0 * ue[k +
								      1][m] +
							     ue[k + 2][m]);
				k = 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (-4.0 * ue[k -
								       1][m] +
							     6.0 * ue[k][m] -
							     4.0 * ue[k +
								      1][m] +
							     ue[k + 2][m]);
			}
			for (m = 0; m < 5; m++) {
				for (k = 1 * 3;
				     k <= grid_points[(2)] - 3 * 1 - 1; k++) {
					forcing[CALC_4D_IDX
						(25, 25, 25, 6, (i), (j), (k),
						 (m))] =
					    forcing[CALC_4D_IDX
						    (25, 25, 25, 6, (i), (j),
						     (k),
						     (m))] - dssp * (ue[k -
									2][m] -
								     4.0 *
								     ue[k -
									1][m] +
								     6.0 *
								     ue[k][m] -
								     4.0 *
								     ue[k +
									1][m] +
								     ue[k +
									2][m]);
				}
			}
			for (m = 0; m < 5; m++) {
				k = grid_points[(2)] - 3;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[k - 2][m] -
							     4.0 * ue[k -
								      1][m] +
							     6.0 * ue[k][m] -
							     4.0 * ue[k +
								      1][m]);
				k = grid_points[(2)] - 2;
				forcing[CALC_4D_IDX
					(25, 25, 25, 6, (i), (j), (k), (m))] =
				    forcing[CALC_4D_IDX
					    (25, 25, 25, 6, (i), (j), (k),
					     (m))] - dssp * (ue[k - 2][m] -
							     4.0 * ue[k -
								      1][m] +
							     5.0 * ue[k][m]);
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 643 of bt.c
//-------------------------------------------------------------------------------
__kernel void exact_rhs_4(__global double *forcing, int __ocl_i_bound,
			  int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	int m;			/* Defined at bt.c : 319 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			forcing[CALC_4D_IDX(25, 25, 25, 6, (i), (j), (k), (m))]
			    =
			    -1.0 *
			    forcing[CALC_4D_IDX
				    (25, 25, 25, 6, (i), (j), (k), (m))];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 705 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_0(__global double *u)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	if (!(i < 24)) {
		return;
	}
	if (!(j < 24)) {
		return;
	}
	if (!(k < 24)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] = 1.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 722 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_1(double dnxm1, __global int *grid_points,
			   double dnym1, double dnzm1, __global double *u,
			   __global double *ce, int __ocl_i_bound)
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
	double xi;		/* Defined at bt.c : 695 */
	int j;			/* Defined at bt.c : 694 */
	double eta;		/* Defined at bt.c : 695 */
	int k;			/* Defined at bt.c : 694 */
	double zeta;		/* Defined at bt.c : 695 */
	int ix;			/* Defined at bt.c : 694 */
	double Pface[2][3][5];	/* Defined at bt.c : 695 */
	int iy;			/* Defined at bt.c : 694 */
	int iz;			/* Defined at bt.c : 694 */
	int m;			/* Defined at bt.c : 694 */
	double Pxi;		/* Defined at bt.c : 695 */
	double Peta;		/* Defined at bt.c : 695 */
	double Pzeta;		/* Defined at bt.c : 695 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			for (k = 0; k < grid_points[(2)]; k++) {
				zeta = (double)k *dnzm1;
				for (ix = 0; ix < 2; ix++) {
					exact_solution_g4((double)ix, eta, zeta,
							  &(Pface[ix][0][0]),
							  ce) /*Arg Exp: ce */ ;
				}
				for (iy = 0; iy < 2; iy++) {
					exact_solution_g4(xi, (double)iy, zeta,
							  &Pface[iy][1][0],
							  ce) /*Arg Exp: ce */ ;
				}
				for (iz = 0; iz < 2; iz++) {
					exact_solution_g4(xi, eta, (double)iz,
							  &Pface[iz][2][0],
							  ce) /*Arg Exp: ce */ ;
				}
				for (m = 0; m < 5; m++) {
					Pxi =
					    xi * Pface[1][0][m] + (1.0 -
								   xi) *
					    Pface[0][0][m];
					Peta =
					    eta * Pface[1][1][m] + (1.0 -
								    eta) *
					    Pface[0][1][m];
					Pzeta =
					    zeta * Pface[1][2][m] + (1.0 -
								     zeta) *
					    Pface[0][2][m];
					u[CALC_4D_IDX
					  (25, 25, 25, 5, (i), (j), (k), (m))] =
				 Pxi + Peta + Pzeta - Pxi * Peta - Pxi * Pzeta -
				 Peta * Pzeta + Pxi * Peta * Pzeta;
				}
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 774 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_2(double dnym1, __global int *grid_points,
			   double dnzm1, double xi, __global double *u, int i,
			   __global double *ce, int __ocl_j_bound)
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
	double eta;		/* Defined at bt.c : 695 */
	int k;			/* Defined at bt.c : 694 */
	double zeta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Exp: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (25, 25, 25, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 794 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_3(double dnym1, double dnzm1, double xi,
			   __global double *u, int i, __global double *ce,
			   int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0);
	int k = get_global_id(1);
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
	double eta;		/* Defined at bt.c : 695 */
	double zeta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		eta = (double)j *dnym1;
		zeta = (double)k *dnzm1;
		exact_solution_g4(xi, eta, zeta, temp, ce) /*Arg Exp: ce */ ;
		for (m = 0; m < 5; m++) {
			u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    temp[m];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 813 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_4(double dnxm1, __global int *grid_points,
			   double dnzm1, double eta, __global double *u, int j,
			   __global double *ce, int __ocl_i_bound)
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
	double xi;		/* Defined at bt.c : 695 */
	int k;			/* Defined at bt.c : 694 */
	double zeta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Exp: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (25, 25, 25, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 832 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_5(double dnxm1, __global int *grid_points,
			   double dnzm1, double eta, __global double *u, int j,
			   __global double *ce, int __ocl_i_bound)
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
	double xi;		/* Defined at bt.c : 695 */
	int k;			/* Defined at bt.c : 694 */
	double zeta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (k = 0; k < grid_points[(2)]; k++) {
			zeta = (double)k *dnzm1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Exp: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (25, 25, 25, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 851 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_6(double dnxm1, __global int *grid_points,
			   double dnym1, double zeta, __global double *u, int k,
			   __global double *ce, int __ocl_i_bound)
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
	double xi;		/* Defined at bt.c : 695 */
	int j;			/* Defined at bt.c : 694 */
	double eta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Exp: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (25, 25, 25, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 870 of bt.c
//-------------------------------------------------------------------------------
__kernel void initialize_7(double dnxm1, __global int *grid_points,
			   double dnym1, double zeta, __global double *u, int k,
			   __global double *ce, int __ocl_i_bound)
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
	double xi;		/* Defined at bt.c : 695 */
	int j;			/* Defined at bt.c : 694 */
	double eta;		/* Defined at bt.c : 695 */
	double temp[5];		/* Defined at bt.c : 695 */
	int m;			/* Defined at bt.c : 694 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		xi = (double)i *dnxm1;
		for (j = 0; j < grid_points[(1)]; j++) {
			eta = (double)j *dnym1;
			exact_solution_g4(xi, eta, zeta, temp,
					  ce) /*Arg Exp: ce */ ;
			for (m = 0; m < 5; m++) {
				u[CALC_4D_IDX
				  (25, 25, 25, 5, (i), (j), (k), (m))] =
			 temp[m];
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 898 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsinit_0(__global double *lhs, int __ocl_i_bound,
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
	int m;			/* Defined at bt.c : 888 */
	int n;			/* Defined at bt.c : 888 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			for (n = 0; n < 5; n++) {
				lhs[CALC_6D_IDX
				    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0),
				     (m), (n))] = 0.0;
				lhs[CALC_6D_IDX
				    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1),
				     (m), (n))] = 0.0;
				lhs[CALC_6D_IDX
				    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2),
				     (m), (n))] = 0.0;
			}
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 918 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsinit_1(__global double *lhs, int __ocl_i_bound,
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
	int m;			/* Defined at bt.c : 888 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (m),
			     (m))] = 1.0;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 949 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsx_0(__global int *grid_points, __global double *u,
		     __global double *fjac, double c2, double c1,
		     __global double *njac, double con43, double c3c4,
		     double c1345, double dt, double tx1, double tx2,
		     __global double *lhs, double dx1, double dx2, double dx3,
		     double dx4, double dx5, int __ocl_j_bound,
		     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
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
	int i;			/* Defined at bt.c : 942 */
	double tmp1;		/* threadprivate: defined at ./header.h : 91 */
	double tmp2;		/* threadprivate: defined at ./header.h : 91 */
	double tmp3;		/* threadprivate: defined at ./header.h : 91 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = 0; i < grid_points[(0)]; i++) {
			//-------------------------------------------
			//Declare prefetching Buffers (BEGIN) : 951
			//-------------------------------------------
			double4 u_0;
			double u_1;
			//-------------------------------------------
			//Declare prefetching buffers (END)
			//-------------------------------------------
			//-------------------------------------------
			//Prefetching (BEGIN) : 951
			//Candidates:
			//      u[i][j][k][0]
			//      u[i][j][k][1]
			//      u[i][j][k][2]
			//      u[i][j][k][3]
			//      u[i][j][k][4]
			//-------------------------------------------
			__global double *p_u_0_0 =
			    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0));
			if ((unsigned long)p_u_0_0 % 64 == 0) {
				u_0 = vload4(0, p_u_0_0);
			} else {
				u_0.x = p_u_0_0[0];
				p_u_0_0++;
				u_0.y = p_u_0_0[0];
				p_u_0_0++;
				u_0.z = p_u_0_0[0];
				p_u_0_0++;
				u_0.w = p_u_0_0[0];
				p_u_0_0++;
			}
			u_1 = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
			//-------------------------------------------
			//Prefetching (END)
			//-------------------------------------------

			tmp1 = 1.0 / u_0.x;
			tmp2 = tmp1 * tmp1;
			tmp3 = tmp1 * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] = 1.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
			    -(u_0.y * tmp2 * u_0.y) +
			    c2 * 0.50 * (u_0.y * u_0.y + u_0.z * u_0.z +
					 u_0.w * u_0.w) * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
			    (2.0 - c2) * (u_0.y / u_0.x);
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] =
			    -c2 * (u_0.z * tmp1);
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] =
			    -c2 * (u_0.w * tmp1);
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] = c2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
			    -(u_0.y * u_0.z) * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] =
			    u_0.z * tmp1;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
			    u_0.y * tmp1;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
			    -(u_0.y * u_0.w) * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] =
			    u_0.w * tmp1;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
			    u_0.y * tmp1;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] = 0.0;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
			    (c2 *
			     (u_0.y * u_0.y + u_0.z * u_0.z +
			      u_0.w * u_0.w) * tmp2 -
			     c1 * (u_1 * tmp1)) * (u_0.y * tmp1);
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
			    c1 * u_1 * tmp1 - 0.50 * c2 * (3.0 * u_0.y * u_0.y +
							   u_0.z * u_0.z +
							   u_0.w * u_0.w) *
			    tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
			    -c2 * (u_0.z * u_0.y) * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
			    -c2 * (u_0.w * u_0.y) * tmp2;
			fjac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
			    c1 * (u_0.y * tmp1);
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
			    -con43 * c3c4 * tmp2 * u_0.y;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
			    con43 * c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
			    -c3c4 * tmp2 * u_0.z;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
			    c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
			    -c3c4 * tmp2 * u_0.w;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
			    c3c4 * tmp1;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] = 0.0;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
			    -(con43 * c3c4 -
			      c1345) * tmp3 * (((u_0.y) * (u_0.y))) - (c3c4 -
								       c1345) *
			    tmp3 * (((u_0.z) * (u_0.z))) - (c3c4 -
							    c1345) * tmp3 *
			    (((u_0.w) * (u_0.w))) - c1345 * tmp2 * u_1;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
			    (con43 * c3c4 - c1345) * tmp2 * u_0.y;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
			    (c3c4 - c1345) * tmp2 * u_0.z;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
			    (c3c4 - c1345) * tmp2 * u_0.w;
			njac[CALC_5D_IDX
			     (24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
			    (c1345) * tmp1;
		}
		for (i = 1; i < grid_points[(0)] - 1; i++) {
			tmp1 = dt * tx1;
			tmp2 = dt * tx2;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (0),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (0))] -
			    tmp1 * dx1;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (0),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (0),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (0),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (0),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (0),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (1),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (1),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (1))] -
			    tmp1 * dx2;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (1),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (1),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (1),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (1),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (2),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (2),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (2),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (2))] -
			    tmp1 * dx3;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (2),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (2),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (2),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (3),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (3),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (3),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (3),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (3))] -
			    tmp1 * dx4;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (3),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (3),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4),
			     (0))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (4),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4),
			     (1))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (4),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4),
			     (2))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (4),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4),
			     (3))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (4),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4),
			     (4))] =
			    -tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i - 1), (j), (k), (4),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i - 1),
								  (j), (k), (4),
								  (4))] -
			    tmp1 * dx5;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0),
			     (0))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (0),
				  (0))] + tmp1 * 2.0 * dx1;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (0), (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (0), (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (0), (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (0), (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (1), (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1),
			     (1))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (1),
				  (1))] + tmp1 * 2.0 * dx2;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (1), (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (1), (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (1), (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (2), (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (2), (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2),
			     (2))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (2),
				  (2))] + tmp1 * 2.0 * dx3;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (2), (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (2), (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (3), (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (3), (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (3), (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3),
			     (3))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (3),
				  (3))] + tmp1 * 2.0 * dx4;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3),
			     (4))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (3), (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4),
			     (0))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (4), (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4),
			     (1))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (4), (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4),
			     (2))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (4), (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4),
			     (3))] =
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (4), (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4),
			     (4))] =
			    1.0 +
			    tmp1 * 2.0 *
			    njac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i), (j), (k), (4),
				  (4))] + tmp1 * 2.0 * dx5;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (0),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (0))] -
			    tmp1 * dx1;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (0),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (0),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (0),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (0),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (0),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (1),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (1),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (1))] -
			    tmp1 * dx2;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (1),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (1),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (1),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (1),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (2),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (2),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (2),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (2))] -
			    tmp1 * dx3;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (2),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (2),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (2),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (3),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (3),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (3),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (3),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (3))] -
			    tmp1 * dx4;
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (3),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (3),
								  (4))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4),
			     (0))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (4),
				  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (0))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4),
			     (1))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (4),
				  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (1))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4),
			     (2))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (4),
				  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (2))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4),
			     (3))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (4),
				  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (3))];
			lhs[CALC_6D_IDX
			    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4),
			     (4))] =
			    tmp2 *
			    fjac[CALC_5D_IDX
				 (24, 24, 23, 5, 5, (i + 1), (j), (k), (4),
				  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5,
								  5, (i + 1),
								  (j), (k), (4),
								  (4))] -
			    tmp1 * dx5;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1235 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsy_0(__global double *u, __global double *fjac, double c2,
		     double c1, __global double *njac, double c3c4,
		     double con43, double c1345, int __ocl_i_bound,
		     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1);
	int k = get_global_id(2) + 1;
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
	double tmp1;		/* threadprivate: defined at ./header.h : 91 */
	double tmp2;		/* threadprivate: defined at ./header.h : 91 */
	double tmp3;		/* threadprivate: defined at ./header.h : 91 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1237
		//-------------------------------------------
		double4 u_2;
		double u_3;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1237
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//      u[i][j][k][4]
		//-------------------------------------------
		__global double *p_u_2_0 =
		    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0));
		if ((unsigned long)p_u_2_0 % 64 == 0) {
			u_2 = vload4(0, p_u_2_0);
		} else {
			u_2.x = p_u_2_0[0];
			p_u_2_0++;
			u_2.y = p_u_2_0[0];
			p_u_2_0++;
			u_2.z = p_u_2_0[0];
			p_u_2_0++;
			u_2.w = p_u_2_0[0];
			p_u_2_0++;
		}
		u_3 = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u_2.x;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] =
		    1.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
		    -(u_2.y * u_2.z) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
		    u_2.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] =
		    u_2.y * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
		    -(u_2.z * u_2.z * tmp2) +
		    0.50 * c2 *
		    ((u_2.y * u_2.y + u_2.z * u_2.z + u_2.w * u_2.w) * tmp2);
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] =
		    -c2 * u_2.y * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
		    (2.0 - c2) * u_2.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] =
		    -c2 * u_2.w * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] =
		    c2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
		    -(u_2.z * u_2.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] =
		    u_2.w * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
		    u_2.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
		    (c2 * (u_2.y * u_2.y + u_2.z * u_2.z + u_2.w * u_2.w) *
		     tmp2 - c1 * u_3 * tmp1) * u_2.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
		    -c2 * u_2.y * u_2.z * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
		    c1 * u_3 * tmp1 -
		    0.50 * c2 *
		    ((u_2.y * u_2.y + 3.0 * u_2.z * u_2.z +
		      u_2.w * u_2.w) * tmp2);
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
		    -c2 * (u_2.z * u_2.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
		    c1 * u_2.z * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
		    -c3c4 * tmp2 * u_2.y;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
		    -con43 * c3c4 * tmp2 * u_2.z;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
		    con43 * c3c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
		    -c3c4 * tmp2 * u_2.w;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
		    -(c3c4 - c1345) * tmp3 * (((u_2.y) * (u_2.y))) -
		    (con43 * c3c4 - c1345) * tmp3 * (((u_2.z) * (u_2.z))) -
		    (c3c4 - c1345) * tmp3 * (((u_2.w) * (u_2.w))) -
		    c1345 * tmp2 * u_3;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
		    (c3c4 - c1345) * tmp2 * u_2.y;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
		    (con43 * c3c4 - c1345) * tmp2 * u_2.z;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
		    (c3c4 - c1345) * tmp2 * u_2.w;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
		    (c1345) * tmp1;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1339 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsy_1(double dt, double ty1, double ty2, __global double *lhs,
		     __global double *fjac, __global double *njac, double dy1,
		     double dy2, double dy3, double dy4, double dy5,
		     int __ocl_i_bound, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	double tmp1;		/* threadprivate: defined at ./header.h : 91 */
	double tmp2;		/* threadprivate: defined at ./header.h : 91 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1341
		//-------------------------------------------
		double2 fjac_0[5];
		double2 njac_0[13];
		double4 njac_1[2];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1341
		//Candidates:
		//      fjac[i][j + 1][k][1][1]
		//      fjac[i][j + 1][k][1][2]
		//      fjac[i][j + 1][k][1][3]
		//      fjac[i][j + 1][k][1][4]
		//      fjac[i][j + 1][k][0][0]
		//      fjac[i][j + 1][k][0][1]
		//      fjac[i][j + 1][k][2][2]
		//      fjac[i][j + 1][k][2][3]
		//      fjac[i][j + 1][k][4][2]
		//      fjac[i][j + 1][k][4][3]
		//      njac[i][j - 1][k][0][0]
		//      njac[i][j - 1][k][0][1]
		//      njac[i][j][k][0][1]
		//      njac[i][j][k][0][2]
		//      njac[i][j - 1][k][2][0]
		//      njac[i][j - 1][k][2][1]
		//      njac[i][j][k][2][0]
		//      njac[i][j][k][2][1]
		//      njac[i][j + 1][k][2][1]
		//      njac[i][j + 1][k][3][0]
		//      njac[i][j + 1][k][3][0]
		//      njac[i][j + 1][k][3][1]
		//      njac[i][j + 1][k][4][0]
		//      njac[i][j + 1][k][4][1]
		//      njac[i][j - 1][k][3][0]
		//      njac[i][j - 1][k][3][1]
		//      njac[i][j - 1][k][1][3]
		//      njac[i][j - 1][k][1][4]
		//      njac[i][j][k][3][2]
		//      njac[i][j][k][3][3]
		//      njac[i][j + 1][k][2][3]
		//      njac[i][j + 1][k][2][4]
		//      njac[i][j - 1][k][4][1]
		//      njac[i][j - 1][k][4][2]
		//      njac[i][j + 1][k][4][3]
		//      njac[i][j + 1][k][4][4]
		//      njac[i][j + 1][k][0][1]
		//      njac[i][j + 1][k][0][2]
		//      njac[i][j + 1][k][0][3]
		//      njac[i][j + 1][k][0][4]
		//      njac[i][j + 1][k][1][0]
		//      njac[i][j + 1][k][1][1]
		//      njac[i][j + 1][k][1][2]
		//      njac[i][j + 1][k][1][3]
		//-------------------------------------------
		__global double *p_fjac_0_0 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (1),
				       (1));
		if ((unsigned long)p_fjac_0_0 % 64 == 0) {
			fjac_0[0] = vload2(0, p_fjac_0_0);
		} else {
			fjac_0[0].x = p_fjac_0_0[0];
			p_fjac_0_0++;
			fjac_0[0].y = p_fjac_0_0[0];
			p_fjac_0_0++;
		}
		__global double *p_fjac_0_1 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (1),
				       (3));
		if ((unsigned long)p_fjac_0_1 % 64 == 0) {
			fjac_0[1] = vload2(0, p_fjac_0_1);
		} else {
			fjac_0[1].x = p_fjac_0_1[0];
			p_fjac_0_1++;
			fjac_0[1].y = p_fjac_0_1[0];
			p_fjac_0_1++;
		}
		__global double *p_fjac_0_2 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
				       (0));
		if ((unsigned long)p_fjac_0_2 % 64 == 0) {
			fjac_0[2] = vload2(0, p_fjac_0_2);
		} else {
			fjac_0[2].x = p_fjac_0_2[0];
			p_fjac_0_2++;
			fjac_0[2].y = p_fjac_0_2[0];
			p_fjac_0_2++;
		}
		__global double *p_fjac_0_3 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
				       (2));
		if ((unsigned long)p_fjac_0_3 % 64 == 0) {
			fjac_0[3] = vload2(0, p_fjac_0_3);
		} else {
			fjac_0[3].x = p_fjac_0_3[0];
			p_fjac_0_3++;
			fjac_0[3].y = p_fjac_0_3[0];
			p_fjac_0_3++;
		}
		__global double *p_fjac_0_4 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
				       (2));
		if ((unsigned long)p_fjac_0_4 % 64 == 0) {
			fjac_0[4] = vload2(0, p_fjac_0_4);
		} else {
			fjac_0[4].x = p_fjac_0_4[0];
			p_fjac_0_4++;
			fjac_0[4].y = p_fjac_0_4[0];
			p_fjac_0_4++;
		}
		__global double *p_njac_0_0 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
				       (0));
		if ((unsigned long)p_njac_0_0 % 64 == 0) {
			njac_0[0] = vload2(0, p_njac_0_0);
		} else {
			njac_0[0].x = p_njac_0_0[0];
			p_njac_0_0++;
			njac_0[0].y = p_njac_0_0[0];
			p_njac_0_0++;
		}
		__global double *p_njac_0_1 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0),
				       (1));
		if ((unsigned long)p_njac_0_1 % 64 == 0) {
			njac_0[1] = vload2(0, p_njac_0_1);
		} else {
			njac_0[1].x = p_njac_0_1[0];
			p_njac_0_1++;
			njac_0[1].y = p_njac_0_1[0];
			p_njac_0_1++;
		}
		__global double *p_njac_0_2 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
				       (0));
		if ((unsigned long)p_njac_0_2 % 64 == 0) {
			njac_0[2] = vload2(0, p_njac_0_2);
		} else {
			njac_0[2].x = p_njac_0_2[0];
			p_njac_0_2++;
			njac_0[2].y = p_njac_0_2[0];
			p_njac_0_2++;
		}
		__global double *p_njac_0_3 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2),
				       (0));
		if ((unsigned long)p_njac_0_3 % 64 == 0) {
			njac_0[3] = vload2(0, p_njac_0_3);
		} else {
			njac_0[3].x = p_njac_0_3[0];
			p_njac_0_3++;
			njac_0[3].y = p_njac_0_3[0];
			p_njac_0_3++;
		}
		__global double *p_njac_0_4 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
				       (1));
		if ((unsigned long)p_njac_0_4 % 64 == 0) {
			njac_0[4] = vload2(0, p_njac_0_4);
		} else {
			njac_0[4].x = p_njac_0_4[0];
			p_njac_0_4++;
			njac_0[4].y = p_njac_0_4[0];
			p_njac_0_4++;
		}
		__global double *p_njac_0_5 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
				       (0));
		if ((unsigned long)p_njac_0_5 % 64 == 0) {
			njac_0[5] = vload2(0, p_njac_0_5);
		} else {
			njac_0[5].x = p_njac_0_5[0];
			p_njac_0_5++;
			njac_0[5].y = p_njac_0_5[0];
			p_njac_0_5++;
		}
		__global double *p_njac_0_6 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
				       (0));
		if ((unsigned long)p_njac_0_6 % 64 == 0) {
			njac_0[6] = vload2(0, p_njac_0_6);
		} else {
			njac_0[6].x = p_njac_0_6[0];
			p_njac_0_6++;
			njac_0[6].y = p_njac_0_6[0];
			p_njac_0_6++;
		}
		__global double *p_njac_0_7 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
				       (0));
		if ((unsigned long)p_njac_0_7 % 64 == 0) {
			njac_0[7] = vload2(0, p_njac_0_7);
		} else {
			njac_0[7].x = p_njac_0_7[0];
			p_njac_0_7++;
			njac_0[7].y = p_njac_0_7[0];
			p_njac_0_7++;
		}
		__global double *p_njac_0_8 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
				       (3));
		if ((unsigned long)p_njac_0_8 % 64 == 0) {
			njac_0[8] = vload2(0, p_njac_0_8);
		} else {
			njac_0[8].x = p_njac_0_8[0];
			p_njac_0_8++;
			njac_0[8].y = p_njac_0_8[0];
			p_njac_0_8++;
		}
		__global double *p_njac_0_9 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3),
				       (2));
		if ((unsigned long)p_njac_0_9 % 64 == 0) {
			njac_0[9] = vload2(0, p_njac_0_9);
		} else {
			njac_0[9].x = p_njac_0_9[0];
			p_njac_0_9++;
			njac_0[9].y = p_njac_0_9[0];
			p_njac_0_9++;
		}
		__global double *p_njac_0_10 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
				       (3));
		if ((unsigned long)p_njac_0_10 % 64 == 0) {
			njac_0[10] = vload2(0, p_njac_0_10);
		} else {
			njac_0[10].x = p_njac_0_10[0];
			p_njac_0_10++;
			njac_0[10].y = p_njac_0_10[0];
			p_njac_0_10++;
		}
		__global double *p_njac_0_11 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
				       (1));
		if ((unsigned long)p_njac_0_11 % 64 == 0) {
			njac_0[11] = vload2(0, p_njac_0_11);
		} else {
			njac_0[11].x = p_njac_0_11[0];
			p_njac_0_11++;
			njac_0[11].y = p_njac_0_11[0];
			p_njac_0_11++;
		}
		__global double *p_njac_0_12 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
				       (3));
		if ((unsigned long)p_njac_0_12 % 64 == 0) {
			njac_0[12] = vload2(0, p_njac_0_12);
		} else {
			njac_0[12].x = p_njac_0_12[0];
			p_njac_0_12++;
			njac_0[12].y = p_njac_0_12[0];
			p_njac_0_12++;
		}
		__global double *p_njac_1_0 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
				       (1));
		if ((unsigned long)p_njac_1_0 % 64 == 0) {
			njac_1[0] = vload4(0, p_njac_1_0);
		} else {
			njac_1[0].x = p_njac_1_0[0];
			p_njac_1_0++;
			njac_1[0].y = p_njac_1_0[0];
			p_njac_1_0++;
			njac_1[0].z = p_njac_1_0[0];
			p_njac_1_0++;
			njac_1[0].w = p_njac_1_0[0];
			p_njac_1_0++;
		}
		__global double *p_njac_1_1 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j + 1), (k), (1),
				       (0));
		if ((unsigned long)p_njac_1_1 % 64 == 0) {
			njac_1[1] = vload4(0, p_njac_1_1);
		} else {
			njac_1[1].x = p_njac_1_1[0];
			p_njac_1_1++;
			njac_1[1].y = p_njac_1_1[0];
			p_njac_1_1++;
			njac_1[1].z = p_njac_1_1[0];
			p_njac_1_1++;
			njac_1[1].w = p_njac_1_1[0];
			p_njac_1_1++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = dt * ty1;
		tmp2 = dt * ty2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
			  (0))] - tmp1 * njac_0[0].x - tmp1 * dy1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
			  (1))] - tmp1 * njac_0[0].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (1),
							  (1))] - tmp1 * dy2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
			  (3))] - tmp1 * njac_0[8].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (1),
			  (4))] - tmp1 * njac_0[8].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
			  (0))] - tmp1 * njac_0[2].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
			  (1))] - tmp1 * njac_0[2].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (2),
							  (2))] - tmp1 * dy3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
			  (0))] - tmp1 * njac_0[7].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
			  (1))] - tmp1 * njac_0[7].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (3),
							  (3))] - tmp1 * dy4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
			  (1))] - tmp1 * njac_0[11].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
			  (2))] - tmp1 * njac_0[11].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j - 1), (k), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j - 1), (k), (4),
							  (4))] - tmp1 * dy5;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (0))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))]
		    + tmp1 * 2.0 * dy1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (1))] =
		    tmp1 * 2.0 * njac_0[1].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (2))] =
		    tmp1 * 2.0 * njac_0[1].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (0), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (0), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (1))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))]
		    + tmp1 * 2.0 * dy2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (0))] =
		    tmp1 * 2.0 * njac_0[3].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (1))] =
		    tmp1 * 2.0 * njac_0[3].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (2))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))]
		    + tmp1 * 2.0 * dy3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (2), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (2), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (1))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (2))] =
		    tmp1 * 2.0 * njac_0[9].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (3))] =
		    1.0 + tmp1 * 2.0 * njac_0[9].y + tmp1 * 2.0 * dy4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (1))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (4))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))]
		    + tmp1 * 2.0 * dy5;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (0))] =
		    tmp2 * fjac_0[2].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
			  (0))] - tmp1 * dy1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (1))] =
		    tmp2 * fjac_0[2].y - tmp1 * njac_1[0].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
			  (2))] - tmp1 * njac_1[0].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
			  (3))] - tmp1 * njac_1[0].z;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (0),
			  (4))] - tmp1 * njac_1[0].w;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (1),
			  (0))] - tmp1 * njac_1[1].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (1))] =
		    tmp2 * fjac_0[0].x - tmp1 * njac_1[1].y - tmp1 * dy2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (2))] =
		    tmp2 * fjac_0[0].y - tmp1 * njac_1[1].z;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (3))] =
		    tmp2 * fjac_0[1].x - tmp1 * njac_1[1].w;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (4))] =
		    tmp2 * fjac_0[1].y -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (1), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j + 1), (k), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
			  (1))] - tmp1 * njac_0[4].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (2))] =
		    tmp2 * fjac_0[3].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
			  (2))] - tmp1 * dy3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (3))] =
		    tmp2 * fjac_0[3].y - tmp1 * njac_0[10].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (2),
			  (4))] - tmp1 * njac_0[10].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
			  (0))] - tmp1 * njac_0[4].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
			  (1))] - tmp1 * njac_0[5].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j + 1), (k), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j + 1), (k), (3),
							  (3))] - tmp1 * dy4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j + 1), (k), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
			  (0))] - tmp1 * njac_0[6].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
			  (1))] - tmp1 * njac_0[6].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (2))] =
		    tmp2 * fjac_0[4].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (4), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (3))] =
		    tmp2 * fjac_0[4].y - tmp1 * njac_0[12].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j + 1), (k), (4),
			  (4))] - tmp1 * njac_0[12].y - tmp1 * dy5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1533 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsz_0(__global double *u, __global double *fjac, double c2,
		     double c1, __global double *njac, double c3c4,
		     double con43, double c3, double c4, double c1345,
		     int __ocl_i_bound, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
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
	double tmp1;		/* threadprivate: defined at ./header.h : 91 */
	double tmp2;		/* threadprivate: defined at ./header.h : 91 */
	double tmp3;		/* threadprivate: defined at ./header.h : 91 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1535
		//-------------------------------------------
		double4 u_4;
		double u_5;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1535
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//      u[i][j][k][4]
		//-------------------------------------------
		__global double *p_u_4_0 =
		    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0));
		if ((unsigned long)p_u_4_0 % 64 == 0) {
			u_4 = vload4(0, p_u_4_0);
		} else {
			u_4.x = p_u_4_0[0];
			p_u_4_0++;
			u_4.y = p_u_4_0[0];
			p_u_4_0++;
			u_4.z = p_u_4_0[0];
			p_u_4_0++;
			u_4.w = p_u_4_0[0];
			p_u_4_0++;
		}
		u_5 = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = 1.0 / u_4.x;
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] =
		    1.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
		    -(u_4.y * u_4.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
		    u_4.w * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] =
		    u_4.y * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
		    -(u_4.z * u_4.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
		    u_4.w * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] =
		    u_4.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
		    -(u_4.w * u_4.w * tmp2) +
		    0.50 * c2 *
		    ((u_4.y * u_4.y + u_4.z * u_4.z + u_4.w * u_4.w) * tmp2);
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] =
		    -c2 * u_4.y * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] =
		    -c2 * u_4.z * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
		    (2.0 - c2) * u_4.w * tmp1;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] =
		    c2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
		    (c2 * (u_4.y * u_4.y + u_4.z * u_4.z + u_4.w * u_4.w) *
		     tmp2 - c1 * (u_5 * tmp1)) * (u_4.w * tmp1);
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
		    -c2 * (u_4.y * u_4.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
		    -c2 * (u_4.z * u_4.w) * tmp2;
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
		    c1 * (u_5 * tmp1) -
		    0.50 * c2 *
		    ((u_4.y * u_4.y + u_4.z * u_4.z +
		      3.0 * u_4.w * u_4.w) * tmp2);
		fjac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
		    c1 * u_4.w * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (0))] =
		    -c3c4 * tmp2 * u_4.y;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (0))] =
		    -c3c4 * tmp2 * u_4.z;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))] =
		    c3c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (3))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (0))] =
		    -con43 * c3c4 * tmp2 * u_4.w;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (1))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (2))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (3))] =
		    con43 * c3 * c4 * tmp1;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3), (4))] =
		    0.0;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (0))] =
		    -(c3c4 - c1345) * tmp3 * (((u_4.y) * (u_4.y))) - (c3c4 -
								      c1345) *
		    tmp3 * (((u_4.z) * (u_4.z))) - (con43 * c3c4 -
						    c1345) * tmp3 * (((u_4.w) *
								      (u_4.
								       w))) -
		    c1345 * tmp2 * u_5;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (1))] =
		    (c3c4 - c1345) * tmp2 * u_4.y;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (2))] =
		    (c3c4 - c1345) * tmp2 * u_4.z;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (3))] =
		    (con43 * c3c4 - c1345) * tmp2 * u_4.w;
		njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))] =
		    (c1345) * tmp1;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1637 of bt.c
//-------------------------------------------------------------------------------
__kernel void lhsz_1(double dt, double tz1, double tz2, __global double *lhs,
		     __global double *fjac, __global double *njac, double dz1,
		     double dz2, double dz3, double dz4, double dz5,
		     int __ocl_i_bound, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	double tmp1;		/* threadprivate: defined at ./header.h : 91 */
	double tmp2;		/* threadprivate: defined at ./header.h : 91 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1639
		//-------------------------------------------
		double2 fjac_1[5];
		double2 njac_2[13];
		double4 njac_3[2];
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1639
		//Candidates:
		//      fjac[i][j][k + 1][1][1]
		//      fjac[i][j][k + 1][1][2]
		//      fjac[i][j][k + 1][1][3]
		//      fjac[i][j][k + 1][1][4]
		//      fjac[i][j][k + 1][0][0]
		//      fjac[i][j][k + 1][0][1]
		//      fjac[i][j][k + 1][2][2]
		//      fjac[i][j][k + 1][2][3]
		//      fjac[i][j][k + 1][4][2]
		//      fjac[i][j][k + 1][4][3]
		//      njac[i][j][k - 1][0][0]
		//      njac[i][j][k - 1][0][1]
		//      njac[i][j][k][0][1]
		//      njac[i][j][k][0][2]
		//      njac[i][j][k - 1][2][0]
		//      njac[i][j][k - 1][2][1]
		//      njac[i][j][k][2][0]
		//      njac[i][j][k][2][1]
		//      njac[i][j][k + 1][2][1]
		//      njac[i][j][k + 1][3][0]
		//      njac[i][j][k + 1][3][0]
		//      njac[i][j][k + 1][3][1]
		//      njac[i][j][k + 1][4][0]
		//      njac[i][j][k + 1][4][1]
		//      njac[i][j][k - 1][3][0]
		//      njac[i][j][k - 1][3][1]
		//      njac[i][j][k - 1][1][3]
		//      njac[i][j][k - 1][1][4]
		//      njac[i][j][k][3][2]
		//      njac[i][j][k][3][3]
		//      njac[i][j][k + 1][2][3]
		//      njac[i][j][k + 1][2][4]
		//      njac[i][j][k - 1][4][1]
		//      njac[i][j][k - 1][4][2]
		//      njac[i][j][k + 1][4][3]
		//      njac[i][j][k + 1][4][4]
		//      njac[i][j][k + 1][0][1]
		//      njac[i][j][k + 1][0][2]
		//      njac[i][j][k + 1][0][3]
		//      njac[i][j][k + 1][0][4]
		//      njac[i][j][k + 1][1][0]
		//      njac[i][j][k + 1][1][1]
		//      njac[i][j][k + 1][1][2]
		//      njac[i][j][k + 1][1][3]
		//-------------------------------------------
		__global double *p_fjac_1_0 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (1),
				       (1));
		if ((unsigned long)p_fjac_1_0 % 64 == 0) {
			fjac_1[0] = vload2(0, p_fjac_1_0);
		} else {
			fjac_1[0].x = p_fjac_1_0[0];
			p_fjac_1_0++;
			fjac_1[0].y = p_fjac_1_0[0];
			p_fjac_1_0++;
		}
		__global double *p_fjac_1_1 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (1),
				       (3));
		if ((unsigned long)p_fjac_1_1 % 64 == 0) {
			fjac_1[1] = vload2(0, p_fjac_1_1);
		} else {
			fjac_1[1].x = p_fjac_1_1[0];
			p_fjac_1_1++;
			fjac_1[1].y = p_fjac_1_1[0];
			p_fjac_1_1++;
		}
		__global double *p_fjac_1_2 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
				       (0));
		if ((unsigned long)p_fjac_1_2 % 64 == 0) {
			fjac_1[2] = vload2(0, p_fjac_1_2);
		} else {
			fjac_1[2].x = p_fjac_1_2[0];
			p_fjac_1_2++;
			fjac_1[2].y = p_fjac_1_2[0];
			p_fjac_1_2++;
		}
		__global double *p_fjac_1_3 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
				       (2));
		if ((unsigned long)p_fjac_1_3 % 64 == 0) {
			fjac_1[3] = vload2(0, p_fjac_1_3);
		} else {
			fjac_1[3].x = p_fjac_1_3[0];
			p_fjac_1_3++;
			fjac_1[3].y = p_fjac_1_3[0];
			p_fjac_1_3++;
		}
		__global double *p_fjac_1_4 =
		    fjac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
				       (2));
		if ((unsigned long)p_fjac_1_4 % 64 == 0) {
			fjac_1[4] = vload2(0, p_fjac_1_4);
		} else {
			fjac_1[4].x = p_fjac_1_4[0];
			p_fjac_1_4++;
			fjac_1[4].y = p_fjac_1_4[0];
			p_fjac_1_4++;
		}
		__global double *p_njac_2_0 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
				       (0));
		if ((unsigned long)p_njac_2_0 % 64 == 0) {
			njac_2[0] = vload2(0, p_njac_2_0);
		} else {
			njac_2[0].x = p_njac_2_0[0];
			p_njac_2_0++;
			njac_2[0].y = p_njac_2_0[0];
			p_njac_2_0++;
		}
		__global double *p_njac_2_1 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0),
				       (1));
		if ((unsigned long)p_njac_2_1 % 64 == 0) {
			njac_2[1] = vload2(0, p_njac_2_1);
		} else {
			njac_2[1].x = p_njac_2_1[0];
			p_njac_2_1++;
			njac_2[1].y = p_njac_2_1[0];
			p_njac_2_1++;
		}
		__global double *p_njac_2_2 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
				       (0));
		if ((unsigned long)p_njac_2_2 % 64 == 0) {
			njac_2[2] = vload2(0, p_njac_2_2);
		} else {
			njac_2[2].x = p_njac_2_2[0];
			p_njac_2_2++;
			njac_2[2].y = p_njac_2_2[0];
			p_njac_2_2++;
		}
		__global double *p_njac_2_3 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2),
				       (0));
		if ((unsigned long)p_njac_2_3 % 64 == 0) {
			njac_2[3] = vload2(0, p_njac_2_3);
		} else {
			njac_2[3].x = p_njac_2_3[0];
			p_njac_2_3++;
			njac_2[3].y = p_njac_2_3[0];
			p_njac_2_3++;
		}
		__global double *p_njac_2_4 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
				       (1));
		if ((unsigned long)p_njac_2_4 % 64 == 0) {
			njac_2[4] = vload2(0, p_njac_2_4);
		} else {
			njac_2[4].x = p_njac_2_4[0];
			p_njac_2_4++;
			njac_2[4].y = p_njac_2_4[0];
			p_njac_2_4++;
		}
		__global double *p_njac_2_5 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
				       (0));
		if ((unsigned long)p_njac_2_5 % 64 == 0) {
			njac_2[5] = vload2(0, p_njac_2_5);
		} else {
			njac_2[5].x = p_njac_2_5[0];
			p_njac_2_5++;
			njac_2[5].y = p_njac_2_5[0];
			p_njac_2_5++;
		}
		__global double *p_njac_2_6 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
				       (0));
		if ((unsigned long)p_njac_2_6 % 64 == 0) {
			njac_2[6] = vload2(0, p_njac_2_6);
		} else {
			njac_2[6].x = p_njac_2_6[0];
			p_njac_2_6++;
			njac_2[6].y = p_njac_2_6[0];
			p_njac_2_6++;
		}
		__global double *p_njac_2_7 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
				       (0));
		if ((unsigned long)p_njac_2_7 % 64 == 0) {
			njac_2[7] = vload2(0, p_njac_2_7);
		} else {
			njac_2[7].x = p_njac_2_7[0];
			p_njac_2_7++;
			njac_2[7].y = p_njac_2_7[0];
			p_njac_2_7++;
		}
		__global double *p_njac_2_8 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
				       (3));
		if ((unsigned long)p_njac_2_8 % 64 == 0) {
			njac_2[8] = vload2(0, p_njac_2_8);
		} else {
			njac_2[8].x = p_njac_2_8[0];
			p_njac_2_8++;
			njac_2[8].y = p_njac_2_8[0];
			p_njac_2_8++;
		}
		__global double *p_njac_2_9 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (3),
				       (2));
		if ((unsigned long)p_njac_2_9 % 64 == 0) {
			njac_2[9] = vload2(0, p_njac_2_9);
		} else {
			njac_2[9].x = p_njac_2_9[0];
			p_njac_2_9++;
			njac_2[9].y = p_njac_2_9[0];
			p_njac_2_9++;
		}
		__global double *p_njac_2_10 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
				       (3));
		if ((unsigned long)p_njac_2_10 % 64 == 0) {
			njac_2[10] = vload2(0, p_njac_2_10);
		} else {
			njac_2[10].x = p_njac_2_10[0];
			p_njac_2_10++;
			njac_2[10].y = p_njac_2_10[0];
			p_njac_2_10++;
		}
		__global double *p_njac_2_11 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
				       (1));
		if ((unsigned long)p_njac_2_11 % 64 == 0) {
			njac_2[11] = vload2(0, p_njac_2_11);
		} else {
			njac_2[11].x = p_njac_2_11[0];
			p_njac_2_11++;
			njac_2[11].y = p_njac_2_11[0];
			p_njac_2_11++;
		}
		__global double *p_njac_2_12 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
				       (3));
		if ((unsigned long)p_njac_2_12 % 64 == 0) {
			njac_2[12] = vload2(0, p_njac_2_12);
		} else {
			njac_2[12].x = p_njac_2_12[0];
			p_njac_2_12++;
			njac_2[12].y = p_njac_2_12[0];
			p_njac_2_12++;
		}
		__global double *p_njac_3_0 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
				       (1));
		if ((unsigned long)p_njac_3_0 % 64 == 0) {
			njac_3[0] = vload4(0, p_njac_3_0);
		} else {
			njac_3[0].x = p_njac_3_0[0];
			p_njac_3_0++;
			njac_3[0].y = p_njac_3_0[0];
			p_njac_3_0++;
			njac_3[0].z = p_njac_3_0[0];
			p_njac_3_0++;
			njac_3[0].w = p_njac_3_0[0];
			p_njac_3_0++;
		}
		__global double *p_njac_3_1 =
		    njac + CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k + 1), (1),
				       (0));
		if ((unsigned long)p_njac_3_1 % 64 == 0) {
			njac_3[1] = vload4(0, p_njac_3_1);
		} else {
			njac_3[1].x = p_njac_3_1[0];
			p_njac_3_1++;
			njac_3[1].y = p_njac_3_1[0];
			p_njac_3_1++;
			njac_3[1].z = p_njac_3_1[0];
			p_njac_3_1++;
			njac_3[1].w = p_njac_3_1[0];
			p_njac_3_1++;
		}
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		tmp1 = dt * tz1;
		tmp2 = dt * tz2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
			  (0))] - tmp1 * njac_2[0].x - tmp1 * dz1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
			  (1))] - tmp1 * njac_2[0].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (0),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (0),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (0), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (0),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (0),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (1),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
			  (1))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (1),
							  (1))] - tmp1 * dz2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (1),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
			  (3))] - tmp1 * njac_2[8].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (1), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (1),
			  (4))] - tmp1 * njac_2[8].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
			  (0))] - tmp1 * njac_2[2].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
			  (1))] - tmp1 * njac_2[2].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (2),
							  (2))] - tmp1 * dz3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (2),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (2), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (2),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (2),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
			  (0))] - tmp1 * njac_2[7].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
			  (1))] - tmp1 * njac_2[7].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (3),
							  (3))] - tmp1 * dz4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (3), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (0))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (4),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (1))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
			  (1))] - tmp1 * njac_2[11].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (2))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
			  (2))] - tmp1 * njac_2[11].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (3))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (4),
							  (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (0), (4), (4))] =
		    -tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k - 1), (4),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k - 1), (4),
							  (4))] - tmp1 * dz5;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (0))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (0), (0))]
		    + tmp1 * 2.0 * dz1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (1))] =
		    tmp1 * 2.0 * njac_2[1].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (2))] =
		    tmp1 * 2.0 * njac_2[1].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (0), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (0), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (0), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (1))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (1), (1))]
		    + tmp1 * 2.0 * dz2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (1), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (1), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (0))] =
		    tmp1 * 2.0 * njac_2[3].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (1))] =
		    tmp1 * 2.0 * njac_2[3].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (2))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (2), (2))]
		    + tmp1 * 2.0 * dz3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (2), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (2), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (2), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (1))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (2))] =
		    tmp1 * 2.0 * njac_2[9].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (3))] =
		    1.0 + tmp1 * 2.0 * njac_2[9].y + tmp1 * 2.0 * dz4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (3), (4))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (3), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (0))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (1))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (1))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (2))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (3))] =
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k), (4), (3))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (1), (4), (4))] =
		    1.0 +
		    tmp1 * 2.0 *
		    njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i), (j), (k), (4), (4))]
		    + tmp1 * 2.0 * dz5;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (0))] =
		    tmp2 * fjac_1[2].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
			  (0))] - tmp1 * dz1;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (1))] =
		    tmp2 * fjac_1[2].y - tmp1 * njac_3[0].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
			  (2))] - tmp1 * njac_3[0].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
			  (3))] - tmp1 * njac_3[0].z;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (0), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (0),
			  (4))] - tmp1 * njac_3[0].w;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (1),
			  (0))] - tmp1 * njac_3[1].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (1))] =
		    tmp2 * fjac_1[0].x - tmp1 * njac_3[1].y - tmp1 * dz2;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (2))] =
		    tmp2 * fjac_1[0].y - tmp1 * njac_3[1].z;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (3))] =
		    tmp2 * fjac_1[1].x - tmp1 * njac_3[1].w;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (1), (4))] =
		    tmp2 * fjac_1[1].y -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (1), (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
			  (0))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k + 1), (2),
							  (0))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
			  (1))] - tmp1 * njac_2[4].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (2))] =
		    tmp2 * fjac_1[3].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
			  (2))] - tmp1 * dz3;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (3))] =
		    tmp2 * fjac_1[3].y - tmp1 * njac_2[10].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (2), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (2),
			  (4))] - tmp1 * njac_2[10].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
			  (0))] - tmp1 * njac_2[4].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
			  (1))] - tmp1 * njac_2[5].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (2))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
			  (2))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k + 1), (3),
							  (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (3))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
			  (3))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k + 1), (3),
							  (3))] - tmp1 * dz4;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (3), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (3),
			  (4))] - tmp1 * njac[CALC_5D_IDX(24, 24, 23, 5, 5, (i),
							  (j), (k + 1), (3),
							  (4))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (0))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
			  (0))] - tmp1 * njac_2[6].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (1))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
			  (1))] - tmp1 * njac_2[6].y;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (2))] =
		    tmp2 * fjac_1[4].x -
		    tmp1 *
		    njac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (4), (2))];
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (3))] =
		    tmp2 * fjac_1[4].y - tmp1 * njac_2[12].x;
		lhs[CALC_6D_IDX
		    (25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (4), (4))] =
		    tmp2 *
		    fjac[CALC_5D_IDX
			 (24, 24, 23, 5, 5, (i), (j), (k + 1), (4),
			  (4))] - tmp1 * njac_2[12].y - tmp1 * dz5;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1824 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_0(__global double *u, __global double *rho_i,
			    __global double *us, __global double *vs,
			    __global double *ws, __global double *square,
			    __global double *qs, int __ocl_i_bound,
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
	double rho_inv;		/* Defined at bt.c : 1816 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1826
		//-------------------------------------------
		double4 u_6;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1826
		//Candidates:
		//      u[i][j][k][0]
		//      u[i][j][k][1]
		//      u[i][j][k][2]
		//      u[i][j][k][3]
		//-------------------------------------------
		__global double *p_u_6_0 =
		    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0));
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

		rho_inv = 1.0 / u_6.x;
		rho_i[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] = rho_inv;
		us[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] = u_6.y * rho_inv;
		vs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] = u_6.z * rho_inv;
		ws[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] = u_6.w * rho_inv;
		square[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] =
		    0.5 * (u_6.y * u_6.y + u_6.z * u_6.z +
			   u_6.w * u_6.w) * rho_inv;
		qs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] =
		    square[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] * rho_inv;
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1849 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_1(__global double *rhs, __global double *forcing,
			    int __ocl_i_bound, int __ocl_j_bound,
			    int __ocl_k_bound)
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
	int m;			/* Defined at bt.c : 1815 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    forcing[CALC_4D_IDX
				    (25, 25, 25, 6, (i), (j), (k), (m))];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1865 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_2(__global double *us, __global double *rhs,
			    double dx1tx1, __global double *u, double tx2,
			    double dx2tx1, double xxcon2, double con43,
			    __global double *square, double c2, double dx3tx1,
			    __global double *vs, double dx4tx1,
			    __global double *ws, double dx5tx1, double xxcon3,
			    __global double *qs, double xxcon4, double xxcon5,
			    __global double *rho_i, double c1,
			    int __ocl_i_bound, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	double uijk;		/* Defined at bt.c : 1816 */
	double up1;		/* Defined at bt.c : 1816 */
	double um1;		/* Defined at bt.c : 1816 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 1867
		//-------------------------------------------
		double u_7[8];
		double square_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 1867
		//Candidates:
		//      u[i - 1][j][k][1]
		//      u[i + 1][j][k][1]
		//      u[i - 1][j][k][2]
		//      u[i - 1][j][k][4]
		//      u[i + 1][j][k][2]
		//      u[i - 1][j][k][3]
		//      u[i + 1][j][k][3]
		//      u[i][j][k][4]
		//      square[i - 1][j][k]
		//-------------------------------------------
		u_7[0] = u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (1))];
		u_7[1] = u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (1))];
		u_7[2] = u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (2))];
		u_7[3] = u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (4))];
		u_7[4] = u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (2))];
		u_7[5] = u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (3))];
		u_7[6] = u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (3))];
		u_7[7] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
		square_0 = square[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		uijk = us[CALC_3D_IDX(25, 25, 25, (i), (j), (k))];
		up1 = us[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))];
		um1 = us[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))];
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		    dx1tx1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (0))] -
		     2.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (0))]) -
		    tx2 * (u_7[1] - u_7[0]);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] +
		    dx2tx1 * (u_7[1] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))]
			      + u_7[0]) + xxcon2 * con43 * (up1 - 2.0 * uijk +
							    um1) -
		    tx2 * (u_7[1] * up1 - u_7[0] * um1 +
			   (u
			    [CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (4))]
			    -
			    square[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))] -
			    u_7[3] + square_0) * c2);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] +
		    dx3tx1 * (u_7[4] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))]
			      + u_7[2]) +
		    xxcon2 * (vs[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))] -
			      2.0 * vs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      vs[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))]) -
		    tx2 * (u_7[4] * up1 - u_7[2] * um1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] +
		    dx4tx1 * (u_7[6] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))]
			      + u_7[5]) +
		    xxcon2 * (ws[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))] -
			      2.0 * ws[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      ws[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))]) -
		    tx2 * (u_7[6] * up1 - u_7[5] * um1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] +
		    dx5tx1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (4))] -
		     2.0 * u_7[7] + u_7[3]) +
		    xxcon3 * (qs[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))] -
			      2.0 * qs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      qs[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))]) +
		    xxcon4 * (up1 * up1 - 2.0 * uijk * uijk + um1 * um1) +
		    xxcon5 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (4))] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))] -
		     2.0 * u_7[7] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
		     u_7[3] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i - 1), (j), (k))]) -
		    tx2 *
		    ((c1 *
		      u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (4))] -
		      c2 * square[CALC_3D_IDX(25, 25, 25, (i + 1), (j), (k))]) *
		     up1 - (c1 * u_7[3] - c2 * square_0) * um1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1928 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_3(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i + 1), (j), (k),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i + 2),
						     (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1942 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_4(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i - 1), (j), (k),
			       (m))] + 6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i + 1), (j), (k),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i + 2),
						     (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1955 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_5(__global double *rhs, double dssp,
			    __global double *u, int __ocl_i_bound,
			    int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 3;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	int m;			/* Defined at bt.c : 1815 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (25, 25, 25, 5, (i - 2), (j), (k),
			       (m))] - 4.0 * u[CALC_4D_IDX(25, 25, 25, 5,
							   (i - 1), (j), (k),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (25, 25, 25, 5, (i + 1), (j), (k),
				(m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i + 2),
						      (j), (k), (m))]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1972 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_6(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i - 2), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (m))] +
		     6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i + 1), (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 1986 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_7(__global double *rhs, int i, double dssp,
			    __global double *u, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i - 2), (j), (k), (m))] -
		     4. *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i - 1), (j), (k), (m))] +
		     5.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2002 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_8(__global double *vs, __global double *rhs,
			    double dy1ty1, __global double *u, double ty2,
			    double dy2ty1, double yycon2, __global double *us,
			    double dy3ty1, double con43,
			    __global double *square, double c2, double dy4ty1,
			    __global double *ws, double dy5ty1, double yycon3,
			    __global double *qs, double yycon4, double yycon5,
			    __global double *rho_i, double c1,
			    int __ocl_i_bound, int __ocl_j_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	double vijk;		/* Defined at bt.c : 1816 */
	double vp1;		/* Defined at bt.c : 1816 */
	double vm1;		/* Defined at bt.c : 1816 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2004
		//-------------------------------------------
		double2 u_8[2];
		double u_9[4];
		double square_1;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2004
		//Candidates:
		//      u[i][j - 1][k][1]
		//      u[i][j - 1][k][2]
		//      u[i][j - 1][k][3]
		//      u[i][j - 1][k][4]
		//      u[i][j + 1][k][1]
		//      u[i][j + 1][k][2]
		//      u[i][j + 1][k][3]
		//      u[i][j][k][4]
		//      square[i][j - 1][k]
		//-------------------------------------------
		__global double *p_u_8_0 =
		    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j - 1), (k), (1));
		if ((unsigned long)p_u_8_0 % 64 == 0) {
			u_8[0] = vload2(0, p_u_8_0);
		} else {
			u_8[0].x = p_u_8_0[0];
			p_u_8_0++;
			u_8[0].y = p_u_8_0[0];
			p_u_8_0++;
		}
		__global double *p_u_8_1 =
		    u + CALC_4D_IDX(25, 25, 25, 5, (i), (j - 1), (k), (3));
		if ((unsigned long)p_u_8_1 % 64 == 0) {
			u_8[1] = vload2(0, p_u_8_1);
		} else {
			u_8[1].x = p_u_8_1[0];
			p_u_8_1++;
			u_8[1].y = p_u_8_1[0];
			p_u_8_1++;
		}
		u_9[0] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (1))];
		u_9[1] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (2))];
		u_9[2] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (3))];
		u_9[3] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
		square_1 = square[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))];
		//-------------------------------------------
		//Prefetching (END)
		//-------------------------------------------

		vijk = vs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))];
		vp1 = vs[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))];
		vm1 = vs[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))];
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		    dy1ty1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (0))] -
		     2.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j - 1), (k), (0))]) -
		    ty2 * (u_9[1] - u_8[0].y);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] +
		    dy2ty1 * (u_9[0] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))]
			      + u_8[0].x) +
		    yycon2 * (us[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))] -
			      2.0 * us[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      us[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))]) -
		    ty2 * (u_9[0] * vp1 - u_8[0].x * vm1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] +
		    dy3ty1 * (u_9[1] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))]
			      + u_8[0].y) + yycon2 * con43 * (vp1 - 2.0 * vijk +
							      vm1) -
		    ty2 * (u_9[1] * vp1 - u_8[0].y * vm1 +
			   (u
			    [CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (4))]
			    -
			    square[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))] -
			    u_8[1].y + square_1) * c2);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] +
		    dy4ty1 * (u_9[2] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))]
			      + u_8[1].x) +
		    yycon2 * (ws[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))] -
			      2.0 * ws[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      ws[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))]) -
		    ty2 * (u_9[2] * vp1 - u_8[1].x * vm1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] +
		    dy5ty1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (4))] -
		     2.0 * u_9[3] + u_8[1].y) +
		    yycon3 * (qs[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))] -
			      2.0 * qs[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
			      qs[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))]) +
		    yycon4 * (vp1 * vp1 - 2.0 * vijk * vijk + vm1 * vm1) +
		    yycon5 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (4))] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))] -
		     2.0 * u_9[3] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i), (j), (k))] +
		     u_8[1].y *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i), (j - 1), (k))]) -
		    ty2 *
		    ((c1 *
		      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (4))] -
		      c2 * square[CALC_3D_IDX(25, 25, 25, (i), (j + 1), (k))]) *
		     vp1 - (c1 * u_8[1].y - c2 * square_1) * vm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2060 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_9(__global double *rhs, int j, double dssp,
			    __global double *u, int __ocl_i_bound,
			    int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j + 1), (k),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i),
						     (j + 2), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2074 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_10(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j - 1), (k),
			       (m))] + 6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j + 1), (k),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i),
						     (j + 2), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2087 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_11(__global double *rhs, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 3;
	int k = get_global_id(2) + 1;
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
	int m;			/* Defined at bt.c : 1815 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j - 2), (k),
			       (m))] - 4.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i),
							   (j - 1), (k),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (25, 25, 25, 5, (i), (j + 1), (k),
				(m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i),
						      (j + 2), (k), (m))]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2104 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_12(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j - 2), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j - 1), (k), (m))] +
		     6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j + 1), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2118 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_13(__global double *rhs, int j, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j - 2), (k), (m))] -
		     4. *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j - 1), (k), (m))] +
		     5. * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2134 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_14(__global double *ws, __global double *rhs,
			     double dz1tz1, __global double *u, double tz2,
			     double dz2tz1, double zzcon2, __global double *us,
			     double dz3tz1, __global double *vs, double dz4tz1,
			     double con43, __global double *square, double c2,
			     double dz5tz1, double zzcon3, __global double *qs,
			     double zzcon4, double zzcon5,
			     __global double *rho_i, double c1,
			     int __ocl_i_bound, int __ocl_j_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 1;
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
	double wijk;		/* Defined at bt.c : 1816 */
	double wp1;		/* Defined at bt.c : 1816 */
	double wm1;		/* Defined at bt.c : 1816 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		//-------------------------------------------
		//Declare prefetching Buffers (BEGIN) : 2136
		//-------------------------------------------
		double2 ws_0;
		double u_10[8];
		double2 us_0;
		double2 vs_0;
		double square_2;
		double2 qs_0;
		double2 rho_i_0;
		//-------------------------------------------
		//Declare prefetching buffers (END)
		//-------------------------------------------
		//-------------------------------------------
		//Prefetching (BEGIN) : 2136
		//Candidates:
		//      ws[i][j][k - 1]
		//      ws[i][j][k]
		//      u[i][j][k - 1][1]
		//      u[i][j][k - 1][3]
		//      u[i][j][k + 1][1]
		//      u[i][j][k - 1][2]
		//      u[i][j][k + 1][2]
		//      u[i][j][k + 1][3]
		//      u[i][j][k - 1][4]
		//      u[i][j][k][4]
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
		__global double *p_ws_0_0 =
		    ws + CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1));
		if ((unsigned long)p_ws_0_0 % 64 == 0) {
			ws_0 = vload2(0, p_ws_0_0);
		} else {
			ws_0.x = p_ws_0_0[0];
			p_ws_0_0++;
			ws_0.y = p_ws_0_0[0];
			p_ws_0_0++;
		}
		u_10[0] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (1))];
		u_10[1] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (3))];
		u_10[2] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (1))];
		u_10[3] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (2))];
		u_10[4] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (2))];
		u_10[5] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (3))];
		u_10[6] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (4))];
		u_10[7] = u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))];
		__global double *p_us_0_0 =
		    us + CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1));
		if ((unsigned long)p_us_0_0 % 64 == 0) {
			us_0 = vload2(0, p_us_0_0);
		} else {
			us_0.x = p_us_0_0[0];
			p_us_0_0++;
			us_0.y = p_us_0_0[0];
			p_us_0_0++;
		}
		__global double *p_vs_0_0 =
		    vs + CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1));
		if ((unsigned long)p_vs_0_0 % 64 == 0) {
			vs_0 = vload2(0, p_vs_0_0);
		} else {
			vs_0.x = p_vs_0_0[0];
			p_vs_0_0++;
			vs_0.y = p_vs_0_0[0];
			p_vs_0_0++;
		}
		square_2 = square[CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1))];
		__global double *p_qs_0_0 =
		    qs + CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1));
		if ((unsigned long)p_qs_0_0 % 64 == 0) {
			qs_0 = vload2(0, p_qs_0_0);
		} else {
			qs_0.x = p_qs_0_0[0];
			p_qs_0_0++;
			qs_0.y = p_qs_0_0[0];
			p_qs_0_0++;
		}
		__global double *p_rho_i_0_0 =
		    rho_i + CALC_3D_IDX(25, 25, 25, (i), (j), (k - 1));
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

		wijk = ws_0.y;
		wp1 = ws[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))];
		wm1 = ws_0.x;
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		    dz1tz1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (0))] -
		     2.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (0))] +
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (0))]) -
		    tz2 * (u_10[5] - u_10[1]);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))] +
		    dz2tz1 * (u_10[2] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (1))]
			      + u_10[0]) +
		    zzcon2 * (us[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))] -
			      2.0 * us_0.y + us_0.x) - tz2 * (u_10[2] * wp1 -
							      u_10[0] * wm1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))] +
		    dz3tz1 * (u_10[4] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (2))]
			      + u_10[3]) +
		    zzcon2 * (vs[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))] -
			      2.0 * vs_0.y + vs_0.x) - tz2 * (u_10[4] * wp1 -
							      u_10[3] * wm1);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))] +
		    dz4tz1 * (u_10[5] -
			      2.0 *
			      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (3))]
			      + u_10[1]) + zzcon2 * con43 * (wp1 - 2.0 * wijk +
							     wm1) -
		    tz2 * (u_10[5] * wp1 - u_10[1] * wm1 +
			   (u
			    [CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (4))]
			    -
			    square[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))] -
			    u_10[6] + square_2) * c2);
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (4))] +
		    dz5tz1 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (4))] -
		     2.0 * u_10[7] + u_10[6]) +
		    zzcon3 * (qs[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))] -
			      2.0 * qs_0.y + qs_0.x) + zzcon4 * (wp1 * wp1 -
								 2.0 * wijk *
								 wijk +
								 wm1 * wm1) +
		    zzcon5 *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (4))] *
		     rho_i[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))] -
		     2.0 * u_10[7] * rho_i_0.y + u_10[6] * rho_i_0.x) -
		    tz2 *
		    ((c1 *
		      u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (4))] -
		      c2 * square[CALC_3D_IDX(25, 25, 25, (i), (j), (k + 1))]) *
		     wp1 - (c1 * u_10[6] - c2 * square_2) * wm1);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2193 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_15(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (5.0 *
			    u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j), (k + 1),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i), (j),
						     (k + 2), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2207 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_16(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp * (-4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j), (k - 1),
			       (m))] + 6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i),
							   (j), (k),
							   (m))] -
			    4.0 *
			    u[CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j), (k + 1),
			       (m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i), (j),
						     (k + 2), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2220 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_17(__global double *rhs, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int k = get_global_id(2) + 3;
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
	int m;			/* Defined at bt.c : 1815 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (m = 0; m < 5; m++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    -
			    dssp *
			    (u
			     [CALC_4D_IDX
			      (25, 25, 25, 5, (i), (j), (k - 2),
			       (m))] - 4.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i),
							   (j), (k - 1),
							   (m))] +
			     6.0 *
			     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
			     4.0 *
			     u[CALC_4D_IDX
			       (25, 25, 25, 5, (i), (j), (k + 1),
				(m))] + u[CALC_4D_IDX(25, 25, 25, 5, (i), (j),
						      (k + 2), (m))]);
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2237 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_18(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 2), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (m))] +
		     6.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k + 1), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2251 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_19(__global double *rhs, int k, double dssp,
			     __global double *u, int __ocl_i_bound,
			     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
		    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] -
		    dssp *
		    (u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 2), (m))] -
		     4.0 *
		     u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k - 1), (m))] +
		     5.0 * u[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]);
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2264 of bt.c
//-------------------------------------------------------------------------------
__kernel void compute_rhs_20(__global int *grid_points, __global double *rhs,
			     double dt, int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	int i;			/* Defined at bt.c : 1815 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (i = 1; i < grid_points[(0)] - 1; i++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    * dt;
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2790 of bt.c
//-------------------------------------------------------------------------------
__kernel void x_backsubstitute_0(__global double *rhs, int i,
				 __global double *lhs, int __ocl_j_bound,
				 int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(j < __ocl_j_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	int n;			/* Defined at bt.c : 2786 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (n = 0; n < 5; n++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    -
			    lhs[CALC_6D_IDX
				(25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (m),
				 (n))] * rhs[CALC_4D_IDX(25, 25, 25, 5, (i + 1),
							 (j), (k), (n))];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2828 of bt.c
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
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
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (0)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (0)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (0)) + ((25 * 5) * (j)) +
				   ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2849 of bt.c
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_1(__global double *lhs, int i, __global double *rhs,
			     int __ocl_j_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
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
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (i - 1)) +
				     ((25 * 5) * (j)) + ((5) * (k))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (i - 1)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				   ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 2881 of bt.c
//-------------------------------------------------------------------------------
__kernel void x_solve_cell_2(__global double *lhs, int isize,
			     __global double *rhs, int i, int __ocl_j_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int j = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
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
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (isize)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (isize - 1)) +
				     ((25 * 5) * (j)) + ((5) * (k))),
				    (((25 * 25 * 5) * (isize)) +
				     ((25 * 5) * (j)) + ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (isize)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (isize - 1)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (isize)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((25 * 25 * 3 * 5 * 5) * (i)) +
			       ((25 * 3 * 5 * 5) * (j)) + ((3 * 5 * 5) * (k)) +
			       ((5 * 5) * (1))),
			      (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
			       ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3437 of bt.c
//-------------------------------------------------------------------------------
__kernel void y_backsubstitute_0(__global double *rhs, int j,
				 __global double *lhs, int __ocl_i_bound,
				 int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	int m = get_global_id(2);
	if (!(i < __ocl_i_bound)) {
		return;
	}
	if (!(k < __ocl_k_bound)) {
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
	int n;			/* Defined at bt.c : 3432 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (n = 0; n < 5; n++) {
			rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))] =
			    rhs[CALC_4D_IDX(25, 25, 25, 5, (i), (j), (k), (m))]
			    -
			    lhs[CALC_6D_IDX
				(25, 25, 25, 3, 5, 5, (i), (j), (k), (2), (m),
				 (n))] * rhs[CALC_4D_IDX(25, 25, 25, 5, (i),
							 (j + 1), (k), (n))];
		}
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3475 of bt.c
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_i_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (0)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (0)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (i)) + ((25 * 5) * (0)) +
				   ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3496 of bt.c
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_1(__global double *lhs, int j, __global double *rhs,
			     int __ocl_i_bound, int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (i)) +
				     ((25 * 5) * (j - 1)) + ((5) * (k))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j - 1)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				   ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3529 of bt.c
//-------------------------------------------------------------------------------
__kernel void y_solve_cell_2(__global double *lhs, int jsize,
			     __global double *rhs, int __ocl_i_bound,
			     int __ocl_k_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int k = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (i)) +
				     ((25 * 5) * (jsize - 1)) + ((5) * (k))),
				    (((25 * 25 * 5) * (i)) +
				     ((25 * 5) * (jsize)) + ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (jsize - 1)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (jsize)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((25 * 25 * 3 * 5 * 5) * (i)) +
			       ((25 * 3 * 5 * 5) * (jsize)) +
			       ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
			      (((25 * 25 * 5) * (i)) + ((25 * 5) * (jsize)) +
			       ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3601 of bt.c
//-------------------------------------------------------------------------------
__kernel void z_backsubstitute_0(__global int *grid_points,
				 __global double *rhs, __global double *lhs,
				 int __ocl_i_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	if (!(i < __ocl_i_bound)) {
		return;
	}
	//-------------------------------------------
	//OpenCL global indexes (END)
	//-------------------------------------------

	//-------------------------------------------
	//Pivate variables (BEGIN)
	//-------------------------------------------
	int j;			/* Defined at bt.c : 3597 */
	int k;			/* Defined at bt.c : 3597 */
	int m;			/* Defined at bt.c : 3597 */
	int n;			/* Defined at bt.c : 3597 */
	//-------------------------------------------
	//Pivate variables (END)
	//-------------------------------------------

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		for (j = 1; j < grid_points[(1)] - 1; j++) {
			for (k = grid_points[(2)] - 2; k >= 0; k--) {
				for (m = 0; m < 5; m++) {
					for (n = 0; n < 5; n++) {
						rhs[CALC_4D_IDX
						    (25, 25, 25, 5, (i), (j),
						     (k), (m))] =
						    rhs[CALC_4D_IDX
							(25, 25, 25, 5, (i),
							 (j), (k),
							 (m))] -
						    lhs[CALC_6D_IDX
							(25, 25, 25, 3, 5, 5,
							 (i), (j), (k), (2),
							 (m),
							 (n))] *
						    rhs[CALC_4D_IDX
							(25, 25, 25, 5, (i),
							 (j), (k + 1), (n))];
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
//Loop defined at line 3643 of bt.c
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_0(__global double *lhs, __global double *rhs,
			     int __ocl_i_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (0)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (0)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				   ((5) * (0))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3665 of bt.c
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_1(__global double *lhs, int k, __global double *rhs,
			     int __ocl_i_bound, int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (k - 1))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (k))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k - 1)) + ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))));
		binvcrhs_p0_p1_p2(lhs, lhs, rhs,
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (1))),
				  (((25 * 25 * 3 * 5 * 5) * (i)) +
				   ((25 * 3 * 5 * 5) * (j)) +
				   ((3 * 5 * 5) * (k)) + ((5 * 5) * (2))),
				  (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				   ((5) * (k))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//Loop defined at line 3702 of bt.c
//-------------------------------------------------------------------------------
__kernel void z_solve_cell_2(__global double *lhs, int ksize,
			     __global double *rhs, int __ocl_i_bound,
			     int __ocl_j_bound)
{
	//-------------------------------------------
	//OpenCL global indexes (BEGIN)
	//-------------------------------------------
	int i = get_global_id(0) + 1;
	int j = get_global_id(1) + 1;
	if (!(i < __ocl_i_bound)) {
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

	//-------------------------------------------
	//OpenCL kernel (BEGIN)
	//-------------------------------------------
	{
		matvec_sub_p0_p1_p2(lhs, rhs, rhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (0))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (ksize - 1))),
				    (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
				     ((5) * (ksize))));
		matmul_sub_p0_p1_p2(lhs, lhs, lhs,
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (0))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize - 1)) +
				     ((5 * 5) * (2))),
				    (((25 * 25 * 3 * 5 * 5) * (i)) +
				     ((25 * 3 * 5 * 5) * (j)) +
				     ((3 * 5 * 5) * (ksize)) +
				     ((5 * 5) * (1))));
		binvrhs_p0_p1(lhs, rhs,
			      (((25 * 25 * 3 * 5 * 5) * (i)) +
			       ((25 * 3 * 5 * 5) * (j)) +
			       ((3 * 5 * 5) * (ksize)) + ((5 * 5) * (1))),
			      (((25 * 25 * 5) * (i)) + ((25 * 5) * (j)) +
			       ((5) * (ksize))));
	}
	//-------------------------------------------
	//OpenCL kernel (END)
	//-------------------------------------------

}

//-------------------------------------------------------------------------------
//OpenCL Kernels (END)
//-------------------------------------------------------------------------------
